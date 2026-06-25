"""Matching Service — business logic for real-time POI matching."""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from backend.core.config import get_config
from backend.domain.entities.match_result import MatchResult
from backend.domain.interfaces.matcher import MatchingStrategy
from backend.domain.interfaces.repository import CacheRepository, POIRepository

log = logging.getLogger("poi.service.matching")

# Sticky-POI Redis key prefix. Separate from the short-lived object cache
# so that the first-matched POI survives object_cache_ttl expirations.
_STICKY_PREFIX = "sticky_poi:"


_CAM_CONTINUITY_THRESHOLD = 0.30  # cosine sim to consider same person
_CAM_BUFFER_MAX_AGE = 60          # seconds before stale buffer eviction


class MatchingService:
    """Applies Cache-Aside pattern for object_id → poi_id lookups.

    If the cache contains a mapping, FAISS is skipped entirely.
    Otherwise, FAISS is searched and the result is cached.

    Sticky-first-match guarantee (stable UUIDs)
    --------------------------------------------
    When a person is first identified as POI-A, that binding is stored in a
    separate "sticky" Redis key with TTL = track_seen_ttl (default 600 s).
    If the short-lived object cache later expires (object_cache_ttl, default
    300 s) and FAISS returns a different POI-B on re-query (due to
    frame-to-frame embedding variation), the sticky key returns POI-A and
    the result is re-cached — preventing the same physical person from
    generating alerts for multiple POIs during one appearance window.

    Embedding-continuity tracking (cam:* fallback IDs)
    ---------------------------------------------------
    Camera-local IDs like ``cam:Camera_01:1`` are recycled across different
    physical people.  Instead of relying on the ID, we compare the current
    face embedding with the previous one for the same cam:* slot.  If cosine
    similarity > _CAM_CONTINUITY_THRESHOLD (0.30) the person hasn't changed →
    reuse the previous POI match.  If below that threshold it's a new person →
    run FAISS fresh.
    """

    def __init__(
        self,
        strategy: MatchingStrategy,
        cache_repo: CacheRepository,
        poi_repo: Optional[POIRepository] = None,
    ) -> None:
        self._strategy = strategy
        self._cache = cache_repo
        self._poi_repo = poi_repo
        self._cfg = get_config()
        # In-memory buffer: cam:* object_id → (normalised_embedding, poi_id, similarity, timestamp)
        self._cam_buffer: dict[str, tuple[np.ndarray, str, float, float]] = {}

    # ── Sticky-POI helpers ────────────────────────────────────────────────

    def _get_sticky_poi(self, object_id: str) -> Optional[tuple[str, float]]:
        """Return (poi_id, similarity) from the sticky key, or None."""
        try:
            raw = self._cache._r.get(f"{_STICKY_PREFIX}{object_id}")  # type: ignore[attr-defined]
            if raw is None:
                return None
            import json as _json
            data = _json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            return data["poi_id"], float(data["similarity"])
        except Exception:
            return None

    def _set_sticky_poi(self, object_id: str, poi_id: str, similarity: float) -> None:
        """Persist sticky mapping with track_seen_ttl so it outlives the object cache."""
        try:
            import json as _json
            value = _json.dumps({"poi_id": poi_id, "similarity": similarity})
            self._cache._r.setex(  # type: ignore[attr-defined]
                f"{_STICKY_PREFIX}{object_id}",
                self._cfg.track_seen_ttl,
                value,
            )
        except Exception:
            pass  # Non-fatal: fall back to regular cache behaviour

    def _delete_sticky_poi(self, object_id: str) -> None:
        """Remove sticky-POI key for the given object."""
        try:
            self._cache._r.delete(f"{_STICKY_PREFIX}{object_id}")  # type: ignore[attr-defined]
        except Exception:
            pass

    # ── Main matching entry point ─────────────────────────────────────────

    @staticmethod
    def _is_stable_id(object_id: str) -> bool:
        """Return True if the object_id is a globally unique Scenescape UUID.

        Camera-local fallback IDs like ``cam:Camera_01:1`` are recycled across
        different physical people and must NOT be treated as stable identifiers
        for sticky-POI or long-lived caching.
        """
        return not object_id.startswith("cam:")

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        """L2-normalize a vector (in-place safe)."""
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _cleanup_cam_buffer(self) -> None:
        """Evict stale entries from the in-memory embedding buffer."""
        now = time.time()
        stale = [k for k, v in self._cam_buffer.items()
                 if now - v[3] > _CAM_BUFFER_MAX_AGE]
        for k in stale:
            del self._cam_buffer[k]

    def match_object(
        self, object_id: str, embedding_vector: list[float]
    ) -> Optional[MatchResult]:
        """Match an object's embedding against the POI index.

        Returns the best MatchResult or None if no match above threshold.

        Priority order for **stable UUIDs**:
          1. Short-lived object cache (object_cache_ttl, default 300 s).
          2. Sticky-POI key (track_seen_ttl, default 600 s).
          3. FAISS search.

        Priority order for **cam:* fallback IDs**:
          1. Embedding-continuity check (in-memory) — if the current face
             embedding is similar to the last one seen for this cam slot,
             reuse the previous match without running FAISS.
          2. FAISS search — result stored in both the Redis cache and the
             in-memory embedding buffer.
        """
        stable_id = self._is_stable_id(object_id)

        # Prepare normalised query vector (needed for continuity + FAISS)
        vector = np.array(embedding_vector, dtype=np.float32)
        if vector.ndim == 2:
            vector = vector[0]
        normed_vec = self._normalize(vector.copy())

        # ── cam:* fast path — embedding continuity ────────────────────────
        if not stable_id:
            # Periodic cleanup of stale buffer entries
            if len(self._cam_buffer) > 50:
                self._cleanup_cam_buffer()

            prev = self._cam_buffer.get(object_id)
            if prev is not None:
                prev_emb, prev_poi, prev_sim, prev_ts = prev
                continuity = float(np.dot(normed_vec, prev_emb))
                age = time.time() - prev_ts

                if continuity >= _CAM_CONTINUITY_THRESHOLD and age < _CAM_BUFFER_MAX_AGE:
                    # Same person — reuse previous match, update embedding
                    # (rolling update allows gradual drift while blocking
                    # abrupt changes from a new person).
                    self._cam_buffer[object_id] = (normed_vec, prev_poi, prev_sim, time.time())
                    log.info(
                        "Continuity hit: object=%s → poi=%s (sim=%.3f, continuity=%.3f, age=%.0fs)",
                        object_id, prev_poi, prev_sim, continuity, age,
                    )
                    return MatchResult(poi_id=prev_poi, similarity_score=prev_sim, faiss_distance=0.0)
                else:
                    # Different person or stale — evict and fall through to FAISS
                    log.info(
                        "Continuity break: object=%s prev_poi=%s continuity=%.3f age=%.0fs — running FAISS",
                        object_id, prev_poi, continuity, age,
                    )
                    del self._cam_buffer[object_id]
                    self._cache.delete_object(object_id)

            # Fall through to FAISS (no cache/sticky for cam:* IDs)
            return self._faiss_search(object_id, vector, normed_vec, stable_id=False)

        # ── Stable UUID path ──────────────────────────────────────────────

        # 1. Short-lived cache
        cached_poi = self._cache.get_poi_for_object(object_id)
        if cached_poi:
            cached_sim = getattr(self._cache, "get_similarity_for_object", lambda _: None)(object_id)
            if cached_sim is None:
                log.debug("Cache hit without similarity: object=%s poi=%s — evicting", object_id, cached_poi)
                self._cache.delete_object(object_id)
            elif cached_sim < self._cfg.similarity_threshold:
                log.debug(
                    "Cache hit below threshold: object=%s poi=%s sim=%.4f threshold=%.2f — evicting",
                    object_id, cached_poi, cached_sim, self._cfg.similarity_threshold,
                )
                self._cache.delete_object(object_id)
            else:
                log.debug("Cache hit: object=%s → poi=%s similarity=%.4f", object_id, cached_poi, cached_sim)
                return MatchResult(poi_id=cached_poi, similarity_score=cached_sim, faiss_distance=0.0)

        # 2. Sticky-POI key
        sticky = self._get_sticky_poi(object_id)
        if sticky is not None:
            sticky_poi_id, sticky_sim = sticky
            if self._poi_repo and not self._poi_repo.get(sticky_poi_id):
                log.info(
                    "Sticky-POI stale: object=%s → poi=%s was deleted — evicting",
                    object_id, sticky_poi_id,
                )
                self._delete_sticky_poi(object_id)
            elif sticky_sim >= self._cfg.similarity_threshold:
                log.info(
                    "Sticky-POI hit: object=%s → poi=%s (sim=%.3f) — re-using first match, skipping FAISS",
                    object_id, sticky_poi_id, sticky_sim,
                )
                self._cache.set_poi_for_object(
                    object_id, sticky_poi_id,
                    ttl=self._cfg.object_cache_ttl,
                    similarity=sticky_sim,
                )
                return MatchResult(poi_id=sticky_poi_id, similarity_score=sticky_sim, faiss_distance=0.0)
            else:
                log.debug(
                    "Sticky-POI below threshold: object=%s poi=%s sim=%.4f — allowing FAISS re-query",
                    object_id, sticky_poi_id, sticky_sim,
                )

        # 3. FAISS search
        return self._faiss_search(object_id, vector, normed_vec, stable_id=True)

    def _faiss_search(
        self, object_id: str, vector: np.ndarray, normed_vec: np.ndarray,
        *, stable_id: bool,
    ) -> Optional[MatchResult]:
        """Run FAISS search and cache/buffer the result."""
        t0 = time.perf_counter()
        matches = self._strategy.match(
            vector,
            top_k=self._cfg.search_top_k,
            threshold=self._cfg.similarity_threshold,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if self._cfg.benchmark_latency:
            log.info("FAISS search: %.2f ms", elapsed_ms)

        if not matches:
            # Debug: log query vector stats on miss to compare with matches
            norm = float(np.linalg.norm(vector))
            log.debug("FAISS miss: object=%s vec_norm=%.4f vec[:5]=%s", object_id, norm, vector[:5].tolist())
            return None

        best = matches[0]

        norm = float(np.linalg.norm(vector))
        log.debug(
            "FAISS match: object=%s poi=%s sim=%.4f vec_norm=%.4f dim=%d",
            object_id, best.poi_id, best.similarity_score, norm, len(vector),
        )

        if stable_id:
            self._cache.set_poi_for_object(
                object_id, best.poi_id,
                ttl=self._cfg.object_cache_ttl,
                similarity=best.similarity_score,
            )
            self._set_sticky_poi(object_id, best.poi_id, best.similarity_score)
        else:
            # cam:* IDs: store in embedding buffer (no Redis cache needed)
            self._cam_buffer[object_id] = (
                normed_vec, best.poi_id, best.similarity_score, time.time(),
            )

        log.info(
            "Match found: object=%s → poi=%s (similarity=%.3f) [FAISS]",
            object_id, best.poi_id, best.similarity_score,
        )
        return best

