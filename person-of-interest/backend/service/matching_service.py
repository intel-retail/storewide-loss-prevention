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


class MatchingService:
    """Applies Cache-Aside pattern for object_id → poi_id lookups.

    If the cache contains a mapping, FAISS is skipped entirely.
    Otherwise, FAISS is searched and the result is cached.

    Sticky-first-match guarantee
    ----------------------------
    When a person is first identified as POI-A, that binding is stored in a
    separate "sticky" Redis key with TTL = track_seen_ttl (default 600 s).
    If the short-lived object cache later expires (object_cache_ttl, default
    300 s) and FAISS returns a different POI-B on re-query (due to
    frame-to-frame embedding variation), the sticky key returns POI-A and
    the result is re-cached — preventing the same physical person from
    generating alerts for multiple POIs during one appearance window.
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

    def match_object(
        self, object_id: str, embedding_vector: list[float]
    ) -> Optional[MatchResult]:
        """Match an object's embedding against the POI index.

        Returns the best MatchResult or None if no match above threshold.

        Priority order:
          1. Short-lived object cache (object_cache_ttl, default 300 s).
          2. Sticky-POI key (track_seen_ttl, default 600 s) — enforces
             first-match-wins so the same person cannot flip between POIs.
          3. FAISS search — result stored in both cache and sticky key.
        """
        # ── 1. Short-lived cache (fast path) ─────────────────────────────
        cached_poi = self._cache.get_poi_for_object(object_id)
        if cached_poi:
            cached_sim = getattr(self._cache, "get_similarity_for_object", lambda _: None)(object_id)
            if cached_sim is None:
                # Similarity missing (legacy entry or corrupt cache) — evict and re-query FAISS
                log.debug("Cache hit without similarity: object=%s poi=%s — evicting", object_id, cached_poi)
                self._cache.delete_object(object_id)
                # Fall through
            elif cached_sim < self._cfg.similarity_threshold:
                # Threshold may have been raised since caching — evict and re-query
                log.debug(
                    "Cache hit below threshold: object=%s poi=%s sim=%.4f threshold=%.2f — evicting",
                    object_id, cached_poi, cached_sim, self._cfg.similarity_threshold,
                )
                self._cache.delete_object(object_id)
                # Fall through
            else:
                log.debug("Cache hit: object=%s → poi=%s similarity=%.4f", object_id, cached_poi, cached_sim)
                return MatchResult(poi_id=cached_poi, similarity_score=cached_sim, faiss_distance=0.0)

        # ── 2. Sticky-POI key (survives cache expiry) ────────────────────
        # If a sticky binding exists from an earlier frame in this appearance
        # window, return that POI without running FAISS.  This prevents the
        # same physical person from being re-assigned to a different POI when
        # the short-lived cache expires and a noisier frame is processed.
        sticky = self._get_sticky_poi(object_id)
        if sticky is not None:
            sticky_poi_id, sticky_sim = sticky
            # Validate the POI still exists — it may have been deleted
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
                # Refresh the short-lived cache so the next frames are fast
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

        # ── 3. FAISS search ───────────────────────────────────────────────
        vector = np.array(embedding_vector, dtype=np.float32)
        if vector.ndim == 2:
            vector = vector[0]

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
            return None

        best = matches[0]
        # Cache with short TTL (fast path for subsequent frames)
        self._cache.set_poi_for_object(
            object_id, best.poi_id, ttl=self._cfg.object_cache_ttl, similarity=best.similarity_score
        )
        # Sticky key: first FAISS match wins for the entire appearance window
        self._set_sticky_poi(object_id, best.poi_id, best.similarity_score)
        log.info(
            "Match found: object=%s → poi=%s (similarity=%.3f) [FAISS]",
            object_id, best.poi_id, best.similarity_score,
        )
        return best

