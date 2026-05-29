"""Matching Service — business logic for real-time POI matching."""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from backend.core.config import get_config
from backend.domain.entities.match_result import MatchResult
from backend.domain.interfaces.matcher import MatchingStrategy
from backend.domain.interfaces.repository import CacheRepository

log = logging.getLogger("poi.service.matching")


class MatchingService:
    """Applies Cache-Aside pattern for object_id → poi_id lookups.

    If the cache contains a mapping, FAISS is skipped entirely.
    Otherwise, FAISS is searched and the result is cached.
    """

    def __init__(
        self,
        strategy: MatchingStrategy,
        cache_repo: CacheRepository,
    ) -> None:
        self._strategy = strategy
        self._cache = cache_repo
        self._cfg = get_config()

    def match_object(
        self, object_id: str, embedding_vector: list[float]
    ) -> Optional[MatchResult]:
        """Match an object's embedding against the POI index.

        Returns the best MatchResult or None if no match above threshold.
        Uses Cache-Aside: checks cache first, falls back to FAISS.
        """
        # Cache-Aside: check cache first
        cached_poi = self._cache.get_poi_for_object(object_id)
        if cached_poi:
            cached_sim = getattr(self._cache, "get_similarity_for_object", lambda _: None)(object_id)
            if cached_sim is None:
                # Similarity missing (legacy entry or corrupt cache) — evict and re-query FAISS
                log.debug("Cache hit without similarity: object=%s poi=%s — evicting, will re-query FAISS", object_id, cached_poi)
                self._cache.delete_object(object_id)
                # Fall through to FAISS search below
            elif cached_sim < self._cfg.similarity_threshold:
                # Threshold may have been raised since caching — evict and re-query FAISS
                log.debug(
                    "Cache hit below threshold: object=%s poi=%s sim=%.4f threshold=%.2f — evicting, will re-query FAISS",
                    object_id, cached_poi, cached_sim, self._cfg.similarity_threshold,
                )
                self._cache.delete_object(object_id)
                # Fall through to FAISS search below
            else:
                log.debug("Cache hit: object=%s → poi=%s similarity=%.4f", object_id, cached_poi, cached_sim)
                return MatchResult(
                    poi_id=cached_poi,
                    similarity_score=cached_sim,
                    faiss_distance=0.0,
                )

        # Cache miss — search FAISS
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
        # Cache the result with similarity score
        self._cache.set_poi_for_object(
            object_id, best.poi_id, ttl=self._cfg.object_cache_ttl, similarity=best.similarity_score
        )
        log.info(
            "Match found: object=%s → poi=%s (similarity=%.3f)",
            object_id,
            best.poi_id,
            best.similarity_score,
        )
        return best
