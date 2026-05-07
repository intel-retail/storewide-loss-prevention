"""Detection FAISS index — stores embeddings for every face seen (offline search).

Unlike the enrolled-POI index (FAISSRepository), this index:
- Rebuilds FAISS in-memory from Redis on every restart (full persistence).
- Stores the embedding vector alongside metadata in Redis with a 7-day TTL.
- Grows continuously; old entries expire automatically via Redis TTL.
- Used by POST /api/v1/search to find any person, enrolled or not.

Redis key schema:
  detection:meta:{faiss_id}  →  JSON {camera_id, track_id, timestamp, bbox}
  detection:vec:{faiss_id}   →  raw float32 bytes (256 × 4 = 1024 bytes)
  detection:next_id          →  int counter (persists across restarts for unique IDs)
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Optional

import faiss
import numpy as np
import redis

from backend.core.config import get_config
from backend.domain.interfaces.repository import DetectionIndexRepository as IDetectionIndexRepository

log = logging.getLogger("poi.detection_index")

_REDIS_META_PREFIX = "detection:meta:"
_REDIS_VEC_PREFIX  = "detection:vec:"
_REDIS_NEXT_ID_KEY = "detection:next_id"


class DetectionIndexRepository(IDetectionIndexRepository):
    """In-memory FAISS index for all detections with Redis-backed metadata."""

    def __init__(self, redis_client: redis.Redis) -> None:
        cfg = get_config()
        self._r = redis_client
        self._dim = cfg.faiss_dimension
        self._ttl = cfg.appearance_ttl_days * 86400  # days → seconds
        self._lock = threading.Lock()

        self._track_seen_ttl = cfg.track_seen_ttl  # short gate TTL (not data TTL)

        # Inner-product index on L2-normalised vectors == cosine similarity
        base = faiss.IndexFlatIP(self._dim)
        self._index = faiss.IndexIDMap(base)

        # Restore next_id counter, then rebuild FAISS from stored vectors.
        stored = self._r.get(_REDIS_NEXT_ID_KEY.encode())
        self._next_id: int = int(stored) if stored else 0

        rebuilt = self._rebuild_from_redis()
        log.info(
            "DetectionIndexRepository initialised: dim=%d ttl_days=%d track_seen_ttl=%ds next_id=%d rebuilt=%d",
            self._dim, cfg.appearance_ttl_days, self._track_seen_ttl, self._next_id, rebuilt,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def add(
        self,
        vector: np.ndarray,
        camera_id: str,
        track_id: str,
        timestamp: str,
        bbox: Optional[list],
    ) -> int:
        """Normalise, add to FAISS, store metadata in Redis. Returns faiss_id."""
        vec = _normalize(vector)
        if vec is None:
            log.debug("DetectionIndex.add: zero/invalid vector skipped")
            return -1

        with self._lock:
            faiss_id = self._next_id
            self._next_id += 1
            self._index.add_with_ids(vec, np.array([faiss_id], dtype=np.int64))

        # Persist next_id so restarts don't reuse IDs
        self._r.set(_REDIS_NEXT_ID_KEY.encode(), self._next_id)

        # Store metadata + raw embedding bytes — both with same 7-day TTL
        meta = {
            "camera_id": camera_id,
            "track_id": track_id,
            "timestamp": timestamp,
            "bbox": bbox,
        }
        pipe = self._r.pipeline()
        pipe.setex(f"{_REDIS_META_PREFIX}{faiss_id}".encode(), self._ttl,
                   json.dumps(meta).encode())
        pipe.setex(f"{_REDIS_VEC_PREFIX}{faiss_id}".encode(),  self._ttl,
                   vec.flatten().astype(np.float32).tobytes())
        pipe.execute()

        log.debug(
            "DetectionIndex.add: faiss_id=%d camera=%s track=%s ts=%s",
            faiss_id, camera_id, track_id, timestamp,
        )
        return faiss_id

    def search(self, vector: np.ndarray, top_k: int = 20) -> list[tuple[int, float]]:
        """Return [(faiss_id, similarity_score), ...] for the top_k nearest vectors."""
        vec = _normalize(vector)
        if vec is None:
            return []

        with self._lock:
            n = self._index.ntotal
            if n == 0:
                return []
            k = min(top_k, n)
            distances, ids = self._index.search(vec, k)

        results = []
        for dist, fid in zip(distances[0], ids[0]):
            if fid < 0:
                continue
            # Only return hits whose metadata hasn't expired in Redis
            if self._r.exists(f"{_REDIS_META_PREFIX}{fid}".encode()):
                results.append((int(fid), float(dist)))

        return results

    def get_metadata(self, faiss_id: int) -> Optional[dict]:
        """Return stored metadata for a faiss_id, or None if expired/missing."""
        raw = self._r.get(f"{_REDIS_META_PREFIX}{faiss_id}".encode())
        if raw is None:
            return None
        try:
            text = raw.decode() if isinstance(raw, bytes) else raw
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

    def total_vectors(self) -> int:
        with self._lock:
            return self._index.ntotal

    def store_frame(self, faiss_id: int, b64_jpeg: str) -> None:
        """Store a base64 JPEG frame keyed by faiss_id with 7-day TTL."""
        key = f"detection:frame:{faiss_id}".encode()
        self._r.setex(key, self._ttl, b64_jpeg.encode() if isinstance(b64_jpeg, str) else b64_jpeg)

    def get_frame(self, faiss_id: int) -> Optional[str]:
        """Return the stored base64 JPEG for a faiss_id, or None if expired/missing."""
        raw = self._r.get(f"detection:frame:{faiss_id}".encode())
        if raw is None:
            return None
        return raw.decode() if isinstance(raw, bytes) else raw

    def claim_track(self, track_id: str, ttl: Optional[int] = None) -> bool:
        """Allow up to N embeddings per track, with minimum time between stores.

        Returns True if a new embedding slot is available (count < max and
        cooldown expired), False otherwise.

        Uses track_seen_ttl (default 120s) for the dedup gate, NOT the 7-day
        data TTL — so that when the SceneScape tracker recycles an integer ID
        for a new person, the gate expires in time and the new person is stored.
        """
        cfg = get_config()
        max_per_track = cfg.detection_embeddings_per_track
        interval = cfg.detection_embedding_interval
        effective_ttl = ttl if ttl is not None else cfg.track_seen_ttl

        # Check embedding count for this track
        count_key = f"detection:track:count:{track_id}".encode()
        count_raw = self._r.get(count_key)
        current_count = int(count_raw) if count_raw else 0
        if current_count >= max_per_track:
            return False

        # Enforce minimum interval between stores (cooldown)
        cooldown_key = f"detection:track:cooldown:{track_id}".encode()
        if not self._r.set(cooldown_key, b"1", ex=interval, nx=True):
            return False  # too soon since last store

        # Increment counter with same TTL as the dedup gate
        pipe = self._r.pipeline()
        pipe.incr(count_key)
        pipe.expire(count_key, effective_ttl)
        pipe.execute()
        return True

    # ── Private ─────────────────────────────────────────────────────────────

    def _rebuild_from_redis(self) -> int:
        """Reload all stored vectors from Redis into FAISS. Returns count rebuilt."""
        keys = self._r.keys(f"{_REDIS_VEC_PREFIX}*".encode())
        if not keys:
            return 0

        vectors, ids = [], []
        for key in keys:
            raw = self._r.get(key)
            if raw is None:
                continue
            try:
                # key is bytes: b"detection:vec:42" → faiss_id = 42
                faiss_id = int(key.decode().split(":")[-1])
                arr = np.frombuffer(raw, dtype=np.float32)
                if arr.shape[0] != self._dim:
                    continue
                vectors.append(arr.copy())
                ids.append(faiss_id)
            except Exception:
                log.debug("Skipping malformed vector key %s", key, exc_info=True)

        if not vectors:
            return 0

        mat = np.vstack(vectors).astype(np.float32)
        id_arr = np.array(ids, dtype=np.int64)
        with self._lock:
            self._index.add_with_ids(mat, id_arr)

        log.info("DetectionIndex: rebuilt %d vectors from Redis", len(ids))
        return len(ids)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _normalize(vector: np.ndarray) -> Optional[np.ndarray]:
    """L2-normalise a 1-D vector and return a (1, dim) float32 array, or None."""
    arr = np.array(vector, dtype=np.float32).flatten()
    norm = np.linalg.norm(arr)
    if norm < 1e-10:
        return None
    arr /= norm
    return arr.reshape(1, -1)
