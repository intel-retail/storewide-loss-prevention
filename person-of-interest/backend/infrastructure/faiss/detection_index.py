"""Detection FAISS index — stores embeddings for every face seen (offline search).

Unlike the enrolled-POI index (FAISSRepository), this index:
- Rebuilds FAISS in-memory from Redis on every restart (full persistence).
- Stores the embedding vector alongside metadata in Redis with a 7-day TTL.
- Grows continuously; old entries expire automatically via Redis TTL.
- Used by POST /api/v1/search to find any person, enrolled or not.

Redis key schema:
  detection:meta:{faiss_id}      →  JSON {camera_id, track_id, timestamp, bbox, role}
  detection:vec:{faiss_id}       →  raw float32 bytes (256 × 4 = 1024 bytes)
  detection:frame:{faiss_id}     →  base64 JPEG frame at detection moment
  detection:next_id              →  int counter (persists across restarts for unique IDs)
  detection:exit_vec:{track_id}  →  rolling exit embedding (overwritten each detection, track_seen_ttl)
  detection:exit_meta:{track_id} →  JSON metadata for the rolling exit vector
  detection:exit_frame:{track_id}→  base64 JPEG frame for the rolling exit
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
_REDIS_EXIT_VEC_PREFIX  = "detection:exit_vec:"
_REDIS_EXIT_META_PREFIX = "detection:exit_meta:"
_REDIS_EXIT_FRAME_PREFIX = "detection:exit_frame:"


class DetectionIndexRepository(IDetectionIndexRepository):
    """In-memory FAISS index for all detections with Redis-backed metadata."""

    def __init__(self, redis_client: redis.Redis) -> None:
        cfg = get_config()
        self._r = redis_client
        self._dim = cfg.faiss_dimension
        self._ttl = cfg.appearance_ttl_days * 86400  # days → seconds
        self._lock = threading.Lock()

        self._track_seen_ttl = cfg.track_seen_ttl  # gate TTL (not data TTL)
        # Exit vectors survive gate_ttl + 5 min so the promoter can read them
        # after the gate expires (confirming person has left) before promoting.
        self._exit_ttl = cfg.track_seen_ttl + 300

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

    # ── Rolling exit vector (overwritten every detection) ───────────────────

    def update_exit(
        self,
        track_id: str,
        vector: np.ndarray,
        camera_id: str,
        timestamp: str,
        bbox: Optional[list],
        b64_frame: Optional[str] = None,
    ) -> None:
        """Overwrite the rolling exit embedding for this tracker track.

        Called on EVERY detection (not just the first), so the stored vector
        is always the most recent face seen for this track.  TTL = track_seen_ttl
        so the exit vector expires when the tracker gate expires — it belongs to
        exactly one person's appearance window.
        """
        vec = _normalize(vector)
        if vec is None:
            return
        meta = {
            "camera_id": camera_id,
            "track_id": track_id,
            "timestamp": timestamp,
            "bbox": bbox,
            "role": "exit",
        }
        ttl = self._exit_ttl  # gate_ttl + 300s buffer for promoter to read after gate expires
        pipe = self._r.pipeline()
        pipe.setex(f"{_REDIS_EXIT_VEC_PREFIX}{track_id}".encode(), ttl,
                   vec.flatten().astype(np.float32).tobytes())
        pipe.setex(f"{_REDIS_EXIT_META_PREFIX}{track_id}".encode(), ttl,
                   json.dumps(meta).encode())
        if b64_frame:
            pipe.setex(f"{_REDIS_EXIT_FRAME_PREFIX}{track_id}".encode(), ttl,
                       b64_frame.encode() if isinstance(b64_frame, str) else b64_frame)
        pipe.execute()

    def search_exits(
        self, query_vec: np.ndarray, track_ids: list[str]
    ) -> dict[str, float]:
        """Compute cosine similarity of query_vec against exit vectors for the given track_ids.

        Returns {track_id: similarity} for tracks that have an exit vector.
        Only computes against the small set of track_ids already found by FAISS —
        so this is always fast regardless of total track count.
        """
        vec = _normalize(query_vec)
        if vec is None:
            return {}
        results: dict[str, float] = {}
        for track_id in track_ids:
            raw = self._r.get(f"{_REDIS_EXIT_VEC_PREFIX}{track_id}".encode())
            if raw is None:
                continue
            exit_vec = np.frombuffer(raw, dtype=np.float32).reshape(1, -1)
            norm = np.linalg.norm(exit_vec)
            if norm < 1e-9:
                continue
            exit_vec = exit_vec / norm
            similarity = float(np.dot(vec, exit_vec.T)[0, 0])
            results[track_id] = round(similarity, 4)
        return results

    def get_exit_meta(self, track_id: str) -> Optional[dict]:
        """Return stored exit metadata for a track_id, or None if expired/missing."""
        raw = self._r.get(f"{_REDIS_EXIT_META_PREFIX}{track_id}".encode())
        if raw is None:
            return None
        try:
            text = raw.decode() if isinstance(raw, bytes) else raw
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

    def get_exit_frame_url_key(self, track_id: str) -> Optional[str]:
        """Return the Redis key for the exit frame if it exists, else None."""
        key = f"{_REDIS_EXIT_FRAME_PREFIX}{track_id}"
        if self._r.exists(key.encode()):
            return key
        return None

    def promote_exits(self) -> int:
        """Scan all pending exit vectors and promote those whose gate has expired.

        A gate expiry means the person has left (tracker stopped emitting the ID).
        The exit vector then represents their final face look and is added to FAISS
        permanently with role='exit', giving a second embedding per appearance.

        Uses a Redis NX key `detection:exit_promoted:{track_id}` (same exit_ttl) to
        prevent double-promotion if the promoter runs multiple times before the
        exit_vec itself expires.

        Returns the number of new exit embeddings promoted.
        """
        promoted = 0
        exit_keys = self._r.keys(f"{_REDIS_EXIT_VEC_PREFIX}*".encode())
        for key in exit_keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            track_id = key_str[len(_REDIS_EXIT_VEC_PREFIX):]

            # Skip if gate still alive — person is still in frame
            gate_key = f"detection:track:seen:{track_id}".encode()
            if self._r.exists(gate_key):
                continue

            # Skip if already promoted in this window
            promoted_key = f"detection:exit_promoted:{track_id}".encode()
            if not self._r.set(promoted_key, b"1", ex=self._exit_ttl, nx=True):
                continue

            # Read the exit vector
            raw_vec = self._r.get(key)
            if raw_vec is None:
                continue
            vec = np.frombuffer(raw_vec, dtype=np.float32).copy()
            vec = _normalize(vec.reshape(1, -1))
            if vec is None:
                continue

            # Read exit metadata
            meta_raw = self._r.get(f"{_REDIS_EXIT_META_PREFIX}{track_id}".encode())
            if meta_raw is None:
                continue
            try:
                meta = json.loads(meta_raw.decode() if isinstance(meta_raw, bytes) else meta_raw)
            except (json.JSONDecodeError, ValueError):
                continue

            # Add to FAISS with role='exit' in metadata
            with self._lock:
                faiss_id = self._next_id
                self._next_id += 1
                self._index.add_with_ids(vec, np.array([faiss_id], dtype=np.int64))

            self._r.set(_REDIS_NEXT_ID_KEY.encode(), self._next_id)

            meta["role"] = "exit"
            pipe = self._r.pipeline()
            pipe.setex(f"{_REDIS_META_PREFIX}{faiss_id}".encode(), self._ttl,
                       json.dumps(meta).encode())
            pipe.setex(f"{_REDIS_VEC_PREFIX}{faiss_id}".encode(), self._ttl,
                       vec.flatten().astype(np.float32).tobytes())
            # Copy exit frame to a permanent per-faiss_id frame key
            frame_raw = self._r.get(f"{_REDIS_EXIT_FRAME_PREFIX}{track_id}".encode())
            if frame_raw:
                pipe.setex(f"detection:frame:{faiss_id}".encode(), self._ttl, frame_raw)
            pipe.execute()

            promoted += 1
            log.info("Promoted exit embedding: track=%s faiss_id=%d", track_id, faiss_id)

        return promoted

    def claim_track(self, track_id: str, ttl: Optional[int] = None) -> bool:
        """Atomically mark a track as stored (NX). Returns True only the first time.

        Used to deduplicate: one embedding per tracker track, not one per frame.
        Uses track_seen_ttl (default 120s), NOT the 7-day data TTL — so that when
        the SceneScape tracker recycles an integer ID for a new person, the gate
        expires in time and the new person is stored as a distinct detection.
        """
        effective_ttl = ttl if ttl is not None else self._track_seen_ttl
        key = f"detection:track:seen:{track_id}".encode()
        return bool(self._r.set(key, b"1", ex=effective_ttl, nx=True))

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
