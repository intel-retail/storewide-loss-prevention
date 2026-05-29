"""FAISS repository — Singleton index with cosine similarity."""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from backend.core.config import get_config
from backend.domain.interfaces.repository import EmbeddingRepository

log = logging.getLogger("poi.faiss")


class FAISSRepository(EmbeddingRepository):
    """Singleton FAISS index using Inner Product (cosine on L2-normed vectors)."""

    _instance: Optional[FAISSRepository] = None
    _lock = threading.Lock()
    _init_lock = threading.Lock()

    def __new__(cls) -> FAISSRepository:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialised = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        with self._init_lock:
            if self._initialised:
                return
            self._do_init()

    def _do_init(self) -> None:
        cfg = get_config()
        self._dim = cfg.faiss_dimension
        self._index_path = cfg.faiss_index_path
        self._id_map_path = cfg.faiss_id_map_path
        self._search_lock = threading.Lock()

        # faiss_id → poi_id mapping (in-memory + persisted)
        self._id_map: dict[int, str] = {}
        self._next_id = 0

        self._index = self._load_or_create()
        self._initialised = True  # must be last — signals init complete

    def _load_or_create(self) -> faiss.IndexIDMap:
        if os.path.exists(self._index_path):
            log.info("Loading FAISS index from %s", self._index_path)
            base = faiss.read_index(self._index_path)
            if os.path.exists(self._id_map_path):
                with open(self._id_map_path) as f:
                    raw = json.load(f)
                    self._id_map = {int(k): v for k, v in raw.items()}
                    self._next_id = max(self._id_map.keys(), default=-1) + 1
            return base
        log.info("Creating new FAISS index (dim=%d)", self._dim)
        flat = faiss.IndexFlatIP(self._dim)
        index = faiss.IndexIDMap(flat)
        return index

    def reconcile(self, valid_poi_ids: set[str]) -> None:
        """Remove FAISS vectors whose poi_id no longer exists in Redis.

        Called at startup to prune stale vectors left from previous
        deployments where Redis was wiped but the FAISS bind-mount persisted.
        """
        with self._search_lock:
            stale_fids = [fid for fid, pid in self._id_map.items()
                          if pid not in valid_poi_ids]
            if not stale_fids:
                return
            stale_pois = {self._id_map[fid] for fid in stale_fids}
            log.warning(
                "Removing %d stale FAISS vectors for %d orphaned POI(s): %s",
                len(stale_fids), len(stale_pois), stale_pois,
            )
            self._index.remove_ids(np.array(stale_fids, dtype=np.int64))
            for fid in stale_fids:
                del self._id_map[fid]
            self._save_to_disk_locked()

    def add(self, poi_id: str, vectors: list[np.ndarray]) -> list[int]:
        ids_assigned = []
        vecs = []
        for v in vectors:
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            vecs.append(v.astype(np.float32))

        if not vecs:
            return ids_assigned

        with self._search_lock:
            for v in vecs:
                fid = self._next_id
                self._next_id += 1
                self._id_map[fid] = poi_id
                ids_assigned.append(fid)
            arr = np.vstack(vecs)
            id_arr = np.array(ids_assigned, dtype=np.int64)
            self._index.add_with_ids(arr, id_arr)
            log.info("Added %d vectors for poi=%s (ids=%s)", len(vecs), poi_id, ids_assigned)
            self._save_to_disk_locked()
        return ids_assigned

    def search(self, vector: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        query = vector.astype(np.float32).reshape(1, -1)
        with self._search_lock:
            if self._index.ntotal == 0:
                return []
            distances, ids = self._index.search(query, min(top_k, self._index.ntotal))
            # Copy id_map lookups while still under lock so results are consistent
            results = []
            for dist, fid in zip(distances[0], ids[0]):
                if fid >= 0:
                    results.append((int(fid), float(dist)))
        return results

    def remove(self, poi_id: str) -> None:
        with self._search_lock:
            ids_to_remove = [fid for fid, pid in self._id_map.items() if pid == poi_id]
            if ids_to_remove:
                self._index.remove_ids(np.array(ids_to_remove, dtype=np.int64))
                for fid in ids_to_remove:
                    del self._id_map[fid]
                log.info("Removed %d vectors for poi=%s", len(ids_to_remove), poi_id)
                self._save_to_disk_locked()

    def get_poi_id_for_faiss_id(self, faiss_id: int) -> Optional[str]:
        with self._search_lock:
            return self._id_map.get(faiss_id)

    def get_indexed_poi_ids(self) -> set[str]:
        """Return the set of poi_ids currently indexed in FAISS."""
        with self._search_lock:
            return set(self._id_map.values())

    def _save_to_disk_locked(self) -> None:
        """Persist index and id_map. Caller MUST hold _search_lock."""
        Path(self._index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, self._index_path)
        with open(self._id_map_path, "w") as f:
            json.dump({str(k): v for k, v in self._id_map.items()}, f)
        log.info("FAISS index saved (%d vectors)", self._index.ntotal)

    def save_to_disk(self) -> None:
        """Public save — acquires lock. Prefer _save_to_disk_locked inside mutations."""
        with self._search_lock:
            self._save_to_disk_locked()

    def total_vectors(self) -> int:
        return self._index.ntotal

    @classmethod
    def reset(cls) -> None:
        cls._instance = None
