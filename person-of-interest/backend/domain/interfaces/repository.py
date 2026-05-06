"""Repository interfaces (ports) for the domain layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from backend.domain.entities.poi import POI
from backend.domain.entities.event import MovementEvent


class POIRepository(ABC):
    """Interface for POI metadata storage."""

    @abstractmethod
    def save(self, poi: POI) -> None: ...

    @abstractmethod
    def get(self, poi_id: str) -> Optional[POI]: ...

    @abstractmethod
    def list_all(self) -> list[POI]: ...

    @abstractmethod
    def delete(self, poi_id: str) -> bool: ...

    @abstractmethod
    def update_status(self, poi_id: str, status: str) -> None: ...


class EmbeddingRepository(ABC):
    """Interface for vector storage (FAISS)."""

    @abstractmethod
    def add(self, poi_id: str, vectors: list[np.ndarray]) -> list[int]: ...

    @abstractmethod
    def search(self, vector: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]: ...

    @abstractmethod
    def remove(self, poi_id: str) -> None: ...

    @abstractmethod
    def get_poi_id_for_faiss_id(self, faiss_id: int) -> Optional[str]: ...

    @abstractmethod
    def save_to_disk(self) -> None: ...

    @abstractmethod
    def total_vectors(self) -> int: ...


class CacheRepository(ABC):
    """Interface for object_id → poi_id cache (Cache-Aside pattern)."""

    @abstractmethod
    def get_poi_for_object(self, object_id: str) -> Optional[str]: ...

    def get_similarity_for_object(self, object_id: str) -> Optional[float]:
        """Return the cached similarity score, or None if not stored."""
        return None

    @abstractmethod
    def set_poi_for_object(self, object_id: str, poi_id: str, ttl: int = 300, similarity: float = 0.0) -> None: ...

    def delete_object(self, object_id: str) -> None:
        """Evict a cache entry. Default no-op; override in concrete implementations."""
        pass


class EventRepository(ABC):
    """Interface for movement event storage."""

    @abstractmethod
    def store_event(self, event: MovementEvent) -> None: ...

    @abstractmethod
    def get_events_for_poi(
        self, poi_id: str, start_time: Optional[str] = None, end_time: Optional[str] = None
    ) -> list[dict]: ...

    @abstractmethod
    def get_recent_alerts(self, limit: int = 50) -> list[dict]: ...

    @abstractmethod
    def store_alert(self, alert: dict) -> None: ...

    @abstractmethod
    def is_alert_sent(self, object_id: str) -> bool: ...

    @abstractmethod
    def mark_alert_sent(self, object_id: str, ttl: int = 300) -> None: ...

    @abstractmethod
    def store_thumbnail(self, object_id: str, b64_jpeg: str, ttl: int = 3600) -> None: ...

    @abstractmethod
    def get_thumbnail(self, object_id: str) -> Optional[str]: ...

    @abstractmethod
    def claim_thumbnail(self, object_id: str, ttl: int = 30) -> bool:
        """Atomically claim the right to capture thumbnail. Returns True if claim succeeded."""
        ...

    @abstractmethod
    def get_region_presence(self, object_id: str, scene_id: str, region_id: str) -> Optional[dict]: ...

    @abstractmethod
    def delete_region_presence(self, object_id: str, scene_id: str, region_id: str) -> None: ...

    @abstractmethod
    def store_region_dwell(self, object_id: str, timestamp: str, scene_id: str, region_id: str, region_name: str,
                           dwell_sec: Optional[float] = None, entry_time: Optional[str] = None,
                           camera_id: Optional[str] = None) -> None: ...

    def get_region_dwells_for_object(self, object_id: str, date_filter: Optional[str] = None) -> list[dict]:
        """Return region dwell records for an object, optionally filtered by date."""
        return []

    def get_track_poi_counts(self, track_id: str) -> dict[str, int]:
        """Return {poi_id: event_count} for all POIs that have events on this track."""
        return {}

    def set_reid_matched(self, camera_id: str, global_uuid: str, metadata: dict, ttl: int = 15) -> None:
        """Store SceneScape reid_state=matched signal for a camera (with TTL)."""

    def get_reid_matched_uuid(self, camera_id: str) -> Optional[str]:
        """Return the current matched global UUID for a camera, or None if no recent signal."""
        return None

    def set_match_metadata(self, object_id: str, metadata: dict, ttl: int = 3600) -> None:
        """Persist reid/match metadata for a matched object."""

    def get_match_metadata(self, object_id: str) -> Optional[dict]:
        """Retrieve stored match metadata for an object."""
        return None


class EmbeddingMappingRepository(ABC):
    """Interface for mapping FAISS internal IDs to POI IDs."""

    @abstractmethod
    def map_faiss_to_poi(self, faiss_id: int, poi_id: str) -> None: ...

    @abstractmethod
    def get_poi_for_faiss(self, faiss_id: int) -> Optional[str]: ...

    @abstractmethod
    def remove_mappings_for_poi(self, poi_id: str) -> None: ...


class DetectionIndexRepository(ABC):
    """Interface for the all-detections FAISS index used by offline search.

    Stores a face embedding for every person detected (not just enrolled POIs).
    Metadata linking each vector back to its camera/track/time is stored in Redis
    with a 7-day TTL and cleaned up automatically.
    """

    @abstractmethod
    def add(
        self,
        vector: np.ndarray,
        camera_id: str,
        track_id: str,
        timestamp: str,
        bbox: Optional[list],
    ) -> int:
        """Add a detection vector and return the assigned faiss_id."""
        ...

    @abstractmethod
    def search(self, vector: np.ndarray, top_k: int = 20) -> list[tuple[int, float]]:
        """Return [(faiss_id, similarity_score), ...] sorted by descending similarity."""
        ...

    @abstractmethod
    def get_metadata(self, faiss_id: int) -> Optional[dict]:
        """Return stored metadata dict for a faiss_id, or None if expired/missing."""
        ...

    @abstractmethod
    def total_vectors(self) -> int:
        """Return current number of vectors in the index."""
        ...

    @abstractmethod
    def claim_track(self, track_id: str, ttl: Optional[int] = None) -> bool:
        """Atomically claim a track ID (Redis NX). Returns True only the first time.

        Used to deduplicate: one embedding per track, not one per frame.
        """
        ...
