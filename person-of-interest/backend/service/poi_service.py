"""POI Service — business logic for POI CRUD operations."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from backend.domain.entities.poi import POI
from backend.domain.interfaces.repository import EmbeddingMappingRepository, EmbeddingRepository, POIRepository
from backend.factory.factories import EmbeddingModelFactory
from backend.utils.builder import POIBuilder

log = logging.getLogger("poi.service.poi")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/data/uploads"))


class POIService:
    """Orchestrates POI creation, listing, and deletion."""

    def __init__(
        self,
        poi_repo: POIRepository,
        embedding_repo: EmbeddingRepository,
        mapping_repo: EmbeddingMappingRepository,
    ) -> None:
        self._poi_repo = poi_repo
        self._embedding_repo = embedding_repo
        self._mapping_repo = mapping_repo

    async def create_poi(
        self,
        images: list[bytes],
        severity: str = "medium",
        notes: str = "",
    ) -> dict:
        poi_id = POI.generate_id()
        builder = POIBuilder().with_id(poi_id).with_severity(severity).with_notes(notes)
        model = EmbeddingModelFactory.create()

        embeddings = []
        for idx, img_bytes in enumerate(images):
            result = model.generate_from_bytes(img_bytes)
            if "error" in result:
                log.warning("Image %d failed: %s", idx, result["error"])
                continue

            emb_id = f"emb-{poi_id}-ref-{idx:02d}"
            # Save image to disk
            img_dir = UPLOAD_DIR / poi_id
            img_dir.mkdir(parents=True, exist_ok=True)
            img_path = img_dir / f"ref_{idx}.jpg"
            img_path.write_bytes(img_bytes)

            builder.add_image(emb_id, f"/uploads/{poi_id}/ref_{idx}.jpg")
            embeddings.append(np.array(result["embedding"], dtype=np.float32))

        if not embeddings:
            return {"error": "No faces detected in any uploaded image"}

        poi = builder.build()

        # Store vectors in FAISS
        faiss_ids = self._embedding_repo.add(poi_id, embeddings)

        # Map FAISS IDs → POI ID
        for fid in faiss_ids:
            self._mapping_repo.map_faiss_to_poi(fid, poi_id)

        # Save metadata in Redis
        self._poi_repo.save(poi)

        log.info("Created POI %s with %d embeddings", poi_id, len(embeddings))
        return poi.to_dict()

    def list_pois(self) -> list[dict]:
        pois = self._poi_repo.list_all()
        return [p.to_dict() for p in pois]

    def get_poi(self, poi_id: str) -> Optional[dict]:
        poi = self._poi_repo.get(poi_id)
        return poi.to_dict() if poi else None

    def delete_poi(self, poi_id: str) -> bool:
        # Remove metadata first (authoritative source) — if this fails,
        # embeddings remain searchable which is the safer failure mode.
        deleted = self._poi_repo.delete(poi_id)
        if not deleted:
            return False
        # Remove from FAISS
        self._embedding_repo.remove(poi_id)
        # Remove FAISS→POI mappings
        self._mapping_repo.remove_mappings_for_poi(poi_id)
        log.info("Deleted POI %s", poi_id)
        return True
