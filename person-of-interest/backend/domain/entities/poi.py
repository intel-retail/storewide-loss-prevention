"""POI domain entity."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class POIStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


@dataclass
class ReferenceImage:
    source: str
    embedding_id: str
    vector_dim: int
    image_path: str


@dataclass
class POI:
    poi_id: str
    severity: Severity
    notes: str
    reference_images: list[ReferenceImage]
    status: POIStatus = POIStatus.ACTIVE
    enrolled_by: str = "system"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    embedding_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "event_type": "poi_enrollment",
            "timestamp": self.created_at,
            "poi_id": self.poi_id,
            "enrolled_by": self.enrolled_by,
            "severity": self.severity.value,
            "notes": self.notes,
            "reference_images": [
                {
                    "source": img.source,
                    "embedding_id": img.embedding_id,
                    "vector_dim": img.vector_dim,
                    "image_path": img.image_path,
                }
                for img in self.reference_images
            ],
            "status": self.status.value,
        }

    @staticmethod
    def generate_id() -> str:
        return f"poi-{uuid.uuid4().hex[:8]}"
