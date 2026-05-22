"""Match result domain entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class MatchResult:
    poi_id: str
    similarity_score: float
    faiss_distance: float
    embedding_id: Optional[str] = None

    @property
    def is_match(self) -> bool:
        return self.similarity_score > 0


@dataclass
class AlertPayload:
    alert_id: str
    poi_id: str
    severity: str
    timestamp: str  # DLStreamer frame timestamp (when camera captured the frame)
    match: dict
    poi_metadata: dict
    status: str = "New"
    dispatched_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    )
    mqtt_received_at: str = ""  # When POI backend received the MQTT message

    def to_dict(self) -> dict:
        d = {
            "event_type": "poi_match_alert",
            "timestamp": self.timestamp,
            "dispatched_at": self.dispatched_at,
            "alert_id": self.alert_id,
            "poi_id": self.poi_id,
            "severity": self.severity,
            "match": self.match,
            "poi_metadata": self.poi_metadata,
            "status": self.status,
        }
        if self.mqtt_received_at:
            d["mqtt_received_at"] = self.mqtt_received_at
        return d
