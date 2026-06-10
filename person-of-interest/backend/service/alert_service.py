"""Alert Service — Observer-based alert dispatch using strategies."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from backend.core.config import get_config
from backend.domain.entities.match_result import AlertPayload, MatchResult
from backend.domain.interfaces.alert import AlertStrategy
from backend.domain.interfaces.repository import EventRepository, POIRepository
from backend.observer.events import EventBus, MatchFoundEvent

try:
    from vlm_metrics_logger import log_end_time, user_log_start_time
except ImportError:
    log_end_time = None
    user_log_start_time = None

log = logging.getLogger("poi.service.alert")


class AlertService:
    """Manages alert generation and dispatch via configured strategies.

    Subscribes to MatchFoundEvent via the EventBus (Observer Pattern)
    and dispatches through all registered AlertStrategy instances.
    """

    def __init__(
        self,
        strategies: list[AlertStrategy],
        event_repo: EventRepository,
        poi_repo: POIRepository,
        event_bus: EventBus,
    ) -> None:
        self._strategies = strategies
        self._event_repo = event_repo
        self._poi_repo = poi_repo
        self._event_bus = event_bus
        self._cfg = get_config()

        # Register as observer
        self._event_bus.subscribe("match_found", self._on_match_found)

    def _on_match_found(self, event: MatchFoundEvent) -> None:
        """Observer callback when a POI match is found."""
        poi_id = event.alert.poi_id
        # Dedup strategy depends on whether the object_id is a stable
        # SceneScape UUID or a camera-local fallback like ``cam:Camera_01:1``.
        #
        # Stable UUIDs are unique per physical person, so dedup is per-object
        # only — one alert per person per dedup window, regardless of which
        # POI matched.  This prevents cross-POI flapping when FAISS returns
        # different results on re-query.
        #
        # Camera-local ``cam:*`` IDs are recycled across different people, so
        # per-object dedup would suppress alerts for completely different
        # individuals.  For these we dedup per object+poi, accepting the
        # (rare) risk of cross-POI flapping in exchange for not missing
        # real alerts.
        if event.object_id.startswith("cam:"):
            dedup_key = f"{event.object_id}:{poi_id}"
        else:
            dedup_key = event.object_id
        # Idempotent: check if alert already sent for this key in this window
        if self._event_repo.is_alert_sent(dedup_key):
            log.debug("Alert already sent for object=%s (matched poi=%s), skipping", event.object_id, poi_id)
            return

        # Log start time for performance metrics — uses the DLStreamer frame
        # timestamp for true end-to-end latency (frame capture → alert dispatch).
        # Two entries are written per alert:
        #   unique_id="person-of-interest" → aggregate across all cameras
        #   unique_id=camera_id            → per-camera (used by stream density
        #                                    benchmark to isolate new-camera E2E)
        camera_id = event.alert.match.get("camera_id", "")
        if user_log_start_time and event.timestamp:
            try:
                frame_ts_ms = int(
                    datetime.fromisoformat(
                        event.timestamp.replace("Z", "+00:00")
                    ).timestamp() * 1000
                )
                user_log_start_time(frame_ts_ms, "USECASE_1", "person-of-interest")
                if camera_id:
                    user_log_start_time(frame_ts_ms, "USECASE_1", camera_id)
            except Exception:
                log.debug("Failed to log start time for alert=%s", event.alert.alert_id)

        # Dispatch to all strategies; only mark sent if ALL succeed
        all_delivered = True
        for strategy in self._strategies:
            try:
                strategy.send(event.alert)
                log.info("Alert dispatched via %s: %s", strategy.name(), event.alert.alert_id)
            except Exception:
                log.exception("Failed to dispatch alert via %s", strategy.name())
                all_delivered = False

        # Only persist and mark sent when ALL strategies delivered, so a transient
        # alert-service outage doesn't permanently suppress the alert.
        if all_delivered:
            self._event_repo.store_alert(event.alert.to_dict())
            # Short dedup TTL for cam:* IDs (recycled) to avoid suppressing
            # alerts for different people; full TTL for stable UUIDs.
            ttl = 30 if event.object_id.startswith("cam:") else self._cfg.alert_dedup_ttl
            self._event_repo.mark_alert_sent(dedup_key, ttl=ttl)
            # Log end time for performance metrics (aggregate + per-camera)
            if log_end_time:
                try:
                    log_end_time("USECASE_1", "person-of-interest")
                    if camera_id:
                        log_end_time("USECASE_1", camera_id)
                except Exception:
                    log.debug("Failed to log end time for alert=%s", event.alert.alert_id)

    def create_alert_payload(
        self,
        match: MatchResult,
        object_id: str,
        timestamp: str,
        camera_id: str,
        region_name: str,
        confidence: float,
        center_of_mass: Optional[dict] = None,
        thumbnail_path: str = "",
        mqtt_receive_time_ms: int = 0,
    ) -> AlertPayload:
        """Build an AlertPayload from a match result."""
        poi = self._poi_repo.get(match.poi_id)
        severity = poi.severity.value if poi else "medium"
        poi_name = poi.notes if poi else ""
        enrollment_date = poi.created_at if poi else ""

        # Count previous alerts (not movement events) for accuracy
        total_previous = self._event_repo.get_alert_count_for_poi(match.poi_id)

        bbox = [0, 0, 0, 0]
        if center_of_mass:
            x = int(center_of_mass.get("x", 0))
            y = int(center_of_mass.get("y", 0))
            w = int(center_of_mass.get("width", 0))
            h = int(center_of_mass.get("height", 0))
            # bounding_box_px uses top-left origin: [x1, y1, x2, y2]
            bbox = [x, y, x + w, y + h]

        alert_id = f"alert-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{match.poi_id}-{uuid.uuid4().hex[:8]}"

        # Convert mqtt_receive_time_ms to ISO timestamp
        mqtt_received_at = ""
        if mqtt_receive_time_ms:
            mqtt_received_at = datetime.fromtimestamp(
                mqtt_receive_time_ms / 1000, tz=timezone.utc
            ).isoformat(timespec="milliseconds")

        return AlertPayload(
            alert_id=alert_id,
            poi_id=match.poi_id,
            severity=severity,
            timestamp=timestamp,
            match={
                "camera_id": camera_id,
                "confidence": confidence,
                "similarity_score": match.similarity_score,
                "bbox": bbox,
                "frame_number": 0,
                "thumbnail_path": thumbnail_path,
            },
            poi_metadata={
                "name": poi_name,
                "notes": "",
                "enrollment_date": enrollment_date,
                "total_previous_matches": total_previous,
            },
            mqtt_received_at=mqtt_received_at,
        )

    def get_recent_alerts(self, limit: int = 50) -> list[dict]:
        return self._event_repo.get_recent_alerts(limit)
