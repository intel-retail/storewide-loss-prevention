"""Alert Service — Observer-based alert dispatch using strategies."""

from __future__ import annotations

import logging
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
        dedup_key = f"{event.object_id}:{poi_id}"
        # Idempotent: check if alert already sent for this object+POI pair
        if self._event_repo.is_alert_sent(dedup_key):
            log.debug("Alert already sent for object=%s poi=%s, skipping", event.object_id, poi_id)
            return

        # Log start time for performance metrics — uses the DLStreamer frame
        # timestamp for true end-to-end latency (frame capture → alert dispatch).
        if user_log_start_time and event.timestamp:
            try:
                frame_ts_ms = int(
                    datetime.fromisoformat(
                        event.timestamp.replace("Z", "+00:00")
                    ).timestamp() * 1000
                )
                user_log_start_time(
                    frame_ts_ms, "USECASE_1", "person-of-interest"
                )
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
            self._event_repo.mark_alert_sent(dedup_key, ttl=self._cfg.alert_dedup_ttl)
            # Log end time for performance metrics
            if log_end_time:
                try:
                    log_end_time("USECASE_1", "person-of-interest")
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

        # Count previous matches
        prev_alerts = self._event_repo.get_events_for_poi(match.poi_id)
        total_previous = len(prev_alerts)

        bbox = [0, 0, 0, 0]
        if center_of_mass:
            x = int(center_of_mass.get("x", 0))
            y = int(center_of_mass.get("y", 0))
            w = int(center_of_mass.get("width", 0))
            h = int(center_of_mass.get("height", 0))
            # bounding_box_px uses top-left origin: [x1, y1, x2, y2]
            bbox = [x, y, x + w, y + h]

        alert_id = f"alert-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{match.poi_id}"

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
