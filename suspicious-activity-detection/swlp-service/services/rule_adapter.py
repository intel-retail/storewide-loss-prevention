# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Rule Engine Adapter — bridges the generic rule engine with LP-specific logic.

Responsibilities:
  - Translates RegionEvent + PersonSession → flat context dict
  - Calls RuleEngine.evaluate() (pure, no side effects)
  - Translates Action results → LP-specific Alert objects, BA triggers
  - Owns session state transitions, loiter dedup, poll loop, frame cleanup
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional

import structlog

from models.events import EventType, RegionEvent, ZoneType
from models.alerts import Alert, AlertType, AlertLevel
from .config import ConfigService
from .session_manager import SessionManager
from rule_engine import RuleEngine, Action
from .alert_service_client import AlertServiceClient

logger = structlog.get_logger(__name__)


class RuleEngineAdapter:
    """LP-specific adapter that wires the Rule Engine Service to this service."""

    def __init__(
        self,
        engine: RuleEngine,
        config: ConfigService,
        session_manager: SessionManager,
        alert_service_client: AlertServiceClient | None = None,
        frame_manager=None,
        ba_publisher=None,
    ) -> None:
        self._engine = engine
        self.config = config
        self.session_mgr = session_manager
        self._alert_client = alert_service_client
        self._frame_mgr = frame_manager
        self._ba_publisher = ba_publisher

        rules_cfg = config.get_rules_config()
        self.ba_poll_interval = rules_cfg.get("ba_poll_interval_seconds", 1)
        self._loiter_threshold = float(rules_cfg.get("loiter_threshold_seconds", 20))
        self._pending_cleanups: set[str] = set()  # track scheduled cleanups by "{object_id}:{region_id}"

        logger.info(
            "RuleEngineAdapter initialized",
            ba_poll_interval=self.ba_poll_interval,
            loiter_threshold=self._loiter_threshold,
            rules_loaded=len(engine.rules),
        )

    def set_alert_client(self, client: AlertServiceClient) -> None:
        self._alert_client = client

    # ---- main entry point (same signature as old RuleEngine.on_event) ---------

    async def on_event(self, event: RegionEvent) -> None:
        """Process a region event: update state, evaluate rules, execute actions."""
        if event.event_type == EventType.PERSON_LOST:
            await self._on_person_lost(event)
            return

        session = self.session_mgr.get_session(event.object_id)
        if not session:
            return

        # ---- State transitions (LP-specific, not in the rule engine) ----
        if event.event_type == EventType.ENTERED:
            if event.zone_type == ZoneType.HIGH_VALUE:
                session.visited_high_value = True
                logger.info(
                    "HIGH_VALUE zone entered",
                    object_id=event.object_id,
                    region=event.region_name,
                    visit_count=session.zone_visit_counts.get(event.region_id, 0),
                )
            elif event.zone_type == ZoneType.CHECKOUT:
                session.visited_checkout = True
                logger.info("Checkout visited", object_id=event.object_id)
            elif event.zone_type == ZoneType.EXIT:
                session.visited_exit = True

        elif event.event_type == EventType.EXITED:
            if event.zone_type == ZoneType.HIGH_VALUE:
                cleanup_key = f"{event.object_id}:{event.region_id}"
                if cleanup_key not in self._pending_cleanups:
                    self._pending_cleanups.add(cleanup_key)
                    asyncio.create_task(
                        self._deferred_frame_cleanup(
                            event.object_id, event.region_id, event.region_name, event.dwell_seconds,
                        )
                    )

        elif event.event_type == EventType.LOITER:
            # Loiter detected by continuous region-data feed — fire alert directly
            if session.loiter_alerted.get(event.region_id):
                return
            alert = Alert(
                alert_type=AlertType.LOITERING,
                alert_level=AlertLevel.WARNING,
                object_id=event.object_id,
                timestamp=event.timestamp,
                scene_id=event.scene_id,
                region_id=event.region_id,
                region_name=event.region_name,
                details={
                    "dwell_seconds": event.dwell_seconds,
                    "threshold": self._loiter_threshold,
                    "source": "region_data_feed",
                },
            )
            logger.warning(
                "Loitering detected (region data feed)",
                object_id=event.object_id,
                zone=event.region_name,
                dwell=event.dwell_seconds,
            )
            session.loiter_alerted[event.region_id] = True
            await self._fire_alert(alert)
            return

        # ---- Map event_type to rule trigger string ----
        trigger_event = (
            "zone_entry" if event.event_type == EventType.ENTERED else "zone_exit"
        )

        # ---- Build flat context dict (no LP dataclasses leak into engine) ----
        context = self._build_context(event, session)

        # ---- Evaluate rules (local rule engine) ----
        actions = self._engine.evaluate(trigger_event, event.zone_type.value, context)

        # ---- Execute actions (LP-specific side effects) ----
        await self._execute_actions(actions, event, session)

    # ---- context builder -----------------------------------------------------

    @staticmethod
    def _build_context(event: RegionEvent, session) -> dict:
        """Flatten event + session into a generic dict for the rule engine."""
        return {
            # Event fields
            "region_id": event.region_id,
            "region_name": event.region_name,
            "dwell_seconds": event.dwell_seconds,
            # Session fields
            "visited_high_value": session.visited_high_value,
            "visited_checkout": session.visited_checkout,
            "visited_exit": session.visited_exit,
            "concealment_suspected": session.concealment_suspected,
            "zone_visit_counts": dict(session.zone_visit_counts),
        }

    # ---- action execution (LP-specific) --------------------------------------

    async def _execute_actions(
        self, actions: list[Action], event: RegionEvent, session
    ) -> None:
        for action in actions:
            if action.type == "alert":
                await self._execute_alert(action, event, session)
            elif action.type == "escalate":
                if action.params.get("service") == "behavioral_analysis":
                    await self._trigger_behavioral_analysis(
                        event.object_id, event.region_id
                    )

    async def _execute_alert(
        self, action: Action, event: RegionEvent, session
    ) -> None:
        """Build and fire an LP Alert from a generic Action."""
        alert_type = AlertType[action.params["alert_type"]]
        severity = action.params.get("severity", "WARNING")

        # Concealment upgrades severity (e.g. checkout bypass → CRITICAL)
        if (
            action.params.get("severity_if_concealment")
            and session.concealment_suspected
        ):
            severity = action.params["severity_if_concealment"]

        alert_level = AlertLevel[severity]

        # Dedup: one alert per zone per session for loitering and unusual-path
        if alert_type == AlertType.LOITERING:
            if session.loiter_alerted.get(event.region_id):
                logger.debug(
                    "Loiter alert already fired for zone",
                    object_id=event.object_id,
                    region_id=event.region_id,
                )
                return
        if alert_type == AlertType.UNUSUAL_PATH:
            if session.unusual_path_alerted.get(event.region_id):
                logger.debug(
                    "Unusual-path alert already fired for zone",
                    object_id=event.object_id,
                    region_id=event.region_id,
                )
                return

        details = self._build_details(alert_type, action.params, event, session)

        alert = Alert(
            alert_type=alert_type,
            alert_level=alert_level,
            object_id=event.object_id,
            timestamp=event.timestamp,
            scene_id=event.scene_id,
            region_id=event.region_id,
            region_name=event.region_name,
            details=details,
        )
        logger.warning(
            "Rule fired",
            rule_id=action.rule_id,
            alert_type=alert_type.value,
            level=alert_level.value,
            object_id=event.object_id,
            region=event.region_name,
        )
        await self._fire_alert(alert)

        # Mark as fired for dedup
        if alert_type == AlertType.LOITERING:
            session.loiter_alerted[event.region_id] = True
        if alert_type == AlertType.UNUSUAL_PATH:
            session.unusual_path_alerted[event.region_id] = True

    @staticmethod
    def _build_details(
        alert_type: AlertType, params: dict, event: RegionEvent, session
    ) -> dict:
        """Build contextual details dict for the alert."""
        if alert_type == AlertType.ZONE_VIOLATION:
            return {"zone_type": event.zone_type.value}
        elif alert_type == AlertType.UNUSUAL_PATH:
            return {
                "visit_count": session.zone_visit_counts.get(event.region_id, 0),
                "threshold": params.get("threshold", 2),
            }
        elif alert_type == AlertType.CHECKOUT_BYPASS:
            return {
                "visited_high_value": session.visited_high_value,
                "visited_checkout": session.visited_checkout,
                "concealment_suspected": session.concealment_suspected,
            }
        elif alert_type == AlertType.LOITERING:
            return {
                "dwell_seconds": round(event.dwell_seconds, 1) if event.dwell_seconds else 0,
                "threshold": params.get("threshold", 120),
            }
        return {}

    # ---- active BA polling ---------------------------------------------------

    async def run_ba_check_loop(self) -> None:
        """
        Background task: periodically publish BA requests for persons in HIGH_VALUE zones.
        Publishes to MQTT ba/requests topic; results come back via ba/results.
        """
        if not self._ba_publisher:
            logger.info("No BA publisher configured — skipping BA poll loop")
            return
        if not self._engine.is_rule_enabled("behavioral_analysis"):
            logger.info("Behavioral analysis rule disabled — skipping BA poll loop")
            return

        logger.info(
            "BA poll loop started (queue mode)",
            interval_seconds=self.ba_poll_interval,
        )

        while True:
            await asyncio.sleep(self.ba_poll_interval)
            try:
                for session in self.session_mgr.get_all_sessions().values():
                    for zone_id in list(session.current_zones.keys()):
                        zone_type = self.config.get_zone_type(zone_id)
                        if zone_type != "HIGH_VALUE":
                            continue
                        if session.ba_alerted.get(zone_id):
                            continue
                        self._publish_ba_request(
                            session.object_id, zone_id
                        )
            except Exception:
                logger.exception("Error in BA check loop")

    # ---- Deferred frame cleanup on HIGH_VALUE zone exit ----------------------

    async def _deferred_frame_cleanup(
        self, object_id: str, region_id: str, region_name: str, dwell: float,
    ) -> None:
        """Cleanup disabled for now."""
        logger.info(
            "HIGH_VALUE zone exited — cleanup DISABLED",
            object_id=object_id,
            region=region_name,
            dwell=dwell,
        )
        cleanup_key = f"{object_id}:{region_id}"
        self._pending_cleanups.discard(cleanup_key)

    # ---- PERSON_LOST handler -------------------------------------------------

    async def _on_person_lost(self, event: RegionEvent) -> None:
        """Cleanup disabled for now."""
        logger.info("Person lost — cleanup DISABLED", object_id=event.object_id)

    # ---- External service calls ----------------------------------------------

    def _publish_ba_request(
        self, object_id: str, region_id: str
    ) -> None:
        """Publish a BA analysis request to the MQTT queue."""
        if not self._ba_publisher:
            return

        session = self.session_mgr.get_session(object_id)
        if not session:
            return

        # Skip if already alerted for this zone
        if session.ba_alerted.get(region_id):
            return

        # Get zone entry timestamp for frame path
        entry_ts_iso = session.current_zones.get(region_id, "")
        entry_timestamp = ""
        if entry_ts_iso:
            entry_timestamp = entry_ts_iso.replace(":", "").replace("-", "").split("+")[0].split(".")[0]

        self._ba_publisher.publish_request(
            person_id=object_id,
            region_id=region_id,
            entry_timestamp=entry_timestamp,
            scene_id=session.scene_id,
        )

    async def on_ba_result(self, result: dict) -> None:
        """
        Handle a BA analysis result received from the MQTT ba/results topic.

        This replaces the old synchronous REST response handling.
        """
        person_id = result.get("person_id", "")
        region_id = result.get("region_id", "")
        status = result.get("status", "")

        session = self.session_mgr.get_session(person_id)
        if not session:
            logger.debug("BA result for unknown session", person_id=person_id)
            return

        if status == "suspicious":
            session.concealment_suspected = True
            session.ba_alerted[region_id] = True
            zone_name = self.config.get_zone_name(region_id)
            alert = Alert(
                alert_type=AlertType.CONCEALMENT,
                alert_level=AlertLevel.WARNING,
                object_id=person_id,
                timestamp=session.last_seen,
                scene_id=session.scene_id,
                region_id=region_id,
                region_name=zone_name,
                details={
                    "confidence": result.get("confidence"),
                    "message": result.get("vlm_response", ""),
                    "frames_analyzed": result.get("frames_analyzed", 0),
                },
            )
            logger.warning(
                "BA queue: concealment detected",
                person_id=person_id,
                region_id=region_id,
                confidence=result.get("confidence"),
            )
            await self._fire_alert(alert)
        elif status == "no_match":
            logger.debug(
                "BA queue: no suspicious pattern",
                person_id=person_id,
                region_id=region_id,
            )
        elif status == "received":
            logger.debug(
                "BA queue: request acknowledged",
                person_id=person_id,
                region_id=region_id,
            )
        else:
            logger.debug(
                "BA queue: status update",
                person_id=person_id,
                status=status,
            )

    # ---- alert dispatch ------------------------------------------------------

    async def _fire_alert(self, alert: Alert) -> None:
        """Persist evidence frames and send to alert-service."""
        if self._frame_mgr and not alert.evidence_keys:
            frame_keys = self._frame_mgr.get_person_frame_keys(alert.object_id)
            if frame_keys:
                stored = []
                for idx, key in enumerate(frame_keys):
                    raw = self._frame_mgr.get_frame(key)
                    if raw:
                        ev_key = self._frame_mgr.store_evidence_frame(
                            alert.alert_id, idx, raw
                        )
                        stored.append(ev_key)
                if stored:
                    alert.evidence_keys = stored
                    logger.info(
                        "Evidence frames stored",
                        alert_id=alert.alert_id,
                        count=len(stored),
                    )

        # Send to alert-service (handles MQTT delivery with dedup)
        if self._alert_client:
            try:
                await self._alert_client.publish_alert(alert)
            except Exception:
                logger.exception("AlertService publish error", alert_id=alert.alert_id)

        logger.warning(
            "ALERT",
            alert_id=alert.alert_id,
            type=alert.alert_type.value,
            level=alert.alert_level.value,
            object_id=alert.object_id,
            region=alert.region_name,
            details=alert.details,
        )
