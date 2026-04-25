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
        frame_capture=None,
    ) -> None:
        self._engine = engine
        self.config = config
        self.session_mgr = session_manager
        self._alert_client = alert_service_client
        self._frame_mgr = frame_manager
        self._ba_publisher = ba_publisher
        self._frame_capture = frame_capture

        rules_cfg = config.get_rules_config()
        self.ba_poll_interval = rules_cfg.get("ba_poll_interval_seconds", 1)
        self._loiter_threshold = float(rules_cfg.get("loiter_threshold_seconds", 20))
        self._pending_cleanups: set[str] = set()  # track scheduled cleanups by "{object_id}:{region_id}"

        # Load service definitions for escalation actions (mqtt_topic + payload templates)
        self._services: dict[str, dict] = self._load_services(config)

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

        session = self.session_mgr.get_session(event.object_id, event.scene_id)
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
                # Stop frame capture for this person+zone
                if self._frame_capture:
                    self._frame_capture.stop_capture(event.object_id, event.region_id)

                cleanup_key = f"{event.object_id}:{event.region_id}"
                if cleanup_key not in self._pending_cleanups:
                    self._pending_cleanups.add(cleanup_key)
                    asyncio.create_task(
                        self._deferred_frame_cleanup(
                            event.object_id, event.region_id, event.region_name,
                            event.dwell_seconds, event.scene_id,
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
        trigger_map = {
            EventType.ENTERED: "zone_entry",
            EventType.EXITED: "zone_exit",
            EventType.CONCEALMENT_DETECTED: "concealment_detected",
        }
        trigger_event = trigger_map.get(event.event_type, "zone_exit")

        # ---- Build flat context dict (no LP dataclasses leak into engine) ----
        context = self._build_context(event, session)

        # ---- Evaluate rules (local rule engine) ----
        actions = self._engine.evaluate(trigger_event, event.zone_type.value, context)

        # ---- Execute actions (LP-specific side effects) ----
        await self._execute_actions(actions, event, session)

    # ---- context builder -----------------------------------------------------

    def _build_context(self, event: RegionEvent, session) -> dict:
        """Flatten event + session into a generic dict for the rule engine."""
        # Zone occupancy: count persons currently in this region
        zone_occupancy = sum(
            1 for s in self.session_mgr.get_all_sessions().values()
            if event.region_id in s.current_zones
        )

        # Region visit history: list of past visits to this region
        region_history = [
            {
                "entry_time": v.entry_time.isoformat(),
                "exit_time": v.exit_time.isoformat() if v.exit_time else None,
                "duration_seconds": v.duration_seconds,
            }
            for v in session.region_visits
            if v.region_id == event.region_id
        ]

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
            # Enriched fields
            "zone_occupancy": zone_occupancy,
            "region_visit_count": len(region_history),
            "region_history": region_history,
            "total_persons_in_zone": zone_occupancy,
            "concealment_count": session.concealment_count.get(event.region_id, 0),
            # BA result fields
            "ba_confidence": session.last_ba_confidence,
            "ba_frames_analyzed": session.last_ba_frames_analyzed,
        }

    # ---- action execution (LP-specific) --------------------------------------

    async def _execute_actions(
        self, actions: list[Action], event: RegionEvent, session
    ) -> None:
        for action in actions:
            if action.type == "alert":
                await self._execute_alert(action, event, session)
            elif action.type == "escalate":
                self._execute_escalation(action, event)

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

        # Dedup: one alert per zone per session for loitering and repeated-visit
        if alert_type == AlertType.LOITERING:
            if session.loiter_alerted.get(event.region_id):
                logger.debug(
                    "Loiter alert already fired for zone",
                    object_id=event.object_id,
                    region_id=event.region_id,
                )
                return
        if alert_type == AlertType.REPEATED_VISIT:
            if session.repeated_visit_alerted.get(event.region_id):
                logger.debug(
                    "Repeated-visit alert already fired for zone",
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
        if alert_type == AlertType.REPEATED_VISIT:
            session.repeated_visit_alerted[event.region_id] = True

    @staticmethod
    def _build_details(
        alert_type: AlertType, params: dict, event: RegionEvent, session
    ) -> dict:
        """Build contextual details dict for the alert."""
        if alert_type == AlertType.ZONE_VIOLATION:
            return {"zone_type": event.zone_type.value}
        elif alert_type == AlertType.REPEATED_VISIT:
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

    # ---- Deferred frame cleanup on HIGH_VALUE zone exit ----------------------

    async def _deferred_frame_cleanup(
        self, object_id: str, region_id: str, region_name: str, dwell: float,
        scene_id: str = "",
    ) -> None:
        """Delete BA frames for this person+zone after a delay on exit."""
        await asyncio.sleep(200)
        logger.info(
            "HIGH_VALUE zone exited — deleting BA frames",
            object_id=object_id,
            region=region_name,
            dwell=dwell,
        )
        self._delete_ba_frames(object_id, region_id, scene_id)
        cleanup_key = f"{object_id}:{region_id}"
        self._pending_cleanups.discard(cleanup_key)

    # ---- PERSON_LOST handler -------------------------------------------------

    async def _on_person_lost(self, event: RegionEvent) -> None:
        """Cleanup disabled for now."""
        logger.info("Person lost — cleanup DISABLED", object_id=event.object_id)

    # ---- External service calls ----------------------------------------------

    def _delete_ba_frames(
        self, object_id: str, region_id: str, scene_id: str
    ) -> None:
        """Delete behavioral-analysis frames for a person+zone after alert."""
        if not self._frame_mgr:
            return
        prefix = f"{scene_id}/{object_id}/{region_id}/" if scene_id else f"{object_id}/{region_id}/"
        try:
            self._frame_mgr._delete_prefix(prefix, bucket=self._frame_mgr.BA_BUCKET)
            logger.info(
                "Deleted BA frames after concealment",
                object_id=object_id,
                region_id=region_id,
                prefix=prefix,
            )
        except Exception:
            logger.exception("Failed to delete BA frames", prefix=prefix)

    @staticmethod
    def _load_services(config: ConfigService) -> dict[str, dict]:
        """Load the 'services' section from rules.yaml."""
        import yaml
        path = config.get_rules_yaml_path()
        if not path.exists():
            return {}
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        services = data.get("services", {})
        if services:
            logger.info("Loaded escalation services", services=list(services.keys()))
        return services

    def _execute_escalation(
        self, action: Action, event: RegionEvent,
    ) -> None:
        """Resolve service config from YAML, build payload, publish via MQTT."""
        if not self._ba_publisher:
            return

        service_name = action.params.get("service", "")
        if not service_name:
            logger.warning("Escalate action missing 'service'", rule_id=action.rule_id)
            return

        svc_cfg = self._services.get(service_name)
        if not svc_cfg:
            logger.warning(
                "Unknown escalation service",
                service=service_name,
                rule_id=action.rule_id,
            )
            return

        mqtt_topic = svc_cfg.get("mqtt_topic", "")
        if not mqtt_topic:
            logger.warning("Service missing mqtt_topic", service=service_name)
            return

        session = self.session_mgr.get_session(event.object_id, scene_id=event.scene_id)
        if not session:
            return

        # Build substitution context for {placeholders} in payload values
        entry_ts_iso = session.current_zones.get(event.region_id, "")
        entry_timestamp = ""
        if entry_ts_iso:
            entry_timestamp = entry_ts_iso.replace(":", "").replace("-", "").split("+")[0].split(".")[0]

        subs = {
            "object_id": event.object_id,
            "region_id": event.region_id,
            "scene_id": event.scene_id or session.scene_id,
            "entry_timestamp": entry_timestamp,
            "region_name": event.region_name,
        }

        # Build payload from service template, substituting {key} placeholders
        payload_template = svc_cfg.get("payload", {})
        payload = {
            k: v.format_map(subs) if isinstance(v, str) else v
            for k, v in payload_template.items()
        }

        # Start frame capture — BA request will be published automatically
        # when min_frames are stored (via _on_batch_ready callback)
        if self._frame_capture:
            self._frame_capture.start_capture(
                event.object_id, event.region_id,
                scene_id=subs["scene_id"],
                entry_timestamp=subs["entry_timestamp"],
            )

        logger.info(
            "Frame capture started for escalation (BA request deferred until frames ready)",
            rule_id=action.rule_id,
            service=service_name,
            topic=mqtt_topic,
        )

    def _on_batch_ready(
        self, person_id: str, region_id: str, scene_id: str, entry_timestamp: str,
    ) -> None:
        """Called by FrameCaptureService when min_frames are stored."""
        svc_cfg = self._services.get("behavioral_analysis", {})
        mqtt_topic = svc_cfg.get("mqtt_topic", "ba/requests")

        payload_template = svc_cfg.get("payload", {})
        subs = {
            "object_id": person_id,
            "region_id": region_id,
            "scene_id": scene_id,
            "entry_timestamp": entry_timestamp,
        }
        payload = {
            k: v.format_map(subs) if isinstance(v, str) else v
            for k, v in payload_template.items()
        }

        self._ba_publisher.publish_raw(mqtt_topic, payload)
        logger.info(
            "BA request published (batch ready)",
            person_id=person_id,
            region_id=region_id,
            topic=mqtt_topic,
            payload=payload,
        )

    async def on_ba_result(self, result: dict) -> None:
        """Handle BA analysis result — update session state, feed rule engine."""
        person_id = result.get("person_id", "")
        region_id = result.get("region_id", "")
        status = result.get("status", "")
        scene_id = result.get("scene_id", "")

        session = self.session_mgr.get_session(person_id, scene_id=scene_id)
        if not session:
            logger.debug("BA result for unknown session", person_id=person_id)
            return

        if status != "suspicious":
            logger.debug("BA result", person_id=person_id, status=status)
            # Reset batch so next cycle of frames triggers a new analysis
            if self._frame_capture:
                self._frame_capture.reset_batch(person_id, region_id)
            return

        # ---- Update session state (facts only, no decisions) ----
        session.concealment_suspected = True
        session.concealment_count[region_id] = session.concealment_count.get(region_id, 0) + 1
        session.last_ba_confidence = float(result.get("confidence", 0.0))
        session.last_ba_frames_analyzed = int(result.get("frames_analyzed", 0))

        logger.warning(
            "BA: concealment detected — feeding rule engine",
            person_id=person_id,
            region_id=region_id,
            confidence=session.last_ba_confidence,
            frames_analyzed=session.last_ba_frames_analyzed,
            concealment_number=session.concealment_count[region_id],
        )

        # ---- Feed back into rule engine as CONCEALMENT_DETECTED event ----
        zone_type_str = self.config.get_zone_type(region_id) or "HIGH_VALUE"
        zone_name = self.config.get_zone_name(region_id) or ""

        event = RegionEvent(
            event_type=EventType.CONCEALMENT_DETECTED,
            object_id=person_id,
            region_id=region_id,
            region_name=zone_name,
            zone_type=ZoneType(zone_type_str),
            timestamp=session.last_seen,
            scene_id=scene_id or session.scene_id,
        )
        await self.on_event(event)

        # Delete analyzed frames so BA only sees fresh ones going forward
        self._delete_ba_frames(person_id, region_id, session.scene_id)

        # Reset batch so next cycle of frames triggers a new analysis
        if self._frame_capture:
            self._frame_capture.reset_batch(person_id, region_id)

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
