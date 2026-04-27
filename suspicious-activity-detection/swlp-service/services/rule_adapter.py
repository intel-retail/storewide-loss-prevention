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
from typing import Optional, Protocol, runtime_checkable

import structlog

from models.events import EventType, RegionEvent, ZoneType
from models.alerts import Alert, AlertType, AlertLevel
from .config import ConfigService
from .session_manager import SessionManager
from rule_engine import RuleEngine, Action
from .alert_service_client import AlertServiceClient

logger = structlog.get_logger(__name__)


@runtime_checkable
class EscalationService(Protocol):
    """Protocol for services that can be invoked by the 'escalate' action type."""
    def start(self, object_id: str, region_id: str, scene_id: str) -> None: ...
    def stop(self, object_id: str, region_id: str) -> None: ...
    def stop_all(self, object_id: str) -> None: ...


class RuleEngineAdapter:
    """LP-specific adapter that wires the Rule Engine Service to this service."""

    def __init__(
        self,
        engine: RuleEngine,
        config: ConfigService,
        session_manager: SessionManager,
        alert_service_client: AlertServiceClient | None = None,
        frame_manager=None,
    ) -> None:
        self._engine = engine
        self.config = config
        self.session_mgr = session_manager
        self._alert_client = alert_service_client
        self._frame_mgr = frame_manager

        rules_cfg = config.get_rules_config()
        self._loiter_threshold = float(rules_cfg.get("loiter_threshold_seconds", 20))
        self._pending_cleanups: set[str] = set()  # "{object_id}:{region_id}"

        # Config-driven session flags: {flag_name: {trigger, zone_type, ...}}
        self._session_flag_defs = config.get_session_flag_defs()

        # Precompute zone_type → list of flag names for fast lookup on zone entry
        self._zone_visited_flags: dict[str, list[str]] = {}
        for flag_name, flag_def in self._session_flag_defs.items():
            if flag_def.get("trigger") == "zone_visited":
                zt = flag_def.get("zone_type", "")
                self._zone_visited_flags.setdefault(zt, []).append(flag_name)

        # External flag definitions: {source_name: [{flag_name, field, match_value}]}
        self._external_flags: dict[str, list[dict]] = {}
        for flag_name, flag_def in self._session_flag_defs.items():
            if flag_def.get("trigger") == "external":
                source = flag_def.get("source", "")
                self._external_flags.setdefault(source, []).append({
                    "flag_name": flag_name,
                    "field": flag_def.get("field", "status"),
                    "match_value": flag_def.get("match_value"),
                })

        # Service registry: {service_name: EscalationService}
        self._service_registry: dict[str, EscalationService] = {}

        logger.info(
            "RuleEngineAdapter initialized",
            loiter_threshold=self._loiter_threshold,
            rules_loaded=len(engine.rules),
            session_flags=list(self._session_flag_defs.keys()),
            zone_visited_flags=self._zone_visited_flags,
        )

    def set_alert_client(self, client: AlertServiceClient) -> None:
        self._alert_client = client

    def register_service(self, name: str, handler: EscalationService) -> None:
        """Register a named escalation service (e.g. 'behavioral_analysis')."""
        self._service_registry[name] = handler
        logger.info("Escalation service registered", service=name)

    # ---- main entry point (same signature as old RuleEngine.on_event) ---------

    async def on_event(self, event: RegionEvent) -> None:
        """Process a region event: update state, evaluate rules, execute actions."""
        if event.event_type == EventType.PERSON_LOST:
            await self._on_person_lost(event)
            return

        session = self.session_mgr.get_session(event.object_id, event.scene_id)
        if not session:
            return

        # ---- Config-driven state transitions (replaces hardcoded if/elif) ----
        if event.event_type == EventType.ENTERED:
            zone_type_str = event.zone_type.value if event.zone_type else ""
            flag_names = self._zone_visited_flags.get(zone_type_str, [])
            for flag_name in flag_names:
                if not session.flags.get(flag_name):
                    session.flags[flag_name] = True
                    logger.info(
                        "Session flag set",
                        flag=flag_name,
                        object_id=event.object_id,
                        region=event.region_name,
                        zone_type=zone_type_str,
                    )

        elif event.event_type == EventType.EXITED:
            # Stop any escalation services that were started for this zone
            for svc in self._service_registry.values():
                svc.stop(event.object_id, event.region_id)
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
        ctx = {
            # Event fields
            "region_id": event.region_id,
            "region_name": event.region_name,
            "dwell_seconds": event.dwell_seconds,
            # Session structural fields
            "zone_visit_counts": dict(session.zone_visit_counts),
        }
        # Merge all dynamic session flags into context
        ctx.update(session.flags)
        return ctx

    # ---- action execution (LP-specific) --------------------------------------

    async def _execute_actions(
        self, actions: list[Action], event: RegionEvent, session
    ) -> None:
        for action in actions:
            if action.type == "alert":
                await self._execute_alert(action, event, session)
            elif action.type == "escalate":
                service_name = action.params.get("service", "")
                handler = self._service_registry.get(service_name)
                if handler and self._engine.is_rule_enabled(action.rule_id):
                    handler.start(
                        event.object_id, event.region_id, event.scene_id
                    )
                elif not handler:
                    logger.warning(
                        "Escalate action references unknown service",
                        service=service_name,
                        rule_id=action.rule_id,
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
                "visited_high_value": session.flags.get("visited_high_value", False),
                "visited_checkout": session.flags.get("visited_checkout", False),
                "concealment_suspected": session.flags.get("concealment_suspected", False),
            }
        elif alert_type == AlertType.LOITERING:
            return {
                "dwell_seconds": round(event.dwell_seconds, 1) if event.dwell_seconds else 0,
                "threshold": params.get("threshold", 120),
            }
        elif alert_type == AlertType.CONCEALMENT:
            ba = getattr(session, "_pending_ba_result", None) or {}
            return {
                "confidence": ba.get("confidence"),
                "message": ba.get("vlm_response") or "",
                "frames_analyzed": ba.get("frames_analyzed", 0),
            }
        return {}

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
        """Cancel any active escalation service tasks for this person."""
        for svc in self._service_registry.values():
            svc.stop_all(event.object_id)
        logger.info("Person lost — escalation tasks cancelled", object_id=event.object_id)

    async def on_ba_result(self, result: dict) -> None:
        """
        Handle a BA analysis result received from the MQTT ba/results topic.
        Routes the result through the rule engine (rule: concealment_detected).
        """
        person_id = result.get("person_id", "")
        region_id = result.get("region_id", "")
        status = result.get("status", "")
        scene_id = result.get("scene_id", "")

        session = self.session_mgr.get_session(person_id, scene_id=scene_id)
        if not session:
            logger.debug("BA result for unknown session", person_id=person_id)
            return

        # Quick non-actionable statuses — no rule firing needed.
        if status in ("received", "no_match"):
            logger.debug(
                "BA queue: status update",
                person_id=person_id,
                region_id=region_id,
                status=status,
            )
            return

        # Per-zone dedup before invoking the engine.
        if status == "suspicious" and session.ba_alerted.get(region_id):
            logger.debug(
                "BA result: concealment alert already fired for zone",
                person_id=person_id,
                region_id=region_id,
            )
            return

        # Resolve zone metadata for the synthetic event.
        zone_name = self.config.get_zone_name(region_id)
        zone_type_str = self.config.get_zone_type(region_id) or "HIGH_VALUE"
        try:
            zone_type = ZoneType[zone_type_str]
        except KeyError:
            zone_type = ZoneType.HIGH_VALUE

        synth_event = RegionEvent(
            event_type=EventType.ENTERED,  # placeholder; engine matches on "ba_result" string
            object_id=person_id,
            region_id=region_id,
            region_name=zone_name,
            zone_type=zone_type,
            timestamp=session.last_seen,
            dwell_seconds=0.0,
            scene_id=session.scene_id,
        )

        # Build context the rule conditions can read.
        context = self._build_context(synth_event, session)
        context["ba_status"] = status
        context["ba_confidence"] = result.get("confidence", 0.0)
        context["ba_frames_analyzed"] = result.get("frames_analyzed", 0)

        # Stash raw BA result so _build_details(CONCEALMENT) can pick it up.
        session._pending_ba_result = result

        # Mark state BEFORE firing so dedup is consistent.
        # Apply external flag definitions from config
        for flag_def in self._external_flags.get("behavioral_analysis", []):
            field_name = flag_def["field"]
            match_val = flag_def["match_value"]
            if result.get(field_name) == match_val:
                session.flags[flag_def["flag_name"]] = True
        if status == "suspicious":
            session.ba_alerted[region_id] = True

        try:
            actions = self._engine.evaluate("ba_result", zone_type.value, context)
            await self._execute_actions(actions, synth_event, session)
        finally:
            session._pending_ba_result = None

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
