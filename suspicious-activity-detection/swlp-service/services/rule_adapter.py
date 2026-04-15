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
from .rule_engine import RuleEngine, Action

logger = structlog.get_logger(__name__)


class RuleEngineAdapter:
    """LP-specific adapter that wires the Rule Engine Service to this service."""

    def __init__(
        self,
        engine: RuleEngine,
        config: ConfigService,
        session_manager: SessionManager,
        alert_callback=None,
        behavioral_analysis_client=None,
        frame_manager=None,
    ) -> None:
        self._engine = engine
        self.config = config
        self.session_mgr = session_manager
        self._alert_callback = alert_callback
        self._ba_client = behavioral_analysis_client
        self._frame_mgr = frame_manager

        rules_cfg = config.get_rules_config()
        self.loiter_poll_interval = rules_cfg.get("loiter_poll_interval_seconds", 60)

        logger.info(
            "RuleEngineAdapter initialized",
            loiter_poll_interval=self.loiter_poll_interval,
            rules_loaded=len(engine.rules),
        )

    def set_alert_callback(self, callback) -> None:
        self._alert_callback = callback

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

    # ---- active loiter polling -----------------------------------------------

    async def run_loiter_check_loop(self) -> None:
        """
        Background task: poll for persons still in HIGH_VALUE zones
        whose dwell exceeds the loitering threshold.
        """
        if not self._engine.is_rule_enabled("loitering"):
            logger.info("Loitering rule disabled — skipping poll loop")
            return

        loiter_rule = self._engine.get_rule("loitering")
        threshold = 120.0
        if loiter_rule:
            for cond in loiter_rule.get("conditions", []):
                if "dwell" in cond.get("field", ""):
                    threshold = float(cond["value"])
                    break

        while True:
            await asyncio.sleep(self.loiter_poll_interval)
            try:
                now = datetime.now(timezone.utc)
                for session in self.session_mgr.get_all_sessions().values():
                    for zone_id, entry_ts_iso in session.current_zones.items():
                        zone_type = self.config.get_zone_type(zone_id)
                        if zone_type != "HIGH_VALUE":
                            continue
                        if session.loiter_alerted.get(zone_id):
                            continue

                        try:
                            entry_ts = datetime.fromisoformat(entry_ts_iso)
                        except (ValueError, TypeError):
                            continue

                        dwell = (now - entry_ts).total_seconds()
                        if dwell > threshold:
                            zone_name = self.config.get_zone_name(zone_id) or zone_id
                            alert = Alert(
                                alert_type=AlertType.LOITERING,
                                alert_level=AlertLevel.WARNING,
                                object_id=session.object_id,
                                timestamp=now,
                                region_id=zone_id,
                                region_name=zone_name,
                                details={
                                    "dwell_seconds": round(dwell, 1),
                                    "threshold": threshold,
                                    "source": "active_poll",
                                },
                            )
                            logger.warning(
                                "Loitering detected (active poll)",
                                object_id=session.object_id,
                                zone=zone_name,
                                dwell=round(dwell, 1),
                            )
                            session.loiter_alerted[zone_id] = True
                            await self._fire_alert(alert)
            except Exception:
                logger.exception("Error in loiter check loop")

    # ---- PERSON_LOST handler -------------------------------------------------

    async def _on_person_lost(self, event: RegionEvent) -> None:
        """Clean up frame storage when a person's session expires."""
        if self._frame_mgr:
            self._frame_mgr.cleanup_person(event.object_id)
        logger.info("Person lost — frames cleaned up", object_id=event.object_id)

    # ---- External service calls ----------------------------------------------

    async def _trigger_behavioral_analysis(
        self, object_id: str, region_id: str
    ) -> None:
        """Send frames to BehavioralAnalysis Service for HIGH_VALUE zone persons."""
        if not self._ba_client or not self._frame_mgr:
            return

        session = self.session_mgr.get_session(object_id)
        if not session:
            return

        frame_keys = self._frame_mgr.get_person_frame_keys(object_id)
        if not frame_keys:
            logger.debug("No frames for behavioral analysis", object_id=object_id)
            return

        frames_b64 = await self._frame_mgr.get_frames_base64(frame_keys)
        if not frames_b64:
            return

        zone_info = {
            "region_id": region_id,
            "zone_type": "HIGH_VALUE",
            "zone_name": self.config.get_zone_name(region_id),
        }

        result = await self._ba_client.analyze(
            object_id, frame_keys, frames_b64, zone_info
        )

        if result and result.get("concealment_suspected"):
            session.concealment_suspected = True
            alert = Alert(
                alert_type=AlertType.CONCEALMENT,
                alert_level=AlertLevel.WARNING,
                object_id=object_id,
                timestamp=session.last_seen,
                region_id=region_id,
                region_name=zone_info["zone_name"],
                details={
                    "confidence": result.get("confidence"),
                    "observation": result.get("observation", ""),
                },
                evidence_keys=frame_keys,
            )
            logger.warning(
                "Behavioral analysis flagged concealment",
                object_id=object_id,
                confidence=result.get("confidence"),
            )
            await self._fire_alert(alert)

    # ---- alert dispatch ------------------------------------------------------

    async def _fire_alert(self, alert: Alert) -> None:
        """Persist evidence frames and dispatch via callback."""
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

        if self._alert_callback:
            try:
                await self._alert_callback(alert)
            except Exception:
                logger.exception("Alert callback error", alert_id=alert.alert_id)
