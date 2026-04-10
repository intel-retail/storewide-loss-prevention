# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Rule Engine — evaluates ENTERED / EXITED / PERSON_LOST events and fires alerts.

Built-in detection rules (part of LP service business logic):
  1. Restricted Zone Violation  (on ENTERED RESTRICTED)
  2. Repeated Visits            (on ENTERED HIGH_VALUE, count > threshold)
  3. Checkout state tracking    (on ENTERED CHECKOUT)
  4. Checkout Bypass            (on ENTERED EXIT without CHECKOUT)
  5. Loitering                  (on EXITED HIGH_VALUE, dwell > threshold)
  6. PERSON_LOST cleanup        (check open high-value visits for loitering)

Conditionally calls external BehavioralAnalysis Service
when specific patterns are detected (e.g., person in HIGH_VALUE zone).
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional

import structlog

from models.events import EventType, RegionEvent, ZoneType
from models.alerts import Alert, AlertType, AlertLevel
from .config import ConfigService
from .session_manager import SessionManager

logger = structlog.get_logger(__name__)


class RuleEngine:
    """Stateless rule evaluator — receives events, produces alerts."""

    def __init__(
        self,
        config: ConfigService,
        session_manager: SessionManager,
        alert_callback=None,
        behavioral_analysis_client=None,
        frame_manager=None,
    ) -> None:
        self.config = config
        self.session_mgr = session_manager
        self._alert_callback = alert_callback
        self._ba_client = behavioral_analysis_client
        self._frame_mgr = frame_manager

        rules = config.get_rules_config()
        self.loiter_threshold = rules.get("loiter_threshold_seconds", 120)
        self.repeat_threshold = rules.get("repeat_visit_threshold", 3)
        self.loiter_poll_interval = rules.get("loiter_poll_interval_seconds", 60)

        logger.info(
            "RuleEngine initialized",
            loiter_threshold=self.loiter_threshold,
            repeat_threshold=self.repeat_threshold,
            loiter_poll_interval=self.loiter_poll_interval,
        )

    def set_alert_callback(self, callback) -> None:
        self._alert_callback = callback

    # ---- main dispatcher -----------------------------------------------------
    async def on_event(self, event: RegionEvent) -> None:
        """Route an event to the appropriate handler based on type + zone."""
        if event.event_type == EventType.ENTERED:
            await self._on_entered(event)
        elif event.event_type == EventType.EXITED:
            await self._on_exited(event)
        elif event.event_type == EventType.PERSON_LOST:
            await self._on_person_lost(event)

    # ---- ENTERED handlers ----------------------------------------------------
    async def _on_entered(self, event: RegionEvent) -> None:
        session = self.session_mgr.get_session(event.object_id)
        if not session:
            return

        if event.zone_type == ZoneType.RESTRICTED:
            await self._handle_restricted_entry(event)

        elif event.zone_type == ZoneType.HIGH_VALUE:
            await self._handle_high_value_entry(event, session)

        elif event.zone_type == ZoneType.CHECKOUT:
            session.visited_checkout = True
            logger.info("Checkout visited", object_id=event.object_id)

        elif event.zone_type == ZoneType.EXIT:
            session.visited_exit = True
            await self._handle_exit_entry(event, session)

    async def _handle_restricted_entry(self, event: RegionEvent) -> None:
        """Immediate zone violation alert."""
        alert = Alert(
            alert_type=AlertType.ZONE_VIOLATION,
            alert_level=AlertLevel.CRITICAL,
            object_id=event.object_id,
            timestamp=event.timestamp,
            region_id=event.region_id,
            region_name=event.region_name,
            details={"zone_type": "RESTRICTED"},
        )
        logger.warning("RESTRICTED zone violation", object_id=event.object_id,
                       region=event.region_name)
        await self._fire_alert(alert)

    async def _handle_high_value_entry(self, event: RegionEvent, session) -> None:
        """Track monitoring state, check repeated visits, trigger behavioral analysis."""
        session.visited_high_value = True
        visit_count = session.zone_visit_counts.get(event.region_id, 0)

        logger.info(
            "HIGH_VALUE zone entered",
            object_id=event.object_id,
            region=event.region_name,
            visit_count=visit_count,
        )

        if visit_count > self.repeat_threshold:
            alert = Alert(
                alert_type=AlertType.UNUSUAL_PATH,
                alert_level=AlertLevel.WARNING,
                object_id=event.object_id,
                timestamp=event.timestamp,
                region_id=event.region_id,
                region_name=event.region_name,
                details={
                    "visit_count": visit_count,
                    "threshold": self.repeat_threshold,
                },
            )
            logger.warning(
                "Repeated high-value visits",
                object_id=event.object_id,
                count=visit_count,
            )
            await self._fire_alert(alert)

        # Trigger pose analysis via external BehavioralAnalysis Service
        await self._trigger_behavioral_analysis(event.object_id, event.region_id)

    async def _handle_exit_entry(self, event: RegionEvent, session) -> None:
        """Evaluate checkout bypass when the person reaches an exit."""
        if session.visited_high_value and not session.visited_checkout:
            level = (
                AlertLevel.CRITICAL
                if session.concealment_suspected
                else AlertLevel.WARNING
            )
            alert = Alert(
                alert_type=AlertType.CHECKOUT_BYPASS,
                alert_level=level,
                object_id=event.object_id,
                timestamp=event.timestamp,
                region_id=event.region_id,
                region_name=event.region_name,
                details={
                    "visited_high_value": True,
                    "visited_checkout": False,
                    "concealment_suspected": session.concealment_suspected,
                },
            )
            logger.warning(
                "Checkout bypass detected",
                object_id=event.object_id,
                level=level.value,
            )
            await self._fire_alert(alert)

    # ---- EXITED handlers -----------------------------------------------------
    async def _on_exited(self, event: RegionEvent) -> None:
        if event.zone_type == ZoneType.HIGH_VALUE:
            await self._check_loitering(event)

    async def _check_loitering(self, event: RegionEvent) -> None:
        """Fire loitering alert if dwell time exceeds threshold (once per zone)."""
        session = self.session_mgr.get_session(event.object_id)

        # Check if loiter alert already triggered for this zone
        if session and session.loiter_alerted.get(event.region_id):
            logger.debug(
                "Loiter alert already fired for zone",
                object_id=event.object_id,
                region_id=event.region_id,
            )
            return

        if event.dwell_seconds and event.dwell_seconds > self.loiter_threshold:
            alert = Alert(
                alert_type=AlertType.LOITERING,
                alert_level=AlertLevel.WARNING,
                object_id=event.object_id,
                timestamp=event.timestamp,
                region_id=event.region_id,
                region_name=event.region_name,
                details={
                    "dwell_seconds": round(event.dwell_seconds, 1),
                    "threshold": self.loiter_threshold,
                },
            )
            logger.warning(
                "Loitering detected",
                object_id=event.object_id,
                dwell=event.dwell_seconds,
            )
            await self._fire_alert(alert)

            # Mark loiter alert as triggered for this zone
            if session:
                session.loiter_alerted[event.region_id] = True

    # ---- active loiter polling -----------------------------------------------
    async def run_loiter_check_loop(self) -> None:
        """
        Background task: poll every loiter_poll_interval seconds for persons
        still inside HIGH_VALUE zones whose dwell exceeds the threshold.

        Catches the case where a person enters but never exits —
        SceneScape's EXITED event never fires, so the event-driven
        _check_loitering() path is never reached.
        """
        while True:
            await asyncio.sleep(self.loiter_poll_interval)
            try:
                now = datetime.now(timezone.utc)
                for session in self.session_mgr.get_all_sessions().values():
                    for zone_id, entry_ts_iso in session.current_zones.items():
                        zone_type = self.config.get_zone_type(zone_id)
                        if zone_type != "HIGH_VALUE":
                            continue

                        # Already alerted for this zone
                        if session.loiter_alerted.get(zone_id):
                            continue

                        try:
                            entry_ts = datetime.fromisoformat(entry_ts_iso)
                        except (ValueError, TypeError):
                            continue

                        dwell = (now - entry_ts).total_seconds()
                        if dwell > self.loiter_threshold:
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
                                    "threshold": self.loiter_threshold,
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
        # Session is already removed by SessionManager before this fires,
        # so we clean up the rolling buffer frames directly by object_id.
        if self._frame_mgr:
            self._frame_mgr.cleanup_person(event.object_id)
        logger.info("Person lost — frames cleaned up", object_id=event.object_id)

    # ---- External service calls (conditional) --------------------------------
    async def _trigger_behavioral_analysis(
        self, object_id: str, region_id: str
    ) -> None:
        """
        Send frames to the BehavioralAnalysis Service for persons in HIGH_VALUE zones.

        The behavioral service owns all analysis logic (pose detection, VLM
        escalation, etc.) — we just send frames and handle the result.
        """
        if not self._ba_client or not self._frame_mgr:
            return

        session = self.session_mgr.get_session(object_id)
        if not session:
            return

        frame_keys = self._frame_mgr.get_person_frame_keys(object_id)
        if not frame_keys:
            logger.debug("No frames available for behavioral analysis", object_id=object_id)
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
        # Persist evidence frames for audit (copy from rolling buffer to alerts/)
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
