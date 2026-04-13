# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Session Manager — owns the live state of every person currently in the store.

Consumes two SceneScape MQTT feeds:
  1. scene-data  (scenescape/data/scene/+/+)   — position updates, camera visibility
  2. region-events (scenescape/event/region/+/+/+) — native ENTERED / EXITED with dwell

The scene-data feed keeps sessions alive (last_seen) and tracks cameras/bbox.
The region-event feed drives ENTERED / EXITED / PERSON_LOST events using
SceneScape's own boundary detection and dwell calculation — no local diffing needed.
"""

import asyncio
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Set

import structlog

from models.session import PersonSession, RegionVisit
from models.events import EventType, RegionEvent, ZoneType
from .config import ConfigService

logger = structlog.get_logger(__name__)


class SessionManager:
    """
    Maintains a PersonSession for every active object_id.

    Scene-data messages keep the session alive (last_seen, cameras, bbox).
    Region-event messages drive ENTERED / EXITED events using SceneScape's
    native boundary detection and dwell calculation.
    Sessions are expired when absent for longer than session_timeout.
    """

    def __init__(self, config: ConfigService) -> None:
        self.config = config
        rules = config.get_rules_config()
        self.session_timeout = rules.get("session_timeout_seconds", 30)

        # Build set of configured camera names for filtering
        self._allowed_cameras = {c["name"] for c in config.get_cameras()} if config.get_cameras() else set()

        self._sessions: Dict[str, PersonSession] = {}
        self._event_handlers: List[Callable] = []
        self._expiry_task: Optional[asyncio.Task] = None

        logger.info("SessionManager initialized", timeout=self.session_timeout,
                    allowed_cameras=sorted(self._allowed_cameras) or "all")

    # ---- event handler registration -----------------------------------------
    def register_event_handler(self, handler: Callable) -> None:
        """Register an async handler that receives RegionEvent objects."""
        self._event_handlers.append(handler)

    # ---- public accessors ---------------------------------------------------
    def get_session(self, object_id: str) -> Optional[PersonSession]:
        return self._sessions.get(object_id)

    def get_all_sessions(self) -> Dict[str, PersonSession]:
        return dict(self._sessions)

    def get_active_count(self) -> int:
        return len(self._sessions)

    # ---- scene-data handler: keeps sessions alive ----------------------------
    async def on_scene_data(
        self, scene_id: str, object_type: str, data: dict
    ) -> None:
        """
        Process a scenescape/data/scene/{scene_id}/{object_type} message.

        Updates session liveness (last_seen), cameras, bbox.
        Does NOT fire ENTERED/EXITED events — those come from on_region_event()
        via SceneScape's native region events.
        """
        # Filter by resolved scene_id (read lazily — resolved after init)
        scene_id_filter = self.config.get_scene_id()
        if scene_id_filter and scene_id != scene_id_filter:
            return

        if object_type not in ("person", "persons"):
            return

        now = datetime.now(timezone.utc)

        objects = data.get("objects", data) if isinstance(data, dict) else data
        if not isinstance(objects, list):
            objects = [objects]

        for obj in objects:
            oid = str(obj.get("id", obj.get("object_id", "")))
            if not oid:
                continue

            cameras = obj.get("visibility", obj.get("camera_ids", obj.get("cameras", [])))
            bbox = obj.get("center_of_mass", obj.get("bounding_box", obj.get("bbox")))

            # Filter: only track persons visible on configured cameras
            if self._allowed_cameras:
                visible_on_configured = [c for c in cameras if c in self._allowed_cameras]
                if not visible_on_configured:
                    continue

            if oid in self._sessions:
                session = self._sessions[oid]
                session.last_seen = now
                session.current_cameras = list(cameras)
                session.bbox = bbox
                # Update camera history
                for cam in cameras:
                    if cam not in session.camera_history:
                        session.camera_history.append(cam)
            else:
                session = PersonSession(
                    object_id=oid,
                    first_seen=now,
                    last_seen=now,
                    current_cameras=list(cameras),
                    bbox=bbox,
                )
                self._sessions[oid] = session
                logger.info("Session created", object_id=oid)

    # ---- region-event handler: drives ENTERED / EXITED ----------------------
    async def on_region_event(
        self, scene_id: str, region_id: str, data: dict
    ) -> None:
        """
        Process a scenescape/event/region/{scene_id}/{region_id}/{suffix} message.

        SceneScape provides native enter/exit lists with dwell time,
        so we consume them directly instead of diffing region sets.
        """
        scene_id_filter = self.config.get_scene_id()
        if scene_id_filter and scene_id != scene_id_filter:
            return

        now = datetime.now(timezone.utc)

        # Process persons that entered this region
        for obj in data.get("entered", []):
            oid = str(obj.get("id", obj.get("object_id", "")))
            if not oid:
                continue
            # Ensure session exists (region event may arrive before scene-data)
            if oid not in self._sessions:
                first_seen_str = obj.get("first_seen")
                first_seen = now
                if first_seen_str:
                    try:
                        first_seen = datetime.fromisoformat(first_seen_str.replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        first_seen = now
                cameras = obj.get("visibility", [])
                session = PersonSession(
                    object_id=oid,
                    first_seen=first_seen,
                    last_seen=now,
                    current_cameras=list(cameras),
                    bbox=obj.get("center_of_mass"),
                )
                self._sessions[oid] = session
                logger.info("Session created from region event", object_id=oid, region_id=region_id)
            else:
                session = self._sessions[oid]
                session.last_seen = now

            await self._fire_enter(session, region_id, now)

        # Process persons that exited this region
        for exit_entry in data.get("exited", []):
            obj = exit_entry.get("object", exit_entry)
            dwell = exit_entry.get("dwell", 0.0)
            oid = str(obj.get("id", obj.get("object_id", "")))
            if not oid:
                continue

            session = self._sessions.get(oid)
            if not session:
                continue
            session.last_seen = now

            await self._fire_exit(session, region_id, now, dwell_override=dwell)

    # ---- session expiry ------------------------------------------------------
    async def _expire_session(self, oid: str) -> None:
        session = self._sessions.get(oid)
        if session is None:
            return

        now = datetime.now(timezone.utc)
        logger.info("Session expired", object_id=oid)

        # Close all open region visits and fire EXITED events.
        # Session stays in _sessions so downstream handlers (e.g. RuleEngine)
        # can still look up loiter_alerted and other state.
        for visit in session.get_open_visits():
            visit.exit_time = now
            zone_type = self.config.get_zone_type(visit.region_id)
            if zone_type:
                event = RegionEvent(
                    event_type=EventType.EXITED,
                    object_id=oid,
                    region_id=visit.region_id,
                    region_name=visit.region_name,
                    zone_type=ZoneType(zone_type),
                    timestamp=now,
                    dwell_seconds=visit.duration_seconds,
                )
                await self._emit(event)

        # Remove session after EXITED events are processed
        del self._sessions[oid]

        # Fire PERSON_LOST
        lost_event = RegionEvent(
            event_type=EventType.PERSON_LOST,
            object_id=oid,
            region_id="",
            region_name="",
            zone_type=ZoneType.HIGH_VALUE,
            timestamp=now,
        )
        await self._emit(lost_event)

    # ---- event helpers -------------------------------------------------------
    async def _fire_enter(
        self, session: PersonSession, region_id: str, now: datetime
    ) -> None:
        zone_type = self.config.get_zone_type(region_id)
        zone_name = self.config.get_zone_name(region_id) or region_id
        if not zone_type:
            logger.warning(
                "Region not mapped to any zone — event dropped",
                region_id=region_id,
                object_id=session.object_id,
                configured_zones=list(self.config.get_zones().keys()),
            )
            return

        # Record the visit
        visit = RegionVisit(
            region_id=region_id,
            region_name=zone_name,
            zone_type=zone_type,
            entry_time=now,
        )
        session.region_visits.append(visit)

        # Update current_zones and zone_visit_counts
        session.enter_zone(region_id, now)

        event = RegionEvent(
            event_type=EventType.ENTERED,
            object_id=session.object_id,
            region_id=region_id,
            region_name=zone_name,
            zone_type=ZoneType(zone_type),
            timestamp=now,
        )
        await self._emit(event)

    async def _fire_exit(
        self, session: PersonSession, region_id: str, now: datetime,
        dwell_override: Optional[float] = None,
    ) -> None:
        zone_type = self.config.get_zone_type(region_id)
        zone_name = self.config.get_zone_name(region_id) or region_id
        if not zone_type:
            logger.warning(
                "Region not mapped to any zone — exit event dropped",
                region_id=region_id,
                object_id=session.object_id,
            )
            return

        visit = session.close_visit(region_id, now)
        # Use SceneScape's dwell time if provided, otherwise fall back to local calc
        dwell = dwell_override if dwell_override is not None else (visit.duration_seconds if visit else 0.0)

        # Update current_zones
        session.exit_zone(region_id)

        event = RegionEvent(
            event_type=EventType.EXITED,
            object_id=session.object_id,
            region_id=region_id,
            region_name=zone_name,
            zone_type=ZoneType(zone_type),
            timestamp=now,
            dwell_seconds=dwell,
        )
        await self._emit(event)

    async def _emit(self, event: RegionEvent) -> None:
        for handler in self._event_handlers:
            try:
                await handler(event)
            except Exception:
                logger.exception("Event handler error", event=event)

    # ---- expiry loop ---------------------------------------------------------
    async def run_expiry_loop(self) -> None:
        """Periodically check for expired sessions."""
        while True:
            await asyncio.sleep(5)
            now = datetime.now(timezone.utc)
            expired = [
                oid
                for oid, s in self._sessions.items()
                if (now - s.last_seen).total_seconds() > self.session_timeout
            ]
            for oid in expired:
                await self._expire_session(oid)
