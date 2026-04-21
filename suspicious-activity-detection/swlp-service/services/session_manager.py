# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Session Manager — owns the live state of every person currently in the store.

Consumes three SceneScape MQTT feeds:
  1. scene-data    (scenescape/data/scene/+/+)    — position updates, camera visibility
  2. region-events (scenescape/event/region/+/+/+) — native ENTERED / EXITED with dwell
  3. region-data   (scenescape/data/region/+/+)    — continuous per-frame object presence
     for real-time loiter detection (dwell computed from SceneScape's entry timestamps)
"""

import asyncio
import time
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
        self._loiter_threshold = float(rules.get("loiter_threshold_seconds", 20))

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
        # Filter by resolved scene_ids (supports multiple scenes)
        accepted_scene_ids = self.config.get_accepted_scene_ids()
        if accepted_scene_ids and scene_id not in accepted_scene_ids:
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
            bbox = obj.get("bounding_box", obj.get("bbox"))

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
                    scene_id=scene_id,
                    current_cameras=list(cameras),
                    bbox=bbox,
                )
                self._sessions[oid] = session
                logger.info("Session created", object_id=oid, scene_id=scene_id)

            # Real-time loiter check using region entry timestamps from scene data
            regions = obj.get("regions", {})
            if regions:
                await self._check_loiter_from_scene_data(session, regions, now)
            else:
                logger.debug("No regions in scene data object",
                             object_id=oid, keys=list(obj.keys())[:10])

    async def _check_loiter_from_scene_data(
        self, session: PersonSession, regions: dict, now: datetime
    ) -> None:
        """Check dwell time for each region in scene data and emit LOITER if threshold exceeded."""
        now_epoch = time.time()
        for region_id, rinfo in regions.items():
            # Already alerted for this zone — skip
            if session.loiter_alerted.get(region_id):
                continue

            zone_type = self.config.get_zone_type(region_id)
            if zone_type != "HIGH_VALUE":
                continue

            entered_str = rinfo.get("entered") if isinstance(rinfo, dict) else None
            if not entered_str:
                continue
            try:
                entered_dt = datetime.fromisoformat(entered_str.replace("Z", "+00:00"))
                entered_epoch = entered_dt.timestamp()
            except (ValueError, TypeError):
                continue

            dwell = now_epoch - entered_epoch
            logger.debug("Loiter check", object_id=session.object_id,
                         region_id=region_id, dwell=round(dwell, 1),
                         threshold=self._loiter_threshold)
            if dwell > self._loiter_threshold:
                zone_name = self.config.get_zone_name(region_id) or region_id
                logger.info("Loiter threshold exceeded (scene data)",
                            object_id=session.object_id,
                            region=zone_name, dwell=round(dwell, 1))
                event = RegionEvent(
                    event_type=EventType.LOITER,
                    object_id=session.object_id,
                    region_id=region_id,
                    region_name=zone_name,
                    zone_type=ZoneType(zone_type),
                    timestamp=now,
                    scene_id=session.scene_id,
                    dwell_seconds=round(dwell, 1),
                )
                await self._emit(event)

    # ---- region-event handler: drives ENTERED / EXITED ----------------------
    async def on_region_event(
        self, scene_id: str, region_id: str, data: dict
    ) -> None:
        """
        Process a scenescape/event/region/{scene_id}/{region_id}/{suffix} message.

        SceneScape provides native enter/exit lists with dwell time,
        so we consume them directly instead of diffing region sets.
        """
        scene_id_filter = self.config.get_accepted_scene_ids()
        if scene_id_filter and scene_id not in scene_id_filter:
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
                    scene_id=scene_id,
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

    # ---- region-data handler: continuous dwell checking ----------------------
    async def on_region_data(
        self, scene_id: str, region_id: str, data: dict
    ) -> None:
        """
        Process a scenescape/data/region/{scene_id}/{region_id} message.

        This is the continuous feed — every frame, SceneScape publishes all
        objects currently inside a region. Each object carries
        regions.{name}.entered (epoch timestamp) from which we compute live dwell.

        Used for loiter detection without any local timestamp tracking or polling.
        """
        logger.info("on_region_data called", scene_id=scene_id, region_id=region_id,
                    num_objects=len(data.get("objects", [])))

        scene_id_filter = self.config.get_accepted_scene_ids()
        if scene_id_filter and scene_id not in scene_id_filter:
            logger.info("region_data: scene filtered", scene_id=scene_id)
            return

        zone_type = self.config.get_zone_type(region_id)
        if zone_type != "HIGH_VALUE":
            logger.info("region_data: zone not HIGH_VALUE", region_id=region_id, zone_type=zone_type)
            return

        now_epoch = time.time()

        for obj in data.get("objects", []):
            oid = str(obj.get("id", ""))
            if not oid:
                continue

            session = self._sessions.get(oid)
            if not session:
                logger.info("region_data: no session for object", object_id=oid)
                continue

            # Already alerted for this zone — skip
            if session.loiter_alerted.get(region_id):
                continue

            # Read SceneScape's entry timestamp from the object's region data
            regions = obj.get("regions", {})
            if not regions:
                logger.info("region_data: no regions in object", object_id=oid,
                            keys=list(obj.keys())[:10])
                continue
            for rname, rinfo in regions.items():
                entered_str = rinfo.get("entered")
                if not entered_str:
                    continue
                try:
                    entered_dt = datetime.fromisoformat(entered_str.replace("Z", "+00:00"))
                    entered_epoch = entered_dt.timestamp()
                except (ValueError, TypeError):
                    continue

                dwell = now_epoch - entered_epoch
                logger.info("region_data dwell", object_id=oid, region=rname,
                            dwell=round(dwell, 1), threshold=self._loiter_threshold)
                if dwell > self._loiter_threshold:
                    zone_name = self.config.get_zone_name(region_id) or region_id
                    event = RegionEvent(
                        event_type=EventType.LOITER,
                        object_id=oid,
                        region_id=region_id,
                        region_name=zone_name,
                        zone_type=ZoneType(zone_type),
                        timestamp=datetime.now(timezone.utc),
                        scene_id=session.scene_id,
                        dwell_seconds=round(dwell, 1),
                    )
                    await self._emit(event)
                    break  # one alert per object per region_data message

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
                    scene_id=session.scene_id,
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
            scene_id=session.scene_id,
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

        # Guard: skip duplicate ENTERED if person is already in this zone
        # (SceneScape may publish on multiple topic suffixes or boundary jitter)
        if session.is_in_zone(region_id):
            logger.debug(
                "Duplicate zone_entry suppressed — person already in zone",
                object_id=session.object_id,
                region_id=region_id,
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
            scene_id=session.scene_id,
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
            scene_id=session.scene_id,
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
