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

        # Build set of configured camera names for filtering
        self._allowed_cameras = {c["name"] for c in config.get_cameras()} if config.get_cameras() else set()

        # Sessions keyed by (scene_id, canonical_object_id) to support
        # multi-scene. Canonical id is the earliest UUID in a re-id chain;
        # all later flickering UUIDs are aliased to it via _oid_alias below.
        self._sessions: Dict[tuple, PersonSession] = {}
        # Canonical-id alias: maps any flickering raw oid SceneScape emits
        # back to the first oid we saw for that physical person, so all
        # downstream state (sessions, frame folders, dedup) stays unified.
        self._oid_alias: Dict[tuple, str] = {}  # (scene_id, raw_oid) -> canonical_oid
        self._event_handlers: List[Callable] = []
        self._match_handlers: List[Callable] = []
        self._expiry_task: Optional[asyncio.Task] = None

        logger.info("SessionManager initialized", timeout=self.session_timeout,
                    allowed_cameras=sorted(self._allowed_cameras) or "all")

    # ---- event handler registration -----------------------------------------
    def register_event_handler(self, handler: Callable) -> None:
        """Register an async handler that receives RegionEvent objects."""
        self._event_handlers.append(handler)

    def register_match_handler(self, handler: Callable) -> None:
        """Register a handler invoked when a session's reid_state flips to 'matched'.

        Receives the PersonSession; may be sync or async.
        """
        self._match_handlers.append(handler)

    async def _notify_match(self, session: PersonSession) -> None:
        for h in self._match_handlers:
            try:
                result = h(session)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Match handler error", object_id=session.object_id)

    # ---- canonical-id resolution -------------------------------------------
    def _resolve_canonical(
        self, scene_id: str, oid: str, prev_chain: Optional[list]
    ) -> str:
        """Map a (possibly flickering) raw oid to a stable canonical oid.

        SceneScape often assigns a fresh UUID to the same physical person
        every couple of seconds; each new track lists older UUIDs in
        ``previous_ids_chain``. We collapse the lineage onto the first
        canonical id we have already recorded for any ancestor in the chain.
        Falls back to ``oid`` itself when no ancestor is known.
        """
        skey = (scene_id, oid)
        if skey in self._oid_alias:
            return self._oid_alias[skey]

        for prev in prev_chain or []:
            prev_str = str(prev) if prev is not None else ""
            if not prev_str:
                continue
            prev_key = (scene_id, prev_str)
            if prev_key in self._oid_alias:
                canonical = self._oid_alias[prev_key]
                self._oid_alias[skey] = canonical
                logger.info("track aliased to canonical",
                            oid=oid[:8], canonical=canonical[:8],
                            chain_len=len(prev_chain or []))
                return canonical

        # First time seeing this lineage — oid is its own root canonical.
        self._oid_alias[skey] = oid
        return oid

    # ---- public accessors ---------------------------------------------------
    def get_session(self, object_id: str, scene_id: str = "") -> Optional[PersonSession]:
        canonical = self._oid_alias.get((scene_id, object_id), object_id)
        return self._sessions.get((scene_id, canonical))

    def get_all_sessions(self) -> Dict[tuple, PersonSession]:
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
            raw_oid = str(obj.get("id", obj.get("object_id", "")))
            if not raw_oid:
                continue
            # Collapse re-id flicker: route every track variant onto the
            # earliest canonical UUID we have already recorded for it.
            prev_chain = obj.get("previous_ids_chain") or []
            oid = self._resolve_canonical(scene_id, raw_oid, prev_chain)

            # Note: we do NOT skip on reid_state. Loiter only needs dwell time,
            # not identity. Provisional tracks may flap (new UUID every ~2s),
            # but we coalesce dwell across re-entries below in on_region_data.
            cameras = obj.get("visibility", obj.get("camera_ids", obj.get("cameras", [])))
            bbox = obj.get("bounding_box", obj.get("bbox"))
            reid_state = str(obj.get("reid_state", "") or "")

            # Filter: only track persons visible on configured cameras
            if self._allowed_cameras:
                visible_on_configured = [c for c in cameras if c in self._allowed_cameras]
                if not visible_on_configured:
                    continue

            skey = (scene_id, oid)
            if skey in self._sessions:
                session = self._sessions[skey]
                session.last_seen = now
                session.current_cameras = list(cameras)
                session.bbox = bbox
                # Promote reid_state once SceneScape upgrades to "matched".
                if reid_state and (not session.reid_state or reid_state == "matched"):
                    prev_state = session.reid_state
                    session.reid_state = reid_state
                    if reid_state == "matched" and prev_state != "matched":
                        await self._notify_match(session)
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
                    reid_state=reid_state,
                )
                self._sessions[skey] = session
                logger.info("Session created", object_id=oid, scene_id=scene_id,
                            reid_state=reid_state)

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
        """Emit a LOITER event for each HIGH_VALUE region the person is in.

        No threshold is applied here — the rule engine evaluates
        rules.yaml's `loitering` rule (condition: dwell_seconds > threshold)
        and decides whether to alert.
        """
        for region_id, rinfo in regions.items():
            # Already alerted for this zone — skip
            if session.loiter_alerted.get(region_id):
                continue

            zone_type = self.config.get_zone_type(region_id)
            if zone_type != "HIGH_VALUE":
                continue

            if not isinstance(rinfo, dict):
                continue
            try:
                dwell = float(rinfo.get("dwell", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue

            zone_name = self.config.get_zone_name(region_id) or region_id
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
            raw_oid = str(obj.get("id", obj.get("object_id", "")))
            if not raw_oid:
                continue
            prev_chain = obj.get("previous_ids_chain") or []
            oid = self._resolve_canonical(scene_id, raw_oid, prev_chain)
            # Ensure session exists (region event may arrive before scene-data)
            skey = (scene_id, oid)
            if skey not in self._sessions:
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
                self._sessions[skey] = session
                logger.info("Session created from region event", object_id=oid, region_id=region_id)
            else:
                session = self._sessions[skey]
                session.last_seen = now

            await self._fire_enter(session, region_id, now)

        # Process persons that exited this region
        for exit_entry in data.get("exited", []):
            obj = exit_entry.get("object", exit_entry)
            dwell = exit_entry.get("dwell", 0.0)
            raw_oid = str(obj.get("id", obj.get("object_id", "")))
            if not raw_oid:
                continue
            prev_chain = obj.get("previous_ids_chain") or []
            oid = self._resolve_canonical(scene_id, raw_oid, prev_chain)

            skey = (scene_id, oid)
            session = self._sessions.get(skey)
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

        Continuous feed: every frame, SceneScape publishes all objects
        currently inside the region, each carrying ``regions.{name}.dwell``.
        We forward this dwell as a LOITER event; the rule engine's
        ``loitering`` rule (rules.yaml) decides whether to actually alert
        based on its ``dwell_seconds > threshold`` condition. Per-session
        dedup is handled in the rule adapter via ``loiter_alerted``.
        """
        scene_id_filter = self.config.get_accepted_scene_ids()
        if scene_id_filter and scene_id not in scene_id_filter:
            return

        zone_type = self.config.get_zone_type(region_id)
        if zone_type != "HIGH_VALUE":
            return

        zone_name = self.config.get_zone_name(region_id) or region_id
        now = datetime.now(timezone.utc)

        for obj in data.get("objects", []):
            raw_oid = str(obj.get("id", ""))
            if not raw_oid:
                continue
            prev_chain = obj.get("previous_ids_chain") or []
            oid = self._resolve_canonical(scene_id, raw_oid, prev_chain)

            # Skip if we've already alerted for this person/zone in this visit.
            session = self._sessions.get((scene_id, oid))
            if session and session.loiter_alerted.get(region_id):
                continue

            # Read SceneScape's authoritative dwell for this region.
            rinfo = (obj.get("regions") or {}).get(region_id)
            if not isinstance(rinfo, dict):
                # Fall back to the first region entry if the topic id key
                # doesn't appear (older SceneScape payload shape).
                for _rname, _rinfo in (obj.get("regions") or {}).items():
                    if isinstance(_rinfo, dict):
                        rinfo = _rinfo
                        break
            if not isinstance(rinfo, dict):
                continue
            try:
                dwell = float(rinfo.get("dwell", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue

            logger.info("region_data dwell",
                        object_id=oid[:8], region_id=region_id,
                        dwell=round(dwell, 1))

            event = RegionEvent(
                event_type=EventType.LOITER,
                object_id=oid,
                region_id=region_id,
                region_name=zone_name,
                zone_type=ZoneType(zone_type),
                timestamp=now,
                scene_id=scene_id,
                dwell_seconds=round(dwell, 1),
            )
            await self._emit(event)

    # ---- session expiry ------------------------------------------------------
    async def _expire_session(self, skey: tuple) -> None:
        session = self._sessions.get(skey)
        if session is None:
            return
        oid = session.object_id

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
                    entry_timestamp=(
                        visit.entry_time.isoformat() if visit.entry_time else ""
                    ),
                )
                await self._emit(event)

        # Remove session after EXITED events are processed
        del self._sessions[skey]

        # Drop any oid aliases that pointed at this canonical session so the
        # alias map doesn't grow unbounded across long runs.
        scene_id_expired, canonical_expired = skey
        stale_aliases = [
            k for k, v in self._oid_alias.items()
            if k[0] == scene_id_expired and v == canonical_expired
        ]
        for k in stale_aliases:
            del self._oid_alias[k]

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

        # Capture entry timestamp BEFORE exit_zone() drops it from current_zones,
        # so downstream consumers (frame cleanup) can scope work to this visit.
        entry_ts_iso = session.current_zones.get(region_id, "")

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
            entry_timestamp=entry_ts_iso,
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
                skey
                for skey, s in self._sessions.items()
                if (now - s.last_seen).total_seconds() > self.session_timeout
            ]
            for skey in expired:
                await self._expire_session(skey)
