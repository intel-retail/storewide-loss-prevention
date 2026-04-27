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

        # Sessions keyed by (scene_id, canonical_object_id) to support
        # multi-scene. Canonical id is the earliest UUID in a re-id chain;
        # all later flickering UUIDs are aliased to it via _oid_alias below.
        self._sessions: Dict[tuple, PersonSession] = {}

        # Maps (scene_id, raw_oid) -> canonical_oid. Populated lazily as
        # SceneScape emits ``previous_ids_chain`` linking new tracks to old.
        # Once an oid has an alias, all handlers route updates to the
        # canonical session, preventing duplicate "ghost" entries in the UI.
        self._oid_alias: Dict[tuple, str] = {}

        self._event_handlers: List[Callable] = []
        self._expiry_task: Optional[asyncio.Task] = None

        # Wallclock-based per-canonical loiter timer, keyed by
        # (scene_id, region_id, canonical_oid). Each entry: {
        #   "first_seen":  float,        # wall-clock when this person entered
        #   "last_update": float,        # wall-clock last seen in region
        #   "alerted":     bool,         # one alert per continuous occupancy
        # }
        # Each canonical person gets their own timer per region, so one
        # person's loitering cannot be attributed to a different person who
        # happens to walk in shortly afterward.
        self._region_loiter_state: Dict[tuple, dict] = {}
        # If a person goes unseen in the region for longer than this, their
        # individual timer resets.
        self._loiter_coalesce_gap = float(
            rules.get("loiter_coalesce_gap_seconds", 5.0)
        )

        logger.info("SessionManager initialized", timeout=self.session_timeout,
                    allowed_cameras=sorted(self._allowed_cameras) or "all")

    # ---- event handler registration -----------------------------------------
    def register_event_handler(self, handler: Callable) -> None:
        """Register an async handler that receives RegionEvent objects."""
        self._event_handlers.append(handler)

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
        return self._sessions.get((scene_id, object_id))

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
                # Promote re-id state if SceneScape has now matched this track.
                new_reid = str(obj.get("reid_state", "") or "")
                if new_reid and session.reid_state != new_reid:
                    session.reid_state = new_reid
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
                    reid_state=str(obj.get("reid_state", "") or ""),
                )
                self._sessions[skey] = session
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

        # Region must currently be occupied by at least one person for this
        # message to count as "continuous occupancy".
        objs = [o for o in data.get("objects", []) if o.get("regions")]
        if not objs:
            return

        # Per-canonical loiter: each unique person in the region maintains
        # their own wallclock timer. We update every canonical seen in this
        # message and emit at most one LOITER per (scene, region, canonical).
        for obj in objs:
            raw_oid = str(obj.get("id", ""))
            if not raw_oid:
                continue
            prev_chain = obj.get("previous_ids_chain") or []
            canonical = self._resolve_canonical(scene_id, raw_oid, prev_chain)

            # Read instant dwell (best-effort, for logging only)
            instant_dwell = 0.0
            for _rname, rinfo in obj["regions"].items():
                try:
                    instant_dwell = float(rinfo.get("dwell", 0.0) or 0.0)
                except (TypeError, ValueError):
                    instant_dwell = 0.0
                break

            pkey = (scene_id, region_id, canonical)
            pstate = self._region_loiter_state.get(pkey)
            if pstate and (now_epoch - pstate["last_update"]) > self._loiter_coalesce_gap:
                # This canonical left the region long enough ago that we
                # consider their previous burst over; restart their timer.
                logger.info("loiter state reset (person silence gap)",
                            object_id=canonical, region_id=region_id,
                            gap=round(now_epoch - pstate["last_update"], 1))
                pstate = None
            if pstate is None:
                pstate = {"first_seen": now_epoch, "last_update": now_epoch,
                          "alerted": False}
                self._region_loiter_state[pkey] = pstate

            pstate["last_update"] = now_epoch

            if pstate["alerted"]:
                continue

            person_dwell = now_epoch - pstate["first_seen"]
            session = self._sessions.get((scene_id, canonical))

            logger.info("region_data dwell",
                        object_id=canonical, region_id=region_id,
                        instant=round(instant_dwell, 1),
                        total=round(person_dwell, 1),
                        threshold=self._loiter_threshold)

            if person_dwell <= self._loiter_threshold:
                continue

            # Hold off alerting until re-id matches; the per-canonical timer
            # keeps counting, so the alert fires as soon as the canonical is
            # promoted to "matched".
            if session is not None and session.reid_state and session.reid_state != "matched":
                continue

            zone_name = self.config.get_zone_name(region_id) or region_id
            event = RegionEvent(
                event_type=EventType.LOITER,
                object_id=canonical,
                region_id=region_id,
                region_name=zone_name,
                zone_type=ZoneType(zone_type),
                timestamp=datetime.now(timezone.utc),
                scene_id=session.scene_id if session else scene_id,
                dwell_seconds=round(person_dwell, 1),
            )
            pstate["alerted"] = True
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

        # Drop this canonical's per-region loiter timers.
        stale_loiter = [
            k for k in self._region_loiter_state
            if k[0] == scene_id_expired and k[2] == canonical_expired
        ]
        for k in stale_loiter:
            del self._region_loiter_state[k]

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
                skey
                for skey, s in self._sessions.items()
                if (now - s.last_seen).total_seconds() > self.session_timeout
            ]
            for skey in expired:
                await self._expire_session(skey)
