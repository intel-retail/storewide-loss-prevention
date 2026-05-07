"""SceneScape regulated scene event consumer — region entry/exit tracking.

Subscribes to: scenescape/regulated/scene/{scene_id}
Payload: objects list with per-person `regions` dict
Region entry/exit is detected by diffing current vs previous region membership.

Payload structure per person object:
  {
    id: str,
    category: "person",
    visibility: [camera_id, ...],
    metadata: {...},
    regions: {
      region_id: {"entered": timestamp},
      ...
    }
  }
"""

from __future__ import annotations

import logging
import re
import time
from typing import Dict, Optional, Set

from backend.service.event_service import EventService
from backend.utils.thumbnail import submit_capture

log = logging.getLogger("poi.consumer.scenescape")

# Topic: scenescape/regulated/scene/{scene_id}  (SceneScape v2026+)
REGION_TOPIC_RE = re.compile(r"scenescape/regulated/scene/(?P<scene_id>[^/]+)$")


class ScenescapeRegionConsumer:
    """Handles region entry/exit events from the regulated scene MQTT topic.

    Uses stateful diffing: detects entry when a region_id appears in a person's
    `regions` dict for the first time, and exit when it disappears.
    """

    # Max age (seconds) before stale entries are evicted from _region_presence
    _PRESENCE_MAX_AGE = 3600  # 1 hour
    _EVICTION_INTERVAL = 60  # seconds between eviction sweeps

    def __init__(self, event_service: EventService, event_repo=None) -> None:
        self._event_service = event_service
        self._event_repo = event_repo  # RedisEventRepository for zone frame storage
        # {object_id: (set of region_ids, last_seen_timestamp)}
        self._region_presence: Dict[str, Set[str]] = {}
        self._last_seen: Dict[str, float] = {}
        self._last_eviction = 0.0

    def handle_event(self, topic: str, payload: dict) -> None:
        m = REGION_TOPIC_RE.match(topic)
        if not m:
            log.debug("Topic %s does not match regulated scene pattern, ignoring", topic)
            return

        self._evict_stale()

        scene_id = m.group("scene_id")
        timestamp = payload.get("timestamp", "")

        objects = payload.get("objects", [])
        # Regulated topic objects is a list; also accept dict for robustness
        if isinstance(objects, dict):
            persons = objects.get("person", [])
        elif isinstance(objects, list):
            persons = [o for o in objects if o.get("category") == "person" or o.get("type") == "person"]
        else:
            return

        current_ids: set = set()

        # ── Build UUID → camera_bounds mapping for bbox-based track resolution ──
        # The regulated scene provides camera_bounds for each UUID, enabling the
        # camera topic handler to map camera-local integer IDs to global UUIDs.
        if self._event_repo:
            cam_uuid_bounds: dict[str, dict[str, dict]] = {}  # camera_id → {uuid → bbox}
            for obj in persons:
                uid = obj.get("id", "")
                cam_bounds = obj.get("camera_bounds", {})
                for cam_id, bbox in cam_bounds.items():
                    if cam_id not in cam_uuid_bounds:
                        cam_uuid_bounds[cam_id] = {}
                    cam_uuid_bounds[cam_id][uid] = bbox
            if cam_uuid_bounds:
                for cam_id, uuid_bounds in cam_uuid_bounds.items():
                    try:
                        self._event_repo.store_uuid_camera_bounds(cam_id, uuid_bounds)
                    except Exception:
                        log.debug("Failed to store UUID camera bounds for %s", cam_id, exc_info=True)
                # Log at most once per eviction interval to avoid flooding
                now = time.monotonic()
                if now - getattr(self, "_last_uuid_log", 0) > self._EVICTION_INTERVAL:
                    self._last_uuid_log = now
                    log.info(
                        "UUID camera bounds updated: %s",
                        {c: len(u) for c, u in cam_uuid_bounds.items()},
                    )

        for obj in persons:
            object_id = obj.get("id", "")
            if not object_id:
                continue
            current_ids.add(object_id)

            cameras = obj.get("visibility", [])
            camera_id = cameras[0] if cameras else None

            # Regions dict: {region_id: {"entered": timestamp, ...}}
            regions_now: Set[str] = set(obj.get("regions", {}).keys())
            regions_before: Set[str] = self._region_presence.get(object_id, set())

            entered_regions = regions_now - regions_before
            exited_regions = regions_before - regions_now

            # Bounding box from payload (may not be present on regulated topic)
            bbox = obj.get("bounding_box_px") or obj.get("bounding_box")

            for region_id in entered_regions:
                region_info = obj.get("regions", {}).get(region_id, {})
                entry_ts = region_info.get("entered", timestamp)
                region_name = region_info.get("name", region_id)
                entry_frame_key = self._capture_zone_frame(
                    object_id, scene_id, region_id, "entry", camera_id, bbox
                )
                try:
                    self._event_service.store_region_entry(
                        object_id, entry_ts, scene_id, region_id, region_name, camera_id,
                        entry_frame_key=entry_frame_key,
                    )
                    log.info(
                        "Region ENTER: obj=%s scene=%s region=%s camera=%s frame=%s",
                        object_id, scene_id, region_id, camera_id,
                        "captured" if entry_frame_key else "none",
                    )
                except Exception:
                    log.exception("Error storing region entry for obj %s region %s", object_id, region_id)

            for region_id in exited_regions:
                region_name = region_id  # name not available on exit; use id
                exit_frame_key = self._capture_zone_frame(
                    object_id, scene_id, region_id, "exit", camera_id, bbox
                )
                try:
                    self._event_service.store_region_exit(
                        object_id, timestamp, scene_id, region_id, region_name,
                        exit_frame_key=exit_frame_key,
                    )
                    log.info(
                        "Region EXIT: obj=%s scene=%s region=%s frame=%s",
                        object_id, scene_id, region_id,
                        "captured" if exit_frame_key else "none",
                    )
                except Exception:
                    log.exception("Error storing region exit for obj %s region %s", object_id, region_id)

            self._region_presence[object_id] = regions_now
            self._last_seen[object_id] = time.monotonic()

        # Clean up presence tracking for objects no longer in scene
        gone_ids = set(self._region_presence.keys()) - current_ids
        for object_id in gone_ids:
            old_regions = self._region_presence.pop(object_id, set())
            self._last_seen.pop(object_id, None)
            for region_id in old_regions:
                try:
                    self._event_service.store_region_exit(
                        object_id, timestamp, scene_id, region_id, region_id,
                        exit_frame_key=None,  # object already gone; no frame available
                    )
                    log.info(
                        "Region EXIT (object left scene): obj=%s scene=%s region=%s",
                        object_id, scene_id, region_id,
                    )
                except Exception:
                    log.exception("Error storing implicit region exit for obj %s", object_id)

    def _capture_zone_frame(
        self,
        object_id: str,
        scene_id: str,
        region_id: str,
        event_type: str,  # "entry" or "exit"
        camera_id: Optional[str],
        bbox,
    ) -> Optional[str]:
        """Capture a frame thumbnail and store it in Redis. Returns the Redis key or None."""
        if not camera_id or not self._event_repo:
            return None
        try:
            future = submit_capture(camera_id, bbox)
            b64 = future.result(timeout=4)
            if not b64:
                return None
            frame_key = f"zone:frame:{object_id}:{scene_id}:{region_id}:{event_type}"
            self._event_repo.store_zone_frame(frame_key, b64)
            return frame_key
        except Exception:
            log.debug(
                "Zone frame capture failed: obj=%s region=%s event=%s",
                object_id, region_id, event_type, exc_info=True,
            )
            return None

    def _evict_stale(self) -> None:
        """Remove entries older than _PRESENCE_MAX_AGE to prevent unbounded growth."""
        now = time.monotonic()
        if now - self._last_eviction < self._EVICTION_INTERVAL:
            return
        self._last_eviction = now
        cutoff = now - self._PRESENCE_MAX_AGE
        stale = [oid for oid, ts in self._last_seen.items() if ts < cutoff]
        for oid in stale:
            self._region_presence.pop(oid, None)
            self._last_seen.pop(oid, None)
        if stale:
            log.info("Evicted %d stale entries from region presence tracker", len(stale))
