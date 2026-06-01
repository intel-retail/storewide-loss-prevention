"""SceneScape region consumer — UUID camera_bounds mapping + native region events.

Subscribes to TWO topics:
  1. scenescape/regulated/scene/{scene_id}
     → Extracts UUID→camera_bounds mapping for bbox-based track resolution.
       Entry/exit is NOT derived here (no stateful diffing).

  2. scenescape/event/region/{scene_id}/{region_id}/{suffix}
     → SceneScape's native region entry/exit events with server-computed dwell.
       Provides explicit ``entered`` / ``exited`` lists — no diffing needed.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

from backend.service.event_service import EventService
from backend.utils.thumbnail import submit_capture

log = logging.getLogger("poi.consumer.scenescape")

# Topic: scenescape/regulated/scene/{scene_id}  — camera_bounds only
REGION_TOPIC_RE = re.compile(r"scenescape/regulated/scene/(?P<scene_id>[^/]+)$")

# Topic: scenescape/event/region/{scene_id}/{region_id}/{suffix}  — native entry/exit
REGION_EVENT_TOPIC_RE = re.compile(
    r"scenescape/event/region/(?P<scene_id>[^/]+)/(?P<region_id>[^/]+)/(?P<suffix>[^/]+)$"
)

# Minimum interval between UUID camera_bounds log lines
_UUID_LOG_INTERVAL = 60  # seconds


class ScenescapeRegionConsumer:
    """Handles SceneScape region tracking via two complementary topics.

    - Regulated scene: UUID→camera_bounds mapping (supports UUID resolution
      in the camera topic MQTT consumer for POI matching).
    - Region event: native ENTERED/EXITED events with dwell from SceneScape.
    """

    def __init__(self, event_service: EventService, event_repo=None) -> None:
        self._event_service = event_service
        self._event_repo = event_repo
        self._last_uuid_log = 0.0

    # ── Regulated scene topic: camera_bounds only ────────────────────────────

    def handle_event(self, topic: str, payload: dict) -> None:
        """Process scenescape/regulated/scene/{scene_id}.

        Extracts UUID→camera_bounds for each camera and stores in Redis.
        Region entry/exit is handled by handle_region_event() instead.
        """
        m = REGION_TOPIC_RE.match(topic)
        if not m:
            log.debug("Topic %s does not match regulated scene pattern, ignoring", topic)
            return

        if not self._event_repo:
            return

        objects = payload.get("objects", [])
        if isinstance(objects, dict):
            persons = objects.get("person", [])
        elif isinstance(objects, list):
            persons = [o for o in objects if o.get("category") == "person" or o.get("type") == "person"]
        else:
            return

        cam_uuid_bounds: dict[str, dict[str, dict]] = {}
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
            now = time.monotonic()
            if now - self._last_uuid_log > _UUID_LOG_INTERVAL:
                self._last_uuid_log = now
                log.info(
                    "UUID camera bounds updated: %s",
                    {c: len(u) for c, u in cam_uuid_bounds.items()},
                )

    # ── Native region event topic: explicit entry/exit ───────────────────────

    def handle_region_event(self, scene_id: str, region_id: str, data: dict) -> None:
        """Process scenescape/event/region/{scene_id}/{region_id}/{suffix}.

        SceneScape sends explicit ``entered`` and ``exited`` lists with
        server-computed dwell time — no stateful diffing required.

        Payload format:
          {
            "entered": [{"id": "uuid", "visibility": [...], "regions": {...}}],
            "exited":  [{"object": {"id": "uuid"}, "dwell": 5.2}],
            "timestamp": "2026-06-01T17:23:01.104Z"
          }
        """
        timestamp = data.get("timestamp", "")

        # ── Process entries ──
        for obj in data.get("entered", []):
            object_id = str(obj.get("id", obj.get("object_id", "")))
            if not object_id:
                continue
            cameras = obj.get("visibility", [])
            camera_id = cameras[0] if cameras else None

            region_info = (obj.get("regions") or {}).get(region_id, {})
            entry_ts = region_info.get("entered", timestamp)
            region_name = region_info.get("name", region_id)

            bbox = obj.get("bounding_box_px") or obj.get("bounding_box")
            entry_frame_key = self._capture_zone_frame(
                object_id, scene_id, region_id, "entry", camera_id, bbox,
            )
            try:
                self._event_service.store_region_entry(
                    object_id, entry_ts, scene_id, region_id, region_name, camera_id,
                    entry_frame_key=entry_frame_key,
                )
                log.info(
                    "Region ENTER (native): obj=%s scene=%s region=%s camera=%s",
                    object_id, scene_id, region_id, camera_id,
                )
            except Exception:
                log.exception("Error storing region entry for obj %s region %s", object_id, region_id)

        # ── Process exits ──
        for exit_entry in data.get("exited", []):
            obj = exit_entry.get("object", exit_entry)
            object_id = str(obj.get("id", obj.get("object_id", "")))
            if not object_id:
                continue
            dwell = exit_entry.get("dwell")
            cameras = obj.get("visibility", [])
            camera_id = cameras[0] if cameras else None

            exit_frame_key = self._capture_zone_frame(
                object_id, scene_id, region_id, "exit", camera_id, None,
            )
            try:
                self._event_service.store_region_exit(
                    object_id, timestamp, scene_id, region_id, region_id,
                    exit_frame_key=exit_frame_key,
                    dwell_override=dwell,
                )
                log.info(
                    "Region EXIT (native): obj=%s scene=%s region=%s dwell=%.1fs",
                    object_id, scene_id, region_id, dwell or 0.0,
                )
            except Exception:
                log.exception("Error storing region exit for obj %s region %s", object_id, region_id)

    # ── Shared helpers ───────────────────────────────────────────────────────

    def _capture_zone_frame(
        self,
        object_id: str,
        scene_id: str,
        region_id: str,
        event_type: str,
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
