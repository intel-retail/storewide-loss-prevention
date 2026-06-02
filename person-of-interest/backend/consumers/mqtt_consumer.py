"""MQTT event consumer — processes SceneScape scene events for POI face matching.

Primary topic: scenescape/data/camera/{camera_id}
  Payload: per-camera DLStreamer output with person detections and face sub_objects.
  Face sub_objects contain face-reid embeddings from face-reidentification-retail-0095
  — the SAME model used during POI enrollment.

  Track key resolution (priority order):
    1. Temporal continuity cache: reuse a previously-resolved UUID for the same
       camera-local person ID (cam:{camera}:{int_id} → UUID).
    2. External topic visibility: if exactly one UUID is visible on this camera
       (from scenescape/external/{scene_id}/person), use it directly.
    3. IoU tiebreaker: if multiple UUIDs are visible, match bounding boxes
       (only non-projected camera_bounds with positive dimensions).
    4. Fallback: f"cam:{camera_id}:{person_int_id}" (may recycle).

Secondary topic (monitoring + visibility): scenescape/external/{scene_id}/person
  Carries global UUIDs, reid_state, visibility (camera list), and body-reid
  embeddings.  Used to build per-camera UUID visibility index for resolution.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import struct
import time
from typing import List, Optional

import numpy as np

from backend.observer.events import EventBus, MatchFoundEvent
from backend.service.alert_service import AlertService
from backend.service.event_service import EventService
from backend.service.matching_service import MatchingService
from backend.utils.thumbnail import grab_frame_now, submit_capture, base64_to_frame, crop_bbox, frame_to_base64_jpeg

log = logging.getLogger("poi.consumer")

# Primary: camera topic — face embeddings from face-reidentification-retail-0095
CAMERA_TOPIC_RE = re.compile(r"scenescape/data/camera/(?P<camera_id>[^/]+)$")

# Secondary (monitoring only): external topic — body embeddings + global UUID + reid_state
EXTERNAL_TOPIC_RE = re.compile(r"scenescape/external/(?P<scene_id>[^/]+)/person$")

# Keep alias for existing imports
TOPIC_RE = CAMERA_TOPIC_RE

# Minimum face detection confidence to attempt FAISS matching
FACE_CONFIDENCE_THRESHOLD = 0.80

# SceneScape reid_state values — kept for logging/observability purposes only.
# These no longer gate FAISS execution; they are recorded to reid:meta for MCP tools.
REID_MATCHED_STATES = {"matched", "query_no_match"}


def _decode_embedding_b64(b64_str: str) -> Optional[List[float]]:
    """Decode a base64-encoded IEEE-754 float32 embedding vector."""
    try:
        raw = base64.b64decode(b64_str)
        n = len(raw) // 4
        if n == 0:
            return None
        return list(struct.unpack(f"{n}f", raw))
    except Exception:
        log.debug("Failed to decode base64 embedding")
        return None


def _parse_embedding(raw) -> Optional[List[float]]:
    """Parse an embedding from any wire format.

    Handles:
      - list / nested list  ([[f1, f2, ...]] or [f1, f2, ...])
      - JSON string         ('[[f1, f2, ...]]' or '[f1, f2, ...]')
      - base64 string       (IEEE-754 float32 packed binary, legacy camera topic)
    """
    if raw is None or raw == "":
        return None

    if isinstance(raw, list):
        flat = raw[0] if raw and isinstance(raw[0], list) else raw
        try:
            return [float(x) for x in flat] if flat else None
        except (TypeError, ValueError):
            return None

    if isinstance(raw, str):
        # Try JSON array first (external topic format)
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                flat = parsed[0] if parsed and isinstance(parsed[0], list) else parsed
                return [float(x) for x in flat] if flat else None
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        # Fall back to base64 binary (legacy camera topic format)
        return _decode_embedding_b64(raw)

    return None


class EventConsumer:
    """Consumes MQTT scene events and orchestrates matching + alerting."""

    def __init__(
        self,
        matching_service: MatchingService,
        event_service: EventService,
        alert_service: AlertService,
        event_bus: EventBus,
        event_repo=None,
        detection_index=None,
    ) -> None:
        self._matching = matching_service
        self._events = event_service
        self._alerts = alert_service
        self._event_bus = event_bus
        self._event_repo = event_repo  # used for thumbnail Redis ops
        self._detection_index = detection_index  # DetectionIndexRepository or None

    def handle_event(self, topic: str, payload: dict) -> None:
        """Route incoming MQTT message to the appropriate handler."""
        if CAMERA_TOPIC_RE.match(topic):
            self._handle_camera_event(topic, payload)
        elif EXTERNAL_TOPIC_RE.match(topic):
            self._handle_external_event(topic, payload)

    # ── Primary: camera topic face embeddings ───────────────────────────────

    def _handle_camera_event(self, topic: str, payload: dict) -> None:
        """Process scenescape/data/camera/{camera_id} messages.

        Extracts face sub_object embeddings (face-reidentification-retail-0095)
        for FAISS matching.  Body-level reid embeddings are ignored — they come
        from person-reidentification-retail-0277, a different embedding space.

        Only persons whose face sub_objects exceed FACE_CONFIDENCE_THRESHOLD are
        processed.  Dedup key is f"cam:{camera_id}:{person_int_id}" with 60s TTL
        to suppress repeated alerts for the same person-in-frame.
        """
        m = CAMERA_TOPIC_RE.match(topic)
        if not m:
            return

        mqtt_receive_time_ms = int(time.time() * 1000)
        camera_id = m.group("camera_id")
        timestamp = payload.get("timestamp", "")

        # Log DLStreamer pipeline latency (frame decode → MQTT arrival)
        if timestamp:
            try:
                from datetime import datetime as _dt
                frame_ts_ms = int(
                    _dt.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp() * 1000
                )
                pipeline_latency_ms = mqtt_receive_time_ms - frame_ts_ms
                if pipeline_latency_ms > 1000:
                    log.info(
                        "DLStreamer pipeline latency: %dms camera=%s",
                        pipeline_latency_ms, camera_id,
                    )
            except Exception:
                log.debug(
                    "Failed to parse frame timestamp for latency calculation: "
                    "timestamp=%s camera=%s",
                    timestamp,
                    camera_id,
                    exc_info=True,
                )

        objects = payload.get("objects", {})
        if isinstance(objects, dict):
            persons = objects.get("person", [])
        elif isinstance(objects, list):
            persons = [o for o in objects
                       if o.get("category") == "person" or o.get("type") == "person"]
        else:
            return

        seen_ids: set = set()

        for obj in persons:
            person_int_id = obj.get("id")
            if person_int_id is None or person_int_id in seen_ids:
                continue
            seen_ids.add(person_int_id)

            # Only process persons that have face sub_objects with usable embeddings.
            # Body-level embedding is intentionally ignored (wrong embedding space).
            embedding_vector: Optional[List[float]] = None
            best_face_conf = 0.0
            best_face_bbox = None

            for face in obj.get("sub_objects", {}).get("face", []):
                face_conf = face.get("confidence", 0.0)
                if face_conf < FACE_CONFIDENCE_THRESHOLD:
                    log.debug(
                        "Skipping low-confidence face: camera=%s person=%s conf=%.3f",
                        camera_id, person_int_id, face_conf,
                    )
                    continue
                raw = face.get("metadata", {}).get("reid", {}).get("embedding_vector", "")
                vec = _parse_embedding(raw)
                if vec and face_conf > best_face_conf:
                    embedding_vector = vec
                    best_face_conf = face_conf
                    best_face_bbox = face.get("bounding_box_px")

            if not embedding_vector:
                log.debug("No face embedding for camera=%s person=%s — skipping FAISS", camera_id, person_int_id)
                continue

            # Validate embedding dimension before FAISS search
            expected_dim = self._matching._cfg.faiss_dimension
            if len(embedding_vector) != expected_dim:
                log.warning(
                    "Wrong embedding dimension: camera=%s person=%s dim=%d expected=%d — skipping",
                    camera_id, person_int_id, len(embedding_vector), expected_dim,
                )
                continue

            log.info(
                "Face embedding found: camera=%s person=%s conf=%.3f dim=%d",
                camera_id, person_int_id, best_face_conf, len(embedding_vector),
            )

            # ── Resolve global UUID from SceneScape ─────────────────────────
            # Resolution strategy (in priority order):
            #   1. Temporal continuity cache: cam:Camera:N → UUID (from prior resolution)
            #   2. External topic visibility: which UUIDs are visible on this camera
            #      - If exactly 1 UUID → unambiguous match
            #      - If multiple → IoU tiebreaker with non-projected bounds
            #   3. Fallback to camera-local ID (cam:Camera:N)
            person_bbox = obj.get("bounding_box_px") or best_face_bbox
            resolved_uuid: Optional[str] = None
            resolution_method: str = "none"

            if self._event_repo:
                # 1. Check temporal continuity cache
                try:
                    cached = self._event_repo.get_uuid_for_camid(camera_id, person_int_id)
                    if cached:
                        resolved_uuid = cached
                        resolution_method = "cached"
                except Exception:
                    pass

                # 2. If not cached, check visibility from external topic
                if not resolved_uuid:
                    try:
                        visible = self._event_repo.get_visible_uuids(camera_id, max_age_s=5.0)
                        if len(visible) == 1:
                            resolved_uuid = visible[0]
                            resolution_method = "visibility_single"
                        elif len(visible) > 1 and person_bbox:
                            # Multiple UUIDs visible — try IoU with non-projected bounds
                            resolved_uuid = self._event_repo.get_uuid_for_camera_bbox(
                                camera_id, person_bbox, iou_threshold=0.3,
                            )
                            if resolved_uuid:
                                resolution_method = "iou_tiebreaker"
                    except Exception:
                        log.debug("UUID visibility lookup failed for camera=%s", camera_id, exc_info=True)

                # Cache the mapping for subsequent frames
                if resolved_uuid and resolution_method != "cached":
                    try:
                        self._event_repo.store_camid_uuid_mapping(
                            camera_id, person_int_id, resolved_uuid,
                        )
                    except Exception:
                        pass

            # Use UUID as track key if resolved; otherwise fall back to camera-local ID.
            if resolved_uuid:
                object_id = resolved_uuid
                log.info(
                    "Resolved UUID [%s]: camera=%s person=%s → uuid=%s",
                    resolution_method, camera_id, person_int_id, resolved_uuid,
                )
            else:
                object_id = f"cam:{camera_id}:{person_int_id}"
                log.debug(
                    "No UUID resolved for camera=%s person=%s — using camera-local ID %s",
                    camera_id, person_int_id, object_id,
                )
            # ── Detection index: store face embeddings for offline search ──
            # We store multiple embeddings per appearance (up to N, spaced by
            # an interval) to improve search recall.
            #
            # Appearance lifecycle:
            #   - claim_track succeeds → new appearance window → store entry
            #   - claim_track fails → existing window → continuity check first,
            #     then rate-limited sampling if same person
            #
            # Continuity check runs BEFORE sampling to prevent wrong-person
            # embeddings from being stored when the tracker recycles IDs.
            new_faiss_id: int = -1
            appearance_id = object_id  # fallback if detection index disabled
            _continuity_ok = True  # True for new appearances, checked for existing
            if self._detection_index is not None:
                _det_vec = np.array(embedding_vector, dtype=np.float32)
                try:
                    if self._detection_index.claim_track(object_id):
                        # New appearance window — create unique appearance_id
                        appearance_id = f"{object_id}@{int(time.time())}"
                        self._detection_index.set_active_appearance(object_id, appearance_id)
                        new_faiss_id = self._detection_index.add(
                            vector=_det_vec,
                            camera_id=camera_id,
                            track_id=appearance_id,
                            timestamp=timestamp,
                            bbox=best_face_bbox,
                        )
                        log.info(
                            "DetectionIndex: new appearance stored %s (object=%s) faiss_id=%d",
                            appearance_id, object_id, new_faiss_id,
                        )
                    else:
                        # Existing appearance window — check continuity first
                        appearance_id = (
                            self._detection_index.get_active_appearance(object_id) or object_id
                        )
                        # Compare new embedding against entry vector to detect
                        # tracker ID recycling (different person reusing same ID)
                        _entry_raw = self._detection_index.get_entry_vector(appearance_id)
                        if _entry_raw is not None:
                            _entry_vec = _entry_raw.reshape(1, -1)
                            _entry_norm = np.linalg.norm(_entry_vec)
                            _new_vec = _det_vec.reshape(1, -1)
                            _new_norm = np.linalg.norm(_new_vec)
                            if _entry_norm > 1e-9 and _new_norm > 1e-9:
                                _sim = float(np.dot(
                                    _entry_vec / _entry_norm,
                                    (_new_vec / _new_norm).T,
                                )[0, 0])
                                if _sim < 0.50:
                                    _continuity_ok = False
                                    log.debug(
                                        "Continuity check: sim=%.3f for %s — "
                                        "skipping sample + exit update",
                                        _sim, appearance_id,
                                    )
                                    # Shorten gate so old exit promotes quickly
                                    # and the recycled ID can start a new appearance
                                    self._detection_index.shorten_track_gate(object_id, ttl=5)
                                    # Clear cached UUID mapping — tracker recycled
                                    # this person_int_id to a different person
                                    if self._event_repo and not object_id.startswith("cam:"):
                                        try:
                                            self._event_repo.clear_camid_uuid_mapping(
                                                camera_id, person_int_id,
                                            )
                                        except Exception:
                                            pass

                        # Refresh gate and active_appearance only after
                        # continuity passes — prevents recycled-ID detections
                        # from keeping a stale gate alive.
                        if _continuity_ok:
                            self._detection_index.refresh_track_gate(object_id)
                            self._detection_index.refresh_active_appearance(object_id)

                        # Only sample if continuity check passed
                        if _continuity_ok and self._detection_index.should_sample(appearance_id):
                            new_faiss_id = self._detection_index.add(
                                vector=_det_vec,
                                camera_id=camera_id,
                                track_id=appearance_id,
                                timestamp=timestamp,
                                bbox=best_face_bbox,
                            )
                            log.debug(
                                "DetectionIndex: sampled additional embedding %s faiss_id=%d",
                                appearance_id, new_faiss_id,
                            )
                except Exception:
                    log.debug("DetectionIndex.add failed for %s", object_id, exc_info=True)

            # ── Track-level entry / last-seen frames ──────────────────────────
            # Grab the frame synchronously RIGHT NOW — the MQTTAdapter already cached
            # the image for this camera (image topic message arrives BEFORE the
            # detection message on the same connection because sscape_adapter publishes
            # them in that order in the same processFrame() call).
            # grab_frame_now() is O(1) and never blocks the consumer thread.
            frame_b64: Optional[str] = grab_frame_now(camera_id, timestamp)
            if not frame_b64:
                log.debug("No frame available for camera=%s at ts=%s", camera_id, timestamp)

            # Crop to person body bbox (head-to-toe) so stored frames show the
            # full person, matching what online alerts display.  Falls back to
            # face bbox, then full camera frame.
            frame_b64_cropped: Optional[str] = None
            _crop_box = person_bbox or best_face_bbox
            if frame_b64 and _crop_box:
                _frame_np = base64_to_frame(frame_b64)
                if _frame_np is not None:
                    _cropped = crop_bbox(_frame_np, _crop_box)
                    if _cropped is not None and _cropped.size > 0:
                        frame_b64_cropped = frame_to_base64_jpeg(_cropped)
            if frame_b64_cropped is None:
                frame_b64_cropped = frame_b64  # fallback to full frame

            # Store cropped frame keyed by faiss_id (unique per detection, never overwritten).
            if new_faiss_id >= 0 and frame_b64_cropped and self._detection_index is not None:
                self._detection_index.store_frame(new_faiss_id, frame_b64_cropped)
                log.info("Detection frame stored: faiss_id=%d track=%s", new_faiss_id, appearance_id)

            # Update the rolling exit vector for this appearance (overwritten
            # each detection).  When the person leaves, the last value stored
            # here becomes the effective "exit" embedding.
            # Only update if continuity check passed (same person still in frame).
            if self._detection_index is not None and _continuity_ok:
                _exit_vec = np.array(embedding_vector, dtype=np.float32)
                try:
                    self._detection_index.update_exit(
                        track_id=appearance_id,
                        vector=_exit_vec,
                        camera_id=camera_id,
                        timestamp=timestamp,
                        bbox=best_face_bbox,
                        b64_frame=frame_b64_cropped,
                    )
                except Exception:
                    log.debug("update_exit failed for %s", appearance_id, exc_info=True)

            if self._event_repo is not None:
                self._capture_track_frames(object_id, frame_b64)

            self._run_matching(
                object_id=object_id,
                embedding_vector=embedding_vector,
                timestamp=timestamp,
                camera_id=camera_id,
                confidence=best_face_conf,
                bounding_box=person_bbox,
                face_bbox=best_face_bbox,
                mqtt_receive_time_ms=mqtt_receive_time_ms,
            )

    # ── Secondary: external topic (monitoring / UUID tracking only) ──────────

    def _capture_track_frames(self, object_id: str, frame_b64: Optional[str]) -> None:
        """Store a pre-fetched frame as entry (first time) and last_seen (always).

        frame_b64 was grabbed synchronously in _handle_camera_event, so it is
        guaranteed to be from the detection moment.  This method only performs
        Redis writes — fast, no network capture, no thread pool needed.
        """
        if not frame_b64:
            return
        from backend.core.config import get_config
        track_ttl = get_config().track_seen_ttl
        try:
            if self._event_repo.claim_track_entry(object_id, ttl=track_ttl):
                self._event_repo.store_track_frame(object_id, "entry", frame_b64)
                log.info("Track entry frame stored: %s", object_id)
        except Exception:
            log.debug("Track entry frame store failed for %s", object_id, exc_info=True)

        try:
            self._event_repo.store_track_frame(object_id, "last_seen", frame_b64)
        except Exception:
            log.debug("Track last-seen frame store failed for %s", object_id, exc_info=True)

    def _handle_external_event(self, topic: str, payload: dict) -> None:
        """Process scenescape/external/{scene_id}/person messages.

        Monitoring only — body-reid embeddings on this topic are from
        person-reidentification-retail-0277, a different embedding space from
        the face-reidentification-retail-0095 model used for POI enrollment.
        FAISS matching is NOT performed here.

        Logs reid_state transitions for observability and stores movement events
        so the timeline tracks all detected persons, not only those with faces.
        """
        m = EXTERNAL_TOPIC_RE.match(topic)
        if not m:
            return

        timestamp = payload.get("timestamp", "")
        scene_name = payload.get("name", m.group("scene_id"))

        objects = payload.get("objects", [])
        if isinstance(objects, dict):
            persons = objects.get("person", [])
        elif isinstance(objects, list):
            persons = [o for o in objects
                       if o.get("type") == "person" or o.get("category") == "person"]
        else:
            return

        # Build per-camera UUID visibility from all persons in this message.
        # Each person has visibility=[Camera_01, Camera_02, ...] listing cameras
        # that can see them.  We aggregate into {camera_id: [uuid, ...]} and
        # store in Redis so the camera topic handler can resolve UUIDs without
        # relying on fragile IoU matching with projected camera_bounds.
        cam_uuids: dict[str, list[str]] = {}

        for obj in persons:
            object_id = obj.get("id")
            if not object_id:
                continue
            reid_state = obj.get("reid_state", "")
            visibility = obj.get("visibility", [])
            camera_id = visibility[0] if visibility else scene_name

            # Accumulate UUID visibility per camera
            for cam in visibility:
                cam_uuids.setdefault(cam, []).append(object_id)

            if self._event_repo:
                # Store reid metadata for observability (MCP tools, match enrichment).
                # This is NOT a gate — the camera topic handler no longer checks this.
                meta = {
                    "global_uuid": object_id,
                    "reid_state": reid_state,
                    "camera_id": camera_id,
                    "timestamp": timestamp,
                    "similarity": obj.get("similarity"),
                    "first_seen": obj.get("first_seen"),
                }
                self._event_repo.set_reid_meta(object_id, meta)
            log.debug(
                "External topic: uuid=%s reid_state=%r camera=%s visibility=%s",
                object_id, reid_state, camera_id, visibility,
            )

            self._events.store_movement(
                object_id=object_id,
                timestamp=timestamp,
                camera_id=camera_id,
                region=camera_id,
            )

        # Store per-camera UUID visibility in Redis
        if self._event_repo and cam_uuids:
            for cam_id, uuid_list in cam_uuids.items():
                try:
                    self._event_repo.store_uuid_visibility(cam_id, uuid_list)
                except Exception:
                    log.debug("Failed to store UUID visibility for %s", cam_id, exc_info=True)

    # ── Shared matching + alerting ──────────────────────────────────────────

    def _run_matching(
        self,
        object_id: str,
        embedding_vector: List[float],
        timestamp: str,
        camera_id: Optional[str],
        confidence: float,
        bounding_box,
        mqtt_receive_time_ms: int = 0,
        face_bbox=None,
    ) -> None:
        """Run FAISS lookup and emit alerts for a confirmed detection.

        Args:
            bounding_box: person-level bbox (used for alert metadata / context).
            face_bbox: face sub_object bbox that produced the embedding
                       (used for thumbnail crop — guarantees same person).
        """
        display_camera = camera_id or "unknown"

        # Record movement before matching attempt
        self._events.store_movement(
            object_id=object_id,
            timestamp=timestamp,
            camera_id=display_camera,
            region=display_camera,
        )

        match = self._matching.match_object(object_id, embedding_vector)
        if match is None:
            return

        log.info(
            "POI match: poi=%s uuid=%s camera=%s similarity=%.3f",
            match.poi_id, object_id, display_camera, match.similarity_score,
        )

        # Write match metadata to Redis (accessible via API / MCP tools)
        if self._event_repo:
            match_meta = {
                "object_id": object_id,
                "poi_id": match.poi_id,
                "similarity_score": round(match.similarity_score, 4),
                "camera_id": display_camera,
                "timestamp": timestamp,
                "confidence": round(confidence, 4),
            }
            # Attach global reid metadata if available
            reid_meta_raw = None
            try:
                _raw = self._event_repo._r.get(f"reid:meta:{object_id}")  # type: ignore[attr-defined]
                if _raw:
                    reid_meta_raw = json.loads(_raw)
            except Exception:
                pass
            if reid_meta_raw:
                match_meta["reid"] = reid_meta_raw
            self._event_repo.set_match_metadata(object_id, match_meta, ttl=3600)
            log.debug("Match metadata written to Redis for uuid=%s", object_id)

        # Capture thumbnail — crop the MATCHED PERSON from the inline-cached frame.
        # Uses the person body bounding_box (head-to-toe) rather than face_bbox
        # so the thumbnail shows the full person, not just a tight face crop.
        # Each alert gets its own thumbnail keyed by a unique alert-thumbnail ID.
        thumbnail_b64: Optional[str] = None
        if camera_id:
            b64 = grab_frame_now(camera_id, timestamp)
            crop_box = bounding_box or face_bbox
            if b64 is not None and crop_box:
                frame = base64_to_frame(b64)
                if frame is not None:
                    cropped = crop_bbox(frame, crop_box)
                    if cropped is not None and cropped.size > 0:
                        thumbnail_b64 = frame_to_base64_jpeg(cropped)
                    else:
                        log.warning(
                            "Face crop failed for uuid=%s camera=%s face_bbox=%s — using full frame",
                            object_id, camera_id, crop_box,
                        )
                if thumbnail_b64 is None:
                    thumbnail_b64 = b64  # fall back to full frame
            if b64 is not None and thumbnail_b64 is None:
                thumbnail_b64 = b64  # no crop_box but valid frame — use full frame
            if thumbnail_b64 is None:
                # MQTT ring buffer empty — fall back to async RTSP capture
                future = submit_capture(camera_id, crop_box, timestamp)
                try:
                    thumbnail_b64 = future.result(timeout=6)
                except Exception:
                    thumbnail_b64 = None
                    log.warning("Thumbnail timed out or failed for uuid=%s camera=%s", object_id, camera_id)
        elif not camera_id:
            log.warning("No camera_id for uuid=%s — thumbnail skipped (visibility empty)", object_id)

        # Create alert first to get the unique alert_id
        alert = self._alerts.create_alert_payload(
            match=match,
            object_id=object_id,
            timestamp=timestamp,
            camera_id=display_camera,
            region_name=display_camera,
            confidence=confidence,
            center_of_mass=bounding_box,
            thumbnail_path="",  # placeholder — set below after storing
            mqtt_receive_time_ms=mqtt_receive_time_ms,
        )

        # Store the thumbnail keyed by alert_id (immutable — each alert gets
        # its own face crop, never overwritten by later detections).
        thumbnail_path = ""
        if thumbnail_b64 and self._event_repo:
            thumb_key = alert.alert_id
            self._event_repo.store_thumbnail(thumb_key, thumbnail_b64, ttl=3600)
            thumbnail_path = f"/api/v1/thumbnail/{thumb_key}"
            # Also store under object_id for backward compat (timeline, search)
            self._event_repo.store_thumbnail(object_id, thumbnail_b64, ttl=3600)
            log.info(
                "Thumbnail stored: alert=%s uuid=%s camera=%s face_crop=%s",
                alert.alert_id, object_id, display_camera, face_bbox is not None,
            )

        # Update alert payload with the per-alert thumbnail path
        alert.match["thumbnail_path"] = thumbnail_path

        # Update movement event with the matched poi_id and thumbnail
        self._events.store_movement(
            object_id=object_id,
            timestamp=timestamp,
            camera_id=display_camera,
            region=display_camera,
            poi_id=match.poi_id,
            thumbnail_path=thumbnail_path or None,
        )

        self._event_bus.publish("match_found", MatchFoundEvent(
            alert=alert,
            object_id=object_id,
            timestamp=timestamp,
            mqtt_receive_time_ms=mqtt_receive_time_ms,
        ))
