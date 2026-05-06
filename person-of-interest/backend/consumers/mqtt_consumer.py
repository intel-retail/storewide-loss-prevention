"""MQTT event consumer — processes SceneScape scene events for POI face matching.

Primary topic: scenescape/data/camera/{camera_id}
  Payload: per-camera DLStreamer output with person detections and face sub_objects.
  Person objects contain body-reid embeddings; face sub_objects (when present) contain
  face-reid embeddings from face-reidentification-retail-0095 — the SAME model used
  during POI enrollment.  Only face sub_object embeddings are used for FAISS matching.

  Dedup key: f"cam:{camera_id}:{person_int_id}" with object_cache_ttl (default 60s).
  This camera-local track key is stable for the lifetime of the GStreamer tracker track
  and gives reliable per-person dedup within a camera view without depending on the
  external topic's global UUID.

Secondary topic (monitoring only): scenescape/external/{scene_id}/person
  Carries global UUIDs, reid_state, and body-reid embeddings (person-reidentification-
  retail-0277).  Body embeddings are a DIFFERENT embedding space from the face model
  and must NOT be used for FAISS comparison against face-enrolled POIs.
  External topic is subscribed to solely for reid:meta observability (MCP tools) and
  movement event recording.  It no longer controls a gate on FAISS execution — face
  detections on the camera topic always proceed to FAISS regardless of reid_state.

Embedding space alignment:
  Enrollment (EmbeddingModelFactory):  face-reidentification-retail-0095, 256-dim,
                                       with landmark alignment preprocessing.
  Runtime (camera topic face objects): face-reidentification-retail-0095, 256-dim,
                                       simple resize preprocessing (no landmark align).
  Both are in the same base model embedding space → FAISS cosine similarity is valid.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import struct
from typing import List, Optional

from backend.observer.events import EventBus, MatchFoundEvent
from backend.service.alert_service import AlertService
from backend.service.event_service import EventService
from backend.service.matching_service import MatchingService
from backend.utils.thumbnail import submit_capture

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

        camera_id = m.group("camera_id")
        timestamp = payload.get("timestamp", "")

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

            log.info(
                "Face embedding found: camera=%s person=%s conf=%.3f dim=%d",
                camera_id, person_int_id, best_face_conf, len(embedding_vector),
            )

            # ── Detection index: store one embedding per unique track (not per frame) ──
            object_id = f"cam:{camera_id}:{person_int_id}"
            if self._detection_index is not None:
                import numpy as _np
                try:
                    # claim_track() uses Redis NX — returns True only the FIRST time
                    # this track_id is seen. Subsequent detections of the same person
                    # (same tracker track) are skipped — no duplicate embeddings.
                    if self._detection_index.claim_track(object_id):
                        self._detection_index.add(
                            vector=_np.array(embedding_vector, dtype=_np.float32),
                            camera_id=camera_id,
                            track_id=object_id,
                            timestamp=timestamp,
                            bbox=best_face_bbox,
                        )
                        log.info("DetectionIndex: new track stored %s", object_id)
                    else:
                        log.debug("DetectionIndex: track already stored, skipping %s", object_id)
                except Exception:
                    log.debug("DetectionIndex.add failed for %s", object_id, exc_info=True)

            # Use a stable camera-local track key for dedup.
            # MatchingService cache-aside handles alert suppression (object_cache_ttl).
            # The external topic global UUID is NOT used here — per-camera tracking
            # is both correct and avoids the multi-person / timing-race problems of
            # a camera→UUID map.
            # NOTE: object_id is already set above for the detection index tap.

            # For thumbnail: use person bbox (wider) rather than face bbox to tolerate
            # timing drift between MQTT detection frame and RTSP grabber frame.
            # Face bbox is only ~30-50px and any movement causes a miss-crop.
            person_bbox = obj.get("bounding_box_px") or best_face_bbox

            # ── Track-level entry / last-seen frames (no-zone fallback) ──────
            # Entry:     captured once per track lifetime via atomic Redis NX claim.
            # Last-seen: always overwritten; acts as the exit frame when track stops.
            if self._event_repo is not None:
                self._capture_track_frames(object_id, camera_id, person_bbox, timestamp)

            self._run_matching(
                object_id=object_id,
                embedding_vector=embedding_vector,
                timestamp=timestamp,
                camera_id=camera_id,
                confidence=best_face_conf,
                bounding_box=person_bbox,
            )

    # ── Secondary: external topic (monitoring / UUID tracking only) ──────────

    def _capture_track_frames(self, object_id: str, camera_id: str, bbox, timestamp: str = "") -> None:
        """Capture entry frame (first detection) and update last-seen frame.

        Uses the detection timestamp to retrieve the buffered MQTT image frame
        whose pipeline timestamp is closest to the detection — this guarantees
        the stored frame shows the same scene as the detection event, not a
        later frame where the person may have moved.
        """
        try:
            if self._event_repo.claim_track_entry(object_id):
                future = submit_capture(camera_id, None, timestamp)
                b64 = future.result(timeout=4)
                if b64:
                    self._event_repo.store_track_frame(object_id, "entry", b64)
                    log.info("Track entry frame captured: %s", object_id)
        except Exception:
            log.debug("Track entry frame failed for %s", object_id, exc_info=True)

        try:
            future = submit_capture(camera_id, None, timestamp)
            b64 = future.result(timeout=2)
            if b64:
                self._event_repo.store_track_frame(object_id, "last_seen", b64)
        except Exception:
            log.debug("Track last-seen frame failed for %s", object_id, exc_info=True)

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

        for obj in persons:
            object_id = obj.get("id")
            if not object_id:
                continue
            reid_state = obj.get("reid_state", "")
            visibility = obj.get("visibility", [])
            camera_id = visibility[0] if visibility else scene_name

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

    # ── Shared matching + alerting ──────────────────────────────────────────

    def _run_matching(
        self,
        object_id: str,
        embedding_vector: List[float],
        timestamp: str,
        camera_id: Optional[str],
        confidence: float,
        bounding_box,
    ) -> None:
        """Run FAISS lookup and emit alerts for a confirmed detection."""
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
                import json as _json
                _raw = self._event_repo._r.get(f"reid:meta:{object_id}")  # type: ignore[attr-defined]
                if _raw:
                    reid_meta_raw = _json.loads(_raw)
            except Exception:
                pass
            if reid_meta_raw:
                match_meta["reid"] = reid_meta_raw
            self._event_repo.set_match_metadata(object_id, match_meta, ttl=3600)
            log.debug("Match metadata written to Redis for uuid=%s", object_id)

        # Capture thumbnail from RTSP — only when we have a valid camera_id
        thumbnail_path = ""
        if camera_id and self._event_repo and self._event_repo.claim_thumbnail(object_id, ttl=30):
            future = submit_capture(camera_id, bounding_box)
            try:
                b64 = future.result(timeout=6)
                if b64:
                    self._event_repo.store_thumbnail(object_id, b64, ttl=3600)
                    thumbnail_path = f"/api/v1/thumbnail/{object_id}"
                    log.info("Thumbnail captured for uuid=%s camera=%s", object_id, camera_id)
                else:
                    log.warning("Thumbnail returned no data for uuid=%s camera=%s", object_id, camera_id)
            except Exception:
                log.warning("Thumbnail timed out or failed for uuid=%s camera=%s", object_id, camera_id)
        elif not camera_id:
            log.warning("No camera_id for uuid=%s — thumbnail skipped (visibility empty)", object_id)
        elif self._event_repo and self._event_repo.get_thumbnail(object_id):
            thumbnail_path = f"/api/v1/thumbnail/{object_id}"

        alert = self._alerts.create_alert_payload(
            match=match,
            object_id=object_id,
            timestamp=timestamp,
            camera_id=display_camera,
            region_name=display_camera,
            confidence=confidence,
            center_of_mass=bounding_box,
            thumbnail_path=thumbnail_path,
        )

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
        ))
