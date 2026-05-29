"""Tests for MQTT EventConsumer.

Primary topic: scenescape/data/camera/{camera_id}  — face embeddings → FAISS
Secondary topic: scenescape/external/{scene_id}/person — monitoring only, no FAISS
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.consumers.mqtt_consumer import (
    CAMERA_TOPIC_RE,
    EXTERNAL_TOPIC_RE,
    FACE_CONFIDENCE_THRESHOLD,
    REID_MATCHED_STATES,
    EventConsumer,
    _parse_embedding,
)
from backend.domain.entities.match_result import AlertPayload, MatchResult
from backend.observer.events import EventBus

# Camera topic (primary — FAISS matching via face embeddings)
_CAM_TOPIC = "scenescape/data/camera/Camera_01"
# External topic (monitoring only — body embeddings, no FAISS)
_EXT_TOPIC = "scenescape/external/scene-1/person"


def _make_face_embedding_b64():
    """Return a 256-d float32 embedding as a list (pre-parsed)."""
    return np.random.randn(256).tolist()


class TestEventConsumer:
    def _make_consumer(self, match_result=None):
        matching = MagicMock()
        matching.match_object.return_value = match_result
        matching._cfg.faiss_dimension = 256

        events = MagicMock()

        alert_service = MagicMock()
        alert_service.create_alert_payload.return_value = AlertPayload(
            alert_id="alert-test",
            poi_id="poi-match",
            severity="high",
            timestamp="t",
            match={},
            poi_metadata={},
        )

        bus = EventBus()
        event_repo = MagicMock()
        event_repo.claim_thumbnail.return_value = False
        event_repo.get_thumbnail.return_value = None
        event_repo.get_uuid_for_camera_bbox.return_value = None

        consumer = EventConsumer(matching, events, alert_service, bus, event_repo=event_repo)
        return consumer, matching, events, alert_service, bus

    def _make_camera_payload(self, person_id=42, face_conf=0.95, face_embedding=None):
        """Build a camera-topic payload with face sub_objects."""
        if face_embedding is None:
            face_embedding = _make_face_embedding_b64()
        return {
            "timestamp": "2025-01-15T12:30:00.000Z",
            "objects": {
                "person": [{
                    "id": person_id,
                    "confidence": 0.92,
                    "bounding_box_px": {"x": 10, "y": 20, "width": 100, "height": 200},
                    "sub_objects": {
                        "face": [{
                            "confidence": face_conf,
                            "bounding_box_px": {"x": 15, "y": 22, "width": 40, "height": 50},
                            "metadata": {
                                "reid": {
                                    "embedding_vector": face_embedding,
                                }
                            },
                        }]
                    },
                }]
            },
        }

    def _make_external_payload(self, object_id="uuid-001", reid_state=None):
        """Build an external-topic payload (body-reid, no face)."""
        obj = {
            "id": object_id,
            "type": "person",
            "visibility": ["Camera_01"],
        }
        if reid_state is not None:
            obj["reid_state"] = reid_state
        return {
            "timestamp": "2025-01-15T12:30:00.000Z",
            "name": "scene-1",
            "objects": [obj],
        }

    # ── Basic routing ───────────────────────────────────────────────────────

    def test_ignores_non_matching_topic(self):
        consumer, matching, _, _, _ = self._make_consumer()
        consumer.handle_event("some/other/topic", {})
        matching.match_object.assert_not_called()

    def test_camera_topic_triggers_faiss(self):
        """Camera topic is primary — face embeddings should go to FAISS."""
        match = MatchResult(poi_id="poi-match", similarity_score=0.9, faiss_distance=0.9)
        consumer, matching, _, _, _ = self._make_consumer(match_result=match)
        consumer.handle_event(_CAM_TOPIC, self._make_camera_payload())
        matching.match_object.assert_called_once()

    def test_external_topic_does_not_trigger_faiss(self):
        """External topic is monitoring only — no FAISS matching."""
        consumer, matching, events, _, _ = self._make_consumer()
        consumer.handle_event(_EXT_TOPIC, self._make_external_payload())
        matching.match_object.assert_not_called()
        # But movement is still stored for timeline
        events.store_movement.assert_called_once()

    # ── Camera topic person filtering ───────────────────────────────────────

    def test_skips_persons_without_face_sub_objects(self):
        """Persons with no face sub_objects are skipped for FAISS."""
        consumer, matching, _, _, _ = self._make_consumer()
        payload = self._make_camera_payload()
        # Remove face sub_objects
        payload["objects"]["person"][0]["sub_objects"] = {}
        consumer.handle_event(_CAM_TOPIC, payload)
        matching.match_object.assert_not_called()

    def test_skips_low_confidence_faces(self):
        """Faces below FACE_CONFIDENCE_THRESHOLD are skipped."""
        consumer, matching, _, _, _ = self._make_consumer()
        payload = self._make_camera_payload(face_conf=0.5)
        consumer.handle_event(_CAM_TOPIC, payload)
        matching.match_object.assert_not_called()

    def test_uses_highest_confidence_face(self):
        """When multiple face sub_objects exist, the highest-confidence one is used."""
        consumer, matching, _, _, _ = self._make_consumer()
        emb_low = np.random.randn(256).tolist()
        emb_high = np.random.randn(256).tolist()
        payload = self._make_camera_payload()
        payload["objects"]["person"][0]["sub_objects"]["face"] = [
            {
                "confidence": 0.85,
                "metadata": {"reid": {"embedding_vector": emb_low}},
            },
            {
                "confidence": 0.98,
                "metadata": {"reid": {"embedding_vector": emb_high}},
            },
        ]
        consumer.handle_event(_CAM_TOPIC, payload)
        matching.match_object.assert_called_once()
        # The embedding passed should be from the higher-confidence face
        call_embedding = matching.match_object.call_args[0][1]
        assert call_embedding == [float(x) for x in emb_high]

    # ── Dedup key format ────────────────────────────────────────────────────

    def test_object_id_is_cam_camera_person_format(self):
        """object_id for camera topic should be f'cam:{camera_id}:{person_id}'."""
        match = MatchResult(poi_id="poi-m", similarity_score=0.9, faiss_distance=0.9)
        consumer, matching, _, _, _ = self._make_consumer(match_result=match)
        consumer.handle_event(_CAM_TOPIC, self._make_camera_payload(person_id=7))
        call_args = matching.match_object.call_args[0]
        assert call_args[0] == "cam:Camera_01:7"

    def test_dedup_same_person_id_in_one_message(self):
        """Same person_id appearing twice in one message should only be processed once."""
        consumer, matching, _, _, _ = self._make_consumer()
        payload = self._make_camera_payload(person_id=42)
        # Duplicate the person entry
        payload["objects"]["person"].append(payload["objects"]["person"][0])
        consumer.handle_event(_CAM_TOPIC, payload)
        matching.match_object.assert_called_once()

    # ── External topic monitoring ───────────────────────────────────────────

    def test_external_topic_stores_movement(self):
        """External topic stores movement events for timeline tracking."""
        consumer, _, events, _, _ = self._make_consumer()
        consumer.handle_event(_EXT_TOPIC, self._make_external_payload(object_id="global-uuid"))
        events.store_movement.assert_called_once()
        call_kwargs = events.store_movement.call_args[1]
        assert call_kwargs["object_id"] == "global-uuid"
        assert call_kwargs["camera_id"] == "Camera_01"

    def test_external_topic_uses_visibility_camera(self):
        """External topic uses visibility[0] as camera_id."""
        consumer, _, events, _, _ = self._make_consumer()
        payload = self._make_external_payload()
        payload["objects"][0]["visibility"] = ["Camera_05"]
        consumer.handle_event(_EXT_TOPIC, payload)
        call_kwargs = events.store_movement.call_args[1]
        assert call_kwargs["camera_id"] == "Camera_05"

    def test_external_topic_falls_back_to_scene_name(self):
        """When visibility is empty, falls back to scene name."""
        consumer, _, events, _, _ = self._make_consumer()
        payload = self._make_external_payload()
        payload["objects"][0]["visibility"] = []
        payload["name"] = "store-scene"
        consumer.handle_event(_EXT_TOPIC, payload)
        call_kwargs = events.store_movement.call_args[1]
        assert call_kwargs["camera_id"] == "store-scene"

    # ── Match + alert flow ──────────────────────────────────────────────────

    def test_no_match_skips_alert(self):
        consumer, matching, events, alert_service, _ = self._make_consumer(match_result=None)
        consumer.handle_event(_CAM_TOPIC, self._make_camera_payload())
        matching.match_object.assert_called_once()
        alert_service.create_alert_payload.assert_not_called()
        # Only pre-match movement stored
        events.store_movement.assert_called_once()

    def test_match_emits_alert_and_stores_twice(self):
        match = MatchResult(poi_id="poi-match", similarity_score=0.9, faiss_distance=0.9)
        consumer, matching, events, alert_service, bus = self._make_consumer(match_result=match)

        consumer.handle_event(_CAM_TOPIC, self._make_camera_payload())

        matching.match_object.assert_called_once()
        alert_service.create_alert_payload.assert_called_once()
        # Two store_movement calls: pre-match + post-match with poi_id
        assert events.store_movement.call_count == 2

    def test_publishes_match_found_event(self):
        match = MatchResult(poi_id="poi-m", similarity_score=0.85, faiss_distance=0.85)
        consumer, _, _, _, bus = self._make_consumer(match_result=match)
        observer_cb = MagicMock()
        bus.subscribe("match_found", observer_cb)

        consumer.handle_event(_CAM_TOPIC, self._make_camera_payload(person_id=99))

        observer_cb.assert_called_once()
        event = observer_cb.call_args[0][0]
        assert event.object_id == "cam:Camera_01:99"

    # ── Objects as list format ──────────────────────────────────────────────

    def test_camera_topic_handles_objects_as_list(self):
        """Camera topic should also handle objects as a list (not just dict)."""
        consumer, matching, _, _, _ = self._make_consumer()
        emb = _make_face_embedding_b64()
        payload = {
            "timestamp": "t",
            "objects": [{
                "id": 1,
                "category": "person",
                "type": "person",
                "sub_objects": {
                    "face": [{
                        "confidence": 0.95,
                        "metadata": {"reid": {"embedding_vector": emb}},
                    }]
                },
            }],
        }
        consumer.handle_event(_CAM_TOPIC, payload)
        matching.match_object.assert_called_once()


# ── _parse_embedding unit tests ─────────────────────────────────────────────

class TestParseEmbedding:
    def test_none_returns_none(self):
        assert _parse_embedding(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_embedding("") is None

    def test_flat_list(self):
        data = list(range(256))
        result = _parse_embedding(data)
        assert result == [float(x) for x in data]

    def test_nested_list(self):
        data = [list(range(256))]
        result = _parse_embedding(data)
        assert len(result) == 256
        assert result[0] == 0.0

    def test_json_string_flat(self):
        data = list(range(256))
        result = _parse_embedding(json.dumps(data))
        assert len(result) == 256

    def test_json_string_nested(self):
        data = [list(range(256))]
        result = _parse_embedding(json.dumps(data))
        assert len(result) == 256
