"""Tests for domain entities."""

from __future__ import annotations

import numpy as np

from backend.domain.entities.embedding import Embedding
from backend.domain.entities.event import MovementEvent, PersonEvent
from backend.domain.entities.match_result import AlertPayload, MatchResult
from backend.domain.entities.poi import POI, POIStatus, ReferenceImage, Severity


class TestPOI:
    def test_generate_id(self):
        pid = POI.generate_id()
        assert pid.startswith("poi-")
        assert len(pid) == 12  # "poi-" + 8 hex chars

    def test_to_dict(self):
        poi = POI(
            poi_id="poi-test1234",
            severity=Severity.HIGH,
            notes="suspicious",
            reference_images=[
                ReferenceImage(
                    source="uploaded_image",
                    embedding_id="emb-01",
                    vector_dim=256,
                    image_path="/uploads/test/ref_0.jpg",
                )
            ],
            status=POIStatus.ACTIVE,
        )
        d = poi.to_dict()
        assert d["event_type"] == "poi_enrollment"
        assert d["poi_id"] == "poi-test1234"
        assert d["severity"] == "high"
        assert d["status"] == "active"
        assert len(d["reference_images"]) == 1
        assert d["reference_images"][0]["embedding_id"] == "emb-01"
        assert "timestamp" in d

    def test_severity_enum(self):
        assert Severity.LOW.value == "low"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.HIGH.value == "high"

    def test_poi_status_enum(self):
        assert POIStatus.ACTIVE.value == "active"
        assert POIStatus.INACTIVE.value == "inactive"


class TestEmbedding:
    def test_dimension(self):
        v = np.random.randn(256).astype(np.float32)
        emb = Embedding(embedding_id="e1", vector=v)
        assert emb.dimension == 256

    def test_normalized_unit_length(self):
        v = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        emb = Embedding(embedding_id="e2", vector=v)
        normed = emb.normalized()
        assert abs(np.linalg.norm(normed) - 1.0) < 1e-6

    def test_normalized_zero_vector(self):
        v = np.zeros(256, dtype=np.float32)
        emb = Embedding(embedding_id="e3", vector=v)
        normed = emb.normalized()
        assert np.allclose(normed, v)


class TestMatchResult:
    def test_is_match_positive(self):
        m = MatchResult(poi_id="poi-1", similarity_score=0.85, faiss_distance=0.85)
        assert m.is_match

    def test_is_match_zero(self):
        m = MatchResult(poi_id="poi-1", similarity_score=0.0, faiss_distance=0.0)
        assert not m.is_match

    def test_is_match_negative(self):
        m = MatchResult(poi_id="poi-1", similarity_score=-0.1, faiss_distance=-0.1)
        assert not m.is_match


class TestAlertPayload:
    def test_to_dict(self):
        alert = AlertPayload(
            alert_id="alert-001",
            poi_id="poi-abc",
            severity="high",
            timestamp="2025-01-15T12:00:00Z",
            match={"camera_id": "cam1", "similarity_score": 0.9},
            poi_metadata={"notes": "test"},
        )
        d = alert.to_dict()
        assert d["event_type"] == "poi_match_alert"
        assert d["alert_id"] == "alert-001"
        assert d["severity"] == "high"
        assert d["status"] == "New"
        assert "dispatched_at" in d
        assert "mqtt_received_at" not in d

    def test_to_dict_with_mqtt_received_at(self):
        alert = AlertPayload(
            alert_id="alert-002",
            poi_id="poi-def",
            severity="medium",
            timestamp="2025-01-15T12:00:00Z",
            match={"camera_id": "cam2", "similarity_score": 0.85},
            poi_metadata={"notes": "test"},
            mqtt_received_at="2025-01-15T12:00:05.123+00:00",
        )
        d = alert.to_dict()
        assert d["mqtt_received_at"] == "2025-01-15T12:00:05.123+00:00"

    def test_to_dict_omits_empty_mqtt_received_at(self):
        alert = AlertPayload(
            alert_id="alert-003",
            poi_id="poi-ghi",
            severity="low",
            timestamp="2025-01-15T12:00:00Z",
            match={"camera_id": "cam3", "similarity_score": 0.7},
            poi_metadata={},
            mqtt_received_at="",
        )
        d = alert.to_dict()
        assert "mqtt_received_at" not in d


class TestMovementEvent:
    def test_to_dict(self):
        evt = MovementEvent(
            object_id="obj-1",
            timestamp="2025-01-15T12:00:00Z",
            camera_id="cam-01",
            region="aisle1",
            poi_id="poi-x",
        )
        d = evt.to_dict()
        assert d["object_id"] == "obj-1"
        assert d["poi_id"] == "poi-x"
        assert d["region"] == "aisle1"
