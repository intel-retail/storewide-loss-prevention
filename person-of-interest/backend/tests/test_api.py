"""Tests for FastAPI endpoints (POI routes)."""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import poi_routes

# Minimal valid JPEG header (SOI + EOI markers) for upload validation tests
_JPEG_STUB = b"\xff\xd8\xff\xe0" + b"\x00" * 20


@pytest.fixture
def app():
    """Create a minimal FastAPI app with POI routes."""
    _app = FastAPI()
    _app.include_router(poi_routes.router, prefix="/api/v1")
    return _app


@pytest.fixture
def mock_poi_service():
    service = MagicMock()
    poi_routes.init(service)
    yield service
    poi_routes._poi_service = None


@pytest.fixture
def client(app, mock_poi_service):
    return TestClient(app)


class TestPOIRoutes:
    def test_list_pois(self, client, mock_poi_service):
        mock_poi_service.list_pois.return_value = [
            {"poi_id": "poi-1", "severity": "high"},
            {"poi_id": "poi-2", "severity": "low"},
        ]
        resp = client.get("/api/v1/poi")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["poi_id"] == "poi-1"

    def test_get_poi_found(self, client, mock_poi_service):
        mock_poi_service.get_poi.return_value = {"poi_id": "poi-x", "severity": "medium"}

        resp = client.get("/api/v1/poi/poi-x")
        assert resp.status_code == 200
        assert resp.json()["poi_id"] == "poi-x"

    def test_get_poi_not_found(self, client, mock_poi_service):
        mock_poi_service.get_poi.return_value = None

        resp = client.get("/api/v1/poi/poi-missing")
        assert resp.status_code == 404

    def test_delete_poi_success(self, client, mock_poi_service):
        mock_poi_service.delete_poi.return_value = True

        resp = client.delete("/api/v1/poi/poi-del")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_delete_poi_not_found(self, client, mock_poi_service):
        mock_poi_service.delete_poi.return_value = False

        resp = client.delete("/api/v1/poi/poi-nope")
        assert resp.status_code == 404

    def test_create_poi_success(self, client, mock_poi_service):
        mock_poi_service.create_poi = AsyncMock(
            return_value={"poi_id": "poi-new", "event_type": "poi_enrollment"}
        )
        # Create a fake image file
        image = io.BytesIO(b"\xff\xd8\xff\xe0fake-jpg-data")
        image.name = "test.jpg"

        resp = client.post(
            "/api/v1/poi",
            files=[("images", ("test.jpg", image, "image/jpeg"))],
            data={"severity": "high", "description": "test"},
        )
        assert resp.status_code == 201
        assert resp.json()["poi_id"] == "poi-new"

    def test_create_poi_no_images(self, client, mock_poi_service):
        resp = client.post(
            "/api/v1/poi",
            data={"severity": "medium", "description": ""},
        )
        # FastAPI returns 422 for missing required file parameter
        assert resp.status_code == 422

    def test_create_poi_face_detection_failure(self, client, mock_poi_service):
        mock_poi_service.create_poi = AsyncMock(
            return_value={"error": "No faces detected in any uploaded image"}
        )
        image = io.BytesIO(_JPEG_STUB)

        resp = client.post(
            "/api/v1/poi",
            files=[("images", ("test.jpg", image, "image/jpeg"))],
            data={"severity": "medium", "description": ""},
        )
        assert resp.status_code == 422

    def test_create_poi_too_many_images(self, client, mock_poi_service):
        images = []
        for i in range(6):
            img = io.BytesIO(_JPEG_STUB)
            images.append(("images", (f"img_{i}.jpg", img, "image/jpeg")))

        resp = client.post(
            "/api/v1/poi",
            files=images,
            data={"severity": "medium", "description": ""},
        )
        assert resp.status_code == 400

    def test_create_poi_invalid_file_header(self, client, mock_poi_service):
        """Upload with valid MIME but invalid magic bytes should be rejected."""
        image = io.BytesIO(b"not-an-image-at-all")
        resp = client.post(
            "/api/v1/poi",
            files=[("images", ("test.jpg", image, "image/jpeg"))],
            data={"severity": "medium", "description": ""},
        )
        assert resp.status_code == 400
        assert "invalid file header" in resp.json()["detail"]
