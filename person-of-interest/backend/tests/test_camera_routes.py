"""Tests for camera API routes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import camera_routes


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(camera_routes.router, prefix="/api/v1")
    return _app


@pytest.fixture(autouse=True)
def reset_adapter():
    camera_routes._scenescape_adapter = None
    yield
    camera_routes._scenescape_adapter = None


@pytest.fixture
def client(app):
    return TestClient(app)


class TestListCameras:
    @patch("backend.api.camera_routes.get_config")
    def test_returns_cameras_from_config(self, mock_cfg, client):
        cfg = MagicMock()
        cfg.camera_streams = "Camera_01,Camera_02"
        cfg.camera_stream_map = {"Camera_01": "retail-cam1"}
        cfg.mediamtx_webrtc_port = 8889
        mock_cfg.return_value = cfg

        resp = client.get("/api/v1/cameras")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert data["mediamtx_webrtc_port"] == 8889
        # Camera_01 uses mapped stream path
        assert data["cameras"][0]["stream_path"] == "retail-cam1"
        # Camera_02 falls back to camera_id
        assert data["cameras"][1]["stream_path"] == "Camera_02"

    @patch("backend.api.camera_routes.get_config")
    def test_scenescape_adapter_takes_priority(self, mock_cfg, client):
        cfg = MagicMock()
        cfg.camera_streams = "Camera_01"
        cfg.camera_stream_map = {}
        cfg.mediamtx_webrtc_port = 9000
        mock_cfg.return_value = cfg

        adapter = MagicMock()
        adapter.list_cameras.return_value = [
            {"camera_id": "ss-cam", "name": "SS Cam", "stream_path": "ss-1"},
        ]
        camera_routes.init(adapter)

        resp = client.get("/api/v1/cameras")
        data = resp.json()
        assert data["count"] == 1
        assert data["cameras"][0]["camera_id"] == "ss-cam"

    @patch("backend.api.camera_routes.get_config")
    def test_falls_back_when_adapter_returns_empty(self, mock_cfg, client):
        cfg = MagicMock()
        cfg.camera_streams = "Fallback_01"
        cfg.camera_stream_map = {}
        cfg.mediamtx_webrtc_port = 8889
        mock_cfg.return_value = cfg

        adapter = MagicMock()
        adapter.list_cameras.return_value = []
        camera_routes.init(adapter)

        resp = client.get("/api/v1/cameras")
        data = resp.json()
        assert data["count"] == 1
        assert data["cameras"][0]["camera_id"] == "Fallback_01"

    @patch("backend.api.camera_routes.get_config")
    def test_empty_config_returns_no_cameras(self, mock_cfg, client):
        cfg = MagicMock()
        cfg.camera_streams = ""
        cfg.camera_stream_map = {}
        cfg.mediamtx_webrtc_port = 8889
        mock_cfg.return_value = cfg

        resp = client.get("/api/v1/cameras")
        data = resp.json()
        assert data["count"] == 0
        assert data["cameras"] == []


class TestGetCamera:
    @patch("backend.api.camera_routes.get_config")
    def test_found_in_config(self, mock_cfg, client):
        cfg = MagicMock()
        cfg.camera_streams = "Camera_01"
        cfg.camera_stream_map = {}
        mock_cfg.return_value = cfg

        resp = client.get("/api/v1/cameras/Camera_01")
        assert resp.status_code == 200
        assert resp.json()["camera_id"] == "Camera_01"

    @patch("backend.api.camera_routes.get_config")
    def test_not_found_returns_404(self, mock_cfg, client):
        cfg = MagicMock()
        cfg.camera_streams = "Camera_01"
        cfg.camera_stream_map = {}
        mock_cfg.return_value = cfg

        resp = client.get("/api/v1/cameras/NoSuchCam")
        assert resp.status_code == 404

    def test_found_via_adapter(self, client):
        adapter = MagicMock()
        adapter.get_camera.return_value = {"camera_id": "ss-1", "name": "SS One"}
        camera_routes.init(adapter)

        resp = client.get("/api/v1/cameras/ss-1")
        assert resp.status_code == 200
        assert resp.json()["camera_id"] == "ss-1"
