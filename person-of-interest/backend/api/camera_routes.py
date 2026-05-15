"""Camera API routes — returns cameras with stream URLs.

Sources (in priority order):
  1. SceneScape REST API (if configured)
  2. RTSP_PREWARM_CAMERAS env var (always available)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend.core.config import get_config

log = logging.getLogger("poi.api.camera")

router = APIRouter(prefix="/cameras", tags=["Cameras"])

_scenescape_adapter = None


def init(scenescape_adapter) -> None:
    global _scenescape_adapter
    _scenescape_adapter = scenescape_adapter


def _cameras_from_config() -> list[dict]:
    """Build camera list from RTSP_PREWARM_CAMERAS env var."""
    cfg = get_config()
    raw = cfg.camera_streams
    if not raw:
        return []
    cameras = []
    for cam_id in raw.split(","):
        cam_id = cam_id.strip()
        if cam_id:
            stream_path = cfg.camera_stream_map.get(cam_id, cam_id)
            cameras.append({
                "camera_id": cam_id,
                "name": cam_id.replace("_", " ").replace("-", " ").title(),
                "stream_path": stream_path,
                "status": "active",
            })
    return cameras


@router.get("")
def list_cameras():
    """List all cameras with stream metadata.

    Returns cameras from SceneScape API when available,
    otherwise falls back to configured camera list.
    Each camera includes a ``stream_path`` for building
    the MediaMTX WebRTC player URL on the client side.

    Declared as sync ``def`` so FastAPI runs the synchronous
    SceneScape adapter calls in its threadpool executor.
    """
    cfg = get_config()

    # Try SceneScape API first
    cameras: list[dict] = []
    if _scenescape_adapter:
        cameras = _scenescape_adapter.list_cameras()

    # Fallback to configured camera list
    if not cameras:
        cameras = _cameras_from_config()

    # Enrich each camera with stream_path if missing
    for cam in cameras:
        if "stream_path" not in cam:
            cam["stream_path"] = cam.get("camera_id", cam.get("uid", ""))
        if "name" not in cam:
            cam["name"] = cam.get("camera_id", "Unknown")

    return {
        "cameras": cameras,
        "count": len(cameras),
        "mediamtx_webrtc_port": cfg.mediamtx_webrtc_port,
    }


@router.get("/{camera_id}")
def get_camera(camera_id: str):
    """Get a single camera from SceneScape.

    Sync handler — adapter performs blocking I/O.
    """
    camera = _scenescape_adapter.get_camera(camera_id) if _scenescape_adapter else None
    if camera is None:
        # Check config fallback
        for cam in _cameras_from_config():
            if cam["camera_id"] == camera_id:
                return cam
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    return camera
