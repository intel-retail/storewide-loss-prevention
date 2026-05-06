"""Thumbnail API — serves RTSP-captured face crops stored in Redis."""

from __future__ import annotations

import base64
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

log = logging.getLogger("poi.api.thumbnail")

router = APIRouter()
_event_repo = None


def init(event_repo) -> None:
    global _event_repo
    _event_repo = event_repo


@router.get("/thumbnail/{object_id}", response_class=Response)
def get_thumbnail(object_id: str):
    """Return the captured face crop for a tracked person as JPEG."""
    if _event_repo is None:
        raise HTTPException(status_code=503, detail="Thumbnail service not ready")

    b64 = _event_repo.get_thumbnail(object_id)
    if not b64:
        raise HTTPException(status_code=404, detail="Thumbnail not found or expired")

    try:
        image_bytes = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=500, detail="Thumbnail data corrupt")

    return Response(content=image_bytes, media_type="image/jpeg")


@router.get("/frames/{encoded_key}", response_class=Response)
def get_frame(encoded_key: str):
    """Serve a stored zone or track frame by its URL-safe base64-encoded Redis key.

    Frame keys are returned by POST /api/v1/search in the entry_frame_url,
    last_seen_frame_url, and zone_appearances[].entry/exit_frame_url fields.
    """
    if _event_repo is None:
        raise HTTPException(status_code=503, detail="Frame service not ready")

    # Decode the URL-safe base64 key (re-add stripped padding)
    try:
        padding = 4 - len(encoded_key) % 4
        redis_key = base64.urlsafe_b64decode(encoded_key + "=" * (padding % 4)).decode()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid frame key encoding")

    # Support both zone frames and track frames
    if redis_key.startswith("zone:frame:"):
        b64 = _event_repo.get_zone_frame(redis_key)
    elif redis_key.startswith("track:frame:"):
        raw = _event_repo._r.get(redis_key)
        b64 = raw.decode() if isinstance(raw, bytes) else raw
    else:
        raise HTTPException(status_code=400, detail="Unknown frame key type")

    if not b64:
        raise HTTPException(status_code=404, detail="Frame not found or expired")

    try:
        image_bytes = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=500, detail="Frame data corrupt")

    return Response(content=image_bytes, media_type="image/jpeg")
