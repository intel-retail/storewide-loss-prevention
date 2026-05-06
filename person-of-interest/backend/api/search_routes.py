"""Search API routes — offline search via detection index."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

log = logging.getLogger("poi.api.search")

router = APIRouter(prefix="/search", tags=["Search"])

_embedding_factory = None
_detection_index = None
_event_repo = None


def init(embedding_factory, detection_index, event_repo) -> None:
    global _embedding_factory, _detection_index, _event_repo
    _embedding_factory = embedding_factory
    _detection_index = detection_index
    _event_repo = event_repo


@router.post("")
async def search_history(
    image: UploadFile = File(...),
    top_k: int = Form(20),
    start_time: str = Form(""),
    end_time: str = Form(""),
):
    """Search for a person across all historical detections by uploading an image.

    Queries the detection index (every face ever seen, 7-day retention).
    Returns appearances grouped by track ID, with entry/exit frames and
    zone dwell information where available.
    """
    if _detection_index is None:
        raise HTTPException(503, "Detection index not available")

    img_bytes = await image.read()
    if not img_bytes:
        raise HTTPException(400, "Image file is required")

    # ── Generate query embedding ──
    result = _embedding_factory.generate_from_bytes(img_bytes)
    if "error" in result:
        raise HTTPException(422, result["error"])

    query_vector = np.array(result["embedding"], dtype=np.float32)

    # ── Search detection index ──
    t0 = time.perf_counter()
    hits = _detection_index.search(query_vector, top_k=top_k)
    query_latency_ms = (time.perf_counter() - t0) * 1000

    if not hits:
        return _empty_response(start_time, end_time, query_latency_ms)

    # ── Group hits by track_id, keep best similarity per track ──
    # A single person may appear hundreds of times; we want one entry per track.
    best_per_track: dict[str, dict] = {}
    for faiss_id, similarity in hits:
        meta = _detection_index.get_metadata(faiss_id)
        if meta is None:
            continue
        ts = meta.get("timestamp", "")
        if start_time and ts and ts < start_time:
            continue
        if end_time and ts and ts > end_time:
            continue
        track_id = meta["track_id"]
        if track_id not in best_per_track or similarity > best_per_track[track_id]["similarity"]:
            best_per_track[track_id] = {
                "track_id": track_id,
                "camera_id": meta.get("camera_id", ""),
                "best_timestamp": ts,
                "similarity": round(float(similarity), 4),
                "bbox": meta.get("bbox"),
            }

    if not best_per_track:
        return _empty_response(start_time, end_time, query_latency_ms)

    # ── Enrich each track with frames and zone history ──
    appearances = []
    for track_id, track in best_per_track.items():
        appearance = _build_appearance(track_id, track)
        appearances.append(appearance)

    # Sort by similarity descending
    appearances.sort(key=lambda a: a["similarity"], reverse=True)

    return {
        "event_type": "offline_search_result",
        "query_range": {"start": start_time, "end": end_time},
        "query_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_appearances": len(appearances),
        "appearances": appearances,
        "search_stats": {
            "vectors_searched": _detection_index.total_vectors(),
            "raw_hits": len(hits),
            "unique_tracks": len(best_per_track),
            "query_latency_ms": round(query_latency_ms, 2),
        },
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_appearance(track_id: str, track: dict) -> dict:
    """Build the appearance dict for one track: frames + zone dwells."""
    camera_id = track["camera_id"]

    # ── Frame URLs ──
    entry_frame_url = None
    last_seen_frame_url = None

    if _event_repo is not None:
        if _event_repo.track_frame_exists(track_id, "entry"):
            key = _event_repo.get_track_frame_key(track_id, "entry")
            entry_frame_url = f"/api/v1/frames/{_encode_key(key)}"
        if _event_repo.track_frame_exists(track_id, "last_seen"):
            key = _event_repo.get_track_frame_key(track_id, "last_seen")
            last_seen_frame_url = f"/api/v1/frames/{_encode_key(key)}"

    # ── Zone dwells (available when zones are configured) ──
    zone_appearances = []
    if _event_repo is not None:
        dwells = _event_repo.get_region_dwells_for_object(track_id)
        for dwell in dwells:
            zone_entry: dict = {
                "zone": dwell.get("region_name") or dwell.get("region_id", ""),
                "scene_id": dwell.get("scene_id", ""),
                "entry_time": dwell.get("entry_time", ""),
                "exit_time": dwell.get("exit_time", ""),
                "dwell_seconds": dwell.get("dwell_sec"),
            }
            # Attach zone entry/exit frame URLs if they exist
            entry_fk = dwell.get("entry_frame_key", "")
            exit_fk = dwell.get("exit_frame_key", "")
            if entry_fk:
                zone_entry["entry_frame_url"] = f"/api/v1/frames/{_encode_key(entry_fk)}"
            if exit_fk:
                zone_entry["exit_frame_url"] = f"/api/v1/frames/{_encode_key(exit_fk)}"
            zone_appearances.append(zone_entry)

        zone_appearances.sort(key=lambda z: z.get("entry_time") or "")

    return {
        "track_id": track_id,
        "camera_id": camera_id,
        "similarity": track["similarity"],
        "best_match_time": track["best_timestamp"],
        "entry_frame_url": entry_frame_url,
        "last_seen_frame_url": last_seen_frame_url,
        "zone_appearances": zone_appearances,
    }


def _encode_key(redis_key: str) -> str:
    """URL-safe encode a Redis key for use in a path segment."""
    import base64
    return base64.urlsafe_b64encode(redis_key.encode()).decode().rstrip("=")


def _empty_response(start_time: str, end_time: str, latency_ms: float) -> dict:
    return {
        "event_type": "offline_search_result",
        "query_range": {"start": start_time, "end": end_time},
        "query_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_appearances": 0,
        "appearances": [],
        "search_stats": {
            "vectors_searched": _detection_index.total_vectors() if _detection_index else 0,
            "raw_hits": 0,
            "unique_tracks": 0,
            "query_latency_ms": round(latency_ms, 2),
        },
    }
