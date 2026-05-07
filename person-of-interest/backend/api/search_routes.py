"""Search API routes — offline search via detection index."""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.core.config import get_config

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

    # ── Collect best entry hit per track (above search similarity threshold) ──
    cfg = get_config()
    threshold = cfg.search_similarity_threshold
    best_entry: dict[str, dict] = {}  # track_id → {faiss_id, similarity, meta}
    for faiss_id, similarity in hits:
        if similarity < threshold:
            continue
        meta = _detection_index.get_metadata(faiss_id)
        if meta is None:
            continue
        ts = meta.get("timestamp", "")
        if start_time and ts and ts < start_time:
            continue
        if end_time and ts and ts > end_time:
            continue
        track_id = meta["track_id"]
        if track_id not in best_entry or similarity > best_entry[track_id]["similarity"]:
            best_entry[track_id] = {"faiss_id": faiss_id, "similarity": similarity, "meta": meta}

    if not best_entry:
        return _empty_response(start_time, end_time, query_latency_ms)

    # ── Check rolling exit vectors for the same tracks ──
    exit_sims = _detection_index.search_exits(query_vector, list(best_entry.keys()))

    # ── Build one grouped appearance per track (entry + exit on same card) ──
    appearances = []
    for track_id, entry in best_entry.items():
        exit_sim = exit_sims.get(track_id)
        appearance = _build_grouped_appearance(
            entry["faiss_id"], entry["similarity"], entry["meta"],
            exit_sim=exit_sim, track_id=track_id,
        )
        appearances.append(appearance)

    # Sort by best similarity (max of entry and exit) descending
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
            "unique_tracks": len(appearances),
            "query_latency_ms": round(query_latency_ms, 2),
        },
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_grouped_appearance(
    faiss_id: int,
    entry_sim: float,
    meta: dict,
    exit_sim: Optional[float],
    track_id: str,
) -> dict:
    """Build one appearance card grouping entry and exit for the same track."""
    camera_id = meta.get("camera_id", "")

    # ── Entry frame ──
    entry_frame_url = None
    if _detection_index is not None and _detection_index.get_frame(faiss_id):
        entry_frame_url = f"/api/v1/frames/{_encode_key(f'detection:frame:{faiss_id}')}"
    if entry_frame_url is None and _event_repo is not None:
        for event_type in ("entry", "last_seen"):
            if _event_repo.track_frame_exists(track_id, event_type):
                key = _event_repo.get_track_frame_key(track_id, event_type)
                entry_frame_url = f"/api/v1/frames/{_encode_key(key)}"
                break

    # ── Exit frame (rolling, only available within track_seen_ttl window) ──
    exit_frame_url = None
    exit_timestamp = None
    if exit_sim is not None and _detection_index is not None:
        exit_frame_key = _detection_index.get_exit_frame_url_key(track_id)
        if exit_frame_key:
            exit_frame_url = f"/api/v1/frames/{_encode_key(exit_frame_key)}"
        exit_meta = _detection_index.get_exit_meta(track_id) or {}
        exit_timestamp = exit_meta.get("timestamp")

    # ── Zone dwells ──
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
            entry_fk = dwell.get("entry_frame_key", "")
            exit_fk = dwell.get("exit_frame_key", "")
            if entry_fk:
                zone_entry["entry_frame_url"] = f"/api/v1/frames/{_encode_key(entry_fk)}"
            if exit_fk:
                zone_entry["exit_frame_url"] = f"/api/v1/frames/{_encode_key(exit_fk)}"
            zone_appearances.append(zone_entry)
        zone_appearances.sort(key=lambda z: z.get("entry_time") or "")

    # Overall similarity = best of entry and exit
    best_sim = max(entry_sim, exit_sim) if exit_sim is not None else entry_sim

    return {
        "faiss_id": faiss_id,
        "track_id": track_id,
        "camera_id": camera_id,
        "similarity": round(float(best_sim), 4),
        "entry_similarity": round(float(entry_sim), 4),
        "exit_similarity": round(float(exit_sim), 4) if exit_sim is not None else None,
        "entry_timestamp": meta.get("timestamp", ""),
        "exit_timestamp": exit_timestamp,
        "entry_frame_url": entry_frame_url,
        "exit_frame_url": exit_frame_url,
        "bbox": meta.get("bbox"),
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
