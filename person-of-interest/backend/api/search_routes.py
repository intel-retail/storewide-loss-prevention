"""Search API routes — offline face search via enrolled POI + detection index.

Two-stage search:
  Stage 1: Match query against enrolled POI index (same embedding space as
           online pipeline).  If a POI is identified, return recorded events.
  Stage 2: Search the detection index (all faces ever seen).  Each FAISS hit
           is a unique detection with its own stored frame.  Applies threshold
           filtering.

POI-matched searches skip detection index merge to avoid cross-domain false
positives (EmbeddingModelFactory vs DLStreamer embedding gap).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.core.config import get_config

log = logging.getLogger("poi.api.search")

router = APIRouter(prefix="/search", tags=["Search"])

# Margin between best and second-best similarity to reject ambiguous matches.
_SEARCH_MARGIN = 0.05

_embedding_factory = None
_detection_index = None
_event_repo = None
_faiss_repo = None  # enrolled POI index


def init(embedding_factory, detection_index, event_repo, faiss_repo=None) -> None:
    global _embedding_factory, _detection_index, _event_repo, _faiss_repo
    _embedding_factory = embedding_factory
    _detection_index = detection_index
    _event_repo = event_repo
    _faiss_repo = faiss_repo


@router.post("")
async def search_history(
    image: UploadFile = File(...),
    top_k: int = Form(20),
    start_time: str = Form(""),
    end_time: str = Form(""),
):
    """Search for a person across all historical detections by uploading an image.

    Works for both enrolled POIs and unknown persons.  Uses face detection
    + re-identification embedding on the query image, then searches both
    the enrolled POI index and the all-detections index.
    """
    if _detection_index is None and _faiss_repo is None:
        raise HTTPException(503, "No search index available")

    img_bytes = await image.read()
    if not img_bytes:
        raise HTTPException(400, "Image file is required")

    # ── Generate query embedding (face detection + reid) ──
    result = _embedding_factory.generate_from_bytes(img_bytes)
    if "error" in result:
        raise HTTPException(422, result["error"])

    query_vector = np.array(result["embedding"], dtype=np.float32)
    cfg = get_config()
    t_start = time.perf_counter()

    # ── Stage 1: enrolled POI index ──
    poi_match = _match_poi_index(query_vector, cfg)

    total_latency_ms = (time.perf_counter() - t_start) * 1000

    if poi_match is not None:
        # POI identified — return only verified POI event appearances
        poi_id = poi_match["poi_id"]
        poi_sim = poi_match["similarity"]
        poi_appearances = _get_poi_event_appearances(
            poi_id, poi_sim, start_time, end_time,
        )
        poi_appearances.sort(key=lambda a: a.get("best_match_time", ""), reverse=True)

        return {
            "event_type": "offline_search_result",
            "query_range": {"start": start_time, "end": end_time},
            "query_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "matched_poi_id": poi_id,
            "total_appearances": len(poi_appearances),
            "appearances": poi_appearances,
            "search_stats": {
                "search_stage": "poi_index",
                "poi_similarity": round(poi_sim, 4),
                "query_latency_ms": round(total_latency_ms, 2),
            },
        }

    # ── Stage 2: detection index (only for non-enrolled persons) ──
    if _detection_index is None or _detection_index.total_vectors() == 0:
        return _empty_response(start_time, end_time, total_latency_ms)

    threshold = cfg.similarity_threshold
    hits = _detection_index.search(query_vector, top_k=top_k)
    query_latency_ms = (time.perf_counter() - t_start) * 1000

    if not hits:
        return _empty_response(start_time, end_time, query_latency_ms)

    # ── Collect best entry hit per track ──
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
            "search_stage": "detection_index",
            "vectors_searched": _detection_index.total_vectors(),
            "raw_hits": len(hits),
            "unique_tracks": len(appearances),
            "query_latency_ms": round(query_latency_ms, 2),
        },
    }


# ── Stage 1 helpers ──────────────────────────────────────────────────────────

def _match_poi_index(
    query_vector: np.ndarray,
    cfg,
) -> Optional[dict]:
    """Return {"poi_id": ..., "similarity": ...} or None."""
    if _faiss_repo is None or _faiss_repo.total_vectors() == 0:
        return None

    hits = _faiss_repo.search(query_vector, top_k=cfg.search_top_k)
    if not hits:
        return None

    above = [(fid, sim) for fid, sim in hits if sim >= cfg.similarity_threshold]
    if not above:
        return None

    best_per_poi: dict[str, float] = {}
    for fid, sim in above:
        poi_id = _faiss_repo.get_poi_id_for_faiss_id(fid)
        if poi_id and (poi_id not in best_per_poi or sim > best_per_poi[poi_id]):
            best_per_poi[poi_id] = sim

    if not best_per_poi:
        return None

    sorted_pois = sorted(best_per_poi.items(), key=lambda x: x[1], reverse=True)
    best_poi_id, best_sim = sorted_pois[0]
    second_best_sim = sorted_pois[1][1] if len(sorted_pois) > 1 else 0.0
    margin = best_sim - second_best_sim

    log.info(
        "POI index: poi=%s sim=%.4f margin=%.4f threshold=%.2f",
        best_poi_id, best_sim, margin, cfg.similarity_threshold,
    )

    if len(sorted_pois) > 1 and margin < _SEARCH_MARGIN:
        log.warning("POI match rejected: ambiguous (margin %.4f)", margin)
        return None

    return {"poi_id": best_poi_id, "similarity": best_sim}


def _get_poi_event_appearances(
    poi_id: str,
    poi_sim: float,
    start_time: str,
    end_time: str,
) -> list[dict]:
    """Retrieve recorded events for a matched POI, grouped by track.

    Filters out tracks where another POI has more events (track purity check)
    to avoid false positives from reused track IDs.
    """
    if _event_repo is None:
        return []

    events = _event_repo.get_events_for_poi(poi_id, start_time or None, end_time or None)
    tracks: dict[str, list[dict]] = {}
    for evt in events:
        oid = evt.get("object_id", "unknown")
        tracks.setdefault(oid, []).append(evt)

    appearances = []
    for track_id, track_events in tracks.items():
        # Track purity check: skip tracks clearly dominated by another POI
        poi_counts = _event_repo.get_track_poi_counts(track_id)
        if poi_counts:
            our_count = poi_counts.get(poi_id, 0)
            total = sum(poi_counts.values())
            purity = our_count / total if total else 0
            best_other = max(
                (c for pid, c in poi_counts.items() if pid != poi_id),
                default=0,
            )
            # Require at least 40% purity — allows ties but rejects clear outsiders
            if purity < 0.4:
                log.info(
                    "Skipping track %s: purity %.1f%% (our=%d, other=%d)",
                    track_id, 100 * purity, our_count, best_other,
                )
                continue

        track_events.sort(key=lambda e: e.get("timestamp", ""))
        first = track_events[0]

        # Build appearance with track-level frames
        entry_frame_url = None
        last_seen_frame_url = None
        if _event_repo is not None:
            if _event_repo.track_frame_exists(track_id, "entry"):
                key = _event_repo.get_track_frame_key(track_id, "entry")
                entry_frame_url = f"/api/v1/frames/{_encode_key(key)}"
            if _event_repo.track_frame_exists(track_id, "last_seen"):
                key = _event_repo.get_track_frame_key(track_id, "last_seen")
                last_seen_frame_url = f"/api/v1/frames/{_encode_key(key)}"

        zone_appearances = _get_zone_appearances(track_id)

        appearance: dict = {
            "track_id": track_id,
            "camera_id": first.get("camera_id", ""),
            "similarity": round(poi_sim, 4),
            "best_match_time": first.get("timestamp", ""),
            "entry_frame_url": entry_frame_url,
            "last_seen_frame_url": last_seen_frame_url,
            "zone_appearances": zone_appearances,
        }
        for evt in track_events:
            tp = evt.get("thumbnail_path", "")
            if tp:
                appearance["thumbnail_url"] = tp
                break
        appearances.append(appearance)

    return appearances


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


def _empty_response(
    start_time: str,
    end_time: str,
    latency_ms: float,
    rejection_reason: str = "",
) -> dict:
    resp: dict = {
        "event_type": "offline_search_result",
        "query_range": {"start": start_time, "end": end_time},
        "query_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_appearances": 0,
        "appearances": [],
        "search_stats": {
            "vectors_searched": _detection_index.total_vectors() if _detection_index else 0,
            "raw_hits": 0,
            "unique_tracks_above_threshold": 0,
            "query_latency_ms": round(latency_ms, 2),
        },
    }
    if rejection_reason:
        resp["rejection_reason"] = rejection_reason
    return resp
