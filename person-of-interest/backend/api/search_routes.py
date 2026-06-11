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


def init(embedding_factory, detection_index, event_repo, **_kwargs) -> None:
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
    
    t0 = time.perf_counter()
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
    # Search with a wide net to ensure enough candidates from all cameras
    # survive time-range and similarity filtering.  With multiple cameras at
    # different angles, the same person may score very differently — a narrow
    # top-k can miss an entire camera's detections.
    
    top_k = max(1, min(top_k, 200))
    total_vecs = _detection_index.total_vectors()
    # Search wide enough to capture cross-camera results where the same person
    # may score very differently due to viewing angle.  With 15k+ vectors and
    # two cameras, a minimum of 2000 ensures both cameras' vectors are reached.
    search_k = min(max(top_k * 50, 2000), total_vecs) if total_vecs > 0 else 2000
    hits = _detection_index.search(query_vector, top_k=search_k)
    

    if hits:
        sims = sorted((s for _, s in hits), reverse=True)
        log.debug(
            "Search: %d hits, top-5 sims=[%s], threshold=%.2f, range=%s→%s",
            len(hits),
            ", ".join(f"{s:.4f}" for s in sims[:5]),
            get_config().search_similarity_threshold,
            start_time, end_time,
        )

    if not hits:
        return _empty_response(start_time, end_time, query_latency_ms)

    # ── Collect best entry AND best exit hit per track ──
    # Promoted exit vectors (added to FAISS by ExitPromoterThread) carry
    # meta["role"]="exit".  We separate them so the search can show both
    # entry and exit data even after the rolling Redis exit keys expire.
    cfg = get_config()
    threshold = cfg.search_similarity_threshold
    best_entry: dict[str, dict] = {}  # track_id → {faiss_id, similarity, meta}
    best_exit: dict[str, dict] = {}   # track_id → {faiss_id, similarity, meta}

    # Filter by threshold first, then batch-fetch metadata in one pipeline call
    above_threshold = [(fid, sim) for fid, sim in hits if sim >= threshold]
    all_meta = _detection_index.batch_get_metadata([fid for fid, _ in above_threshold])

    for faiss_id, similarity in above_threshold:
        meta = all_meta.get(faiss_id)
        if meta is None:
            continue
        ts = meta.get("timestamp", "")
        if start_time and ts and ts < start_time:
            continue
        if end_time and ts and ts > end_time:
            continue
        track_id = meta["track_id"]
        role = meta.get("role", "entry")
        if role == "exit":
            if track_id not in best_exit or similarity > best_exit[track_id]["similarity"]:
                best_exit[track_id] = {"faiss_id": faiss_id, "similarity": similarity, "meta": meta}
        else:
            if track_id not in best_entry or similarity > best_entry[track_id]["similarity"]:
                best_entry[track_id] = {"faiss_id": faiss_id, "similarity": similarity, "meta": meta}

    if not best_entry:
        return _empty_response(start_time, end_time, query_latency_ms)

    # ── Check rolling exit vectors for the same tracks ──
    exit_sims = _detection_index.search_exits(query_vector, list(best_entry.keys()))

    # Rolling exits belong to the same tracker track as the entry — they are
    # the last face seen before the person left.  Since the entry similarity
    # already confirmed identity, we trust the exit frame without re-checking
    # against the query threshold (the person may look very different at exit:
    # back of head, far away, different angle).

    # Fallback: use promoted FAISS exit vectors for tracks where the rolling
    # Redis exit has already expired (TTL=15min).  Promoted exits are permanent
    # in FAISS and carry the same track_id.
    for track_id in best_entry:
        if track_id not in exit_sims and track_id in best_exit:
            exit_sims[track_id] = best_exit[track_id]["similarity"]

    # ── Pre-fetch all zone dwells in batch ──
    # Collect unique base object IDs (UUIDs without @timestamp) — dwells are
    # keyed by raw UUID only.  Appearance IDs (uuid@ts) never have dwell data.
    dwell_cache: dict[str, list[dict]] = {}
    if _event_repo is not None:
        lookup_ids_set: set[str] = set()
        for track_id in best_entry:
            base_id = track_id.rsplit("@", 1)[0] if "@" in track_id else track_id
            lookup_ids_set.add(base_id)
        dwell_cache = _event_repo.batch_get_region_dwells(lookup_ids_set)

    # ── Build one grouped appearance per track (entry + exit on same card) ──
    appearances = []
    for track_id, entry in best_entry.items():
        exit_sim = exit_sims.get(track_id)
        promoted_exit = best_exit.get(track_id)
        appearance = _build_grouped_appearance(
            entry["faiss_id"], entry["similarity"], entry["meta"],
            exit_sim=exit_sim, track_id=track_id,
            promoted_exit=promoted_exit,
            dwell_cache=dwell_cache,
        )
        appearances.append(appearance)

    # Sort by best similarity (max of entry and exit) descending
    appearances.sort(key=lambda a: a["similarity"], reverse=True)
    query_latency_ms = (time.perf_counter() - t0) * 1000
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
    promoted_exit: Optional[dict] = None,
    dwell_cache: Optional[dict[str, list[dict]]] = None,
) -> dict:
    """Build one appearance card grouping entry and exit for the same track."""
    camera_id = meta.get("camera_id", "")

    # ── Entry frame ──
    entry_frame_url = None
    if _detection_index is not None and _detection_index.has_frame(faiss_id):
        entry_frame_url = f"/api/v1/frames/{_encode_key(f'detection:frame:{faiss_id}')}"
    if entry_frame_url is None and _event_repo is not None:
        for event_type in ("entry", "last_seen"):
            if _event_repo.track_frame_exists(track_id, event_type):
                key = _event_repo.get_track_frame_key(track_id, event_type)
                entry_frame_url = f"/api/v1/frames/{_encode_key(key)}"
                break

    # ── Exit frame and timestamp ──
    # Priority: rolling Redis exit (fresh, within 15min) → promoted FAISS exit (permanent)
    #         → durable final_exit record (from promotion or SceneScape exit)
    exit_frame_url = None
    exit_timestamp = None
    exit_bbox = None
    if exit_sim is not None and _detection_index is not None:
        # Try rolling Redis exit first (still within track_seen_ttl)
        exit_frame_key = _detection_index.get_exit_frame_url_key(track_id)
        if exit_frame_key:
            exit_frame_url = f"/api/v1/frames/{_encode_key(exit_frame_key)}"
        exit_meta = _detection_index.get_exit_meta(track_id) or {}
        exit_timestamp = exit_meta.get("timestamp")
        exit_bbox = exit_meta.get("bbox")

        # Fallback to promoted FAISS exit if rolling Redis exit expired
        if promoted_exit and not exit_timestamp:
            exit_timestamp = promoted_exit["meta"].get("timestamp")
            exit_bbox = exit_bbox or promoted_exit["meta"].get("bbox")
        if promoted_exit and not exit_frame_url:
            exit_faiss_id = promoted_exit["faiss_id"]
            if _detection_index.has_frame(exit_faiss_id):
                exit_frame_url = f"/api/v1/frames/{_encode_key(f'detection:frame:{exit_faiss_id}')}"

    # Final fallback: durable final_exit record — always available after
    # promotion or SceneScape region exit, regardless of FAISS top-k.
    # Fill any missing exit fields independently.
    if _detection_index is not None and (
        not exit_timestamp or not exit_frame_url or not exit_bbox
    ):
        final_exit = _detection_index.get_final_exit(track_id)
        if final_exit:
            if not exit_timestamp:
                exit_timestamp = final_exit.get("timestamp")
            if not exit_bbox:
                exit_bbox = final_exit.get("bbox")
            if not exit_frame_url:
                fe_faiss_id = final_exit.get("faiss_id")
                fe_frame_key = final_exit.get("frame_key")
                if fe_faiss_id is not None and _detection_index.has_frame(fe_faiss_id):
                    exit_frame_url = f"/api/v1/frames/{_encode_key(f'detection:frame:{fe_faiss_id}')}"
                elif fe_frame_key and _event_repo and _event_repo.has_zone_frame(fe_frame_key):
                    exit_frame_url = f"/api/v1/frames/{_encode_key(fe_frame_key)}"
            if exit_sim is None:
                exit_sim = final_exit.get("similarity")

    # ── Zone dwells ──
    # Region dwells are keyed by the SceneScape UUID (object_id used by
    # ScenescapeRegionConsumer), while the detection index track_id is an
    # appearance_id like "cam:Camera_02:1@1715100000" or "uuid-abc@1715100000".
    # Extract the base object_id (before @timestamp) and also try the full
    # track_id to cover both ID spaces.
    zone_appearances = []
    if dwell_cache is not None:
        base_object_id = track_id.rsplit("@", 1)[0] if "@" in track_id else track_id
        lookup_ids = {track_id, base_object_id}
        seen_dwell_keys: set = set()
        for lookup_id in lookup_ids:
            dwells = dwell_cache.get(lookup_id, [])
            for dwell in dwells:
                # Dedup in case both IDs resolve to the same dwell records
                dwell_key = (dwell.get("region_id", ""), dwell.get("entry_time", ""))
                if dwell_key in seen_dwell_keys:
                    continue
                seen_dwell_keys.add(dwell_key)
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
        "exit_bbox": exit_bbox,
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
