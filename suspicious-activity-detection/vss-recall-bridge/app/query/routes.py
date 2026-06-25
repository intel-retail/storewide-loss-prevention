"""Thin query proxy: POST /recall/search -> VSS stateful search (build-spec §7).

Pure mapping onto VSS; no candidate set, no post-filter (except optional in-video
position), no DB. Every field comes straight off the VSS hit.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from ..config import get_settings
from ..models import RecallHit, RecallSearchRequest, RecallSearchResponse

router = APIRouter()


def build_tag_query(cameras: list[str] | None) -> str | None:
    """Camera ids -> comma-separated tag string (subset/OR matched by VSS)."""

    if not cameras:
        return None
    return ",".join(cameras)


def pad_window(
    start: datetime | None, end: datetime | None, pad_seconds: float
) -> tuple[datetime | None, datetime | None]:
    """Widen the requested window by one chunk to absorb ingest drift (design §3.2)."""

    if start is None and end is None:
        return None, None
    delta = timedelta(seconds=pad_seconds or 0)
    padded_start = start - delta if start is not None else None
    padded_end = end + delta if end is not None else None
    return padded_start, padded_end


def to_hit(meta: dict) -> RecallHit:
    """Map a raw VSS hit metadata dict to a RecallHit (straight field copy)."""

    tags = meta.get("tags")
    if isinstance(tags, str):
        tags = [t for t in tags.split(",") if t]

    return RecallHit(
        video_id=meta.get("video_id") or meta.get("videoId") or "",
        tags=tags or [],
        capture_time=meta.get("created_at") or meta.get("capture_time"),
        segment_start=float(meta.get("segment_start", 0) or 0),
        segment_end=float(meta.get("segment_end", 0) or 0),
        score=float(meta.get("relevance_score", meta.get("score", 0)) or 0),
        video_url=meta.get("video_url") or meta.get("videoUrl") or "",
    )


@router.post("/search", response_model=RecallSearchResponse)
async def search(req: RecallSearchRequest, request: Request) -> RecallSearchResponse:
    settings = get_settings()

    known: set[str] = request.app.state.camera_ids
    if req.cameras:
        unknown = [c for c in req.cameras if c not in known]
        if unknown:
            raise HTTPException(status_code=400, detail=f"unknown cameras: {unknown}")

    tags = build_tag_query(req.cameras)
    start, end = pad_window(req.time_start, req.time_end, settings.window_pad_seconds)

    raw = await request.app.state.vss.search(
        query=req.query, tags=tags, time_start=start, time_end=end
    )

    hits = raw
    if req.video_pos_start is not None or req.video_pos_end is not None:
        lo = req.video_pos_start or 0
        hi = req.video_pos_end if req.video_pos_end is not None else float("inf")
        hits = [
            m
            for m in hits
            if float(m.get("segment_start", 0) or 0) < hi
            and float(m.get("segment_end", 0) or 0) > lo
        ]

    results = [to_hit(m) for m in hits]
    results.sort(key=lambda h: h.score, reverse=True)
    return RecallSearchResponse(results=results[: req.limit])


@router.get("/clips/{clip_id}")
async def get_clip(clip_id: str, request: Request) -> Response:
    """Proxy clip bytes through the authenticated bridge instead of exposing the
    raw MinIO URL to the browser (build-spec §7.3, §8).

    The upstream download endpoint does not support HTTP Range, so the (small)
    clip is buffered and served with Range support to make the player seekable.
    """

    vss = request.app.state.vss
    data = await vss.fetch_clip(clip_id)
    total = len(data)
    headers = {"Accept-Ranges": "bytes", "Content-Type": "video/mp4"}

    range_header = request.headers.get("range") or request.headers.get("Range")
    if range_header and range_header.startswith("bytes="):
        spec = range_header.split("=", 1)[1].split(",", 1)[0].strip()
        start_s, _, end_s = spec.partition("-")
        try:
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else total - 1
        except ValueError:
            start, end = 0, total - 1
        start = max(0, start)
        end = min(end, total - 1)
        if start > end:
            start = 0
        chunk = data[start : end + 1]
        headers.update(
            {
                "Content-Range": f"bytes {start}-{end}/{total}",
                "Content-Length": str(len(chunk)),
            }
        )
        return Response(content=chunk, status_code=206, headers=headers)

    headers["Content-Length"] = str(total)
    return Response(content=data, status_code=200, headers=headers)

