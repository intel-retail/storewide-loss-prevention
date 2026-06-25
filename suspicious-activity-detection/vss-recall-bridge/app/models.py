"""Pydantic request/response DTOs for the optional query proxy (build-spec §7.1)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class RecallSearchRequest(BaseModel):
    query: str  # appearance text, required
    cameras: list[str] | None = None  # optional camera_id filter -> tags
    time_start: datetime | None = None  # optional real-world window start
    time_end: datetime | None = None  # optional real-world window end
    video_pos_start: float | None = None  # optional in-video seconds filter
    video_pos_end: float | None = None
    limit: int = Field(default=20, ge=1, le=100)


class RecallHit(BaseModel):
    """1:1 with a VSS hit, no enrichment."""

    video_id: str
    tags: list[str]
    capture_time: datetime | None = None
    segment_start: float  # in-video seconds
    segment_end: float
    score: float
    video_url: str


class RecallSearchResponse(BaseModel):
    results: list[RecallHit]
