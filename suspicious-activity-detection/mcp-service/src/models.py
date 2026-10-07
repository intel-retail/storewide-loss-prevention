"""Typed structures returned by the SAD tools (standard MCP structured output)."""

from __future__ import annotations

from typing import NotRequired, TypedDict


class Activity(TypedDict):
    """One suspicious-activity event, flattened for the agent."""

    ref_id: str
    ts_ms: int
    timestamp: str
    event_name: str
    use_case: str
    zone: str
    pose: str
    severity: str
    camera_id: str
    object_id: str
    description: str
    frame: str
    frame_refs: list[str]
    frame_count: NotRequired[int]
    station: str
    shift: str


class TrendCount(TypedDict):
    """Count of matching events for one station/shift bucket."""

    station: str
    shift: str
    count: int


class DailyCount(TypedDict):
    """Count of matching events for one store-local day."""

    date: str
    count: int


class ZoneDailySummary(TypedDict):
    """Pre-computed daily breakdown, total, and trend for one zone."""

    zone: str
    total: int
    days_with_events: list[DailyCount]
    trend: str
    summary: str


class DailyCountsResult(TypedDict):
    """Daily counts over a store-local window, across all matching zones."""

    window_start: str
    window_end: str
    timezone: str
    total: int
    zones: list[ZoneDailySummary]
