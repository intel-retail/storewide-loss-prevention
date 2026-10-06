"""Typed structures returned by the SAD tools (standard MCP structured output)."""

from __future__ import annotations

from typing import TypedDict


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
    station: str
    shift: str


class TrendCount(TypedDict):
    """Count of matching events for one station/shift bucket."""

    station: str
    shift: str
    count: int
