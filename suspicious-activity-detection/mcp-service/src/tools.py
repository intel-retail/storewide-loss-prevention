"""FastMCP tools and SAD event-ingestion entry point."""

from __future__ import annotations

from typing import Annotated

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

import queries
from config import configured_zones, get_settings
from delivery import push_event_to_hub
from events import EVENT_TYPE, SCHEMA, EventEnvelope, ingest_alert as make_alert_event
from frame_store import SeaweedFrameStore
from models import Activity, DailyCountsResult, TrendCount
from queries import SEVERITY_RANK
from store import DurableEventStore

_settings = get_settings()
store = DurableEventStore(
    _settings.log_path,
    service="suspicious_activity",
    store_id=_settings.store_id,
)
mcp = FastMCP("suspicious_activity")
frame_store = SeaweedFrameStore(
    _settings.seaweedfs_endpoint,
    bucket=_settings.seaweedfs_alerts_bucket,
)

_DEFAULT_LIMIT = 20
_MAX_LIMIT = 100
_MAX_FRAME_REFS = 1
# Large lists overflow the agent's context and get truncated to the oldest rows.
_LimitParam = Annotated[
    int,
    Field(
        ge=1,
        le=_MAX_LIMIT,
        description="Maximum records to return; the newest matching records are kept.",
    ),
]


def _newest(activities: list[Activity], limit: int) -> list[Activity]:
    return activities[-limit:]


_USE_CASES = {"retail", "kitchen"}


def _validate_filters(
    zone: str | None = None,
    event_name: str | None = None,
    use_case: str | None = None,
) -> None:
    """Fail loudly on unknown filter values; a silent [] reads as 'no incidents'."""
    if use_case and use_case not in _USE_CASES:
        raise ToolError(f"Unknown use_case '{use_case}'. Valid values: {sorted(_USE_CASES)}")
    if not zone and not event_name:
        return
    activities = queries.all_activities(store)
    checks = (
        ("zone", zone, set(configured_zones(_settings.zone_config_path))
         | {a["zone"] for a in activities if a.get("zone")}),
        ("event_name", event_name, {"food_safety_violation"}
         | {a["event_name"] for a in activities if a.get("event_name")}),
    )
    for name, value, valid in checks:
        if value and value not in valid:
            raise ToolError(
                f"Unknown {name} '{value}'. Valid values: {sorted(valid)}. "
                "Retry with a valid value or omit the filter."
            )


def _should_deliver(event: EventEnvelope) -> bool:
    """Only violations reach the agent inbox; every event stays queryable."""
    if not _settings.event_delivery_enabled:
        return False
    minimum = SEVERITY_RANK.get(_settings.event_hub_min_severity, 0)
    severity = str(event.payload.get("severity", "")).lower()
    return SEVERITY_RANK.get(severity, 0) >= minimum


def ingest_alert(
    zone: str,
    pose: str,
    severity: str,
    camera_id: str,
    object_id: str,
    description: str,
    ref_id: str | None = None,
    event_name: str = "report_suspicious_activity",
    use_case: str = "retail",
    frame: str = "",
    station: str = "",
    shift: str = "unknown",
    ts_ms: int | None = None,
) -> None:
    """Persist an alert before attempting configured event-hub delivery."""
    event = make_alert_event(
        _settings.store_id,
        zone,
        pose,
        severity,
        camera_id,
        object_id,
        description,
        ref_id,
        event_name,
        use_case,
        frame,
        station,
        shift,
        ts_ms,
    )
    _, inserted = store.append_once(event)
    if inserted and _should_deliver(event):
        push_event_to_hub(_settings.event_hub_url, event)


@mcp.tool(annotations={"readOnlyHint": True})
def describe() -> dict:
    """Describe the SAD service identity, event schema, and read tools."""
    return {
        "service": "suspicious_activity",
        "store_id": _settings.store_id,
        "event_types": {EVENT_TYPE: SCHEMA},
        "read_tools": [
            "Get_all_activities",
            "Get_activity_by_zone",
            "Get_activity_by_zone_timestamp",
            "Search_retrospective_frames",
            "Get_trend_counts",
            "Get_daily_counts",
            "Get_event_count",
            "Get_all_zones",
        ],
        "act_tools": [],
        "event_delivery": "one-way hub push" if _settings.event_hub_url else "not configured",
    }


@mcp.tool(annotations={"readOnlyHint": True})
def Get_all_activities(limit: _LimitParam = _DEFAULT_LIMIT) -> list[Activity]:
    """List the most recent suspicious-activity events (sample only, not a total).

    Includes kitchen food-safety violations (event_name 'food_safety_violation'):
    dropped-and-returned food and objects picked up / grasped from the floor.
    For how often or which station, call Get_trend_counts; for totals, Get_event_count.
    """
    return _newest(queries.all_activities(store), limit)


@mcp.tool(annotations={"readOnlyHint": True})
def Get_activity_by_zone(
    zone: Annotated[str, Field(description="Zone name, e.g. 'kitchen-prep'.")],
    limit: _LimitParam = _DEFAULT_LIMIT,
) -> list[Activity]:
    """List the most recent events for one zone (sample only, not a total)."""
    _validate_filters(zone=zone)
    return _newest(queries.activity_by_zone(store, zone), limit)


@mcp.tool(annotations={"readOnlyHint": True})
def Get_activity_by_zone_timestamp(
    zone: Annotated[str, Field(description="Zone name to filter by.")],
    start_time: Annotated[
        str | None,
        Field(description=f"Inclusive ISO datetime; local values use {_settings.store_timezone}."),
    ] = None,
    end_time: Annotated[
        str | None,
        Field(description=f"Inclusive ISO datetime; local values use {_settings.store_timezone}."),
    ] = None,
    limit: _LimitParam = _DEFAULT_LIMIT,
) -> list[Activity]:
    """List the most recent zone events in an ISO-8601 time range (store timezone)."""
    _validate_filters(zone=zone)
    start_ms, end_ms = queries.parse_time_range(start_time, end_time, _settings.store_timezone)
    return _newest(queries.activity_by_zone_timestamp(store, zone, start_ms, end_ms), limit)


@mcp.tool(annotations={"readOnlyHint": True})
def Search_retrospective_frames(
    query: Annotated[
        str | None,
        Field(description=(
            "Optional literal keyword filter; every meaningful word must match. "
            "Prefer structured filters for food-safety events."
        )),
    ] = None,
    start_time: Annotated[
        str | None,
        Field(description=f"Inclusive ISO datetime; local values use {_settings.store_timezone}."),
    ] = None,
    end_time: Annotated[
        str | None,
        Field(description=f"Inclusive ISO datetime; local values use {_settings.store_timezone}."),
    ] = None,
    zone: Annotated[
        str | None, Field(description="Optional zone, e.g. 'kitchen-prep'.")
    ] = None,
    event_name: Annotated[
        str | None, Field(description="Optional event name, e.g. 'food_safety_violation'.")
    ] = None,
    use_case: Annotated[
        str | None, Field(description="Optional use case: retail or kitchen.")
    ] = None,
    limit: _LimitParam = _DEFAULT_LIMIT,
) -> list[Activity]:
    """Return the most recent matching incident records with frame references.

    Use to list individual incidents or frame evidence. The result is a sample
    capped by limit, not a total: for how often or which station call
    Get_trend_counts, and for totals call Get_event_count.
    """
    _validate_filters(zone, event_name, use_case)
    start_ms, end_ms = queries.parse_time_range(start_time, end_time, _settings.store_timezone)
    activities = _newest(
        queries.retrospective_frame_search(
            store,
            query=query,
            start_ms=start_ms,
            end_ms=end_ms,
            zone=zone,
            event_name=event_name,
            use_case=use_case,
        ),
        limit,
    )
    for activity in activities:
        frame = activity.get("frame", "")
        refs = [frame] if frame else frame_store.find_alert_frames(
            activity.get("object_id", ""),
            activity.get("ref_id", ""),
        )
        activity["frame_count"] = len(refs)
        activity["frame_refs"] = refs[:_MAX_FRAME_REFS]
        if refs and not frame:
            activity["frame"] = refs[0]
    return activities


@mcp.tool(annotations={"readOnlyHint": True})
def Get_trend_counts(
    start_time: Annotated[
        str | None,
        Field(description=f"Inclusive ISO datetime; local values use {_settings.store_timezone}."),
    ] = None,
    end_time: Annotated[
        str | None,
        Field(description=f"Inclusive ISO datetime; local values use {_settings.store_timezone}."),
    ] = None,
    event_name: Annotated[
        str | None, Field(description="Optional event name, e.g. 'food_safety_violation'.")
    ] = None,
    use_case: Annotated[
        str | None, Field(description="Optional use case: retail or kitchen.")
    ] = None,
    zone: Annotated[
        str | None, Field(description="Optional zone, e.g. 'kitchen-prep'.")
    ] = None,
) -> list[TrendCount]:
    """Answer how often and at which station/shift: total matching events per bucket.

    Use for frequency questions such as 'how often, and which station?' about
    food dropped on the floor and put back (event_name 'food_safety_violation').
    """
    _validate_filters(zone, event_name, use_case)
    start_ms, end_ms = queries.parse_time_range(start_time, end_time, _settings.store_timezone)
    return queries.trend_counts(
        store,
        start_ms=start_ms,
        end_ms=end_ms,
        event_name=event_name,
        use_case=use_case,
        zone=zone,
    )


@mcp.tool(annotations={"readOnlyHint": True})
def Get_daily_counts(
    days: Annotated[
        int, Field(ge=1, le=90, description="Number of store-local days, today included.")
    ] = 7,
    event_name: Annotated[
        str | None, Field(description="Optional event name, e.g. 'food_safety_violation'.")
    ] = None,
    use_case: Annotated[
        str | None, Field(description="Optional use case: retail or kitchen.")
    ] = None,
    zone: Annotated[
        str | None, Field(description="Optional zone, e.g. 'kitchen-prep'.")
    ] = None,
) -> DailyCountsResult:
    """Count events per day and zone for the last N days, with totals and trend.

    Use for 'last 7 days', 'each day', 'daily counts', 'compare days', or
    'upward trend' questions. The service computes the date window, per-zone
    totals, and trend; report each zone's `summary` as returned.
    """
    _validate_filters(zone, event_name, use_case)
    return queries.daily_counts(
        store,
        days,
        _settings.store_timezone,
        event_name=event_name,
        use_case=use_case,
        zone=zone,
    )


@mcp.tool(annotations={"readOnlyHint": True})
def Get_event_count(
    start_time: Annotated[
        str | None,
        Field(description=f"Inclusive ISO datetime; local values use {_settings.store_timezone}."),
    ] = None,
    end_time: Annotated[
        str | None,
        Field(description=f"Inclusive ISO datetime; local values use {_settings.store_timezone}."),
    ] = None,
    zone: Annotated[str | None, Field(description="Optional zone, e.g. kitchen-prep.")] = None,
    event_name: Annotated[
        str | None, Field(description="Optional event name, e.g. food_safety_violation.")
    ] = None,
    use_case: Annotated[
        str | None, Field(description="Optional use case: retail or kitchen.")
    ] = None,
    minimum_severity: Annotated[
        str | None,
        Field(description="Optional severity threshold; high includes high and critical."),
    ] = None,
) -> int:
    """Return a compact count of events matching the supplied filters."""
    _validate_filters(zone, event_name, use_case)
    start_ms, end_ms = queries.parse_time_range(start_time, end_time, _settings.store_timezone)
    return queries.event_count(
        store,
        start_ms=start_ms,
        end_ms=end_ms,
        zone=zone,
        event_name=event_name,
        use_case=use_case,
        minimum_severity=minimum_severity,
    )


@mcp.tool(annotations={"readOnlyHint": True})
def Get_all_zones() -> list[str]:
    """List configured zones for the current use case."""
    zones = configured_zones(_settings.zone_config_path)
    return zones or queries.all_zones(store)