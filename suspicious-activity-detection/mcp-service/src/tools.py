"""FastMCP tools and SAD event-ingestion entry point."""

from __future__ import annotations

from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

import queries
from config import configured_zones, get_settings
from delivery import push_event_to_hub
from events import EVENT_TYPE, SCHEMA, EventEnvelope, ingest_alert as make_alert_event
from frame_store import SeaweedFrameStore
from models import Activity, TrendCount
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
            "Get_event_count",
            "Get_all_zones",
        ],
        "act_tools": [],
        "event_delivery": "one-way hub push" if _settings.event_hub_url else "not configured",
    }


@mcp.tool(annotations={"readOnlyHint": True})
def Get_all_activities() -> list[Activity]:
    """List every recorded suspicious-activity event, oldest first.

    Includes kitchen food-safety violations (event_name 'food_safety_violation'):
    dropped-and-returned food and objects picked up / grasped from the floor.
    """
    return queries.all_activities(store)


@mcp.tool(annotations={"readOnlyHint": True})
def Get_activity_by_zone(
    zone: Annotated[str, Field(description="Zone name, e.g. 'kitchen-prep'.")],
) -> list[Activity]:
    """List suspicious-activity events for a single zone."""
    return queries.activity_by_zone(store, zone)


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
) -> list[Activity]:
    """List zone events in an ISO-8601 time range using the store's configured timezone."""
    start_ms, end_ms = queries.parse_time_range(start_time, end_time, _settings.store_timezone)
    return queries.activity_by_zone_timestamp(store, zone, start_ms, end_ms)


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
) -> list[Activity]:
    """Search logged SAD events and SeaweedFS for matching alert frames."""
    start_ms, end_ms = queries.parse_time_range(start_time, end_time, _settings.store_timezone)
    activities = queries.retrospective_frame_search(
        store,
        query=query,
        start_ms=start_ms,
        end_ms=end_ms,
        zone=zone,
        event_name=event_name,
        use_case=use_case,
    )
    for activity in activities:
        frame = activity.get("frame", "")
        if frame:
            activity["frame_refs"] = [frame]
        else:
            activity["frame_refs"] = frame_store.find_alert_frames(
                activity.get("object_id", ""),
                activity.get("ref_id", ""),
            )
            if activity["frame_refs"]:
                activity["frame"] = activity["frame_refs"][0]
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
    """Count matching SAD events by station and shift."""
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