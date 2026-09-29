"""Wires the ServiceServer: registers the event type and read tools.

Runtime actions are intentionally not exposed; kitchen zone/rule setup is
configuration under configs/usecase/kitchen/.
"""

from __future__ import annotations

from typing import Annotated

from mcp_service_sdk import ServiceConfig, ServiceServer
from pydantic import Field

import queries
from config import configured_zones, get_settings
from events import EVENT_TYPE, SCHEMA
from events import ingest_alert as _ingest_alert
from models import Activity, TrendCount

_settings = get_settings()

svc = ServiceServer.from_config(
    ServiceConfig(
        service="suspicious_activity",
        store_id=_settings.store_id,
        log_backend=_settings.log_backend,
        log_path=_settings.log_path,
        delivery=_settings.delivery,
        webhook_url=_settings.webhook_url,
        expose_subscribe=_settings.expose_subscribe,
    )
)
svc.register_event_type(EVENT_TYPE, schema=SCHEMA)


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
    """Pipeline hand-off: the SAD MQTT consumer calls this to publish a violation."""
    _ingest_alert(
        svc,
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


# -- read tools -----------------------------------------------------------
# Descriptions come from the docstrings; parameter docs from Annotated Field.
@svc.read_tool("Get_all_activities")
def Get_all_activities() -> list[Activity]:
    """List every recorded suspicious-activity event, oldest first."""
    return queries.all_activities(svc.log)


@svc.read_tool("Get_activity_by_zone")
def Get_activity_by_zone(
    zone: Annotated[str, Field(description="Zone name, e.g. 'kitchen-prep'.")],
) -> list[Activity]:
    """List suspicious-activity events for a single zone."""
    return queries.activity_by_zone(svc.log, zone)


@svc.read_tool("Get_activity_by_zone_timestamp")
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
    return queries.activity_by_zone_timestamp(svc.log, zone, start_ms, end_ms)


@svc.read_tool("Search_retrospective_frames")
def Search_retrospective_frames(
    query: Annotated[
        str | None,
        Field(description="Optional text filter, e.g. 'floor food area'."),
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
        str | None, Field(description="Optional zone filter, e.g. 'kitchen-prep'.")
    ] = None,
    event_name: Annotated[
        str | None,
        Field(description="Optional event filter, e.g. 'food_safety_violation'."),
    ] = None,
    use_case: Annotated[
        str | None, Field(description="Optional use case filter: retail or kitchen.")
    ] = None,
) -> list[Activity]:
    """Search logged SAD events and return any frame references already attached to them."""
    start_ms, end_ms = queries.parse_time_range(start_time, end_time, _settings.store_timezone)
    return queries.retrospective_frame_search(
        svc.log,
        query=query,
        start_ms=start_ms,
        end_ms=end_ms,
        zone=zone,
        event_name=event_name,
        use_case=use_case,
    )


@svc.read_tool("Get_trend_counts")
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
        str | None,
        Field(description="Optional event filter, e.g. 'food_safety_violation'."),
    ] = None,
    use_case: Annotated[
        str | None, Field(description="Optional use case filter: retail or kitchen.")
    ] = None,
) -> list[TrendCount]:
    """Count matching SAD events by station and shift."""
    start_ms, end_ms = queries.parse_time_range(start_time, end_time, _settings.store_timezone)
    return queries.trend_counts(
        svc.log,
        start_ms=start_ms,
        end_ms=end_ms,
        event_name=event_name,
        use_case=use_case,
    )


@svc.read_tool("Get_event_count")
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
        Field(description="Optional severity threshold: low, medium, high, or critical. High includes high and critical."),
    ] = None,
) -> int:
    """Return a compact count of events matching the supplied filters."""
    start_ms, end_ms = queries.parse_time_range(start_time, end_time, _settings.store_timezone)
    return queries.event_count(
        svc.log,
        start_ms=start_ms,
        end_ms=end_ms,
        zone=zone,
        event_name=event_name,
        use_case=use_case,
        minimum_severity=minimum_severity,
    )


@svc.read_tool("Get_all_zones")
def Get_all_zones() -> list[str]:
    """List configured zones for the current use case."""
    zones = configured_zones(_settings.zone_config_path)
    if zones:
        return zones
    return queries.all_zones(svc.log)
