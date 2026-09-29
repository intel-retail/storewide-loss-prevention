"""SAD event schema and the pipeline ingest hand-off.

An "activity" is one SAD violation event. The SAD MQTT consumer (topics
alerts/#, ba/results) calls `ingest_alert` to publish one; emit() writes it to
the durable log first, then fans out to the agent.
"""

from __future__ import annotations

from typing import Any

EVENT_TYPE = "report_suspicious_activity"

SCHEMA: dict[str, str] = {
    "event_name": "str",       # e.g. food_safety_violation, loitering
    "use_case": "str",         # retail | kitchen
    "zone": "str",
    "pose": "str",            # e.g. reach-over, item-conceal, slip
    "severity": "str",        # low | medium | high | critical
    "camera_id": "str",
    "object_id": "str",       # SceneScape cross-camera person id
    "description": "str",     # VLM one-line summary
    "frame": "str",           # SeaweedFS frame/object URI or id
    "station": "str",         # prep station / checkout / zone group
    "shift": "str",           # breakfast | lunch | dinner | overnight | unknown
}


def ingest_alert(
    svc: Any,
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
) -> Any:
    return svc.emit(
        EVENT_TYPE,
        {
            "event_name": event_name,
            "use_case": use_case,
            "zone": zone,
            "pose": pose,
            "severity": severity,
            "camera_id": camera_id,
            "object_id": object_id,
            "description": description,
            "frame": frame,
            "station": station or zone,
            "shift": shift,
        },
        ref_id=ref_id,  # MQTT message id -> idempotent replay
        ts_ms=ts_ms,
    )
