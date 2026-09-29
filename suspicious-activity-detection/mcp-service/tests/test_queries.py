"""Unit tests for the SAD query helpers (pure log queries)."""

from __future__ import annotations

from mcp_service_sdk import SQLiteLog

import events
import queries


def _seed(log: SQLiteLog) -> None:
    def ev(zone: str, ref: str, ts: int):
        from mcp_service_sdk.envelope import EventEnvelope

        return EventEnvelope(
            event_type=events.EVENT_TYPE,
            service="suspicious_activity",
            store_id="store_001",
            payload={
                "event_name": "food_safety_violation" if zone == "kitchen-prep" else "loitering",
                "use_case": "kitchen" if zone == "kitchen-prep" else "retail",
                "zone": zone,
                "pose": "floor_to_food_area" if zone == "kitchen-prep" else "loiter",
                "severity": "high",
                "camera_id": "lp-camera1",
                "object_id": ref,
                "description": "item picked from floor and placed back in the food area",
                "frame": f"s3://behavioral-frames/{ref}.jpg",
                "station": "prep" if zone == "kitchen-prep" else "checkout",
                "shift": "lunch",
            },
            ref_id=ref,
            ts_ms=ts,
        )

    log.append(ev("kitchen-prep", "a", 100))
    log.append(ev("kitchen-prep", "b", 200))
    log.append(ev("checkout-2", "c", 300))
    log.append(ev("kitchen-prep", "a", 100))  # idempotent duplicate


def test_all_activities_idempotent():
    log = SQLiteLog(service="t")
    _seed(log)
    assert len(queries.all_activities(log)) == 3


def test_activity_by_zone():
    log = SQLiteLog(service="t")
    _seed(log)
    assert len(queries.activity_by_zone(log, "kitchen-prep")) == 2
    assert len(queries.activity_by_zone(log, "checkout-2")) == 1


def test_activity_by_zone_timestamp():
    log = SQLiteLog(service="t")
    _seed(log)
    assert len(queries.activity_by_zone_timestamp(log, "kitchen-prep", start_ms=150)) == 1
    assert len(queries.activity_by_zone_timestamp(log, "kitchen-prep", end_ms=150)) == 1


def test_activity_includes_readable_store_local_timestamp():
    log = SQLiteLog(service="t")
    _seed(log)

    activity = queries.activity_by_zone(log, "kitchen-prep")[0]

    assert activity["timestamp"] == "January 1, 1970 at 5:30:00 AM IST"
    assert activity["ts_ms"] == 100


def test_parse_store_local_iso_time_range():
    assert queries.parse_time_range(
        "2026-09-29T13:00:00",
        "2026-09-29T15:30:00",
        "Asia/Kolkata",
    ) == (1790667000000, 1790676000000)


def test_parse_offset_iso_time_range():
    assert queries.parse_time_range(
        "2026-09-29T13:00:00+05:30",
        "2026-09-29T15:30:00+05:30",
        "UTC",
    ) == (1790667000000, 1790676000000)


def test_parse_time_range_rejects_reversed_bounds():
    import pytest

    with pytest.raises(ValueError, match="start_time"):
        queries.parse_time_range(
            "2026-09-29T15:30:00",
            "2026-09-29T13:00:00",
            "Asia/Kolkata",
        )


def test_time_bounded_query_finds_events_after_oldest_read_limit():
    from mcp_service_sdk.envelope import EventEnvelope

    log = SQLiteLog(service="t")
    for index in range(10_001):
        log.append(EventEnvelope(
            event_type=events.EVENT_TYPE,
            service="suspicious_activity",
            store_id="store_001",
            payload={"zone": "other", "event_name": "loitering"},
            ref_id=f"old-{index}",
            ts_ms=1000 + index,
        ))
    log.append(EventEnvelope(
        event_type=events.EVENT_TYPE,
        service="suspicious_activity",
        store_id="store_001",
        payload={"zone": "kitchen-prep", "event_name": "food_safety_violation"},
        ref_id="recent-match",
        ts_ms=50_000,
    ))

    results = queries.activity_by_zone_timestamp(
        log, "kitchen-prep", start_ms=50_000, end_ms=50_000
    )

    assert [event["ref_id"] for event in results] == ["recent-match"]
    assert [event["ref_id"] for event in queries.activity_by_zone(log, "kitchen-prep")] == [
        "recent-match"
    ]


def test_all_zones():
    log = SQLiteLog(service="t")
    _seed(log)
    assert queries.all_zones(log) == ["checkout-2", "kitchen-prep"]


def test_retrospective_frame_search():
    log = SQLiteLog(service="t")
    _seed(log)
    results = queries.retrospective_frame_search(
        log,
        query="floor",
        use_case="kitchen",
        event_name="food_safety_violation",
        start_ms=50,
        end_ms=250,
    )
    assert [r["ref_id"] for r in results] == ["a", "b"]


def test_retrospective_frame_search_matches_food_safety_intent():
    log = SQLiteLog(service="t")
    _seed(log)
    results = queries.retrospective_frame_search(
        log,
        query="dropped on the floor and put back in the food area",
        use_case="kitchen",
        event_name="food_safety_violation",
    )
    assert [r["ref_id"] for r in results] == ["a", "b"]


def test_trend_counts():
    log = SQLiteLog(service="t")
    _seed(log)
    assert queries.trend_counts(log, use_case="kitchen") == [
        {"station": "prep", "shift": "lunch", "count": 2}
    ]


def test_event_count_high_includes_critical_and_filters_scope():
    from mcp_service_sdk.envelope import EventEnvelope

    log = SQLiteLog(service="t")
    _seed(log)
    log.append(EventEnvelope(
        event_type=events.EVENT_TYPE,
        service="suspicious_activity",
        store_id="store_001",
        payload={
            "event_name": "food_safety_violation",
            "use_case": "kitchen",
            "zone": "kitchen-prep",
            "severity": "critical",
        },
        ref_id="critical-kitchen-event",
        ts_ms=400,
    ))

    assert queries.event_count(
        log,
        zone="kitchen-prep",
        event_name="food_safety_violation",
        use_case="kitchen",
        minimum_severity="high",
    ) == 3
    assert queries.event_count(
        log,
        zone="kitchen-prep",
        event_name="food_safety_violation",
        use_case="kitchen",
        minimum_severity="critical",
    ) == 1
