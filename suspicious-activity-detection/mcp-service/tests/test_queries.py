"""Unit tests for the SAD query helpers over its owned SQLite store."""

from __future__ import annotations

import events
import queries
from events import EventEnvelope
from store import DurableEventStore


def _new_log() -> DurableEventStore:
    return DurableEventStore(":memory:", "test", "store_001")


def _seed(log: DurableEventStore) -> None:
    def make_event(zone: str, ref: str, ts: int) -> EventEnvelope:
        kitchen = zone == "kitchen-prep"
        return EventEnvelope(
            event_type=events.EVENT_TYPE,
            service="suspicious_activity",
            store_id="store_001",
            payload={
                "event_name": "food_safety_violation" if kitchen else "loitering",
                "use_case": "kitchen" if kitchen else "retail",
                "zone": zone,
                "pose": "floor_to_food_area" if kitchen else "loiter",
                "severity": "high",
                "camera_id": "lp-camera1",
                "object_id": ref,
                "description": "item picked from floor and placed back in the food area",
                "frame": f"s3://behavioral-frames/{ref}.jpg",
                "station": "prep" if kitchen else "checkout",
                "shift": "lunch",
            },
            ref_id=ref,
            ts_ms=ts,
        )

    log.append(make_event("kitchen-prep", "a", 100))
    log.append(make_event("kitchen-prep", "b", 200))
    log.append(make_event("checkout-2", "c", 300))
    log.append(make_event("kitchen-prep", "a", 100))


def test_all_activities_idempotent():
    log = _new_log()
    _seed(log)
    assert len(queries.all_activities(log)) == 3


def test_activity_by_zone():
    log = _new_log()
    _seed(log)
    assert len(queries.activity_by_zone(log, "kitchen-prep")) == 2
    assert len(queries.activity_by_zone(log, "checkout-2")) == 1


def test_activity_by_zone_timestamp():
    log = _new_log()
    _seed(log)
    assert len(queries.activity_by_zone_timestamp(log, "kitchen-prep", start_ms=150)) == 1
    assert len(queries.activity_by_zone_timestamp(log, "kitchen-prep", end_ms=150)) == 1


def test_activity_includes_readable_store_local_timestamp():
    log = _new_log()
    _seed(log)
    activity = queries.activity_by_zone(log, "kitchen-prep")[0]
    assert activity["timestamp"] == "January 1, 1970 at 5:30:00 AM IST"
    assert activity["ts_ms"] == 100


def test_parse_store_local_iso_time_range():
    assert queries.parse_time_range(
        "2026-09-29T13:00:00", "2026-09-29T15:30:00", "Asia/Kolkata"
    ) == (1790667000000, 1790676000000)


def test_parse_offset_iso_time_range():
    assert queries.parse_time_range(
        "2026-09-29T13:00:00+05:30", "2026-09-29T15:30:00+05:30", "UTC"
    ) == (1790667000000, 1790676000000)


def test_parse_time_range_rejects_reversed_bounds():
    import pytest

    with pytest.raises(ValueError, match="start_time"):
        queries.parse_time_range(
            "2026-09-29T15:30:00", "2026-09-29T13:00:00", "Asia/Kolkata"
        )


def test_time_bounded_query_finds_events_after_oldest_read_limit():
    log = _new_log()
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

    result = queries.activity_by_zone_timestamp(
        log, "kitchen-prep", start_ms=50_000, end_ms=50_000
    )
    assert [item["ref_id"] for item in result] == ["recent-match"]
    assert [item["ref_id"] for item in queries.activity_by_zone(log, "kitchen-prep")] == [
        "recent-match"
    ]


def test_all_zones():
    log = _new_log()
    _seed(log)
    assert queries.all_zones(log) == ["checkout-2", "kitchen-prep"]


def test_retrospective_frame_search():
    log = _new_log()
    _seed(log)
    result = queries.retrospective_frame_search(
        log,
        query="floor",
        use_case="kitchen",
        event_name="food_safety_violation",
        start_ms=50,
        end_ms=250,
    )
    assert [item["ref_id"] for item in result] == ["a", "b"]


def test_retrospective_frame_search_matches_food_safety_intent():
    log = _new_log()
    _seed(log)
    result = queries.retrospective_frame_search(
        log,
        query="dropped on the floor and put back in the food area",
        use_case="kitchen",
        event_name="food_safety_violation",
    )
    assert [item["ref_id"] for item in result] == ["a", "b"]


def test_daily_counts_zero_fills_store_local_days():
    log = _new_log()
    day_ms = 86_400_000
    now_ms = 10 * day_ms + 12 * 3_600_000
    for ref, ts in (
        ("d1", now_ms - 2 * day_ms),
        ("d2", now_ms),
        ("d3", now_ms - 60_000),
        ("old", now_ms - 9 * day_ms),
    ):
        log.append(
            EventEnvelope(
                event_type=events.EVENT_TYPE,
                service="suspicious_activity",
                store_id="store_001",
                payload={"event_name": "food_safety_violation", "use_case": "kitchen", "zone": "kitchen-prep"},
                ref_id=ref,
                ts_ms=ts,
            )
        )
    result = queries.daily_counts(log, 3, "UTC", now_ms=now_ms, use_case="kitchen")
    assert (result["window_start"], result["window_end"], result["total"]) == (
        "1970-01-09",
        "1970-01-11",
        3,
    )
    [zone] = result["zones"]
    assert zone["zone"] == "kitchen-prep"
    assert zone["days_with_events"] == [
        {"date": "1970-01-09", "count": 1},
        {"date": "1970-01-11", "count": 2},
    ]
    assert zone["trend"] == "up"
    assert "3 events" in zone["summary"]


def test_trend_counts():
    log = _new_log()
    _seed(log)
    assert queries.trend_counts(log, use_case="kitchen") == [
        {"station": "prep", "shift": "lunch", "count": 2}
    ]


def test_trend_counts_scoped_to_zone():
    log = _new_log()
    _seed(log)
    assert queries.trend_counts(log, zone="kitchen-prep") == [
        {"station": "prep", "shift": "lunch", "count": 2}
    ]
    assert queries.trend_counts(log, zone="no-such-zone") == []


def test_event_count_high_includes_critical_and_filters_scope():
    log = _new_log()
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
    filters = {
        "zone": "kitchen-prep",
        "event_name": "food_safety_violation",
        "use_case": "kitchen",
    }
    assert queries.event_count(log, minimum_severity="high", **filters) == 3
    assert queries.event_count(log, minimum_severity="critical", **filters) == 1