from __future__ import annotations

import sqlite3

from events import EventEnvelope
from store import DurableEventStore


def event(ref_id: str, ts_ms: int) -> EventEnvelope:
    return EventEnvelope(
        event_type="report_suspicious_activity",
        service="suspicious_activity",
        store_id="store_001",
        payload={"zone": "kitchen-prep", "severity": "critical"},
        ref_id=ref_id,
        ts_ms=ts_ms,
    )


def test_opens_existing_sdk_schema_and_keeps_history(tmp_path):
    path = tmp_path / "sad_log.sqlite"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE events (seq INTEGER PRIMARY KEY AUTOINCREMENT, "
        "ref_id TEXT UNIQUE NOT NULL, event_type TEXT NOT NULL, "
        "ts_ms INTEGER NOT NULL, envelope TEXT NOT NULL)"
    )
    existing = event("existing-1", 100)
    connection.execute(
        "INSERT INTO events (ref_id, event_type, ts_ms, envelope) VALUES (?, ?, ?, ?)",
        (existing.ref_id, existing.event_type, existing.ts_ms, existing.to_json()),
    )
    connection.commit()
    connection.close()

    store = DurableEventStore(str(path), "suspicious_activity", "store_001")

    assert [item.ref_id for item in store.read(event_type=existing.event_type)] == ["existing-1"]
    assert store.append_once(existing) == (1, False)


def test_append_is_durable_idempotent_and_replay_is_ordered(tmp_path):
    path = tmp_path / "events.sqlite"
    store = DurableEventStore(str(path), "suspicious_activity", "store_001")
    first = event("event-1", 200)
    second = event("event-2", 100)

    assert store.append_once(first) == (1, True)
    assert store.append_once(first) == (1, False)
    assert store.append(second) == 2
    store.close()

    reopened = DurableEventStore(str(path), "suspicious_activity", "store_001")
    assert [item.ref_id for item in reopened.replay()] == ["event-1", "event-2"]
    assert [item.ref_id for item in reopened.replay(from_seq=1)] == ["event-2"]


def test_time_filtered_read_includes_bounds_and_latest_history(tmp_path):
    store = DurableEventStore(str(tmp_path / "events.sqlite"), "svc", "store_001")
    for index in range(1002):
        store.append(event(f"event-{index}", index))

    assert [item.ref_id for item in store.read(start_ms=1000, end_ms=1001)] == [
        "event-1000", "event-1001"
    ]
    assert [item.ref_id for item in store.read(limit=2, newest_first=True)] == [
        "event-1001", "event-1000"
    ]

