from __future__ import annotations

from dataclasses import replace

import tools
from store import DurableEventStore


def test_ingest_persists_before_hub_push_and_suppresses_duplicate(monkeypatch):
    event_store = DurableEventStore(":memory:", "suspicious_activity", "store_001")
    monkeypatch.setattr(tools, "store", event_store)
    monkeypatch.setattr(
        tools,
        "_settings",
        replace(
            tools._settings,
            event_hub_url="http://event-hub/events",
            event_delivery_enabled=True,
            event_hub_min_severity="critical",
        ),
    )
    dispatches = []

    def capture_push(hub_url, event):
        saved = event_store.read(event_type=event.event_type)
        assert any(item.ref_id == event.ref_id for item in saved)
        dispatches.append((hub_url, event.ref_id, event.to_json()))
        return True

    monkeypatch.setattr(tools, "push_event_to_hub", capture_push)
    arguments = {
        "zone": "kitchen-prep",
        "pose": "floor_to_food_area",
        "severity": "critical",
        "camera_id": "camera-1",
        "object_id": "person-1",
        "description": "Item picked from the floor and returned to the food area",
        "ref_id": "alert-1",
        "event_name": "food_safety_violation",
        "use_case": "kitchen",
        "ts_ms": 12345,
    }

    tools.ingest_alert(**arguments)
    tools.ingest_alert(**arguments)

    assert len(dispatches) == 1
    assert dispatches[0][0:2] == ("http://event-hub/events", "alert-1")


def _ingest_with(monkeypatch, severity: str, **settings_overrides) -> list:
    event_store = DurableEventStore(":memory:", "suspicious_activity", "store_001")
    monkeypatch.setattr(tools, "store", event_store)
    settings = {
        "event_hub_url": "http://event-hub/events",
        "event_delivery_enabled": True,
        "event_hub_min_severity": "critical",
        **settings_overrides,
    }
    monkeypatch.setattr(tools, "_settings", replace(tools._settings, **settings))
    dispatches = []
    monkeypatch.setattr(tools, "push_event_to_hub", lambda url, event: dispatches.append(event.ref_id))
    tools.ingest_alert(
        zone="kitchen-prep",
        pose="floor_to_food_area",
        severity=severity,
        camera_id="camera-1",
        object_id="person-1",
        description="Item picked from the floor",
        ref_id=f"alert-{severity}",
        event_name="food_safety_violation",
        use_case="kitchen",
        ts_ms=12345,
    )
    return dispatches


def test_below_minimum_severity_is_logged_but_not_delivered(monkeypatch):
    assert _ingest_with(monkeypatch, "medium") == []
    assert _ingest_with(monkeypatch, "critical") == ["alert-critical"]


def test_delivery_off_suppresses_hub_push_for_benchmark_runs(monkeypatch):
    assert _ingest_with(monkeypatch, "critical", event_delivery_enabled=False) == []