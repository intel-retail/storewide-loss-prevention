"""Tests for ScenescapeRegionConsumer — region entry/exit via native events + camera_bounds."""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from backend.consumers.scenescape_consumer import ScenescapeRegionConsumer


@pytest.fixture
def event_service():
    return MagicMock()


@pytest.fixture
def consumer(event_service):
    return ScenescapeRegionConsumer(event_service)


# Regulated scene topic format: scenescape/regulated/scene/{scene_id}
_TOPIC = "scenescape/regulated/scene/scene-001"


def _make_payload(timestamp, persons):
    return {
        "timestamp": timestamp,
        "objects": persons,
    }


def _make_person(person_id, regions=None, category="person"):
    p = {"id": person_id, "category": category}
    if regions is not None:
        p["regions"] = regions
    return p


# ── Native region event tests (scenescape/event/region topic) ────────────────

def test_region_entry_detected(consumer, event_service):
    """Native region event with 'entered' triggers store_region_entry."""
    data = {
        "timestamp": "2025-01-15T12:00:00Z",
        "entered": [{"id": "person-1", "visibility": ["cam1"],
                      "regions": {"zone-a": {"entered": "2025-01-15T12:00:00Z", "name": "Zone A"}}}],
        "exited": [],
    }
    consumer.handle_region_event("scene-001", "zone-a", data)
    event_service.store_region_entry.assert_called_once_with(
        "person-1", "2025-01-15T12:00:00Z", "scene-001", "zone-a", "Zone A", "cam1",
        entry_frame_key=None,
    )
    event_service.store_region_exit.assert_not_called()


def test_region_exit_detected(consumer, event_service):
    """Native region event with 'exited' triggers store_region_exit with dwell."""
    data = {
        "timestamp": "2025-01-15T12:05:00Z",
        "entered": [],
        "exited": [{"object": {"id": "person-1"}, "dwell": 5.2}],
    }
    consumer.handle_region_event("scene-001", "zone-a", data)
    event_service.store_region_exit.assert_called_once()
    call_args = event_service.store_region_exit.call_args
    assert call_args[0][0] == "person-1"
    assert call_args[0][3] == "zone-a"
    assert call_args[1]["dwell_override"] == 5.2


def test_non_matching_topic_ignored(consumer, event_service):
    """Topics not matching the regulated pattern are silently ignored."""
    consumer.handle_event("scenescape/event/bfb9f86b/objects", {"timestamp": "t", "objects": []})
    event_service.store_region_entry.assert_not_called()
    event_service.store_region_exit.assert_not_called()


def test_service_exception_does_not_propagate(consumer, event_service):
    """Exceptions in event_service are caught and do not raise."""
    event_service.store_region_entry.side_effect = RuntimeError("Redis down")
    data = {
        "timestamp": "t1",
        "entered": [{"id": "person-2", "regions": {"zone-b": {"name": "Zone B"}}}],
        "exited": [],
    }
    # Should not raise
    consumer.handle_region_event("scene-001", "zone-b", data)


def test_person_leaving_scene_triggers_exit(consumer, event_service):
    """Native exit event with object wrapper triggers exit."""
    data = {
        "timestamp": "t1",
        "entered": [],
        "exited": [{"object": {"id": "person-3"}, "dwell": 10.0}],
    }
    consumer.handle_region_event("scene-001", "zone-a", data)
    event_service.store_region_exit.assert_called_once()


def test_multiple_regions_tracked(consumer, event_service):
    """Multiple entries in a single event message are all processed."""
    data = {
        "timestamp": "t1",
        "entered": [
            {"id": "p1", "regions": {"zone-a": {"name": "Zone A"}}},
            {"id": "p2", "regions": {"zone-a": {"name": "Zone A"}}},
        ],
        "exited": [],
    }
    consumer.handle_region_event("scene-001", "zone-a", data)
    assert event_service.store_region_entry.call_count == 2


# ── Regulated topic tests (camera_bounds only, no entry/exit) ────────────────

def test_non_person_ignored(consumer, event_service):
    """Non-person objects should not trigger region events."""
    obj = _make_person("cart-1", regions={"zone-a": {"name": "Zone A"}}, category="cart")
    payload = _make_payload("t1", [obj])
    consumer.handle_event(_TOPIC, payload)
    event_service.store_region_entry.assert_not_called()
    event_service.store_region_exit.assert_not_called()


def test_regulated_topic_no_entry_exit(consumer, event_service):
    """Regulated topic handler no longer generates entry/exit events (only camera_bounds)."""
    person = _make_person("person-1", regions={"zone-a": {"name": "Zone A"}})
    payload = _make_payload("2025-01-15T12:00:00Z", [person])
    consumer.handle_event(_TOPIC, payload)
    event_service.store_region_entry.assert_not_called()
    event_service.store_region_exit.assert_not_called()
