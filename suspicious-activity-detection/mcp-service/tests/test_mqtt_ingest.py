"""Unit tests for alert MQTT payload normalization."""

from __future__ import annotations

from mqtt_ingest import normalize_alert


def test_normalize_kitchen_food_safety_alert():
    event = normalize_alert(
        "alerts/food_safety_violation",
        {
            "alert_id": "alert-1",
            "alert_type": "FOOD_SAFETY_VIOLATION",
            "alert_level": "CRITICAL",
            "metadata": {
                "person_id": "p-1001",
                "zone_id": "kitchen-prep",
                "zone_name": "kitchen-prep",
                "severity": "CRITICAL",
            },
            "payload": {
                "event_type": "food_safety_violation",
                "description": "Item picked from floor and placed back in food area",
                "frames_analyzed": 8,
            },
        },
        use_case="kitchen",
    )

    assert event == {
        "zone": "kitchen-prep",
        "pose": "food_safety_violation",
        "severity": "critical",
        "camera_id": "unknown",
        "object_id": "p-1001",
        "description": "Item picked from floor and placed back in food area",
        "ref_id": "alert-1",
        "event_name": "food_safety_violation",
        "use_case": "kitchen",
        "frame": "",
        "station": "kitchen-prep",
        "shift": "unknown",
        "ts_ms": None,
    }


def test_normalize_preserves_alert_source_timestamp():
    event = normalize_alert(
        "alerts/food_safety_violation",
        {
            "timestamp": "2026-09-29T09:28:00Z",
            "alert_type": "FOOD_SAFETY_VIOLATION",
            "metadata": {"zone_name": "kitchen-prep"},
            "payload": {"event_type": "food_safety_violation"},
        },
        use_case="kitchen",
    )

    assert event["ts_ms"] == 1790674080000


def test_normalize_accepts_epoch_seconds_and_ignores_invalid_timestamp():
    base_alert = {
        "alert_type": "LOITERING",
        "metadata": {"zone_id": "aisle1"},
        "payload": {},
    }

    assert normalize_alert(
        "alerts/loitering",
        {**base_alert, "timestamp": 1790674080},
        use_case="retail",
    )["ts_ms"] == 1790674080000
    assert normalize_alert(
        "alerts/loitering",
        {**base_alert, "timestamp": "not-a-time"},
        use_case="retail",
    )["ts_ms"] is None


def test_normalize_retail_alert_without_alert_id_gets_stable_ref():
    alert = {
        "alert_type": "LOITERING",
        "metadata": {"person_id": "p-2001", "zone_id": "aisle1"},
        "payload": {"dwell_seconds": 60},
    }

    event = normalize_alert("alerts/loitering", alert, use_case="retail")

    assert event["zone"] == "aisle1"
    assert event["object_id"] == "p-2001"
    assert event["event_name"] == "loitering"
    assert event["use_case"] == "retail"
    assert event["ref_id"].startswith("mqtt-")
    assert event["ref_id"] == normalize_alert("alerts/loitering", alert, use_case="retail")["ref_id"]