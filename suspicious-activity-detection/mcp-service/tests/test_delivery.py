from __future__ import annotations

from events import EventEnvelope
import delivery


def test_hub_delivery_posts_inbox_event_and_retries(monkeypatch):
    event = EventEnvelope(
        event_type="report_suspicious_activity",
        service="suspicious_activity",
        store_id="store_001",
        payload={"severity": "critical"},
        ref_id="r1",
        ts_ms=100,
    )
    calls = []

    def post(url, posted_event, timeout_s):
        calls.append((url, delivery.to_hub_event(posted_event), timeout_s))
        return len(calls) == 2

    monkeypatch.setattr(delivery, "_post", post)
    monkeypatch.setattr(delivery.time, "sleep", lambda _: None)

    assert delivery.push_event_to_hub("http://hub/events", event)
    assert len(calls) == 2
    assert calls[-1][0] == "http://hub/events"
    assert calls[-1][1]["event_id"] == "r1"


def test_hub_event_maps_envelope_and_omits_top_level_store_id():
    body = delivery.to_hub_event(EventEnvelope(
        event_type="report_suspicious_activity",
        service="suspicious_activity",
        store_id="store_001",
        payload={"severity": "critical", "zone": "kitchen-prep"},
        ref_id="alert-1",
        ts_ms=1790674080000,
    ))

    # A top-level store_id is compared against the inbox restaurant id and rejected.
    assert "store_id" not in body
    assert body["event_id"] == "alert-1"
    assert body["event_type"] == "report_suspicious_activity"
    assert body["occurred_at"] == "2026-09-29T09:28:00Z"
    assert body["data"]["zone"] == "kitchen-prep"
    assert body["data"]["store_id"] == "store_001"


def test_missing_hub_url_does_not_attempt_delivery(monkeypatch):
    monkeypatch.setattr(delivery, "_post", lambda *_: (_ for _ in ()).throw(AssertionError()))
    assert not delivery.push_event_to_hub(None, EventEnvelope(
        "event", "service", "store", {}, ref_id="r2", ts_ms=100
    ))