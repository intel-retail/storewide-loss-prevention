from __future__ import annotations

import json

from events import EventEnvelope
import delivery


def test_hub_delivery_posts_standard_envelope_and_retries(monkeypatch):
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
        calls.append((url, json.loads(posted_event.to_json()), timeout_s))
        return len(calls) == 2

    monkeypatch.setattr(delivery, "_post", post)
    monkeypatch.setattr(delivery.time, "sleep", lambda _: None)

    assert delivery.push_event_to_hub("http://hub/events", event)
    assert len(calls) == 2
    assert calls[-1][0] == "http://hub/events"
    assert calls[-1][1]["ref_id"] == "r1"


def test_missing_hub_url_does_not_attempt_delivery(monkeypatch):
    monkeypatch.setattr(delivery, "_post", lambda *_: (_ for _ in ()).throw(AssertionError()))
    assert not delivery.push_event_to_hub(None, EventEnvelope(
        "event", "service", "store", {}, ref_id="r2", ts_ms=100
    ))