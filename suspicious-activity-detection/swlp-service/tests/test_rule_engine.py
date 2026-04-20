# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for RuleEngine (generic) and RuleEngineAdapter (LP-specific)."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

from models.events import EventType, RegionEvent, ZoneType
from models.alerts import AlertType, AlertLevel
from models.session import PersonSession
from rule_engine import RuleEngine, Action
from services.rule_adapter import RuleEngineAdapter

# Path to production rules.yaml
_RULES_YAML = Path(__file__).resolve().parent.parent.parent / "configs" / "rules.yaml"


# ===========================================================================
# Test the generic RuleEngine (no LP imports inside engine)
# ===========================================================================


class TestRuleEngineGeneric:
    """Tests for the self-contained rule engine package."""

    def _engine(self):
        return RuleEngine(rules_path=_RULES_YAML)

    def test_load_rules(self):
        engine = self._engine()
        assert len(engine.rules) >= 5
        ids = [r["id"] for r in engine.rules]
        assert "restricted_zone" in ids
        assert "loitering" in ids
        assert "checkout_bypass" in ids

    def test_restricted_zone_match(self):
        engine = self._engine()
        actions = engine.evaluate("zone_entry", "RESTRICTED", {})
        assert len(actions) == 1
        assert actions[0].type == "alert"
        assert actions[0].params["alert_type"] == "ZONE_VIOLATION"
        assert actions[0].rule_id == "restricted_zone"

    def test_loitering_match(self):
        engine = self._engine()
        actions = engine.evaluate("zone_exit", "HIGH_VALUE", {"dwell_seconds": 150.0})
        assert any(a.params.get("alert_type") == "LOITERING" for a in actions)

    def test_loitering_no_match_below_threshold(self):
        engine = self._engine()
        actions = engine.evaluate("zone_exit", "HIGH_VALUE", {"dwell_seconds": 10.0})
        assert not any(a.params.get("alert_type") == "LOITERING" for a in actions)

    def test_checkout_bypass_match(self):
        engine = self._engine()
        ctx = {"visited_high_value": True, "visited_checkout": False}
        actions = engine.evaluate("zone_entry", "EXIT", ctx)
        assert any(a.params.get("alert_type") == "CHECKOUT_BYPASS" for a in actions)

    def test_checkout_bypass_no_match_when_checked_out(self):
        engine = self._engine()
        ctx = {"visited_high_value": True, "visited_checkout": True}
        actions = engine.evaluate("zone_entry", "EXIT", ctx)
        assert not any(a.params.get("alert_type") == "CHECKOUT_BYPASS" for a in actions)

    def test_repeated_visits_match(self):
        engine = self._engine()
        ctx = {"zone_visit_counts": {"r1": 4}, "region_id": "r1"}
        actions = engine.evaluate("zone_entry", "HIGH_VALUE", ctx)
        assert any(a.params.get("alert_type") == "UNUSUAL_PATH" for a in actions)

    def test_repeated_visits_no_match_below_threshold(self):
        engine = self._engine()
        ctx = {"zone_visit_counts": {"r1": 1}, "region_id": "r1"}
        actions = engine.evaluate("zone_entry", "HIGH_VALUE", ctx)
        assert not any(a.params.get("alert_type") == "UNUSUAL_PATH" for a in actions)

    def test_behavioral_analysis_escalate(self):
        engine = self._engine()
        ctx = {"zone_visit_counts": {}, "region_id": "r1"}
        actions = engine.evaluate("zone_entry", "HIGH_VALUE", ctx)
        assert any(a.type == "escalate" for a in actions)

    def test_disabled_rule_skipped(self):
        rules = [
            {
                "id": "test_disabled",
                "enabled": False,
                "trigger": {"event_type": "zone_entry", "zone_type": "RESTRICTED"},
                "conditions": [],
                "actions": [{"type": "alert", "params": {"alert_type": "TEST"}}],
            }
        ]
        engine = RuleEngine(rules=rules)
        actions = engine.evaluate("zone_entry", "RESTRICTED", {})
        assert len(actions) == 0

    def test_is_rule_enabled(self):
        engine = self._engine()
        assert engine.is_rule_enabled("loitering") is True
        assert engine.is_rule_enabled("nonexistent") is False


# ===========================================================================
# Test the LP-specific adapter (integration with Alert models)
# ===========================================================================


class FakeConfig:
    def get_rules_config(self):
        return {
            "session_timeout_seconds": 30,
        }

    def get_rules_yaml_path(self):
        return _RULES_YAML

    def get_zone_type(self, region_id):
        return None

    def get_zone_name(self, region_id):
        return None


class FakeSessionManager:
    def __init__(self):
        self._sessions = {}

    def add(self, session):
        self._sessions[session.object_id] = session

    def get_session(self, object_id):
        return self._sessions.get(object_id)


@pytest.fixture
def setup():
    config = FakeConfig()
    sm = FakeSessionManager()
    alerts = []

    async def collect(alert):
        alerts.append(alert)

    engine = RuleEngine(rules_path=_RULES_YAML)
    adapter = RuleEngineAdapter(engine, config, sm, alert_callback=collect)
    return adapter, sm, alerts


def _make_event(event_type, zone_type, object_id="42", dwell=None):
    return RegionEvent(
        event_type=event_type,
        object_id=object_id,
        region_id="r1",
        region_name="Test Region",
        zone_type=zone_type,
        timestamp=datetime.now(timezone.utc),
        dwell_seconds=dwell,
    )


@pytest.mark.asyncio
async def test_restricted_zone_immediate_alert(setup):
    adapter, sm, alerts = setup
    session = PersonSession(object_id="42", first_seen=datetime.now(timezone.utc), last_seen=datetime.now(timezone.utc))
    sm.add(session)

    event = _make_event(EventType.ENTERED, ZoneType.RESTRICTED)
    await adapter.on_event(event)

    assert len(alerts) == 1
    assert alerts[0].alert_type == AlertType.ZONE_VIOLATION
    assert alerts[0].alert_level == AlertLevel.CRITICAL


@pytest.mark.asyncio
async def test_loitering_alert(setup):
    adapter, sm, alerts = setup
    session = PersonSession(object_id="42", first_seen=datetime.now(timezone.utc), last_seen=datetime.now(timezone.utc))
    sm.add(session)

    event = _make_event(EventType.EXITED, ZoneType.HIGH_VALUE, dwell=150.0)
    await adapter.on_event(event)

    assert len(alerts) == 1
    assert alerts[0].alert_type == AlertType.LOITERING


@pytest.mark.asyncio
async def test_checkout_bypass(setup):
    adapter, sm, alerts = setup
    session = PersonSession(
        object_id="42",
        first_seen=datetime.now(timezone.utc),
        last_seen=datetime.now(timezone.utc),
        visited_high_value=True,
        visited_checkout=False,
    )
    sm.add(session)

    event = _make_event(EventType.ENTERED, ZoneType.EXIT)
    await adapter.on_event(event)

    assert len(alerts) == 1
    assert alerts[0].alert_type == AlertType.CHECKOUT_BYPASS
    assert alerts[0].alert_level == AlertLevel.WARNING


@pytest.mark.asyncio
async def test_checkout_bypass_critical_with_concealment(setup):
    adapter, sm, alerts = setup
    session = PersonSession(
        object_id="42",
        first_seen=datetime.now(timezone.utc),
        last_seen=datetime.now(timezone.utc),
        visited_high_value=True,
        visited_checkout=False,
        concealment_suspected=True,
    )
    sm.add(session)

    event = _make_event(EventType.ENTERED, ZoneType.EXIT)
    await adapter.on_event(event)

    assert len(alerts) == 1
    assert alerts[0].alert_level == AlertLevel.CRITICAL


@pytest.mark.asyncio
async def test_repeated_visits_alert(setup):
    adapter, sm, alerts = setup
    session = PersonSession(
        object_id="42",
        first_seen=datetime.now(timezone.utc),
        last_seen=datetime.now(timezone.utc),
        zone_visit_counts={"r1": 4},
    )
    sm.add(session)

    event = _make_event(EventType.ENTERED, ZoneType.HIGH_VALUE)
    await adapter.on_event(event)

    assert len(alerts) == 1
    assert alerts[0].alert_type == AlertType.UNUSUAL_PATH


@pytest.mark.asyncio
async def test_loitering_dedup_per_zone(setup):
    """Loiter alert should only fire once per zone per session."""
    adapter, sm, alerts = setup
    session = PersonSession(object_id="42", first_seen=datetime.now(timezone.utc), last_seen=datetime.now(timezone.utc))
    sm.add(session)

    event1 = _make_event(EventType.EXITED, ZoneType.HIGH_VALUE, dwell=150.0)
    await adapter.on_event(event1)
    assert len(alerts) == 1

    # Second exit from same zone should not fire again
    event2 = _make_event(EventType.EXITED, ZoneType.HIGH_VALUE, dwell=200.0)
    await adapter.on_event(event2)
    assert len(alerts) == 1  # still 1, not 2
