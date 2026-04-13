# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for RuleEngine — alert generation from region events."""

import asyncio
from datetime import datetime, timezone

import pytest

from models.events import EventType, RegionEvent, ZoneType
from models.alerts import AlertType, AlertLevel
from models.session import PersonSession
from services.rule_engine import RuleEngine


class FakeConfig:
    def get_rules_config(self):
        return {
            "loiter_threshold_seconds": 120,
            "repeat_visit_threshold": 3,
            "session_timeout_seconds": 30,
        }

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

    engine = RuleEngine(config, sm, alert_callback=collect)
    return engine, sm, alerts


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
    engine, sm, alerts = setup
    session = PersonSession(object_id="42", first_seen=datetime.now(timezone.utc), last_seen=datetime.now(timezone.utc))
    sm.add(session)

    event = _make_event(EventType.ENTERED, ZoneType.RESTRICTED)
    await engine.on_event(event)

    assert len(alerts) == 1
    assert alerts[0].alert_type == AlertType.ZONE_VIOLATION
    assert alerts[0].alert_level == AlertLevel.CRITICAL


@pytest.mark.asyncio
async def test_loitering_alert(setup):
    engine, sm, alerts = setup
    session = PersonSession(object_id="42", first_seen=datetime.now(timezone.utc), last_seen=datetime.now(timezone.utc))
    sm.add(session)

    event = _make_event(EventType.EXITED, ZoneType.HIGH_VALUE, dwell=150.0)
    await engine.on_event(event)

    assert len(alerts) == 1
    assert alerts[0].alert_type == AlertType.LOITERING


@pytest.mark.asyncio
async def test_checkout_bypass(setup):
    engine, sm, alerts = setup
    session = PersonSession(
        object_id="42",
        first_seen=datetime.now(timezone.utc),
        last_seen=datetime.now(timezone.utc),
        visited_high_value=True,
        visited_checkout=False,
    )
    sm.add(session)

    event = _make_event(EventType.ENTERED, ZoneType.EXIT)
    await engine.on_event(event)

    assert len(alerts) == 1
    assert alerts[0].alert_type == AlertType.CHECKOUT_BYPASS
    assert alerts[0].alert_level == AlertLevel.WARNING


@pytest.mark.asyncio
async def test_checkout_bypass_critical_with_concealment(setup):
    engine, sm, alerts = setup
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
    await engine.on_event(event)

    assert len(alerts) == 1
    assert alerts[0].alert_level == AlertLevel.CRITICAL


@pytest.mark.asyncio
async def test_repeated_visits_alert(setup):
    engine, sm, alerts = setup
    session = PersonSession(
        object_id="42",
        first_seen=datetime.now(timezone.utc),
        last_seen=datetime.now(timezone.utc),
        zone_visit_counts={"r1": 4},
    )
    sm.add(session)

    event = _make_event(EventType.ENTERED, ZoneType.HIGH_VALUE)
    await engine.on_event(event)

    assert len(alerts) == 1
    assert alerts[0].alert_type == AlertType.UNUSUAL_PATH


@pytest.mark.asyncio
async def test_loitering_dedup_per_zone(setup):
    """Loiter alert should only fire once per zone per session."""
    engine, sm, alerts = setup
    session = PersonSession(object_id="42", first_seen=datetime.now(timezone.utc), last_seen=datetime.now(timezone.utc))
    sm.add(session)

    event1 = _make_event(EventType.EXITED, ZoneType.HIGH_VALUE, dwell=150.0)
    await engine.on_event(event1)
    assert len(alerts) == 1

    # Second exit from same zone should not fire again
    event2 = _make_event(EventType.EXITED, ZoneType.HIGH_VALUE, dwell=200.0)
    await engine.on_event(event2)
    assert len(alerts) == 1  # still 1, not 2
