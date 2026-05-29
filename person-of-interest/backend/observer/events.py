"""Observer pattern for match-found events and alert triggering."""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Callable

from backend.domain.entities.match_result import AlertPayload

log = logging.getLogger("poi.observer")


@dataclass
class MatchFoundEvent:
    """Domain event emitted when a POI match is found."""

    alert: AlertPayload
    object_id: str
    timestamp: str
    mqtt_receive_time_ms: int = 0  # wall-clock ms when MQTT message was received


class EventBus:
    """Simple in-process event bus — Observer Pattern.

    Observers register callbacks that are invoked when events are published.
    Thread-safe: subscribe/publish may be called from different threads
    (MQTT callback thread vs FastAPI request threads).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._listeners: dict[str, list[Callable]] = {}

    def subscribe(self, event_type: str, callback: Callable) -> None:
        with self._lock:
            self._listeners.setdefault(event_type, []).append(callback)
        log.debug("Subscriber registered for %s", event_type)

    def publish(self, event_type: str, event) -> None:
        with self._lock:
            callbacks = list(self._listeners.get(event_type, []))
        for cb in callbacks:
            try:
                cb(event)
            except Exception:
                log.exception("Error in event handler for %s", event_type)

    async def publish_async(self, event_type: str, event) -> None:
        with self._lock:
            callbacks = list(self._listeners.get(event_type, []))
        for cb in callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(event)
                else:
                    cb(event)
            except Exception:
                log.exception("Error in async event handler for %s", event_type)
