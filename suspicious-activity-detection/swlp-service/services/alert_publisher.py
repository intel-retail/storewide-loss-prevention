# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Alert Publisher — receives Alert objects and distributes them to
MQTT, the in-memory alert store (for REST API), and logs.
"""

import json
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional

import structlog

from models.alerts import Alert
from .config import ConfigService
from .mqtt_service import MQTTService

logger = structlog.get_logger(__name__)


class AlertPublisher:
    """
    Publishes alerts to:
      1. MQTT topic  lp/alerts/{alert_type}
      2. In-memory ring buffer (available via REST API)
      3. Structured log
    """

    def __init__(
        self,
        config: ConfigService,
        mqtt_service: MQTTService,
        max_history: int = 500,
    ) -> None:
        self.config = config
        self.mqtt = mqtt_service
        self.prefix = config.get_alert_topic_prefix()
        self._history: Deque[Alert] = deque(maxlen=max_history)

        logger.info("AlertPublisher initialized", topic_prefix=self.prefix)

    # ---- publish -------------------------------------------------------------
    async def publish(self, alert: Alert) -> None:
        """Distribute an alert to all channels."""
        # 1. In-memory store
        self._history.append(alert)

        # 2. MQTT
        topic = f"{self.prefix}/{alert.alert_type.value}"
        self.mqtt.publish(topic, alert.to_dict())

        # 3. Log
        logger.warning(
            "ALERT",
            alert_id=alert.alert_id,
            type=alert.alert_type.value,
            level=alert.alert_level.value,
            object_id=alert.object_id,
            region=alert.region_name,
            details=alert.details,
        )

    # ---- query ---------------------------------------------------------------
    def get_recent(self, limit: int = 50) -> List[Dict]:
        """Return the most recent alerts as dicts."""
        items = list(self._history)[-limit:]
        return [a.to_dict() for a in reversed(items)]

    def get_by_type(self, alert_type: str, limit: int = 50) -> List[Dict]:
        filtered = [a for a in self._history if a.alert_type.value == alert_type]
        return [a.to_dict() for a in filtered[-limit:]]

    def get_by_person(self, object_id: str) -> List[Dict]:
        return [a.to_dict() for a in self._history if a.object_id == object_id]

    @property
    def total_count(self) -> int:
        return len(self._history)
