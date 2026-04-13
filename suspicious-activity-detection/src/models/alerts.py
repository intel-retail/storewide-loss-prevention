# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Alert data models for loss prevention."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import uuid


class AlertType(str, Enum):
    CONCEALMENT = "CONCEALMENT"
    CHECKOUT_BYPASS = "CHECKOUT_BYPASS"
    LOITERING = "LOITERING"
    UNUSUAL_PATH = "UNUSUAL_PATH"
    ZONE_VIOLATION = "ZONE_VIOLATION"


class AlertLevel(str, Enum):
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    """A suspicious-activity alert produced by the system."""
    alert_type: AlertType
    alert_level: AlertLevel
    object_id: str
    timestamp: datetime
    region_id: Optional[str] = None
    region_name: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    evidence_keys: List[str] = field(default_factory=list)  # MinIO keys
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "alert_level": self.alert_level.value,
            "object_id": self.object_id,
            "timestamp": self.timestamp.isoformat(),
            "region_id": self.region_id,
            "region_name": self.region_name,
            "details": self.details,
            "evidence_keys": self.evidence_keys,
        }
