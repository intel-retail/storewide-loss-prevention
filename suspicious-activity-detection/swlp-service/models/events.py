# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Event types emitted by the Session Manager."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class EventType(str, Enum):
    ENTERED = "ENTERED"
    EXITED = "EXITED"
    PERSON_LOST = "PERSON_LOST"


class ZoneType(str, Enum):
    HIGH_VALUE = "HIGH_VALUE"
    CHECKOUT = "CHECKOUT"
    EXIT = "EXIT"
    RESTRICTED = "RESTRICTED"


@dataclass
class RegionEvent:
    """Event produced when a person enters or exits a region."""
    event_type: EventType
    object_id: str
    region_id: str
    region_name: str
    zone_type: ZoneType
    timestamp: datetime
    dwell_seconds: Optional[float] = None   # populated on EXIT
    minio_thumbnail_key: Optional[str] = None
