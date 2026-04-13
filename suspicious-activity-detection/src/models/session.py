# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Person session data model for loss prevention tracking."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set


@dataclass
class RegionVisit:
    """Record of a person visiting a specific region."""
    region_id: str
    region_name: str
    zone_type: str
    entry_time: datetime
    exit_time: Optional[datetime] = None

    @property
    def duration_seconds(self) -> float:
        end = self.exit_time or datetime.now(timezone.utc)
        return (end - self.entry_time).total_seconds()


@dataclass
class PersonSession:
    """
    Live state of a tracked person in the store.

    Created when SceneScape first reports an object_id.
    Updated on every subsequent scene/region event.
    Expired when the ID is absent for longer than session_timeout.

    Fields aligned with the Store-wide Loss Prevention specification:
      object_id          — SceneScape's persistent person identifier
      first_seen         — Session creation timestamp
      last_seen          — Last activity timestamp for expiry
      visited_checkout   — Whether the person entered any checkout zone
      visited_high_value — Whether the person entered any high-value zone
      zone_visit_counts  — Count of entries per region_id
      current_zones      — Currently occupied zones with entry timestamps
      loiter_alerted     — Tracks if a loiter alert has already been triggered per zone
      concealment_suspected — Set to true if behavioral analysis confirms suspicious behavior
    """
    object_id: str
    first_seen: datetime
    last_seen: datetime

    # Current position
    current_cameras: List[str] = field(default_factory=list)
    bbox: Optional[Dict] = None  # {x, y, w, h} on primary camera

    # Current zones: {region_id: entry_timestamp_iso}
    current_zones: Dict[str, str] = field(default_factory=dict)

    # History
    camera_history: List[str] = field(default_factory=list)
    region_visits: List[RegionVisit] = field(default_factory=list)

    # Zone visit counts: {region_id: count}
    zone_visit_counts: Dict[str, int] = field(default_factory=dict)

    # Behavioral flags
    visited_checkout: bool = False
    visited_exit: bool = False
    visited_high_value: bool = False
    concealment_suspected: bool = False

    # Loiter alert tracking: {region_id: True} — prevents duplicate alerts
    loiter_alerted: Dict[str, bool] = field(default_factory=dict)

    # Frame references (SeaweedFS keys for rolling buffer — cropped person frames)
    frame_buffer: List[str] = field(default_factory=list)
    max_frame_buffer: int = 20  # ~10s at 2fps per spec

    def get_open_visits(self) -> List[RegionVisit]:
        """Return region visits that have not been closed."""
        return [v for v in self.region_visits if v.exit_time is None]

    def close_visit(self, region_id: str, exit_time: datetime) -> Optional[RegionVisit]:
        """Close an open visit for a given region."""
        for visit in self.region_visits:
            if visit.region_id == region_id and visit.exit_time is None:
                visit.exit_time = exit_time
                return visit
        return None

    def add_frame_key(self, key: str) -> None:
        """Append a frame key, evicting the oldest if buffer is full."""
        self.frame_buffer.append(key)
        if len(self.frame_buffer) > self.max_frame_buffer:
            self.frame_buffer.pop(0)

    def is_in_zone(self, region_id: str) -> bool:
        """Check if the person is currently in a specific zone."""
        return region_id in self.current_zones

    def enter_zone(self, region_id: str, timestamp: datetime) -> None:
        """Record zone entry."""
        self.current_zones[region_id] = timestamp.isoformat()
        self.zone_visit_counts[region_id] = self.zone_visit_counts.get(region_id, 0) + 1

    def exit_zone(self, region_id: str) -> Optional[str]:
        """Record zone exit. Returns the entry timestamp if was present."""
        return self.current_zones.pop(region_id, None)
