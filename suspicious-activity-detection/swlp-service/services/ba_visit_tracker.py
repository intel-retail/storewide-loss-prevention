# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Per-visit accounting for BA request/result lifecycle.

A "visit" is a unique tuple of (scene_id, person_id, region_id,
entry_timestamp). For each visit we track:

* ``requests_sent``     -- count of ``ba/requests`` published by the
  orchestrator for this visit.
* ``results_received``  -- count of ``ba/results`` received for this
  visit (any status).
* ``exited``            -- True once the rule adapter has seen the
  HV-zone EXIT event for the visit.
* ``alerted``           -- True once the visit has produced at least one
  ``suspicious`` result that was acted on (frames copied to alert
  bucket). Used to suppress (a) duplicate alert-frame copies and (b)
  bucket cleanup -- a suspicious visit's frames stay in the BA bucket
  as evidence.

The cleanup rule (see RuleEngineAdapter.on_ba_result) is:

    if status in {"no_match", "no_enough_data"}:
        if exited and requests_sent == results_received and not alerted:
            cleanup_visit(...)

EXITED handler also calls ``check_drained`` to catch the case where the
last result arrived before EXIT was processed.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


VisitKey = Tuple[str, str, str, str]  # (scene_id, person_id, region_id, entry_timestamp)


@dataclass
class _VisitState:
    requests_sent: int = 0
    results_received: int = 0
    exited: bool = False
    alerted: bool = False


class BAVisitTracker:
    """Thread-safe per-visit counters."""

    def __init__(self) -> None:
        self._visits: Dict[VisitKey, _VisitState] = {}
        self._lock = threading.Lock()

    @staticmethod
    def make_key(
        scene_id: str, person_id: str, region_id: str, entry_timestamp: str,
    ) -> VisitKey:
        return (scene_id, person_id, region_id, entry_timestamp)

    # ---- mutators --------------------------------------------------------

    def note_request(self, key: VisitKey) -> None:
        with self._lock:
            state = self._visits.setdefault(key, _VisitState())
            state.requests_sent += 1

    def note_result(self, key: VisitKey) -> None:
        with self._lock:
            state = self._visits.setdefault(key, _VisitState())
            state.results_received += 1

    def mark_exited(self, key: VisitKey) -> None:
        with self._lock:
            state = self._visits.setdefault(key, _VisitState())
            state.exited = True

    def mark_alerted(self, key: VisitKey) -> bool:
        """Set the alerted flag. Returns True if this is the first time."""
        with self._lock:
            state = self._visits.setdefault(key, _VisitState())
            if state.alerted:
                return False
            state.alerted = True
            return True

    def forget(self, key: VisitKey) -> None:
        with self._lock:
            self._visits.pop(key, None)

    # ---- queries ---------------------------------------------------------

    def is_alerted(self, key: VisitKey) -> bool:
        with self._lock:
            state = self._visits.get(key)
            return bool(state and state.alerted)

    def is_drained(self, key: VisitKey) -> bool:
        """True iff EXITED, request/result counts match, and not alerted."""
        with self._lock:
            state = self._visits.get(key)
            if state is None:
                return False
            return (
                state.exited
                and state.requests_sent == state.results_received
                and state.requests_sent > 0
                and not state.alerted
            )

    def snapshot(self, key: VisitKey) -> Optional[_VisitState]:
        with self._lock:
            state = self._visits.get(key)
            if state is None:
                return None
            return _VisitState(
                requests_sent=state.requests_sent,
                results_received=state.results_received,
                exited=state.exited,
                alerted=state.alerted,
            )
