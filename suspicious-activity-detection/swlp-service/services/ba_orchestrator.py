# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Behavioral Analysis Orchestrator -- owns the per-visit BA pipeline.

For each (person, region) HIGH_VALUE visit, runs a single asyncio task that:
  1. Periodically requests frames from cameras seeing the person (`getimage`).
  2. At a slower cadence, publishes a BA analysis request to `ba/requests`.

Stops cleanly on:
  - explicit stop() call (zone exit / person lost / shutdown), OR
  - person leaves the zone (detected from session state), OR
  - ba_alerted[region_id] is True (concealment confirmed → no need to keep asking).

Decoupling goals:
  - rule_adapter does NOT know about MQTT topics, frame fps, BA cadence.
  - frame_manager continues to own bucket writes (independent MQTT consumer).
"""

import asyncio
import time
from typing import Any, Dict, Optional, Protocol

import structlog

logger = structlog.get_logger(__name__)


class _BAPublisher(Protocol):
    def publish_request(
        self, person_id: str, region_id: str, entry_timestamp: str,
        scene_id: str = "",
    ) -> None: ...


class _MQTT(Protocol):
    def publish_raw(self, topic: str, payload: str) -> None: ...


class _SessionManager(Protocol):
    def get_session(self, object_id: str, scene_id: str = "") -> Any: ...


class BehavioralAnalysisOrchestrator:
    """Owns BA visit tasks and per-visit frame/request cadence."""

    def __init__(
        self,
        ba_publisher: _BAPublisher,
        mqtt_service: _MQTT,
        session_manager: _SessionManager,
        analysis_fps: float = 5.0,
        ba_poll_interval: float = 1.0,
        ba_initial_delay: float = 2.0,
    ) -> None:
        self._ba = ba_publisher
        self._mqtt = mqtt_service
        self._sessions = session_manager
        self._analysis_fps = max(float(analysis_fps), 0.1)
        self._frame_interval = 1.0 / self._analysis_fps
        self._ba_poll_interval = float(ba_poll_interval)
        self._initial_delay = float(ba_initial_delay)

        # Active per-visit tasks, keyed by "{object_id}:{region_id}"
        self._tasks: Dict[str, asyncio.Task] = {}

        logger.info(
            "BehavioralAnalysisOrchestrator initialized",
            analysis_fps=self._analysis_fps,
            ba_poll_interval=self._ba_poll_interval,
            ba_initial_delay=self._initial_delay,
        )

    # ---- public API ----------------------------------------------------------

    def start(self, object_id: str, region_id: str, scene_id: str) -> None:
        """Start a BA visit task for one HV-zone visit.

        If a task is already running for this canonical person+region, leave
        it alone — repeated ENTERED events from re-id flicker would otherwise
        cancel it and reset the initial delay, preventing BA from ever firing.
        """
        key = self._key(object_id, region_id)
        prev = self._tasks.get(key)
        if prev and not prev.done():
            logger.debug("BA visit task already active, ignoring re-start",
                         object_id=object_id, region_id=region_id)
            return
        self._tasks[key] = asyncio.create_task(
            self._run(object_id, region_id, scene_id)
        )

    def stop(self, object_id: str, region_id: str) -> None:
        """Schedule cancellation of the BA visit task after a short grace period.

        SceneScape can emit spurious EXITED events while a person is still
        physically in the region (track UUID flicker). Cancelling immediately
        would abort BA mid-analysis and let the next ENTERED restart its 2 s
        initial delay, so concealment never gets evaluated. We instead let
        the running task notice ``region_id not in current_zones`` on its own,
        which is itself debounced via ``_run``'s grace window.
        """
        # Intentionally a no-op: the running task self-terminates after the
        # absence-grace window in ``_run`` if the person truly left.
        return

    def stop_all(self, object_id: str) -> None:
        """Cancel every BA visit task for a person (used on PERSON_LOST)."""
        prefix = f"{object_id}:"
        for key in [k for k in self._tasks if k.startswith(prefix)]:
            task = self._tasks.pop(key, None)
            if task and not task.done():
                task.cancel()

    def active_count(self) -> int:
        return sum(1 for t in self._tasks.values() if not t.done())

    # ---- internals -----------------------------------------------------------

    @staticmethod
    def _key(object_id: str, region_id: str) -> str:
        return f"{object_id}:{region_id}"

    async def _run(self, object_id: str, region_id: str, scene_id: str) -> None:
        next_ba_at = time.monotonic() + self._initial_delay
        # Grace window for absence: lets us survive SceneScape track-id
        # flicker (where a person is briefly missing from current_zones).
        absence_grace = max(self._initial_delay + 1.0, 5.0)
        absent_since: Optional[float] = None
        logger.info(
            "BA visit task started",
            object_id=object_id,
            region_id=region_id,
            analysis_fps=self._analysis_fps,
            ba_interval=self._ba_poll_interval,
            absence_grace=absence_grace,
        )
        try:
            while True:
                session = self._sessions.get_session(object_id, scene_id=scene_id)
                if not session:
                    return
                if region_id not in session.current_zones:
                    if absent_since is None:
                        absent_since = time.monotonic()
                    elif time.monotonic() - absent_since >= absence_grace:
                        return
                    await asyncio.sleep(self._frame_interval)
                    continue
                absent_since = None
                if session.ba_alerted.get(region_id):
                    return

                # Skip while re-id is still provisional: this canonical may be
                # superseded once previous_ids_chain links it to an existing
                # session. Avoids generating BA requests + frame uploads for
                # short-lived ghost canonicals.
                if session.reid_state and session.reid_state != "matched":
                    await asyncio.sleep(self._frame_interval)
                    continue

                # 1. Trigger frame capture (camera reply lands in FrameManager).
                for cam in list(session.current_cameras):
                    try:
                        self._mqtt.publish_raw(
                            f"scenescape/cmd/camera/{cam}", "getimage"
                        )
                    except Exception:
                        logger.exception(
                            "getimage publish failed",
                            object_id=object_id,
                            camera=cam,
                        )

                # 2. Publish BA request at the slower cadence.
                now = time.monotonic()
                if now >= next_ba_at:
                    self._publish_request(session, object_id, region_id)
                    next_ba_at = now + self._ba_poll_interval

                await asyncio.sleep(self._frame_interval)
        except asyncio.CancelledError:
            logger.debug(
                "BA visit task cancelled",
                object_id=object_id,
                region_id=region_id,
            )
            raise
        except Exception:
            logger.exception(
                "BA visit task crashed",
                object_id=object_id,
                region_id=region_id,
            )
        finally:
            self._tasks.pop(self._key(object_id, region_id), None)

    def _publish_request(self, session: Any, object_id: str, region_id: str) -> None:
        # Skip if already alerted (also re-checked by the loop, but cheap).
        if session.ba_alerted.get(region_id):
            return
        entry_ts_iso = session.current_zones.get(region_id, "")
        entry_timestamp = ""
        if entry_ts_iso:
            entry_timestamp = (
                entry_ts_iso.replace(":", "")
                .replace("-", "")
                .split("+")[0]
                .split(".")[0]
            )
        self._ba.publish_request(
            person_id=object_id,
            region_id=region_id,
            entry_timestamp=entry_timestamp,
            scene_id=session.scene_id,
        )
