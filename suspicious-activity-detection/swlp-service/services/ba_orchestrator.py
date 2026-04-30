# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Behavioral Analysis Orchestrator -- owns the per-visit getimage cadence.

For each (person, region) HIGH_VALUE visit we run a single asyncio task that
publishes ``getimage`` to every camera seeing the person at ``analysis_fps``.
Camera replies are stored in the ``behavioral-frames`` bucket by FrameManager;
on each successful frame store, swlp-service publishes one ``ba/requests``
message so the behavioural-analysis service can run a single-shot analysis.

There is no visit-lifecycle ``start``/``exit`` payload anymore: BA is
stateless and reacts to every published request independently.

Stops cleanly on:
  - explicit stop() call (driven by SceneScape EXITED event), OR
  - explicit stop_all() call (PERSON_LOST).
"""

import asyncio
from typing import Any, Dict, Protocol

import structlog

logger = structlog.get_logger(__name__)


class _MQTT(Protocol):
    def publish_raw(self, topic: str, payload: str) -> None: ...


class _SessionManager(Protocol):
    def get_session(self, object_id: str, scene_id: str = "") -> Any: ...


class BehavioralAnalysisOrchestrator:
    """Owns BA visit tasks and per-visit frame-capture cadence."""

    def __init__(
        self,
        mqtt_service: _MQTT,
        session_manager: _SessionManager,
        analysis_fps: float = 5.0,
        ba_initial_delay: float = 0.0,
    ) -> None:
        self._mqtt = mqtt_service
        self._sessions = session_manager
        self._analysis_fps = max(float(analysis_fps), 0.1)
        self._frame_interval = 1.0 / self._analysis_fps
        self._initial_delay = float(ba_initial_delay)

        # Active per-visit tasks, keyed by "{object_id}:{region_id}"
        self._tasks: Dict[str, asyncio.Task] = {}

        logger.info(
            "BehavioralAnalysisOrchestrator initialized",
            analysis_fps=self._analysis_fps,
            initial_delay=self._initial_delay,
        )

    # ---- public API ----------------------------------------------------------

    def start(self, object_id: str, region_id: str, scene_id: str) -> None:
        """Begin a frame-capture task for one HV-zone visit.

        Idempotent: re-entry events from re-id flicker are ignored if the
        existing task is still alive.
        """
        key = self._key(object_id, region_id)
        prev = self._tasks.get(key)
        if prev and not prev.done():
            logger.debug(
                "BA visit task already active, ignoring re-start",
                object_id=object_id, region_id=region_id,
            )
            return
        self._tasks[key] = asyncio.create_task(
            self._run(object_id, region_id, scene_id)
        )

    def stop(self, object_id: str, region_id: str) -> None:
        """Cancel the visit task for one (person, region) pair."""
        key = self._key(object_id, region_id)
        task = self._tasks.pop(key, None)
        if task and not task.done():
            task.cancel()

    def stop_all(self, object_id: str) -> None:
        """Cancel every visit task for a person (used on PERSON_LOST)."""
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
        logger.info(
            "BA visit task started",
            object_id=object_id,
            region_id=region_id,
            analysis_fps=self._analysis_fps,
        )

        try:
            if self._initial_delay > 0:
                await asyncio.sleep(self._initial_delay)

            while True:
                session = self._sessions.get_session(object_id, scene_id=scene_id)
                if not session:
                    return

                # Trigger frame capture (camera reply lands in FrameManager,
                # which stores the frame and triggers ba/requests publish).
                for cam in list(session.current_cameras):
                    try:
                        self._mqtt.publish_raw(
                            f"scenescape/cmd/camera/{cam}", "getimage"
                        )
                    except Exception:
                        logger.exception(
                            "getimage publish failed",
                            object_id=object_id, camera=cam,
                        )

                await asyncio.sleep(self._frame_interval)

        except asyncio.CancelledError:
            logger.debug(
                "BA visit task cancelled",
                object_id=object_id, region_id=region_id,
            )
            raise
        except Exception:
            logger.exception(
                "BA visit task crashed",
                object_id=object_id, region_id=region_id,
            )
        finally:
            self._tasks.pop(self._key(object_id, region_id), None)
