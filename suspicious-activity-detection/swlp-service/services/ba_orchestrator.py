# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Behavioral Analysis Orchestrator -- owns the per-visit BA cadence.

For each (person, region) HIGH_VALUE visit we run a single asyncio task
that repeats the following "BA cycle":

    1. emit ``frames_per_request`` getimage commands evenly spread across
       ``request_interval`` seconds. Camera replies are stored in the
       behavioral-frames bucket by FrameCaptureService.
    2. publish one ``ba/requests`` message so the behavioural-analysis
       service can run a single-shot analysis on the accumulated frames.

With ``frames_per_request=5`` and ``request_interval=1.0`` this means
5 frames captured per second and one BA request fired every second.

BA itself is stateless: each request causes BA to fetch the latest K
frames from the bucket and run pose+VLM once.

Stops cleanly on:
  - explicit stop() call (driven by SceneScape EXITED event), OR
  - explicit stop_all() call (PERSON_LOST).
"""

import asyncio
from typing import Any, Dict, Protocol

import structlog

logger = structlog.get_logger(__name__)


def _compact_ts(entry_iso: str) -> str:
    """Compact ISO timestamp for SeaweedFS bucket-prefix consistency."""
    if not entry_iso:
        return ""
    return (
        entry_iso.replace(":", "")
        .replace("-", "")
        .split("+")[0]
        .split(".")[0]
    )


class _MQTT(Protocol):
    def publish_raw(self, topic: str, payload: str) -> None: ...


class _SessionManager(Protocol):
    def get_session(self, object_id: str, scene_id: str = "") -> Any: ...


class _BAPublisher(Protocol):
    def publish_request(
        self, *, person_id: str, region_id: str,
        entry_timestamp: str, scene_id: str,
    ) -> None: ...


class BehavioralAnalysisOrchestrator:
    """Owns BA visit tasks and per-visit frame-capture + request cadence."""

    def __init__(
        self,
        mqtt_service: _MQTT,
        session_manager: _SessionManager,
        ba_publisher: _BAPublisher,
        config,
        frames_per_request: int = 5,
        request_interval: float = 1.0,
        ba_initial_delay: float = 0.0,
    ) -> None:
        self._mqtt = mqtt_service
        self._sessions = session_manager
        self._ba = ba_publisher
        self._config = config
        self._frames_per_request = max(int(frames_per_request), 1)
        self._request_interval = max(float(request_interval), 0.05)
        self._frame_interval = self._request_interval / self._frames_per_request
        self._initial_delay = float(ba_initial_delay)

        # Active per-visit tasks, keyed by "{object_id}:{region_id}"
        self._tasks: Dict[str, asyncio.Task] = {}

        logger.info(
            "BehavioralAnalysisOrchestrator initialized",
            frames_per_request=self._frames_per_request,
            request_interval=self._request_interval,
            frame_interval=self._frame_interval,
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
            frames_per_request=self._frames_per_request,
            request_interval=self._request_interval,
        )

        try:
            if self._initial_delay > 0:
                await asyncio.sleep(self._initial_delay)

            while True:
                session = self._sessions.get_session(object_id, scene_id=scene_id)
                if not session:
                    return

                # 1) Emit N getimage commands across the interval. Camera
                #    replies land in FrameCaptureService and are stored in
                #    the behavioral-frames bucket.
                for _ in range(self._frames_per_request):
                    cams = list(session.current_cameras)
                    for cam in cams:
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

                # 2) After the batch of frames, publish exactly one BA
                #    request so BA processes the latest window once.
                entry_ts_iso = session.current_zones.get(region_id, "")
                try:
                    self._ba.publish_request(
                        person_id=object_id,
                        region_id=region_id,
                        entry_timestamp=_compact_ts(entry_ts_iso),
                        scene_id=scene_id,
                    )
                except Exception:
                    logger.exception(
                        "ba/requests publish failed",
                        object_id=object_id, region_id=region_id,
                    )

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
