# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Frame Capture Service — explicit start/stop frame capture per person+zone.

Replaces the implicit frame_request_loop + on_camera_image wiring in main.py.
Owns the full lifecycle:
  start_capture(person_id, region_id, ...) → begins requesting frames at N FPS
  stop_capture(person_id, region_id)       → stops requesting, schedules frame cleanup
"""

import asyncio
import base64
from datetime import datetime, timezone
from typing import Callable, Optional

import structlog

from .config import ConfigService
from .frame_manager import FrameManager
from .session_manager import SessionManager

logger = structlog.get_logger(__name__)


class _CaptureState:
    """Per-capture tracking: task, entry_timestamp, frame count, notify flag."""
    __slots__ = ("task", "entry_timestamp", "frame_count", "ba_notified")

    def __init__(self, task: asyncio.Task, entry_timestamp: str) -> None:
        self.task = task
        self.entry_timestamp = entry_timestamp
        self.frame_count: int = 0
        self.ba_notified: bool = False


class FrameCaptureService:
    """Manages per-person frame capture with explicit start/stop lifecycle."""

    def __init__(
        self,
        config: ConfigService,
        frame_manager: FrameManager,
        session_manager: SessionManager,
        mqtt_service=None,
        min_frames_for_ba: int = 3,
    ) -> None:
        self._config = config
        self._frame_mgr = frame_manager
        self._session_mgr = session_manager
        self._mqtt = mqtt_service
        self._min_frames = min_frames_for_ba

        rules_cfg = config.get_rules_config()
        analysis_fps = int(rules_cfg.get("behavioural_analysis_fps", 1))
        self._frame_interval = 1.0 / max(analysis_fps, 1)

        # Active captures: {"{person_id}:{region_id}" → _CaptureState}
        self._active_captures: dict[str, _CaptureState] = {}

        # Callback invoked when a batch reaches min_frames
        self._on_batch_ready: Optional[Callable] = None

        logger.info(
            "FrameCaptureService initialized",
            frame_interval=self._frame_interval,
            min_frames_for_ba=self._min_frames,
        )

    def set_on_batch_ready(self, callback: Callable) -> None:
        """Register callback(person_id, region_id, scene_id, entry_timestamp)."""
        self._on_batch_ready = callback

    def start_capture(
        self,
        person_id: str,
        region_id: str,
        scene_id: str = "",
        entry_timestamp: str = "",
    ) -> None:
        """Start capturing frames for a person in a zone."""
        capture_key = f"{person_id}:{region_id}"
        if capture_key in self._active_captures:
            logger.debug("Capture already active", capture_key=capture_key)
            return

        task = asyncio.create_task(
            self._capture_loop(person_id, region_id, scene_id, entry_timestamp)
        )
        self._active_captures[capture_key] = _CaptureState(task, entry_timestamp)
        logger.info(
            "Frame capture STARTED",
            person_id=person_id,
            region_id=region_id,
            scene_id=scene_id,
            fps=round(1.0 / self._frame_interval),
        )

    def stop_capture(self, person_id: str, region_id: str) -> None:
        """Stop capturing frames for a person in a zone."""
        capture_key = f"{person_id}:{region_id}"
        state = self._active_captures.pop(capture_key, None)
        if state:
            if not state.task.done():
                state.task.cancel()
            logger.info(
                "Frame capture STOPPED",
                person_id=person_id,
                region_id=region_id,
            )

    def is_capturing(self, person_id: str, region_id: str) -> bool:
        capture_key = f"{person_id}:{region_id}"
        return capture_key in self._active_captures

    def get_active_captures(self) -> list[str]:
        return list(self._active_captures.keys())

    def reset_batch(self, person_id: str, region_id: str) -> None:
        """Reset frame counter after BA processes a batch, allowing next cycle."""
        capture_key = f"{person_id}:{region_id}"
        state = self._active_captures.get(capture_key)
        if state:
            state.frame_count = 0
            state.ba_notified = False
            logger.debug(
                "Batch reset — ready for next analysis cycle",
                person_id=person_id,
                region_id=region_id,
            )

    # ---- internal capture loop -----------------------------------------------

    async def _capture_loop(
        self,
        person_id: str,
        region_id: str,
        scene_id: str,
        entry_timestamp: str,
    ) -> None:
        """Request and store frames at configured FPS until cancelled."""
        capture_key = f"{person_id}:{region_id}"
        try:
            while True:
                session = self._session_mgr.get_session(person_id, scene_id=scene_id)
                if not session:
                    logger.warning(
                        "Capture loop ending — session not found",
                        person_id=person_id,
                        scene_id=scene_id,
                    )
                    break
                if region_id not in session.current_zones:
                    logger.info(
                        "Capture loop ending — person left zone",
                        person_id=person_id,
                        region_id=region_id,
                        current_zones=list(session.current_zones.keys()),
                    )
                    break

                # Request image from cameras that see this person
                if self._mqtt and session.current_cameras:
                    for cam in session.current_cameras:
                        self._mqtt.publish_raw(
                            f"scenescape/cmd/camera/{cam}", "getimage"
                        )
                    logger.debug(
                        "Frame requested",
                        person_id=person_id,
                        cameras=session.current_cameras,
                    )
                else:
                    logger.debug(
                        "No cameras for person",
                        person_id=person_id,
                        has_mqtt=bool(self._mqtt),
                        cameras=session.current_cameras,
                    )

                await asyncio.sleep(self._frame_interval)
        except asyncio.CancelledError:
            pass
        finally:
            self._active_captures.pop(capture_key, None)

    # ---- camera image handler (called by MQTT) -------------------------------

    async def on_camera_image(self, camera_name: str, data: dict) -> None:
        """Store frame for any person with an active capture on this camera."""
        if not self._active_captures:
            return

        image_b64 = data.get("image", data.get("data", ""))
        if not image_b64:
            logger.debug("Camera image has no data", camera=camera_name)
            return
        image_bytes = base64.b64decode(image_b64)
        ts = datetime.now(timezone.utc)

        stored_any = False
        for session in self._session_mgr.get_all_sessions().values():
            if camera_name not in session.current_cameras:
                continue

            # Only store if there's an active capture for this person
            for zone_id in session.current_zones:
                capture_key = f"{session.object_id}:{zone_id}"
                state = self._active_captures.get(capture_key)
                if state is None:
                    continue

                key = self._frame_mgr.store_person_frame(
                    session.object_id, image_bytes, ts,
                    region_id=zone_id,
                    entry_timestamp=state.entry_timestamp,
                    scene_id=session.scene_id,
                )
                session.add_frame_key(key)
                stored_any = True

                # Track batch and notify when min_frames reached
                state.frame_count += 1
                if (
                    state.frame_count >= self._min_frames
                    and not state.ba_notified
                    and self._on_batch_ready
                ):
                    state.ba_notified = True
                    logger.info(
                        "Batch ready — triggering BA request",
                        person_id=session.object_id,
                        region_id=zone_id,
                        frame_count=state.frame_count,
                    )
                    self._on_batch_ready(
                        session.object_id, zone_id,
                        session.scene_id, state.entry_timestamp,
                    )

        if not stored_any:
            logger.debug(
                "Camera image received but no matching capture",
                camera=camera_name,
                active_captures=list(self._active_captures.keys()),
            )

    # ---- shutdown ------------------------------------------------------------

    async def stop_all(self) -> None:
        """Cancel all active capture tasks."""
        for capture_key, state in list(self._active_captures.items()):
            state.task.cancel()
        self._active_captures.clear()
        logger.info("All frame captures stopped")
