# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Frame capture service.

Receives camera-image MQTT messages, decides which active HIGH_VALUE-zone
sessions the frame belongs to, and stores the cropped frame to SeaweedFS
via ``FrameManager``.

This service does NOT publish ba/requests -- the
``BehavioralAnalysisOrchestrator`` owns the BA cadence and emits one
ba/requests per batch of stored frames.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


class FrameCaptureService:
    """Glue between camera image events and frame storage."""

    def __init__(
        self,
        config,
        session_manager,
        frame_manager,
    ) -> None:
        self._config = config
        self._sessions = session_manager
        self._frame_mgr = frame_manager

    async def on_camera_image(self, camera_name: str, data: dict) -> None:
        """Handle a fresh image from one camera.

        For each active session whose person is currently in a HIGH_VALUE
        zone visible to ``camera_name``, store the frame.
        """
        image_b64 = data.get("image", data.get("data", ""))
        if not image_b64:
            return

        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception:
            logger.exception("Invalid base64 image", camera=camera_name)
            return

        ts = datetime.now(timezone.utc)

        for session in self._sessions.get_all_sessions().values():
            if camera_name not in session.current_cameras:
                continue

            # Find the HIGH_VALUE zone the person is in (if any).
            zone_id: Optional[str] = None
            for zid in session.current_zones:
                if self._config.get_zone_type(zid) == "HIGH_VALUE":
                    zone_id = zid
                    break
            if zone_id is None:
                continue

            entry_ts_iso = session.current_zones.get(zone_id, "")
            try:
                key = self._frame_mgr.store_person_frame(
                    session.object_id, image_bytes, ts,
                    region_id=zone_id,
                    entry_timestamp=entry_ts_iso,
                    scene_id=session.scene_id,
                )
            except Exception:
                logger.exception(
                    "Failed to store person frame",
                    person_id=session.object_id, region_id=zone_id,
                )
                continue
            session.add_frame_key(key)
