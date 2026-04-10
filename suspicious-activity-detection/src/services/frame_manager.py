# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Frame Manager — manages cropped person frames in SeaweedFS (S3-compatible).

Bucket structure:
  loss-prevention-frames/
  ├── {object_id}/
  │   ├── {timestamp_1}.jpg    # Cropped person frame
  │   ├── {timestamp_2}.jpg
  │   └── ...                  # Rolling buffer of last 20 frames (~10s at 2fps)
  └── alerts/
      └── {alert_id}/
          └── evidence/        # Frames sent to behavioral analysis, retained for audit

Only stores cropped person frames for individuals currently in HIGH_VALUE zones.
Storage rate: 2 fps per person in a high-value zone.
Rolling buffer: 20 frames per person.
"""

import base64
import io
from datetime import datetime, timezone
from typing import Dict, List, Optional

import structlog

from .config import ConfigService

logger = structlog.get_logger(__name__)

try:
    from minio import Minio
    from minio.error import S3Error
except ImportError:
    Minio = None
    S3Error = Exception
    logger.warning("minio package not installed — FrameManager will be no-op")


class FrameManager:
    """
    Manages cropped person frames in SeaweedFS via S3-compatible API.

    Only stores crops for persons in HIGH_VALUE zones.
    Maintains a rolling buffer of 20 frames per person.
    """

    BUCKET = "loss-prevention-frames"
    ROLLING_BUFFER_SIZE = 20  # ~10 seconds at 2fps
    ALERT_EVIDENCE_PREFIX = "alerts"

    def __init__(self, config: ConfigService) -> None:
        seaweed_cfg = config.get_seaweedfs_config()
        self.endpoint = seaweed_cfg.get("endpoint", "seaweedfs:8333")
        self.access_key = seaweed_cfg.get("access_key", "")
        self.secret_key = seaweed_cfg.get("secret_key", "")
        self.secure = seaweed_cfg.get("secure", False)
        self.retention_hours = seaweed_cfg.get("evidence_retention_hours", 24)
        self.exit_retention_seconds = seaweed_cfg.get("exit_retention_seconds", 60)

        # Per-person key tracking for rolling buffer management
        self._person_keys: Dict[str, List[str]] = {}

        self.client: Optional["Minio"] = None
        if Minio:
            self.client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure,
            )

        logger.info(
            "FrameManager initialized",
            endpoint=self.endpoint,
            bucket=self.BUCKET,
            buffer_size=self.ROLLING_BUFFER_SIZE,
        )

    async def ensure_bucket(self) -> None:
        """Create the frame bucket if it doesn't exist. Retries on connection failure."""
        if not self.client:
            return
        import asyncio
        for attempt in range(5):
            try:
                if not self.client.bucket_exists(self.BUCKET):
                    self.client.make_bucket(self.BUCKET)
                    logger.info("Created bucket", bucket=self.BUCKET)
                else:
                    logger.info("Bucket exists", bucket=self.BUCKET)
                return
            except Exception:
                if attempt < 4:
                    wait = 2 * (attempt + 1)
                    logger.warning(
                        "SeaweedFS not ready, retrying",
                        attempt=attempt + 1,
                        wait=wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.exception("Bucket check/create failed after retries", bucket=self.BUCKET)

    # ---- Store cropped person frame ------------------------------------------
    def store_person_frame(
        self, object_id: str, image_bytes: bytes, ts: Optional[datetime] = None
    ) -> str:
        """
        Store a cropped person frame in the rolling buffer.
        Evicts the oldest frame if the buffer exceeds ROLLING_BUFFER_SIZE.
        Returns the SeaweedFS object key.
        """
        ts = ts or datetime.now(timezone.utc)
        key = f"{object_id}/{ts.strftime('%Y%m%dT%H%M%S_%f')}.jpg"
        self._put(key, image_bytes)

        # Track keys for rolling buffer management
        if object_id not in self._person_keys:
            self._person_keys[object_id] = []
        self._person_keys[object_id].append(key)

        # Evict oldest if over buffer size
        while len(self._person_keys[object_id]) > self.ROLLING_BUFFER_SIZE:
            old_key = self._person_keys[object_id].pop(0)
            self._delete(old_key)

        return key

    # ---- Store alert evidence ------------------------------------------------
    def store_evidence_frame(
        self, alert_id: str, idx: int, image_bytes: bytes
    ) -> str:
        """Store an evidence frame for audit retention."""
        key = f"{self.ALERT_EVIDENCE_PREFIX}/{alert_id}/evidence/frame_{idx:03d}.jpg"
        self._put(key, image_bytes)
        return key

    # ---- Read frames ---------------------------------------------------------
    def get_frame(self, key: str) -> Optional[bytes]:
        """Read frame bytes by key."""
        return self._get(key)

    async def get_frames_base64(self, keys: List[str]) -> List[str]:
        """Fetch multiple frames and return as base64-encoded strings."""
        results = []
        for key in keys:
            raw = self._get(key)
            if raw:
                results.append(base64.b64encode(raw).decode("ascii"))
        return results

    def get_person_frame_keys(self, object_id: str) -> List[str]:
        """Return the current rolling buffer keys for a person."""
        return list(self._person_keys.get(object_id, []))

    # ---- Cleanup -------------------------------------------------------------
    def cleanup_person(self, object_id: str) -> None:
        """
        Remove all frames for a person (called after exit_retention_seconds
        or session expiry).
        """
        keys = self._person_keys.pop(object_id, [])
        for key in keys:
            self._delete(key)
        if keys:
            logger.info("Cleaned up person frames", object_id=object_id, count=len(keys))

    def cleanup_person_frames_deferred(self, object_id: str) -> List[str]:
        """
        Return keys to delete later (after exit_retention_seconds).
        Does NOT delete immediately — caller schedules deletion.
        """
        return list(self._person_keys.get(object_id, []))

    # ---- Internal helpers ----------------------------------------------------
    def _put(self, key: str, data: bytes) -> None:
        if not self.client:
            return
        try:
            self.client.put_object(
                self.BUCKET, key, io.BytesIO(data), length=len(data),
                content_type="image/jpeg",
            )
        except S3Error:
            logger.exception("SeaweedFS put failed", key=key)

    def _get(self, key: str) -> Optional[bytes]:
        if not self.client:
            return None
        resp = None
        try:
            resp = self.client.get_object(self.BUCKET, key)
            return resp.read()
        except S3Error:
            logger.debug("SeaweedFS get miss", key=key)
            return None
        finally:
            if resp is not None:
                try:
                    resp.close()
                    resp.release_conn()
                except Exception:
                    pass

    def _delete(self, key: str) -> None:
        if not self.client:
            return
        try:
            self.client.remove_object(self.BUCKET, key)
        except S3Error:
            logger.debug("SeaweedFS delete miss", key=key)
