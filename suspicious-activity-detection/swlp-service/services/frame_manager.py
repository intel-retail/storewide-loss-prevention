# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Frame Manager -- manages person frames in SeaweedFS (S3-compatible).

Bucket structure:
  loss-prevention-frames/
  +-- {object_id}/
  |   +-- {timestamp_1}.jpg    # Full camera frame
  |   +-- {timestamp_2}.jpg
  |   +-- ...                  # Rolling buffer of last 20 frames (~10s at 2fps)
  +-- alerts/
      +-- {alert_id}/
          +-- evidence/        # Frames sent to behavioral analysis, retained for audit

Only stores frames for individuals currently in HIGH_VALUE zones.
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
    Manages person frames in SeaweedFS via S3-compatible API.

    Only stores frames for persons in HIGH_VALUE zones.
    Maintains a rolling buffer of 20 frames per person.
    Also mirrors frames to the behavioral-frames bucket for the BA service.
    """

    BUCKET = "loss-prevention-frames"
    BA_BUCKET = "behavioral-frames"
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
        self._person_ba_keys: Dict[str, List[str]] = {}

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
        """Create the frame buckets if they don't exist. Retries on connection failure."""
        if not self.client:
            return
        import asyncio
        for attempt in range(5):
            try:
                for bucket in (self.BUCKET, self.BA_BUCKET):
                    if not self.client.bucket_exists(bucket):
                        self.client.make_bucket(bucket)
                        logger.info("Created bucket", bucket=bucket)
                    else:
                        logger.info("Bucket exists", bucket=bucket)
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

    # ---- Store person frame --------------------------------------------------
    def store_person_frame(
        self, object_id: str, image_bytes: bytes, ts: Optional[datetime] = None,
        region_id: Optional[str] = None,
        entry_timestamp: Optional[str] = None,
        scene_id: Optional[str] = None,
    ) -> str:
        """
        Store a full camera frame in the rolling buffer.
        Also mirrors to behavioral-frames bucket for BA service consumption.
        Evicts the oldest frame if the buffer exceeds ROLLING_BUFFER_SIZE.
        Returns the SeaweedFS object key.
        """
        ts = ts or datetime.now(timezone.utc)
        # Key: {scene_id}/{object_id}/{timestamp}.jpg
        prefix = f"{scene_id}/{object_id}" if scene_id else object_id
        key = f"{prefix}/{ts.strftime('%Y%m%dT%H%M%S_%f')}.jpg"
        self._put(key, image_bytes)

        # Mirror to behavioral-frames bucket:
        # {scene_id}/{person_id}/{region_id}/{entry_timestamp}/frames/{ts_ms}.jpg
        ts_ms = int(ts.timestamp() * 1000)
        # Convert entry_timestamp ISO string to compact folder name
        entry_folder = ""
        if entry_timestamp:
            entry_folder = entry_timestamp.replace(":", "").replace("-", "").replace("T", "T").split("+")[0].split(".")[0]
        if region_id and entry_folder:
            ba_key = f"{prefix}/{region_id}/{entry_folder}/frames/{ts_ms}.jpg"
        elif region_id:
            ba_key = f"{prefix}/{region_id}/frames/{ts_ms}.jpg"
        else:
            ba_key = f"{prefix}/frames/{ts_ms}.jpg"
        self._put(ba_key, image_bytes, bucket=self.BA_BUCKET)

        # Track keys for rolling buffer management
        if object_id not in self._person_keys:
            self._person_keys[object_id] = []
        self._person_keys[object_id].append(key)

        # Evict oldest if over buffer size
        while len(self._person_keys[object_id]) > self.ROLLING_BUFFER_SIZE:
            old_key = self._person_keys[object_id].pop(0)
            self._delete(old_key)

        # Track BA keys (no eviction — BA service owns lifecycle)
        if object_id not in self._person_ba_keys:
            self._person_ba_keys[object_id] = []
        self._person_ba_keys[object_id].append(ba_key)

        return key

    # ---- Store alert evidence ------------------------------------------------
    def store_evidence_frame(
        self, alert_id: str, idx: int, image_bytes: bytes,
        scene_id: Optional[str] = None,
    ) -> str:
        """Store an evidence frame for audit retention."""
        prefix = f"{scene_id}/{self.ALERT_EVIDENCE_PREFIX}" if scene_id else self.ALERT_EVIDENCE_PREFIX
        key = f"{prefix}/{alert_id}/evidence/frame_{idx:03d}.jpg"
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
    def cleanup_person(self, object_id: str, scene_id: Optional[str] = None) -> None:
        """
        Remove ALL frames for a person across every visit (called on session
        expiry / PERSON_LOST). Use ``cleanup_visit`` for per-visit cleanup
        on zone EXIT.
        """
        keys = self._person_keys.pop(object_id, [])
        ba_keys = self._person_ba_keys.pop(object_id, [])
        for key in keys:
            self._delete(key)
        # Clean up behavioral-frames bucket
        prefix = f"{scene_id}/{object_id}/" if scene_id else f"{object_id}/"
        self._delete_prefix(prefix, bucket=self.BA_BUCKET)
        if keys:
            logger.info("Cleaned up person frames", object_id=object_id, count=len(keys))

    def cleanup_visit(
        self,
        object_id: str,
        region_id: str,
        entry_timestamp: str,
        scene_id: Optional[str] = None,
    ) -> None:
        """
        Remove ONLY this visit's behavioral-frames prefix:
            {scene_id}/{object_id}/{region_id}/{entry_folder}/

        Other visits (different entry_timestamp) for the same person and
        region remain intact, so the BA service can still process them.
        Rolling-buffer keys in the LP bucket are person-wide and not
        touched here — they age out naturally via the ROLLING_BUFFER_SIZE
        eviction in ``store_person_frame``.
        """
        if not entry_timestamp:
            logger.warning(
                "cleanup_visit called without entry_timestamp — skipping",
                object_id=object_id,
                region_id=region_id,
            )
            return
        # Mirror the entry_folder transform used by store_person_frame so
        # the prefix matches what was actually written.
        entry_folder = (
            entry_timestamp.replace(":", "").replace("-", "").split("+")[0].split(".")[0]
        )
        person_prefix = f"{scene_id}/{object_id}" if scene_id else object_id
        prefix = f"{person_prefix}/{region_id}/{entry_folder}/"
        logger.info(
            "cleanup_visit start",
            object_id=object_id,
            region_id=region_id,
            entry_timestamp=entry_timestamp,
            scene_id=scene_id,
            bucket=self.BA_BUCKET,
            prefix=prefix,
        )
        deleted = self._delete_prefix(prefix, bucket=self.BA_BUCKET)
        # Forget any tracked BA keys that fell under this prefix so the
        # in-memory list doesn't grow unbounded across visits.
        ba_keys = self._person_ba_keys.get(object_id, [])
        if ba_keys:
            kept = [k for k in ba_keys if not k.startswith(prefix)]
            self._person_ba_keys[object_id] = kept
            logger.info(
                "cleanup_visit pruned in-memory key list",
                object_id=object_id,
                before=len(ba_keys),
                after=len(kept),
            )
        logger.info(
            "cleanup_visit done",
            object_id=object_id,
            region_id=region_id,
            entry_timestamp=entry_timestamp,
            prefix=prefix,
            deleted=deleted,
        )

    def cleanup_person_frames_deferred(self, object_id: str) -> List[str]:
        """
        Return keys to delete later (after exit_retention_seconds).
        Does NOT delete immediately — caller schedules deletion.
        """
        return list(self._person_keys.get(object_id, []))

    # ---- Internal helpers ----------------------------------------------------
    def _put(self, key: str, data: bytes, bucket: Optional[str] = None) -> None:
        if not self.client:
            return
        bucket = bucket or self.BUCKET
        try:
            self.client.put_object(
                bucket, key, io.BytesIO(data), length=len(data),
                content_type="image/jpeg",
            )
        except S3Error:
            logger.exception("SeaweedFS put failed", key=key, bucket=bucket)

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

    def _delete(self, key: str, bucket: Optional[str] = None) -> None:
        if not self.client:
            return
        bucket = bucket or self.BUCKET
        try:
            self.client.remove_object(bucket, key)
        except S3Error:
            logger.debug("SeaweedFS delete miss", key=key)

    def _delete_prefix(self, prefix: str, bucket: Optional[str] = None) -> int:
        """Delete all objects under a prefix in the given bucket.

        Returns the number of objects successfully removed.
        """
        if not self.client:
            return 0
        bucket = bucket or self.BUCKET
        deleted = 0
        failed = 0
        try:
            objects = list(
                self.client.list_objects(bucket, prefix=prefix, recursive=True)
            )
            for obj in objects:
                try:
                    self.client.remove_object(bucket, obj.object_name)
                    deleted += 1
                except S3Error:
                    failed += 1
                    logger.warning(
                        "SeaweedFS object delete failed",
                        bucket=bucket,
                        key=obj.object_name,
                    )
            logger.info(
                "_delete_prefix complete",
                bucket=bucket,
                prefix=prefix,
                listed=len(objects),
                deleted=deleted,
                failed=failed,
            )
        except S3Error:
            logger.exception(
                "SeaweedFS list_objects failed during prefix delete",
                bucket=bucket,
                prefix=prefix,
            )
        return deleted
