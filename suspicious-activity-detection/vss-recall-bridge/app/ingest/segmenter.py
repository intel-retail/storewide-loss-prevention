"""RTSP -> MP4 segmenter, one asyncio task per camera (build-spec §6.2).

Near-real-time: each closed segment is uploaded immediately, so VSS's ``created_at``
tracks capture time within one chunk length.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from ..clients.vss_client import VssClient
from ..config import Camera, Settings
from .pipeline import process_clip

logger = logging.getLogger(__name__)

_RESTART_BACKOFF_SECONDS = 2.0
_WATCH_INTERVAL_SECONDS = 1.0


async def _watch_segments(
    camera: Camera, clips_dir: Path, vss: VssClient, settings: Settings
) -> None:
    """Process each completed segment exactly once.

    The newest file is still being written by ffmpeg, so every file *except* the
    last (sorted by name == start time) is treated as complete.
    """

    processed: set[str] = set()
    prefix = f"{camera.camera_id}_"

    while True:
        files = sorted(clips_dir.glob(f"{prefix}*.mp4"))
        for path in files[:-1]:  # exclude the in-progress newest segment
            if path.name in processed:
                continue
            processed.add(path.name)
            try:
                await process_clip(
                    camera=camera,
                    clip_path=str(path),
                    vss=vss,
                    clips_dir=str(clips_dir),
                )
            except Exception:  # one bad clip must not kill the camera task
                logger.exception("failed to process segment %s", path)
        await asyncio.sleep(_WATCH_INTERVAL_SECONDS)


async def run_segmenter(camera: Camera, settings: Settings, vss: VssClient) -> None:
    """Run ffmpeg segmentation for a camera, restarting on exit/crash."""

    clips_dir = Path(settings.clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)
    segment_seconds = camera.segment_seconds or settings.default_segment_seconds
    out_template = str(clips_dir / f"{camera.camera_id}_%Y%m%dT%H%M%S.mp4")

    while True:
        logger.info("starting ffmpeg segmenter for %s", camera.camera_id)
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-rtsp_transport",
            "tcp",
            "-i",
            camera.rtsp_url,
            "-an",
            "-c:v",
            "copy",
            "-f",
            "segment",
            "-segment_time",
            str(segment_seconds),
            "-reset_timestamps",
            "1",
            "-strftime",
            "1",
            out_template,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        watcher = asyncio.create_task(_watch_segments(camera, clips_dir, vss, settings))
        try:
            rc = await proc.wait()
        finally:
            watcher.cancel()
        logger.warning("ffmpeg for %s exited rc=%s; restarting", camera.camera_id, rc)
        await asyncio.sleep(_RESTART_BACKOFF_SECONDS)
