"""RTSP -> MP4 segmenter, one asyncio task per camera (build-spec §6.2).

Near-real-time: each closed segment is uploaded immediately, so VSS's ``created_at``
tracks capture time within one chunk length.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from pathlib import Path
from urllib.parse import urlparse

from ..clients.vss_client import VssClient
from ..config import Camera, Settings
from .pipeline import process_clip

logger = logging.getLogger(__name__)

_RESTART_BACKOFF_SECONDS = 2.0
_WATCH_INTERVAL_SECONDS = 1.0
# A just-closed segment needs a moment for ffmpeg to flush its moov atom; wait
# this long after the last write before treating a segment as complete.
_SEGMENT_SETTLE_SECONDS = 2.0
# Startup/reconnect: poll the RTSP host until DNS resolves and its port accepts
# connections before launching ffmpeg, so we never fire into an unresolvable
# hostname (the mediaserver startup race).
_SOURCE_WAIT_BACKOFF_SECONDS = 1.0
_SOURCE_WAIT_MAX_BACKOFF_SECONDS = 10.0
# TESTING ONLY: stop uploading after this many clips per camera (0 = unlimited).
_MAX_UPLOADS_FOR_TESTING = 5


async def _is_complete_mp4(path: Path) -> bool:
    """True if ffprobe can read the file's duration (i.e. the moov atom is present).

    Segments left truncated by an ffmpeg crash have no moov and would fail the
    remux; this lets the watcher drop them instead of erroring on every one.
    """

    proc = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    return proc.returncode == 0 and out.strip() not in (b"", b"N/A")


async def _watch_segments(
    camera: Camera, clips_dir: Path, vss: VssClient, settings: Settings
) -> None:
    """Process each completed segment exactly once.

    The newest file is still being written by ffmpeg, so every file *except* the
    last (sorted by name == start time) is treated as complete.
    """

    processed: set[str] = set()
    prefix = f"{camera.camera_id}_"
    uploaded = 0

    while True:
        files = sorted(clips_dir.glob(f"{prefix}*.mp4"))
        for path in files[:-1]:  # exclude the in-progress newest segment
            if path.name in processed:
                continue
            # TESTING ONLY: stop after the configured number of uploads.
            if _MAX_UPLOADS_FOR_TESTING and uploaded >= _MAX_UPLOADS_FOR_TESTING:
                logger.info(
                    "reached %s-clip test cap for %s; not uploading further segments",
                    _MAX_UPLOADS_FOR_TESTING,
                    camera.camera_id,
                )
                return
            # Skip a segment that is still settling (just rotated / being written).
            try:
                if time.time() - path.stat().st_mtime < _SEGMENT_SETTLE_SECONDS:
                    continue
            except FileNotFoundError:
                continue
            # Discard truncated orphans (e.g. left by an ffmpeg crash) once.
            if not await _is_complete_mp4(path):
                processed.add(path.name)
                logger.warning("discarding incomplete segment %s (no moov)", path)
                try:
                    os.unlink(path)
                except OSError:
                    pass
                continue
            processed.add(path.name)
            try:
                await process_clip(
                    camera=camera,
                    clip_path=str(path),
                    vss=vss,
                    clips_dir=str(clips_dir),
                )
                uploaded += 1
            except Exception:  # one bad clip must not kill the camera task
                logger.exception("failed to process segment %s", path)
        await asyncio.sleep(_WATCH_INTERVAL_SECONDS)


async def _wait_for_source(rtsp_url: str, camera_id: str) -> None:
    """Block until the RTSP host resolves in DNS and its TCP port is listening.

    Prevents the startup race where ffmpeg launches before the mediaserver
    container is attached to the network (DNS failure) or before its port is
    open, which otherwise surfaces as noisy ``rc=251`` crash-restart cycles.
    """

    parsed = urlparse(rtsp_url)
    host = parsed.hostname
    if not host:
        return
    port = parsed.port or 554
    attempt = 0
    while True:
        writer = None
        try:
            _, writer = await asyncio.open_connection(host, port)
            if attempt:
                logger.info(
                    "RTSP source %s:%s reachable for %s", host, port, camera_id
                )
            return
        except (OSError, socket.gaierror):
            attempt += 1
            if attempt == 1:
                logger.info(
                    "waiting for RTSP source %s:%s for %s", host, port, camera_id
                )
            await asyncio.sleep(
                min(
                    _SOURCE_WAIT_BACKOFF_SECONDS * attempt,
                    _SOURCE_WAIT_MAX_BACKOFF_SECONDS,
                )
            )
        finally:
            if writer is not None:
                writer.close()
                try:
                    await writer.wait_closed()
                except OSError:
                    pass


async def run_segmenter(camera: Camera, settings: Settings, vss: VssClient) -> None:
    """Run ffmpeg segmentation for a camera, restarting on exit/crash."""

    clips_dir = Path(settings.clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)
    segment_seconds = camera.segment_seconds or settings.default_segment_seconds
    out_template = str(clips_dir / f"{camera.camera_id}_%Y%m%dT%H%M%S.mp4")

    while True:
        await _wait_for_source(camera.rtsp_url, camera.camera_id)
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
            stderr=asyncio.subprocess.PIPE,
        )
        watcher = asyncio.create_task(_watch_segments(camera, clips_dir, vss, settings))
        try:
            _, stderr = await proc.communicate()
        finally:
            watcher.cancel()
        rc = proc.returncode
        tail = ""
        if stderr:
            lines = stderr.decode("utf-8", "replace").strip().splitlines()
            tail = " | ".join(lines[-4:])
        logger.warning(
            "ffmpeg for %s exited rc=%s; restarting%s",
            camera.camera_id,
            rc,
            f" — {tail}" if tail else "",
        )
        await asyncio.sleep(_RESTART_BACKOFF_SECONDS)
