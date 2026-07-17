"""ffmpeg faststart remux so VSS accepts the clip (build-spec §6.1)."""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


async def _run_ffmpeg(args: list[str]) -> int:
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        *args,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.debug("ffmpeg failed (%s): %s", proc.returncode, stderr.decode(errors="ignore")[-500:])
    return proc.returncode or 0


async def faststart_remux(src: str, dst: str) -> None:
    """Remux ``src`` into a streamable (moov-first) MP4 at ``dst``.

    Tries stream copy first; falls back to re-encoding video if copy fails.
    """

    rc = await _run_ffmpeg(["-y", "-i", src, "-c", "copy", "-movflags", "+faststart", dst])
    if rc == 0:
        return

    logger.info("stream-copy remux failed for %s; re-encoding", src)
    rc = await _run_ffmpeg(
        ["-y", "-i", src, "-c:v", "libx264", "-movflags", "+faststart", dst]
    )
    if rc != 0:
        raise RuntimeError(f"faststart remux failed for {src} (rc={rc})")
