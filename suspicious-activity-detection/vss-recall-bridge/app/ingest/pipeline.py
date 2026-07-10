"""Shared clip pipeline: remux -> upload -> embed -> cleanup (build-spec §6.5).

No local state is written. The bridge stores nothing about a clip; tags and the
playback URL live in VSS after upload.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

from ..clients.vss_client import VssClient
from ..config import Camera
from .remux import faststart_remux
from .tagger import build_tags

logger = logging.getLogger(__name__)


def _safe_unlink(path: str | os.PathLike[str]) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as exc:  # pragma: no cover - best effort cleanup
        logger.warning("could not delete %s: %s", path, exc)


async def process_clip(
    *,
    camera: Camera,
    clip_path: str,
    vss: VssClient,
    clips_dir: str,
    keep_source: bool = False,
) -> str:
    """Remux, upload and embed one clip; return its VSS videoId.

    Raises on failure so the caller can decide whether to retry/continue. A single
    bad clip must not kill a camera task (handled by the segmenter loop).
    """

    Path(clips_dir).mkdir(parents=True, exist_ok=True)
    remuxed_path = os.path.join(clips_dir, f".remux_{uuid.uuid4().hex}.mp4")

    try:
        await faststart_remux(clip_path, remuxed_path)
        tags = build_tags(camera)
        name = os.path.splitext(os.path.basename(clip_path))[0]
        video_id = await vss.upload(file_path=remuxed_path, name=name, tags=tags)
        await vss.trigger_embedding(video_id)
        logger.info("ingested %s -> %s (tags=%s)", name, video_id, tags)
        return video_id
    finally:
        _safe_unlink(remuxed_path)
        if not keep_source:
            _safe_unlink(clip_path)
