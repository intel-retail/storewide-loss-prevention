"""One-shot ingest of an existing MP4 (demos / backfill) (build-spec §6.3).

Same pipeline as the segmenter minus the live source. ``created_at`` = upload time,
so time filtering on *old* backfilled footage is only accurate with R2 (design §4);
fine for demos and recently recorded files.

CLI:
    python -m app.ingest.file_ingest --file /path/to/clip.mp4 --camera cam1
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from ..clients.vss_client import VssClient
from ..config import Camera, Settings, get_settings, load_cameras
from .pipeline import process_clip

logger = logging.getLogger(__name__)


async def ingest_file(
    camera: Camera, source_file: str, vss: VssClient, settings: Settings
) -> str:
    """Remux + upload + embed an existing MP4. Returns the VSS videoId."""

    return await process_clip(
        camera=camera,
        clip_path=source_file,
        vss=vss,
        clips_dir=settings.clips_dir,
        keep_source=True,  # never delete the user's source file
    )


async def _run_cli(file_path: str, camera_id: str) -> str:
    settings = get_settings()
    cameras = {
        c.camera_id: c
        for c in load_cameras(
            settings.scene_config,
            settings.rtsp_base_url,
            settings.store_id,
            settings.default_segment_seconds,
        )
    }
    camera = cameras.get(camera_id)
    if camera is None:
        raise SystemExit(
            f"camera '{camera_id}' not found in {settings.scene_config}"
        )

    vss = VssClient(
        settings.vss_base_url,
        settings.http_timeout_seconds,
        settings.search_poll_seconds,
        settings.search_poll_timeout_seconds,
    )
    try:
        return await ingest_file(camera, file_path, vss, settings)
    finally:
        await vss.aclose()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest one MP4 into VSS for a camera.")
    parser.add_argument("--file", required=True, help="path to the MP4 to ingest")
    parser.add_argument("--camera", required=True, help="camera name from scene-config.yaml")
    args = parser.parse_args()

    video_id = asyncio.run(_run_cli(args.file, args.camera))
    print(video_id)


if __name__ == "__main__":
    main()
