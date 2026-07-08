"""FastAPI app: ingest workers (lifespan) + optional thin query proxy (build-spec §9)."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .clients.vss_client import VssClient
from .config import get_settings, load_cameras
from .ingest.file_ingest import ingest_file
from .ingest.segmenter import run_segmenter
from .query.routes import router as query_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    cameras = load_cameras(
        settings.scene_config,
        settings.rtsp_base_url,
        settings.store_id,
        settings.default_segment_seconds,
    )
    app.state.vss = VssClient(
        settings.vss_base_url,
        settings.http_timeout_seconds,
        settings.search_poll_seconds,
        settings.search_poll_timeout_seconds,
        settings.dataprep_base_url,
    )
    app.state.camera_ids = {c.camera_id for c in cameras}
    # Any tag present in scene-config is a valid search filter: camera ids,
    # the store id, and the scene's zone names.
    known_tags: set[str] = set()
    for c in cameras:
        known_tags.add(c.camera_id)
        if c.store_id:
            known_tags.add(c.store_id)
        known_tags.update(c.extra_tags)
    app.state.known_tags = known_tags

    tasks: list[asyncio.Task] = []
    for cam in cameras:
        if not cam.enabled:
            continue
        if cam.rtsp_url:
            tasks.append(asyncio.create_task(run_segmenter(cam, settings, app.state.vss)))
        elif cam.source_file:
            tasks.append(
                asyncio.create_task(ingest_file(cam, cam.source_file, app.state.vss, settings))
            )
    app.state.ingest_tasks = tasks
    logger.info("started %d ingest task(s)", len(tasks))

    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        await app.state.vss.aclose()


app = FastAPI(title="vss-recall-bridge", version="0.1.0", lifespan=lifespan)
app.include_router(
    query_router,
    prefix="/api/v1/lp/recall",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
