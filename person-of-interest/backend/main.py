"""POI Re-identification System — Application Entry Point.

Wires all layers together following Clean Architecture:
  API → Service → Domain ← Infrastructure
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api import camera_routes, poi_routes, search_routes, thumbnail_routes
from backend.utils.thumbnail import prewarm_grabbers
from backend.consumers.mqtt_consumer import EventConsumer
from backend.consumers.scenescape_consumer import ScenescapeRegionConsumer
from backend.core.config import get_config
from backend.factory.factories import EmbeddingModelFactory
from backend.infrastructure.faiss.repository import FAISSRepository
from backend.infrastructure.faiss.detection_index import DetectionIndexRepository
from backend.infrastructure.mqtt.adapter import MQTTAdapter
from backend.infrastructure.redis.repository import (
    RedisCacheRepository,
    RedisEmbeddingMappingRepository,
    RedisEventRepository,
    RedisPOIRepository,
)
from backend.infrastructure.scenescape.adapter import ScenescapeAPIAdapter
from backend.observer.events import EventBus
from backend.service.alert_service import AlertService
from backend.service.event_service import EventService
from backend.service.matching_service import MatchingService
from backend.service.poi_service import POIService
from backend.strategy.alert import AlertServiceStrategy
from backend.strategy.matching import CosineSimilarityStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("poi.main")

# Global references for cleanup
_mqtt_adapter = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    global _mqtt_adapter
    cfg = get_config()
    log.info("=== POI Re-identification System Starting ===")

    # ── Infrastructure ──
    log.info("Initializing FAISS repository...")
    faiss_repo = FAISSRepository()

    log.info("Initializing Redis repositories...")
    poi_repo = RedisPOIRepository()
    cache_repo = RedisCacheRepository()
    event_repo = RedisEventRepository()
    mapping_repo = RedisEmbeddingMappingRepository()

    log.info("Initializing detection index...")
    import redis as _redis_mod
    # decode_responses=False — DetectionIndexRepository stores raw float32 bytes for vectors
    _redis_client = _redis_mod.Redis(
        host=cfg.redis_host, port=cfg.redis_port, db=cfg.redis_db, decode_responses=False
    )
    detection_index = DetectionIndexRepository(_redis_client) if cfg.detection_index_enabled else None

    log.info("Initializing SceneScape adapter...")
    scenescape = ScenescapeAPIAdapter()

    # ── Strategy ──
    matching_strategy = CosineSimilarityStrategy(faiss_repo)

    # Single delivery strategy: forward all alerts to the alert-service.
    # The alert-service owns log, WebSocket, and MQTT delivery.
    alert_strategies = [AlertServiceStrategy(cfg.alert_service_url)]

    # ── Observer ──
    event_bus = EventBus()

    # ── Service ──
    poi_service = POIService(poi_repo, faiss_repo, mapping_repo)
    matching_service = MatchingService(matching_strategy, cache_repo)
    event_service = EventService(event_repo)
    alert_service = AlertService(alert_strategies, event_repo, poi_repo, event_bus)

    # ── Consumer ──
    consumer = EventConsumer(
        matching_service, event_service, alert_service, event_bus,
        event_repo=event_repo,
        detection_index=detection_index,
    )
    region_consumer = ScenescapeRegionConsumer(event_service, event_repo=event_repo)

    # ── MQTT ──
    if cfg.mqtt_host:
        log.info("Starting MQTT consumer...")
        _mqtt_adapter = MQTTAdapter(on_event=consumer.handle_event, on_region_event=region_consumer.handle_event)
        try:
            _mqtt_adapter.start()
        except Exception:
            log.exception("Failed to start MQTT — running without live events")
            _mqtt_adapter = None

    # ── Inject services into API routes ──
    poi_routes.init(poi_service)
    search_routes.init(
        EmbeddingModelFactory.create(), detection_index, event_repo
    )
    camera_routes.init(scenescape)
    thumbnail_routes.init(event_repo, detection_index)

    # ── Exit promoter — promotes exit embeddings to FAISS after persons leave ──
    if detection_index is not None:
        from backend.infrastructure.faiss.exit_promoter import ExitPromoterThread
        _exit_promoter = ExitPromoterThread(detection_index, interval_sec=30)
        _exit_promoter.start()
        log.info("ExitPromoterThread started")

    # Pre-warm RTSP grabbers so frames are cached before first match event
    import os as _os
    _prewarm = [c.strip() for c in _os.getenv("RTSP_PREWARM_CAMERAS", "").split(",") if c.strip()]
    if _prewarm:
        log.info("Pre-warming RTSP grabbers for cameras: %s", _prewarm)
        prewarm_grabbers(_prewarm)

    log.info("FAISS: %d vectors indexed", faiss_repo.total_vectors())
    log.info("=== POI System Ready on %s:%d ===", cfg.api_host, cfg.api_port)

    yield

    # ── Shutdown ──
    log.info("Shutting down...")
    if _mqtt_adapter:
        _mqtt_adapter.stop()
    faiss_repo.save_to_disk()
    log.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="POI Re-identification System",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    app.include_router(poi_routes.router, prefix="/api/v1")
    app.include_router(search_routes.router, prefix="/api/v1")
    app.include_router(camera_routes.router, prefix="/api/v1")
    app.include_router(thumbnail_routes.router, prefix="/api/v1")

    # Health check
    @app.get("/api/v1/status")
    async def status():
        faiss_repo = FAISSRepository()
        return {
            "status": "running",
            "faiss_vectors": faiss_repo.total_vectors(),
            "mqtt_connected": _mqtt_adapter.is_connected if _mqtt_adapter else False,
        }

    # Alerts endpoints
    @app.get("/api/v1/alerts")
    async def get_alerts():
        event_repo = RedisEventRepository()
        return event_repo.get_recent_alerts(50)

    @app.delete("/api/v1/alerts")
    async def clear_alerts():
        import httpx
        _cfg = get_config()
        event_repo = RedisEventRepository()
        deleted = event_repo.clear_alerts()
        log.info("Alerts cleared via API: %d Redis keys deleted", deleted)

        # Also clear the alert-service in-memory WebSocket history buffer
        try:
            async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
                resp = await client.delete(f"{_cfg.alert_service_url}/api/v1/alerts")
                resp.raise_for_status()
                log.info("Alert-service history cleared: %s", resp.json())
        except Exception:
            log.warning("Could not clear alert-service history (non-fatal)", exc_info=True)

        return {"status": "cleared", "deleted": deleted}

    # Mount uploads directory
    upload_dir = os.getenv("UPLOAD_DIR", "/data/uploads")
    os.makedirs(upload_dir, exist_ok=True)
    app.mount("/uploads", StaticFiles(directory=upload_dir), name="uploads")

    return app


app = create_app()


def main():
    cfg = get_config()
    log_level = cfg.log_level.lower()
    uvicorn.run(
        "backend.main:app",
        host=cfg.api_host,
        port=cfg.api_port,
        log_level=log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
