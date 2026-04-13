# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Store-wide Loss Prevention — main application entry point.

Wires together the four core responsibilities:
  1. MQTT Subscription and Event Routing
  2. Session State Management
  3. Business Logic (Detection Rules)
  4. Frame Manager (SeaweedFS)

External services (called conditionally):
  - BehavioralAnalysis Service (pose analysis + VLM confirmation)
  - Rule Service (advanced rule evaluation)
"""

import asyncio
import base64
import io
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import structlog
import uvicorn
from fastapi import FastAPI
from PIL import Image

from api.routes import router
from services.config import ConfigService
from services.mqtt_service import MQTTService
from services.session_manager import SessionManager
from services.rule_engine import RuleEngine
from services.frame_manager import FrameManager
from services.alert_publisher import AlertPublisher
from services.scenescape_client import SceneScapeClient
from services.behavioral_analysis_client import BehavioralAnalysisClient

# ---- Structured logging setup -----------------------------------------------
logging.basicConfig(format="%(message)s", stream=__import__("sys").stdout, level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


# ---- Frame cropping helper ---------------------------------------------------

def _crop_to_bbox(image_bytes: bytes, bbox) -> bytes:
    """
    Crop image to bounding box if valid pixel-space bbox is available.

    Supports bbox formats:
      - dict with {x, y, w, h}  (top-left + dimensions)
      - dict with {x1, y1, x2, y2}  (corner coordinates)
      - list/tuple [x1, y1, x2, y2] or [x, y, w, h]
    Falls back to full frame if bbox is missing or unusable.
    """
    if not bbox:
        return image_bytes
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img_w, img_h = img.size

        # Parse bbox into (left, upper, right, lower) for PIL
        if isinstance(bbox, dict):
            if "x1" in bbox and "y1" in bbox:
                left, upper = int(bbox["x1"]), int(bbox["y1"])
                right = int(bbox.get("x2", img_w))
                lower = int(bbox.get("y2", img_h))
            elif "x" in bbox and "y" in bbox and "w" in bbox and "h" in bbox:
                left, upper = int(bbox["x"]), int(bbox["y"])
                right, lower = left + int(bbox["w"]), upper + int(bbox["h"])
            else:
                return image_bytes
        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            vals = [int(v) for v in bbox]
            # Heuristic: if 3rd/4th values are small relative to image,
            # treat as [x, y, w, h]; otherwise [x1, y1, x2, y2]
            if vals[2] <= img_w and vals[3] <= img_h and vals[2] > vals[0]:
                left, upper, right, lower = vals
            else:
                left, upper = vals[0], vals[1]
                right, lower = vals[0] + vals[2], vals[1] + vals[3]
        else:
            return image_bytes

        # Clamp to image bounds
        left = max(0, min(left, img_w - 1))
        upper = max(0, min(upper, img_h - 1))
        right = max(left + 1, min(right, img_w))
        lower = max(upper + 1, min(lower, img_h))

        # Only crop if the region is meaningfully smaller than the full frame
        crop_area = (right - left) * (lower - upper)
        if crop_area >= img_w * img_h * 0.95:
            return image_bytes

        cropped = img.crop((left, upper, right, lower))
        buf = io.BytesIO()
        cropped.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception:
        return image_bytes


# ---- FastAPI lifespan --------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start all services, yield, then tear down."""
    logger.info("Starting Store-wide Loss Prevention")

    # 1. Config
    config = ConfigService()
    app.state.config = config

    # 1b. Discover zones from SceneScape API (match by region name)
    ss_client = SceneScapeClient(config)
    app.state.scenescape_client = ss_client
    ss_user = os.environ.get("SCENESCAPE_API_USER", "")
    ss_pass = os.environ.get("SCENESCAPE_API_PASSWORD", "")
    if ss_user and ss_pass:
        # Authenticate with retry — web container may still be starting
        authenticated = False
        for attempt in range(5):
            authenticated = await ss_client.authenticate(ss_user, ss_pass)
            if authenticated:
                break
            wait = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
            logger.warning("SceneScape auth failed, retrying", attempt=attempt + 1, retry_in=wait)
            await asyncio.sleep(wait)

        if not authenticated:
            logger.error("SceneScape API authentication failed after retries")
        else:
            # Resolve scene_name → scene_id via SceneScape API
            scene_name = config.get_scene_name()
            if scene_name:
                scene_id = await ss_client.resolve_scene_id(scene_name)
                if scene_id:
                    config.set_scene_id(scene_id)
                    logger.info("Scene resolved from name", scene_name=scene_name, scene_id=scene_id)
                else:
                    logger.error("Could not resolve scene_name to scene_id", scene_name=scene_name)

            # Discover and map zones (already authenticated, skip re-auth)
            regions = await ss_client.fetch_regions()
            if regions:
                discovered = ss_client.map_zones(regions)
                if discovered:
                    config.merge_zones(discovered)
                    logger.info("Zone discovery complete", zones=len(config.get_zones()))
                else:
                    logger.warning("Zone discovery found no matching regions")
            else:
                logger.warning("No regions found in SceneScape")
    else:
        logger.warning(
            "SCENESCAPE_API_USER / SCENESCAPE_API_PASSWORD not set, "
            "zone discovery skipped. Use POST /api/v1/lp/zones/discover "
            "or PUT /api/v1/lp/zones/{region_id} to add zones at runtime."
        )

    # 2. Frame Manager (SeaweedFS)
    frame_mgr = FrameManager(config)
    await frame_mgr.ensure_bucket()
    app.state.frame_manager = frame_mgr

    # 3. MQTT
    mqtt_svc = MQTTService(config)
    await mqtt_svc.initialize()
    loop = asyncio.get_running_loop()
    mqtt_svc.set_event_loop(loop)
    app.state.mqtt_service = mqtt_svc

    # 4. Alert publisher
    alert_pub = AlertPublisher(config, mqtt_svc)
    app.state.alert_publisher = alert_pub

    # 5. Session manager
    session_mgr = SessionManager(config)
    app.state.session_manager = session_mgr

    # 6. External service clients (called conditionally)
    ba_client = BehavioralAnalysisClient(config)
    app.state.behavioral_analysis_client = ba_client

    # 7. Rule engine (business logic with external service integration)
    rule_engine = RuleEngine(
        config, session_mgr,
        alert_callback=alert_pub.publish,
        behavioral_analysis_client=ba_client,
        frame_manager=frame_mgr,
    )
    app.state.rule_engine = rule_engine

    # ---- Wire callbacks ----
    # Session manager fires events → rule engine (business logic)
    session_mgr.register_event_handler(rule_engine.on_event)

    # MQTT scene data → session manager (liveness: cameras, bbox, last_seen)
    mqtt_svc.register_scene_data_handler(session_mgr.on_scene_data)

    # MQTT region events → session manager (enter/exit with dwell from SceneScape)
    mqtt_svc.register_region_event_handler(session_mgr.on_region_event)

    # MQTT camera images → frame storage (cropped person frames for HIGH_VALUE zones)
    async def on_camera_image(camera_name: str, data: dict) -> None:
        image_b64 = data.get("image", data.get("data", ""))
        if not image_b64:
            return
        image_bytes = base64.b64decode(image_b64)
        ts = datetime.now(timezone.utc)

        # Store cropped frames only for persons in HIGH_VALUE zones
        for session in session_mgr.get_all_sessions().values():
            if camera_name not in session.current_cameras:
                continue

            # Check if person is currently in any HIGH_VALUE zone
            in_high_value = False
            for zone_id in session.current_zones:
                if config.get_zone_type(zone_id) == "HIGH_VALUE":
                    in_high_value = True
                    break

            if in_high_value:
                # Crop to person bounding box if available
                frame_bytes = _crop_to_bbox(image_bytes, session.bbox)
                key = frame_mgr.store_person_frame(
                    session.object_id, frame_bytes, ts
                )
                session.add_frame_key(key)

    mqtt_svc.register_camera_image_handler(on_camera_image)

    # Background task: request frames from cameras that see people in HIGH_VALUE zones
    async def frame_request_loop() -> None:
        """Periodically send 'getimage' to cameras with active HIGH_VALUE sessions."""
        FRAME_REQUEST_INTERVAL = 0.5  # ~2 fps
        while True:
            try:
                cameras_needed: set[str] = set()
                for session in session_mgr.get_all_sessions().values():
                    for zone_id in session.current_zones:
                        if config.get_zone_type(zone_id) == "HIGH_VALUE":
                            cameras_needed.update(session.current_cameras)
                            break

                for cam in cameras_needed:
                    mqtt_svc.publish_raw(
                        f"scenescape/cmd/camera/{cam}", "getimage"
                    )

            except Exception:
                logger.exception("Error in frame request loop")

            await asyncio.sleep(FRAME_REQUEST_INTERVAL)

    # Start background tasks
    mqtt_task = asyncio.create_task(mqtt_svc.start())
    expiry_task = asyncio.create_task(session_mgr.run_expiry_loop())
    frame_req_task = asyncio.create_task(frame_request_loop())
    loiter_task = asyncio.create_task(rule_engine.run_loiter_check_loop())

    logger.info(
        "Store-wide Loss Prevention started",
        store_id=config.get_store_id(),
        zones=len(config.get_zones()),
        cameras=len(config.get_cameras()),
    )

    yield

    # ---- Shutdown ----
    logger.info("Shutting down Store-wide Loss Prevention")
    await mqtt_svc.stop()
    expiry_task.cancel()
    frame_req_task.cancel()
    loiter_task.cancel()
    mqtt_task.cancel()


# ---- App ---------------------------------------------------------------------

app = FastAPI(
    title="Store-wide Loss Prevention",
    description="Store-wide Loss Prevention: Suspicious Activity Detection",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1/lp")


@app.get("/health")
async def health():
    return {"status": "healthy"}


# ---- Entry point -------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8082,
        log_level="info",
    )
