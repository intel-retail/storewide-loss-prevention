"""Application configuration — Singleton pattern."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field


def _parse_stream_map(raw: str) -> dict:
    """Parse 'Camera_01:retail-cam1,Camera_02:retail-cam2' into a dict."""
    mapping = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" in pair:
            cam_id, stream = pair.split(":", 1)
            mapping[cam_id.strip()] = stream.strip()
    return mapping


@dataclass
class Config:
    """Centralised, immutable application config loaded from env vars.

    Implements the Singleton pattern — only one instance exists.
    """

    # MQTT / SceneScape
    mqtt_host: str = ""
    mqtt_port: int = 1883
    mqtt_topic_event: str = ""
    mqtt_ca_cert: str = ""
    scene_uid: str = ""

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    appearance_ttl_days: int = 7

    # FAISS — enrolled POI index
    faiss_dimension: int = 256
    faiss_index_path: str = "/data/faiss/poi.index"
    faiss_id_map_path: str = "/data/faiss/id_map.json"

    # FAISS — detection index (all faces seen, 7-day TTL)
    detection_index_enabled: bool = True
    detection_index_top_k: int = 20
    detection_embeddings_per_track: int = 5
    detection_embedding_interval: int = 10  # seconds between stored embeddings
    # TTL for the per-track dedup gate (claim_track NX key).
    # Must be short so recycled tracker IDs (same int id reused for a new person)
    # are not blocked. 120s covers any realistic single-person dwell time.
    track_seen_ttl: int = 600

    # Thresholds
    similarity_threshold: float = 0.6
    search_similarity_threshold: float = 0.65
    search_top_k: int = 10

    # Embedding / OpenVINO
    model_base: str = "/models/intel"
    det_model: str = ""
    lm_model: str = ""
    reid_model: str = ""
    inference_device: str = "CPU"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # SceneScape API
    scenescape_api_url: str = ""
    scenescape_api_token: str = ""

    # Alert
    alert_webhook_url: str = ""
    alert_service_url: str = ""
    delivery_handlers: list[str] = field(default_factory=lambda: ["log"])

    # Camera / MediaMTX
    camera_streams: str = ""  # comma-separated: "Camera_01,Camera_02"
    camera_stream_map: dict = field(default_factory=dict)  # camera_id → mediamtx stream path
    mediamtx_webrtc_port: int = 8889

    # Logging
    log_level: str = "INFO"

    # Cache
    object_cache_ttl: int = 300  # seconds
    alert_dedup_ttl: int = 300

    # Benchmark
    benchmark_latency: bool = False

    _instance: Config | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __class_getitem__(cls, _):
        return cls

    @classmethod
    def get_instance(cls) -> Config:
        """Thread-safe singleton accessor."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls._from_env()
        return cls._instance

    @classmethod
    def _from_env(cls) -> Config:
        model_base = os.getenv("MODEL_BASE", "/models/intel")
        scene_uid = os.getenv("SCENE_UID", "")
        mqtt_topic = os.getenv(
            "MQTT_TOPIC_EVENT",
            "scenescape/data/camera/+",
        )
        handlers_raw = os.getenv("DELIVERY_HANDLERS", "log")
        handlers = [h.strip() for h in handlers_raw.split(",") if h.strip()]

        return cls(
            mqtt_host=os.getenv("MQTT_HOST", ""),
            mqtt_port=int(os.getenv("MQTT_PORT", "1883")),
            mqtt_topic_event=mqtt_topic,
            mqtt_ca_cert=os.getenv("MQTT_CA_CERT", ""),
            scene_uid=scene_uid,
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_db=int(os.getenv("REDIS_DB", "0")),
            appearance_ttl_days=int(os.getenv("APPEARANCE_TTL_DAYS", "7")),
            faiss_dimension=int(os.getenv("FAISS_DIMENSION", "256")),
            faiss_index_path=os.getenv("FAISS_INDEX_PATH", "/data/faiss/poi.index"),
            faiss_id_map_path=os.getenv("FAISS_ID_MAP_PATH", "/data/faiss/id_map.json"),
            detection_index_enabled=os.getenv("DETECTION_INDEX_ENABLED", "true").lower() == "true",
            detection_index_top_k=int(os.getenv("DETECTION_INDEX_TOP_K", "20")),
            detection_embeddings_per_track=int(os.getenv("DETECTION_EMBEDDINGS_PER_TRACK", "5")),
            detection_embedding_interval=int(os.getenv("DETECTION_EMBEDDING_INTERVAL", "10")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.6")),
            search_similarity_threshold=float(os.getenv("SEARCH_SIMILARITY_THRESHOLD", "0.65")),
            search_top_k=int(os.getenv("SEARCH_TOP_K", "10")),
            model_base=model_base,
            det_model=os.getenv(
                "DET_MODEL",
                f"{model_base}/face-detection-retail-0004/FP32/face-detection-retail-0004.xml",
            ),
            lm_model=os.getenv(
                "LM_MODEL",
                f"{model_base}/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml",
            ),
            reid_model=os.getenv(
                "REID_MODEL",
                f"{model_base}/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml",
            ),
            inference_device=os.getenv("INFERENCE_DEVICE", "CPU"),
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("API_PORT", "8000")),
            scenescape_api_url=os.getenv("SCENESCAPE_API_URL", ""),
            scenescape_api_token=os.getenv("SCENESCAPE_API_TOKEN", ""),
            alert_webhook_url=os.getenv("ALERT_WEBHOOK_URL", ""),
            alert_service_url=os.getenv("ALERT_SERVICE_URL", "http://alert-service:8000"),
            delivery_handlers=handlers,
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            camera_streams=os.getenv("RTSP_PREWARM_CAMERAS", ""),
            camera_stream_map=_parse_stream_map(os.getenv("CAMERA_STREAM_MAP", "")),
            mediamtx_webrtc_port=int(os.getenv("MEDIAMTX_WEBRTC_PORT", "8889")),
            object_cache_ttl=int(os.getenv("OBJECT_CACHE_TTL", "300")),
            alert_dedup_ttl=int(os.getenv("ALERT_DEDUP_TTL", "300")),
            track_seen_ttl=int(os.getenv("TRACK_SEEN_TTL", "600")),
            benchmark_latency=os.getenv("BENCHMARK_LATENCY", "false").lower() == "true",
        )

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None


# Module-level convenience
_lock = threading.Lock()
_instance: Config | None = None


def get_config() -> Config:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = Config._from_env()
    return _instance


def reset_config() -> None:
    global _instance
    _instance = None
    Config.reset()
