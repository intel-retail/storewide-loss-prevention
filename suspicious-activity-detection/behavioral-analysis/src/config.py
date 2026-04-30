# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Configuration settings for BehavioralAnalysis Service."""

import logging
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Service configuration loaded from environment variables."""

    # Service settings
    debug: bool = False
    log_level: str = "INFO"

    # Pose confidence threshold
    pose_confidence_threshold: float = 0.5

    # Pose model settings
    yolo_pose_model: str = "/models/yolo_models/yolo11n-pose/yolo11n-pose.xml"
    gst_inference_device: str = "CPU"

    # Legacy GStreamer settings (kept for backwards compatibility)
    person_detector_model: str = "/omz_models/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml"
    rtmpose_model: str = "/models/rtmpose_models/rtmpose/rtmpose.xml"
    gvapython_module: str = "/app/src/pose_logger_rtmpose.py"
    gst_detect_threshold: float = 0.15
    gst_pipeline_timeout: int = 120

    # Frame analysis settings
    min_frames_for_detection: int = 3
    max_frames_to_fetch: int = 20
    pose_frames_count: int = 10

    # SeaweedFS settings
    seaweedfs_endpoint: str = "http://localhost:8333"
    seaweedfs_bucket: str = "behavioral-frames"
    seaweedfs_access_key: str = ""
    seaweedfs_secret_key: str = ""
    seaweedfs_max_frame_age: int = 120  # seconds

    # VLM settings
    vlm_endpoint: str = "http://ovms-vlm:8001"
    vlm_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vlm_enabled: bool = True
    vlm_timeout: float = 300.0
    # Max concurrent VLM requests in flight against ovms-vlm. Continuous
    # batching is fine but unbounded fan-in lets the cache and per-request
    # latency grow without bound. 1–2 keeps OVMS responsive on a single
    # GPU.
    vlm_max_concurrency: int = 1
    vlm_max_tokens: int = 500
    vlm_temperature: float = 0.1
    vlm_max_image_size: int = 512

    # Pattern config file path
    pattern_config_path: str = "/app/config/patterns.yaml"

    # MQTT settings (for BA request/result queue)
    mqtt_host: str = "broker.scenescape.intel.com"
    mqtt_port: int = 1883
    ba_request_topic: str = "ba/requests"
    ba_result_topic: str = "ba/results"

    class Config:
        env_prefix = ""  # No prefix, use exact variable names
        case_sensitive = False


def load_pattern_config(path: str) -> dict[str, Any]:
    """Load pattern definitions from YAML config file."""
    config_path = Path(path)
    if not config_path.exists():
        logger.warning(f"Pattern config not found: {path}, using defaults")
        return {}

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    patterns = config.get("patterns", {})
    enabled = {k: v for k, v in patterns.items() if v.get("enabled", True)}
    logger.info(f"Loaded {len(enabled)} enabled patterns from {path}")
    return patterns
