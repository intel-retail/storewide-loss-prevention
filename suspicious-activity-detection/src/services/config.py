# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Configuration service for Store-wide Loss Prevention."""

import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class ConfigService:
    """Loads and exposes app_config.json and zone_config.json."""

    def __init__(self) -> None:
        self._config_dir = Path(os.environ.get("CONFIG_DIR", "/app/configs"))
        if not self._config_dir.exists():
            # Fallback for local development: configs/ next to src/
            self._config_dir = Path(__file__).resolve().parent.parent.parent / "configs"
        self._app_cfg = self._load_json("app_config.json")
        self._zone_cfg = self._load_json("zone_config.json")

        # Zone name → type from config: {"jewelry_zone": "HIGH_VALUE", ...}
        self._zone_name_map: Dict[str, str] = dict(self._zone_cfg.get("zones", {}))
        # Runtime zone map: {region_uuid: {name, type}} — populated by SceneScapeClient
        self._zones: Dict[str, dict] = {}
        # Resolved at runtime from scene_name via SceneScape API
        self._resolved_scene_id: Optional[str] = None
        self._zone_lock = threading.Lock()

        logger.info(
            "ConfigService initialized",
            store_id=self.get_store_id(),
            num_cameras=len(self.get_cameras()),
            num_zones=len(self._zones),
        )

    # ---- loaders ----
    def _load_json(self, filename: str) -> dict:
        path = self._config_dir / filename
        if not path.exists():
            logger.warning("Config file not found, using empty dict", path=str(path))
            return {}
        with open(path, "r") as f:
            return json.load(f)

    # ---- store ----
    def get_store_id(self) -> str:
        return self._app_cfg.get("store", {}).get("id", "store_001")

    def get_store_name(self) -> str:
        return self._app_cfg.get("store", {}).get("name", "retail_store_1")

    # ---- cameras (derived from zone_config.camera_name, fallback to app_config) ----
    def get_cameras(self) -> List[dict]:
        cam = self._zone_cfg.get("camera_name", "")
        if cam:
            return [{
                "name": cam,
                "number": 1,
                "description": self._zone_cfg.get("scene_name", ""),
                "data_topic": f"scenescape/data/camera/{cam}",
                "image_topic": f"scenescape/image/camera/{cam}",
            }]
        return self._app_cfg.get("cameras", [])

    def get_camera_topics(self) -> List[str]:
        return [c["data_topic"] for c in self.get_cameras()]

    def get_image_topics(self) -> List[str]:
        return [c["image_topic"] for c in self.get_cameras()]

    # ---- mqtt ----
    def get_mqtt_config(self) -> dict:
        return self._app_cfg.get("mqtt", {})

    def get_scene_name(self) -> Optional[str]:
        """Return configured scene name for lookup, or None to accept all scenes."""
        return self._zone_cfg.get("scene_name")

    def get_scene_id(self) -> Optional[str]:
        """Return resolved scene UUID, or None to accept all scenes."""
        return self._resolved_scene_id

    def set_scene_id(self, scene_id: str) -> None:
        """Set the resolved scene UUID at runtime (from scene_name lookup)."""
        self._resolved_scene_id = scene_id
        logger.info("Scene ID resolved", scene_id=scene_id)

    def get_scene_data_topic(self) -> str:
        return self.get_mqtt_config().get(
            "scene_data_topic_pattern", "scenescape/data/scene/+/+"
        )

    def get_region_event_topic(self) -> str:
        return self.get_mqtt_config().get(
            "region_event_topic_pattern", "scenescape/event/region/+/+/+"
        )

    def get_image_topic_pattern(self) -> str:
        return self.get_mqtt_config().get(
            "image_topic_pattern", "scenescape/image/camera/+"
        )

    def get_alert_topic_prefix(self) -> str:
        return self.get_mqtt_config().get("alert_topic_prefix", "lp/alerts")

    # ---- seaweedfs ----
    def get_seaweedfs_config(self) -> dict:
        return self._app_cfg.get("seaweedfs", {})

    # ---- external services ----
    def get_behavioral_analysis_config(self) -> dict:
        return self._app_cfg.get("behavioral_analysis", {})

    # ---- rules ----
    def get_rules_config(self) -> dict:
        return self._app_cfg.get("rules", {})

    # ---- zones (dynamic) ----
    def get_zones(self) -> Dict[str, dict]:
        """Return {region_uuid: {name, type}} — live, thread-safe."""
        with self._zone_lock:
            return dict(self._zones)

    def get_zone_type(self, region_id: str) -> Optional[str]:
        with self._zone_lock:
            zone = self._zones.get(region_id)
        return zone["type"] if zone else None

    def get_zone_name(self, region_id: str) -> Optional[str]:
        with self._zone_lock:
            zone = self._zones.get(region_id)
        return zone["name"] if zone else None

    def set_zone(self, region_id: str, name: str, zone_type: str, **extra) -> None:
        """Add or update a single zone mapping at runtime."""
        with self._zone_lock:
            self._zones[region_id] = {"name": name, "type": zone_type, **extra}
        logger.info("Zone set", region_id=region_id, name=name, type=zone_type)

    def remove_zone(self, region_id: str) -> bool:
        """Remove a zone mapping. Returns True if it existed."""
        with self._zone_lock:
            removed = self._zones.pop(region_id, None)
        if removed:
            logger.info("Zone removed", region_id=region_id)
        return removed is not None

    def merge_zones(self, new_zones: Dict[str, dict]) -> int:
        """Merge discovered zones into the live map. Returns count added."""
        added = 0
        with self._zone_lock:
            for rid, zinfo in new_zones.items():
                if rid not in self._zones:
                    self._zones[rid] = zinfo
                    added += 1
        logger.info("Zones merged", added=added, total=len(self._zones))
        return added

    # ---- zone name map ----
    def get_zone_name_map(self) -> Dict[str, str]:
        """Return {region_name: zone_type} from zone_config."""
        return dict(self._zone_name_map)

    # ---- scenescape api ----
    def get_scenescape_api_config(self) -> dict:
        return self._zone_cfg.get("scenescape_api", {})
