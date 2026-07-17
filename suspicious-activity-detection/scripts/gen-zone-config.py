#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Translate configs/scene-config.yaml -> configs/zone_config.json.

scene-config.yaml is the single source of truth the operator edits. The vendored
SceneScape scripts (init.sh, export-scene.sh, import-scene.sh) and
download_sample_video.sh still consume the legacy configs/zone_config.json, so we
generate that file from scene-config.yaml before running them. The generated JSON
is a build artifact (gitignored) and should not be edited by hand.

Runtime settings (scenes/cameras/zones/mqtt) live under the normal keys; the
deploy-only bits (scene_zip, sample_video_fps, per-camera sample-video URLs) live
inline on each scene, and the global model_precision at the top level. The
scene-understanding-service ignores the extra scene keys.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    sys.exit(
        "ERROR: PyYAML is required to generate zone_config.json from scene-config.yaml.\n"
        "       Install it with:  python3 -m pip install pyyaml"
    )

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
SCENE_CONFIG = CONFIGS_DIR / "scene-config.yaml"
ZONE_CONFIG = CONFIGS_DIR / "zone_config.json"


def build_zone_config(data: dict) -> dict:
    scenes = data.get("scenes") or []
    if not scenes:
        sys.exit(f"ERROR: no 'scenes' defined in {SCENE_CONFIG}")
    scene = scenes[0]

    scene_name = scene.get("scene_name", "")
    if not scene_name:
        sys.exit(f"ERROR: scenes[0].scene_name is required in {SCENE_CONFIG}")

    camera_names = [str(c) for c in (scene.get("cameras") or [])]
    if not camera_names:
        sys.exit(f"ERROR: scenes[0].cameras must list at least one camera in {SCENE_CONFIG}")

    sample_urls = scene.get("sample_video_urls") or {}
    fps = scene.get("sample_video_fps", 30)

    cameras = []
    for name in camera_names:
        cameras.append(
            {
                "name": name,
                "video": f"{name}.mp4",
                "video_url": sample_urls.get(name, ""),
                "fps": fps,
            }
        )

    return {
        "scene_name": scene_name,
        "scene_zip": scene.get("scene_zip", f"{scene_name.replace(' ', '-')}.zip"),
        "model_precision": data.get("model_precision", "FP32"),
        "scenescape_api": {
            "base_url": (data.get("scenescape_api") or {}).get(
                "base_url", "https://web.scenescape.intel.com"
            )
        },
        "cameras": cameras,
        "zones": scene.get("zones") or {},
    }


def main() -> None:
    if not SCENE_CONFIG.exists():
        sys.exit(f"ERROR: {SCENE_CONFIG} not found")

    data = yaml.safe_load(SCENE_CONFIG.read_text()) or {}
    zone_config = build_zone_config(data)
    ZONE_CONFIG.write_text(json.dumps(zone_config, indent=2) + "\n")
    print(f"  Generated {ZONE_CONFIG} from {SCENE_CONFIG.name}")


if __name__ == "__main__":
    main()
