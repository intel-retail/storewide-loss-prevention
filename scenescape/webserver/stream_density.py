#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0

"""
Stream Density scene cloning utility.

Clones a SceneScape scene .zip file N times with unique scene names,
camera IDs, and UIDs for stream density testing.

Usage (called by scene-import.sh):
    python3 stream_density.py clone-zip <base_zip> <output_dir> <scene_name> <camera_name> <density>

Also provides ``expand_scene_configs`` for use by swlp-service's ConfigService.
"""

import json
import os
import shutil
import sys
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List


def _unique_uid() -> str:
    return str(uuid.uuid4())


def clone_scene_zip(
    base_zip_path: str,
    output_dir: str,
    base_scene_name: str,
    base_camera_name: str,
    density: int,
) -> List[str]:
    """
    Clone *base_zip_path* ``density`` times into *output_dir*.

    Each clone gets:
      - A unique scene name:  ``{base_scene_name}``  (copy 1 = original),
        ``{base_scene_name}-2``, ``{base_scene_name}-3``, …
      - A unique camera name: ``{base_camera_name}`` (copy 1),
        ``{base_camera_name}-2``, …
      - Fresh UIDs for scene, cameras, and regions.

    Returns a list of generated zip file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Read the base zip
    with zipfile.ZipFile(base_zip_path, "r") as zf:
        # Find the scene JSON and map image
        json_name = None
        other_files: Dict[str, bytes] = {}
        for name in zf.namelist():
            data = zf.read(name)
            if name.endswith(".json"):
                json_name = name
                base_json = json.loads(data)
            else:
                other_files[name] = data

    if not json_name or not base_json:
        raise ValueError(f"No scene JSON found in {base_zip_path}")

    generated: List[str] = []

    for i in range(2, density + 1):
        scene_name = f"{base_scene_name}-{i}"
        camera_name = f"{base_camera_name}-{i}"

        # Deep copy and update the scene JSON
        scene_data = json.loads(json.dumps(base_json))

        # New scene UID
        new_scene_uid = _unique_uid()
        scene_data["uid"] = new_scene_uid
        scene_data["name"] = scene_name

        # Update cameras
        for cam in scene_data.get("cameras", []):
            cam["uid"] = camera_name
            cam["name"] = camera_name
            cam["scene"] = new_scene_uid

        # Update regions with new UIDs
        for region in scene_data.get("regions", []):
            region["uid"] = _unique_uid()
            region["scene"] = new_scene_uid

        # Build the new zip
        #
        # SceneScape's ImportScene.extractZip() flattens all files into a
        # directory named after the zip (minus .zip).  loadScene() then:
        #   1. Expects exactly ONE .json file in the extract dir
        #   2. Matches resource files by checking:  scene_name in filename
        #
        # So each clone needs:
        #   - A unique zip filename  → unique extract dir (no JSON collisions)
        #   - A unique JSON filename → one JSON per extract dir
        #   - Resource files renamed to contain scene_name for matching
        safe_name = scene_name.replace(" ", "-").lower()
        clone_zip_name = f"{safe_name}.zip"
        clone_zip_path = os.path.join(output_dir, clone_zip_name)

        with zipfile.ZipFile(clone_zip_path, "w", zipfile.ZIP_DEFLATED) as zf_out:
            # Write JSON with scene_name-based filename
            new_json_name = f"{safe_name}/{scene_name}.json"
            zf_out.writestr(new_json_name, json.dumps(scene_data, indent=2))

            # Copy resource files renamed to include the scene_name
            for orig_name, data in other_files.items():
                ext = os.path.splitext(orig_name)[1]  # e.g. ".png"
                new_resource_name = f"{safe_name}/{scene_name}{ext}"
                zf_out.writestr(new_resource_name, data)

        generated.append(clone_zip_path)

    return generated


def expand_scene_configs(base_scene: dict, density: int) -> List[dict]:
    """
    Expand a single base scene config dict into *density* copies.

    Used by swlp-service's ``ConfigService`` to generate scene subscription
    configs when ``stream_density > 1``.

    Each copy gets a unique scene_name and camera_name suffix.
    """
    if density <= 1:
        return [base_scene]

    configs = []
    base_name = base_scene.get("scene_name", "scene")
    base_camera = base_scene.get("cameras", [base_scene.get("camera_name", "camera")])[0] \
        if isinstance(base_scene.get("cameras"), list) else base_scene.get("camera_name", "camera")

    for i in range(1, density + 1):
        suffix = "" if i == 1 else f"-{i}"
        scene = dict(base_scene)
        scene["scene_name"] = f"{base_name}{suffix}"

        cam_name = f"{base_camera}{suffix}"
        if "cameras" in scene and isinstance(scene["cameras"], list):
            scene["cameras"] = [cam_name]
        else:
            scene["camera_name"] = cam_name

        configs.append(scene)

    return configs


def main():
    if len(sys.argv) < 2:
        print("Usage: stream_density.py clone-zip <base_zip> <output_dir> <scene_name> <camera_name> <density>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "clone-zip":
        if len(sys.argv) != 7:
            print("Usage: stream_density.py clone-zip <base_zip> <output_dir> <scene_name> <camera_name> <density>")
            sys.exit(1)

        base_zip = sys.argv[2]
        output_dir = sys.argv[3]
        scene_name = sys.argv[4]
        camera_name = sys.argv[5]
        density = int(sys.argv[6])

        generated = clone_scene_zip(base_zip, output_dir, scene_name, camera_name, density)
        for path in generated:
            print(path)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
