#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Clone a SceneScape scene .zip N times with unique scene names and camera IDs.

Usage:
    python3 clone_scene_zip.py <base_zip> <output_dir> <scene_name> <camera_name> <density>

Outputs comma-separated zip filenames to stdout.
Each clone gets:
  - Scene name:  "{scene_name} {i}"
  - Camera uid/name: "{camera_name}-{i}"
  - Resource files renamed to match new scene name
"""

import json
import os
import shutil
import sys
import tempfile
import zipfile


def clone_zip(base_zip, output_dir, scene_name, camera_name, index):
    new_scene_name = f"{scene_name} {index}"
    new_camera = f"{camera_name}-{index}"
    base_stem = os.path.splitext(os.path.basename(base_zip))[0]
    new_zip_name = f"{base_stem}-{index}.zip"
    new_zip_path = os.path.join(output_dir, new_zip_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract (flatten like SceneScape does)
        with zipfile.ZipFile(base_zip, "r") as zf:
            for member in zf.namelist():
                filename = os.path.basename(member)
                if not filename:
                    continue
                with zf.open(member) as src, open(os.path.join(tmpdir, filename), "wb") as dst:
                    dst.write(src.read())

        files = os.listdir(tmpdir)
        json_files = [f for f in files if f.lower().endswith(".json")]
        resource_files = [f for f in files if not f.lower().endswith(".json")]

        if not json_files:
            print(f"ERROR: No JSON file in {base_zip}", file=sys.stderr)
            sys.exit(1)

        # Modify scene JSON
        json_path = os.path.join(tmpdir, json_files[0])
        with open(json_path, "r") as f:
            data = json.load(f)

        data["name"] = new_scene_name
        for cam in data.get("cameras", []):
            cam["uid"] = new_camera
            cam["name"] = new_camera

        # Write modified JSON with new name
        new_json_name = f"{new_scene_name}.json"
        new_json_path = os.path.join(tmpdir, new_json_name)
        with open(new_json_path, "w") as f:
            json.dump(data, f, indent=2)
        if json_path != new_json_path:
            os.remove(json_path)

        # Rename resource files to match new scene name
        renamed_resources = []
        for rf in resource_files:
            ext = os.path.splitext(rf)[1]
            new_rf_name = f"{new_scene_name}{ext}"
            new_rf_path = os.path.join(tmpdir, new_rf_name)
            old_rf_path = os.path.join(tmpdir, rf)
            if old_rf_path != new_rf_path:
                shutil.move(old_rf_path, new_rf_path)
            renamed_resources.append(new_rf_name)

        # Create zip (flat, no directory prefix)
        with zipfile.ZipFile(new_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(new_json_path, new_json_name)
            for rf_name in renamed_resources:
                zf.write(os.path.join(tmpdir, rf_name), rf_name)

    return new_zip_name


def main():
    if len(sys.argv) != 6:
        print(f"Usage: {sys.argv[0]} <base_zip> <output_dir> <scene_name> <camera_name> <density>",
              file=sys.stderr)
        sys.exit(1)

    base_zip = sys.argv[1]
    output_dir = sys.argv[2]
    scene_name = sys.argv[3]
    camera_name = sys.argv[4]
    density = int(sys.argv[5])

    if density < 1:
        print("ERROR: density must be >= 1", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    zip_names = []
    for i in range(1, density + 1):
        name = clone_zip(base_zip, output_dir, scene_name, camera_name, i)
        zip_names.append(name)

    # Print comma-separated zip names to stdout
    print(",".join(zip_names))


if __name__ == "__main__":
    main()
