#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENESCAPE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
APP_DIR="${1:-}"

if [ -z "${APP_DIR}" ]; then
    echo "Usage: $0 <app-dir>"
    exit 1
fi

APP_DIR="$(cd "${APP_DIR}" && pwd)"
ZONE_CONFIG="${ZONE_CONFIG:-${APP_DIR}/configs/zone_config.json}"
ENV_FILE="${APP_DIR}/docker/.env"

if [ ! -f "${ZONE_CONFIG}" ]; then
    echo "Skipping SceneScape cleanup: zone_config.json not found at ${ZONE_CONFIG}"
    exit 0
fi

if [ ! -f "${ENV_FILE}" ]; then
    echo "Skipping SceneScape cleanup: env file not found at ${ENV_FILE}"
    exit 0
fi

readarray -t SCENE_INFO < <(python3 -c '
import json, sys
cfg = json.load(open(sys.argv[1]))
scene_name = cfg.get("scene_name", "").strip()
base_url = cfg.get("scenescape_api", {}).get("base_url", "https://localhost").strip()
print(scene_name)
print(base_url)
for camera in cfg.get("cameras", []):
    if isinstance(camera, dict) and camera.get("name"):
        print(camera["name"])
' "${ZONE_CONFIG}")

SCENE_NAME="${SCENE_INFO[0]:-}"
CONFIG_SCENESCAPE_URL="${SCENE_INFO[1]:-https://localhost}"
CAMERA_NAMES=("${SCENE_INFO[@]:2}")

if [ -z "${SCENE_NAME}" ]; then
    echo "Skipping SceneScape cleanup: scene_name is empty in ${ZONE_CONFIG}"
    exit 0
fi

SUPASS="$(grep '^SUPASS=' "${ENV_FILE}" | head -n1 | cut -d= -f2-)"
SCENESCAPE_USER="${SCENESCAPE_USER:-admin}"

if [ -z "${SUPASS}" ]; then
    echo "Skipping SceneScape cleanup: SUPASS not found in ${ENV_FILE}"
    exit 0
fi

python3 - <<'PY' "${CONFIG_SCENESCAPE_URL}" "${SCENESCAPE_USER}" "${SUPASS}" "${SCENE_NAME}" "${CAMERA_NAMES[@]}"
import json
import ssl
import sys
import urllib.error
import urllib.request
from urllib.parse import urlparse

config_base_url, username, password, scene_name = sys.argv[1:5]
camera_names = sys.argv[5:]
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def unique_preserve_order(values):
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


parsed = urlparse(config_base_url if "://" in config_base_url else f"https://{config_base_url}")
config_host = parsed.hostname or ""
config_scheme = parsed.scheme or "https"
fallback_base = f"{config_scheme}://{config_host}" if config_host else ""

candidate_urls = unique_preserve_order([
    "https://localhost",
    "https://127.0.0.1",
    config_base_url,
    fallback_base,
])

def request(path, method="GET", token=None, payload=None):
    headers = {}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
    if token:
        headers["Authorization"] = f"token {token}"
    req = urllib.request.Request(base_url.rstrip("/") + path, data=data, headers=headers, method=method)
    return urllib.request.urlopen(req, context=ctx, timeout=30)

token = ""
base_url = ""
auth_error = ""
for candidate in candidate_urls:
    try:
        base_url = candidate
        auth = request("/api/v1/auth", method="POST", payload={"username": username, "password": password})
        token = json.loads(auth.read().decode()).get("token", "")
        if token:
            print(f"Using SceneScape endpoint: {base_url}")
            break
    except Exception as exc:
        auth_error = f"{candidate}: {exc}"
        token = ""

if not token:
    print(f"Skipping SceneScape cleanup: failed to authenticate ({auth_error})")
    sys.exit(0)

try:
    response = request("/api/v1/scenes", token=token)
    scenes = json.loads(response.read().decode())
except Exception as exc:
    print(f"Skipping SceneScape cleanup: failed to list scenes ({exc})")
    sys.exit(0)

if isinstance(scenes, dict):
    scene_list = scenes.get("results", scenes.get("scenes", []))
elif isinstance(scenes, list):
    scene_list = scenes
else:
    print(f"Skipping SceneScape cleanup: unexpected scenes payload type {type(scenes).__name__}")
    sys.exit(0)

matches = []
for scene in scene_list:
    if not isinstance(scene, dict):
        continue
    name = (scene.get("name") or scene.get("scene_name") or "").strip()
    uid = scene.get("uid") or scene.get("id") or ""
    if not uid:
        continue
    if name == scene_name or name.startswith(f"{scene_name}-"):
        matches.append((name, uid))

if not matches:
    print(f"No SceneScape scenes matched '{scene_name}'.")
    sys.exit(0)

deleted = 0
for name, uid in matches:
    try:
        request(f"/api/v1/scene/{uid}", method="DELETE", token=token).read()
        print(f"Deleted SceneScape scene: {name} ({uid})")
        deleted += 1
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="ignore")
        print(f"Failed to delete SceneScape scene {name} ({uid}): HTTP {exc.code} {body}")
    except Exception as exc:
        print(f"Failed to delete SceneScape scene {name} ({uid}): {exc}")

print(f"SceneScape cleanup finished. Deleted {deleted} scene(s).")

deleted_cameras = 0
if camera_names:
    try:
        response = request("/api/v1/cameras", token=token)
        cameras_payload = json.loads(response.read().decode())
        if isinstance(cameras_payload, dict):
            camera_list = cameras_payload.get("results", cameras_payload.get("cameras", []))
        elif isinstance(cameras_payload, list):
            camera_list = cameras_payload
        else:
            camera_list = []
    except Exception as exc:
        print(f"Skipping orphan camera cleanup: failed to list cameras ({exc})")
        camera_list = []

    wanted = set(camera_names)
    for camera in camera_list:
        if not isinstance(camera, dict):
            continue
        name = (camera.get("name") or camera.get("uid") or "").strip()
        uid = (camera.get("uid") or camera.get("id") or "").strip()
        if not uid or name not in wanted:
            continue
        try:
            request(f"/api/v1/camera/{uid}", method="DELETE", token=token).read()
            print(f"Deleted orphan SceneScape camera: {name} ({uid})")
            deleted_cameras += 1
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="ignore")
            print(f"Failed to delete SceneScape camera {name} ({uid}): HTTP {exc.code} {body}")
        except Exception as exc:
            print(f"Failed to delete SceneScape camera {name} ({uid}): {exc}")

    print(f"Orphan camera cleanup finished. Deleted {deleted_cameras} camera(s).")
PY