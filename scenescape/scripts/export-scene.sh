#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Export SceneScape scene configuration (scene, cameras, regions, map image)
# to a zip file that can be used by scene-import.sh / init.sh.
#
# Usage:
#   ./scenescape/scripts/export-scene.sh <app-dir> [--scene "scene name"]
#
# Examples:
#   ./scenescape/scripts/export-scene.sh person-of-interest
#   ./scenescape/scripts/export-scene.sh person-of-interest --scene "conference room"
#
# The script reads SceneScape connection info from zone_config.json,
# fetches the scene/cameras/regions via REST API, downloads the map image,
# and packages everything into a zip under scenescape/webserver/.
#
# Output: scenescape/webserver/<scene-zip-name>.zip
# The zip filename is taken from zone_config.json "scene_zip" field.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENESCAPE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
APP_DIR="${1:-}"

if [ -z "${APP_DIR}" ]; then
    echo "Usage: $0 <app-dir> [--scene \"scene name\"]"
    echo "  <app-dir> is the application directory containing configs/"
    exit 1
fi

APP_DIR="$(cd "${APP_DIR}" && pwd)"
ZONE_CONFIG="${APP_DIR}/configs/zone_config.json"
ENV_FILE="${APP_DIR}/docker/.env"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

# Parse optional --scene argument
shift
SCENE_NAME_OVERRIDE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --scene) SCENE_NAME_OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Read zone_config.json ----
if [ ! -f "${ZONE_CONFIG}" ]; then
    echo -e "${RED}ERROR: zone_config.json not found at ${ZONE_CONFIG}${NC}"
    exit 1
fi

eval "$(python3 -c "
import json
cfg = json.load(open('${ZONE_CONFIG}'))
print(f'SCENE_NAME=\"{cfg.get(\"scene_name\", \"\")}\"')
print(f'SCENE_ZIP=\"{cfg.get(\"scene_zip\", \"\")}\"')
api = cfg.get('scenescape_api', {})
print(f'API_BASE_URL=\"{api.get(\"base_url\", \"https://localhost\")}\"')
ss = cfg.get('scenescape', {})
print(f'API_USER=\"{ss.get(\"api_user\", \"admin\")}\"')
" 2>/dev/null)"

# Override scene name if --scene was passed
if [ -n "${SCENE_NAME_OVERRIDE}" ]; then
    SCENE_NAME="${SCENE_NAME_OVERRIDE}"
fi

if [ -z "${SCENE_NAME}" ]; then
    echo -e "${RED}ERROR: No scene_name in zone_config.json and no --scene flag${NC}"
    exit 1
fi

if [ -z "${SCENE_ZIP}" ]; then
    # Generate zip name from scene name
    SCENE_ZIP="$(echo "${SCENE_NAME}" | tr ' ' '-' | tr '[:upper:]' '[:lower:]').zip"
fi

# Read SUPASS from .env or environment
if [ -z "${SUPASS}" ] && [ -f "${ENV_FILE}" ]; then
    SUPASS=$(grep '^SUPASS=' "${ENV_FILE}" 2>/dev/null | cut -d= -f2-)
fi
if [ -z "${SUPASS}" ]; then
    SUPASS=$(cat "${SCENESCAPE_DIR}/secrets/supass" 2>/dev/null || echo "")
fi
if [ -z "${SUPASS}" ]; then
    echo -e "${RED}ERROR: SUPASS not found. Run init.sh first or set SUPASS env var.${NC}"
    exit 1
fi

# ---- SceneScape API connection ----
SCENESCAPE_URL="${API_BASE_URL}"
CA_CERT="${SCENESCAPE_DIR}/secrets/certs/scenescape-ca.pem"
CURL_TLS_FLAGS="-k"
if [ -f "${CA_CERT}" ]; then
    CURL_TLS_FLAGS="--cacert ${CA_CERT}"
fi

OUTPUT_DIR="${SCENESCAPE_DIR}/webserver"
WORK_DIR=$(mktemp -d)
trap "rm -rf ${WORK_DIR}" EXIT

echo -e "${GREEN}=== SceneScape Scene Export ===${NC}"
echo "  Scene:      ${SCENE_NAME}"
echo "  Output:     ${OUTPUT_DIR}/${SCENE_ZIP}"
echo "  API:        ${SCENESCAPE_URL}"
echo ""

# ---- Authenticate ----
echo -e "${YELLOW}[1/4] Authenticating...${NC}"
AUTH_RESPONSE=$(curl -s ${CURL_TLS_FLAGS} \
    -X POST "${SCENESCAPE_URL}/api/v1/auth" \
    -H "Content-Type: application/json" \
    -d "{\"username\": \"${API_USER}\", \"password\": \"${SUPASS}\"}" 2>/dev/null)

TOKEN=$(echo "$AUTH_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('token',''))" 2>/dev/null || echo "")

if [ -z "${TOKEN}" ]; then
    echo -e "${RED}ERROR: Failed to authenticate. Response: ${AUTH_RESPONSE}${NC}"
    exit 1
fi
echo "  Authenticated successfully."

# ---- Fetch scenes ----
echo -e "${YELLOW}[2/4] Fetching scene data...${NC}"
SCENES_RESPONSE=$(curl -s ${CURL_TLS_FLAGS} \
    -H "Authorization: token ${TOKEN}" \
    "${SCENESCAPE_URL}/api/v1/scenes" 2>/dev/null)

# Extract target scene and its cameras/regions using Python
python3 << EXPORT_SCRIPT
import json, sys, os, base64, urllib.request, ssl

scenes_raw = '''${SCENES_RESPONSE}'''
try:
    scenes = json.loads(scenes_raw)
except json.JSONDecodeError:
    print(f"ERROR: Failed to parse scenes response", file=sys.stderr)
    sys.exit(1)

# Handle both list and dict-with-results formats
if isinstance(scenes, dict):
    scene_list = scenes.get("results", scenes.get("scenes", []))
elif isinstance(scenes, list):
    scene_list = scenes
else:
    print(f"ERROR: Unexpected scenes format: {type(scenes)}", file=sys.stderr)
    sys.exit(1)

scene_name = "${SCENE_NAME}"
target = None
for s in scene_list:
    name = s.get("name", "")
    if name.lower() == scene_name.lower():
        target = s
        break

if not target:
    available = [s.get("name", "?") for s in scene_list]
    print(f"ERROR: Scene '{scene_name}' not found. Available: {available}", file=sys.stderr)
    sys.exit(1)

scene_uid = target.get("uid", "")
print(f"  Found scene: {target['name']} (uid={scene_uid})")

# Build the export JSON (same format as SceneScape import expects)
export = {
    "uid": scene_uid,
    "name": target.get("name", ""),
    "map_type": target.get("map_type", "map_upload"),
    "use_tracker": target.get("use_tracker", True),
    "output_lla": target.get("output_lla", False),
    "map": "/media/base_image.png",
    "scale": target.get("scale", 50),
    "regulated_rate": target.get("regulated_rate", 30),
    "external_update_rate": target.get("external_update_rate", 30),
    "camera_calibration": target.get("camera_calibration", "Manual"),
}

# Copy cameras
cameras = target.get("cameras", [])
export["cameras"] = []
for cam in cameras:
    cam_export = {
        "uid": cam.get("uid", cam.get("name", "")),
        "name": cam.get("name", ""),
        "scene": scene_uid,
        "cv_subsystem": cam.get("cv_subsystem", "AUTO"),
        "undistort": cam.get("undistort", False),
        "use_camera_pipeline": cam.get("use_camera_pipeline", False),
    }
    # Copy calibration data if present
    for key in ("intrinsics", "transform_type", "transforms", "translation",
                "rotation", "scale", "resolution", "modelconfig"):
        if key in cam:
            cam_export[key] = cam[key]
    export["cameras"].append(cam_export)

print(f"  Cameras: {[c['name'] for c in export['cameras']]}")

# Copy regions
regions = target.get("regions", [])
export["regions"] = []
for reg in regions:
    reg_export = {
        "uid": reg.get("uid", ""),
        "name": reg.get("name", ""),
        "points": reg.get("points", []),
        "scene": scene_uid,
        "buffer_size": reg.get("buffer_size", 0),
        "height": reg.get("height", 1),
        "volumetric": reg.get("volumetric", False),
    }
    if "color_ranges" in reg:
        reg_export["color_ranges"] = reg["color_ranges"]
    export["regions"].append(reg_export)

print(f"  Regions: {[r['name'] for r in export['regions']]}")

# Write scene JSON
work_dir = "${WORK_DIR}"
json_path = os.path.join(work_dir, f"{target['name']}.json")
with open(json_path, "w") as f:
    json.dump(export, f, indent=2)
print(f"  Wrote: {json_path}")

# Write map URL for download
map_url = target.get("map", "")
if map_url:
    with open(os.path.join(work_dir, "map_url.txt"), "w") as f:
        f.write(map_url)
EXPORT_SCRIPT

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to extract scene data.${NC}"
    exit 1
fi

# ---- Download map image ----
echo -e "${YELLOW}[3/4] Downloading map image...${NC}"
MAP_IMAGE="${WORK_DIR}/${SCENE_NAME}.png"
MAP_URL=""
if [ -f "${WORK_DIR}/map_url.txt" ]; then
    MAP_URL=$(cat "${WORK_DIR}/map_url.txt")
fi

if [ -n "${MAP_URL}" ]; then
    # Handle relative URLs
    case "${MAP_URL}" in
        http*) FULL_MAP_URL="${MAP_URL}" ;;
        *)     FULL_MAP_URL="${SCENESCAPE_URL}${MAP_URL}" ;;
    esac

    curl -s ${CURL_TLS_FLAGS} \
        -H "Authorization: token ${TOKEN}" \
        -o "${MAP_IMAGE}" \
        "${FULL_MAP_URL}" 2>/dev/null

    if [ -f "${MAP_IMAGE}" ] && [ -s "${MAP_IMAGE}" ]; then
        echo "  Downloaded map image ($(du -h "${MAP_IMAGE}" | cut -f1))"
    else
        echo -e "${YELLOW}  WARNING: Could not download map image from ${FULL_MAP_URL}${NC}"
        echo "  Creating placeholder image..."
        python3 -c "
import struct, zlib
# Minimal 1x1 white PNG
def png():
    sig = b'\x89PNG\r\n\x1a\n'
    def chunk(t, d): return struct.pack('>I',len(d)) + t + d + struct.pack('>I',zlib.crc32(t+d)&0xffffffff)
    ihdr = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
    return sig + chunk(b'IHDR',ihdr) + chunk(b'IDAT',zlib.compress(b'\x00\xff\xff\xff')) + chunk(b'IEND',b'')
open('${MAP_IMAGE}','wb').write(png())
"
    fi
else
    echo "  No map URL found — creating placeholder..."
    python3 -c "
import struct, zlib
def png():
    sig = b'\x89PNG\r\n\x1a\n'
    def chunk(t, d): return struct.pack('>I',len(d)) + t + d + struct.pack('>I',zlib.crc32(t+d)&0xffffffff)
    ihdr = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
    return sig + chunk(b'IHDR',ihdr) + chunk(b'IDAT',zlib.compress(b'\x00\xff\xff\xff')) + chunk(b'IEND',b'')
open('${MAP_IMAGE}','wb').write(png())
"
fi
rm -f "${WORK_DIR}/map_url.txt"

# ---- Package zip ----
echo -e "${YELLOW}[4/4] Creating zip file...${NC}"

# Backup existing zip if present
if [ -f "${OUTPUT_DIR}/${SCENE_ZIP}" ]; then
    echo "  Backing up existing ${SCENE_ZIP} → ${SCENE_ZIP}.bak"
    cp "${OUTPUT_DIR}/${SCENE_ZIP}" "${OUTPUT_DIR}/${SCENE_ZIP}.bak"
fi

cd "${WORK_DIR}"
zip -j "${OUTPUT_DIR}/${SCENE_ZIP}" "${SCENE_NAME}.json" "${SCENE_NAME}.png"

echo ""
echo -e "${GREEN}=== Export complete ===${NC}"
echo ""
echo "  Output: ${OUTPUT_DIR}/${SCENE_ZIP}"
echo ""
echo "Contents:"
unzip -l "${OUTPUT_DIR}/${SCENE_ZIP}"
echo ""
echo "This zip can be imported by:"
echo "  1. scene-import.sh (automatic on docker compose up)"
echo "  2. init.sh (set scene_zip in zone_config.json)"
echo "  3. SceneScape UI > Import Scene"
