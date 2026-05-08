#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Import a SceneScape scene configuration from a zip file.
# If the scene already exists, prompts the user before replacing.
#
# Usage:
#   ./scenescape/scripts/import-scene.sh <app-dir>
#
# The script reads scene_zip and scene_name from zone_config.json,
# locates the zip under scenescape/webserver/, authenticates with
# the SceneScape REST API, and imports (or replaces) the scene.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENESCAPE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
APP_DIR="${1:-}"

if [ -z "${APP_DIR}" ]; then
    echo "Usage: $0 <app-dir>"
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
print(f'API_BASE_URL=\"{api.get(\"base_url\", \"https://localhost:8443\")}\"')
ss = cfg.get('scenescape', {})
print(f'API_USER=\"{ss.get(\"api_user\", \"admin\")}\"')
" 2>/dev/null)"

if [ -z "${SCENE_NAME}" ]; then
    echo -e "${RED}ERROR: No scene_name in zone_config.json${NC}"
    exit 1
fi

if [ -z "${SCENE_ZIP}" ]; then
    SCENE_ZIP="$(echo "${SCENE_NAME}" | tr ' ' '-' | tr '[:upper:]' '[:lower:]').zip"
fi

ZIP_PATH="${SCENESCAPE_DIR}/webserver/${SCENE_ZIP}"
if [ ! -f "${ZIP_PATH}" ]; then
    echo -e "${RED}ERROR: Scene zip not found: ${ZIP_PATH}${NC}"
    echo "Run 'make export-scene' first to create the zip, or place it manually."
    exit 1
fi

# Read SUPASS
if [ -z "${SUPASS}" ] && [ -f "${ENV_FILE}" ]; then
    SUPASS=$(grep '^SUPASS=' "${ENV_FILE}" 2>/dev/null | cut -d= -f2-)
fi
if [ -z "${SUPASS}" ]; then
    SUPASS=$(cat "${SCENESCAPE_DIR}/secrets/supass" 2>/dev/null || echo "")
fi
if [ -z "${SUPASS}" ]; then
    echo -e "${RED}ERROR: SUPASS not found. Run init first or set SUPASS env var.${NC}"
    exit 1
fi

# ---- SceneScape API connection ----
SCENESCAPE_URL="${API_BASE_URL}"
CA_CERT="${SCENESCAPE_DIR}/secrets/certs/scenescape-ca.pem"
CURL_TLS_FLAGS="-k"
if [ -f "${CA_CERT}" ]; then
    CURL_TLS_FLAGS="--cacert ${CA_CERT}"
fi

echo -e "${GREEN}=== SceneScape Scene Import ===${NC}"
echo "  Scene:  ${SCENE_NAME}"
echo "  Zip:    ${ZIP_PATH}"
echo "  API:    ${SCENESCAPE_URL}"
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

# ---- Check if scene already exists ----
echo -e "${YELLOW}[2/4] Checking for existing scene...${NC}"
SCENES_RESPONSE=$(curl -s ${CURL_TLS_FLAGS} \
    -H "Authorization: token ${TOKEN}" \
    "${SCENESCAPE_URL}/api/v1/scenes" 2>/dev/null)

EXISTING_UID=$(python3 -c "
import json, sys
scenes_raw = '''${SCENES_RESPONSE}'''
try:
    scenes = json.loads(scenes_raw)
except:
    sys.exit(0)
scene_list = scenes if isinstance(scenes, list) else scenes.get('results', scenes.get('scenes', []))
for s in scene_list:
    if s.get('name', '').lower() == '${SCENE_NAME}'.lower():
        print(s.get('uid', ''))
        break
" 2>/dev/null || echo "")

if [ -n "${EXISTING_UID}" ]; then
    echo -e "  ${YELLOW}Scene '${SCENE_NAME}' already exists (uid=${EXISTING_UID}).${NC}"
    read -p "  Replace existing scene? [y/N] " CONFIRM
    if [ "${CONFIRM}" != "y" ] && [ "${CONFIRM}" != "Y" ]; then
        echo "  Import cancelled."
        exit 0
    fi

    # Delete existing scene
    echo -e "${YELLOW}[3/4] Deleting existing scene...${NC}"
    DELETE_RESPONSE=$(curl -s ${CURL_TLS_FLAGS} \
        -X DELETE \
        -H "Authorization: token ${TOKEN}" \
        "${SCENESCAPE_URL}/api/v1/scenes/${EXISTING_UID}" 2>/dev/null)
    echo "  Deleted scene ${EXISTING_UID}."
else
    echo "  No existing scene found. Proceeding with import."
fi

# ---- Extract and import scene JSON + map ----
echo -e "${YELLOW}[$([ -n "${EXISTING_UID}" ] && echo "4" || echo "3")/4] Importing scene from zip...${NC}"

WORK_DIR=$(mktemp -d)
trap "rm -rf ${WORK_DIR}" EXIT

unzip -o -d "${WORK_DIR}" "${ZIP_PATH}" >/dev/null

# Find the JSON file in the zip
SCENE_JSON=$(find "${WORK_DIR}" -name "*.json" -type f | head -1)
if [ -z "${SCENE_JSON}" ]; then
    echo -e "${RED}ERROR: No JSON file found in ${SCENE_ZIP}${NC}"
    exit 1
fi

# Find the map image in the zip
MAP_IMAGE=$(find "${WORK_DIR}" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | head -1)

# Read scene UID and data from JSON
SCENE_UID=$(python3 -c "import json; print(json.load(open('${SCENE_JSON}')).get('uid',''))" 2>/dev/null)
SCENE_DATA=$(python3 << 'IMPORT_SCRIPT'
import json, sys

with open("SCENE_JSON_PATH") as f:
    data = json.load(f)

cameras = data.pop("cameras", [])
regions = data.pop("regions", [])

# Remove map field (will upload separately)
data.pop("map", None)

# Output as structured JSON for the rest of the script
output = {"scene": data, "cameras": cameras, "regions": regions}
print(json.dumps(output))
IMPORT_SCRIPT
)
# Fix the placeholder path
SCENE_DATA=$(echo "${SCENE_DATA}" | sed "s|SCENE_JSON_PATH|${SCENE_JSON}|g")
SCENE_DATA=$(python3 -c "
import json, sys
with open('${SCENE_JSON}') as f:
    data = json.load(f)
cameras = data.pop('cameras', [])
regions = data.pop('regions', [])
data.pop('map', None)
output = {'scene': data, 'cameras': cameras, 'regions': regions}
print(json.dumps(output))
" 2>/dev/null)

# Create scene
SCENE_CREATE_DATA=$(echo "${SCENE_DATA}" | python3 -c "
import json, sys
d = json.load(sys.stdin)
s = d['scene']
print(json.dumps({
    'uid': s.get('uid', ''),
    'name': s.get('name', ''),
    'map_type': s.get('map_type', 'map_upload'),
    'use_tracker': s.get('use_tracker', True),
    'output_lla': s.get('output_lla', False),
    'scale': s.get('scale', 50),
    'regulated_rate': s.get('regulated_rate', 30),
    'external_update_rate': s.get('external_update_rate', 30),
    'camera_calibration': s.get('camera_calibration', 'Manual'),
}))
")

echo "  Creating scene..."
if [ -n "${MAP_IMAGE}" ]; then
    CREATE_RESPONSE=$(curl -s ${CURL_TLS_FLAGS} \
        -X POST \
        -H "Authorization: token ${TOKEN}" \
        -F "data=${SCENE_CREATE_DATA}" \
        -F "map=@${MAP_IMAGE}" \
        "${SCENESCAPE_URL}/api/v1/scenes" 2>/dev/null)
else
    CREATE_RESPONSE=$(curl -s ${CURL_TLS_FLAGS} \
        -X POST \
        -H "Authorization: token ${TOKEN}" \
        -H "Content-Type: application/json" \
        -d "${SCENE_CREATE_DATA}" \
        "${SCENESCAPE_URL}/api/v1/scenes" 2>/dev/null)
fi

# Check for error
ERROR=$(echo "${CREATE_RESPONSE}" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('error','') or r.get('detail',''))" 2>/dev/null || echo "")
if [ -n "${ERROR}" ]; then
    echo -e "${RED}ERROR creating scene: ${ERROR}${NC}"
    echo "  Full response: ${CREATE_RESPONSE}"
    exit 1
fi
echo "  Scene created."

# Create cameras
CAM_COUNT=$(echo "${SCENE_DATA}" | python3 -c "import json,sys; print(len(json.load(sys.stdin).get('cameras',[])))" 2>/dev/null)
if [ "${CAM_COUNT}" -gt 0 ] 2>/dev/null; then
    echo "  Creating ${CAM_COUNT} camera(s)..."
    echo "${SCENE_DATA}" | python3 -c "
import json, sys
d = json.load(sys.stdin)
for cam in d.get('cameras', []):
    print(json.dumps(cam))
" 2>/dev/null | while read -r CAM_JSON; do
        CAM_NAME=$(echo "${CAM_JSON}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('name',''))" 2>/dev/null)
        curl -s ${CURL_TLS_FLAGS} \
            -X POST \
            -H "Authorization: token ${TOKEN}" \
            -H "Content-Type: application/json" \
            -d "${CAM_JSON}" \
            "${SCENESCAPE_URL}/api/v1/cameras" >/dev/null 2>&1
        echo "    Camera: ${CAM_NAME}"
    done
fi

# Create regions
REG_COUNT=$(echo "${SCENE_DATA}" | python3 -c "import json,sys; print(len(json.load(sys.stdin).get('regions',[])))" 2>/dev/null)
if [ "${REG_COUNT}" -gt 0 ] 2>/dev/null; then
    echo "  Creating ${REG_COUNT} region(s)..."
    echo "${SCENE_DATA}" | python3 -c "
import json, sys
d = json.load(sys.stdin)
for reg in d.get('regions', []):
    print(json.dumps(reg))
" 2>/dev/null | while read -r REG_JSON; do
        REG_NAME=$(echo "${REG_JSON}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('name',''))" 2>/dev/null)
        curl -s ${CURL_TLS_FLAGS} \
            -X POST \
            -H "Authorization: token ${TOKEN}" \
            -H "Content-Type: application/json" \
            -d "${REG_JSON}" \
            "${SCENESCAPE_URL}/api/v1/regions" >/dev/null 2>&1
        echo "    Region: ${REG_NAME}"
    done
fi

echo ""
echo -e "${GREEN}=== Import complete ===${NC}"
echo "  Scene '${SCENE_NAME}' imported successfully."
echo ""
echo "  View at: ${SCENESCAPE_URL}"
