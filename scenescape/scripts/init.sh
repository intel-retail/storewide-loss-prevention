#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Initialize secrets, read zone_config.json, generate DLStreamer config,
# and generate .env for the full-stack deployment.
#
# Usage: ./scenescape/scripts/init.sh <app-dir>
# Example: ../scenescape/scripts/init.sh /path/to/suspicious-activity-detection

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENESCAPE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
APP_DIR="${1:-}"

if [ -z "${APP_DIR}" ]; then
    echo "Usage: $0 <app-dir>"
    echo "  <app-dir> is the application directory containing configs/ and docker/"
    exit 1
fi

APP_DIR="$(cd "${APP_DIR}" && pwd)"
APP_NAME="$(basename "${APP_DIR}")"
SECRETS_DIR="${SCENESCAPE_DIR}/secrets"
ENV_FILE="${APP_DIR}/docker/.env"
SAMPLE_DATA_DIR="${SCENESCAPE_DIR}/sample_data"
ZONE_CONFIG="${APP_DIR}/configs/zone_config.json"
DLSTREAMER_CONFIG="${SCENESCAPE_DIR}/dlstreamer-pipeline-server/${APP_NAME}-pipeline-config.json"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo -e "${GREEN}=== Storewide Loss Prevention - Full Stack Init ===${NC}"
echo ""

# ---- Step 1: Generate SceneScape secrets ----
echo -e "${YELLOW}[1/4] Generating SceneScape secrets...${NC}"
SECRETS_GENERATED=0
if [ -f "${SECRETS_DIR}/django/secrets.py" ] && [ -f "${SECRETS_DIR}/certs/scenescape-ca.pem" ]; then
    echo "  Secrets already exist, skipping generation."
    echo "  (To regenerate: make clean-secrets && make run-scenescape)"
else
    chmod +x "${SECRETS_DIR}/generate_secrets.sh"
    bash "${SECRETS_DIR}/generate_secrets.sh"
    SECRETS_GENERATED=1
fi

# ---- Step 2: Read zone_config.json ----
echo -e "${YELLOW}[2/4] Reading zone_config.json...${NC}"
if [ ! -f "${ZONE_CONFIG}" ]; then
    echo -e "${RED}ERROR: zone_config.json not found at ${ZONE_CONFIG}${NC}"
    exit 1
fi

# Read base scene config and stream_density
CONFIG_VALUES=$(python3 -c "
import json
cfg = json.load(open('${ZONE_CONFIG}'))
# Support both flat format and scenes array (backward compat)
scenes = cfg.get('scenes', [])
if scenes:
    s = scenes[0]
    sn = s.get('scene_name', '')
    cn = s.get('cameras', [s.get('camera_name', '')])[0] if s.get('cameras') or s.get('camera_name') else ''
    vf = s.get('video_file', '')
    sz = s.get('scene_zip', '')
    sd = cfg.get('stream_density', 1)
else:
    sn = cfg.get('scene_name', '')
    cn = cfg.get('camera_name', '')
    vf = cfg.get('video_file', '')
    sz = cfg.get('scene_zip', '')
    sd = cfg.get('stream_density', 1)
print(f'{sn}|{cn}|{vf}|{sz}|{sd}')
" 2>/dev/null)
IFS='|' read -r SCENE_NAME CAMERA_NAME VIDEO_FILE SCENE_ZIP_BASE STREAM_DENSITY <<< "${CONFIG_VALUES}"

# Defaults
STREAM_DENSITY="${STREAM_DENSITY:-1}"
VIDEO_FILE="${VIDEO_FILE:-lp-camera1.mp4}"
SCENE_ZIP_BASE="${SCENE_ZIP_BASE:-storewide-loss-prevention.zip}"

echo "  Scene name:      ${SCENE_NAME}"
echo "  Camera name:     ${CAMERA_NAME}"
echo "  Video file:      ${VIDEO_FILE}"
echo "  Base scene zip:  ${SCENE_ZIP_BASE}"
echo "  Stream density:  ${STREAM_DENSITY}"

if [ -z "${SCENE_NAME}" ] || [ -z "${CAMERA_NAME}" ]; then
    echo -e "${RED}ERROR: zone_config.json must have scene_name and camera_name${NC}"
    exit 1
fi

# Validate base scene zip exists
BASE_ZIP_PATH="${SCENESCAPE_DIR}/webserver/${SCENE_ZIP_BASE}"
if [ ! -f "${BASE_ZIP_PATH}" ]; then
    echo -e "${YELLOW}WARNING: Base scene zip not found at ${BASE_ZIP_PATH}${NC}"
    echo "  Scene import will be skipped. Import manually via SceneScape UI."
fi

# Validate video exists
VIDEO_PATH="${SAMPLE_DATA_DIR}/${VIDEO_FILE}"
if [ ! -f "${VIDEO_PATH}" ]; then
    echo -e "${YELLOW}WARNING: Video not found at ${VIDEO_PATH}${NC}"
    echo "  Place your video file in scenescape/sample_data/"
fi

SCENE_ZIP="${SCENE_ZIP_BASE}"

# ---- Step 3: Generate DLStreamer config + camera streams ----
echo -e "${YELLOW}[3/4] Generating DLStreamer pipeline config + camera streams...${NC}"

DLSTREAMER_TEMPLATE="${APP_DIR}/configs/pipeline-config.json"
if [ ! -f "${DLSTREAMER_TEMPLATE}" ]; then
    echo -e "${RED}ERROR: DLStreamer template not found at ${DLSTREAMER_TEMPLATE}${NC}"
    exit 1
fi

# Generate single DLStreamer pipeline config
# For multi-stream density, run: make stream-density
sed "s/{{CAMERA_NAME}}/${CAMERA_NAME}/g" "${DLSTREAMER_TEMPLATE}" > "${DLSTREAMER_CONFIG}"
echo "  Generated ${DLSTREAMER_CONFIG}"
echo "  Using pre-existing ${APP_DIR}/docker/docker-compose.cameras.yaml"

# Copy stream_density.py to webserver dir for scene-import container (if available)
STREAM_DENSITY_SCRIPT="${APP_DIR}/stream_density.py"
if [ -f "${STREAM_DENSITY_SCRIPT}" ]; then
    cp "${STREAM_DENSITY_SCRIPT}" "${SCENESCAPE_DIR}/webserver/stream_density.py"
fi

# ---- Step 4: Generate .env file ----
echo -e "${YELLOW}[4/4] Generating docker/.env...${NC}"

# Source AI-model settings from configs/.env.example (single source of truth).
# User edits VLM_MODEL_NAME / VLM_SOURCE_MODEL / VLM_PRECISION / TARGET_DEVICE /
# YOLO_MODEL_NAME / VLM_ENABLED there; init.sh propagates
# them into docker/.env, and download_models/model_download.sh reads the same
# file. Only AI-model keys are pulled — placeholders like SUPASS=<auto-generated>
# in .env.example are intentionally NOT sourced (those come from secrets/
# zone_config.json below).
ENV_EXAMPLE="${APP_DIR}/configs/.env.example"
AI_KEYS_REGEX='^(VLM_ENABLED|VLM_MODEL_NAME|VLM_SOURCE_MODEL|VLM_PRECISION|TARGET_DEVICE|YOLO_MODEL_NAME)='
if [ -f "${ENV_EXAMPLE}" ]; then
    AI_ENV_TMP="$(mktemp)"
    grep -E "${AI_KEYS_REGEX}" "${ENV_EXAMPLE}" > "${AI_ENV_TMP}" || true
    set -a
    # shellcheck disable=SC1090
    . "${AI_ENV_TMP}"
    set +a
    rm -f "${AI_ENV_TMP}"
    echo "  Loaded AI-model settings from ${ENV_EXAMPLE}"
fi
VLM_ENABLED="${VLM_ENABLED:-true}"
VLM_MODEL_NAME="${VLM_MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
VLM_SOURCE_MODEL="${VLM_SOURCE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
VLM_PRECISION="${VLM_PRECISION:-int8}"
TARGET_DEVICE="${TARGET_DEVICE:-GPU}"
YOLO_MODEL_NAME="${YOLO_MODEL_NAME:-yolo26n-pose}"

# Read generated secrets — honor SUPASS from environment if set
SUPASS="${SUPASS:-$(cat "${SECRETS_DIR}/supass" 2>/dev/null || echo "")}"
DBPASS=$(sed -nr "/DATABASE_PASSWORD=/s/.*'([^']+)'/\1/p" "${SECRETS_DIR}/django/secrets.py" 2>/dev/null || echo "")
CONTROLLER_AUTH=$(cat "${SECRETS_DIR}/controller.auth" 2>/dev/null || echo "")

USER_UID=$(id -u)
USER_GID=$(id -g)

# If secrets were freshly generated, remove stale DB volumes
if [ "${SECRETS_GENERATED}" = "1" ]; then
    echo "  New secrets generated — removing stale DB volumes..."
    docker volume rm storewide-lp_vol-db storewide-lp_vol-migrations 2>/dev/null || true
fi

if [ -f "${ENV_FILE}" ]; then
    echo "  ${ENV_FILE} already exists. Backing up to ${ENV_FILE}.bak"
    cp "${ENV_FILE}" "${ENV_FILE}.bak"
fi

cat > "${ENV_FILE}" <<EOF
# Auto-generated by init.sh — $(date -Iseconds)
SECRETSDIR=${SECRETS_DIR}
SUPASS=${SUPASS}
DATABASE_PASSWORD=${DBPASS}
CONTROLLER_AUTH=${CONTROLLER_AUTH}
UID=${USER_UID}
GID=${USER_GID}

# Scene (auto-populated from zone_config.json by init.sh)
SCENE_NAME=${SCENE_NAME}
CAMERA_NAME=${CAMERA_NAME}
SCENE_ZIP=${SCENE_ZIP}
VIDEO_FILE=${VIDEO_FILE}
STREAM_DENSITY=${STREAM_DENSITY}

# DLStreamer pipeline config (app-specific, generated by init.sh)
PIPELINE_CONFIG=../scenescape/dlstreamer-pipeline-server/${APP_NAME}-pipeline-config.json

# OpenVINO Models (comma-separated)
MODELS=person-detection-retail-0013,person-reidentification-retail-0277
MODEL_PRECISION=FP32

# SceneScape image versions
SCENESCAPE_REGISTRY=
SCENESCAPE_VERSION=v2026.0.0
DLSTREAMER_VERSION=2026.1.0-20260331-weekly-ubuntu24

# Store
STORE_NAME=Retail
STORE_ID=store_001

# LP
LP_SERVICE_PORT=8082
LOG_LEVEL=INFO

# SeaweedFS
SEAWEEDFS_S3_PORT=8333
SEAWEEDFS_MASTER_PORT=9333
SEAWEEDFS_VOLUME_PORT=8080

# SceneScape API (for zone auto-discovery)
SCENESCAPE_API_USER=admin
SCENESCAPE_API_PASSWORD=${SUPASS}

# AI Models (sourced from configs/.env.example)
VLM_ENABLED=${VLM_ENABLED}
VLM_MODEL_NAME=${VLM_MODEL_NAME}
VLM_SOURCE_MODEL=${VLM_SOURCE_MODEL}
VLM_PRECISION=${VLM_PRECISION}
TARGET_DEVICE=${TARGET_DEVICE}
YOLO_MODEL_NAME=${YOLO_MODEL_NAME}
EOF

echo ""
echo -e "${GREEN}=== Init complete ===${NC}"
echo ""
echo "Generated files:"
echo "  Secrets:          ${SECRETS_DIR}/"
echo "  DLStreamer config: ${DLSTREAMER_CONFIG}"
echo "  Env:              ${ENV_FILE}"
echo ""
echo "Scene: ${SCENE_NAME}"
echo "  Camera: ${CAMERA_NAME}  Video: ${VIDEO_FILE}  Zip: ${SCENE_ZIP}"
echo "  Stream density: ${STREAM_DENSITY}"
echo -e "  SUPASS: ${YELLOW}${SUPASS}${NC}"
echo ""
echo "To change scene/camera: edit configs/zone_config.json, then re-run init.sh"
echo ""
echo "Next steps:"
echo "  1. Place your video in scenescape/sample_data/${VIDEO_FILE}"
echo "  2. Place your scene zip in scenescape/webserver/${SCENE_ZIP}"
echo "  3. Start from your app directory:"
echo "       make run-scenescape   (SceneScape only)"
echo "       make demo             (full stack)"
echo ""
echo "  4. Open SceneScape UI:  https://localhost"
echo "     Login: admin / ${SUPASS}"
