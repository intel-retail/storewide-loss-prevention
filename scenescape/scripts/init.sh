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

SCENE_NAME=$(python3 -c "import json; print(json.load(open('${ZONE_CONFIG}')).get('scene_name',''))" 2>/dev/null)
CAMERA_NAME=$(python3 -c "import json; print(json.load(open('${ZONE_CONFIG}')).get('camera_name',''))" 2>/dev/null)
CAMERA_NAME_2=$(python3 -c "import json; print(json.load(open('${ZONE_CONFIG}')).get('camera_name_2',''))" 2>/dev/null)
SCENE_ZIP=$(python3 -c "import json; print(json.load(open('${ZONE_CONFIG}')).get('scene_zip',''))" 2>/dev/null)
VIDEO_FILE=$(python3 -c "import json; print(json.load(open('${ZONE_CONFIG}')).get('video_file',''))" 2>/dev/null)
VIDEO_FILE_2=$(python3 -c "import json; print(json.load(open('${ZONE_CONFIG}')).get('video_file_2',''))" 2>/dev/null)
MODELS=$(python3 -c "import json; print(json.load(open('${ZONE_CONFIG}')).get('models',''))" 2>/dev/null)

# Allow env var overrides
SCENE_ZIP="${SCENE_ZIP:-storewide-loss-prevention.zip}"
VIDEO_FILE="${VIDEO_FILE:-lp-camera1.mp4}"
VIDEO_FILE_2="${VIDEO_FILE_2:-}"

echo "  Scene name:  ${SCENE_NAME}"
echo "  Camera name: ${CAMERA_NAME}"
if [ -n "${CAMERA_NAME_2}" ]; then
    echo "  Camera name 2: ${CAMERA_NAME_2}"
fi
echo "  Scene zip:   ${SCENE_ZIP}"
echo "  Video file:  ${VIDEO_FILE}"
if [ -n "${VIDEO_FILE_2}" ]; then
    echo "  Video file 2: ${VIDEO_FILE_2}"
fi

if [ -z "${SCENE_NAME}" ] || [ -z "${CAMERA_NAME}" ]; then
    echo -e "${RED}ERROR: zone_config.json must have scene_name and camera_name${NC}"
    exit 1
fi

# Validate scene zip exists
SCENE_ZIP_PATH="${SCENESCAPE_DIR}/webserver/${SCENE_ZIP}"
if [ -n "${SCENE_ZIP}" ] && [ ! -f "${SCENE_ZIP_PATH}" ]; then
    echo -e "${YELLOW}WARNING: Scene zip not found at ${SCENE_ZIP_PATH}${NC}"
    echo "  Scene import will be skipped. Import manually via SceneScape UI."
fi

# Validate video exists
VIDEO_PATH="${SAMPLE_DATA_DIR}/${VIDEO_FILE}"
if [ ! -f "${VIDEO_PATH}" ]; then
    echo -e "${YELLOW}WARNING: Video not found at ${VIDEO_PATH}${NC}"
    echo "  Place your video file in scenescape/sample_data/"
fi

if [ -n "${VIDEO_FILE_2}" ]; then
    VIDEO_PATH_2="${SAMPLE_DATA_DIR}/${VIDEO_FILE_2}"
    if [ ! -f "${VIDEO_PATH_2}" ]; then
        echo -e "${YELLOW}WARNING: Video 2 not found at ${VIDEO_PATH_2}${NC}"
        echo "  Place your video file in scenescape/sample_data/"
    fi
fi

# ---- Step 3: Generate DLStreamer config.json (per camera) ----
echo -e "${YELLOW}[3/4] Generating DLStreamer pipeline configs...${NC}"

DLSTREAMER_TEMPLATE="${APP_DIR}/configs/pipeline-config.json"
if [ ! -f "${DLSTREAMER_TEMPLATE}" ]; then
    echo -e "${RED}ERROR: DLStreamer template not found at ${DLSTREAMER_TEMPLATE}${NC}"
    exit 1
fi

# Validate app-specific controller configs exist
if [ ! -f "${APP_DIR}/configs/tracker-config.json" ]; then
    echo -e "${YELLOW}WARNING: tracker-config.json not found in ${APP_DIR}/configs/${NC}"
    echo "  SceneScape will use default from scenescape/controller/tracker-config.json"
fi
if [ ! -f "${APP_DIR}/configs/reid-config.json" ]; then
    echo -e "${YELLOW}WARNING: reid-config.json not found in ${APP_DIR}/configs/${NC}"
    echo "  SceneScape will use default from scenescape/controller/reid-config.json"
fi

# Resolve controller config paths (app-specific or SceneScape default)
if [ -f "${APP_DIR}/configs/tracker-config.json" ]; then
    TRACKER_CONFIG="${APP_DIR}/configs/tracker-config.json"
else
    TRACKER_CONFIG="${SCENESCAPE_DIR}/controller/tracker-config.json"
fi
if [ -f "${APP_DIR}/configs/reid-config.json" ]; then
    REID_CONFIG="${APP_DIR}/configs/reid-config.json"
else
    REID_CONFIG="${SCENESCAPE_DIR}/controller/reid-config.json"
fi

DLSTREAMER_OUTPUT_DIR="${SCENESCAPE_DIR}/dlstreamer-pipeline-server"

# Camera 1 — always generated
PIPELINE_CONFIG_1="${DLSTREAMER_OUTPUT_DIR}/${APP_NAME}-${CAMERA_NAME}-pipeline-config.json"
sed "s/{{CAMERA_NAME}}/${CAMERA_NAME}/g" "${DLSTREAMER_TEMPLATE}" > "${PIPELINE_CONFIG_1}"
echo "  Camera 1: ${PIPELINE_CONFIG_1}"
echo "    Pipeline: reid_${CAMERA_NAME}  cameraid: ${CAMERA_NAME}"

# Camera 2 — generated only if defined in zone_config.json
PIPELINE_CONFIG_2=""
if [ -n "${CAMERA_NAME_2}" ]; then
    PIPELINE_CONFIG_2="${DLSTREAMER_OUTPUT_DIR}/${APP_NAME}-${CAMERA_NAME_2}-pipeline-config.json"
    sed "s/{{CAMERA_NAME}}/${CAMERA_NAME_2}/g" "${DLSTREAMER_TEMPLATE}" > "${PIPELINE_CONFIG_2}"
    echo "  Camera 2: ${PIPELINE_CONFIG_2}"
    echo "    Pipeline: reid_${CAMERA_NAME_2}  cameraid: ${CAMERA_NAME_2}"
fi

# ---- Step 4: Generate .env file ----
echo -e "${YELLOW}[4/4] Generating docker/.env...${NC}"

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
CAMERA_NAME_2=${CAMERA_NAME_2}
SCENE_ZIP=${SCENE_ZIP}
VIDEO_FILE=${VIDEO_FILE}
VIDEO_FILE_2=${VIDEO_FILE_2}

# DLStreamer pipeline config (app-specific, generated by init.sh per camera)
PIPELINE_CONFIG=${PIPELINE_CONFIG_1}
PIPELINE_CONFIG_2=${PIPELINE_CONFIG_2}

# Controller configs (app-specific — tracker + reid)
TRACKER_CONFIG=${TRACKER_CONFIG}
REID_CONFIG=${REID_CONFIG}

# OpenVINO Models (comma-separated, from zone_config.json or default)
MODELS=${MODELS:-person-detection-retail-0013,person-reidentification-retail-0277}
MODEL_PRECISION=FP32

# SceneScape image versions
SCENESCAPE_REGISTRY=
SCENESCAPE_VERSION=latest
SCENESCAPE_CONTROLLER_IMAGE=scenescape-controller
SCENESCAPE_MANAGER_IMAGE=scenescape-manager
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
EOF

echo ""
echo -e "${GREEN}=== Init complete ===${NC}"
echo ""
echo "Generated files:"
echo "  Secrets:              ${SECRETS_DIR}/"
echo "  Pipeline config (1):  ${PIPELINE_CONFIG_1}"
if [ -n "${PIPELINE_CONFIG_2}" ]; then
echo "  Pipeline config (2):  ${PIPELINE_CONFIG_2}"
fi
echo "  Tracker config:       ${APP_DIR}/configs/tracker-config.json"
echo "  Reid config:          ${APP_DIR}/configs/reid-config.json"
echo "  Env:                  ${ENV_FILE}"
echo ""
echo "Scene: ${SCENE_NAME}"
echo "  Camera: ${CAMERA_NAME}  Video: ${VIDEO_FILE}  Zip: ${SCENE_ZIP}"
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
