#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Initialize secrets, read zone_config.json, generate DLStreamer config,
# and generate .env for the full-stack deployment.
#
# Usage: ./init.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECRETS_DIR="${SCRIPT_DIR}/scenescape/secrets"
ENV_FILE="${SCRIPT_DIR}/docker/.env"
SAMPLE_DATA_DIR="${SCRIPT_DIR}/scenescape/sample_data"
ZONE_CONFIG="${SCRIPT_DIR}/configs/zone_config.json"
DLSTREAMER_CONFIG="${SCRIPT_DIR}/scenescape/dlstreamer-pipeline-server/config.json"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo -e "${GREEN}=== Storewide Loss Prevention - Full Stack Init ===${NC}"
echo ""

# ---- Step 1: Generate SceneScape secrets ----
echo -e "${YELLOW}[1/4] Generating SceneScape secrets...${NC}"
chmod +x "${SECRETS_DIR}/generate_secrets.sh"
bash "${SECRETS_DIR}/generate_secrets.sh"

# ---- Step 2: Read zone_config.json ----
echo -e "${YELLOW}[2/4] Reading zone_config.json...${NC}"
if [ ! -f "${ZONE_CONFIG}" ]; then
    echo -e "${RED}ERROR: zone_config.json not found at ${ZONE_CONFIG}${NC}"
    exit 1
fi

SCENE_NAME=$(python3 -c "import json; print(json.load(open('${ZONE_CONFIG}')).get('scene_name',''))" 2>/dev/null)
SCENE_ZIP=$(python3 -c "import json; print(json.load(open('${ZONE_CONFIG}')).get('scene_zip',''))" 2>/dev/null)
CAMERA_NAME=$(python3 -c "import json; print(json.load(open('${ZONE_CONFIG}')).get('camera_name',''))" 2>/dev/null)
VIDEO_FILE=$(python3 -c "import json; print(json.load(open('${ZONE_CONFIG}')).get('video_file',''))" 2>/dev/null)

echo "  Scene name:  ${SCENE_NAME}"
echo "  Scene zip:   ${SCENE_ZIP}"
echo "  Camera name: ${CAMERA_NAME}"
echo "  Video file:  ${VIDEO_FILE}"

if [ -z "${SCENE_NAME}" ] || [ -z "${CAMERA_NAME}" ] || [ -z "${VIDEO_FILE}" ]; then
    echo -e "${RED}ERROR: zone_config.json must have scene_name, camera_name, and video_file${NC}"
    exit 1
fi

# Validate scene zip exists
SCENE_ZIP_PATH="${SCRIPT_DIR}/scenescape/webserver/${SCENE_ZIP}"
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

# ---- Step 3: Generate DLStreamer config.json ----
echo -e "${YELLOW}[3/4] Generating DLStreamer pipeline config...${NC}"

cat > "${DLSTREAMER_CONFIG}" <<DLEOF
{
  "config": {
    "logging": {
      "C_LOG_LEVEL": "INFO",
      "PY_LOG_LEVEL": "INFO"
    },
    "pipelines": [
      {
        "name": "reid_${CAMERA_NAME}",
        "source": "gstreamer",
        "pipeline": "rtspsrc location=rtsp://mediaserver:8554/retail-cam1 latency=200 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! gvapython class=PostDecodeTimestampCapture function=processFrame module=/home/pipeline-server/user_scripts/gvapython/sscape/sscape_adapter.py name=timesync ! gvadetect model=/home/pipeline-server/models/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml model-proc=/home/pipeline-server/models/object_detection/person/person-detection-retail-0013.json name=detection ! gvainference model=/home/pipeline-server/models/intel/person-reidentification-retail-0277/FP32/person-reidentification-retail-0277.xml inference-region=roi-list ! gvametaconvert add-tensor-data=true name=metaconvert ! gvapython class=PostInferenceDataPublish function=processFrame module=/home/pipeline-server/user_scripts/gvapython/sscape/sscape_adapter.py name=datapublisher ! gvametapublish name=destination ! appsink sync=true",
        "auto_start": true,
        "parameters": {
          "type": "object",
          "properties": {
            "ntp_config": {
              "element": { "name": "timesync", "property": "kwarg", "format": "json" },
              "type": "object",
              "properties": { "ntpServer": { "type": "string" } }
            },
            "camera_config": {
              "element": { "name": "datapublisher", "property": "kwarg", "format": "json" },
              "type": "object",
              "properties": {
                "cameraid": { "type": "string" },
                "metadatagenpolicy": { "type": "string" },
                "publish_frame": { "type": "boolean" },
                "detection_labels": { "type": "array", "items": { "type": "string" } }
              }
            }
          }
        },
        "payload": {
          "parameters": {
            "ntp_config": { "ntpServer": "ntpserv" },
            "camera_config": {
              "cameraid": "${CAMERA_NAME}",
              "metadatagenpolicy": "reidPolicy",
              "detection_labels": ["person"]
            }
          }
        }
      }
    ]
  }
}
DLEOF

echo "  Generated ${DLSTREAMER_CONFIG}"
echo "  Pipeline: reid_${CAMERA_NAME}  cameraid: ${CAMERA_NAME}"

# ---- Step 4: Generate .env file ----
echo -e "${YELLOW}[4/4] Generating docker/.env...${NC}"

# Read generated secrets — honor SUPASS from environment if set
SUPASS="${SUPASS:-$(cat "${SECRETS_DIR}/supass" 2>/dev/null || echo "")}"
DBPASS=$(sed -nr "/DATABASE_PASSWORD=/s/.*'([^']+)'/\1/p" "${SECRETS_DIR}/django/secrets.py" 2>/dev/null || echo "")
CONTROLLER_AUTH=$(cat "${SECRETS_DIR}/controller.auth" 2>/dev/null || echo "")

USER_UID=$(id -u)
USER_GID=$(id -g)

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

# Scene (from zone_config.json)
SCENE_NAME=${SCENE_NAME}
SCENE_ZIP=${SCENE_ZIP}
CAMERA_NAME=${CAMERA_NAME}
VIDEO_FILE=${VIDEO_FILE}

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
echo -e "  SUPASS: ${YELLOW}${SUPASS}${NC}"
echo ""
echo "To change scene/video/camera: edit configs/zone_config.json, then re-run ./init.sh"
echo ""
echo "Next steps:"
echo "  1. Place your video in scenescape/sample_data/${VIDEO_FILE}"
echo "  2. Place your scene zip in scenescape/webserver/${SCENE_ZIP}"
echo "  3. Start the full stack:"
echo "       make demo   (or: docker compose -f docker/docker-compose.full.yaml up -d)"
echo ""
echo "  4. Open SceneScape UI:  https://localhost"
echo "     Login: admin / ${SUPASS}"
echo "  5. Open Gradio UI:      http://localhost:7860"
