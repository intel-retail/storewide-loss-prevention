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
# Accept resource config as $2 (positional arg) or RESOURCE_CONFIG env var.
# Positional arg takes precedence, but only if non-empty.
if [ -n "${2:-}" ]; then
    RESOURCE_CONFIG="$2"
fi

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

# Ensure key files are readable inside Docker containers (fixes Permission denied
# on /run/secrets/certs/scenescape-web.key when the container user differs from
# the host user who generated the keys).
chmod 644 "${SECRETS_DIR}"/certs/*.key "${SECRETS_DIR}"/ca/*.key 2>/dev/null || true

# ---- Step 2: Read zone_config.json (single source of truth) ----
echo -e "${YELLOW}[2/4] Reading zone_config.json...${NC}"
if [ ! -f "${ZONE_CONFIG}" ]; then
    echo -e "${RED}ERROR: zone_config.json not found at ${ZONE_CONFIG}${NC}"
    exit 1
fi

# Extract all configuration from zone_config.json in one pass
eval "$(python3 -c "
import json, sys

cfg = json.load(open('${ZONE_CONFIG}'))

# Scene
print(f'SCENE_NAME=\"{cfg.get(\"scene_name\", \"\")}\"')
print(f'SCENE_ZIP=\"{cfg.get(\"scene_zip\", \"\")}\"')

# Cameras (new array format or legacy flat fields)
cameras = cfg.get('cameras', [])
if cameras:
    print(f'CAMERA_NAME=\"{cameras[0].get(\"name\", \"\")}\"')
    print(f'VIDEO_FILE=\"{cameras[0].get(\"video\", \"\")}\"')
    if len(cameras) > 1:
        print(f'CAMERA_NAME_2=\"{cameras[1].get(\"name\", \"\")}\"')
        print(f'VIDEO_FILE_2=\"{cameras[1].get(\"video\", \"\")}\"')
    else:
        print('CAMERA_NAME_2=\"\"')
        print('VIDEO_FILE_2=\"\"')
else:
    # Legacy flat fields
    print(f'CAMERA_NAME=\"{cfg.get(\"camera_name\", \"\")}\"')
    print(f'VIDEO_FILE=\"{cfg.get(\"video_file\", \"\")}\"')
    print(f'CAMERA_NAME_2=\"{cfg.get(\"camera_name_2\", \"\")}\"')
    print(f'VIDEO_FILE_2=\"{cfg.get(\"video_file_2\", \"\")}\"')

# Models
print(f'MODELS=\"{cfg.get(\"models\", \"\")}\"')
print(f'MODEL_PRECISION=\"{cfg.get(\"model_precision\", \"FP32\")}\"')

# SceneScape images
ss = cfg.get('scenescape', {})
print(f'SCENESCAPE_REGISTRY=\"{ss.get(\"registry\", \"\")}\"')
print(f'SCENESCAPE_VERSION=\"{ss.get(\"version\", \"latest\")}\"')
print(f'SCENESCAPE_CONTROLLER_IMAGE=\"{ss.get(\"controller_image\", \"intel/scenescape-controller\")}\"')
print(f'SCENESCAPE_MANAGER_IMAGE=\"{ss.get(\"manager_image\", \"intel/scenescape-manager\")}\"')
print(f'DLSTREAMER_VERSION=\"{ss.get(\"dlstreamer_version\", \"2026.1.0-20260331-weekly-ubuntu24\")}\"')
print(f'SCENESCAPE_API_USER=\"{ss.get(\"api_user\", \"admin\")}\"')

# SceneScape API
api = cfg.get('scenescape_api', {})
print(f'SCENESCAPE_API_URL=\"{api.get(\"base_url\", \"https://localhost\")}\"')

# Store
store = cfg.get('store', {})
print(f'STORE_NAME=\"{store.get(\"name\", \"Retail\")}\"')
print(f'STORE_ID=\"{store.get(\"id\", \"store_001\")}\"')

# Services
svc = cfg.get('services', {})
print(f'LP_SERVICE_PORT=\"{svc.get(\"lp_service_port\", 8082)}\"')
print(f'LOG_LEVEL=\"{svc.get(\"log_level\", \"INFO\")}\"')
print(f'SEAWEEDFS_S3_PORT=\"{svc.get(\"seaweedfs_s3_port\", 8333)}\"')
print(f'SEAWEEDFS_MASTER_PORT=\"{svc.get(\"seaweedfs_master_port\", 9333)}\"')
print(f'SEAWEEDFS_VOLUME_PORT=\"{svc.get(\"seaweedfs_volume_port\", 8080)}\"')

# Benchmark
bm = cfg.get('benchmark', {})
print(f'BENCHMARK_TARGET_LATENCY_MS=\"{bm.get(\"target_latency_ms\", 2000)}\"')
print(f'BENCHMARK_LATENCY_METRIC=\"{bm.get(\"latency_metric\", \"avg\")}\"')
print(f'BENCHMARK_SCENE_INCREMENT=\"{bm.get(\"scene_increment\", 1)}\"')
print(f'BENCHMARK_INIT_DURATION=\"{bm.get(\"init_duration\", 90)}\"')
print(f'BENCHMARK_STABILISE_DURATION=\"{bm.get(\"stabilise_duration\", 30)}\"')
print(f'BENCHMARK_MAX_ITERATIONS=\"{bm.get(\"max_iterations\", 50)}\"')
print(f'BENCHMARK_MIN_THROUGHPUT_RATIO=\"{bm.get(\"min_throughput_ratio\", 0.5)}\"')
print(f'RESULTS_PATH=\"{bm.get(\"results_path\", \"./results\")}\"')
" 2>/dev/null)"

# Apply defaults for required fields
SCENE_ZIP="${SCENE_ZIP:-storewide-loss-prevention.zip}"
VIDEO_FILE="${VIDEO_FILE:-lp-camera1.mp4}"
MODELS="${MODELS:-person-detection-retail-0013,person-reidentification-retail-0277}"

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
echo "  Models:      ${MODELS}"

if [ -z "${SCENE_NAME}" ] || [ -z "${CAMERA_NAME}" ]; then
    echo -e "${RED}ERROR: zone_config.json must have scene_name and cameras[0].name${NC}"
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

# ---- Source AI-model settings from configs/.env.example (single source of truth) ----
ENV_EXAMPLE="${APP_DIR}/configs/.env.example"
AI_KEYS_REGEX='^(VLM_ENABLED|VLM_MODEL_NAME|VLM_PRECISION|TARGET_DEVICE|YOLO_MODEL_NAME|DETECT_MODEL|DETECT_MODEL_PRECISION|REID_MODEL|REID_MODEL_PRECISION|OVMS_IMAGE_TAG|MODEL_PRECISION|SCENESCAPE_REGISTRY|SCENESCAPE_VERSION|DLSTREAMER_VERSION|TZ)='
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

# Source device resource config (all-gpu-cpu.env, all-gpu.env, or all-cpu.env)
RESOURCE_CONFIG="${RESOURCE_CONFIG:-configs/res/all-gpu-cpu.env}"
RESOURCE_CONFIG_PATH="${APP_DIR}/${RESOURCE_CONFIG}"
if [ -f "${RESOURCE_CONFIG_PATH}" ]; then
    echo "  Loading resource config: ${RESOURCE_CONFIG}"
    set -a
    # shellcheck disable=SC1090
    . "${RESOURCE_CONFIG_PATH}"
    set +a
else
    echo -e "${RED}ERROR: Resource config not found at ${RESOURCE_CONFIG_PATH}${NC}"
    echo "Valid options: $(ls -1 "${APP_DIR}/configs/res/"*.env 2>/dev/null | xargs -I{} basename {} | tr '\n' ' ')"
    exit 1
fi

DETECT_DEVICE="${DETECT_DEVICE:-GPU}"
REID_DEVICE="${REID_DEVICE:-CPU}"
DETECT_MODEL="${DETECT_MODEL:-yolo11s}"
DETECT_MODEL_PRECISION="${DETECT_MODEL_PRECISION:-FP16}"
REID_MODEL="${REID_MODEL:-person-reidentification-retail-0277}"
REID_MODEL_PRECISION="${REID_MODEL_PRECISION:-FP16}"

# Auto-derive model-proc and labels: YOLO models need both, OpenVINO models skip labels
if [[ "${DETECT_MODEL}" == yolo* ]]; then
    DETECT_MODEL_PROC="${DETECT_MODEL_PROC:-yolo-v8.json}"
    DETECT_LABELS="labels-file=/home/pipeline-server/models/detect/${DETECT_MODEL}/labels.txt"
else
    DETECT_MODEL_PROC="${DETECT_MODEL_PROC:-${DETECT_MODEL}.json}"
    DETECT_LABELS=""
fi
MODEL_PRECISION="${MODEL_PRECISION:-FP32}"

echo "  Detect: ${DETECT_MODEL} (${DETECT_MODEL_PRECISION}) on ${DETECT_DEVICE}  ReID: ${REID_MODEL} (${REID_MODEL_PRECISION}) on ${REID_DEVICE}"

# Defaults for pipeline element variables (if not set by resource config)
DECODE="${DECODE:-rtph264depay ! h264parse ! vah264dec ! vapostproc ! video/x-raw(memory:VAMemory)}"
PRE_PROCESS="${PRE_PROCESS:-pre-process-backend=va-surface-sharing}"
PRE_PROCESS_CONFIG="${PRE_PROCESS_CONFIG:-}"
# FACE_PRE_PROCESS: pre-process backend for face-detection gvadetect.
# Face-detection runs after personreid which outputs system memory, so it must use
# opencv (not va-surface-sharing) even in GPU mode.
FACE_PRE_PROCESS="${FACE_PRE_PROCESS:-pre-process-backend=opencv}"
FACE_PRE_PROCESS_CONFIG="${FACE_PRE_PROCESS_CONFIG:-}"
DETECTION_OPTIONS="${DETECTION_OPTIONS:-ie-config=GPU_THROUGHPUT_STREAMS=2 nireq=2}"
REID_PRE_PROCESS="${REID_PRE_PROCESS:-pre-process-backend=opencv}"
REID_OPTIONS="${REID_OPTIONS:-nireq=2}"
POST_DETECT="${POST_DETECT:-}"
POST_INFERENCE="${POST_INFERENCE:-}"
QUEUE_OPTIONS="${QUEUE_OPTIONS:-max-size-buffers=1 leaky=downstream}"
DETECT_THRESHOLD="${DETECT_THRESHOLD:-0.5}"
INFERENCE_INTERVAL="${INFERENCE_INTERVAL:-3}"

echo "  Resource config: ${RESOURCE_CONFIG}"

# Build !-delimited element chains; insert leading " ! " only when non-empty
POST_DETECT_CHAIN=""
if [ -n "${POST_DETECT}" ]; then
    POST_DETECT_CHAIN="! ${POST_DETECT}"
fi
POST_INFERENCE_CHAIN=""
if [ -n "${POST_INFERENCE}" ]; then
    POST_INFERENCE_CHAIN="! ${POST_INFERENCE}"
fi

# Camera 1 — always generated
PIPELINE_CONFIG_1="${DLSTREAMER_OUTPUT_DIR}/${APP_NAME}-${CAMERA_NAME}-pipeline-config.json"
sed -e "s|{{CAMERA_NAME}}|${CAMERA_NAME}|g" \
    -e "s|{{DETECT_MODEL_PROC}}|${DETECT_MODEL_PROC}|g" \
    -e "s|{{DETECT_LABELS}}|${DETECT_LABELS}|g" \
    -e "s|{{DETECT_MODEL}}|${DETECT_MODEL}|g" \
    -e "s|{{DETECT_MODEL_PRECISION}}|${DETECT_MODEL_PRECISION}|g" \
    -e "s|{{DETECT_DEVICE}}|${DETECT_DEVICE}|g" \
    -e "s|{{REID_MODEL}}|${REID_MODEL}|g" \
    -e "s|{{REID_MODEL_PRECISION}}|${REID_MODEL_PRECISION}|g" \
    -e "s|{{REID_DEVICE}}|${REID_DEVICE}|g" \
    -e "s|{{MODEL_PRECISION}}|${MODEL_PRECISION}|g" \
    -e "s|{{DECODE}}|${DECODE}|g" \
    -e "s|{{PRE_PROCESS}}|${PRE_PROCESS}|g" \
    -e "s|{{PRE_PROCESS_CONFIG}}|${PRE_PROCESS_CONFIG}|g" \
    -e "s|{{FACE_PRE_PROCESS}}|${FACE_PRE_PROCESS}|g" \
    -e "s|{{FACE_PRE_PROCESS_CONFIG}}|${FACE_PRE_PROCESS_CONFIG}|g" \
    -e "s|{{DETECTION_OPTIONS}}|${DETECTION_OPTIONS}|g" \
    -e "s|{{REID_PRE_PROCESS}}|${REID_PRE_PROCESS}|g" \
    -e "s|{{REID_OPTIONS}}|${REID_OPTIONS}|g" \
    -e "s|{{POST_DETECT}}|${POST_DETECT_CHAIN}|g" \
    -e "s|{{POST_INFERENCE}}|${POST_INFERENCE_CHAIN}|g" \
    -e "s|{{QUEUE_OPTIONS}}|${QUEUE_OPTIONS}|g" \
    -e "s|{{DETECT_THRESHOLD}}|${DETECT_THRESHOLD}|g" \
    -e "s|{{INFERENCE_INTERVAL}}|${INFERENCE_INTERVAL}|g" \
    "${DLSTREAMER_TEMPLATE}" > "${PIPELINE_CONFIG_1}"
echo "  Camera 1: ${PIPELINE_CONFIG_1}"
echo "    Pipeline: reid_${CAMERA_NAME}  cameraid: ${CAMERA_NAME}"

# Camera 2 — generated only if defined in zone_config.json
PIPELINE_CONFIG_2=""
if [ -n "${CAMERA_NAME_2}" ]; then
    PIPELINE_CONFIG_2="${DLSTREAMER_OUTPUT_DIR}/${APP_NAME}-${CAMERA_NAME_2}-pipeline-config.json"
    sed -e "s|{{CAMERA_NAME}}|${CAMERA_NAME_2}|g" \
        -e "s|{{DETECT_MODEL_PROC}}|${DETECT_MODEL_PROC}|g" \
        -e "s|{{DETECT_LABELS}}|${DETECT_LABELS}|g" \
        -e "s|{{DETECT_MODEL}}|${DETECT_MODEL}|g" \
        -e "s|{{DETECT_MODEL_PRECISION}}|${DETECT_MODEL_PRECISION}|g" \
        -e "s|{{DETECT_DEVICE}}|${DETECT_DEVICE}|g" \
        -e "s|{{REID_MODEL}}|${REID_MODEL}|g" \
        -e "s|{{REID_MODEL_PRECISION}}|${REID_MODEL_PRECISION}|g" \
        -e "s|{{REID_DEVICE}}|${REID_DEVICE}|g" \
        -e "s|{{MODEL_PRECISION}}|${MODEL_PRECISION}|g" \
        -e "s|{{DECODE}}|${DECODE}|g" \
        -e "s|{{PRE_PROCESS}}|${PRE_PROCESS}|g" \
        -e "s|{{PRE_PROCESS_CONFIG}}|${PRE_PROCESS_CONFIG}|g" \
        -e "s|{{FACE_PRE_PROCESS}}|${FACE_PRE_PROCESS}|g" \
        -e "s|{{FACE_PRE_PROCESS_CONFIG}}|${FACE_PRE_PROCESS_CONFIG}|g" \
        -e "s|{{DETECTION_OPTIONS}}|${DETECTION_OPTIONS}|g" \
        -e "s|{{REID_PRE_PROCESS}}|${REID_PRE_PROCESS}|g" \
        -e "s|{{REID_OPTIONS}}|${REID_OPTIONS}|g" \
        -e "s|{{POST_DETECT}}|${POST_DETECT_CHAIN}|g" \
        -e "s|{{POST_INFERENCE}}|${POST_INFERENCE_CHAIN}|g" \
        -e "s|{{QUEUE_OPTIONS}}|${QUEUE_OPTIONS}|g" \
        -e "s|{{DETECT_THRESHOLD}}|${DETECT_THRESHOLD}|g" \
        -e "s|{{INFERENCE_INTERVAL}}|${INFERENCE_INTERVAL}|g" \
        "${DLSTREAMER_TEMPLATE}" > "${PIPELINE_CONFIG_2}"
    echo "  Camera 2: ${PIPELINE_CONFIG_2}"
    echo "    Pipeline: reid_${CAMERA_NAME_2}  cameraid: ${CAMERA_NAME_2}"
else
    # Fallback to camera 1 config so Docker Compose config resolution doesn't fail
    PIPELINE_CONFIG_2="${PIPELINE_CONFIG_1}"
    echo "  Camera 2: not configured (using camera 1 config as placeholder)"
fi

# ---- Step 4: Generate .env file ----
echo -e "${YELLOW}[4/4] Generating docker/.env...${NC}"

# Read generated secrets — honor SUPASS from environment if set
SUPASS="${SUPASS:-$(cat "${SECRETS_DIR}/supass" 2>/dev/null || echo "")}"
DBPASS=$(sed -nr "/DATABASE_PASSWORD=/s/.*'([^']+)'/\1/p" "${SECRETS_DIR}/django/secrets.py" 2>/dev/null || echo "")
CONTROLLER_AUTH=$(cat "${SECRETS_DIR}/controller.auth" 2>/dev/null || echo "")

USER_UID=$(id -u)
USER_GID=$(id -g)

# If secrets were freshly generated, remove stale DB volumes so PostgreSQL
# reinitializes with the new password.  Only remove volumes belonging to the
# storewide-lp compose project (set in scenescape/docker-compose.yaml).
if [ "${SECRETS_GENERATED}" = "1" ]; then
    echo "  New secrets generated — removing stale DB volumes..."
    for vol in storewide-lp_vol-db storewide-lp_vol-migrations; do
        docker volume rm "$vol" 2>/dev/null && echo "    Removed $vol" || true
    done
fi

mkdir -p "$(dirname "${ENV_FILE}")"

if [ -f "${ENV_FILE}" ]; then
    echo "  ${ENV_FILE} already exists. Backing up to ${ENV_FILE}.bak"
    cp "${ENV_FILE}" "${ENV_FILE}.bak"
fi

cat > "${ENV_FILE}" <<EOF
# Auto-generated by init.sh from ${APP_DIR}/configs/zone_config.json
# Regenerate: make init  (or ../scenescape/scripts/init.sh ${APP_DIR})
# Generated: $(date -Iseconds)

# ---- Secrets (auto-generated) ----
SECRETSDIR=${SECRETS_DIR}
SUPASS=${SUPASS}
DATABASE_PASSWORD=${DBPASS}
CONTROLLER_AUTH=${CONTROLLER_AUTH}
UID=${USER_UID}
GID=${USER_GID}

# ---- Scene (from zone_config.json) ----
SCENE_NAME=${SCENE_NAME}
CAMERA_NAME=${CAMERA_NAME}
CAMERA_NAME_2=${CAMERA_NAME_2}
SCENE_ZIP=${SCENE_ZIP}
VIDEO_FILE=${VIDEO_FILE}
VIDEO_FILE_2=${VIDEO_FILE_2}

# ---- DLStreamer pipeline config (generated per camera) ----
PIPELINE_CONFIG=${PIPELINE_CONFIG_1}
PIPELINE_CONFIG_2=${PIPELINE_CONFIG_2}

# ---- Controller configs (from app configs/) ----
TRACKER_CONFIG=${TRACKER_CONFIG}
REID_CONFIG=${REID_CONFIG}

# ---- OpenVINO Models (from zone_config.json) ----
MODELS=${MODELS}
MODEL_PRECISION=${MODEL_PRECISION}

# ---- SceneScape images (from zone_config.json) ----
SCENESCAPE_REGISTRY=${SCENESCAPE_REGISTRY}
SCENESCAPE_VERSION=${SCENESCAPE_VERSION}
SCENESCAPE_CONTROLLER_IMAGE=${SCENESCAPE_CONTROLLER_IMAGE}
SCENESCAPE_MANAGER_IMAGE=${SCENESCAPE_MANAGER_IMAGE}
DLSTREAMER_VERSION=${DLSTREAMER_VERSION}

# ---- Store (from zone_config.json) ----
STORE_NAME=${STORE_NAME}
STORE_ID=${STORE_ID}

# ---- Services (from zone_config.json) ----
LP_SERVICE_PORT=${LP_SERVICE_PORT}
LOG_LEVEL=${LOG_LEVEL}
TZ=${TZ:-UTC}

# ---- SeaweedFS (from zone_config.json) ----
SEAWEEDFS_S3_PORT=${SEAWEEDFS_S3_PORT}
SEAWEEDFS_MASTER_PORT=${SEAWEEDFS_MASTER_PORT}
SEAWEEDFS_VOLUME_PORT=${SEAWEEDFS_VOLUME_PORT}

# ---- SceneScape API (from zone_config.json scenescape_api.base_url) ----
SCENESCAPE_API_URL=${SCENESCAPE_API_URL}
SCENESCAPE_API_USER=${SCENESCAPE_API_USER}
SCENESCAPE_API_PASSWORD=${SUPASS}

# ---- Benchmark (from zone_config.json) ----
BENCHMARK_TARGET_LATENCY_MS=${BENCHMARK_TARGET_LATENCY_MS}
BENCHMARK_LATENCY_METRIC=${BENCHMARK_LATENCY_METRIC}
BENCHMARK_SCENE_INCREMENT=${BENCHMARK_SCENE_INCREMENT}
BENCHMARK_INIT_DURATION=${BENCHMARK_INIT_DURATION}
BENCHMARK_STABILISE_DURATION=${BENCHMARK_STABILISE_DURATION}
BENCHMARK_MAX_ITERATIONS=${BENCHMARK_MAX_ITERATIONS}
BENCHMARK_MIN_THROUGHPUT_RATIO=${BENCHMARK_MIN_THROUGHPUT_RATIO}
RESULTS_PATH=${RESULTS_PATH}

# ---- AI Models (sourced from configs/.env.example) ----
VLM_ENABLED=${VLM_ENABLED:-true}
VLM_MODEL_NAME=${VLM_MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}
VLM_PRECISION=${VLM_PRECISION:-int8}
TARGET_DEVICE=${TARGET_DEVICE:-GPU}
YOLO_MODEL_NAME=${YOLO_MODEL_NAME:-yolo26n-pose}
DETECT_MODEL=${DETECT_MODEL}
DETECT_MODEL_PRECISION=${DETECT_MODEL_PRECISION}
REID_MODEL=${REID_MODEL}
REID_MODEL_PRECISION=${REID_MODEL_PRECISION}
OVMS_IMAGE_TAG=${OVMS_IMAGE_TAG:-2026.1-gpu}

# ---- Device Resource Config ----
DECODE=${DECODE}
DETECT_DEVICE=${DETECT_DEVICE}
REID_DEVICE=${REID_DEVICE}
PRE_PROCESS=${PRE_PROCESS}
DETECTION_OPTIONS=${DETECTION_OPTIONS}
REID_PRE_PROCESS=${REID_PRE_PROCESS}
REID_OPTIONS=${REID_OPTIONS}
POST_DETECT=${POST_DETECT}
POST_INFERENCE=${POST_INFERENCE}
QUEUE_OPTIONS=${QUEUE_OPTIONS}
DETECT_THRESHOLD=${DETECT_THRESHOLD}
INFERENCE_INTERVAL=${INFERENCE_INTERVAL}

# ---- WebRTC / MediaMTX ----
HOST_IP=${HOST_IP}

# ---- Host user identity (for Docker bind-mount file ownership) ----
HOST_UID=$(id -u)
HOST_GID=$(id -g)
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
echo "  Tracker config:       ${TRACKER_CONFIG}"
echo "  Reid config:          ${REID_CONFIG}"
echo "  Env:                  ${ENV_FILE}"
echo ""
echo "All values sourced from: ${ZONE_CONFIG}"
echo ""
echo "Scene: ${SCENE_NAME}"
echo "  Camera: ${CAMERA_NAME}  Video: ${VIDEO_FILE}  Zip: ${SCENE_ZIP}"
echo -e "  SUPASS: ${YELLOW}${SUPASS}${NC}"
echo ""
echo "To change any setting: edit configs/zone_config.json, then re-run init.sh"
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
