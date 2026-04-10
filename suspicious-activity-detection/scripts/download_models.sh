#!/bin/bash -e
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Download OpenVINO models required by the DLStreamer inference pipeline.
# Models are downloaded via wget from the OpenVINO Model Zoo storage,
# following the same approach as SceneScape's model_installer.
#
# Usage:
#   ./scripts/download_models.sh [--precisions FP32,FP16]
#   make download-models

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_PROC_DIR="${PROJECT_DIR}/scenescape/dlstreamer-pipeline-server/model-proc-files"
VOLUME_NAME="storewide-lp_vol-models"

# Source MODELS and MODEL_PRECISION from .env if available
ENV_FILE="${PROJECT_DIR}/docker/.env"
if [ -f "${ENV_FILE}" ]; then
    MODELS="${MODELS:-$(grep ^MODELS= "${ENV_FILE}" 2>/dev/null | cut -d= -f2-)}"
    MODEL_PRECISION="${MODEL_PRECISION:-$(grep ^MODEL_PRECISION= "${ENV_FILE}" 2>/dev/null | cut -d= -f2-)}"
fi

# OpenVINO Model Zoo download URL (same as SceneScape model_installer)
OMZ_BASE_URL="https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1"

# Models required by the DLStreamer pipeline (from .env or defaults)
DEFAULT_MODELS="person-detection-retail-0013,person-reidentification-retail-0277"
IFS=',' read -ra MODELS <<< "${MODELS:-${DEFAULT_MODELS}}"

# Default precision; override with MODEL_PRECISION env or --precisions flag
PRECISIONS="${MODEL_PRECISION:-FP32}"
if [ "$1" = "--precisions" ] && [ -n "$2" ]; then
    PRECISIONS="$2"
fi

echo "=== Downloading OpenVINO Models ==="
echo "  Precisions: ${PRECISIONS}"

# Create the volume if it doesn't exist
docker volume create "${VOLUME_NAME}" 2>/dev/null || true

# Build the download commands
DOWNLOAD_CMDS=""
for model in "${MODELS[@]}"; do
    IFS=',' read -ra PREC_LIST <<< "${PRECISIONS}"
    for prec in "${PREC_LIST[@]}"; do
        prec=$(echo "${prec}" | xargs)  # trim whitespace
        DOWNLOAD_CMDS+="echo \"  Downloading ${model} (${prec})...\" && "
        DOWNLOAD_CMDS+="mkdir -p /models/intel/${model}/${prec} && "
        DOWNLOAD_CMDS+="wget -nv -O /models/intel/${model}/${prec}/${model}.xml ${OMZ_BASE_URL}/${model}/${prec}/${model}.xml && "
        DOWNLOAD_CMDS+="wget -nv -O /models/intel/${model}/${prec}/${model}.bin ${OMZ_BASE_URL}/${model}/${prec}/${model}.bin && "
    done
done
DOWNLOAD_CMDS="${DOWNLOAD_CMDS% && }"

echo "[1/2] Downloading models into Docker volume '${VOLUME_NAME}'..."
docker run --rm \
    -v "${VOLUME_NAME}":/models \
    alpine:3.23 \
    sh -c "apk add --no-cache wget >/dev/null 2>&1 && ${DOWNLOAD_CMDS}"

# Copy model-proc files (same as SceneScape's copy-config-files)
echo "[2/2] Copying model-proc files..."
if [ -d "${MODEL_PROC_DIR}" ] && [ -n "$(ls -A "${MODEL_PROC_DIR}"/*.json 2>/dev/null)" ]; then
    docker run --rm \
        -v "${MODEL_PROC_DIR}":/src:ro \
        -v "${VOLUME_NAME}":/models \
        alpine:3.23 \
        sh -c "mkdir -p /models/object_detection/person && cp -v /src/*.json /models/object_detection/person/"
else
    echo "WARNING: No model-proc JSON files found in ${MODEL_PROC_DIR}"
fi

echo ""
echo "=== Model download complete ==="
echo "Volume: ${VOLUME_NAME}"
echo "Models:"
for model in "${MODELS[@]}"; do
    echo "  - intel/${model} (${PRECISIONS})"
done
