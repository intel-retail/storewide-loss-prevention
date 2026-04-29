#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Model Download Script for Suspicious Activity Detection
# Downloads and exports:
#   1. Qwen/Qwen2.5-VL-7B-Instruct VLM (for OVMS)
#   2. yolo26n-pose (for behavioral analysis)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
MODELS_DIR="${PROJECT_ROOT}/models"

###############################################
# CONFIGURATION — load AI-model settings from configs/.env.example
# (single source of truth for VLM_*/TARGET_DEVICE/YOLO_*).
# Falls back to docker/.env if .env.example is missing.
# Only AI-model keys are sourced — placeholder keys like SUPASS,
# CAMERA_NAME, etc. in .env.example are intentionally ignored.
###############################################
ENV_EXAMPLE="${PROJECT_ROOT}/configs/.env.example"
ENV_FILE="${PROJECT_ROOT}/docker/.env"
AI_KEYS_REGEX='^(VLM_ENABLED|VLM_MODEL_NAME|VLM_PRECISION|TARGET_DEVICE|YOLO_MODEL_NAME)='

SOURCE_FILE=""
if [ -f "${ENV_EXAMPLE}" ]; then
    SOURCE_FILE="${ENV_EXAMPLE}"
elif [ -f "${ENV_FILE}" ]; then
    SOURCE_FILE="${ENV_FILE}"
fi

if [ -n "${SOURCE_FILE}" ]; then
    AI_ENV_TMP="$(mktemp)"
    grep -E "${AI_KEYS_REGEX}" "${SOURCE_FILE}" > "${AI_ENV_TMP}" || true
    set -a
    # shellcheck disable=SC1090
    . "${AI_ENV_TMP}"
    set +a
    rm -f "${AI_ENV_TMP}"
    echo "  Loaded AI-model settings from ${SOURCE_FILE}"
else
    echo "  No configs/.env.example or docker/.env found, using defaults"
fi

VLM_MODEL_NAME="${VLM_MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
VLM_PRECISION="${VLM_PRECISION:-int8}"
TARGET_DEVICE="${TARGET_DEVICE:-GPU}"
YOLO_MODEL_NAME="${YOLO_MODEL_NAME:-yolo26n-pose}"

# Where OVMS expects models
VLM_MODELS_DIR="${MODELS_DIR}/vlm_models"
YOLO_MODELS_DIR="${MODELS_DIR}/yolo_models"

POTENTIAL_SOURCE_DIRS=(
    "${HOME}/ovms-vlm/models"
    "/opt/ovms/models"
    "${PROJECT_ROOT}/../ovms-vlm/models"
)

echo "=========================================="
echo "Model Setup — Suspicious Activity Detection"
echo "=========================================="
echo "  VLM Model:     ${VLM_MODEL_NAME} (${VLM_PRECISION}, ${TARGET_DEVICE})"
echo "  YOLO Model:    ${YOLO_MODEL_NAME}"
echo "  Models Dir:    ${MODELS_DIR}"
echo ""

###############################################
# HELPERS
###############################################
check_vlm_model() {
    local model_path="$1"
    if [ ! -d "${model_path}" ]; then
        return 1
    fi
    if [ -f "${model_path}/graph.pbtxt" ] && \
       ls "${model_path}"/*.xml > /dev/null 2>&1; then
        echo "  ✓ VLM model found at ${model_path}"
        return 0
    fi
    return 1
}

check_yolo_model() {
    local target_dir="${YOLO_MODELS_DIR}/${YOLO_MODEL_NAME}"
    if [ -f "${target_dir}/${YOLO_MODEL_NAME}.xml" ] && [ -f "${target_dir}/${YOLO_MODEL_NAME}.bin" ]; then
        echo "  ✓ YOLO model found (OpenVINO IR)"
        return 0
    fi
    return 1
}

patch_graph_pbtxt_paths() {
    local model_name="$1"
    local graph_file="${VLM_MODELS_DIR}/${model_name}/graph.pbtxt"
    if [ ! -f "${graph_file}" ]; then
        return 0
    fi
    if grep -qF "${VLM_MODELS_DIR}" "${graph_file}"; then
        sed -i "s|${VLM_MODELS_DIR}|/models/vlm_models|g" "${graph_file}"
        echo "  ✓ graph.pbtxt paths patched (host path → /models/vlm_models)"
    fi
}

update_graph_pbtxt_device() {
    local model_name="$1"
    local graph_file="${VLM_MODELS_DIR}/${model_name}/graph.pbtxt"
    if [ ! -f "${graph_file}" ]; then
        return 0
    fi
    patch_graph_pbtxt_paths "${model_name}"
    local current_device
    current_device=$(grep -oP '(?<=device: ")[^"]+' "${graph_file}" || true)
    if [ "${current_device}" = "${TARGET_DEVICE}" ]; then
        echo "  ✓ graph.pbtxt device already set to ${TARGET_DEVICE}"
        return 0
    fi
    sed -i "s|device: \"${current_device}\"|device: \"${TARGET_DEVICE}\"|g" "${graph_file}"
    echo "  ✓ graph.pbtxt device updated: ${current_device} → ${TARGET_DEVICE}"
}

###############################################
# PYTHON ENVIRONMENT SETUP (for VLM export)
###############################################
_PYTHON_ENV_READY=0
ensure_python_env() {
    if [ "${_PYTHON_ENV_READY}" -eq 1 ]; then
        return 0
    fi

    if [ ! -f "${SCRIPT_DIR}/export_model.py" ]; then
        echo "  Downloading OVMS export tools..."
        EXPORT_BASE_URL="https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2026/0/demos/common/export_models"
        curl -fsSL "${EXPORT_BASE_URL}/export_model.py" -o "${SCRIPT_DIR}/export_model.py"
        curl -fsSL "${EXPORT_BASE_URL}/requirements.txt" -o "${SCRIPT_DIR}/export_requirements.txt"
        echo "  ✓ Export tools downloaded"
    fi

    if [ ! -d "${SCRIPT_DIR}/venv" ] || [ ! -f "${SCRIPT_DIR}/venv/bin/pip" ]; then
        echo "  Creating Python virtual environment..."
        python3 -m venv "${SCRIPT_DIR}/venv" --clear
    fi

    source "${SCRIPT_DIR}/venv/bin/activate"

    # Skip pip installs if requirements haven't changed
    local req_hash
    req_hash=$(md5sum "${SCRIPT_DIR}/export_requirements.txt" 2>/dev/null | cut -d' ' -f1)
    local marker="${SCRIPT_DIR}/venv/.deps_installed_${req_hash}"
    if [ ! -f "${marker}" ]; then
        pip install -q --upgrade pip
        pip install -q -r "${SCRIPT_DIR}/export_requirements.txt"
        rm -f "${SCRIPT_DIR}"/venv/.deps_installed_* 2>/dev/null
        touch "${marker}"
        echo "  ✓ Python environment ready (packages installed)"
    else
        echo "  ✓ Python environment ready (cached)"
    fi
    _PYTHON_ENV_READY=1
}

###############################################
# PARALLEL MODEL DOWNLOAD
###############################################

# --- VLM download function ---
download_vlm() {
echo "------------------------------------------"
echo "[1/2] VLM: ${VLM_MODEL_NAME}"
echo "------------------------------------------"

mkdir -p "${VLM_MODELS_DIR}"
VLM_TARGET_PATH="${VLM_MODELS_DIR}/${VLM_MODEL_NAME}"

if check_vlm_model "${VLM_TARGET_PATH}"; then
    echo "  ✓ VLM model already exists"
    update_graph_pbtxt_device "${VLM_MODEL_NAME}"
else
    # Check external source directories
    VLM_FOUND=0
    for SOURCE_DIR in "${POTENTIAL_SOURCE_DIRS[@]}"; do
        if check_vlm_model "${SOURCE_DIR}/${VLM_MODEL_NAME}"; then
            echo "  Copying VLM model from ${SOURCE_DIR}..."
            mkdir -p "$(dirname "${VLM_MODELS_DIR}/${VLM_MODEL_NAME}")"
            cp -r "${SOURCE_DIR}/${VLM_MODEL_NAME}" "$(dirname "${VLM_MODELS_DIR}/${VLM_MODEL_NAME}")/"
            update_graph_pbtxt_device "${VLM_MODEL_NAME}"
            VLM_FOUND=1
            break
        fi
    done

    if [ "${VLM_FOUND}" -eq 0 ]; then
        echo "  Downloading and exporting VLM from HuggingFace..."
        ensure_python_env

        target_device_args=()
        if [ "${TARGET_DEVICE}" != "CPU" ]; then
            target_device_args=(--target_device "${TARGET_DEVICE}")
        fi

        python "${SCRIPT_DIR}/export_model.py" text_generation \
            --source_model "${VLM_MODEL_NAME}" \
            --weight-format "${VLM_PRECISION}" \
            --pipeline_type VLM_CB \
            "${target_device_args[@]}" \
            --cache_size 32 \
            --max_num_seqs 1 \
            --enable_prefix_caching True \
            --config_file_path "${VLM_MODELS_DIR}/config.json" \
            --model_repository_path "${VLM_MODELS_DIR}" \
            --model_name "${VLM_MODEL_NAME}"

        if check_vlm_model "${VLM_TARGET_PATH}"; then
            patch_graph_pbtxt_paths "${VLM_MODEL_NAME}"
            echo "  ✓ VLM export successful"
        else
            echo "  ✗ VLM export failed"
            return 1
        fi
    fi
fi

# Generate OVMS config.json
echo ""
echo "  Generating OVMS config.json..."
cat > "${VLM_MODELS_DIR}/config.json" << EOF
{
    "model_config_list": [],
    "mediapipe_config_list": [
        {
            "name": "${VLM_MODEL_NAME}",
            "base_path": "${VLM_MODEL_NAME}"
        }
    ]
}
EOF
echo "  ✓ config.json written"

}

# --- YOLO download function ---
download_yolo() {
echo ""
echo "------------------------------------------"
echo "[2/2] YOLO: ${YOLO_MODEL_NAME}"
echo "------------------------------------------"

mkdir -p "${YOLO_MODELS_DIR}"

if check_yolo_model; then
    echo "  ✓ YOLO model already exists"
else
    echo "  Downloading and exporting ${YOLO_MODEL_NAME}..."

    if [ ! -d "${SCRIPT_DIR}/yolo-venv" ] || [ ! -f "${SCRIPT_DIR}/yolo-venv/bin/pip" ]; then
        echo "  Creating YOLO Python environment..."
        python3 -m venv "${SCRIPT_DIR}/yolo-venv" --clear
    fi
    source "${SCRIPT_DIR}/yolo-venv/bin/activate"

    # Skip pip installs if marker exists
    local yolo_marker="${SCRIPT_DIR}/yolo-venv/.deps_installed"
    if [ ! -f "${yolo_marker}" ]; then
        pip install -q --upgrade pip
        pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
        pip install -q ultralytics openvino
        touch "${yolo_marker}"
        echo "  ✓ YOLO dependencies installed"
    else
        echo "  ✓ YOLO dependencies cached"
    fi

    YOLO_MODELS_DIR="${YOLO_MODELS_DIR}" YOLO_MODEL_NAME="${YOLO_MODEL_NAME}" \
    python3 - << 'PYEOF'
import os, shutil, glob
from pathlib import Path
from ultralytics import YOLO

models_dir = Path(os.environ["YOLO_MODELS_DIR"])
model_name = os.environ["YOLO_MODEL_NAME"]
model_pt = models_dir / f"{model_name}.pt"
export_dir = models_dir / f"{model_name}_openvino_model"
target_dir = models_dir / model_name

# Download base weights
if not model_pt.exists():
    print(f"  Downloading {model_name}.pt ...")
    orig = os.getcwd()
    os.chdir(str(models_dir))
    YOLO(f"{model_name}.pt")
    os.chdir(orig)
    print(f"  ✓ Downloaded: {model_pt}")
else:
    print(f"  {model_name}.pt already exists")

# Export to OpenVINO FP32
if not export_dir.exists() and not target_dir.exists():
    print(f"  Exporting to OpenVINO FP32 ...")
    orig = os.getcwd()
    os.chdir(str(models_dir))
    YOLO(str(model_pt)).export(format="openvino", half=False)
    os.chdir(orig)
    print(f"  ✓ FP32 export: {export_dir}")

# Move only .xml and .bin into target_dir, clean up the rest
if export_dir.exists():
    target_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("*.xml", "*.bin"):
        for f in export_dir.glob(ext):
            dest = target_dir / f"{model_name}{f.suffix}"
            shutil.move(str(f), str(dest))
            print(f"  ✓ Moved {f.name} -> {dest}")
    shutil.rmtree(str(export_dir))
    print(f"  ✓ Cleaned up {export_dir.name}")

# Remove .pt file (no longer needed)
if model_pt.exists():
    model_pt.unlink()
    print(f"  ✓ Removed {model_pt.name}")

print("YOLO export complete.")
PYEOF

    deactivate 2>/dev/null || true

    if check_yolo_model; then
        echo "  ✓ YOLO model ready"
    else
        echo "  ✗ YOLO export failed"
        return 1
    fi
fi
}

###############################################
# RUN DOWNLOADS IN PARALLEL
###############################################
VLM_LOG=$(mktemp)
YOLO_LOG=$(mktemp)
trap 'rm -f "${VLM_LOG}" "${YOLO_LOG}"' EXIT

download_vlm > "${VLM_LOG}" 2>&1 &
VLM_PID=$!

download_yolo > "${YOLO_LOG}" 2>&1 &
YOLO_PID=$!

echo "Downloading VLM and YOLO models in parallel..."
echo "  VLM PID:     ${VLM_PID}"
echo "  YOLO PID:    ${YOLO_PID}"
echo ""

# Show progress while waiting
VLM_DONE=0
YOLO_DONE=0
VLM_LINES=0
YOLO_LINES=0
while true; do
    # Check if processes finished
    if [ ${VLM_DONE} -eq 0 ] && ! kill -0 ${VLM_PID} 2>/dev/null; then
        wait ${VLM_PID}
        VLM_RC=$?
        VLM_DONE=1
    fi
    if [ ${YOLO_DONE} -eq 0 ] && ! kill -0 ${YOLO_PID} 2>/dev/null; then
        wait ${YOLO_PID}
        YOLO_RC=$?
        YOLO_DONE=1
    fi

    # Stream new lines from VLM log
    NEW_VLM=$(wc -l < "${VLM_LOG}")
    if [ "${NEW_VLM}" -gt "${VLM_LINES}" ]; then
        sed -n "$((VLM_LINES + 1)),${NEW_VLM}p" "${VLM_LOG}" | sed 's/^/  [VLM]  /'
        VLM_LINES=${NEW_VLM}
    fi

    # Stream new lines from YOLO log
    NEW_YOLO=$(wc -l < "${YOLO_LOG}")
    if [ "${NEW_YOLO}" -gt "${YOLO_LINES}" ]; then
        sed -n "$((YOLO_LINES + 1)),${NEW_YOLO}p" "${YOLO_LOG}" | sed 's/^/  [YOLO] /'
        YOLO_LINES=${NEW_YOLO}
    fi

    # All done? Break.
    if [ ${VLM_DONE} -eq 1 ] && [ ${YOLO_DONE} -eq 1 ]; then
        break
    fi

    sleep 2
done

FAILED=0

if [ ${VLM_RC} -ne 0 ]; then
    echo "  ✗ VLM download/export failed (exit code ${VLM_RC})"
    FAILED=1
fi

if [ ${YOLO_RC} -ne 0 ]; then
    echo "  ✗ YOLO download/export failed (exit code ${YOLO_RC})"
    FAILED=1
fi

if [ ${FAILED} -ne 0 ]; then
    echo "One or more downloads failed."
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ All Model Setup Complete!"
echo "=========================================="
echo "  VLM:     ${VLM_MODELS_DIR}/${VLM_MODEL_NAME}"
echo "  YOLO:    ${YOLO_MODELS_DIR}/${YOLO_MODEL_NAME}"
echo "=========================================="
