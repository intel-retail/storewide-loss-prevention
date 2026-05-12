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
AI_KEYS_REGEX='^(VLM_ENABLED|VLM_MODEL_NAME|VLM_PRECISION|TARGET_DEVICE|YOLO_MODEL_NAME|DETECT_MODEL|REID_MODEL)='

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
YOLO_MODEL_NAME="${YOLO_MODEL_NAME:-yolo11n-pose}"
DETECT_MODEL="${DETECT_MODEL:-yolo11s}"
REID_MODEL="${REID_MODEL:-person-reidentification-retail-0277}"

# Model directories (unified paths)
VLM_MODELS_DIR="${MODELS_DIR}/vlm_models"
YOLO_MODELS_DIR="${MODELS_DIR}/yolo_models"
DETECT_DIR="${MODELS_DIR}/detect_models"
REID_DIR="${MODELS_DIR}/reid_models"

# OpenVINO Model Zoo download URL (for non-YOLO models)
OMZ_BASE_URL="https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1"
MODEL_PROC_DIR="${PROJECT_ROOT}/../scenescape/dlstreamer-pipeline-server/model-proc-files"

POTENTIAL_SOURCE_DIRS=(
    "${HOME}/ovms-vlm/models"
    "/opt/ovms/models"
    "${PROJECT_ROOT}/../ovms-vlm/models"
)

echo "=========================================="
echo "Model Setup — Suspicious Activity Detection"
echo "=========================================="
echo "  VLM Model:     ${VLM_MODEL_NAME} (${VLM_PRECISION}, ${TARGET_DEVICE})"
echo "  YOLO Pose:     ${YOLO_MODEL_NAME}"
echo "  Detect Model:  ${DETECT_MODEL}"
echo "  ReID Model:    ${REID_MODEL}"
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
    else
        sed -i "s|device: \"${current_device}\"|device: \"${TARGET_DEVICE}\"|g" "${graph_file}"
        echo "  ✓ graph.pbtxt device updated: ${current_device} → ${TARGET_DEVICE}"
    fi
    # Ensure max_num_seqs is set to 1 (single-request concurrency)
    local current_seqs
    current_seqs=$(grep -oP '(?<=max_num_seqs: )\d+' "${graph_file}" || echo "0")
    if [ "${current_seqs}" -ne 1 ]; then
        sed -i "s|max_num_seqs: ${current_seqs}|max_num_seqs: 1|g" "${graph_file}"
        echo "  ✓ graph.pbtxt max_num_seqs updated: ${current_seqs} → 1"
    fi
    # Ensure cache_size is 4 GB (not 32) to reduce memory footprint
    local current_cache
    current_cache=$(grep -oP '(?<=cache_size: )\d+' "${graph_file}" || echo "0")
    if [ "${current_cache}" -ne 4 ]; then
        sed -i "s|cache_size: ${current_cache}|cache_size: 4|g" "${graph_file}"
        echo "  ✓ graph.pbtxt cache_size updated: ${current_cache} → 4"
    fi
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
            --cache_size 4 \
            --max_num_seqs 1 \
            --enable_prefix_caching True \
            --config_file_path "${VLM_MODELS_DIR}/config.json" \
            --model_repository_path "${VLM_MODELS_DIR}" \
            --model_name "${VLM_MODEL_NAME}"

        if check_vlm_model "${VLM_TARGET_PATH}"; then
            update_graph_pbtxt_device "${VLM_MODEL_NAME}"
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

# Patch graph.pbtxt device to match TARGET_DEVICE
local GRAPH_FILE="${VLM_MODELS_DIR}/${VLM_MODEL_NAME}/graph.pbtxt"
if [ -f "${GRAPH_FILE}" ]; then
    sed -i "s|device: \"[^\"]*\"|device: \"${TARGET_DEVICE:-GPU}\"|g" "${GRAPH_FILE}"
    echo "  ✓ graph.pbtxt device set to ${TARGET_DEVICE:-GPU}"
fi

}

# --- Detection model download function (YOLO or OpenVINO) ---
download_detect() {
echo ""
echo "------------------------------------------"
echo "[2/4] Detect Model: ${DETECT_MODEL}"
echo "------------------------------------------"

mkdir -p "${DETECT_DIR}"
local target_dir="${DETECT_DIR}/${DETECT_MODEL}"

if [ -f "${target_dir}/${DETECT_MODEL}.xml" ] && [ -f "${target_dir}/${DETECT_MODEL}.bin" ]; then
    echo "  ✓ Detect model already exists"
else
    if [[ "${DETECT_MODEL}" == yolo* ]]; then
        # --- YOLO model: export via ultralytics ---
        echo "  Downloading and exporting ${DETECT_MODEL} (YOLO)..."

        if [ ! -d "${SCRIPT_DIR}/yolo-detect-venv" ] || [ ! -f "${SCRIPT_DIR}/yolo-detect-venv/bin/pip" ]; then
            echo "  Creating YOLO detect Python environment..."
            python3 -m venv "${SCRIPT_DIR}/yolo-detect-venv" --clear
        fi
        source "${SCRIPT_DIR}/yolo-detect-venv/bin/activate"

        local detect_marker="${SCRIPT_DIR}/yolo-detect-venv/.deps_installed"
        if [ ! -f "${detect_marker}" ]; then
            pip install -q --upgrade pip
            pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
            pip install -q ultralytics openvino
            touch "${detect_marker}"
            echo "  ✓ YOLO detect dependencies installed"
        else
            echo "  ✓ YOLO detect dependencies cached"
        fi

        DETECT_DIR="${DETECT_DIR}" DETECT_MODEL="${DETECT_MODEL}" \
        python3 - << 'PYEOF'
import os, shutil
from pathlib import Path
from ultralytics import YOLO

models_dir = Path(os.environ["DETECT_DIR"])
model_name = os.environ["DETECT_MODEL"]
model_pt = models_dir / f"{model_name}.pt"
export_dir = models_dir / f"{model_name}_openvino_model"
target_dir = models_dir / model_name

if not model_pt.exists():
    print(f"  Downloading {model_name}.pt ...")
    orig = os.getcwd()
    os.chdir(str(models_dir))
    YOLO(f"{model_name}.pt")
    os.chdir(orig)
    print(f"  ✓ Downloaded: {model_pt}")

if not export_dir.exists() and not target_dir.exists():
    print(f"  Exporting to OpenVINO FP16 ...")
    orig = os.getcwd()
    os.chdir(str(models_dir))
    YOLO(str(model_pt)).export(format="openvino", dynamic=False, half=True, imgsz=640)
    os.chdir(orig)

if export_dir.exists():
    target_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("*.xml", "*.bin"):
        for f in export_dir.glob(ext):
            dest = target_dir / f"{model_name}{f.suffix}"
            shutil.move(str(f), str(dest))
            print(f"  ✓ Moved {f.name} -> {dest}")
    shutil.rmtree(str(export_dir))

if model_pt.exists():
    model_pt.unlink()
    print(f"  ✓ Removed {model_pt.name}")
PYEOF

        deactivate 2>/dev/null || true
    else
        # --- OpenVINO Model Zoo: download via wget ---
        local prec="${MODEL_PRECISION:-FP32}"
        echo "  Downloading ${DETECT_MODEL} (${prec}) from OpenVINO Model Zoo..."
        mkdir -p "${target_dir}"
        wget -nv -O "${target_dir}/${DETECT_MODEL}.xml" \
            "${OMZ_BASE_URL}/${DETECT_MODEL}/${prec}/${DETECT_MODEL}.xml"
        wget -nv -O "${target_dir}/${DETECT_MODEL}.bin" \
            "${OMZ_BASE_URL}/${DETECT_MODEL}/${prec}/${DETECT_MODEL}.bin"
    fi

    if [ -f "${target_dir}/${DETECT_MODEL}.xml" ] && [ -f "${target_dir}/${DETECT_MODEL}.bin" ]; then
        echo "  ✓ Detect model ready"
    else
        echo "  ✗ Detect model download/export failed"
        return 1
    fi
fi

# Copy model-proc file
if [[ "${DETECT_MODEL}" == yolo* ]]; then
    # YOLO models use yolo-v8.json
    if [ ! -f "${target_dir}/yolo-v8.json" ]; then
        local MODELPROC_SRC="${MODEL_PROC_DIR}/yolo-v8.json"
        if [ -f "${MODELPROC_SRC}" ]; then
            cp "${MODELPROC_SRC}" "${target_dir}/yolo-v8.json"
            echo "  ✓ yolo-v8.json model-proc copied"
        else
            local container_image="docker.io/intel/dlstreamer-pipeline-server:${DLSTREAMER_VERSION:-2026.1.0-20260331-weekly-ubuntu24}"
            if docker create --name modelproctmp "${container_image}" true >/dev/null 2>&1; then
                docker cp "modelproctmp:/opt/intel/dlstreamer/samples/gstreamer/model_proc/public/yolo-v8.json" "${target_dir}/yolo-v8.json" 2>/dev/null && \
                    echo "  ✓ yolo-v8.json extracted from DLStreamer image" || \
                    echo "  ⚠ Failed to extract yolo-v8.json"
                docker rm modelproctmp >/dev/null 2>&1
            fi
        fi
    else
        echo "  ✓ yolo-v8.json model-proc already exists"
    fi
    # Copy labels.txt
    local LABELS_SRC="${MODEL_PROC_DIR}/labels.txt"
    if [ -f "${LABELS_SRC}" ] && [ ! -f "${target_dir}/labels.txt" ]; then
        cp "${LABELS_SRC}" "${target_dir}/labels.txt"
        echo "  ✓ labels.txt copied"
    elif [ -f "${target_dir}/labels.txt" ]; then
        echo "  ✓ labels.txt already exists"
    fi
else
    # OpenVINO models use {model_name}.json
    if [ ! -f "${target_dir}/${DETECT_MODEL}.json" ]; then
        local MODELPROC_SRC="${MODEL_PROC_DIR}/${DETECT_MODEL}.json"
        if [ -f "${MODELPROC_SRC}" ]; then
            cp "${MODELPROC_SRC}" "${target_dir}/${DETECT_MODEL}.json"
            echo "  ✓ ${DETECT_MODEL}.json model-proc copied"
        else
            local container_image="docker.io/intel/dlstreamer-pipeline-server:${DLSTREAMER_VERSION:-2026.1.0-20260331-weekly-ubuntu24}"
            if docker create --name modelproctmp "${container_image}" true >/dev/null 2>&1; then
                docker cp "modelproctmp:/opt/intel/dlstreamer/samples/gstreamer/model_proc/intel/${DETECT_MODEL}.json" "${target_dir}/${DETECT_MODEL}.json" 2>/dev/null && \
                    echo "  ✓ ${DETECT_MODEL}.json extracted from DLStreamer image" || \
                    echo "  ⚠ Failed to extract ${DETECT_MODEL}.json"
                docker rm modelproctmp >/dev/null 2>&1
            fi
        fi
    else
        echo "  ✓ ${DETECT_MODEL}.json model-proc already exists"
    fi
fi
}

# --- ReID model download function (OpenVINO Model Zoo) ---
download_reid() {
echo ""
echo "------------------------------------------"
echo "[3/4] ReID Model: ${REID_MODEL}"
echo "------------------------------------------"

mkdir -p "${REID_DIR}"
local target_dir="${REID_DIR}/${REID_MODEL}"

if [ -f "${target_dir}/${REID_MODEL}.xml" ] && [ -f "${target_dir}/${REID_MODEL}.bin" ]; then
    echo "  ✓ ReID model already exists"
else
    local prec="${MODEL_PRECISION:-FP32}"
    echo "  Downloading ${REID_MODEL} (${prec}) from OpenVINO Model Zoo..."
    mkdir -p "${target_dir}"
    wget -nv -O "${target_dir}/${REID_MODEL}.xml" \
        "${OMZ_BASE_URL}/${REID_MODEL}/${prec}/${REID_MODEL}.xml"
    wget -nv -O "${target_dir}/${REID_MODEL}.bin" \
        "${OMZ_BASE_URL}/${REID_MODEL}/${prec}/${REID_MODEL}.bin"

    if [ -f "${target_dir}/${REID_MODEL}.xml" ] && [ -f "${target_dir}/${REID_MODEL}.bin" ]; then
        echo "  ✓ ReID model ready"
    else
        echo "  ✗ ReID model download failed"
        return 1
    fi
fi
}

# --- YOLO pose download function ---
download_yolo() {
echo ""
echo "------------------------------------------"
echo "[4/4] YOLO Pose: ${YOLO_MODEL_NAME}"
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
DETECT_LOG=$(mktemp)
REID_LOG=$(mktemp)
YOLO_LOG=$(mktemp)
trap 'rm -f "${VLM_LOG}" "${DETECT_LOG}" "${REID_LOG}" "${YOLO_LOG}"' EXIT

download_vlm > "${VLM_LOG}" 2>&1 &
VLM_PID=$!

download_detect > "${DETECT_LOG}" 2>&1 &
DETECT_PID=$!

download_reid > "${REID_LOG}" 2>&1 &
REID_PID=$!

download_yolo > "${YOLO_LOG}" 2>&1 &
YOLO_PID=$!

echo "Downloading all models in parallel..."
echo "  VLM PID:    ${VLM_PID}"
echo "  Detect PID: ${DETECT_PID}"
echo "  ReID PID:   ${REID_PID}"
echo "  YOLO PID:   ${YOLO_PID}"
echo ""

# Show progress while waiting
VLM_DONE=0
DETECT_DONE=0
REID_DONE=0
YOLO_DONE=0
VLM_LINES=0
DETECT_LINES=0
REID_LINES=0
YOLO_LINES=0
while true; do
    if [ ${VLM_DONE} -eq 0 ] && ! kill -0 ${VLM_PID} 2>/dev/null; then
        wait ${VLM_PID}
        VLM_RC=$?
        VLM_DONE=1
    fi
    if [ ${DETECT_DONE} -eq 0 ] && ! kill -0 ${DETECT_PID} 2>/dev/null; then
        wait ${DETECT_PID}
        DETECT_RC=$?
        DETECT_DONE=1
    fi
    if [ ${REID_DONE} -eq 0 ] && ! kill -0 ${REID_PID} 2>/dev/null; then
        wait ${REID_PID}
        REID_RC=$?
        REID_DONE=1
    fi
    if [ ${YOLO_DONE} -eq 0 ] && ! kill -0 ${YOLO_PID} 2>/dev/null; then
        wait ${YOLO_PID}
        YOLO_RC=$?
        YOLO_DONE=1
    fi

    NEW_VLM=$(wc -l < "${VLM_LOG}")
    if [ "${NEW_VLM}" -gt "${VLM_LINES}" ]; then
        sed -n "$((VLM_LINES + 1)),${NEW_VLM}p" "${VLM_LOG}" | sed 's/^/  [VLM]    /'
        VLM_LINES=${NEW_VLM}
    fi
    NEW_DETECT=$(wc -l < "${DETECT_LOG}")
    if [ "${NEW_DETECT}" -gt "${DETECT_LINES}" ]; then
        sed -n "$((DETECT_LINES + 1)),${NEW_DETECT}p" "${DETECT_LOG}" | sed 's/^/  [DETECT] /'
        DETECT_LINES=${NEW_DETECT}
    fi
    NEW_REID=$(wc -l < "${REID_LOG}")
    if [ "${NEW_REID}" -gt "${REID_LINES}" ]; then
        sed -n "$((REID_LINES + 1)),${NEW_REID}p" "${REID_LOG}" | sed 's/^/  [REID]   /'
        REID_LINES=${NEW_REID}
    fi
    NEW_YOLO=$(wc -l < "${YOLO_LOG}")
    if [ "${NEW_YOLO}" -gt "${YOLO_LINES}" ]; then
        sed -n "$((YOLO_LINES + 1)),${NEW_YOLO}p" "${YOLO_LOG}" | sed 's/^/  [POSE]   /'
        YOLO_LINES=${NEW_YOLO}
    fi

    if [ ${VLM_DONE} -eq 1 ] && [ ${DETECT_DONE} -eq 1 ] && [ ${REID_DONE} -eq 1 ] && [ ${YOLO_DONE} -eq 1 ]; then
        break
    fi

    sleep 2
done

FAILED=0

if [ ${VLM_RC} -ne 0 ]; then
    echo "  ✗ VLM download/export failed (exit code ${VLM_RC})"
    echo "  --- VLM error log ---"
    cat "${VLM_LOG}"
    echo "  --- end VLM log ---"
    FAILED=1
fi

if [ ${DETECT_RC} -ne 0 ]; then
    echo "  ✗ Detect model download/export failed (exit code ${DETECT_RC})"
    echo "  --- Detect error log ---"
    cat "${DETECT_LOG}"
    echo "  --- end Detect log ---"
    FAILED=1
fi

if [ ${REID_RC} -ne 0 ]; then
    echo "  ✗ ReID model download failed (exit code ${REID_RC})"
    echo "  --- ReID error log ---"
    cat "${REID_LOG}"
    echo "  --- end ReID log ---"
    FAILED=1
fi

if [ ${YOLO_RC} -ne 0 ]; then
    echo "  ✗ YOLO pose download/export failed (exit code ${YOLO_RC})"
    echo "  --- YOLO error log ---"
    cat "${YOLO_LOG}"
    echo "  --- end YOLO log ---"
    FAILED=1
fi

if [ ${FAILED} -ne 0 ]; then
    echo "One or more downloads failed. See error logs above."
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ All Model Setup Complete!"
echo "=========================================="
echo "  VLM:         ${VLM_MODELS_DIR}/${VLM_MODEL_NAME}"
echo "  Detect:      ${DETECT_DIR}/${DETECT_MODEL}"
echo "  ReID:        ${REID_DIR}/${REID_MODEL}"
echo "  YOLO Pose:   ${YOLO_MODELS_DIR}/${YOLO_MODEL_NAME}"
echo "=========================================="
