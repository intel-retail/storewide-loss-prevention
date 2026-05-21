#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Download and convert sample video using zone_config.json settings.
# 1. Reads video_url and video_file from configs/zone_config.json
# 2. Downloads the raw video
# 3. Converts to AVC H.264 at specified resolution/fps using format_avc_mp4.sh
# 4. Places the result in scenescape/sample_data/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
SAMPLE_DATA_DIR="${PROJECT_ROOT}/../scenescape/sample_data"
ZONE_CONFIG="${PROJECT_ROOT}/configs/zone_config.json"
FORMAT_SCRIPT="${PROJECT_ROOT}/../performance-tools/benchmark-scripts/format_avc_mp4.sh"
SAMPLE_MEDIA_DIR="${PROJECT_ROOT}/../performance-tools/sample-media"

# Load video_url, video_file, and camera_fps from zone_config.json
if [ ! -f "${ZONE_CONFIG}" ]; then
    echo "ERROR: zone_config.json not found at ${ZONE_CONFIG}"
    exit 1
fi

CAMERA_URL=$(python3 -c "import json; print(json.load(open('${ZONE_CONFIG}')).get('video_url', ''))" 2>/dev/null)
FILENAME=$(python3 -c "import json; print(json.load(open('${ZONE_CONFIG}')).get('video_file', ''))" 2>/dev/null)
CAMERA_FPS=$(python3 -c "import json; print(json.load(open('${ZONE_CONFIG}')).get('camera_fps', 15))" 2>/dev/null)

# Conversion defaults — camera_fps from zone_config, overridable via environment
VIDEO_WIDTH="${VIDEO_WIDTH:-1920}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-1080}"
VIDEO_FPS="${VIDEO_FPS:-${CAMERA_FPS}}"

if [ -z "${CAMERA_URL}" ]; then
    echo "ERROR: video_url not found in ${ZONE_CONFIG}"
    exit 1
fi

if [ -z "${FILENAME}" ]; then
    echo "ERROR: video_file not found in ${ZONE_CONFIG}"
    exit 1
fi

echo "=========================================="
echo "Sample Video Download & Convert"
echo "=========================================="
echo "  URL:         ${CAMERA_URL}"
echo "  Output:      ${SAMPLE_DATA_DIR}/${FILENAME}"
echo "  Resolution:  ${VIDEO_WIDTH}x${VIDEO_HEIGHT} @ ${VIDEO_FPS}fps"
echo ""

mkdir -p "${SAMPLE_DATA_DIR}"

OUTPUT_PATH="${SAMPLE_DATA_DIR}/${FILENAME}"

if [ -f "${OUTPUT_PATH}" ]; then
    echo "  ✓ Video already exists: ${OUTPUT_PATH}"
    exit 0
fi

# --- Step 1: Download & convert using format_avc_mp4.sh ---
if [ -f "${FORMAT_SCRIPT}" ]; then
    echo "  Converting via format_avc_mp4.sh (${VIDEO_WIDTH}x${VIDEO_HEIGHT} @ ${VIDEO_FPS}fps)..."
    mkdir -p "${SAMPLE_MEDIA_DIR}"

    # Derive the bench filename that format_avc_mp4.sh will produce
    BASENAME="${FILENAME%.mp4}"
    BENCH_FILE="${BASENAME}-${VIDEO_WIDTH}-${VIDEO_FPS}-bench.mp4"

    # Run format_avc_mp4.sh from its expected directory
    pushd "${PROJECT_ROOT}/../performance-tools/benchmark-scripts" > /dev/null
    bash format_avc_mp4.sh "${FILENAME}" "${CAMERA_URL}" "${VIDEO_WIDTH}" "${VIDEO_HEIGHT}" "${VIDEO_FPS}"
    popd > /dev/null

    # Move the converted file to sample_data with the expected name
    if [ -f "${SAMPLE_MEDIA_DIR}/${BENCH_FILE}" ]; then
        mv "${SAMPLE_MEDIA_DIR}/${BENCH_FILE}" "${OUTPUT_PATH}"
        echo "  ✓ Converted video saved: ${OUTPUT_PATH}"
    else
        echo "  ✗ Conversion failed — bench file not found: ${BENCH_FILE}"
        exit 1
    fi
else
    # Fallback: direct download without conversion
    echo "  format_avc_mp4.sh not found, downloading raw video..."
    wget -q --show-progress -O "${OUTPUT_PATH}" "${CAMERA_URL}"

    if [ -f "${OUTPUT_PATH}" ] && [ -s "${OUTPUT_PATH}" ]; then
        FILE_SIZE=$(du -h "${OUTPUT_PATH}" | cut -f1)
        echo "  ✓ Download complete: ${OUTPUT_PATH} (${FILE_SIZE})"
    else
        echo "  ✗ Download failed"
        rm -f "${OUTPUT_PATH}"
        exit 1
    fi
fi
