#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Download and convert sample video using zone_config.json settings.
# 1. Reads video_url and video_file from the selected use case's zone_config.json
# 2. Uses local file:// media when available, otherwise downloads the raw video
# 3. Converts downloaded media to AVC H.264 at specified resolution/fps using format_avc_mp4.sh
# 4. Places the result in scenescape/sample_data/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
SAMPLE_DATA_DIR="${PROJECT_ROOT}/../scenescape/sample_data"
USE_CASE="${USE_CASE:-retail}"
ZONE_CONFIG="${ZONE_CONFIG:-${PROJECT_ROOT}/configs/usecase/${USE_CASE}/zone_config.json}"
FORMAT_SCRIPT="${PROJECT_ROOT}/../performance-tools/benchmark-scripts/format_avc_mp4.sh"
SAMPLE_MEDIA_DIR="${PROJECT_ROOT}/../performance-tools/sample-media"

# Load camera video settings from zone_config.json
if [ ! -f "${ZONE_CONFIG}" ]; then
    echo "ERROR: zone_config.json not found at ${ZONE_CONFIG}"
    exit 1
fi

# Enumerate cameras. New format: a "cameras" array of {name, video, video_url, fps}.
# Legacy format: flat video_url/video_file/camera_fps fields (single camera).
# Emit one "url<TAB>file<TAB>fps" line per camera.
CAMERAS_TSV=$(python3 -c "
import json
cfg = json.load(open('${ZONE_CONFIG}'))
cams = cfg.get('cameras', [])
rows = []
if cams and isinstance(cams[0], dict):
    for c in cams:
        url = c.get('video_url', '')
        vid = c.get('video', c.get('video_file', ''))
        fps = c.get('fps', c.get('camera_fps', 15))
        rows.append(f'{url}\t{vid}\t{fps}')
else:
    rows.append(f\"{cfg.get('video_url','')}\t{cfg.get('video_file','')}\t{cfg.get('camera_fps',15)}\")
print('\n'.join(rows))
" 2>/dev/null)

# Conversion defaults — overridable via environment
VIDEO_WIDTH="${VIDEO_WIDTH:-1920}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-1080}"

if [ -z "${CAMERAS_TSV}" ]; then
    echo "ERROR: no cameras found in ${ZONE_CONFIG}"
    exit 1
fi

mkdir -p "${SAMPLE_DATA_DIR}"

resolve_file_url() {
    local url="$1"
    local path="${url#file://}"

    if [[ "${path}" == localhost/* ]]; then
        path="/${path#localhost/}"
    fi
    if [ -f "${path}" ]; then
        printf '%s\n' "${path}"
        return 0
    fi
    if [[ "${path}" == /storewide-loss-prevention/* ]]; then
        local repo_path
        repo_path="$(dirname "${PROJECT_ROOT}")/${path#/storewide-loss-prevention/}"
        if [ -f "${repo_path}" ]; then
            printf '%s\n' "${repo_path}"
            return 0
        fi
    fi
    return 1
}

# --- Download & convert a single camera video ---
# Args: <url> <filename> <fps>
download_one() {
    local camera_url="$1"
    local filename="$2"
    local fps="$3"
    local video_fps="${VIDEO_FPS:-${fps}}"
    local usecase_dir="${SAMPLE_DATA_DIR}/${USE_CASE}"
    local canonical_path="${usecase_dir}/${filename}"
    local output_path="${SAMPLE_DATA_DIR}/${filename}"

    if [ -z "${camera_url}" ]; then
        echo "  ✗ Skipping '${filename}': video_url is empty in ${ZONE_CONFIG}"
        return 1
    fi
    if [ -z "${filename}" ]; then
        echo "  ✗ Skipping camera: video filename is empty in ${ZONE_CONFIG}"
        return 1
    fi

    echo "=========================================="
    echo "Sample Video Download & Convert"
    echo "=========================================="
    echo "  URL:         ${camera_url}"
    echo "  Canonical:   ${canonical_path}"
    echo "  Active:      ${output_path}"
    echo "  Resolution:  ${VIDEO_WIDTH}x${VIDEO_HEIGHT} @ ${video_fps}fps"
    echo ""

    mkdir -p "${usecase_dir}"

    # Local file:// source: stage (overwrite) so switching use cases restages the
    # correct video even when the flat output filename is shared (e.g. lp-camera1.mp4).
    if [[ "${camera_url}" == file://* ]]; then
        local source_path
        if ! source_path="$(resolve_file_url "${camera_url}")"; then
            echo "  ✗ Local video not found for ${camera_url}"
            return 1
        fi
        if [ ! -f "${canonical_path}" ] || ! cmp -s "${source_path}" "${canonical_path}"; then
            cp "${source_path}" "${canonical_path}"
            echo "  ✓ Updated canonical video: ${source_path} -> ${canonical_path}"
        fi
        if [ -f "${output_path}" ] && cmp -s "${canonical_path}" "${output_path}"; then
            echo "  ✓ Video already staged: ${output_path}"
            return 0
        fi
        cp "${canonical_path}" "${output_path}"
        echo "  ✓ Staged local video: ${canonical_path} -> ${output_path}"
        return 0
    fi

    # --- Download & convert using format_avc_mp4.sh ---
    if [ ! -f "${canonical_path}" ] && [ -f "${FORMAT_SCRIPT}" ]; then
        echo "  Converting via format_avc_mp4.sh (${VIDEO_WIDTH}x${VIDEO_HEIGHT} @ ${video_fps}fps)..."
        mkdir -p "${SAMPLE_MEDIA_DIR}"

        # Derive the bench filename that format_avc_mp4.sh will produce
        local basename="${filename%.mp4}"
        local bench_file="${basename}-${VIDEO_WIDTH}-${video_fps}-bench.mp4"

        # Run format_avc_mp4.sh from its expected directory
        pushd "${PROJECT_ROOT}/../performance-tools/benchmark-scripts" > /dev/null
        bash format_avc_mp4.sh "${filename}" "${camera_url}" "${VIDEO_WIDTH}" "${VIDEO_HEIGHT}" "${video_fps}"
        popd > /dev/null

        # Save the converted file under the use-case folder, then stage it flat.
        if [ -f "${SAMPLE_MEDIA_DIR}/${bench_file}" ]; then
            mv "${SAMPLE_MEDIA_DIR}/${bench_file}" "${canonical_path}"
            echo "  ✓ Converted video saved: ${canonical_path}"
        else
            echo "  ✗ Conversion failed — bench file not found: ${bench_file}"
            return 1
        fi
    elif [ ! -f "${canonical_path}" ]; then
        # Fallback: direct download without conversion
        echo "  format_avc_mp4.sh not found, downloading raw video..."
        curl -fL --progress-bar -o "${canonical_path}" "${camera_url}"

        if [ -f "${canonical_path}" ] && [ -s "${canonical_path}" ]; then
            local file_size
            file_size=$(du -h "${canonical_path}" | cut -f1)
            echo "  ✓ Download complete: ${canonical_path} (${file_size})"
        else
            echo "  ✗ Download failed"
            rm -f "${canonical_path}"
            return 1
        fi
    else
        echo "  ✓ Canonical video already exists: ${canonical_path}"
    fi

    if [ -f "${output_path}" ] && cmp -s "${canonical_path}" "${output_path}"; then
        echo "  ✓ Video already staged: ${output_path}"
        return 0
    fi

    cp "${canonical_path}" "${output_path}"
    echo "  ✓ Staged active video: ${canonical_path} -> ${output_path}"
}

# --- Download every camera's video ---
FAILED=0
while IFS=$'\t' read -r cam_url cam_file cam_fps; do
    [ -z "${cam_url}${cam_file}" ] && continue
    download_one "${cam_url}" "${cam_file}" "${cam_fps}" || FAILED=1
done <<< "${CAMERAS_TSV}"

if [ "${FAILED}" != "0" ]; then
    exit 1
fi
