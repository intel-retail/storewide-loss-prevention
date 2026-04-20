#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Import SceneScape scene .zip file(s) via the REST API.
# Runs as a sidecar container after the web service is healthy.
#
# When STREAM_DENSITY > 1, clones the base scene zip on-the-fly with unique
# scene names and camera IDs, uploads each clone, then cleans up.

set -e

# Install curl (not present in python:3.12-slim)
apt-get update -qq && apt-get install -y -qq curl > /dev/null 2>&1

SCENE_ZIP_NAME="${SCENE_ZIP:-}"
STREAM_DENSITY="${STREAM_DENSITY:-1}"
SCENE_NAME="${SCENE_NAME:-}"
CAMERA_NAME="${CAMERA_NAME:-}"
SCENESCAPE_URL="${SCENESCAPE_URL:-https://web.scenescape.intel.com}"
SCENESCAPE_USER="${SCENESCAPE_USER:-admin}"
SCENESCAPE_PASSWORD="${SCENESCAPE_PASSWORD:-${SUPASS}}"
CA_CERT="${CA_CERT:-/run/secrets/certs/scenescape-ca.pem}"
MAX_RETRIES="${MAX_RETRIES:-60}"
RETRY_INTERVAL="${RETRY_INTERVAL:-5}"

echo "=== SceneScape Scene Import ==="
echo "  Stream density: ${STREAM_DENSITY}"

# Build list of zip files to import
# If STREAM_DENSITY > 1, clone the base zip on-the-fly
ZIP_FILES=()
CLONE_DIR=""

if [ "${STREAM_DENSITY}" -gt 1 ] && [ -n "${SCENE_ZIP_NAME}" ]; then
    BASE_ZIP="/webserver/${SCENE_ZIP_NAME}"
    if [ ! -f "${BASE_ZIP}" ]; then
        echo "ERROR: Base scene zip not found: ${BASE_ZIP}"
        exit 1
    fi
    CLONE_DIR=$(mktemp -d)
    echo "  Cloning base zip ${STREAM_DENSITY} times..."
    python3 /scripts/clone_scene_zip.py \
        "${BASE_ZIP}" "${CLONE_DIR}" "${SCENE_NAME}" "${CAMERA_NAME}" "${STREAM_DENSITY}" > /dev/null
    for f in "${CLONE_DIR}"/*.zip; do
        [ -f "$f" ] && ZIP_FILES+=("$f")
    done
    echo "  Generated ${#ZIP_FILES[@]} cloned zips in ${CLONE_DIR}"
elif [ -n "${SCENE_ZIP_NAME}" ]; then
    IFS=',' read -ra ZIP_NAMES <<< "${SCENE_ZIP_NAME}"
    for name in "${ZIP_NAMES[@]}"; do
        name=$(echo "$name" | xargs)
        [ -n "$name" ] && ZIP_FILES+=("/webserver/${name}")
    done
else
    for f in /webserver/*.zip; do
        [ -f "$f" ] && ZIP_FILES+=("$f")
    done
fi

if [ ${#ZIP_FILES[@]} -eq 0 ]; then
    echo "ERROR: No .zip files found and SCENE_ZIP is not set."
    exit 1
fi

echo "  Found ${#ZIP_FILES[@]} zip file(s) to import."
echo "  API URL:  ${SCENESCAPE_URL}"
echo "  User:     ${SCENESCAPE_USER}"

# Build curl TLS flags
CURL_TLS_FLAGS="-k"
if [ -f "${CA_CERT}" ]; then
    CURL_TLS_FLAGS="--cacert ${CA_CERT}"
fi

# Wait for SceneScape web to be healthy
echo "Waiting for SceneScape web service..."
for i in $(seq 1 ${MAX_RETRIES}); do
    HEALTH=$(curl -s ${CURL_TLS_FLAGS} "${SCENESCAPE_URL}/api/v1/database-ready" 2>/dev/null || echo "")
    if echo "$HEALTH" | grep -q "true"; then
        echo "  Web service is ready (attempt ${i}/${MAX_RETRIES})"
        break
    fi
    if [ "$i" -eq "${MAX_RETRIES}" ]; then
        echo "ERROR: Web service did not become ready after ${MAX_RETRIES} attempts"
        exit 1
    fi
    echo "  Waiting... (attempt ${i}/${MAX_RETRIES})"
    sleep ${RETRY_INTERVAL}
done

# Authenticate and get token
echo "Authenticating..."
AUTH_RESPONSE=$(curl -s ${CURL_TLS_FLAGS} \
    -X POST "${SCENESCAPE_URL}/api/v1/auth" \
    -H "Content-Type: application/json" \
    -d "{\"username\": \"${SCENESCAPE_USER}\", \"password\": \"${SCENESCAPE_PASSWORD}\"}" 2>/dev/null)

TOKEN=$(echo "$AUTH_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('token',''))" 2>/dev/null || echo "")

if [ -z "${TOKEN}" ]; then
    echo "ERROR: Failed to authenticate. Response: ${AUTH_RESPONSE}"
    exit 1
fi
echo "  Authenticated successfully."

# Import each zip file
IMPORT_SUCCESS=0
IMPORT_FAIL=0

for SCENE_ZIP in "${ZIP_FILES[@]}"; do
    ZIP_BASENAME=$(basename "${SCENE_ZIP}")
    echo ""
    echo "--- Importing: ${ZIP_BASENAME} ---"

    if [ ! -f "${SCENE_ZIP}" ]; then
        echo "  WARNING: File not found: ${SCENE_ZIP}. Skipping."
        IMPORT_FAIL=$((IMPORT_FAIL + 1))
        continue
    fi

    echo "  Uploading ${ZIP_BASENAME}..."
    IMPORT_RESPONSE=$(curl -s ${CURL_TLS_FLAGS} \
        -X POST "${SCENESCAPE_URL}/api/v1/import-scene/" \
        -H "Authorization: token ${TOKEN}" \
        -F "zipFile=@${SCENE_ZIP}" 2>/dev/null)

    echo "  Import response: ${IMPORT_RESPONSE}"
    IMPORT_SUCCESS=$((IMPORT_SUCCESS + 1))
done

# Cleanup cloned zips
if [ -n "${CLONE_DIR}" ] && [ -d "${CLONE_DIR}" ]; then
    rm -rf "${CLONE_DIR}"
    echo "  Cleaned up temporary clones."
fi

echo ""
echo "=== Scene Import Summary ==="
echo "  Total:     ${#ZIP_FILES[@]}"
echo "  Imported:  ${IMPORT_SUCCESS}"
echo "  Failed:    ${IMPORT_FAIL}"

if [ ${IMPORT_FAIL} -gt 0 ]; then
    echo "  Some imports failed. Check logs above or use SceneScape UI > Import Scene."
fi
