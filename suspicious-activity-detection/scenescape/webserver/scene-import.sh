#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Import a SceneScape scene .zip via the REST API.
# Runs as a sidecar container after the web service is healthy.
#
# Expects env vars: SCENE_NAME, SCENE_ZIP (filename only)
# The zip file is mounted at /webserver/<SCENE_ZIP>

set -e

# Install curl (not present in python:3.12-slim)
apt-get update -qq && apt-get install -y -qq curl > /dev/null 2>&1

SCENE_NAME="${SCENE_NAME:-}"
SCENE_ZIP_NAME="${SCENE_ZIP:-}"
SCENESCAPE_URL="${SCENESCAPE_URL:-https://web.scenescape.intel.com}"
SCENESCAPE_USER="${SCENESCAPE_USER:-admin}"
SCENESCAPE_PASSWORD="${SCENESCAPE_PASSWORD:-${SUPASS}}"
CA_CERT="${CA_CERT:-/run/secrets/certs/scenescape-ca.pem}"
MAX_RETRIES="${MAX_RETRIES:-60}"
RETRY_INTERVAL="${RETRY_INTERVAL:-5}"

echo "=== SceneScape Scene Import ==="

if [ -z "${SCENE_NAME}" ] || [ -z "${SCENE_ZIP_NAME}" ]; then
    echo "ERROR: SCENE_NAME and SCENE_ZIP env vars are required."
    echo "  These are set automatically by init.sh from zone_config.json."
    exit 1
fi

SCENE_ZIP="/webserver/${SCENE_ZIP_NAME}"

echo "  Scene name:  ${SCENE_NAME}"
echo "  Scene ZIP:   ${SCENE_ZIP}"
echo "  API URL:     ${SCENESCAPE_URL}"
echo "  User:        ${SCENESCAPE_USER}"

# Validate scene zip exists
if [ ! -f "${SCENE_ZIP}" ]; then
    echo "ERROR: Scene file not found: ${SCENE_ZIP}"
    exit 1
fi

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

# Check if scene already exists by name
echo "Checking for existing scene '${SCENE_NAME}'..."
SCENES_RESPONSE=$(curl -s ${CURL_TLS_FLAGS} \
    -H "Authorization: token ${TOKEN}" \
    "${SCENESCAPE_URL}/api/v1/scenes" 2>/dev/null)

ALREADY_EXISTS=$(echo "$SCENES_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    results = data.get('results', data) if isinstance(data, dict) else data
    if isinstance(results, list):
        for s in results:
            if s.get('name') == '${SCENE_NAME}':
                print('yes')
                break
        else:
            print('no')
    else:
        print('no')
except:
    print('no')
" 2>/dev/null || echo "no")

if [ "${ALREADY_EXISTS}" = "yes" ]; then
    echo "  Scene '${SCENE_NAME}' already exists. Skipping import."
    echo "=== Scene import skipped (already configured) ==="
    exit 0
fi

# Import the scene .zip
echo "Importing scene from ${SCENE_ZIP}..."
IMPORT_RESPONSE=$(curl -s ${CURL_TLS_FLAGS} \
    -X POST "${SCENESCAPE_URL}/api/v1/import-scene/" \
    -H "Authorization: token ${TOKEN}" \
    -F "zipFile=@${SCENE_ZIP}" 2>/dev/null)

echo "  Import response: ${IMPORT_RESPONSE}"

# Verify import
sleep 3
SCENES_AFTER=$(curl -s ${CURL_TLS_FLAGS} \
    -H "Authorization: token ${TOKEN}" \
    "${SCENESCAPE_URL}/api/v1/scenes" 2>/dev/null)

FOUND_AFTER=$(echo "$SCENES_AFTER" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    results = data.get('results', data) if isinstance(data, dict) else data
    if isinstance(results, list):
        for s in results:
            if s.get('name') == '${SCENE_NAME}':
                print('found')
                break
        else:
            print('not_found')
    else:
        print('not_found')
except:
    print('error')
" 2>/dev/null || echo "error")

if [ "${FOUND_AFTER}" = "found" ]; then
    echo "=== Scene '${SCENE_NAME}' imported successfully ==="
else
    echo "WARNING: Scene '${SCENE_NAME}' import may have failed. Check SceneScape UI."
    echo "  You can manually import the scene via: SceneScape UI > Import Scene"
fi
