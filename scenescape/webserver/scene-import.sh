#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Import SceneScape scene .zip file(s) via the REST API.
# Runs as a sidecar container after the web service is healthy.
#
# When STREAM_DENSITY > 1, clones the base scene zip on-the-fly with unique
# scene names and camera IDs, uploads each clone, then cleans up.
#
# Uses Python's built-in urllib for HTTP calls (no curl/apt-get needed,
# avoids proxy issues inside Docker containers).

set -e

SCENE_ZIP_NAME="${SCENE_ZIP:-}"
STREAM_DENSITY="${STREAM_DENSITY:-1}"
SCENE_NAME="${SCENE_NAME:-}"
CAMERA_NAME="${CAMERA_NAME:-}"
# Number of base cameras already present before cloning starts.
# Cloned camera indices begin at BASE_CAMERA_COUNT+1 so they never collide
# with an existing base camera (e.g. Camera_02).
# POI sets this to 2 (Camera_01 + Camera_02); other apps default to 1.
BASE_CAMERA_COUNT="${BASE_CAMERA_COUNT:-1}"
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
    if [ ! -f "/webserver/stream_density.py" ]; then
        echo "ERROR: stream_density.py not found — cannot clone zips for density ${STREAM_DENSITY}"
        exit 1
    fi
    BASE_ZIP="/webserver/${SCENE_ZIP_NAME}"
    if [ ! -f "${BASE_ZIP}" ]; then
        echo "ERROR: Base scene zip not found: ${BASE_ZIP}"
        exit 1
    fi
    CLONE_DIR=$(mktemp -d)
    echo "  Cloning base zip ${STREAM_DENSITY} times..."
    python3 /webserver/stream_density.py clone-zip \
        "${BASE_ZIP}" "${CLONE_DIR}" "${SCENE_NAME}" "${CAMERA_NAME}" "${STREAM_DENSITY}" "${BASE_CAMERA_COUNT}" > /dev/null
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

# Wait for SceneScape web to be healthy
echo "Waiting for SceneScape web service..."
for i in $(seq 1 ${MAX_RETRIES}); do
    HEALTH=$(python3 -c "
import urllib.request, ssl, os
ctx = ssl.create_default_context()
ca = os.environ.get('CA_CERT', '')
if ca and os.path.isfile(ca):
    ctx.load_verify_locations(ca)
else:
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
try:
    url = os.environ.get('SCENESCAPE_URL', '').rstrip('/') + '/api/v1/database-ready'
    r = urllib.request.urlopen(url, context=ctx, timeout=5)
    print(r.read().decode())
except Exception:
    print('')
")
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
TOKEN=$(python3 -c "
import urllib.request, ssl, json, os, sys
ctx = ssl.create_default_context()
ca = os.environ.get('CA_CERT', '')
if ca and os.path.isfile(ca):
    ctx.load_verify_locations(ca)
else:
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
base_url = os.environ.get('SCENESCAPE_URL', '').rstrip('/')
data = json.dumps({
    'username': os.environ.get('SCENESCAPE_USER', ''),
    'password': os.environ.get('SCENESCAPE_PASSWORD', os.environ.get('SUPASS', '')),
}).encode()
req = urllib.request.Request(f'{base_url}/api/v1/auth', data=data,
                            headers={'Content-Type': 'application/json'}, method='POST')
try:
    r = urllib.request.urlopen(req, context=ctx, timeout=30)
    print(json.loads(r.read().decode()).get('token', ''))
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    print('')
")

if [ -z "${TOKEN}" ]; then
    echo "ERROR: Failed to authenticate with SceneScape."
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
    IMPORT_RESPONSE=$(SCENE_ZIP_PATH="${SCENE_ZIP}" AUTH_TOKEN="${TOKEN}" python3 -c "
import urllib.request, urllib.error, ssl, os, uuid, sys
ctx = ssl.create_default_context()
ca = os.environ.get('CA_CERT', '')
if ca and os.path.isfile(ca):
    ctx.load_verify_locations(ca)
else:
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
boundary = uuid.uuid4().hex
zip_path = os.environ['SCENE_ZIP_PATH']
filename = os.path.basename(zip_path)
base_url = os.environ.get('SCENESCAPE_URL', '').rstrip('/')
token = os.environ.get('AUTH_TOKEN', '')
with open(zip_path, 'rb') as f:
    file_data = f.read()
body = (
    b'--' + boundary.encode() + b'\r\n'
    b'Content-Disposition: form-data; name=\"zipFile\"; filename=\"' + filename.encode() + b'\"\r\n'
    b'Content-Type: application/zip\r\n\r\n'
    + file_data + b'\r\n'
    b'--' + boundary.encode() + b'--\r\n'
)
req = urllib.request.Request(
    f'{base_url}/api/v1/import-scene/',
    data=body,
    headers={
        'Authorization': f'token {token}',
        'Content-Type': f'multipart/form-data; boundary={boundary}',
    },
    method='POST',
)
try:
    r = urllib.request.urlopen(req, context=ctx, timeout=120)
    print(r.read().decode())
except urllib.error.HTTPError as e:
    print(f'HTTP {e.code}: {e.read().decode()}')
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    print(f'ERROR: {e}')
")

    echo "  Import response: ${IMPORT_RESPONSE}"
    if echo "${IMPORT_RESPONSE}" | grep -qE '^(HTTP [45][0-9]{2}:|ERROR:)'; then
        IMPORT_FAIL=$((IMPORT_FAIL + 1))
    else
        IMPORT_SUCCESS=$((IMPORT_SUCCESS + 1))
    fi
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
