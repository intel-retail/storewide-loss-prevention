#!/bin/bash -e
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Install script for Storewide Loss Prevention.
# Generates secrets, downloads sample videos, and prepares the environment.
#
# Usage:
#   ./scripts/install.sh [HOST_IP]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SECRETS_DIR="${PROJECT_DIR}/scenescape/secrets"

echo "=== Storewide Loss Prevention - Install ==="

# ---- Generate secrets (if not already present) ----
if [ ! -f "${SECRETS_DIR}/browser.auth" ]; then
    echo "[1/2] Generating SceneScape secrets..."
    chmod +x "${SECRETS_DIR}/generate_secrets.sh"
    bash "${SECRETS_DIR}/generate_secrets.sh"
else
    echo "[1/2] Secrets already exist, skipping generation."
fi

# ---- Set HOST_IP ----
HOST_IP_ARG="$1"
if [ -z "$HOST_IP_ARG" ]; then
    HOST_IP=$(hostname -I 2>/dev/null | cut -f1 -d' ')
    if [ -z "$HOST_IP" ]; then
        HOST_IP=$(ip route get 1 2>/dev/null | awk '{print $7}')
    fi
    [ -z "$HOST_IP" ] && HOST_IP="localhost"
else
    HOST_IP="$HOST_IP_ARG"
fi
echo "[2/2] Configured HOST_IP: $HOST_IP"

# ---- Fix ownership ----
sudo chown -R "$USER:$USER" "${SECRETS_DIR}" 2>/dev/null || true

echo ""
echo "=== Install complete ==="
echo "  Secrets:  ${SECRETS_DIR}/"
echo "  HOST_IP:  ${HOST_IP}"
echo ""
echo "Next steps:"
echo "  ./scripts/init.sh            # download videos & generate .env"
echo "  make demo            # start full stack"
