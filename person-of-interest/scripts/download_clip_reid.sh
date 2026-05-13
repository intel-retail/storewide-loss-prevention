#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Download CLIP-ReID (ViT-B-16) Market-1501 pretrained model from
# https://github.com/Syliz517/CLIP-ReID, export to ONNX, convert to
# OpenVINO IR, and install into the storewide-lp_vol-models Docker volume.
#
# Usage:
#   bash scripts/download_clip_reid.sh
#
# Requirements (installed automatically inside a temporary container):
#   Python 3.10+, PyTorch, OpenVINO dev tools, gdown

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
VOLUME_NAME="storewide-lp_vol-models"
MODEL_NAME="clip-reid-market1501"
PRECISION="FP32"
DEST="/models/intel/${MODEL_NAME}/${PRECISION}"

# Google Drive file ID for ViT-CLIP-ReID Market-1501 checkpoint
# Source: https://github.com/Syliz517/CLIP-ReID README → Trained models → ViT-CLIP-ReID → Market column
GDRIVE_FILE_ID="1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7"

echo "=== CLIP-ReID Model Download & Export ==="
echo "  Model:     ${MODEL_NAME}"
echo "  Precision: ${PRECISION}"
echo "  Volume:    ${VOLUME_NAME}"
echo ""

# Ensure volume exists
docker volume create "${VOLUME_NAME}" 2>/dev/null || true

# Check if model already exists
EXISTING=$(docker run --rm -v "${VOLUME_NAME}":/models alpine:3.23 \
    sh -c "[ -f ${DEST}/${MODEL_NAME}.xml ] && [ -f ${DEST}/${MODEL_NAME}.bin ] && echo yes || echo no")

if [ "${EXISTING}" = "yes" ]; then
    echo "  Model already exists at ${DEST}. Skipping download."
    echo "  To re-download, remove the model first:"
    echo "    docker run --rm -v ${VOLUME_NAME}:/models alpine:3.23 rm -rf ${DEST}"
    exit 0
fi

echo "[1/3] Downloading pretrained checkpoint from Google Drive..."
echo "[2/3] Exporting to ONNX..."
echo "[3/3] Converting to OpenVINO IR (${PRECISION})..."
echo ""
echo "  This runs inside a temporary Docker container and may take a few minutes."
echo ""

# Run everything in a single Python container with PyTorch + OpenVINO
docker run --rm \
    -v "${VOLUME_NAME}":/models \
    -e http_proxy="${http_proxy:-}" \
    -e https_proxy="${https_proxy:-}" \
    -e HTTP_PROXY="${HTTP_PROXY:-}" \
    -e HTTPS_PROXY="${HTTPS_PROXY:-}" \
    -e no_proxy="${no_proxy:-}" \
    -e NO_PROXY="${NO_PROXY:-}" \
    python:3.12-slim bash -c "
set -e

echo '--- Installing dependencies ---'
pip install --no-cache-dir -q gdown torch torchvision openvino-dev onnx onnxscript yacs timm ftfy regex 2>&1 | tail -1

WORK=/tmp/clip-reid
mkdir -p \${WORK} && cd \${WORK}

echo '--- Cloning CLIP-ReID repository ---'
apt-get update -qq && apt-get install -y -qq git >/dev/null 2>&1
git clone --depth 1 https://github.com/Syliz517/CLIP-ReID.git repo 2>&1 | tail -1
cd repo

echo '--- Downloading pretrained checkpoint ---'
gdown '${GDRIVE_FILE_ID}' -O checkpoint.pth 2>&1 | tail -3

echo '--- Patching for CPU-only export ---'
python3 -c \"
import glob
for f in glob.glob('model/*.py') + glob.glob('model/**/*.py', recursive=True):
    txt = open(f).read()
    orig = txt
    txt = txt.replace('.to(\\\"cuda\\\")', '.to(\\\"cpu\\\")')
    txt = txt.replace('.cuda()', '')
    txt = txt.replace('\\\"cuda\\\"', '\\\"cpu\\\"')
    if txt != orig:
        open(f, 'w').write(txt)
        print(f'  Patched: {f}')
\"

echo '--- Exporting to ONNX ---'
python3 -c \"
import sys, os
sys.path.insert(0, os.getcwd())
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import torch.nn as nn

from config import cfg
cfg.defrost()
cfg.MODEL.NAME = 'ViT-B-16'
cfg.MODEL.PRETRAIN_CHOICE = 'self'
cfg.INPUT.SIZE_TRAIN = [256, 128]
cfg.INPUT.SIZE_TEST = [256, 128]
cfg.MODEL.STRIDE_SIZE = [16, 16]
cfg.MODEL.SIE_CAMERA = True
cfg.MODEL.SIE_VIEW = False
cfg.MODEL.SIE_COE = 3.0
cfg.MODEL.COS_LAYER = False
cfg.MODEL.NECK = 'bnneck'
cfg.MODEL.ID_LOSS_TYPE = 'softmax'
cfg.MODEL.METRIC_LOSS_TYPE = 'triplet'
cfg.TEST.NECK_FEAT = 'before'
cfg.DATASETS.NAMES = 'market1501'
cfg.freeze()

from model.make_model_clipreid import make_model
model = make_model(cfg, num_class=751, camera_num=6, view_num=1)
model.eval()
model.cpu()

state_dict = torch.load('checkpoint.pth', map_location='cpu')
new_sd = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_sd, strict=False)
print('  Checkpoint loaded successfully')

dummy_input = torch.randn(1, 3, 256, 128)

class ReIDFeatureExtractor(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.model = m
    def forward(self, x):
        return self.model(x)

wrapper = ReIDFeatureExtractor(model)
wrapper.eval()

torch.onnx.export(
    wrapper, dummy_input, 'clip_reid_market1501.onnx',
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=14, do_constant_folding=True,
)
print('  ONNX export complete: clip_reid_market1501.onnx')
\"

echo '--- Converting to OpenVINO IR ---'
mkdir -p ${DEST}
ovc clip_reid_market1501.onnx \
    --output_model ${DEST}/${MODEL_NAME}.xml \
    --compress_to_fp16=False 2>&1 | tail -3

echo ''
echo '--- Verifying output ---'
ls -la ${DEST}/
echo ''
echo '=== CLIP-ReID model export complete ==='
echo '  ${DEST}/${MODEL_NAME}.xml'
echo '  ${DEST}/${MODEL_NAME}.bin'
"

echo ""
echo "=== Done ==="
echo "Model installed to Docker volume '${VOLUME_NAME}' at:"
echo "  intel/${MODEL_NAME}/${PRECISION}/${MODEL_NAME}.xml"
echo "  intel/${MODEL_NAME}/${PRECISION}/${MODEL_NAME}.bin"
