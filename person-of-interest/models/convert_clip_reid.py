#!/usr/bin/env python3
"""Convert CLIP-ReID Market-1501 PyTorch checkpoint → ONNX → OpenVINO IR.

Usage:
    python models/convert_clip_reid.py \
        --weights models/intel/clip-reid-market1501/ViT-B-16_market1501.pth \
        --output-dir models/intel/clip-reid-market1501

Produces:
    FP32/clip-reid-market1501.xml  + .bin
    FP16/clip-reid-market1501.xml  + .bin
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

REPO_URL = "https://github.com/Syliz517/CLIP-ReID.git"

# ── Market-1501 ViT-CLIP-ReID config ────────────────────────────
NUM_CLASSES = 751      # Market-1501 training identities
CAMERA_NUM = 6         # Market-1501 cameras
VIEW_NUM = 1
INPUT_H, INPUT_W = 256, 128
STRIDE = 16
PATCH_SIZE = 16
H_RES = (INPUT_H - PATCH_SIZE) // STRIDE + 1   # 16
W_RES = (INPUT_W - PATCH_SIZE) // STRIDE + 1    # 8


class InferenceWrapper(nn.Module):
    """Thin wrapper that runs only the visual encoder + BN,
    returning the 1280-d ReID embedding (768 + 512)."""

    def __init__(self, full_model):
        super().__init__()
        self.image_encoder = full_model.image_encoder
        self.bottleneck = full_model.bottleneck
        self.bottleneck_proj = full_model.bottleneck_proj

    def forward(self, x):
        # image_encoder returns (x11, x12, xproj) for ViT
        _, image_features, image_features_proj = self.image_encoder(x, None)
        img_feature = image_features[:, 0]        # [B, 768]
        img_feature_proj = image_features_proj[:, 0]  # [B, 512]
        feat = self.bottleneck(img_feature)           # BN → [B, 768]
        feat_proj = self.bottleneck_proj(img_feature_proj)  # BN → [B, 512]
        return torch.cat([feat, feat_proj], dim=1)    # [B, 1280]


def clone_repo(dest: str) -> str:
    """Clone CLIP-ReID repo into a temp directory and patch for CPU."""
    print(f"[1/5] Cloning CLIP-ReID repo …")
    subprocess.check_call(
        ["git", "clone", "--depth", "1", REPO_URL, dest],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Patch: remove .to("cuda") so model builds on CPU
    model_file = os.path.join(dest, "model", "make_model_clipreid.py")
    with open(model_file, "r") as f:
        src = f.read()
    src = src.replace('clip_model.to("cuda")', 'clip_model.to("cpu")')
    src = src.replace("tokenized_prompts = clip.tokenize(ctx_init).cuda()",
                       "tokenized_prompts = clip.tokenize(ctx_init)")
    with open(model_file, "w") as f:
        f.write(src)
    return dest


def build_model(repo_dir: str, weights_path: str) -> nn.Module:
    """Build the full CLIP-ReID model and load weights."""
    print("[2/5] Building model and loading weights …")
    sys.path.insert(0, repo_dir)

    # Minimal cfg object matching configs/person/vit_clipreid.yml
    from yacs.config import CfgNode as CN

    cfg = CN()
    cfg.MODEL = CN()
    cfg.MODEL.NAME = "ViT-B-16"
    cfg.MODEL.COS_LAYER = False
    cfg.MODEL.NECK = "bnneck"
    cfg.MODEL.STRIDE_SIZE = [STRIDE, STRIDE]
    cfg.MODEL.SIE_CAMERA = False
    cfg.MODEL.SIE_VIEW = False
    cfg.MODEL.SIE_COE = 1.0
    cfg.MODEL.PRETRAIN_CHOICE = "imagenet"

    cfg.INPUT = CN()
    cfg.INPUT.SIZE_TRAIN = [INPUT_H, INPUT_W]

    cfg.TEST = CN()
    cfg.TEST.NECK_FEAT = "before"

    cfg.DATASETS = CN()
    cfg.DATASETS.NAMES = "market1501"

    from model.make_model_clipreid import make_model  # noqa: E402

    model = make_model(cfg, NUM_CLASSES, CAMERA_NUM, VIEW_NUM)
    state = torch.load(weights_path, map_location="cpu")
    # Handle 'module.' prefix from DataParallel training
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model


def export_onnx(wrapper: nn.Module, onnx_path: str):
    """Export the inference wrapper to ONNX."""
    print("[3/5] Exporting to ONNX …")
    dummy = torch.randn(1, 3, INPUT_H, INPUT_W)
    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["reid_embedding"],
        dynamic_axes={"input": {0: "batch"}, "reid_embedding": {0: "batch"}},
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"     ONNX saved → {onnx_path}")


def convert_to_openvino(onnx_path: str, output_dir: str):
    """Convert ONNX → OpenVINO IR (FP32 and FP16) using openvino.convert_model."""
    print("[4/5] Converting to OpenVINO IR …")
    import openvino as ov

    model_name = "clip-reid-market1501"
    core = ov.Core()

    for precision in ("FP32", "FP16"):
        out = os.path.join(output_dir, precision)
        os.makedirs(out, exist_ok=True)
        ov_model = ov.convert_model(onnx_path)
        if precision == "FP16":
            ov_model = ov.serialize(
                ov_model,
                os.path.join(out, f"{model_name}.xml"),
            )
            # Re-convert with compress_to_fp16
            import openvino.runtime.passes as passes
            ov_model = ov.convert_model(onnx_path)
            ov.save_model(ov_model, os.path.join(out, f"{model_name}.xml"),
                          compress_to_fp16=True)
        else:
            ov.save_model(ov_model, os.path.join(out, f"{model_name}.xml"),
                          compress_to_fp16=False)
        print(f"     {precision} → {out}/{model_name}.xml")


def main():
    parser = argparse.ArgumentParser(description="Convert CLIP-ReID to OpenVINO IR")
    parser.add_argument(
        "--weights",
        default="models/intel/clip-reid-market1501/ViT-B-16_market1501.pth",
        help="Path to PyTorch checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        default="models/intel/clip-reid-market1501",
        help="Directory for OpenVINO IR output",
    )
    args = parser.parse_args()

    weights = os.path.abspath(args.weights)
    output_dir = os.path.abspath(args.output_dir)
    onnx_path = os.path.join(output_dir, "clip-reid-market1501.onnx")

    if not os.path.isfile(weights):
        sys.exit(f"Weights not found: {weights}")

    tmp_dir = tempfile.mkdtemp(prefix="clip_reid_")
    repo_dir = os.path.join(tmp_dir, "CLIP-ReID")

    try:
        clone_repo(repo_dir)
        full_model = build_model(repo_dir, weights)
        wrapper = InferenceWrapper(full_model)
        wrapper.eval()

        with torch.no_grad():
            export_onnx(wrapper, onnx_path)

        convert_to_openvino(onnx_path, output_dir)

        # Clean up ONNX intermediate
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

        print("[5/5] Done! OpenVINO IR files are in:")
        for precision in ("FP32", "FP16"):
            xml = os.path.join(output_dir, precision, "clip-reid-market1501.xml")
            if os.path.exists(xml):
                size_mb = os.path.getsize(xml.replace(".xml", ".bin")) / 1e6
                print(f"     {xml}  ({size_mb:.1f} MB)")

    finally:
        # Always clean up the temp clone
        shutil.rmtree(tmp_dir, ignore_errors=True)
        # Remove repo from sys.path
        if repo_dir in sys.path:
            sys.path.remove(repo_dir)


if __name__ == "__main__":
    main()
