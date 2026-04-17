# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Lightweight YOLO-Pose inference using OpenVINO Runtime directly.

Replaces ultralytics.YOLO to avoid pulling in PyTorch (~2 GB).
Expects an OpenVINO IR model (.xml/.bin) exported from yolo11n-pose.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

logger = logging.getLogger(__name__)

_INPUT_SIZE = 640  # YOLO default


@dataclass
class _KeypointsResult:
    """Mirrors the subset of ultralytics result API used by PoseAnalyzer."""

    xy: np.ndarray  # Shape: [num_persons, 17, 2]
    conf: np.ndarray  # Shape: [num_persons, 17]


@dataclass
class _PoseResult:
    """Single-image result matching the ultralytics interface we use."""

    keypoints: _KeypointsResult | None


class YOLOPoseOV:
    """
    Drop-in replacement for ``ultralytics.YOLO`` (pose task only).

    Only the ``__call__`` interface used by ``PoseAnalyzer`` is implemented:

        results = model(frame, verbose=False)
        kp_xy   = results[0].keypoints.xy[0].cpu().numpy()
        kp_conf = results[0].keypoints.conf[0].cpu().numpy()
    """

    def __init__(self, model_path: str, device: str = "AUTO"):
        path = Path(model_path)
        if path.suffix != ".xml":
            raise ValueError(f"Expected .xml model path, got: {path}")

        core = ov.Core()
        logger.info("Compiling YOLO-Pose model %s on %s", path, device)
        self._model = core.compile_model(str(path), device)
        self._input_layer = self._model.input(0)
        self._output_layer = self._model.output(0)

    # ------------------------------------------------------------------
    def __call__(
        self, image: np.ndarray, *, verbose: bool = False  # noqa: ARG002
    ) -> list[_PoseResult]:
        """Run inference on a single BGR image."""
        img, ratio, (pad_w, pad_h) = self._preprocess(image)
        output = self._model([img])[self._output_layer]  # (1, 56, 8400)
        results = self._postprocess(output, ratio, pad_w, pad_h)
        return results

    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess(
        image: np.ndarray,
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Letterbox + normalise to (1, 3, 640, 640) float32."""
        h, w = image.shape[:2]
        ratio = min(_INPUT_SIZE / h, _INPUT_SIZE / w)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = (_INPUT_SIZE - new_w) // 2
        pad_h = (_INPUT_SIZE - new_h) // 2
        padded = np.full((_INPUT_SIZE, _INPUT_SIZE, 3), 114, dtype=np.uint8)
        padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # (1,3,640,640)
        return blob, ratio, (pad_w, pad_h)

    # ------------------------------------------------------------------
    def _postprocess(
        self,
        output: np.ndarray,
        ratio: float,
        pad_w: int,
        pad_h: int,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
    ) -> list[_PoseResult]:
        """
        Parse YOLO-Pose output tensor.

        output shape: (1, 56, 8400)
          - 0:4   = bbox (cx, cy, w, h)
          - 4     = objectness score
          - 5:56  = 17 keypoints * 3 (x, y, conf)
        """
        predictions = output[0].T  # (8400, 56)

        scores = predictions[:, 4]
        mask = scores > conf_threshold
        predictions = predictions[mask]
        scores = scores[mask]

        if len(predictions) == 0:
            return [_PoseResult(keypoints=None)]

        # NMS
        boxes_xywh = predictions[:, :4]
        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)
        keep = self._nms(boxes_xyxy, scores, iou_threshold)
        predictions = predictions[keep]

        # Extract keypoints — columns 5..56 → (N, 17, 3)
        raw_kp = predictions[:, 5:].reshape(-1, 17, 3)
        kp_xy = raw_kp[:, :, :2].copy()
        kp_conf = raw_kp[:, :, 2].copy()

        # Undo letterbox transform
        kp_xy[:, :, 0] = (kp_xy[:, :, 0] - pad_w) / ratio
        kp_xy[:, :, 1] = (kp_xy[:, :, 1] - pad_h) / ratio

        return [_PoseResult(keypoints=_KeypointsResult(xy=kp_xy, conf=kp_conf))]

    # ------------------------------------------------------------------
    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        result = np.empty_like(boxes)
        result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return result

    @staticmethod
    def _nms(
        boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
    ) -> list[int]:
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep: list[int] = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        return keep
