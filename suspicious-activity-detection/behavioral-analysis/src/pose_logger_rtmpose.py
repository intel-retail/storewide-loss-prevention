# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
"""
GStreamer gvapython bridge to PoseAnalyzer using RTMPose OpenVINO model.

RTMPose uses SimCC (Simple Coordinate Classification) output:
  simcc_x : [1, 17, W_ext]  where W_ext = input_W * simcc_split_ratio (default 2.0)
  simcc_y : [1, 17, H_ext]  where H_ext = input_H * simcc_split_ratio (default 2.0)

Common RTMPose input sizes:
  RTMPose-t / s / m / l : 256 x 192  (H x W)
  -> simcc_x : [1, 17, 384]
  -> simcc_y : [1, 17, 512]

Pipeline (person detector + RTMPose):
  gst-launch-1.0 filesrc location=video.mp4 ! decodebin ! videoconvert !
    video/x-raw,format=BGRx !
    gvadetect model=person_detector.xml device=CPU threshold=0.4 !
    gvainference model=rtmpose.xml device=CPU object-class=person !
    gvapython module=pose_logger_rtmpose.py function=process_frame !
    fakesink

Or single-stage (no detector, full-frame inference):
  gst-launch-1.0 filesrc location=video.mp4 ! decodebin ! videoconvert !
    video/x-raw,format=BGRx !
    gvainference model=rtmpose.xml device=CPU !
    gvapython module=pose_logger_rtmpose.py function=process_frame !
    fakesink
"""

import sys
import os
import json
import logging
import inspect
import numpy as np
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(inspect.getfile(lambda: None)))
sys.path.insert(0, _THIS_DIR)

from pose_analyzer import PoseAnalyzer, Pose
from config import Settings, load_pattern_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output files (JSONL - one JSON object per line for fast O(1) appends)
# Support env-var override for subprocess integration
FRAMES_OUTPUT_PATH = os.environ.get(
    "POSE_FRAMES_OUTPUT", os.path.join(_THIS_DIR, "pose_frames.jsonl")
)
ALERTS_OUTPUT_PATH = os.environ.get(
    "POSE_ALERTS_OUTPUT", os.path.join(_THIS_DIR, "pose_alerts.jsonl")
)

# RTMPose model configuration
_settings         = Settings()
_NUM_KP           = 17          # COCO-17 keypoints
_CONF_THRESH      = _settings.pose_confidence_threshold
_SIMCC_SPLIT      = 2.0         # SimCC split ratio (matches export config)
_INPUT_W          = 192         # model input width  (change if using larger variant)
_INPUT_H          = 256         # model input height (change if using larger variant)
_SIMCC_W          = int(_INPUT_W * _SIMCC_SPLIT)   # 384
_SIMCC_H          = int(_INPUT_H * _SIMCC_SPLIT)   # 512

_KP_NAMES = [
    "nose",
    "left_eye",      "right_eye",
    "left_ear",      "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow",    "right_elbow",
    "left_wrist",    "right_wrist",
    "left_hip",      "right_hip",
    "left_knee",     "right_knee",
    "left_ankle",    "right_ankle",
]


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _init_output_files():
    for path in (FRAMES_OUTPUT_PATH, ALERTS_OUTPUT_PATH):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            pass
    logger.info("Output files initialised -> %s | %s",
                FRAMES_OUTPUT_PATH, ALERTS_OUTPUT_PATH)


def _append_line(path, entry):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error("Failed to write to %s: %s", path, e)


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# Load pattern config from patterns.yaml (same file used by main.py)
_pattern_config = load_pattern_config(_settings.pattern_config_path)

_analyzer = PoseAnalyzer(
    min_frames=3,
    confidence_threshold=_CONF_THRESH,
    pattern_config=_pattern_config,
)

_pose_window      = []
_WINDOW_SIZE      = 30
_frame_counter    = 0
_first_frame_done = False

# Stores tensor sizes seen on first frame so we can auto-detect layout
_detected_layout  = None   # set once on first successful decode

_init_output_files()


# ---------------------------------------------------------------------------
# SimCC decoder (primary RTMPose output format)
# ---------------------------------------------------------------------------

def _decode_simcc(simcc_x, simcc_y, img_w, img_h, region_rect=None):
    """
    Decode RTMPose SimCC outputs into COCO-17 keypoints.

    simcc_x : numpy array [K, W_ext]  or [1, K, W_ext]
    simcc_y : numpy array [K, H_ext]  or [1, K, H_ext]

    Returns (xy_norm [K, 2], conf [K]) normalised to [0, 1] in frame coords,
    or None if mean confidence is below threshold.
    """
    # Squeeze batch dimension if present
    if simcc_x.ndim == 3:
        simcc_x = simcc_x[0]   # [K, W_ext]
    if simcc_y.ndim == 3:
        simcc_y = simcc_y[0]   # [K, H_ext]

    K = simcc_x.shape[0]

    # Softmax for numerical stability (optional but improves confidence scores)
    def _softmax(arr):
        e = np.exp(arr - arr.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    sx = _softmax(simcc_x)   # [K, W_ext]
    sy = _softmax(simcc_y)   # [K, H_ext]

    x_idx  = np.argmax(sx, axis=-1).astype(np.float32)   # [K]
    y_idx  = np.argmax(sy, axis=-1).astype(np.float32)   # [K]
    x_conf = sx[np.arange(K), np.argmax(sx, axis=-1)]    # [K]
    y_conf = sy[np.arange(K), np.argmax(sy, axis=-1)]    # [K]

    # Per-keypoint confidence = geometric mean of x and y peak probabilities
    conf = np.sqrt(x_conf * y_conf)

    # Convert SimCC index -> pixel coordinate in model input space
    x_px = x_idx / _SIMCC_SPLIT   # [0, INPUT_W]
    y_px = y_idx / _SIMCC_SPLIT   # [0, INPUT_H]

    # If inference was on a cropped region, map back to full-frame coordinates
    if region_rect is not None:
        rx, ry, rw, rh = (region_rect.x, region_rect.y,
                          region_rect.w, region_rect.h)
        rw = max(rw, 1)
        rh = max(rh, 1)
        x_px = rx + x_px * (rw / _INPUT_W)
        y_px = ry + y_px * (rh / _INPUT_H)

    # Normalise to [0, 1] using full frame dimensions
    xy_norm = np.stack([x_px / max(img_w, 1),
                        y_px / max(img_h, 1)], axis=-1)   # [K, 2]

    mean_conf = float(conf.mean())
    if mean_conf < _CONF_THRESH:
        logger.info("SimCC mean confidence %.3f below threshold %.3f — pose dropped",
                     mean_conf, _CONF_THRESH)
        return None

    return xy_norm, conf


# ---------------------------------------------------------------------------
# Fallback decoder: single keypoint tensor [K, 3] or [K, 2]
# ---------------------------------------------------------------------------

def _decode_keypoint_tensor(data, img_w, img_h, region_rect=None):
    """
    Decode a flat keypoint tensor where each keypoint is stored as
    (x, y, score) or (x, y).  Coordinates may be in pixel space or [0,1].

    Supports:
      Layout A: [K, 3]  - x, y, score per keypoint (51 floats for K=17)
      Layout B: [K, 2]  - x, y only                (34 floats for K=17)
    """
    arr = np.array(data, dtype=np.float32)
    n   = arr.size

    if n == _NUM_KP * 3:
        kp   = arr.reshape(_NUM_KP, 3)
        xy   = kp[:, :2]
        conf = kp[:, 2]
    elif n == _NUM_KP * 2:
        kp   = arr.reshape(_NUM_KP, 2)
        xy   = kp
        conf = np.ones(_NUM_KP, dtype=np.float32)
    else:
        return None

    # Map region-relative coordinates to full frame if needed
    if region_rect is not None:
        rx, ry, rw, rh = (region_rect.x, region_rect.y,
                          max(region_rect.w, 1), max(region_rect.h, 1))
        # Detect if coordinates are already normalised [0,1] or pixel
        if xy.max() <= 1.0:
            xy[:, 0] = rx + xy[:, 0] * rw
            xy[:, 1] = ry + xy[:, 1] * rh

    # Normalise to [0, 1]
    xy_norm = xy.copy()
    if xy_norm.max() > 1.0:
        xy_norm[:, 0] /= max(img_w, 1)
        xy_norm[:, 1] /= max(img_h, 1)

    if float(conf.mean()) < _CONF_THRESH:
        return None

    return xy_norm, conf


# ---------------------------------------------------------------------------
# Tensor layout auto-detection
# ---------------------------------------------------------------------------

def _try_decode_all(tensors, img_w, img_h, region_rect=None):
    """
    Try all known RTMPose tensor layouts.

    Layout 1 (SimCC dual output):
      tensor[0] -> simcc_x [1, 17, 384]   (size = 6528)
      tensor[1] -> simcc_y [1, 17, 512]   (size = 8704)

    Layout 2 (SimCC combined flat):
      tensor[0] -> simcc_x + simcc_y concatenated flat (size = 15232)

    Layout 3 (Direct keypoints):
      tensor[0] -> [17, 3] or [17, 2]      (size = 51 or 34)

    Layout 4 (Transposed SimCC):
      tensor[0] -> simcc_x [1, 384, 17]   (size = 6528)
      tensor[1] -> simcc_y [1, 512, 17]   (size = 8704)
    """
    global _detected_layout

    simcc_x_size = _NUM_KP * _SIMCC_W   # 17 * 384 = 6528
    simcc_y_size = _NUM_KP * _SIMCC_H   # 17 * 512 = 8704

    data_list = []
    for t in tensors:
        d = t.data()
        if d is not None and len(d) > 0:
            data_list.append(np.array(d, dtype=np.float32))

    if not data_list:
        return None

    # Log layout once
    if _detected_layout is None:
        sizes = [a.size for a in data_list]
        logger.info("First decode attempt | tensor sizes=%s | "
                    "expected simcc_x=%d simcc_y=%d",
                    sizes, simcc_x_size, simcc_y_size)

    # --- Layout 1: two separate SimCC tensors ---
    if (len(data_list) >= 2 and
            data_list[0].size == simcc_x_size and
            data_list[1].size == simcc_y_size):
        sx = data_list[0].reshape(_NUM_KP, _SIMCC_W)
        sy = data_list[1].reshape(_NUM_KP, _SIMCC_H)
        result = _decode_simcc(sx, sy, img_w, img_h, region_rect)
        if result:
            if _detected_layout is None:
                logger.info("Using Layout 1: dual SimCC tensors")
                _detected_layout = 1
            return result

    # --- Layout 4: transposed SimCC [W_ext, K] and [H_ext, K] ---
    if (len(data_list) >= 2 and
            data_list[0].size == simcc_x_size and
            data_list[1].size == simcc_y_size):
        sx = data_list[0].reshape(_SIMCC_W, _NUM_KP).T   # [K, W_ext]
        sy = data_list[1].reshape(_SIMCC_H, _NUM_KP).T   # [K, H_ext]
        result = _decode_simcc(sx, sy, img_w, img_h, region_rect)
        if result:
            if _detected_layout is None:
                logger.info("Using Layout 4: transposed SimCC tensors")
                _detected_layout = 4
            return result

    # --- Layout 2: single concatenated SimCC tensor ---
    if (len(data_list) >= 1 and
            data_list[0].size == simcc_x_size + simcc_y_size):
        flat = data_list[0]
        sx   = flat[:simcc_x_size].reshape(_NUM_KP, _SIMCC_W)
        sy   = flat[simcc_x_size:].reshape(_NUM_KP, _SIMCC_H)
        result = _decode_simcc(sx, sy, img_w, img_h, region_rect)
        if result:
            if _detected_layout is None:
                logger.info("Using Layout 2: concatenated SimCC tensor")
                _detected_layout = 2
            return result

    # --- Layout 3: direct keypoint tensor [K, 3] or [K, 2] ---
    for arr in data_list:
        result = _decode_keypoint_tensor(arr, img_w, img_h, region_rect)
        if result:
            if _detected_layout is None:
                logger.info("Using Layout 3: direct keypoint tensor size=%d",
                            arr.size)
                _detected_layout = 3
            return result

    # Log unrecognised sizes once
    if _detected_layout is None:
        logger.warning("Unrecognised tensor sizes=%s | "
                       "check _INPUT_W=%d _INPUT_H=%d _SIMCC_SPLIT=%.1f",
                       [a.size for a in data_list],
                       _INPUT_W, _INPUT_H, _SIMCC_SPLIT)

    return None


# ---------------------------------------------------------------------------
# GVA frame extraction
# ---------------------------------------------------------------------------

def _extract_pose_from_gva(video_frame, timestamp):
    global _first_frame_done

    try:
        info  = video_frame.video_info()
        img_w = info.width
        img_h = info.height
    except Exception:
        img_w, img_h = _INPUT_W, _INPUT_H

    # Detailed debug on first frame only
    if not _first_frame_done:
        _first_frame_done = True
        try:
            frame_tensors = list(video_frame.tensors())
            regions       = list(video_frame.regions())
            logger.info("Frame 1 | frame_tensors=%d | regions=%d | img=%dx%d",
                        len(frame_tensors), len(regions), img_w, img_h)
            for i, t in enumerate(frame_tensors):
                d = t.data()
                logger.info("  frame_tensor[%d] size=%d name=%s",
                            i,
                            len(d) if d is not None else 0,
                            t.name() if hasattr(t, "name") else "N/A")
            for r_idx, roi in enumerate(regions[:3]):   # log first 3 regions
                rect = roi.rect()
                roi_tensors = list(roi.tensors())
                logger.info("  region[%d] label=%s conf=%.2f "
                            "rect=(%d,%d,%d,%d) tensors=%d",
                            r_idx, roi.label(), roi.confidence(),
                            rect.x, rect.y, rect.w, rect.h,
                            len(roi_tensors))
                for t_idx, t in enumerate(roi_tensors):
                    d = t.data()
                    logger.info("    tensor[%d] size=%d name=%s",
                                t_idx,
                                len(d) if d is not None else 0,
                                t.name() if hasattr(t, "name") else "N/A")
        except Exception as e:
            logger.warning("Debug inspection error: %s", e)

    # --- Primary: region-level tensors (gvainference with object-class=person) ---
    try:
        regions = list(video_frame.regions())
        if not regions:
            logger.debug("Frame %d: gvadetect found 0 regions", timestamp)
        for region in regions:
            rect    = region.rect()
            tensors = list(region.tensors())
            if not tensors:
                logger.debug("Frame %d: region (%d,%d,%d,%d) has 0 tensors",
                             timestamp, rect.x, rect.y, rect.w, rect.h)
                continue
            result = _try_decode_all(tensors, img_w, img_h,
                                     region_rect=rect)
            if result is not None:
                return Pose(keypoints=result[0],
                            confidences=result[1],
                            timestamp=timestamp)
    except Exception as e:
        logger.warning("Region tensor error: %s", e)

    # --- Secondary: frame-level tensors (gvainference, full-frame mode) ---
    try:
        tensors = list(video_frame.tensors())
        if tensors:
            result = _try_decode_all(tensors, img_w, img_h,
                                     region_rect=None)
            if result is not None:
                return Pose(keypoints=result[0],
                            confidences=result[1],
                            timestamp=timestamp)
    except Exception as e:
        logger.warning("Frame tensor error: %s", e)

    return None


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _pose_to_dict(pose):
    keypoints = {}
    for idx, name in enumerate(_KP_NAMES):
        keypoints[name] = {
            "x":          round(float(pose.keypoints[idx][0]), 6),
            "y":          round(float(pose.keypoints[idx][1]), 6),
            "confidence": round(float(pose.confidences[idx]),  4),
        }
    return {"timestamp": pose.timestamp, "keypoints": keypoints}


# ---------------------------------------------------------------------------
# gvapython entry point
# ---------------------------------------------------------------------------

def process_frame(frame):
    global _frame_counter, _pose_window

    _frame_counter += 1

    pose = _extract_pose_from_gva(frame, _frame_counter)

    if pose is not None:
        _append_line(FRAMES_OUTPUT_PATH, {
            "frame":       _frame_counter,
            "captured_at": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "pose":        _pose_to_dict(pose),
        })
        _pose_window.append(pose)
        if len(_pose_window) > _WINDOW_SIZE:
            _pose_window.pop(0)
        logger.info("Frame %d | pose captured | window=%d",
                    _frame_counter, len(_pose_window))
    else:
        # Log diagnostic info for every frame that fails pose extraction
        try:
            regions = list(frame.regions())
            tensors = list(frame.tensors())
            logger.warning(
                "Frame %d | NO POSE | regions=%d frame_tensors=%d",
                _frame_counter, len(regions), len(tensors),
            )
        except Exception:
            logger.warning("Frame %d | NO POSE (inspection failed)", _frame_counter)

    if len(_pose_window) >= _analyzer.min_frames:
        results = _analyzer.detect_all_patterns(_pose_window)
        for result in results:
            if result.matched:
                alert = {
                    "alert_id":    "alert_" + str(_frame_counter),
                    "frame":       _frame_counter,
                    "detected_at": datetime.utcnow().isoformat(
                                       timespec="milliseconds") + "Z",
                    "pattern":     result.pattern_id,
                    "confidence":  round(result.confidence, 4),
                    "description": result.description,
                    "window_size": len(_pose_window),
                }
                _append_line(ALERTS_OUTPUT_PATH, alert)
                logger.warning(
                    "ALERT | pattern=%s | confidence=%.2f | %s | frame=%d",
                    result.pattern_id, result.confidence,
                    result.description, _frame_counter,
                )
                _pose_window.clear()
                break

    return True
