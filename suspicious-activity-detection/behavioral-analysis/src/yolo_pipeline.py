# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
YOLO-Pose pipeline runner for pose extraction + pattern detection.

Pure-Python replacement for the GStreamer DL Streamer pipeline.
Uses YOLOv11n-pose via OpenVINO for single-stage person detection + keypoint
estimation, then feeds poses to PoseAnalyzer for pattern matching.
"""

import logging

import numpy as np

from config import Settings, load_pattern_config
from pose_analyzer import PatternResult, Pose, PoseAnalyzer
from yolo_pose_ov import YOLOPoseOV

logger = logging.getLogger(__name__)

# Lazy-initialized singleton — compiled once, reused across calls.
_yolo_model: YOLOPoseOV | None = None


def _get_model(settings: Settings) -> YOLOPoseOV:
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLOPoseOV(
            model_path=settings.yolo_pose_model,
            device=settings.gst_inference_device,
        )
    return _yolo_model


async def run_yolo_pipeline(
    frames: list[tuple[np.ndarray, int]],
    entity_id: str,
    settings: Settings,
) -> PatternResult | None:
    """
    Run YOLO-Pose pipeline for pose extraction + pattern detection.

    Drop-in replacement for ``run_gst_pipeline`` — same signature and return type.

    Args:
        frames: List of (frame_image_bgr, timestamp_ms) tuples.
        entity_id: Person identifier (for logging).
        settings: Settings object with model paths and thresholds.

    Returns:
        PatternResult if analysis completed (matched or not).
        None if pipeline failed critically.
    """
    model = _get_model(settings)
    conf_threshold = settings.pose_confidence_threshold

    poses: list[Pose] = []
    frame_images = [f[0] for f in frames]
    frame_timestamps = [f[1] for f in frames]

    logger.info(
        "Entity %s: running YOLO-Pose pipeline (%d frames)",
        entity_id, len(frame_images),
    )

    for i, img in enumerate(frame_images):
        results = model(img, verbose=False)
        kp_result = results[0].keypoints if results else None

        if kp_result is None or kp_result.xy.shape[0] == 0:
            logger.debug("Entity %s: frame %d — no person detected", entity_id, i + 1)
            continue

        # Take the highest-confidence detection (largest person, typically)
        # YOLO output is sorted by score after NMS.
        kp_xy = kp_result.xy[0]    # (17, 2)
        kp_conf = kp_result.conf[0]  # (17,)
        mean_conf = float(kp_conf.mean())

        if mean_conf < conf_threshold:
            logger.debug(
                "Entity %s: frame %d — mean kp conf %.3f below %.3f",
                entity_id, i + 1, mean_conf, conf_threshold,
            )
            continue

        pose = Pose(
            keypoints=np.array(kp_xy),
            confidences=np.array(kp_conf),
            timestamp=frame_timestamps[i] if i < len(frame_timestamps) else None,
        )
        poses.append(pose)

    logger.info(
        "Entity %s: YOLO-Pose extracted %d/%d poses",
        entity_id, len(poses), len(frame_images),
    )

    if not poses:
        return PatternResult(
            matched=False,
            confidence=0.0,
            pattern_id="shelf_to_waist",
            description="No usable poses extracted (YOLO pipeline)",
        )

    # Run pattern detection
    pattern_config = load_pattern_config(settings.pattern_config_path)
    analyzer = PoseAnalyzer(
        min_frames=settings.min_frames_for_detection,
        confidence_threshold=conf_threshold,
        pattern_config=pattern_config,
    )
    results = analyzer.detect_all_patterns(poses)

    # Return the best match (or first no-match)
    matched = [r for r in results if r.matched]
    if matched:
        best = max(matched, key=lambda r: r.confidence)
        logger.info(
            "Entity %s: YOLO pipeline detected pattern=%s confidence=%.3f",
            entity_id, best.pattern_id, best.confidence,
        )
        return best

    # No match — return the first result for context
    if results:
        return results[0]

    return PatternResult(
        matched=False,
        confidence=0.0,
        pattern_id="shelf_to_waist",
        description="No suspicious pattern detected (YOLO pipeline)",
    )
