# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
GStreamer DL Streamer pipeline runner for pose extraction + pattern detection.

Saves frames as numbered JPEG files, runs gst-launch-1.0 with:
  multifilesrc → jpegdec → gvadetect (person detector) → gvainference (RTMPose)
  → gvapython (pose_logger_rtmpose.py) → fakesink
then reads the JSONL alert output.
"""

import asyncio
import contextlib
import json
import logging
import os
import tempfile

import cv2
import numpy as np

from pose_analyzer import PatternResult

logger = logging.getLogger(__name__)


async def run_gst_pipeline(frames, entity_id: str, settings):
    """
    Run GStreamer DL Streamer pipeline for pose extraction + pattern detection.

    Args:
        frames: List of (frame_image, timestamp) tuples
        entity_id: Person identifier (for logging)
        settings: Settings object with model paths and pipeline config

    Returns:
        PatternResult if a pattern was detected or no match.
        None if pipeline failed or not enough poses could be extracted.
    """
    debug_dir = os.environ.get("GST_DEBUG_DIR")
    if debug_dir:
        tmpdir = os.path.join(debug_dir, f"ba_gst_{entity_id}")
        os.makedirs(tmpdir, exist_ok=True)
        logger.info(f"Entity {entity_id}: debug output in {tmpdir}")
    with (tempfile.TemporaryDirectory(prefix=f"ba_gst_{entity_id}_") if not debug_dir else contextlib.nullcontext(tmpdir)) as tmpdir:
        frames_dir = os.path.join(tmpdir, "frames")
        frames_output = os.path.join(tmpdir, "pose_frames.jsonl")
        alerts_output = os.path.join(tmpdir, "pose_alerts.jsonl")

        frame_images = [f[0] for f in frames]
        num_written = _write_frames_as_jpeg(frame_images, frames_dir)
        if num_written == 0:
            logger.error(f"Entity {entity_id}: failed to write frames as JPEG")
            return None

        h, w = frame_images[0].shape[:2]
        logger.info(
            f"Entity {entity_id}: {num_written} frames written, "
            f"resolution={w}x{h}"
        )
        location_pattern = os.path.join(frames_dir, "%05d.jpg")

        cmd = [
            "gst-launch-1.0",
            "multifilesrc",
                f"location={location_pattern}",
                "index=0",
                f"stop-index={num_written - 1}",
                f"caps=image/jpeg,framerate=2/1", "!",
            "jpegdec", "!",
            "videoconvert", "!",
            "video/x-raw,format=BGRx", "!",
            "gvadetect",
                f"model={settings.person_detector_model}",
                f"device={settings.gst_inference_device}",
                f"threshold={settings.gst_detect_threshold}",
                "nireq=1", "!",
            "gvainference",
                f"model={settings.rtmpose_model}",
                f"device={settings.gst_inference_device}",
                "inference-region=roi-list",
                "nireq=1", "!",
            "gvapython",
                f"module={settings.gvapython_module}",
                "function=process_frame", "!",
            "fakesink",
        ]

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["GST_DEBUG"] = "2"
        env["POSE_FRAMES_OUTPUT"] = frames_output
        env["POSE_ALERTS_OUTPUT"] = alerts_output

        logger.info(
            f"Entity {entity_id}: running GStreamer pipeline "
            f"({len(frame_images)} frames)"
        )

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        try:
            stdout, _unused = await asyncio.wait_for(
                proc.communicate(),
                timeout=settings.gst_pipeline_timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            logger.error(
                f"Entity {entity_id}: GStreamer pipeline timed out "
                f"after {settings.gst_pipeline_timeout}s"
            )
            return None

        combined_output = stdout.decode(errors="replace") if stdout else ""

        if proc.returncode != 0:
            logger.error(
                f"Entity {entity_id}: GStreamer pipeline failed "
                f"(rc={proc.returncode}): {combined_output[-500:]}"
            )
            return None

        # Log pose_logger_rtmpose output (regions, poses, confidence)
        for line in combined_output.splitlines():
            # Forward pose_logger_rtmpose.py log lines (INFO/WARNING)
            if any(kw in line for kw in (
                "Frame ", "NO POSE", "pose captured", "SimCC",
                "regions=", "tensors=", "Layout", "ALERT",
            )):
                logger.info(f"Entity {entity_id}: [gvapython] {line.rstrip()}")

        # Parse pose frames JSONL for diagnostics
        pose_count = 0
        if os.path.exists(frames_output):
            with open(frames_output, "r") as f:
                for line in f:
                    if line.strip():
                        pose_count += 1
        logger.info(
            f"Entity {entity_id}: poses extracted from "
            f"{pose_count}/{len(frame_images)} frames"
        )

        # Warn and save debug info if extraction rate is low
        if pose_count < len(frame_images) * 0.5 and debug_dir:
            logger.warning(
                f"Entity {entity_id}: low pose extraction rate "
                f"({pose_count}/{len(frame_images)}). "
                f"Debug frames preserved in {tmpdir}"
            )
        elif pose_count < len(frame_images) * 0.5:
            logger.warning(
                f"Entity {entity_id}: low pose extraction rate "
                f"({pose_count}/{len(frame_images)}). "
                f"Set GST_DEBUG_DIR to preserve frames for inspection."
            )

        # Parse alerts JSONL
        alerts = []
        if os.path.exists(alerts_output):
            with open(alerts_output, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        alerts.append(json.loads(line))

        if alerts:
            best = max(alerts, key=lambda a: a.get("confidence", 0))
            logger.info(
                f"Entity {entity_id}: GStreamer pipeline detected "
                f"pattern={best.get('pattern')} "
                f"confidence={best.get('confidence', 0):.3f}"
            )
            return PatternResult(
                matched=True,
                confidence=best.get("confidence", 0.0),
                pattern_id=best.get("pattern", "shelf_to_waist"),
                description=best.get("description", "Detected by GStreamer pipeline"),
            )
        else:
            logger.info(
                f"Entity {entity_id}: GStreamer pipeline — no pattern detected"
            )
            return PatternResult(
                matched=False,
                confidence=0.0,
                pattern_id="shelf_to_waist",
                description="No suspicious pattern detected (GStreamer pipeline)",
            )


def _write_frames_as_jpeg(
    frame_images: list[np.ndarray], frames_dir: str
) -> int:
    """Write frames as numbered JPEG files for multifilesrc.

    Files are named 00000.jpg, 00001.jpg, etc.

    Returns:
        Number of frames written (0 on failure).
    """
    if not frame_images:
        return 0

    os.makedirs(frames_dir, exist_ok=True)
    h, w = frame_images[0].shape[:2]
    count = 0

    for i, img in enumerate(frame_images):
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        path = os.path.join(frames_dir, f"{i:05d}.jpg")
        ok = cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            count += 1

    return count
