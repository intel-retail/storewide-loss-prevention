# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Pose Analyzer

Detects suspicious activity patterns from pose sequences.
Pose extraction is handled by the YOLO pose pipeline.
When a pose pattern matches, optionally sends frames to VLM for confirmation.
The service is generic — patterns and prompts are loaded from config.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from vlm_client import VLMClient

logger = logging.getLogger(__name__)


# COCO 17-keypoint indices
class Keypoints:
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


@dataclass
class Pose:
    """Single frame pose data."""

    keypoints: np.ndarray  # Shape: [17, 2] - x, y coordinates
    confidences: np.ndarray  # Shape: [17] - confidence per keypoint
    timestamp: Optional[int] = None

    def get_keypoint(self, idx: int) -> tuple[float, float, float]:
        """Get keypoint (x, y, confidence)."""
        return (
            self.keypoints[idx][0],
            self.keypoints[idx][1],
            self.confidences[idx],
        )

    @property
    def left_wrist(self) -> tuple[float, float, float]:
        return self.get_keypoint(Keypoints.LEFT_WRIST)

    @property
    def right_wrist(self) -> tuple[float, float, float]:
        return self.get_keypoint(Keypoints.RIGHT_WRIST)

    @property
    def left_hip(self) -> tuple[float, float, float]:
        return self.get_keypoint(Keypoints.LEFT_HIP)

    @property
    def right_hip(self) -> tuple[float, float, float]:
        return self.get_keypoint(Keypoints.RIGHT_HIP)

    @property
    def left_shoulder(self) -> tuple[float, float, float]:
        return self.get_keypoint(Keypoints.LEFT_SHOULDER)

    @property
    def right_shoulder(self) -> tuple[float, float, float]:
        return self.get_keypoint(Keypoints.RIGHT_SHOULDER)

    @property
    def waist_midpoint(self) -> tuple[float, float]:
        """Calculate waist midpoint from hips."""
        lh = self.left_hip
        rh = self.right_hip
        return ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)

    @property
    def chest_midpoint(self) -> tuple[float, float]:
        """Calculate chest midpoint from shoulders."""
        ls = self.left_shoulder
        rs = self.right_shoulder
        return ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)


@dataclass
class PatternResult:
    """Result of pattern detection."""

    matched: bool
    confidence: float
    pattern_id: str
    description: str
    key_frames: list[int] = field(default_factory=list)
    vlm_result: Optional[dict[str, Any]] = None
    vlm_confirmed: Optional[bool] = None


class PoseAnalyzer:
    """
    Analyzes pose sequences to detect suspicious patterns.

    Pose extraction is handled externally by the GStreamer DL Streamer pipeline.
    This class provides pattern matching on pose sequences and VLM confirmation.
    """

    def __init__(
        self,
        min_frames: int = 10,
        confidence_threshold: float = 0.5,
        vlm_client: Optional[VLMClient] = None,
        pattern_config: Optional[dict[str, Any]] = None,
    ):
        self.min_frames = min_frames
        self.confidence_threshold = confidence_threshold
        self.vlm_client = vlm_client
        self.pattern_config = pattern_config or {}
        logger.info("PoseAnalyzer initialized (pattern detection + VLM confirmation)")

    def is_loaded(self) -> bool:
        """Check if analyzer is ready."""
        return True

    def detect_pattern(
        self,
        pose_sequence: list[Pose],
        pattern_id: str = "shelf_to_waist",
    ) -> PatternResult:
        """
        Detect suspicious activity pattern in pose sequence.

        Args:
            pose_sequence: List of Pose objects (chronological order)
            pattern_id: Pattern to detect

        Returns:
            PatternResult indicating if pattern was detected
        """
        # Check if pattern is enabled in config
        pattern_cfg = self.pattern_config.get(pattern_id, {})
        if pattern_cfg and not pattern_cfg.get("enabled", True):
            return PatternResult(
                matched=False,
                confidence=0.0,
                pattern_id=pattern_id,
                description=f"Pattern '{pattern_id}' is disabled",
            )

        if pattern_id == "shelf_to_waist":
            return self._detect_shelf_to_waist(pose_sequence, pattern_cfg)
        else:
            logger.warning(
                "Pattern '%s' is configured but has no implementation — skipping",
                pattern_id,
            )
            return PatternResult(
                matched=False,
                confidence=0.0,
                pattern_id=pattern_id,
                description=f"Pattern '{pattern_id}' not implemented",
            )

    def detect_all_patterns(
        self,
        pose_sequence: list[Pose],
    ) -> list[PatternResult]:
        """
        Run all enabled patterns against a pose sequence.

        Args:
            pose_sequence: List of Pose objects (chronological order)

        Returns:
            List of PatternResult for each enabled pattern
        """
        pattern_ids = list(self.pattern_config.keys()) if self.pattern_config else ["shelf_to_waist"]
        results = []
        for pattern_id in pattern_ids:
            cfg = self.pattern_config.get(pattern_id, {})
            if cfg and not cfg.get("enabled", True):
                continue
            results.append(self.detect_pattern(pose_sequence, pattern_id))
        return results

    async def analyze_with_vlm(
        self,
        frames: list[tuple[np.ndarray, int]],
        pose_result: PatternResult,
        frame_key_prefix: str = "",
    ) -> PatternResult:
        """
        Send frames to VLM for visual confirmation after pose match.

        Only called when:
        1. Pose pattern matched
        2. VLM is enabled globally
        3. VLM is enabled for this pattern in config

        Args:
            frames: List of (frame_image, timestamp) tuples
            pose_result: The result from pose-based detection

        Returns:
            Updated PatternResult with VLM confirmation
        """
        if not self.vlm_client:
            return pose_result

        pattern_cfg = self.pattern_config.get(pose_result.pattern_id, {})
        vlm_cfg = pattern_cfg.get("vlm", {})

        if not vlm_cfg.get("enabled", True):
            logger.debug(f"VLM disabled for pattern {pose_result.pattern_id}")
            return pose_result

        prompt = vlm_cfg.get("prompt", "")
        if not prompt:
            logger.warning(f"No VLM prompt configured for pattern {pose_result.pattern_id}")
            return pose_result

        # Sample frames evenly for VLM
        num_frames = vlm_cfg.get("num_frames", 4)
        sampled = self._sample_frames(frames, num_frames)
        frame_images = [f[0] for f in sampled]
        sampled_ts = [int(f[1]) for f in sampled]
        sampled_keys = (
            [f"{frame_key_prefix}{ts}.jpg" for ts in sampled_ts]
            if frame_key_prefix else []
        )

        logger.info(
            "Sending %d frames to VLM for pattern '%s' "
            "(input_pool=%d, sampled_keys=%s)",
            len(frame_images), pose_result.pattern_id,
            len(frames), sampled_keys or sampled_ts,
        )

        vlm_result = await self.vlm_client.analyze(frame_images, prompt)

        if not vlm_result.success:
            logger.warning(f"VLM analysis failed: {vlm_result.error}")
            # Fall back to pose-only result
            pose_result.vlm_confirmed = None
            return pose_result

        parsed = vlm_result.parsed
        pose_result.vlm_result = parsed

        # Check if VLM confirms the suspicious behavior
        vlm_suspicious = parsed.get("suspicious", False) if parsed else False
        vlm_confidence = parsed.get("confidence", 0.0) if parsed else 0.0
        vlm_reasoning = parsed.get("reasoning", "") if parsed else ""

        pose_result.vlm_confirmed = vlm_suspicious

        if vlm_suspicious:
            # Combine pose and VLM confidence
            combined = (pose_result.confidence + vlm_confidence) / 2
            pose_result.confidence = combined
            pose_result.description = (
                f"{pose_result.description} | VLM confirms: {vlm_reasoning}"
            )
        else:
            # VLM disagrees — lower confidence but keep match
            pose_result.confidence = pose_result.confidence * 0.5
            pose_result.description = (
                f"{pose_result.description} | VLM disagrees: {vlm_reasoning}"
            )

        logger.info(
            f"VLM confirmation: suspicious={vlm_suspicious}, "
            f"vlm_confidence={vlm_confidence:.2f}, "
            f"combined_confidence={pose_result.confidence:.3f}"
        )

        return pose_result

    @staticmethod
    def _sample_frames(
        frames: list[tuple[np.ndarray, int]],
        n: int,
    ) -> list[tuple[np.ndarray, int]]:
        """Evenly sample n frames from a sequence."""
        if len(frames) <= n:
            return frames
        indices = np.linspace(0, len(frames) - 1, n, dtype=int)
        return [frames[i] for i in indices]

    def _detect_shelf_to_waist(
        self,
        poses: list[Pose],
        pattern_cfg: dict[str, Any] = None,
    ) -> PatternResult:
        """
        Detect shelf-to-waist hand movement pattern.

        Pattern: Hand moves from above chest level to waist/pocket area.

        Detection logic (sliding window):
        For each possible split point in the sequence, check whether the
        early portion has enough "hand raised" frames and the late portion
        has enough "hand at waist" frames.  This avoids requiring the
        transition to happen exactly at the midpoint.

        Body-relative threshold:
        Instead of a fixed normalised distance, the waist proximity
        threshold is expressed as a fraction of the person's torso length
        (shoulder-midpoint to hip-midpoint), so it scales with distance
        from the camera.
        """
        if pattern_cfg is None:
            pattern_cfg = {}
        pose_cfg = pattern_cfg.get("pose", {})

        min_raised = pose_cfg.get("min_hand_raised_frames", 2)
        min_at_waist = pose_cfg.get("min_hand_at_waist_frames", 2)
        waist_ratio = pose_cfg.get("waist_proximity_ratio", 0.6)

        if len(poses) < self.min_frames:
            return PatternResult(
                matched=False,
                confidence=0.0,
                pattern_id="shelf_to_waist",
                description=f"Not enough frames: {len(poses)}/{self.min_frames}",
            )

        # Pre-classify each frame per wrist: "raised", "at_waist", or None
        for wrist_name, wrist_getter in [
            ("left", lambda p: p.left_wrist),
            ("right", lambda p: p.right_wrist),
        ]:
            raised_flags: list[bool] = []
            waist_flags: list[bool] = []

            for pose in poses:
                wrist = wrist_getter(pose)
                ls = pose.left_shoulder
                rs = pose.right_shoulder
                lh = pose.left_hip
                rh = pose.right_hip

                # Skip frame if wrist or body keypoints are low confidence
                body_ok = (
                    wrist[2] >= self.confidence_threshold
                    and ls[2] >= self.confidence_threshold
                    and rs[2] >= self.confidence_threshold
                    and lh[2] >= self.confidence_threshold
                    and rh[2] >= self.confidence_threshold
                )

                if not body_ok:
                    raised_flags.append(False)
                    waist_flags.append(False)
                    continue

                chest = pose.chest_midpoint
                waist = pose.waist_midpoint

                # Torso length for body-relative threshold
                torso_len = self._euclidean_distance(chest, waist)
                if torso_len < 1e-4:
                    raised_flags.append(False)
                    waist_flags.append(False)
                    continue

                wrist_pt = (wrist[0], wrist[1])

                # Hand above waist? (lower y = higher in image)
                # Uses waist midpoint as reference — retail shelves are
                # typically at chest-to-waist height from store cameras.
                is_raised = wrist[1] < waist[1]

                # Hand near waist? (distance < waist_ratio * torso_len)
                dist_to_waist = self._euclidean_distance(wrist_pt, waist)
                is_at_waist = dist_to_waist < (waist_ratio * torso_len)

                raised_flags.append(is_raised)
                waist_flags.append(is_at_waist)

            # Sliding split: try every split point from min_raised .. len-min_at_waist
            n = len(poses)
            best_conf = 0.0
            best_split = -1
            best_raised = 0
            best_waist = 0

            for split in range(min_raised, n - min_at_waist + 1):
                r_count = sum(raised_flags[:split])
                w_count = sum(waist_flags[split:])

                if r_count >= min_raised and w_count >= min_at_waist:
                    conf = (r_count + w_count) / n
                    if conf > best_conf:
                        best_conf = conf
                        best_split = split
                        best_raised = r_count
                        best_waist = w_count

            if best_split >= 0:
                return PatternResult(
                    matched=True,
                    confidence=min(1.0, best_conf),
                    pattern_id="shelf_to_waist",
                    description=(
                        f"{wrist_name.capitalize()} hand: raised in "
                        f"{best_raised} frames (0-{best_split - 1}), "
                        f"at waist in {best_waist} frames "
                        f"({best_split}-{n - 1})"
                    ),
                )

        # No pattern detected
        return PatternResult(
            matched=False,
            confidence=0.0,
            pattern_id="shelf_to_waist",
            description="Hand movement pattern not detected",
        )

    @staticmethod
    def _euclidean_distance(
        p1: tuple[float, float], p2: tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two points."""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
