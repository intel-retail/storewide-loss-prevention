"""
Pose Analyzer using YOLO-Pose

Extracts keypoints from frames and detects suspicious activity patterns.
"""

import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np
from ultralytics import YOLO

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
            self.confidences[idx]
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
    key_frames: list[int] = None  # Frame indices that triggered detection


class PoseAnalyzer:
    """
    Analyzes pose sequences to detect suspicious patterns.
    
    Uses YOLO-Pose for keypoint extraction and rule-based pattern matching.
    """
    
    def __init__(
        self,
        model_path: str = "yolo11n-pose.pt",
        min_frames: int = 10,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the pose analyzer.
        
        Args:
            model_path: Path to YOLO-Pose model weights
            min_frames: Minimum frames required for pattern detection
            confidence_threshold: Minimum keypoint confidence to consider valid
        """
        self.model_path = model_path
        self.min_frames = min_frames
        self.confidence_threshold = confidence_threshold
        self.model: Optional[YOLO] = None
        
        self._load_model()
    
    def _load_model(self):
        """Load YOLO-Pose model."""
        try:
            logger.info(f"Loading YOLO-Pose model: {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def extract_poses(self, frames: list[tuple[np.ndarray, int]]) -> list[Pose]:
        """
        Extract pose keypoints from frames.
        
        Args:
            frames: List of (frame_image, timestamp) tuples
        
        Returns:
            List of Pose objects
        """
        poses = []
        
        for frame_img, timestamp in frames:
            try:
                # Run YOLO-Pose inference
                results = self.model(frame_img, verbose=False)
                
                for result in results:
                    if result.keypoints is None or len(result.keypoints) == 0:
                        continue
                    
                    # Get first person's keypoints (frame should be cropped to single person)
                    kp_xy = result.keypoints.xy[0].cpu().numpy()  # Shape: [17, 2]
                    kp_conf = result.keypoints.conf[0].cpu().numpy()  # Shape: [17]
                    
                    # Normalize coordinates to 0-1 range
                    h, w = frame_img.shape[:2]
                    kp_xy_norm = kp_xy.copy()
                    kp_xy_norm[:, 0] /= w
                    kp_xy_norm[:, 1] /= h
                    
                    pose = Pose(
                        keypoints=kp_xy_norm,
                        confidences=kp_conf,
                        timestamp=timestamp
                    )
                    poses.append(pose)
                    break  # Only first person per frame
                    
            except Exception as e:
                logger.warning(f"Failed to extract pose from frame: {e}")
                continue
        
        return poses
    
    def detect_pattern(
        self,
        pose_sequence: list[Pose],
        pattern_id: str = "shelf_to_waist"
    ) -> PatternResult:
        """
        Detect suspicious activity pattern in pose sequence.
        
        Args:
            pose_sequence: List of Pose objects (chronological order)
            pattern_id: Pattern to detect
        
        Returns:
            PatternResult indicating if pattern was detected
        """
        if pattern_id == "shelf_to_waist":
            return self._detect_shelf_to_waist(pose_sequence)
        else:
            return PatternResult(
                matched=False,
                confidence=0.0,
                pattern_id=pattern_id,
                description=f"Unknown pattern: {pattern_id}"
            )
    
    def _detect_shelf_to_waist(self, poses: list[Pose]) -> PatternResult:
        """
        Detect shelf-to-waist hand movement pattern.
        
        Pattern: Hand moves from above chest level to waist/pocket area.
        
        Detection logic:
        1. First half of window: hand should be ABOVE chest (reaching to shelf)
        2. Second half of window: hand should be NEAR waist (concealing)
        """
        if len(poses) < self.min_frames:
            return PatternResult(
                matched=False,
                confidence=0.0,
                pattern_id="shelf_to_waist",
                description=f"Not enough frames: {len(poses)}/{self.min_frames}"
            )
        
        # Split into first half (reaching) and second half (concealing)
        mid = len(poses) // 2
        first_half = poses[:mid]
        second_half = poses[mid:]
        
        # Check both wrists
        for wrist_name, wrist_getter in [("left", lambda p: p.left_wrist), 
                                          ("right", lambda p: p.right_wrist)]:
            
            # Step 1: Check if hand was above chest in first half
            hand_raised_count = 0
            for pose in first_half:
                wrist = wrist_getter(pose)
                chest = pose.chest_midpoint
                
                # Check confidence
                if wrist[2] < self.confidence_threshold:
                    continue
                
                # Hand above chest? (lower y = higher in image)
                if wrist[1] < chest[1]:
                    hand_raised_count += 1
            
            # Need at least 2 frames with hand raised
            if hand_raised_count < 2:
                continue
            
            # Step 2: Check if hand moved to waist in second half
            hand_at_waist_count = 0
            for pose in second_half:
                wrist = wrist_getter(pose)
                waist = pose.waist_midpoint
                
                # Check confidence
                if wrist[2] < self.confidence_threshold:
                    continue
                
                # Hand near waist? (within 15% of frame height)
                distance = self._euclidean_distance(
                    (wrist[0], wrist[1]),
                    waist
                )
                
                if distance < 0.15:  # 15% normalized distance
                    hand_at_waist_count += 1
            
            # Need at least 3 frames with hand at waist
            if hand_at_waist_count < 3:
                continue
            
            # Pattern detected!
            confidence = min(1.0, (hand_raised_count + hand_at_waist_count) / (mid + len(second_half)))
            
            return PatternResult(
                matched=True,
                confidence=confidence,
                pattern_id="shelf_to_waist",
                description=f"{wrist_name.capitalize()} hand: raised in {hand_raised_count} frames, "
                           f"at waist in {hand_at_waist_count} frames"
            )
        
        # No pattern detected
        return PatternResult(
            matched=False,
            confidence=0.0,
            pattern_id="shelf_to_waist",
            description="Hand movement pattern not detected"
        )
    
    @staticmethod
    def _euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
