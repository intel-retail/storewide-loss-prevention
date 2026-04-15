"""
Tests for PoseAnalyzer
"""

import pytest
import numpy as np
from pose_analyzer import PoseAnalyzer, Pose, Keypoints, PatternResult


class TestPose:
    """Tests for Pose dataclass."""
    
    def test_waist_midpoint(self):
        """Test waist midpoint calculation."""
        keypoints = np.zeros((17, 2))
        keypoints[Keypoints.LEFT_HIP] = [0.3, 0.6]
        keypoints[Keypoints.RIGHT_HIP] = [0.7, 0.6]
        confidences = np.ones(17)
        
        pose = Pose(keypoints=keypoints, confidences=confidences)
        waist = pose.waist_midpoint
        
        assert waist == (0.5, 0.6)
    
    def test_chest_midpoint(self):
        """Test chest midpoint calculation."""
        keypoints = np.zeros((17, 2))
        keypoints[Keypoints.LEFT_SHOULDER] = [0.3, 0.3]
        keypoints[Keypoints.RIGHT_SHOULDER] = [0.7, 0.3]
        confidences = np.ones(17)
        
        pose = Pose(keypoints=keypoints, confidences=confidences)
        chest = pose.chest_midpoint
        
        assert chest == (0.5, 0.3)


class TestPatternDetection:
    """Tests for pattern detection logic."""
    
    def create_pose(
        self,
        wrist_y: float,
        chest_y: float = 0.3,
        waist_y: float = 0.6,
        confidence: float = 0.9
    ) -> Pose:
        """Create a pose with specified wrist position."""
        keypoints = np.zeros((17, 2))
        confidences = np.ones(17) * confidence
        
        # Set shoulders (chest reference)
        keypoints[Keypoints.LEFT_SHOULDER] = [0.4, chest_y]
        keypoints[Keypoints.RIGHT_SHOULDER] = [0.6, chest_y]
        
        # Set hips (waist reference)
        keypoints[Keypoints.LEFT_HIP] = [0.4, waist_y]
        keypoints[Keypoints.RIGHT_HIP] = [0.6, waist_y]
        
        # Set right wrist at specified y position
        keypoints[Keypoints.RIGHT_WRIST] = [0.5, wrist_y]
        keypoints[Keypoints.LEFT_WRIST] = [0.3, 0.5]  # Neutral position
        
        return Pose(keypoints=keypoints, confidences=confidences)
    
    def test_suspicious_pattern_detected(self):
        """Test that shelf-to-waist pattern is detected."""
        # Create pose sequence:
        # First half: hand above chest (y < 0.3)
        # Second half: hand at waist (y ~ 0.6)
        
        poses = []
        
        # First 5 frames: hand raised (y = 0.15, above chest at 0.3)
        for _ in range(5):
            poses.append(self.create_pose(wrist_y=0.15))
        
        # Last 5 frames: hand at waist (y = 0.58, near waist at 0.6)
        for _ in range(5):
            poses.append(self.create_pose(wrist_y=0.58))
        
        # Create analyzer with mocked model (we test pattern logic only)
        analyzer = PoseAnalyzer.__new__(PoseAnalyzer)
        analyzer.min_frames = 10
        analyzer.confidence_threshold = 0.5
        
        result = analyzer._detect_shelf_to_waist(poses)
        
        assert result.matched is True
        assert result.pattern_id == "shelf_to_waist"
        assert result.confidence > 0.5
    
    def test_no_pattern_hand_stays_low(self):
        """Test that pattern is not detected when hand stays low."""
        poses = []
        
        # All frames: hand at waist level (no raised frames)
        for _ in range(10):
            poses.append(self.create_pose(wrist_y=0.55))
        
        analyzer = PoseAnalyzer.__new__(PoseAnalyzer)
        analyzer.min_frames = 10
        analyzer.confidence_threshold = 0.5
        
        result = analyzer._detect_shelf_to_waist(poses)
        
        assert result.matched is False
    
    def test_no_pattern_hand_stays_high(self):
        """Test that pattern is not detected when hand stays high."""
        poses = []
        
        # All frames: hand raised (no waist frames)
        for _ in range(10):
            poses.append(self.create_pose(wrist_y=0.15))
        
        analyzer = PoseAnalyzer.__new__(PoseAnalyzer)
        analyzer.min_frames = 10
        analyzer.confidence_threshold = 0.5
        
        result = analyzer._detect_shelf_to_waist(poses)
        
        assert result.matched is False
    
    def test_not_enough_frames(self):
        """Test that pattern is not detected with insufficient frames."""
        poses = [self.create_pose(wrist_y=0.15) for _ in range(5)]
        
        analyzer = PoseAnalyzer.__new__(PoseAnalyzer)
        analyzer.min_frames = 10
        analyzer.confidence_threshold = 0.5
        
        result = analyzer._detect_shelf_to_waist(poses)
        
        assert result.matched is False
        assert "Not enough frames" in result.description
    
    def test_low_confidence_keypoints_ignored(self):
        """Test that low confidence keypoints are ignored."""
        poses = []
        
        # First half: hand raised but LOW confidence
        for _ in range(5):
            poses.append(self.create_pose(wrist_y=0.15, confidence=0.3))
        
        # Second half: hand at waist with good confidence
        for _ in range(5):
            poses.append(self.create_pose(wrist_y=0.58, confidence=0.9))
        
        analyzer = PoseAnalyzer.__new__(PoseAnalyzer)
        analyzer.min_frames = 10
        analyzer.confidence_threshold = 0.5
        
        result = analyzer._detect_shelf_to_waist(poses)
        
        # Should not match because first half has low confidence
        assert result.matched is False


class TestEuclideanDistance:
    """Test distance calculation."""
    
    def test_same_point(self):
        dist = PoseAnalyzer._euclidean_distance((0.5, 0.5), (0.5, 0.5))
        assert dist == 0.0
    
    def test_horizontal_distance(self):
        dist = PoseAnalyzer._euclidean_distance((0.0, 0.5), (1.0, 0.5))
        assert dist == 1.0
    
    def test_vertical_distance(self):
        dist = PoseAnalyzer._euclidean_distance((0.5, 0.0), (0.5, 1.0))
        assert dist == 1.0
    
    def test_diagonal_distance(self):
        dist = PoseAnalyzer._euclidean_distance((0.0, 0.0), (0.3, 0.4))
        assert abs(dist - 0.5) < 0.001
