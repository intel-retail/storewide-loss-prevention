"""
Configuration settings for BehavioralAnalysis Service.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Service configuration loaded from environment variables."""
    
    # Service settings
    debug: bool = False
    log_level: str = "INFO"
    
    # YOLO-Pose model settings
    yolo_model_path: str = "yolo11n-pose.pt"
    pose_confidence_threshold: float = 0.5
    
    # Frame analysis settings
    min_frames_for_detection: int = 10
    max_frames_to_fetch: int = 20
    
    # SeaweedFS settings
    seaweedfs_endpoint: str = "http://localhost:8333"
    seaweedfs_bucket: str = "behavioral-frames"
    seaweedfs_access_key: str = ""
    seaweedfs_secret_key: str = ""
    seaweedfs_max_frame_age: int = 30  # seconds
    
    class Config:
        env_prefix = ""  # No prefix, use exact variable names
        case_sensitive = False
