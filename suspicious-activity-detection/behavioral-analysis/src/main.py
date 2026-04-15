"""
BehavioralAnalysis Service

Analyzes pose sequences to detect suspicious activity patterns.
Uses YOLO-Pose for keypoint extraction and pattern matching for detection.
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from pose_analyzer import PoseAnalyzer
from frame_store import FrameStore
from config import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    # Startup
    logger.info("Starting BehavioralAnalysis Service")
    app.state.pose_analyzer = PoseAnalyzer(
        model_path=settings.yolo_model_path,
        min_frames=settings.min_frames_for_detection,
        confidence_threshold=settings.pose_confidence_threshold
    )
    app.state.frame_store = FrameStore(
        endpoint=settings.seaweedfs_endpoint,
        bucket=settings.seaweedfs_bucket,
        access_key=settings.seaweedfs_access_key,
        secret_key=settings.seaweedfs_secret_key
    )
    logger.info("Service initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down BehavioralAnalysis Service")


app = FastAPI(
    title="BehavioralAnalysis Service",
    description="Pose-based suspicious activity detection",
    version="1.0.0",
    lifespan=lifespan
)


# ─────────────────────────────────────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """Request to analyze frames for an entity."""
    entity_id: str
    pattern_id: str = "shelf_to_waist"  # Pattern to detect


class AnalyzeResponse(BaseModel):
    """Response from pose analysis."""
    entity_id: str
    status: str  # "no_data" | "accumulating" | "no_match" | "suspicious"
    frames_available: int
    frames_required: int
    confidence: Optional[float] = None
    pattern_id: Optional[str] = None
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    seaweedfs_connected: bool


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health."""
    frame_store: FrameStore = app.state.frame_store
    pose_analyzer: PoseAnalyzer = app.state.pose_analyzer
    
    return HealthResponse(
        status="healthy",
        model_loaded=pose_analyzer.is_loaded(),
        seaweedfs_connected=await frame_store.check_connection()
    )


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze_activity(request: AnalyzeRequest):
    """
    Analyze frames for suspicious activity.
    
    Flow:
    1. Fetch frames for entity_id from SeaweedFS
    2. If not enough frames, return "no_data" or "accumulating"
    3. Extract pose keypoints from each frame
    4. Run pattern detection on pose sequence
    5. Return result
    """
    frame_store: FrameStore = app.state.frame_store
    pose_analyzer: PoseAnalyzer = app.state.pose_analyzer
    
    entity_id = request.entity_id
    pattern_id = request.pattern_id
    min_frames = settings.min_frames_for_detection
    
    try:
        # Step 1: Fetch frames from SeaweedFS
        frames = await frame_store.get_frames(
            entity_id=entity_id,
            max_frames=settings.max_frames_to_fetch
        )
        
        frames_available = len(frames)
        logger.info(f"Entity {entity_id}: {frames_available} frames available")
        
        # Step 2: Check if we have enough frames
        if frames_available == 0:
            return AnalyzeResponse(
                entity_id=entity_id,
                status="no_data",
                frames_available=0,
                frames_required=min_frames,
                message="No frames available for this entity"
            )
        
        if frames_available < min_frames:
            return AnalyzeResponse(
                entity_id=entity_id,
                status="accumulating",
                frames_available=frames_available,
                frames_required=min_frames,
                message=f"Need {min_frames - frames_available} more frames"
            )
        
        # Step 3: Extract poses from frames
        pose_sequence = pose_analyzer.extract_poses(frames)
        
        if len(pose_sequence) < min_frames:
            return AnalyzeResponse(
                entity_id=entity_id,
                status="accumulating",
                frames_available=frames_available,
                frames_required=min_frames,
                message="Could not extract poses from enough frames"
            )
        
        # Step 4: Run pattern detection
        result = pose_analyzer.detect_pattern(
            pose_sequence=pose_sequence,
            pattern_id=pattern_id
        )
        
        # Step 5: Return result
        if result.matched:
            return AnalyzeResponse(
                entity_id=entity_id,
                status="suspicious",
                frames_available=frames_available,
                frames_required=min_frames,
                confidence=result.confidence,
                pattern_id=pattern_id,
                message=result.description
            )
        else:
            return AnalyzeResponse(
                entity_id=entity_id,
                status="no_match",
                frames_available=frames_available,
                frames_required=min_frames,
                confidence=result.confidence,
                pattern_id=pattern_id,
                message="No suspicious pattern detected"
            )
    
    except Exception as e:
        logger.error(f"Error analyzing entity {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/entities/{entity_id}/frames")
async def clear_entity_frames(entity_id: str):
    """Clear all frames for an entity (called when tracking stops)."""
    frame_store: FrameStore = app.state.frame_store
    
    try:
        deleted_count = await frame_store.delete_frames(entity_id)
        return {"entity_id": entity_id, "deleted_frames": deleted_count}
    except Exception as e:
        logger.error(f"Error clearing frames for {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=settings.debug
    )
