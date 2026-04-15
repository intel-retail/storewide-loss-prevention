# BehavioralAnalysis Service

Pose-based suspicious activity detection service using YOLO-Pose.

## Overview

This service analyzes cropped person frames to detect suspicious hand movement patterns (e.g., shelf-to-waist concealment gesture). Frames are stored in SeaweedFS by the Loss Prevention service, and this service reads them for analysis.

## Architecture

```
Loss Prevention Service               BehavioralAnalysis Service
        │                                       │
        │ 1. Store frame                        │
        ├──────────────────────► SeaweedFS      │
        │                            │          │
        │ 2. POST /analyze           │          │
        ├───────────────────────────────────────►
        │                            │          │
        │                            │ 3. Fetch frames
        │                            ◄──────────┤
        │                            │          │
        │                            │ 4. Run YOLO-Pose
        │                            │          │
        │ 5. Return result           │          │
        ◄───────────────────────────────────────┤
```

## API

### POST /api/v1/analyze

Analyze frames for suspicious activity.

**Request:**
```json
{
  "entity_id": "person_42",
  "pattern_id": "shelf_to_waist"
}
```

**Response:**
```json
{
  "entity_id": "person_42",
  "status": "suspicious",
  "frames_available": 12,
  "frames_required": 10,
  "confidence": 0.85,
  "pattern_id": "shelf_to_waist",
  "message": "Right hand: raised in 3 frames, at waist in 4 frames"
}
```

**Status values:**
| Status | Meaning |
|--------|---------|
| `no_data` | No frames available for this entity |
| `accumulating` | Not enough frames yet (keep collecting) |
| `no_match` | Enough frames analyzed, no suspicious pattern detected |
| `suspicious` | Suspicious pattern detected! |

### DELETE /api/v1/entities/{entity_id}/frames

Clear all frames for an entity (call when tracking stops).

### GET /health

Health check endpoint.

## Frame Storage Pattern

Frames are stored in SeaweedFS with this structure:

```
bucket: behavioral-frames
└── {entity_id}/
    └── frames/
        ├── 1713088800000.jpg   (timestamp in milliseconds)
        ├── 1713088800500.jpg
        ├── 1713088801000.jpg
        └── ...
```

### How Loss Prevention Should Store Frames

```python
import boto3
import cv2
from datetime import datetime

# Initialize S3 client for SeaweedFS
s3 = boto3.client(
    's3',
    endpoint_url='http://seaweedfs:8333',
    aws_access_key_id='',
    aws_secret_access_key=''
)

BUCKET = 'behavioral-frames'

def store_frame(entity_id: str, frame: np.ndarray):
    """Store a cropped person frame to SeaweedFS."""
    # Generate timestamp key
    timestamp = int(datetime.utcnow().timestamp() * 1000)
    key = f"{entity_id}/frames/{timestamp}.jpg"
    
    # Encode frame as JPEG
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    # Upload to SeaweedFS
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=buffer.tobytes(),
        ContentType='image/jpeg'
    )
    
    return key
```

### How Loss Prevention Should Call Analyze

```python
import httpx

BEHAVIORAL_URL = "http://behavioral-analysis:8080"

async def check_suspicious_activity(entity_id: str) -> dict:
    """Call BehavioralAnalysis service to check for suspicious activity."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BEHAVIORAL_URL}/api/v1/analyze",
            json={
                "entity_id": entity_id,
                "pattern_id": "shelf_to_waist"
            },
            timeout=5.0
        )
        return response.json()

# Usage in frame processing loop:
result = await check_suspicious_activity("person_42")

if result["status"] == "suspicious":
    # Fire CONCEALMENT alert!
    await alert_service.publish(
        alert_type="CONCEALMENT",
        entity_id="person_42",
        confidence=result["confidence"],
        message=result["message"]
    )
elif result["status"] == "accumulating":
    # Keep collecting frames
    pass
elif result["status"] == "no_match":
    # Normal behavior, continue monitoring
    pass
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_MODEL_PATH` | `yolo11n-pose.pt` | Path to YOLO-Pose model |
| `MIN_FRAMES_FOR_DETECTION` | `10` | Minimum frames needed |
| `MAX_FRAMES_TO_FETCH` | `20` | Maximum frames to analyze |
| `POSE_CONFIDENCE_THRESHOLD` | `0.5` | Minimum keypoint confidence |
| `SEAWEEDFS_ENDPOINT` | `http://localhost:8333` | SeaweedFS S3 endpoint |
| `SEAWEEDFS_BUCKET` | `behavioral-frames` | Bucket for frames |
| `SEAWEEDFS_ACCESS_KEY` | `` | S3 access key |
| `SEAWEEDFS_SECRET_KEY` | `` | S3 secret key |

## Running Locally

```bash
cd behavioral-analysis

# Install dependencies
pip install -e .

# Run service
cd src
python main.py
```

## Docker

```bash
# Build
docker build -t behavioral-analysis .

# Run
docker run -p 8080:8080 \
  -e SEAWEEDFS_ENDPOINT=http://seaweedfs:8333 \
  behavioral-analysis
```

## Pattern Detection: Shelf-to-Waist

The service detects the "shelf-to-waist" concealment pattern:

1. **First half of frames:** Hand should be ABOVE chest level (reaching toward shelf)
2. **Second half of frames:** Hand should be NEAR waist/pocket area (concealing)

```
Keypoints used (COCO 17-point):
─────────────────────────────────
- Left/Right Wrist (indices 9, 10)
- Left/Right Shoulder (indices 5, 6) → chest midpoint
- Left/Right Hip (indices 11, 12) → waist midpoint

Detection criteria:
─────────────────────────────────
- At least 2 frames with hand above chest (first half)
- At least 3 frames with hand near waist (second half)
- Keypoint confidence > 0.5
- Distance to waist < 15% of frame height
```
