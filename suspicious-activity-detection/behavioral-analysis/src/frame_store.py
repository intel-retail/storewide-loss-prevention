"""
Frame Store Client for SeaweedFS

Handles reading and writing cropped person frames to SeaweedFS.
"""

import asyncio
import logging
from io import BytesIO
from typing import Optional
import aioboto3
import cv2
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FrameStore:
    """
    Async client for storing and retrieving frames from SeaweedFS.
    
    Storage Structure:
    ─────────────────
    bucket: behavioral-frames
    └── {entity_id}/
        └── frames/
            ├── {timestamp_1}.jpg
            ├── {timestamp_2}.jpg
            └── ...
    
    Frames are stored with timestamp as filename for easy ordering.
    """
    
    def __init__(
        self,
        endpoint: str,
        bucket: str = "behavioral-frames",
        access_key: str = "",
        secret_key: str = "",
        max_frame_age_seconds: int = 30
    ):
        """
        Initialize SeaweedFS client.
        
        Args:
            endpoint: SeaweedFS S3 endpoint (e.g., "http://localhost:8333")
            bucket: Bucket name for storing frames
            access_key: S3 access key
            secret_key: S3 secret key
            max_frame_age_seconds: Maximum age of frames to fetch
        """
        self.endpoint = endpoint
        self.bucket = bucket
        self.access_key = access_key
        self.secret_key = secret_key
        self.max_frame_age_seconds = max_frame_age_seconds
        
        self.session = aioboto3.Session()
    
    def _get_client(self):
        """Get S3 client context manager."""
        return self.session.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )
    
    async def check_connection(self) -> bool:
        """Check if SeaweedFS is accessible."""
        try:
            async with self._get_client() as client:
                await client.head_bucket(Bucket=self.bucket)
                return True
        except Exception as e:
            logger.warning(f"SeaweedFS connection check failed: {e}")
            return False
    
    async def ensure_bucket(self):
        """Create bucket if it doesn't exist."""
        try:
            async with self._get_client() as client:
                try:
                    await client.head_bucket(Bucket=self.bucket)
                except:
                    await client.create_bucket(Bucket=self.bucket)
                    logger.info(f"Created bucket: {self.bucket}")
        except Exception as e:
            logger.error(f"Failed to ensure bucket: {e}")
            raise
    
    async def store_frame(
        self,
        entity_id: str,
        frame: np.ndarray,
        timestamp: Optional[int] = None
    ) -> str:
        """
        Store a cropped person frame.
        
        Args:
            entity_id: Entity identifier (person tracking ID)
            frame: OpenCV image (BGR numpy array)
            timestamp: Unix timestamp in milliseconds (auto-generated if not provided)
        
        Returns:
            S3 key of stored frame
        """
        if timestamp is None:
            timestamp = int(datetime.utcnow().timestamp() * 1000)
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        # S3 key: {entity_id}/frames/{timestamp}.jpg
        key = f"{entity_id}/frames/{timestamp}.jpg"
        
        try:
            async with self._get_client() as client:
                await client.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=frame_bytes,
                    ContentType='image/jpeg'
                )
            logger.debug(f"Stored frame: {key}")
            return key
        except Exception as e:
            logger.error(f"Failed to store frame {key}: {e}")
            raise
    
    async def get_frames(
        self,
        entity_id: str,
        max_frames: int = 20,
        max_age_seconds: Optional[int] = None
    ) -> list[tuple[np.ndarray, int]]:
        """
        Get frames for an entity, sorted by timestamp (oldest first).
        
        Args:
            entity_id: Entity identifier
            max_frames: Maximum number of frames to return
            max_age_seconds: Only return frames newer than this age
        
        Returns:
            List of (frame_image, timestamp) tuples
        """
        if max_age_seconds is None:
            max_age_seconds = self.max_frame_age_seconds
        
        prefix = f"{entity_id}/frames/"
        
        try:
            async with self._get_client() as client:
                # List all frames for this entity
                response = await client.list_objects_v2(
                    Bucket=self.bucket,
                    Prefix=prefix
                )
                
                if 'Contents' not in response:
                    return []
                
                # Filter and sort by timestamp
                cutoff_time = int((datetime.utcnow() - timedelta(seconds=max_age_seconds)).timestamp() * 1000)
                
                frame_keys = []
                for obj in response['Contents']:
                    key = obj['Key']
                    # Extract timestamp from filename
                    try:
                        filename = key.split('/')[-1]
                        timestamp = int(filename.replace('.jpg', ''))
                        
                        if timestamp >= cutoff_time:
                            frame_keys.append((key, timestamp))
                    except ValueError:
                        continue
                
                # Sort by timestamp (oldest first)
                frame_keys.sort(key=lambda x: x[1])
                
                # Limit to max_frames (take most recent)
                if len(frame_keys) > max_frames:
                    frame_keys = frame_keys[-max_frames:]
                
                # Fetch frames
                frames = []
                for key, timestamp in frame_keys:
                    try:
                        response = await client.get_object(
                            Bucket=self.bucket,
                            Key=key
                        )
                        body = await response['Body'].read()
                        
                        # Decode JPEG to numpy array
                        nparr = np.frombuffer(body, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            frames.append((frame, timestamp))
                    except Exception as e:
                        logger.warning(f"Failed to fetch frame {key}: {e}")
                        continue
                
                logger.debug(f"Fetched {len(frames)} frames for entity {entity_id}")
                return frames
                
        except Exception as e:
            logger.error(f"Failed to get frames for {entity_id}: {e}")
            return []
    
    async def delete_frames(self, entity_id: str) -> int:
        """
        Delete all frames for an entity.
        
        Args:
            entity_id: Entity identifier
        
        Returns:
            Number of frames deleted
        """
        prefix = f"{entity_id}/frames/"
        deleted_count = 0
        
        try:
            async with self._get_client() as client:
                # List all frames
                response = await client.list_objects_v2(
                    Bucket=self.bucket,
                    Prefix=prefix
                )
                
                if 'Contents' not in response:
                    return 0
                
                # Delete each frame
                for obj in response['Contents']:
                    try:
                        await client.delete_object(
                            Bucket=self.bucket,
                            Key=obj['Key']
                        )
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete {obj['Key']}: {e}")
                
                logger.info(f"Deleted {deleted_count} frames for entity {entity_id}")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to delete frames for {entity_id}: {e}")
            return deleted_count
    
    async def cleanup_old_frames(
        self,
        entity_id: str,
        max_frames: int = 20
    ) -> int:
        """
        Remove old frames beyond the rolling buffer limit.
        
        Args:
            entity_id: Entity identifier
            max_frames: Maximum frames to keep
        
        Returns:
            Number of frames deleted
        """
        prefix = f"{entity_id}/frames/"
        deleted_count = 0
        
        try:
            async with self._get_client() as client:
                # List all frames
                response = await client.list_objects_v2(
                    Bucket=self.bucket,
                    Prefix=prefix
                )
                
                if 'Contents' not in response:
                    return 0
                
                # Sort by timestamp
                frame_keys = []
                for obj in response['Contents']:
                    key = obj['Key']
                    try:
                        filename = key.split('/')[-1]
                        timestamp = int(filename.replace('.jpg', ''))
                        frame_keys.append((key, timestamp))
                    except ValueError:
                        continue
                
                frame_keys.sort(key=lambda x: x[1])
                
                # Delete oldest frames if over limit
                frames_to_delete = len(frame_keys) - max_frames
                if frames_to_delete > 0:
                    for key, _ in frame_keys[:frames_to_delete]:
                        try:
                            await client.delete_object(
                                Bucket=self.bucket,
                                Key=key
                            )
                            deleted_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to delete {key}: {e}")
                
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup frames for {entity_id}: {e}")
            return deleted_count
