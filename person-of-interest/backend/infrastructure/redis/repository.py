"""Redis-backed repository implementations."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from backend.core.config import get_config
from backend.domain.entities.event import MovementEvent
from backend.domain.entities.poi import POI, POIStatus, ReferenceImage, Severity
from backend.domain.interfaces.repository import (
    CacheRepository,
    EmbeddingMappingRepository,
    EventRepository,
    POIRepository,
)
from backend.infrastructure.redis.client import RedisClient

log = logging.getLogger("poi.redis.repo")


class RedisPOIRepository(POIRepository):
    """Stores POI metadata in Redis hashes."""

    PREFIX = "poi:"

    def __init__(self) -> None:
        self._r = RedisClient().client

    def save(self, poi: POI) -> None:
        key = f"{self.PREFIX}{poi.poi_id}"
        self._r.set(key, json.dumps(poi.to_dict()))
        self._r.sadd("poi:index", poi.poi_id)

    def get(self, poi_id: str) -> Optional[POI]:
        raw = self._r.get(f"{self.PREFIX}{poi_id}")
        if raw is None:
            return None
        return self._deserialize(json.loads(raw))

    def list_all(self) -> list[POI]:
        poi_ids = self._r.smembers("poi:index")
        pois = []
        for pid in sorted(poi_ids, reverse=True):
            poi = self.get(pid)
            if poi:
                pois.append(poi)
        return pois

    def delete(self, poi_id: str) -> bool:
        key = f"{self.PREFIX}{poi_id}"
        deleted = self._r.delete(key)
        self._r.srem("poi:index", poi_id)
        return deleted > 0

    def update_status(self, poi_id: str, status: str) -> None:
        poi = self.get(poi_id)
        if poi:
            poi.status = POIStatus(status)
            self.save(poi)

    @staticmethod
    def _deserialize(data: dict) -> POI:
        return POI(
            poi_id=data["poi_id"],
            severity=Severity(data["severity"]),
            notes=data.get("notes", ""),
            reference_images=[
                ReferenceImage(
                    source=img.get("source", "uploaded_image"),
                    embedding_id=img.get("embedding_id", ""),
                    vector_dim=img.get("vector_dim", 256),
                    image_path=img.get("image_path", ""),
                )
                for img in data.get("reference_images", [])
            ],
            status=POIStatus(data.get("status", "active")),
            enrolled_by=data.get("enrolled_by", "system"),
            created_at=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            embedding_ids=[img.get("embedding_id", "") for img in data.get("reference_images", [])],
        )


class RedisCacheRepository(CacheRepository):
    """Cache-Aside: object_id → poi_id mapping."""

    PREFIX = "object:"

    def __init__(self) -> None:
        self._r = RedisClient().client

    def get_poi_for_object(self, object_id: str) -> Optional[str]:
        raw = self._r.get(f"{self.PREFIX}{object_id}")
        if raw is None:
            return None
        text = raw.decode() if isinstance(raw, bytes) else raw
        try:
            data = json.loads(text)
            return data.get("poi_id") if isinstance(data, dict) else text
        except (json.JSONDecodeError, TypeError):
            # Legacy colon-separated format: "poi_id:similarity"
            if ":" in text:
                return text.split(":", 1)[0]
            return text

    def get_similarity_for_object(self, object_id: str) -> Optional[float]:
        """Return the cached similarity score for object_id, or None."""
        raw = self._r.get(f"{self.PREFIX}{object_id}")
        if raw is None:
            return None
        text = raw.decode() if isinstance(raw, bytes) else raw
        try:
            data = json.loads(text)
            return float(data["similarity"]) if isinstance(data, dict) and "similarity" in data else None
        except (json.JSONDecodeError, TypeError, ValueError):
            # Legacy colon-separated format
            if ":" in text:
                try:
                    return float(text.split(":", 1)[1])
                except ValueError:
                    return None
            return None

    def set_poi_for_object(self, object_id: str, poi_id: str, ttl: int = 300, similarity: float = 0.0) -> None:
        value = json.dumps({"poi_id": poi_id, "similarity": similarity})
        self._r.setex(f"{self.PREFIX}{object_id}", ttl, value)

    def delete_object(self, object_id: str) -> None:
        self._r.delete(f"{self.PREFIX}{object_id}")


class RedisEventRepository(EventRepository):
    """Stores movement events and alerts in Redis."""

    def __init__(self) -> None:
        self._r = RedisClient().client
        self._cfg = get_config()

    def store_event(self, event: MovementEvent) -> None:
        key = f"event:{event.object_id}:{event.timestamp}"
        self._r.setex(
            key,
            self._cfg.appearance_ttl_days * 86400,
            json.dumps(event.to_dict()),
        )
        if event.poi_id:
            self._r.sadd(f"events:poi:{event.poi_id}", key)
            self._r.expire(f"events:poi:{event.poi_id}", self._cfg.appearance_ttl_days * 86400)

    def get_events_for_poi(
        self, poi_id: str, start_time: Optional[str] = None, end_time: Optional[str] = None
    ) -> list[dict]:
        keys = self._r.smembers(f"events:poi:{poi_id}")
        events = []
        for key in keys:
            raw = self._r.get(key)
            if raw:
                evt = json.loads(raw)
                ts = evt.get("timestamp", "")
                if start_time and ts < start_time:
                    continue
                if end_time and ts > end_time:
                    continue
                events.append(evt)
        events.sort(key=lambda e: e.get("timestamp", ""))
        return events

    def store_alert(self, alert: dict) -> None:
        alert_id = alert.get("alert_id", "")
        self._r.lpush("alerts:recent", json.dumps(alert))
        self._r.ltrim("alerts:recent", 0, 999)
        self._r.set(f"alert:{alert_id}", json.dumps(alert))
        self._r.expire(f"alert:{alert_id}", self._cfg.appearance_ttl_days * 86400)

    def get_recent_alerts(self, limit: int = 50) -> list[dict]:
        raw_list = self._r.lrange("alerts:recent", 0, limit - 1)
        return [json.loads(r) for r in raw_list]

    def clear_alerts(self) -> int:
        """Delete all alert records and the recent-alerts list. Returns count deleted."""
        deleted = 0
        self._r.delete("alerts:recent")
        deleted += 1
        cursor = 0
        while True:
            cursor, keys = self._r.scan(cursor, match="alert:*", count=200)
            if keys:
                self._r.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break
        return deleted

    def is_alert_sent(self, object_id: str) -> bool:
        return self._r.exists(f"alert:sent:{object_id}") > 0

    def mark_alert_sent(self, object_id: str, ttl: int = 300) -> None:
        self._r.setex(f"alert:sent:{object_id}", ttl, "1")

    def store_thumbnail(self, object_id: str, b64_jpeg: str, ttl: int = 3600) -> None:
        """Store base64 JPEG thumbnail for an object. TTL defaults to 1 hour."""
        self._r.setex(f"thumbnail:{object_id}", ttl, b64_jpeg)

    def get_thumbnail(self, object_id: str) -> Optional[str]:
        raw = self._r.get(f"thumbnail:{object_id}")
        return raw.decode() if isinstance(raw, bytes) else raw

    def claim_thumbnail(self, object_id: str, ttl: int = 30) -> bool:
        """Atomically claim the right to capture thumbnail (NX). Returns True if claim acquired."""
        return bool(self._r.set(f"thumbnail:claiming:{object_id}", "1", ex=ttl, nx=True))

    def store_region_presence(self, object_id, timestamp, scene_id, region_id, region_name, camera_id=None,
                               entry_frame_key=None):
        """Store region entry presence record."""
        import json
        key = f"region:presence:{scene_id}:{region_id}:{object_id}"
        data = {
            "first_seen": timestamp,
            "region_name": region_name,
            "camera_id": camera_id or "",
            "entry_frame_key": entry_frame_key or "",
        }
        self._r.setex(key, 86400, json.dumps(data))  # 24h TTL — must outlive any realistic dwell

    def get_region_presence(self, object_id, scene_id, region_id):
        """Get region presence record."""
        import json
        key = f"region:presence:{scene_id}:{region_id}:{object_id}"
        raw = self._r.get(key)
        return json.loads(raw) if raw else None

    def delete_region_presence(self, object_id, scene_id, region_id):
        """Delete region presence record after exit."""
        key = f"region:presence:{scene_id}:{region_id}:{object_id}"
        self._r.delete(key)

    def scan_all_region_presence(self) -> list[dict]:
        """Return all active region presence records.

        Each item: {"object_id": str, "scene_id": str, "region_id": str, ...data}
        Used to rebuild in-memory state after container restart.
        """
        import json
        results: list[dict] = []
        cursor = 0
        while True:
            cursor, keys = self._r.scan(cursor, match="region:presence:*", count=500)
            for key in keys:
                raw = self._r.get(key)
                if not raw:
                    continue
                # Key format: region:presence:{scene_id}:{region_id}:{object_id}
                parts = (key if isinstance(key, str) else key.decode()).split(":", 4)
                if len(parts) < 5:
                    continue
                _, _, scene_id, region_id, object_id = parts
                data = json.loads(raw)
                data.update({"object_id": object_id, "scene_id": scene_id, "region_id": region_id})
                results.append(data)
            if cursor == 0:
                break
        return results

    def store_region_dwell(self, object_id, timestamp, scene_id, region_id, region_name, dwell_sec=None,
                           entry_time=None, camera_id=None, entry_frame_key=None, exit_frame_key=None):
        """Store region dwell record (entry + exit + duration + frame keys)."""
        import json
        from datetime import datetime, timezone
        date_key = timestamp[:10] if len(timestamp) >= 10 else datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = f"region:dwell:{object_id}:{scene_id}:{region_id}:{date_key}"
        data = {
            "object_id": object_id,
            "scene_id": scene_id,
            "region_id": region_id,
            "region_name": region_name,
            "entry_time": entry_time or "",
            "exit_time": timestamp,
            "dwell_sec": dwell_sec,
            "camera_id": camera_id or "",
            "entry_frame_key": entry_frame_key or "",
            "exit_frame_key": exit_frame_key or "",
        }
        self._r.setex(key, 86400 * 7, json.dumps(data))  # 7 day TTL

    def store_zone_frame(self, frame_key: str, b64_jpeg: str, ttl: int = 86400 * 7) -> None:
        """Store a base64-encoded JPEG frame for a zone entry or exit event."""
        self._r.setex(frame_key, ttl, b64_jpeg)

    def get_zone_frame(self, frame_key: str) -> Optional[str]:
        """Return the stored base64 JPEG for a zone frame key, or None if expired."""
        val = self._r.get(frame_key)
        return val.decode() if isinstance(val, bytes) else val

    def claim_track_entry(self, object_id: str, ttl: int = 120) -> bool:
        """Atomically claim the entry frame slot for a track (NX).

        Returns True only the first time this track_id is seen within the TTL
        window — used to capture the entry frame once per tracker track lifetime.
        TTL must match track_seen_ttl (default 120s), NOT the 7-day data TTL,
        so that when the tracker recycles an integer ID the new person's entry
        frame is not blocked by a stale key from a previous occupant.
        """
        return bool(self._r.set(f"track:entry:claimed:{object_id}", "1", ex=ttl, nx=True))

    def store_track_frame(self, object_id: str, event_type: str, b64_jpeg: str,
                          ttl: int = 86400 * 7) -> str:
        """Store a base64 JPEG as the entry or last_seen frame for a track.

        event_type: "entry" (written once, NX) or "last_seen" (always overwritten).
        Returns the Redis key.
        """
        key = f"track:frame:{object_id}:{event_type}"
        self._r.setex(key, ttl, b64_jpeg)
        return key

    def get_track_frame_key(self, object_id: str, event_type: str) -> str:
        """Return the Redis key for a track frame without fetching the value."""
        return f"track:frame:{object_id}:{event_type}"

    def track_frame_exists(self, object_id: str, event_type: str) -> bool:
        """Return True if a frame is stored for this track/event_type."""
        return bool(self._r.exists(f"track:frame:{object_id}:{event_type}"))

    def set_reid_meta(self, global_uuid: str, metadata: dict, ttl: int = 120) -> None:
        """Store reid metadata for a global UUID (for MCP tool observability)."""
        self._r.setex(f"reid:meta:{global_uuid}", ttl, json.dumps(metadata))

    # ── UUID ↔ camera visibility + resolution ────────────────────────────

    def store_uuid_visibility(
        self, camera_id: str, uuids: list[str], ttl: int = 10,
    ) -> None:
        """Store which UUIDs are currently visible on a camera.

        Called from the external topic handler.  The list is a snapshot
        from SceneScape's controller — replaced every frame (~1-3 Hz).
        Stored as a Redis hash with the current epoch timestamp so
        readers can enforce freshness.
        """
        import time
        key = f"uuid:visible:{camera_id}"
        pipe = self._r.pipeline()
        pipe.delete(key)
        if uuids:
            now_str = str(time.time())
            mapping = {uid: now_str for uid in uuids}
            pipe.hset(key, mapping=mapping)
            pipe.expire(key, ttl)
        pipe.execute()

    def get_visible_uuids(
        self, camera_id: str, max_age_s: float = 5.0,
    ) -> list[str]:
        """Return UUIDs currently visible on a camera (freshness-checked).

        Only returns UUIDs whose visibility was updated within max_age_s
        seconds.  Returns empty list if data is stale or absent.
        """
        import time
        key = f"uuid:visible:{camera_id}"
        raw_map = self._r.hgetall(key)
        if not raw_map:
            return []

        now = time.time()
        result: list[str] = []
        for uid_bytes, ts_bytes in raw_map.items():
            uid = uid_bytes.decode() if isinstance(uid_bytes, bytes) else uid_bytes
            try:
                ts = float(ts_bytes)
            except (ValueError, TypeError):
                continue
            if now - ts <= max_age_s:
                result.append(uid)
        return result

    def store_camid_uuid_mapping(
        self, camera_id: str, person_int_id: int, uuid: str, ttl: int = 600,
    ) -> None:
        """Cache a confirmed camera-local person ID → UUID mapping.

        Once we establish that cam:Camera_01:1 is UUID X, this cache
        allows all subsequent detections for person 1 on Camera_01 to
        use the same UUID without re-resolving visibility each frame.
        TTL matches track_seen_ttl (default 600s).
        """
        key = f"uuid_map:{camera_id}:{person_int_id}"
        self._r.setex(key, ttl, uuid)

    def get_uuid_for_camid(
        self, camera_id: str, person_int_id: int,
    ) -> Optional[str]:
        """Look up cached UUID for a camera-local person ID."""
        raw = self._r.get(f"uuid_map:{camera_id}:{person_int_id}")
        if raw is None:
            return None
        return raw.decode() if isinstance(raw, bytes) else raw

    def clear_camid_uuid_mapping(
        self, camera_id: str, person_int_id: int,
    ) -> None:
        """Clear the cached UUID for a camera-local person ID.

        Called when continuity check fails (different person reusing same ID).
        """
        self._r.delete(f"uuid_map:{camera_id}:{person_int_id}")

    # ── UUID ↔ camera bbox mapping (populated from regulated scene topic) ──

    def store_uuid_camera_bounds(
        self, camera_id: str, uuid_bounds: dict[str, dict], ttl: int = 15,
    ) -> None:
        """Store all UUID→bbox mappings for a camera as a single Redis hash.

        uuid_bounds: {uuid: {"x": int, "y": int, "width": int, "height": int}}
        TTL of 15s allows for MQTT jitter/lag while the regulated scene
        topic refreshes at ~3Hz. Previous 5s was too aggressive and caused
        UUID resolution failures during brief network stalls.
        """
        key = f"uuid:cam_bounds:{camera_id}"
        pipe = self._r.pipeline()
        pipe.delete(key)
        if uuid_bounds:
            mapping = {uid: json.dumps(bbox) for uid, bbox in uuid_bounds.items()}
            pipe.hset(key, mapping=mapping)
            pipe.expire(key, ttl)
        pipe.execute()

    def get_uuid_for_camera_bbox(
        self, camera_id: str, bbox: dict, iou_threshold: float = 0.3,
    ) -> Optional[str]:
        """Find the SceneScape global UUID whose camera_bounds best overlaps with bbox.

        Returns the UUID with highest IoU above iou_threshold, or None.
        """
        key = f"uuid:cam_bounds:{camera_id}"
        raw_map = self._r.hgetall(key)
        if not raw_map:
            return None

        best_uuid: Optional[str] = None
        best_iou = 0.0

        for uid_bytes, bbox_bytes in raw_map.items():
            uid = uid_bytes.decode() if isinstance(uid_bytes, bytes) else uid_bytes
            try:
                ref = json.loads(bbox_bytes)
            except (json.JSONDecodeError, ValueError):
                continue
            # Skip projected bounds — they use 3D-projected coordinates
            # that don't match the camera topic's pixel-space bboxes.
            if ref.get("projected"):
                continue
            # Skip invalid bounds (negative or zero dimensions)
            if ref.get("width", 0) <= 0 or ref.get("height", 0) <= 0:
                continue
            iou = _compute_iou(bbox, ref)
            if iou > best_iou:
                best_iou = iou
                best_uuid = uid

        if best_iou >= iou_threshold:
            return best_uuid
        return None

    def get_region_dwells_for_object(self, object_id: str, date_filter: Optional[str] = None) -> list[dict]:
        """Return region dwell records for an object, optionally filtered by date.

        Args:
            object_id: The object ID to look up.
            date_filter: If provided, only return dwells matching this date (YYYY-MM-DD).
        """
        pattern = f"region:dwell:{object_id}:*"
        keys: list = []
        cursor = 0
        while True:
            cursor, batch = self._r.scan(cursor, match=pattern, count=100)
            keys.extend(batch)
            if cursor == 0:
                break
        dwells = []
        for key in keys:
            raw = self._r.get(key)
            if raw:
                dwell = json.loads(raw)
                if date_filter:
                    exit_ts = dwell.get("exit_time", "")
                    dwell_date = exit_ts[:10] if len(exit_ts) >= 10 else ""
                    if dwell_date != date_filter:
                        continue
                dwells.append(dwell)
        return dwells

    def get_track_poi_counts(self, track_id: str) -> dict[str, int]:
        """Return {poi_id: event_count} for all POIs that have events on this track."""
        pattern = f"event:{track_id}:*"
        cursor = 0
        counts: dict[str, int] = {}
        while True:
            cursor, batch = self._r.scan(cursor, match=pattern, count=200)
            for key in batch:
                try:
                    raw = self._r.get(key)
                    if raw:
                        evt = json.loads(raw)
                        pid = evt.get("poi_id", "")
                        if pid:
                            counts[pid] = counts.get(pid, 0) + 1
                except Exception:
                    pass
            if cursor == 0:
                break
        return counts

    # Kept for backwards-compatibility; no longer called from the consumer.
    def set_reid_matched(self, camera_id: str, global_uuid: str, metadata: dict, ttl: int = 15) -> None:
        """Deprecated: use set_reid_meta instead. Writes gate key + meta."""
        self._r.setex(f"reid:matched:{camera_id}", ttl, global_uuid)
        self._r.setex(f"reid:meta:{global_uuid}", ttl * 8, json.dumps(metadata))

    def get_reid_matched_uuid(self, camera_id: str) -> Optional[str]:
        """Deprecated: reid gate removed. Kept for MCP tool compatibility."""
        val = self._r.get(f"reid:matched:{camera_id}")
        return val.decode() if isinstance(val, bytes) else val

    def set_match_metadata(self, object_id: str, metadata: dict, ttl: int = 3600) -> None:
        """Persist FAISS+reid metadata for an object (1h TTL)."""
        self._r.setex(f"match:meta:{object_id}", ttl, json.dumps(metadata))

    def get_match_metadata(self, object_id: str) -> Optional[dict]:
        raw = self._r.get(f"match:meta:{object_id}")
        return json.loads(raw) if raw else None


class RedisEmbeddingMappingRepository(EmbeddingMappingRepository):
    """Maps FAISS internal integer IDs to POI string IDs."""

    def __init__(self) -> None:
        self._r = RedisClient().client

    def map_faiss_to_poi(self, faiss_id: int, poi_id: str) -> None:
        self._r.set(f"faiss2poi:{faiss_id}", poi_id)
        self._r.sadd(f"poi2faiss:{poi_id}", str(faiss_id))

    def get_poi_for_faiss(self, faiss_id: int) -> Optional[str]:
        return self._r.get(f"faiss2poi:{faiss_id}")

    def remove_mappings_for_poi(self, poi_id: str) -> None:
        faiss_ids = self._r.smembers(f"poi2faiss:{poi_id}")
        pipe = self._r.pipeline()
        for fid in faiss_ids:
            pipe.delete(f"faiss2poi:{fid}")
        pipe.delete(f"poi2faiss:{poi_id}")
        pipe.execute()


# ── Module-level helpers ────────────────────────────────────────────────────


def _normalize_bbox(bbox) -> dict:
    """Convert any bbox format to {x, y, width, height} dict.

    Handles: [x1, y1, x2, y2] list, {x, y, width, height} dict,
    and {x1, y1, x2, y2} or {left, top, right, bottom} dicts.
    Returns a dict with keys x, y, width, height. Returns zero-bbox on failure.
    """
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        return {"x": int(min(x1, x2)), "y": int(min(y1, y2)),
                "width": int(abs(x2 - x1)), "height": int(abs(y2 - y1))}
    if isinstance(bbox, dict):
        if "width" in bbox and "height" in bbox:
            return bbox
        if "x2" in bbox and "y2" in bbox:
            x1 = float(bbox.get("x1", bbox.get("x", 0)))
            y1 = float(bbox.get("y1", bbox.get("y", 0)))
            x2 = float(bbox["x2"])
            y2 = float(bbox["y2"])
            return {"x": int(min(x1, x2)), "y": int(min(y1, y2)),
                    "width": int(abs(x2 - x1)), "height": int(abs(y2 - y1))}
    return {"x": 0, "y": 0, "width": 0, "height": 0}


def _compute_iou(a, b) -> float:
    """Compute Intersection-over-Union for two bounding boxes.

    Accepts any bbox format: [x1,y1,x2,y2] list or {x,y,width,height} dict.
    """
    a = _normalize_bbox(a)
    b = _normalize_bbox(b)

    ax1 = a.get("x", 0)
    ay1 = a.get("y", 0)
    ax2 = ax1 + a.get("width", 0)
    ay2 = ay1 + a.get("height", 0)

    bx1 = b.get("x", 0)
    by1 = b.get("y", 0)
    bx2 = bx1 + b.get("width", 0)
    by2 = by1 + b.get("height", 0)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0
