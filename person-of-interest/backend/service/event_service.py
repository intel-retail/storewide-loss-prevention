"""Event Service — stores movement events and supports historical queries."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from backend.domain.entities.event import MovementEvent
from backend.domain.interfaces.repository import EventRepository

log = logging.getLogger("poi.service.event")


class EventService:
    """Handles movement event storage and historical search aggregation."""

    def __init__(self, event_repo: EventRepository) -> None:
        self._repo = event_repo

    def store_movement(
        self,
        object_id: str,
        timestamp: str,
        camera_id: str,
        region: str,
        poi_id: Optional[str] = None,
        embedding_ref: Optional[str] = None,
        thumbnail_path: Optional[str] = None,
    ) -> None:
        event = MovementEvent(
            object_id=object_id,
            timestamp=timestamp,
            camera_id=camera_id,
            region=region,
            poi_id=poi_id,
            embedding_reference=embedding_ref,
            thumbnail_path=thumbnail_path,
        )
        self._repo.store_event(event)

    def search_history(
        self,
        poi_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> dict:
        events = self._repo.get_events_for_poi(poi_id, start_time, end_time)
        if not events:
            return {
                "event_type": "poi_history_result",
                "poi_id": poi_id,
                "visits": [],
                "total_visits": 0,
                "search_stats": {"vectors_searched": 0, "query_latency_ms": 0},
            }

        # Aggregate events into visits (grouped by date)
        visits_by_date: dict[str, list[dict]] = defaultdict(list)
        for evt in events:
            ts = evt.get("timestamp", "")
            date = ts[:10] if len(ts) >= 10 else "unknown"
            visits_by_date[date].append(evt)

        # Collect all object_ids for region dwell lookup
        all_object_ids = {evt.get("object_id", "") for evt in events if evt.get("object_id")}

        visits = []
        for date, day_events in sorted(visits_by_date.items()):
            timestamps = [e["timestamp"] for e in day_events]
            cameras = list({e.get("camera_id", "") for e in day_events})
            regions = list({e.get("region", "") for e in day_events if e.get("region")})
            entry = min(timestamps)
            exit_time = max(timestamps) if len(timestamps) > 1 else None
            duration = None
            if exit_time and entry != exit_time:
                try:
                    from datetime import datetime
                    t0 = datetime.fromisoformat(entry.replace("Z", "+00:00"))
                    t1 = datetime.fromisoformat(exit_time.replace("Z", "+00:00"))
                    duration = (t1 - t0).total_seconds()
                except Exception:
                    pass

            # Thumbnail: prefer thumbnail_path persisted on matched events
            thumbnail = ""
            for evt in day_events:
                tp = evt.get("thumbnail_path", "")
                if tp:
                    thumbnail = tp
                    break
            # Fallback: try Redis thumbnail cache for any object_id in this day
            if not thumbnail:
                for evt in day_events:
                    obj_id = evt.get("object_id", "")
                    if obj_id:
                        cached_thumb = self._repo.get_thumbnail(obj_id)
                        if cached_thumb:
                            thumbnail = f"/api/v1/thumbnail/{obj_id}"
                            break

            # Enrich regions from dwell data (date-scoped)
            region_dwells: list[dict] = []
            for oid in all_object_ids:
                region_dwells.extend(
                    self._repo.get_region_dwells_for_object(oid, date_filter=date)
                )
            for dwell in region_dwells:
                rname = dwell.get("region_name", "")
                if rname and rname not in regions:
                    regions.append(rname)

            # Filter out camera-ID-only entries if real region names exist
            real_regions = [r for r in regions if not r.startswith("lp-camera") and not r.startswith("cam:")]
            display_regions = real_regions if real_regions else regions

            visits.append(
                {
                    "date": date,
                    "entry_time": entry,
                    "exit_time": exit_time,
                    "cameras_visited": cameras,
                    "regions": display_regions,
                    "region_name": display_regions[0] if display_regions else "",
                    "duration_sec": duration,
                    "region_dwells": [
                        {
                            "region_name": d.get("region_name", ""),
                            "entry_time": d.get("entry_time", ""),
                            "exit_time": d.get("exit_time", ""),
                            "dwell_sec": d.get("dwell_sec"),
                            "camera_id": d.get("camera_id", ""),
                        }
                        for d in region_dwells
                    ] if region_dwells else [],
                    "thumbnail": thumbnail,
                    "alert_id": "",
                }
            )

        return {
            "event_type": "poi_history_result",
            "poi_id": poi_id,
            "query_range": {"start": start_time or "", "end": end_time or ""},
            "visits": visits,
            "total_visits": len(visits),
            "search_stats": {
                "vectors_searched": len(events),
                "query_latency_ms": 0,
            },
        }

    def store_region_entry(self, object_id, timestamp, scene_id, region_id, region_name, camera_id=None,
                           entry_frame_key=None):
        """Record region entry — store presence key in Redis."""
        self._repo.store_region_presence(
            object_id, timestamp, scene_id, region_id, region_name, camera_id,
            entry_frame_key=entry_frame_key,
        )

    def store_region_exit(self, object_id, timestamp, scene_id, region_id, region_name,
                          exit_frame_key=None):
        """Record region exit — calculate dwell time and store."""
        entry_data = self._repo.get_region_presence(object_id, scene_id, region_id)
        dwell_sec = None
        entry_time = None
        if entry_data:
            # Prefer the region_name from entry (human-readable) over the exit fallback
            region_name = entry_data.get("region_name") or region_name
            entry_time = entry_data.get("first_seen")
            try:
                from datetime import datetime, timezone
                t0 = datetime.fromisoformat(entry_data["first_seen"].replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                dwell_sec = (t1 - t0).total_seconds()
            except Exception:
                pass
        self._repo.store_region_dwell(
            object_id, timestamp, scene_id, region_id, region_name, dwell_sec,
            entry_time=entry_time,
            camera_id=entry_data.get("camera_id") if entry_data else None,
            entry_frame_key=entry_data.get("entry_frame_key") if entry_data else None,
            exit_frame_key=exit_frame_key,
        )
        self._repo.delete_region_presence(object_id, scene_id, region_id)
