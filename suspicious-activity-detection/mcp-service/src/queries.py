"""Pure query logic over the durable log. No MCP or agent concerns here.

An "activity" is a flattened SAD violation event: the envelope's ref_id + ts_ms
merged with the payload fields.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import datetime, timezone
import os
from typing import Any
from zoneinfo import ZoneInfo

from events import EVENT_TYPE
from models import Activity, TrendCount

_MAX = 10_000
_SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}
_STORE_TIMEZONE = os.getenv("SAD_TIMEZONE", "Asia/Kolkata")

_STOPWORDS = {
    "all",
    "and",
    "any",
    "anything",
    "are",
    "for",
    "found",
    "how",
    "into",
    "often",
    "put",
    "record",
    "records",
    "related",
    "show",
    "station",
    "stations",
    "the",
    "which",
    "with",
}

_FOOD_SAFETY_ALIASES = (
    "food safety violation dropped drop floor item items food area put back returned return prep kitchen"
)


def _to_activity(event: Any) -> Activity:
    """Flatten an event envelope into an Activity row."""
    local_time = datetime.fromtimestamp(
        event.ts_ms / 1000, tz=timezone.utc
    ).astimezone(ZoneInfo(_STORE_TIMEZONE))
    timestamp = (
        f"{local_time.strftime('%B')} {local_time.day}, {local_time.year} at "
        f"{local_time.strftime('%I').lstrip('0')}:{local_time.strftime('%M:%S %p %Z')}"
    )
    return {
        "ref_id": event.ref_id,
        "ts_ms": event.ts_ms,
        "timestamp": timestamp,
        **event.payload,
    }


def parse_time_range(
    start_time: str | None,
    end_time: str | None,
    store_timezone: str,
) -> tuple[int | None, int | None]:
    """Convert ISO datetimes to epoch milliseconds using the configured store timezone."""
    zone = ZoneInfo(store_timezone)

    def parse(value: str | None) -> int | None:
        if value is None:
            return None
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=zone)
        return int(parsed.timestamp() * 1000)

    start_ms = parse(start_time)
    end_ms = parse(end_time)
    if start_ms is not None and end_ms is not None and start_ms > end_ms:
        raise ValueError("start_time must be earlier than or equal to end_time")
    return start_ms, end_ms


def _matches_text(activity: Activity, query: str | None) -> bool:
    if not query:
        return True
    needle = query.lower()
    text = _activity_search_text(activity)
    if needle in text:
        return True

    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", needle)
        if len(token) > 2 and token not in _STOPWORDS
    ]
    if not tokens:
        return True
    return all(token in text for token in tokens)


def _activity_search_text(activity: Activity) -> str:
    fields = (
        activity.get("event_name", ""),
        activity.get("use_case", ""),
        activity.get("zone", ""),
        activity.get("pose", ""),
        activity.get("description", ""),
        activity.get("frame", ""),
    )
    text = " ".join(str(field).lower() for field in fields)
    if activity.get("event_name") == "food_safety_violation":
        text = f"{text} {_FOOD_SAFETY_ALIASES}"
    return text


def all_activities(log: Any, limit: int = _MAX) -> list[Activity]:
    """All activities from the log, oldest first."""
    return [_to_activity(e) for e in log.read(event_type=EVENT_TYPE, limit=limit)]


def _iter_activities(
    log: Any,
    start_ms: int | None,
    end_ms: int | None,
) -> Iterator[Activity]:
    newest_first = start_ms is None and end_ms is None
    activities = [
        _to_activity(event)
        for event in log.read(
            event_type=EVENT_TYPE,
            limit=_MAX,
            start_ms=start_ms,
            end_ms=end_ms,
            newest_first=newest_first,
        )
    ]
    yield from reversed(activities) if newest_first else activities


def activity_by_zone(log: Any, zone: str, limit: int = _MAX) -> list[Activity]:
    """Activities filtered to a single zone."""
    events = log.read(event_type=EVENT_TYPE, limit=limit, newest_first=True)
    matches = [_to_activity(event) for event in events if event.payload.get("zone") == zone]
    return list(reversed(matches))


def activity_by_zone_timestamp(
    log: Any,
    zone: str,
    start_ms: int | None = None,
    end_ms: int | None = None,
    limit: int = _MAX,
) -> list[Activity]:
    """Activities for a zone within an optional epoch-ms range."""
    out: list[Activity] = []
    for a in _iter_activities(log, start_ms, end_ms):
        if a.get("zone") != zone:
            continue
        out.append(a)
        if len(out) >= limit:
            break
    return out


def retrospective_frame_search(
    log: Any,
    query: str | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
    zone: str | None = None,
    event_name: str | None = None,
    use_case: str | None = None,
    limit: int = _MAX,
) -> list[Activity]:
    """Search logged SAD events with frame references for retrospective review."""
    out: list[Activity] = []
    for a in _iter_activities(log, start_ms, end_ms):
        if zone and a.get("zone") != zone:
            continue
        if event_name and a.get("event_name") != event_name:
            continue
        if use_case and a.get("use_case") != use_case:
            continue
        if not _matches_text(a, query):
            continue
        out.append(a)
        if len(out) >= limit:
            break
    return out


def trend_counts(
    log: Any,
    start_ms: int | None = None,
    end_ms: int | None = None,
    event_name: str | None = None,
    use_case: str | None = None,
    limit: int = _MAX,
) -> list[TrendCount]:
    """Count matching events by station and shift."""
    buckets: dict[tuple[str, str], int] = {}
    matched = 0
    for a in _iter_activities(log, start_ms, end_ms):
        if event_name and a.get("event_name") != event_name:
            continue
        if use_case and a.get("use_case") != use_case:
            continue
        key = (a.get("station") or a.get("zone") or "unknown", a.get("shift") or "unknown")
        buckets[key] = buckets.get(key, 0) + 1
        matched += 1
        if matched >= limit:
            break
    return [
        {"station": station, "shift": shift, "count": count}
        for (station, shift), count in sorted(buckets.items())
    ]


def event_count(
    log: Any,
    start_ms: int | None = None,
    end_ms: int | None = None,
    zone: str | None = None,
    event_name: str | None = None,
    use_case: str | None = None,
    minimum_severity: str | None = None,
    limit: int = _MAX,
) -> int:
    """Count matching events; a minimum severity includes all higher ranks."""
    minimum_rank = _SEVERITY_RANK.get(minimum_severity.lower()) if minimum_severity else None
    count = 0
    for activity in _iter_activities(log, start_ms, end_ms):
        if zone and activity.get("zone") != zone:
            continue
        if event_name and activity.get("event_name") != event_name:
            continue
        if use_case and activity.get("use_case") != use_case:
            continue
        if minimum_rank is not None:
            rank = _SEVERITY_RANK.get(str(activity.get("severity", "")).lower(), 0)
            if rank < minimum_rank:
                continue
        count += 1
        if count >= limit:
            break
    return count


def all_zones(log: Any, limit: int = _MAX) -> list[str]:
    """Distinct zone names that have any activity."""
    return sorted({a["zone"] for a in all_activities(log, limit) if a.get("zone")})
