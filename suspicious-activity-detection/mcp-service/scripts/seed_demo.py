"""Seed kitchen food-safety violations into the SAD durable log.

Run once so the Hermes read tools (Get_all_activities, Get_activity_by_zone,
Search_retrospective_frames, Get_trend_counts, Get_all_zones) return sample data
during a local developer demo:

    .venv/bin/python scripts/seed_demo.py

Writes to the same persistent log the running MCP server reads (see config.py).
Re-runs are idempotent because each event carries a stable ref_id.
"""

from __future__ import annotations

import queries
from tools import ingest_alert, store

_EVENTS = [
    {
        "zone": "kitchen-prep",
        "pose": "floor_to_food_area",
        "severity": "critical",
        "camera_id": "lp-camera1",
        "object_id": "p-1001",
        "description": "Object picked from the prep floor and placed back in the food area",
        "ref_id": "seed-kitchen-1",
        "event_name": "food_safety_violation",
        "use_case": "kitchen",
        "frame": "s3://behavioral-frames/kitchen/seed-kitchen-1.jpg",
        "station": "prep",
        "shift": "lunch",
    },
    {
        "zone": "checkout-2",
        "pose": "item-conceal",
        "severity": "high",
        "camera_id": "lp-camera2",
        "object_id": "p-2001",
        "description": "Possible item concealment at self-checkout",
        "ref_id": "seed-retail-1",
        "event_name": "concealment",
        "use_case": "retail",
        "frame": "s3://behavioral-frames/retail/seed-retail-1.jpg",
        "station": "checkout",
        "shift": "lunch",
    },
]


def main() -> None:
    for event in _EVENTS:
        ingest_alert(**event)
    activities = queries.all_activities(store)
    print(f"seeded {len(activities)} activities across zones {queries.all_zones(store)}")


if __name__ == "__main__":
    main()
