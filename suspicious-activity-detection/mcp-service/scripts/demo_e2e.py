"""End-to-end demo of the SAD read tools (no MCP client needed).

    python scripts/demo_e2e.py     # after `pip install -e .`
"""

from __future__ import annotations

from tools import ingest_alert, store


def main() -> None:
    # SAD pipeline publishes activities (the MQTT hand-off seam).
    ingest_alert("kitchen-prep", "item-drop-return", "high", "cam-3", "p-1001",
                         "Object dropped on floor and returned to prep area", ref_id="mqtt-1",
                         event_name="food_safety_violation", use_case="kitchen",
                         frame="s3://behavioral-frames/kitchen/mqtt-1.jpg", station="prep", shift="lunch")
    ingest_alert("kitchen-prep", "reach-over", "medium", "cam-3", "p-1002",
                         "Reach-over sneeze guard", ref_id="mqtt-2",
                         event_name="food_safety_violation", use_case="kitchen",
                         frame="s3://behavioral-frames/kitchen/mqtt-2.jpg", station="prep", shift="lunch")
    ingest_alert("checkout-2", "item-conceal", "high", "cam-7", "p-1003",
                         "Possible concealment at self-checkout", ref_id="mqtt-3",
                         event_name="concealment", use_case="retail",
                         frame="s3://behavioral-frames/retail/mqtt-3.jpg", station="checkout", shift="lunch")
    ingest_alert("checkout-2", "item-conceal", "high", "cam-7", "p-1003",
                         "Duplicate MQTT redelivery", ref_id="mqtt-3",
                         event_name="concealment", use_case="retail",
                         frame="s3://behavioral-frames/retail/mqtt-3.jpg", station="checkout", shift="lunch")  # idempotent

    acts = queries.all_activities(store)
    print("Get_all_activities:", len(acts), "(idempotent = 3)")
    print("Get_all_zones:", queries.all_zones(store))
    print("Get_activity_by_zone(kitchen-prep):",
          len(queries.activity_by_zone(store, "kitchen-prep")))

    print("Get_activity_by_zone_timestamp(kitchen-prep):",
          len(queries.activity_by_zone_timestamp(store, "kitchen-prep")))
    print("Search_retrospective_frames(floor/kitchen):",
          len(queries.retrospective_frame_search(store, query="floor", use_case="kitchen")))
    print("Get_trend_counts(kitchen):",
          queries.trend_counts(store, use_case="kitchen"))
    print("\nOK — SAD queries and service-owned SQLite store work end-to-end.")


if __name__ == "__main__":
    main()
