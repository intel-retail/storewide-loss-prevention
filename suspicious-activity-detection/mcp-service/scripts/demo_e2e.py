"""End-to-end demo of the SAD read tools (no MCP client needed).

    python scripts/demo_e2e.py     # after `pip install -e .`
"""

from __future__ import annotations

from tools import ingest_alert, svc


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

    print("Get_all_activities:", len(svc._read_tools["Get_all_activities"].fn()), "(idempotent = 3)")
    print("Get_all_zones:", svc._read_tools["Get_all_zones"].fn())
    print("Get_activity_by_zone(kitchen-prep):",
          len(svc._read_tools["Get_activity_by_zone"].fn("kitchen-prep")))

    print("Get_activity_by_zone_timestamp(kitchen-prep):",
          len(svc._read_tools["Get_activity_by_zone_timestamp"].fn("kitchen-prep")))
    print("Search_retrospective_frames(floor/kitchen):",
          len(svc._read_tools["Search_retrospective_frames"].fn(query="floor", use_case="kitchen")))
    print("Get_trend_counts(kitchen):",
          svc._read_tools["Get_trend_counts"].fn(use_case="kitchen"))

    d = svc.describe()
    print("describe read tools:", list(d["read_tools"]))
    print("describe act tools:", d["act_tools"])
    print("\nOK — SAD MCP (standard package) works end-to-end.")


if __name__ == "__main__":
    main()
