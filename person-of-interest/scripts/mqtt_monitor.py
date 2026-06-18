#!/usr/bin/env python3
"""Monitor Scenescape MQTT topics for face detections and region events."""

import json
import sys
from datetime import datetime

import paho.mqtt.client as mqtt

BROKER_HOST = "localhost"
BROKER_PORT = 1883

TOPICS = [
    ("scenescape/data/camera/+", 0),
    ("scenescape/data/scene/#", 0),
    ("scenescape/event/region/+/+/objects", 0),
]

COLORS = {
    "camera": "\033[32m",    # green
    "face": "\033[36m",      # cyan
    "scene": "\033[35m",     # magenta
    "objects": "\033[33m",   # yellow
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
}


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"{COLORS['bold']}Connected to broker {BROKER_HOST}:{BROKER_PORT}{COLORS['reset']}")
        for topic, qos in TOPICS:
            client.subscribe(topic, qos)
            print(f"  Subscribed: {topic}")
        print(f"{COLORS['dim']}{'─' * 70}{COLORS['reset']}")
        print("Waiting for messages...\n")
    else:
        print(f"Connection failed (rc={rc})")
        sys.exit(1)


def on_message(client, userdata, msg):
    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    topic = msg.topic
    parts = topic.split("/")

    try:
        data = json.loads(msg.payload.decode())
    except (json.JSONDecodeError, UnicodeDecodeError):
        print(f"[{now}] {topic}  (binary payload, {len(msg.payload)} bytes)")
        return

    # Camera data topic: scenescape/data/camera/{camera_id}
    if parts[1] == "data" and parts[2] == "camera":
        objects = data.get("objects", {})
        # Skip frames with no detections to reduce noise
        if not any(objects.values()):
            return

        color = COLORS["camera"]
        camera_id = parts[3] if len(parts) > 3 else "?"
        timestamp = data.get("timestamp", "?")
        rate = data.get("rate", 0)

        print(f"{color}[{now}] CAMERA DATA  camera={camera_id}  ts={timestamp}  fps={rate:.1f}{COLORS['reset']}")
        for category, items in objects.items():
            print(f"  {category}: {len(items)} detected")
            for i, obj in enumerate(items[:5]):
                obj_id = obj.get("id", "?")
                conf = obj.get("confidence", 0)
                bbox = obj.get("bounding_box_px", obj.get("bbox", "?"))
                has_reid = "reid" in obj.get("metadata", {})
                reid_tag = " [reid]" if has_reid else ""
                print(f"    [{i}] id={obj_id}  conf={conf:.3f}  bbox={bbox}{reid_tag}")
        print()

    # Face detection topic: scenescape/data/scene/{scene_id}/face
    elif "face" in parts:
        color = COLORS["face"]
        scene_id = parts[3] if len(parts) > 3 else "?"
        timestamp = data.get("timestamp", "?")
        objects = data.get("objects", [])

        print(f"{color}[{now}] FACE DETECTION  scene={scene_id}  ts={timestamp}{COLORS['reset']}")
        print(f"  Objects: {len(objects)}")
        for i, obj in enumerate(objects):
            obj_id = obj.get("id", "?")
            category = obj.get("category", "?")
            bbox = obj.get("bounding_box", obj.get("bbox", "?"))
            reid = obj.get("reid", obj.get("embedding", None))
            reid_info = f"  reid_len={len(reid)}" if isinstance(reid, list) else ""
            print(f"    [{i}] id={obj_id}  category={category}  bbox={bbox}{reid_info}")

            # Print additional attributes if present
            for key in ["confidence", "world_coordinate", "velocity", "age"]:
                if key in obj:
                    print(f"         {key}={obj[key]}")

        print()

    # Scene data topic: scenescape/data/scene/{scene_id}/{thing_type}
    elif parts[1] == "data" and parts[2] == "scene" and "face" not in parts:
        color = COLORS["scene"]
        scene_id = parts[3] if len(parts) > 3 else "?"
        thing_type = parts[4] if len(parts) > 4 else "?"
        timestamp = data.get("timestamp", "?")
        objects = data.get("objects", [])

        print(f"{color}[{now}] SCENE DATA  scene={scene_id}  type={thing_type}  ts={timestamp}{COLORS['reset']}")
        print(f"  Objects: {len(objects)}")
        for i, obj in enumerate(objects[:5]):
            obj_id = obj.get("id", "?")
            category = obj.get("category", "?")
            world = obj.get("world_coordinate", obj.get("position", "?"))
            print(f"    [{i}] id={obj_id}  category={category}  world={world}")
        if len(objects) > 5:
            print(f"    ... (+{len(objects) - 5} more)")
        print()

    # Region event topic: scenescape/event/region/{scene_id}/{region_id}/objects
    elif "event" in parts and "objects" in parts:
        color = COLORS["objects"]
        scene_id = parts[3] if len(parts) > 3 else "?"
        region_id = parts[4] if len(parts) > 4 else "?"

        entered = data.get("entered", [])
        exited = data.get("exited", [])
        objects_in = data.get("objects", [])
        timestamp = data.get("timestamp", "?")

        print(f"{color}[{now}] REGION EVENT  scene={scene_id}  region={region_id}  ts={timestamp}{COLORS['reset']}")
        print(f"  Entered: {len(entered)}  Exited: {len(exited)}  Inside: {len(objects_in)}")

        if entered:
            ids = [str(o.get("id", o) if isinstance(o, dict) else o) for o in entered]
            print(f"    Entered IDs: {', '.join(ids)}")
        if exited:
            ids = [str(o.get("id", o) if isinstance(o, dict) else o) for o in exited]
            print(f"    Exited IDs:  {', '.join(ids)}")
        if objects_in:
            ids = [str(o.get("id", o) if isinstance(o, dict) else o) for o in objects_in[:10]]
            suffix = f" ... (+{len(objects_in)-10} more)" if len(objects_in) > 10 else ""
            print(f"    Inside IDs:  {', '.join(ids)}{suffix}")

        print()

    # Fallback — unknown pattern
    else:
        print(f"[{now}] {topic}")
        print(f"  {json.dumps(data, indent=2)[:500]}\n")


def main():
    print(f"\n{COLORS['bold']}Scenescape MQTT Monitor{COLORS['reset']}")
    print(f"Topics:")
    for topic, _ in TOPICS:
        print(f"  • {topic}")
    print()

    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id="mqtt-monitor",
    )
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(BROKER_HOST, BROKER_PORT)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    except ConnectionRefusedError:
        print(f"Cannot connect to {BROKER_HOST}:{BROKER_PORT} — is the broker running?")
        sys.exit(1)


if __name__ == "__main__":
    main()
