#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
UUID Monitor — SceneScape Person Detection Analyzer

Subscribes to SceneScape MQTT topics, tracks unique person UUIDs across
cameras, and captures annotated frames at the moment each new UUID is
first detected.

Usage:
    python3 uuid_monitor.py
    python3 uuid_monitor.py --duration 200 --output /tmp/captures
    MONITOR_DURATION=120 python3 uuid_monitor.py

Environment / CLI config:
    MQTT_BROKER       Broker host        (default: localhost)
    MQTT_PORT         Broker port        (default: 1883)
    MONITOR_DURATION  Run time seconds   (default: 100)
    OUTPUT_DIR        Frame output dir   (default: tools/debug_faces/uuid_captures)
"""

import argparse
import base64
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import paho.mqtt.client as mqtt

# ── Logging ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("uuid_monitor")

# ── Configuration ──────────────────────────────────────
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MONITOR_DURATION = int(os.getenv("MONITOR_DURATION", "100"))
OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR",
    str(Path(__file__).parent / "debug_faces" / "uuid_captures"),
)

DATA_TOPIC = "scenescape/data/camera/+"
IMAGE_TOPIC = "scenescape/image/camera/+"
CMD_TOPIC_FMT = "scenescape/cmd/camera/{camera}"

# ── State ──────────────────────────────────────────────
uuid_stats: dict = {}       # uuid -> {cameras, count, first_seen, last_seen}
captured_per_camera: dict = {}  # camera -> set of uuids already captured
pending_captures: dict = {} # camera -> set of uuids awaiting image
stats_lock = threading.Lock()
total_messages = 0
messages_with_persons = 0
running = True


def _uuid_key(camera: str, uid: str) -> str:
    return f"{camera}:{uid}"


def on_connect(client: mqtt.Client, _userdata, _flags, rc):
    if rc != 0:
        log.error("MQTT connection failed (rc=%d)", rc)
        return
    log.info("Connected to MQTT broker %s:%d", MQTT_BROKER, MQTT_PORT)
    client.subscribe(DATA_TOPIC)
    client.subscribe(IMAGE_TOPIC)
    log.info("Subscribed to %s and %s", DATA_TOPIC, IMAGE_TOPIC)


def on_data_message(client: mqtt.Client, _userdata, camera: str, persons: list):
    """Process a camera data message — track UUIDs and trigger image capture."""
    global total_messages, messages_with_persons

    with stats_lock:
        total_messages += 1
        if not persons:
            return
        messages_with_persons += 1

        new_uuids = []
        new_on_camera = []
        now = datetime.now(timezone.utc).isoformat()
        cam_captured = captured_per_camera.setdefault(camera, set())

        for person in persons:
            uid = str(person.get("id", "unknown"))
            confidence = person.get("confidence", 0.0)
            bbox = person.get("bounding_box_px", {})

            if uid not in uuid_stats:
                uuid_stats[uid] = {
                    "cameras": set(),
                    "count": 0,
                    "first_seen": now,
                    "last_seen": now,
                    "first_camera": camera,
                    "max_confidence": confidence,
                    "bbox_sample": bbox,
                }
                new_uuids.append(uid)
                log.info(
                    "NEW UUID %s on %s (confidence=%.2f, bbox=%s)",
                    uid, camera, confidence, bbox,
                )
            elif camera not in uuid_stats[uid]["cameras"]:
                log.info(
                    "UUID %s now seen on %s (previously on %s)",
                    uid, camera, ", ".join(uuid_stats[uid]["cameras"]),
                )

            # Track if this UUID is new on THIS camera
            if uid not in cam_captured:
                new_on_camera.append(uid)

            stats = uuid_stats[uid]
            stats["cameras"].add(camera)
            stats["count"] += 1
            stats["last_seen"] = now
            if confidence > stats["max_confidence"]:
                stats["max_confidence"] = confidence

        # Trigger annotated frame capture when any UUID is first seen on this camera
        if new_on_camera:
            cam_captured.update(new_on_camera)
            pending_captures.setdefault(camera, set()).update(new_on_camera)
            client.publish(CMD_TOPIC_FMT.format(camera=camera), "getimage")
            log.info("Triggered getimage on %s for UUIDs %s", camera, new_on_camera)


def on_image_message(_client: mqtt.Client, _userdata, camera: str, payload: dict,
                     output_dir: str):
    """Save annotated frame when received from DLStreamer."""
    with stats_lock:
        uuids_waiting = pending_captures.pop(camera, set())

    if not uuids_waiting:
        return

    image_b64 = payload.get("image", "")
    if not image_b64:
        log.warning("Empty image payload from %s", camera)
        return

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception:
        log.exception("Failed to decode image from %s", camera)
        return

    uuids_label = "_".join(sorted(uuids_waiting))
    filename = f"{camera}_uuid_{uuids_label}.jpg"
    filepath = Path(output_dir) / filename

    # Don't overwrite — append a counter if file exists
    if filepath.exists():
        counter = 1
        while filepath.exists():
            filename = f"{camera}_uuid_{uuids_label}_{counter}.jpg"
            filepath = Path(output_dir) / filename
            counter += 1

    filepath.write_bytes(image_bytes)
    size_kb = len(image_bytes) // 1024
    log.info("Saved annotated frame: %s (%dKB) for UUIDs %s", filename, size_kb, uuids_waiting)


def on_message(client: mqtt.Client, userdata, msg: mqtt.MQTTMessage):
    """Route incoming MQTT messages to appropriate handlers."""
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return

    topic = msg.topic
    output_dir = userdata.get("output_dir", OUTPUT_DIR)

    if topic.startswith("scenescape/data/camera/"):
        camera = payload.get("id", topic.split("/")[-1])
        persons = payload.get("objects", {}).get("person", [])
        on_data_message(client, userdata, camera, persons)

    elif topic.startswith("scenescape/image/camera/"):
        camera = topic.split("/")[-1]
        on_image_message(client, userdata, camera, payload, output_dir)


def print_progress():
    """Print a live progress line."""
    with stats_lock:
        n_uuids = len(uuid_stats)
        cameras_with_ids = {}
        for uid, s in uuid_stats.items():
            for cam in s["cameras"]:
                cameras_with_ids.setdefault(cam, set()).add(uid)

    parts = [f"UUIDs: {n_uuids}"]
    for cam in sorted(cameras_with_ids):
        ids = sorted(cameras_with_ids[cam], key=lambda x: int(x) if x.isdigit() else 0)
        parts.append(f"{cam}: {ids}")
    sys.stdout.write(f"\r  {' | '.join(parts)}    ")
    sys.stdout.flush()


def print_summary(output_dir: str, elapsed: float):
    """Print final summary table."""
    print("\n")
    print("=" * 70)
    print("  UUID Monitor — Final Summary")
    print("=" * 70)
    print(f"  Duration:              {elapsed:.0f}s")
    print(f"  Total MQTT messages:   {total_messages}")
    print(f"  Messages with persons: {messages_with_persons}")
    print(f"  Unique UUIDs:          {len(uuid_stats)}")
    print()

    if uuid_stats:
        header = f"{'UUID':<8} {'Camera(s)':<25} {'Count':<8} {'MaxConf':<8} {'First Seen'}"
        print(f"  {header}")
        print(f"  {'-' * len(header)}")
        for uid in sorted(uuid_stats, key=lambda x: int(x) if x.isdigit() else 0):
            s = uuid_stats[uid]
            cams = ", ".join(sorted(s["cameras"]))
            print(
                f"  {uid:<8} {cams:<25} {s['count']:<8} {s['max_confidence']:<8.2f} {s['first_seen']}"
            )
        print()

    # List saved frames
    out = Path(output_dir)
    frames = sorted(out.glob("*.jpg"))
    if frames:
        print("  Captured frames:")
        for f in frames:
            print(f"    {f.name} ({f.stat().st_size // 1024}KB)")
    else:
        print("  No annotated frames captured.")

    # Save summary JSON
    summary = {
        "duration_seconds": elapsed,
        "total_messages": total_messages,
        "messages_with_persons": messages_with_persons,
        "unique_uuids": len(uuid_stats),
        "uuids": {
            uid: {
                "cameras": sorted(s["cameras"]),
                "detection_count": s["count"],
                "max_confidence": s["max_confidence"],
                "first_seen": s["first_seen"],
                "last_seen": s["last_seen"],
                "first_camera": s["first_camera"],
            }
            for uid, s in uuid_stats.items()
        },
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary saved to: {summary_path}")
    print("=" * 70)


def main():
    global running

    parser = argparse.ArgumentParser(description="SceneScape UUID Monitor")
    parser.add_argument(
        "--duration", type=int,
        default=MONITOR_DURATION,
        help=f"Monitor duration in seconds (default: {MONITOR_DURATION})",
    )
    parser.add_argument(
        "--output", type=str,
        default=OUTPUT_DIR,
        help=f"Output directory for captured frames (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--broker", type=str,
        default=MQTT_BROKER,
        help=f"MQTT broker host (default: {MQTT_BROKER})",
    )
    parser.add_argument(
        "--port", type=int,
        default=MQTT_PORT,
        help=f"MQTT broker port (default: {MQTT_PORT})",
    )
    args = parser.parse_args()

    # Prepare output directory
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    # Clean previous captures
    for f in out.glob("*.jpg"):
        f.unlink()
    for f in out.glob("*.json"):
        f.unlink()

    # Graceful shutdown
    def handle_signal(_sig, _frame):
        global running
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # MQTT client
    client = mqtt.Client(userdata={"output_dir": str(out)})
    client.on_connect = on_connect
    client.on_message = on_message

    log.info("Connecting to %s:%d ...", args.broker, args.port)
    try:
        client.connect(args.broker, args.port, keepalive=60)
    except Exception:
        log.exception("Cannot connect to MQTT broker")
        sys.exit(1)

    client.loop_start()

    print(f"\n  Monitoring SceneScape for {args.duration}s ...")
    print(f"  Output: {out}\n")

    start = time.time()
    try:
        while running and (time.time() - start) < args.duration:
            print_progress()
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        client.loop_stop()
        client.disconnect()

    elapsed = time.time() - start
    print_summary(str(out), elapsed)


if __name__ == "__main__":
    main()
