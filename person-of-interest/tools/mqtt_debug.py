#!/usr/bin/env python3
"""
Simple MQTT dump tool for Scenescape.

- Subscribes to camera + scene topics
- Saves each message as JSON
- Organizes into folders per topic
"""

import json
import logging
import os
import signal
import threading
from datetime import datetime
from pathlib import Path

import paho.mqtt.client as mqtt

# ── CONFIG ─────────────────────────────────────────────
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

# Topics
CAMERA_TOPICS = [
    "scenescape/data/camera/Camera_01",
    "scenescape/data/camera/Camera_02",
]

SCENE_TOPIC = "scenescape/regulated/scene/+"

OUTPUT_DIR = Path("./mqtt_dump")

# ── LOGGING ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("mqtt_dump")

shutdown_event = threading.Event()

# ── HELPERS ────────────────────────────────────────────
def get_folder_from_topic(topic: str) -> Path:
    """Create folder structure based on topic."""
    parts = topic.split("/")

    if topic.startswith("scenescape/data/camera/"):
        camera = parts[-1]
        return OUTPUT_DIR / "camera" / camera

    if topic.startswith("scenescape/regulated/scene/"):
        scene_id = parts[-1]
        return OUTPUT_DIR / "scene" / scene_id[:8]

    return OUTPUT_DIR / "unknown"


def save_payload(topic: str, payload: bytes):
    folder = get_folder_from_topic(topic)
    folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    file_path = folder / f"{timestamp}.json"

    try:
        data = json.loads(payload)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        log.info(f"Saved → {topic} → {file_path}")

    except json.JSONDecodeError:
        # fallback for non-json
        file_path = file_path.with_suffix(".txt")
        with open(file_path, "wb") as f:
            f.write(payload)

        log.warning(f"Non-JSON → {topic} → {file_path}")


# ── MQTT CALLBACKS ─────────────────────────────────────
def on_connect(client, userdata, flags, rc, properties):
    if rc == 0:
        log.info(f"Connected to MQTT {MQTT_BROKER}:{MQTT_PORT}")

        for topic in CAMERA_TOPICS:
            client.subscribe(topic)
            log.info(f"Subscribed → {topic}")

        client.subscribe(SCENE_TOPIC)
        log.info(f"Subscribed → {SCENE_TOPIC}")

    else:
        log.error(f"Connection failed: {rc}")


def on_message(client, userdata, msg):
    if shutdown_event.is_set():
        return

    try:
        data = json.loads(msg.payload)
    except json.JSONDecodeError:
        save_payload(msg.topic, msg.payload)
        return

    # Skip empty scene messages (~30/sec with no objects)
    if msg.topic.startswith("scenescape/regulated/scene/"):
        if not data.get("objects"):
            return

    # Skip empty camera messages (no detections)
    if msg.topic.startswith("scenescape/data/camera/"):
        objects = data.get("objects", {})
        has_any = any(objects.get(cat) for cat in objects)
        if not has_any:
            return

    save_payload(msg.topic, msg.payload)


def on_disconnect(client, userdata, flags, rc, properties):
    if not shutdown_event.is_set():
        log.warning("Disconnected. Reconnecting...")


# ── MAIN ───────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    client.reconnect_delay_set(min_delay=1, max_delay=30)

    def shutdown(signum, frame):
        log.info("Shutting down...")
        shutdown_event.set()
        client.disconnect()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    log.info("=== MQTT DUMP TOOL STARTED ===")
    log.info(f"Broker: {MQTT_BROKER}:{MQTT_PORT}")
    log.info(f"Output: {OUTPUT_DIR.resolve()}")

    try:
        client.connect(MQTT_BROKER, MQTT_PORT)
        client.loop_start()

        while not shutdown_event.is_set():
            shutdown_event.wait(1)

        client.loop_stop()

    finally:
        log.info("Stopped.")


if __name__ == "__main__":
    main()