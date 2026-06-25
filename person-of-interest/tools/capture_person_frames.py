#!/usr/bin/env python3
"""Capture and crop person frames from Scenescape camera feeds.

Subscribes to Scenescape per-camera MQTT data, detects persons via bounding
boxes, grabs frames, crops each person, and saves both full frames and
cropped person images to disk.

Two frame-source modes:
  --frame-source rtsp   Grab frames from RTSP stream (default, ~100ms lag)
  --frame-source mqtt   Use exact frames published by DLStreamer pipeline
                        (requires publish_frame:true in pipeline config)

Usage:
    python3 tools/capture_person_frames.py [OPTIONS]

Environment variables (or CLI flags):
    MQTT_BROKER     MQTT broker host (default: localhost)
    MQTT_PORT       MQTT broker port (default: 1883)
    CAMERA_ID       Camera ID to monitor (default: lp-camera1)
    RTSP_URL        RTSP stream URL (default: rtsp://localhost:8554/lp-camera1)
    OUTPUT_DIR      Output directory for saved frames (default: ./captured_frames)
    COOLDOWN        Min seconds between captures for same person ID (default: 5)
    FRAME_SOURCE    Frame source: rtsp or mqtt (default: rtsp)
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
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import paho.mqtt.client as mqtt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("capture_person_frames")

# ── Globals ──────────────────────────────────────────────────────────
shutdown_event = threading.Event()
capture_lock = threading.Lock()
last_capture_times: dict[int, float] = {}  # person_id -> last capture epoch

# MQTT frame buffer: timestamp -> decoded cv2 image
frame_buffer_lock = threading.Lock()
frame_buffer: dict[str, cv2.typing.MatLike] = {}
FRAME_BUFFER_MAX = 30  # keep last N frames


def parse_args():
    p = argparse.ArgumentParser(description="Capture person frames from Scenescape")
    p.add_argument("--mqtt-broker", default=os.getenv("MQTT_BROKER", "localhost"))
    p.add_argument("--mqtt-port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    p.add_argument("--camera-id", default=os.getenv("CAMERA_ID", "lp-camera1"))
    p.add_argument("--rtsp-url", default=os.getenv("RTSP_URL", "rtsp://localhost:8554/lp-camera1"))
    p.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "./captured_frames"))
    p.add_argument("--cooldown", type=float, default=float(os.getenv("COOLDOWN", "5")),
                    help="Min seconds between captures for the same person ID")
    p.add_argument("--save-full-frame", action="store_true", default=True,
                    help="Also save the full frame alongside crops")
    p.add_argument("--bbox-padding", type=int, default=10,
                    help="Pixels of padding around the bounding box crop")
    p.add_argument("--max-captures", type=int, default=0,
                    help="Stop after N total captures (0 = unlimited)")
    p.add_argument("--frame-source", choices=["rtsp", "mqtt"],
                    default=os.getenv("FRAME_SOURCE", "rtsp"),
                    help="Frame source: rtsp (grab from stream) or mqtt (exact frames from pipeline)")
    return p.parse_args()


def grab_frame(rtsp_url: str) -> cv2.typing.MatLike | None:
    """Grab a single frame from the RTSP stream."""
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    try:
        ret, frame = cap.read()
        if not ret:
            log.warning("Failed to grab frame from %s", rtsp_url)
            return None
        return frame
    finally:
        cap.release()


def decode_mqtt_frame(b64_image: str) -> cv2.typing.MatLike | None:
    """Decode a base64-encoded JPEG image from MQTT into a cv2 frame."""
    try:
        jpeg_bytes = base64.b64decode(b64_image)
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        log.warning("Failed to decode MQTT frame: %s", e)
        return None


def buffer_mqtt_frame(timestamp: str, frame: cv2.typing.MatLike):
    """Store a frame in the buffer keyed by timestamp."""
    with frame_buffer_lock:
        frame_buffer[timestamp] = frame
        # Evict oldest frames if buffer is full
        while len(frame_buffer) > FRAME_BUFFER_MAX:
            oldest = next(iter(frame_buffer))
            del frame_buffer[oldest]


def get_buffered_frame(timestamp: str) -> cv2.typing.MatLike | None:
    """Retrieve and remove a frame from the buffer by exact timestamp."""
    with frame_buffer_lock:
        return frame_buffer.pop(timestamp, None)


def crop_person(frame, bbox: dict, padding: int = 10) -> cv2.typing.MatLike | None:
    """Crop a person from the frame using bounding_box_px.

    bbox format: {"x": int, "y": int, "width": int, "height": int}
    """
    h, w = frame.shape[:2]
    x = max(0, bbox["x"] - padding)
    y = max(0, bbox["y"] - padding)
    x2 = min(w, bbox["x"] + bbox["width"] + padding)
    y2 = min(h, bbox["y"] + bbox["height"] + padding)

    if x2 <= x or y2 <= y:
        log.warning("Invalid bounding box: %s", bbox)
        return None

    return frame[y:y2, x:x2]


def should_capture(person_id: int, cooldown: float) -> bool:
    """Check if enough time has passed since last capture for this person."""
    now = time.time()
    last = last_capture_times.get(person_id, 0)
    return (now - last) >= cooldown


def save_images(
    frame,
    persons: list[dict],
    camera_id: str,
    timestamp: str,
    output_dir: Path,
    padding: int,
    cooldown: float,
    save_full: bool,
) -> int:
    """Crop and save person images. Returns number of persons saved."""
    saved = 0
    ts_safe = timestamp.replace(":", "-").replace(".", "-")

    for person in persons:
        pid = person.get("id", 0)
        bbox = person.get("bounding_box_px")
        if not bbox:
            continue

        with capture_lock:
            if not should_capture(pid, cooldown):
                continue
            last_capture_times[pid] = time.time()

        # Crop person
        crop = crop_person(frame, bbox, padding)
        if crop is None or crop.size == 0:
            continue

        # Create per-person directory
        person_dir = output_dir / f"person_{pid}"
        person_dir.mkdir(parents=True, exist_ok=True)

        # Save cropped person
        crop_path = person_dir / f"{camera_id}_{ts_safe}_person{pid}_crop.jpg"
        cv2.imwrite(str(crop_path), crop)
        log.info("Saved crop: %s (%dx%d)", crop_path.name, crop.shape[1], crop.shape[0])
        saved += 1

        # Save full frame (once per timestamp)
        if save_full:
            full_dir = output_dir / "full_frames"
            full_dir.mkdir(parents=True, exist_ok=True)
            full_path = full_dir / f"{camera_id}_{ts_safe}.jpg"
            if not full_path.exists():
                cv2.imwrite(str(full_path), frame)
                log.info("Saved full frame: %s (%dx%d)", full_path.name, frame.shape[1], frame.shape[0])

    return saved


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    topic = f"scenescape/data/camera/{args.camera_id}"
    image_topic = f"scenescape/image/camera/{args.camera_id}"
    total_captures = 0
    use_mqtt_frames = args.frame_source == "mqtt"

    log.info("Starting person frame capture")
    log.info("  MQTT: %s:%d  topic: %s", args.mqtt_broker, args.mqtt_port, topic)
    log.info("  Frame source: %s", args.frame_source.upper())
    if not use_mqtt_frames:
        log.info("  RTSP: %s", args.rtsp_url)
    else:
        log.info("  Image topic: %s", image_topic)
        log.info("  (requires publish_frame:true in pipeline config)")
    log.info("  Output: %s", output_dir.resolve())
    log.info("  Cooldown: %.1fs  Padding: %dpx", args.cooldown, args.bbox_padding)

    def on_connect(client, userdata, flags, rc, properties):
        if rc == 0:
            log.info("Connected to MQTT broker, subscribing to %s", topic)
            client.subscribe(topic)
            if use_mqtt_frames:
                client.subscribe(image_topic)
                log.info("Subscribed to image topic: %s", image_topic)
        else:
            log.error("MQTT connection failed: rc=%s", rc)

    def on_message(client, userdata, msg):
        nonlocal total_captures
        if shutdown_event.is_set():
            return

        try:
            data = json.loads(msg.payload)
        except json.JSONDecodeError:
            return

        # Handle image frames from MQTT (publish_frame mode)
        if msg.topic == image_topic:
            b64_img = data.get("image")
            ts = data.get("timestamp")
            if b64_img and ts:
                frame = decode_mqtt_frame(b64_img)
                if frame is not None:
                    buffer_mqtt_frame(ts, frame)
                    log.debug("Buffered MQTT frame: %s (%dx%d)", ts, frame.shape[1], frame.shape[0])
            return

        # Handle detection data
        persons = data.get("objects", {}).get("person", [])
        if not persons:
            return

        # Filter to persons not recently captured
        eligible = [p for p in persons if p.get("bounding_box_px")
                    and should_capture(p.get("id", 0), args.cooldown)]
        if not eligible:
            return

        timestamp = data.get("timestamp", datetime.now().isoformat())
        log.info("Detected %d person(s) (%d eligible) at %s",
                 len(persons), len(eligible), timestamp)

        # Get frame based on source mode
        if use_mqtt_frames:
            frame = get_buffered_frame(timestamp)
            if frame is None:
                log.warning("No MQTT frame for timestamp %s (buffer has %d frames)",
                            timestamp, len(frame_buffer))
                return
            log.info("Using exact MQTT frame for %s", timestamp)
        else:
            frame = grab_frame(args.rtsp_url)
            if frame is None:
                return

        saved = save_images(
            frame, eligible, args.camera_id, timestamp,
            output_dir, args.bbox_padding, args.cooldown, args.save_full_frame,
        )
        total_captures += saved

        if args.max_captures > 0 and total_captures >= args.max_captures:
            log.info("Reached max captures (%d), shutting down", args.max_captures)
            shutdown_event.set()
            client.disconnect()

    def on_disconnect(client, userdata, flags, rc, properties):
        if not shutdown_event.is_set():
            log.warning("Disconnected from MQTT (rc=%s), will reconnect...", rc)

    # Set up MQTT client
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    client.reconnect_delay_set(min_delay=1, max_delay=30)

    # Signal handling for graceful shutdown
    def handle_signal(signum, _frame):
        log.info("Received signal %d, shutting down...", signum)
        shutdown_event.set()
        client.disconnect()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Connect and run
    try:
        client.connect(args.mqtt_broker, args.mqtt_port)
        client.loop_start()

        while not shutdown_event.is_set():
            shutdown_event.wait(timeout=1)

        client.loop_stop()
    except KeyboardInterrupt:
        pass
    finally:
        log.info("Total person crops saved: %d", total_captures)
        log.info("Output directory: %s", output_dir.resolve())


if __name__ == "__main__":
    main()
