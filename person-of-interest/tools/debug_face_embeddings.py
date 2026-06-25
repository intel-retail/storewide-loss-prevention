#!/usr/bin/env python3
"""Debug face embeddings: capture face crops from RTSP that match MQTT detections.

Subscribes to Scenescape MQTT data topic, extracts face bounding boxes
(nested as sub_objects under person), grabs the corresponding RTSP frame,
crops the exact face region, and saves it alongside the embedding metadata.

This lets you visually verify which face image produced each embedding.

Usage:
    python3 tools/debug_face_embeddings.py \
        --mqtt-broker localhost \
        --camera-id Camera_01 \
        --rtsp-url rtsp://localhost:8554/Camera_01

    python3 tools/debug_face_embeddings.py \
        --mqtt-broker localhost \
        --camera-id Camera_02 \
        --rtsp-url rtsp://localhost:8554/Camera_02
"""

import argparse
import base64
import json
import logging
import os
import signal
import struct
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
log = logging.getLogger("debug_face_embeddings")

shutdown_event = threading.Event()
capture_lock = threading.Lock()
last_capture_times: dict[int, float] = {}


def parse_args():
    p = argparse.ArgumentParser(description="Debug face embeddings from Scenescape")
    p.add_argument("--mqtt-broker", default=os.getenv("MQTT_BROKER", "localhost"))
    p.add_argument("--mqtt-port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    p.add_argument("--camera-id", default=os.getenv("CAMERA_ID", "Camera_01"))
    p.add_argument("--rtsp-url", default=os.getenv("RTSP_URL", ""))
    p.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "./debug_faces"))
    p.add_argument("--cooldown", type=float, default=2.0,
                   help="Min seconds between captures for same person ID")
    p.add_argument("--max-captures", type=int, default=50,
                   help="Stop after N face captures (0 = unlimited)")
    p.add_argument("--bbox-padding", type=int, default=5,
                   help="Pixels of padding around face crop")
    p.add_argument("--dump-mqtt", action="store_true",
                   help="Dump raw MQTT JSON to file for inspection")
    return p.parse_args()


def grab_frame_rtsp(rtsp_url: str) -> np.ndarray | None:
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    try:
        ret, frame = cap.read()
        return frame if ret else None
    finally:
        cap.release()


def crop_bbox(frame: np.ndarray, bbox: dict, padding: int = 5) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x = max(0, int(bbox["x"]) - padding)
    y = max(0, int(bbox["y"]) - padding)
    x2 = min(w, int(bbox["x"] + bbox["width"]) + padding)
    y2 = min(h, int(bbox["y"] + bbox["height"]) + padding)
    if x2 <= x or y2 <= y:
        return None
    return frame[y:y2, x:x2]


def decode_embedding(b64_str: str) -> list[float] | None:
    try:
        raw = base64.b64decode(b64_str)
        n_floats = len(raw) // 4
        return list(struct.unpack(f"{n_floats}f", raw))
    except Exception as e:
        log.warning("Failed to decode embedding: %s", e)
        return None


def main():
    args = parse_args()
    if not args.rtsp_url:
        args.rtsp_url = f"rtsp://localhost:8554/{args.camera_id}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    topic = f"scenescape/data/camera/{args.camera_id}"
    total_captures = 0

    log.info("=== Face Embedding Debug Tool ===")
    log.info("  MQTT: %s:%d  topic: %s", args.mqtt_broker, args.mqtt_port, topic)
    log.info("  RTSP: %s", args.rtsp_url)
    log.info("  Output: %s", output_dir.resolve())

    def on_connect(client, userdata, flags, rc, properties):
        if rc == 0:
            log.info("Connected to MQTT, subscribing to %s", topic)
            client.subscribe(topic)
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

        # Dump raw MQTT for debugging
        if args.dump_mqtt:
            dump_path = output_dir / "mqtt_dump.jsonl"
            with open(dump_path, "a") as f:
                f.write(json.dumps(data) + "\n")

        timestamp = data.get("timestamp", "unknown")
        objects = data.get("objects", {})
        persons = objects.get("person", [])

        if not persons:
            return

        # Check for face sub_objects or top-level face detections
        faces_found = []
        for person in persons:
            pid = person.get("id", 0)
            person_bbox = person.get("bounding_box_px")

            # Face as sub_object of person (chained pipeline)
            sub_objects = person.get("sub_objects", {})
            face_list = sub_objects.get("face", [])
            for face in face_list:
                face_bbox = face.get("bounding_box_px")
                reid_meta = face.get("metadata", {}).get("reid", {})
                faces_found.append({
                    "person_id": pid,
                    "face_bbox": face_bbox,
                    "person_bbox": person_bbox,
                    "reid": reid_meta,
                    "confidence": face.get("confidence", 0),
                    "source": "sub_object",
                })

            # Person-level reid (face embedding on person crop)
            person_reid = person.get("metadata", {}).get("reid", {})
            if person_reid:
                faces_found.append({
                    "person_id": pid,
                    "face_bbox": None,
                    "person_bbox": person_bbox,
                    "reid": person_reid,
                    "confidence": person.get("confidence", 0),
                    "source": "person_level",
                })

        # Also check top-level face objects (if detection_labels includes "face")
        top_faces = objects.get("face", [])
        for face in top_faces:
            face_bbox = face.get("bounding_box_px")
            reid_meta = face.get("metadata", {}).get("reid", {})
            faces_found.append({
                "person_id": 0,
                "face_bbox": face_bbox,
                "person_bbox": None,
                "reid": reid_meta,
                "confidence": face.get("confidence", 0),
                "source": "top_level",
            })

        if not faces_found:
            # Log when persons exist but no faces detected
            log.debug("Persons detected but no faces at %s (persons: %d)", timestamp, len(persons))
            return

        # Cooldown check
        eligible = []
        for f in faces_found:
            pid = f["person_id"]
            now = time.time()
            with capture_lock:
                last = last_capture_times.get(pid, 0)
                if (now - last) >= args.cooldown:
                    last_capture_times[pid] = now
                    eligible.append(f)

        if not eligible:
            return

        # Grab RTSP frame
        frame = grab_frame_rtsp(args.rtsp_url)
        if frame is None:
            log.warning("Failed to grab RTSP frame")
            return

        ts_safe = timestamp.replace(":", "-").replace(".", "-")

        for face_info in eligible:
            pid = face_info["person_id"]
            person_dir = output_dir / f"person_{pid}"
            person_dir.mkdir(parents=True, exist_ok=True)

            cap_id = total_captures + 1
            prefix = f"{args.camera_id}_{ts_safe}_p{pid}"

            # Save person crop
            if face_info["person_bbox"]:
                person_crop = crop_bbox(frame, face_info["person_bbox"], padding=10)
                if person_crop is not None and person_crop.size > 0:
                    cv2.imwrite(str(person_dir / f"{prefix}_person.jpg"), person_crop)

            # Save face crop
            if face_info["face_bbox"]:
                face_crop = crop_bbox(frame, face_info["face_bbox"], padding=args.bbox_padding)
                if face_crop is not None and face_crop.size > 0:
                    face_path = person_dir / f"{prefix}_face.jpg"
                    cv2.imwrite(str(face_path), face_crop)
                    log.info("FACE CROP: %s (%dx%d) conf=%.2f src=%s",
                             face_path.name, face_crop.shape[1], face_crop.shape[0],
                             face_info["confidence"], face_info["source"])

            # Save embedding metadata
            reid = face_info.get("reid", {})
            if reid:
                embedding = decode_embedding(reid.get("embedding_vector", ""))
                meta = {
                    "timestamp": timestamp,
                    "camera_id": args.camera_id,
                    "person_id": pid,
                    "source": face_info["source"],
                    "confidence": face_info["confidence"],
                    "model_name": reid.get("model_name", ""),
                    "embedding_dim": len(embedding) if embedding else 0,
                    "face_bbox": face_info["face_bbox"],
                    "person_bbox": face_info["person_bbox"],
                }
                meta_path = person_dir / f"{prefix}_meta.json"
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)

                log.info("  Embedding: %dD model=%s", meta["embedding_dim"], meta["model_name"])
            else:
                log.warning("  No reid embedding for person %d at %s", pid, timestamp)

            # Save full frame (once per capture batch)
            full_dir = output_dir / "full_frames"
            full_dir.mkdir(parents=True, exist_ok=True)
            full_path = full_dir / f"{args.camera_id}_{ts_safe}.jpg"
            if not full_path.exists():
                cv2.imwrite(str(full_path), frame)

            total_captures += 1

            if args.max_captures > 0 and total_captures >= args.max_captures:
                log.info("Reached max captures (%d), shutting down", args.max_captures)
                shutdown_event.set()
                client.disconnect()
                return

    def on_disconnect(client, userdata, flags, rc, properties):
        if not shutdown_event.is_set():
            log.warning("MQTT disconnected (rc=%s), will reconnect...", rc)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    client.reconnect_delay_set(min_delay=1, max_delay=30)

    def handle_signal(signum, _frame):
        log.info("Shutting down...")
        shutdown_event.set()
        client.disconnect()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        client.connect(args.mqtt_broker, args.mqtt_port)
        client.loop_start()
        while not shutdown_event.is_set():
            shutdown_event.wait(timeout=1)
        client.loop_stop()
    except KeyboardInterrupt:
        pass
    finally:
        log.info("Total face captures: %d", total_captures)
        log.info("Output: %s", output_dir.resolve())


if __name__ == "__main__":
    main()
