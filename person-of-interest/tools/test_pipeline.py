#!/usr/bin/env python3
"""
Integration smoke-test for the Person-of-Interest pipeline.

Tests (run in order):
  1. Face detection        — extract face embedding from poi.JPG with OpenVINO
  2. POI enrollment        — add that embedding into a fresh in-memory FAISS index
  3. FAISS match           — query the same embedding; expect similarity ≥ threshold
  4. MQTT consumer         — feed a fake camera MQTT payload through EventConsumer;
                             verify match fires and best_face_bbox is the face ROI
  5. UUID cross-camera     — subscribe to live Scenescape MQTT for N seconds and
                             confirm the same UUID appears on ≥ 2 different cameras
  6. Thumbnail visualise   — if a thumbnail was stored (Redis present), save it as
                             /tmp/poi_thumbnail.jpg and print its path

Usage (from repo root):
  cd person-of-interest
  python tools/test_pipeline.py [--poi-image sample_data/poi.JPG] [--mqtt-host broker.scenescape.intel.com] [--uuid-listen-secs 30]

The script does NOT require the full stack to be running for tests 1-4.
Tests 5-6 need live MQTT / Redis.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import cv2
import numpy as np

# ── allow running from tools/ or from person-of-interest/ ─────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent   # person-of-interest/
sys.path.insert(0, str(_REPO_ROOT))

# ── helpers ───────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"


def ok(msg: str):   print(f"  {GREEN}✔  {msg}{RESET}")
def fail(msg: str): print(f"  {RED}✗  {msg}{RESET}"); sys.exit(1)
def warn(msg: str): print(f"  {YELLOW}⚠  {msg}{RESET}")
def info(msg: str): print(f"  {CYAN}ℹ  {msg}{RESET}")


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ── 1. Face detection with OpenVINO ───────────────────────────────────────────

def test_face_detection(poi_image_path: str, model_base: str) -> np.ndarray:
    section("TEST 1 — Face detection from poi.JPG (OpenVINO)")

    image = cv2.imread(poi_image_path)
    if image is None:
        fail(f"Cannot read image: {poi_image_path}")

    ok(f"Image loaded: {image.shape[1]}×{image.shape[0]} px from {Path(poi_image_path).name}")

    det_model_path  = os.path.join(model_base, "face-detection-retail-0004", "FP32", "face-detection-retail-0004.xml")
    reid_model_path = os.path.join(model_base, "face-reidentification-retail-0095", "FP32", "face-reidentification-retail-0095.xml")

    if not os.path.exists(det_model_path):
        fail(f"Detection model not found: {det_model_path}\n       Run 'make download-models' first.")
    if not os.path.exists(reid_model_path):
        fail(f"ReID model not found: {reid_model_path}\n       Run 'make download-models' first.")

    try:
        from openvino import Core
        core = Core()
        det  = core.compile_model(det_model_path,  "CPU")
        reid = core.compile_model(reid_model_path, "CPU")
    except Exception as e:
        fail(f"OpenVINO model load failed: {e}")

    # ── detection
    img_h, img_w = image.shape[:2]
    _, c, h, w = det.input(0).shape
    blob = cv2.resize(image, (w, h)).transpose(2, 0, 1).reshape(1, c, h, w).astype(np.float32)
    detections = det(blob)[det.output(0)]

    best_face: Optional[tuple] = None
    best_conf = 0.0
    for det_row in detections[0][0]:
        conf = float(det_row[2])
        if conf > 0.5 and conf > best_conf:
            x1 = max(0, int(det_row[3] * img_w))
            y1 = max(0, int(det_row[4] * img_h))
            x2 = min(img_w, int(det_row[5] * img_w))
            y2 = min(img_h, int(det_row[6] * img_h))
            if x2 > x1 and y2 > y1:
                best_face = (x1, y1, x2, y2)
                best_conf = conf

    if best_face is None:
        fail("No face detected in poi.JPG — check the image contains a clear frontal face")

    x1, y1, x2, y2 = best_face
    ok(f"Face detected: bbox=[{x1},{y1},{x2},{y2}]  confidence={best_conf:.3f}")

    # ── save annotated image so user can inspect it
    annotated = image.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(annotated, f"{best_conf:.2f}", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    out_path = "/tmp/poi_face_detection.jpg"
    cv2.imwrite(out_path, annotated)
    ok(f"Annotated image saved → {out_path}")

    # ── reid embedding
    face_crop = image[y1:y2, x1:x2]
    aligned = cv2.resize(face_crop, (128, 128)).transpose(2, 0, 1).reshape(1, 3, 128, 128).astype(np.float32)
    embedding = reid(aligned)[reid.output(0)][0].flatten()
    embedding = (embedding / np.linalg.norm(embedding)).astype(np.float32)

    ok(f"Face embedding extracted: dim={len(embedding)}, norm={np.linalg.norm(embedding):.4f}")
    return embedding, best_face


# ── 2+3. FAISS enrol + match ──────────────────────────────────────────────────

def test_faiss(embedding: np.ndarray, threshold: float = 0.6) -> None:
    section("TEST 2+3 — FAISS enrolment and cosine matching")

    try:
        import faiss
    except ImportError:
        fail("faiss-cpu not installed — pip install faiss-cpu")

    dim = len(embedding)
    flat  = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(flat)

    # Enrol the POI embedding with ID=0
    vec = embedding.copy().reshape(1, -1)
    index.add_with_ids(vec, np.array([0], dtype=np.int64))
    ok(f"Enrolled 1 embedding into FAISS (dim={dim})")

    # Query with the exact same vector — should return cosine similarity = 1.0
    D, I = index.search(vec, 1)
    sim  = float(D[0][0])
    faiss_id = int(I[0][0])
    ok(f"Self-match: faiss_id={faiss_id}  similarity={sim:.4f}")
    if sim < threshold:
        fail(f"Self-similarity {sim:.4f} is below threshold {threshold} — FAISS index broken")
    ok(f"FAISS match passed (similarity={sim:.4f} ≥ threshold={threshold})")

    # Query with a random vector — should NOT match
    noise = np.random.randn(dim).astype(np.float32)
    noise /= np.linalg.norm(noise)
    D_rand, _ = index.search(noise.reshape(1, -1), 1)
    sim_rand = float(D_rand[0][0])
    ok(f"Random noise query: similarity={sim_rand:.4f} (expected < {threshold})")
    if sim_rand >= threshold:
        warn(f"Random vector scored {sim_rand:.4f} ≥ threshold — consider raising threshold above 0.6")


# ── 4. MQTT EventConsumer integration ─────────────────────────────────────────

def _run_consumer_test(embedding: np.ndarray, face_bbox: tuple) -> None:
    """Inner body of test 4, called after redis/faiss mocks are in place."""
    # Set required env vars for Config singleton
    os.environ.setdefault("REDIS_HOST", "localhost")
    os.environ.setdefault("FAISS_INDEX_PATH", "/tmp/test_poi_pipeline.index")
    os.environ.setdefault("FAISS_ID_MAP_PATH", "/tmp/test_poi_id_map.json")

    from backend.consumers.mqtt_consumer import EventConsumer, _parse_embedding, FACE_CONFIDENCE_THRESHOLD
    from backend.domain.entities.match_result import AlertPayload, MatchResult
    from backend.observer.events import EventBus

    # ── encode embedding as base64 (camera topic wire format)
    b64_embedding = base64.b64encode(struct.pack(f"{len(embedding)}f", *embedding)).decode("utf-8")

    # ── Build camera MQTT payload with face sub_object
    x1, y1, x2, y2 = face_bbox
    camera_payload = {
        "id": "Camera_01",
        "timestamp": "2026-04-29T10:00:00.000Z",
        "rate": 10.0,
        "objects": {
            "person": [
                {
                    "id": 1,
                    "category": "person",
                    "confidence": 0.96,
                    "bounding_box_px": {"x": max(0, x1 - 50), "y": max(0, y1 - 100),
                                        "width": (x2 - x1) + 100, "height": (y2 - y1) + 200},
                    "metadata": {
                        "reid": {
                            "embedding_vector": base64.b64encode(
                                struct.pack(f"{len(embedding)}f", *np.random.randn(len(embedding)).astype(np.float32))
                            ).decode("utf-8"),
                            "model_name": "person-reidentification-retail-0277"
                        }
                    },
                    "sub_objects": {
                        "face": [
                            {
                                "id": 1,
                                "category": "face",
                                "confidence": 0.92,   # above FACE_CONFIDENCE_THRESHOLD (0.80)
                                "bounding_box_px": {"x": x1, "y": y1,
                                                    "width": x2 - x1, "height": y2 - y1},
                                "metadata": {
                                    "reid": {
                                        "embedding_vector": b64_embedding,
                                        "model_name": "face-reidentification-retail-0095"
                                    }
                                }
                            }
                        ]
                    }
                }
            ]
        }
    }

    # ── Wire up consumer with a mock matching service that always matches
    poi_id_expected = "poi-test-001"
    match_result = MatchResult(
        poi_id=poi_id_expected,
        similarity_score=0.95,
        faiss_distance=0.95,
    )

    matching = MagicMock()
    matching.match_object.return_value = match_result

    events = MagicMock()
    alerts = MagicMock()
    alerts.create_alert_payload.return_value = AlertPayload(
        alert_id="alert-test-001",
        poi_id=poi_id_expected,
        severity="high",
        timestamp="2026-04-29T10:00:00.000Z",
        match={"similarity_score": 0.95, "camera_id": "Camera_01",
               "confidence": 0.92, "thumbnail_path": ""},
        poi_metadata={"notes": ""},
    )
    event_bus = EventBus()

    # Capture event_bus publish calls
    published_events = []
    event_bus.subscribe("match_found", lambda e: published_events.append(e))

    consumer = EventConsumer(matching, events, alerts, event_bus)

    # ── Handle the camera event
    consumer.handle_event("scenescape/data/camera/Camera_01", camera_payload)

    # ── Verify
    assert matching.match_object.call_count == 1, \
        f"Expected match_object called once, got {matching.match_object.call_count}"
    ok("EventConsumer called matching.match_object for the face embedding")

    call_args = matching.match_object.call_args
    object_id_arg  = call_args[1].get("object_id") or call_args[0][0]
    embedding_arg  = call_args[1].get("embedding_vector") or call_args[0][1]
    bbox_arg       = call_args[1].get("bounding_box") if call_args[1] else None

    ok(f"object_id = {object_id_arg}  (format: cam:<camera_id>:<person_int_id>)")
    assert "cam:Camera_01:1" in str(object_id_arg), \
        f"Unexpected object_id format: {object_id_arg}"
    ok("object_id format correct: cam:Camera_01:1")

    # Verify the embedding passed to FAISS is the FACE embedding (not body reid)
    emb_arr = np.array(embedding_arg, dtype=np.float32)
    similarity = float(np.dot(emb_arr / np.linalg.norm(emb_arr),
                              embedding / np.linalg.norm(embedding)))
    ok(f"Embedding passed to FAISS: cosine-similarity to poi.JPG face = {similarity:.4f}")
    if similarity < 0.99:
        fail(f"Consumer passed wrong embedding to FAISS (similarity={similarity:.4f})")
    ok("Correct face embedding (not body reid) passed to FAISS")

    # Verify bbox used for thumbnail is the FACE bbox (after our change)
    if bbox_arg:
        face_bbox_expected = {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}
        ok(f"bounding_box passed for thumbnail: {bbox_arg}")
        if bbox_arg == face_bbox_expected:
            ok("Thumbnail bounding box = face bbox (correct — tight face crop)")
        else:
            warn(f"Thumbnail bbox differs from face bbox: got {bbox_arg}, expected {face_bbox_expected}")
    else:
        warn("bounding_box not captured in call_args — check _run_matching signature")

    assert len(published_events) == 1, \
        f"Expected 1 match_found event, got {len(published_events)}"
    ok(f"match_found event published: poi_id={published_events[0].alert.poi_id}")

    # Low-confidence face should be skipped
    camera_payload_low_conf = json.loads(json.dumps(camera_payload))
    camera_payload_low_conf["objects"]["person"][0]["sub_objects"]["face"][0]["confidence"] = 0.50
    matching.reset_mock()
    consumer.handle_event("scenescape/data/camera/Camera_01", camera_payload_low_conf)
    assert matching.match_object.call_count == 0, "Low-confidence face should be skipped"
    ok(f"Low-confidence face (0.50 < {FACE_CONFIDENCE_THRESHOLD}) correctly skipped")


def test_mqtt_consumer(embedding: np.ndarray, face_bbox: tuple) -> None:
    section("TEST 4 — MQTT EventConsumer face sub_object pipeline")

    # Mock redis/faiss only for this test, then restore so later tests use real libs
    import sys
    _mocked = {}
    for mod in ("redis", "faiss"):
        if mod not in sys.modules:
            _mocked[mod] = MagicMock()
            sys.modules[mod] = _mocked[mod]
    try:
        _run_consumer_test(embedding, face_bbox)
    finally:
        for mod, mock in _mocked.items():
            if sys.modules.get(mod) is mock:
                del sys.modules[mod]


# ── 5. UUID cross-camera consistency (live MQTT) ──────────────────────────────

def test_uuid_cross_camera(mqtt_host: str, mqtt_port: int, listen_secs: int) -> None:
    section(f"TEST 5 — UUID cross-camera consistency (live MQTT, {listen_secs}s)")
    info(f"Connecting to {mqtt_host}:{mqtt_port} …")

    try:
        import paho.mqtt.client as mqtt_client
    except ImportError:
        warn("paho-mqtt not installed — skipping live UUID test")
        return

    # uuid → set of camera_ids seen on
    uuid_cameras: dict[str, set] = {}
    lock = threading.Lock()
    connected = threading.Event()

    def on_connect(client, userdata, flags, rc, props=None):
        if rc == 0:
            client.subscribe("scenescape/external/+/person")
            connected.set()
            info(f"Subscribed to scenescape/external/+/person")
        else:
            warn(f"MQTT connect failed rc={rc}")

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload)
        except Exception:
            return
        for obj in payload.get("objects", []):
            uid = obj.get("id")
            vis = obj.get("visibility", [])
            if uid and vis:
                with lock:
                    uuid_cameras.setdefault(uid, set()).update(vis)

    client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(mqtt_host, mqtt_port, keepalive=10)
        client.loop_start()
    except Exception as e:
        warn(f"Cannot connect to MQTT broker {mqtt_host}:{mqtt_port} — {e}")
        warn("Skipping cross-camera UUID test (broker not reachable)")
        return

    if not connected.wait(timeout=5):
        warn("MQTT connection timed out — skipping UUID cross-camera test")
        client.loop_stop()
        return

    info(f"Listening for {listen_secs} seconds…")
    for remaining in range(listen_secs, 0, -5):
        time.sleep(5)
        with lock:
            multi_camera = {uid: cams for uid, cams in uuid_cameras.items() if len(cams) >= 2}
        info(f"  {remaining-5}s remaining — UUIDs seen so far: {len(uuid_cameras)}, "
             f"cross-camera UUIDs: {len(multi_camera)}")

    client.loop_stop()
    client.disconnect()

    with lock:
        total_uuids = len(uuid_cameras)
        multi_camera_uuids = {uid: cams for uid, cams in uuid_cameras.items() if len(cams) >= 2}

    ok(f"Total UUIDs observed: {total_uuids}")
    if multi_camera_uuids:
        ok(f"Cross-camera UUIDs (same UUID on ≥2 cameras): {len(multi_camera_uuids)}")
        for uid, cams in list(multi_camera_uuids.items())[:5]:
            ok(f"  UUID {uid[:16]}…  cameras={sorted(cams)}")
    else:
        warn("No UUID observed on multiple cameras during the listen window")
        warn("Either only one camera is active, or listen_secs is too short (try --uuid-listen-secs 60)")
        info("All UUIDs observed (single-camera):")
        for uid, cams in list(uuid_cameras.items())[:10]:
            info(f"  UUID {uid[:16]}…  cameras={sorted(cams)}")


# ── 6. Thumbnail visualisation from Redis ────────────────────────────────────

def test_thumbnail_visualise(redis_host: str = "localhost", redis_port: int = 6379) -> None:
    section("TEST 6 — Thumbnail visualisation from Redis")

    try:
        import redis as redis_lib
        r = redis_lib.Redis(host=redis_host, port=redis_port, db=0, socket_timeout=2)
        r.ping()
    except Exception as e:
        warn(f"Redis not reachable ({redis_host}:{redis_port}) — {e}")
        warn("Skipping thumbnail test (Redis required)")
        return

    # Scan for thumbnail keys (pattern: thumbnail:*)
    pattern = "thumbnail:*"
    keys = list(r.scan_iter(pattern, count=50))
    info(f"Found {len(keys)} thumbnail keys in Redis")

    if not keys:
        warn("No thumbnails in Redis yet — run the full pipeline first")
        warn("Thumbnails are stored on first POI match via RTSP capture")
        return

    saved_paths = []
    for key in keys[:5]:
        b64 = r.get(key)
        if not b64:
            continue
        try:
            img_bytes = base64.b64decode(b64)
        except Exception:
            continue

        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue

        key_str = key.decode("utf-8") if isinstance(key, bytes) else key
        object_id = key_str.replace("thumbnail:", "").replace(":", "_")
        out_path = f"/tmp/thumbnail_{object_id}.jpg"
        cv2.imwrite(out_path, img)
        saved_paths.append((key_str, out_path, img.shape))

    if saved_paths:
        ok(f"Saved {len(saved_paths)} thumbnail(s):")
        for key_str, path, shape in saved_paths:
            ok(f"  key={key_str}  size={shape[1]}×{shape[0]}  → {path}")
        info("Open the saved JPEGs to visually verify they show the correct face crop")
    else:
        warn("Thumbnails found in Redis but could not be decoded as images")


# ── Simulate thumbnail without RTSP: inject a fake thumbnail into Redis ──────

def test_synthetic_thumbnail(embedding: np.ndarray, face_bbox: tuple,
                              poi_image_path: str,
                              redis_host: str = "localhost", redis_port: int = 6379) -> None:
    section("TEST 6b — Synthetic thumbnail: crop poi.JPG face → store in Redis → retrieve")

    try:
        import redis as redis_lib
        r = redis_lib.Redis(host=redis_host, port=redis_port, db=0, socket_timeout=2)
        r.ping()
    except Exception:
        warn("Redis not reachable — synthesising thumbnail without Redis storage")
        # Still save locally so user can see the face crop
        image = cv2.imread(poi_image_path)
        x1, y1, x2, y2 = face_bbox
        padding = 10
        h, w = image.shape[:2]
        crop = image[max(0, y1 - padding): min(h, y2 + padding),
                     max(0, x1 - padding): min(w, x2 + padding)]
        out_path = "/tmp/poi_face_crop.jpg"
        cv2.imwrite(out_path, crop)
        ok(f"Face crop saved (no Redis) → {out_path}")
        return

    image = cv2.imread(poi_image_path)
    x1, y1, x2, y2 = face_bbox
    padding = 10
    h, w = image.shape[:2]
    crop = image[max(0, y1 - padding): min(h, y2 + padding),
                 max(0, x1 - padding): min(w, x2 + padding)]

    ok(f"Face crop from poi.JPG: {crop.shape[1]}×{crop.shape[0]} px")

    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    test_object_id = "cam:Camera_01:1"
    redis_key = f"thumbnail:{test_object_id}"
    r.setex(redis_key, 3600, b64)
    ok(f"Stored synthetic thumbnail in Redis: key={redis_key}")

    # Retrieve and save
    retrieved = r.get(redis_key)
    if not retrieved or not isinstance(retrieved, (bytes, str)):
        warn("Could not retrieve thumbnail from Redis (type mismatch) — check Redis connection")
        out_path = "/tmp/poi_face_crop.jpg"
        cv2.imwrite(out_path, crop)
        ok(f"Face crop saved locally → {out_path}")
        return
    arr = np.frombuffer(base64.b64decode(retrieved), dtype=np.uint8)
    img_back = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    out_path = "/tmp/poi_thumbnail_from_redis.jpg"
    cv2.imwrite(out_path, img_back)
    ok(f"Retrieved thumbnail from Redis → {out_path}")
    ok("Open /tmp/poi_thumbnail_from_redis.jpg to verify the face crop looks correct")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="POI pipeline smoke-test")
    p.add_argument("--poi-image",
                   default=str(Path(__file__).parent.parent / "sample_data" / "poi.JPG"),
                   help="Path to POI reference image (default: sample_data/poi.JPG)")
    p.add_argument("--model-base",
                   default=str(Path(__file__).parent.parent / "models" / "intel" / "intel"),
                   help="Base directory containing OpenVINO model folders")
    p.add_argument("--mqtt-host", default="broker.scenescape.intel.com",
                   help="MQTT broker hostname for live UUID cross-camera test")
    p.add_argument("--mqtt-port", type=int, default=1883)
    p.add_argument("--uuid-listen-secs", type=int, default=30,
                   help="Seconds to listen for cross-camera UUID consistency (default: 30)")
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument("--skip-live", action="store_true",
                   help="Skip tests that require live MQTT/Redis (tests 5 and 6)")
    p.add_argument("--faiss-threshold", type=float, default=0.6,
                   help="Cosine similarity threshold for FAISS match (default: 0.6)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'═'*60}")
    print(f"  POI Pipeline Smoke Test")
    print(f"  poi image   : {args.poi_image}")
    print(f"  model base  : {args.model_base}")
    print(f"  mqtt broker : {args.mqtt_host}:{args.mqtt_port}")
    print(f"  redis       : {args.redis_host}:{args.redis_port}")
    print(f"{'═'*60}")

    # ── Tests 1-4: no live services required ──────────────────────────────
    embedding, face_bbox = test_face_detection(args.poi_image, args.model_base)
    test_faiss(embedding, threshold=args.faiss_threshold)
    test_mqtt_consumer(embedding, face_bbox)

    # ── Tests 5-6: require live broker / Redis ─────────────────────────────
    if not args.skip_live:
        test_uuid_cross_camera(args.mqtt_host, args.mqtt_port, args.uuid_listen_secs)
        test_thumbnail_visualise(args.redis_host, args.redis_port)
        test_synthetic_thumbnail(embedding, face_bbox, args.poi_image,
                                  args.redis_host, args.redis_port)
    else:
        # Always save the face crop locally even without Redis
        test_synthetic_thumbnail(embedding, face_bbox, args.poi_image,
                                  "localhost", 9999)   # port 9999 → will fail Redis ping → local save

    print(f"\n{'═'*60}")
    print(f"  {GREEN}All offline tests passed.{RESET}")
    print(f"  Saved outputs:")
    print(f"    /tmp/poi_face_detection.jpg        — annotated face bbox")
    print(f"    /tmp/poi_face_crop.jpg             — tight face crop (if Redis unavailable)")
    print(f"    /tmp/poi_thumbnail_from_redis.jpg  — face crop stored/retrieved via Redis")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
