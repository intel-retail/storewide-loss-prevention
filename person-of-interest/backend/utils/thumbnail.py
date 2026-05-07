"""Thumbnail capture utility — persistent per-camera frame grabber + bbox crop.

Two capture strategies are supported per camera:

1. MQTT image topic (preferred when available):
   SceneScape DLStreamer adapter publishes annotated frames on demand to
   scenescape/image/camera/{camera_id} when a "getimage" command is sent.
   We subscribe to that topic, cache the latest received frame, and also
   proactively request a fresh frame at match time.  This eliminates RTSP
   timing drift entirely because the image comes from the same pipeline that
   produced the detection bounding boxes.

2. RTSP grabber (fallback):
   A background thread continuously reads the RTSP stream and caches the
   latest frame.  Used for cameras that do not have publish_image configured.
"""

from __future__ import annotations

import base64
import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("poi.thumbnail")

# RTSP base URL pattern; override via RTSP_BASE_URL env var
_RTSP_BASE_URL = os.getenv("RTSP_BASE_URL", "rtsp://mediaserver:8554")

# MQTT broker for SceneScape image topic (same broker used by the MQTT consumer)
_MQTT_HOST = os.getenv("MQTT_HOST", "broker.scenescape.intel.com")
_MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

# Shared thread pool for async capture submissions
_executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="thumbnail")


def build_rtsp_url(camera_id: str) -> str:
    return f"{_RTSP_BASE_URL.rstrip('/')}/{camera_id}"


# ---------------------------------------------------------------------------
# MQTT image subscriber (preferred source — same frame as detection)
# ---------------------------------------------------------------------------

def _parse_pipeline_ts(ts_str: str) -> float:
    """Parse a pipeline ISO-8601 timestamp string to a Unix timestamp float.

    Handles the 'Z' suffix used by sscape_adapter (e.g. '2026-05-07T12:34:56.789Z').
    """
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()


class _MqttImageSubscriber:
    """Subscribes to scenescape/image/camera/{camera_id} and maintains a
    timestamp-indexed ring buffer of recent frames.

    Why a ring buffer instead of request-and-wait:
      sscape_adapter publishes ONE frame per `getimage` command *for the next
      video frame it processes*.  By the time our backend receives a detection
      and sends `getimage`, DLStreamer is already 1-3 frames ahead — the person
      may have moved.

      The image payload includes the same `timestamp` field as the detection
      payload (both set from `postdecode_timestamp`).  By continuously sending
      `getimage` at a heartbeat rate and buffering every response with its
      pipeline timestamp, we can instantly retrieve the frame whose timestamp
      is closest to any given detection timestamp — this is the actual frame
      that was being processed when the detection was published.

    Heartbeat: sends `getimage` every _HEARTBEAT_INTERVAL seconds so the
    buffer is always populated with recent frames.
    """

    # Number of (timestamp, b64) pairs to keep.  At the heartbeat rate this
    # covers ~15 seconds of history — more than enough to match any detection.
    _RING_BUFFER_SIZE = 30

    # How often to proactively send `getimage` (seconds).
    # 300 ms → ~3 frames/sec cached; matches well with a 10-FPS pipeline.
    _HEARTBEAT_INTERVAL = 0.3

    def __init__(self, camera_id: str, mqtt_host: str, mqtt_port: int) -> None:
        import paho.mqtt.client as mqtt  # type: ignore[import]
        self._camera_id = camera_id
        self._host = mqtt_host
        self._port = mqtt_port

        # Ring buffer: deque of (pipeline_timestamp_str, base64_jpeg)
        self._ring: deque[tuple[str, str]] = deque(maxlen=self._RING_BUFFER_SIZE)
        self._latest_b64: Optional[str] = None
        self._cond = threading.Condition()

        self._image_topic = f"scenescape/image/camera/{camera_id}"
        self._cmd_topic   = f"scenescape/cmd/camera/{camera_id}"

        self._client = mqtt.Client(client_id=f"poi-thumbnail-{camera_id}")
        self._client.on_connect    = self._on_connect
        self._client.on_message    = self._on_message
        self._client.on_disconnect = self._on_disconnect

        # MQTT network loop thread
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"mqtt-img-{camera_id}"
        )
        self._thread.start()

        # Heartbeat thread — sends getimage periodically so the ring is always warm
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name=f"mqtt-hb-{camera_id}"
        )
        self._hb_thread.start()

    def _run(self) -> None:
        while True:
            try:
                self._client.connect(self._host, self._port, keepalive=30)
                self._client.loop_forever()
            except Exception as exc:
                log.warning("MQTT image subscriber disconnected camera=%s: %s", self._camera_id, exc)
            time.sleep(3)

    def _heartbeat_loop(self) -> None:
        """Continuously send getimage so the ring buffer stays populated."""
        # Give the MQTT connection a moment to establish before the first request
        time.sleep(3.0)
        while True:
            self.request_frame()
            time.sleep(self._HEARTBEAT_INTERVAL)

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            client.subscribe(self._image_topic, qos=0)
            log.info("MQTT image subscriber connected: camera=%s topic=%s",
                     self._camera_id, self._image_topic)
        else:
            log.warning("MQTT image subscriber failed rc=%d camera=%s", rc, self._camera_id)

    def _on_disconnect(self, client, userdata, rc) -> None:
        log.debug("MQTT image subscriber disconnected rc=%d camera=%s", rc, self._camera_id)

    def _on_message(self, client, userdata, msg) -> None:
        """Store the received frame in the ring buffer with its pipeline timestamp."""
        try:
            import json as _json
            data = _json.loads(msg.payload)
            b64 = data.get("image")
            frame_ts = data.get("timestamp", "")   # same field as detection payload
            if b64:
                with self._cond:
                    self._ring.append((frame_ts, b64))
                    self._latest_b64 = b64
                    self._cond.notify_all()
                log.debug("MQTT image received camera=%s ts=%s len=%d",
                          self._camera_id, frame_ts, len(b64))
        except Exception as exc:
            log.debug("MQTT image parse error camera=%s: %s", self._camera_id, exc)

    def request_frame(self) -> None:
        """Ask the DLStreamer adapter to publish a fresh frame (fire-and-forget)."""
        try:
            self._client.publish(self._cmd_topic, "getimage", qos=0)
        except Exception as exc:
            log.debug("Failed to send getimage camera=%s: %s", self._camera_id, exc)

    def get_frame_for_timestamp(self, detection_ts: str, max_age_sec: float = 3.0) -> Optional[str]:
        """Return the buffered frame whose pipeline timestamp is closest to detection_ts.

        The sscape_adapter sets the same `postdecode_timestamp` on both the
        detection payload and the image payload for each frame, so the best
        match will be the exact frame that produced the detection.

        Returns None if the ring buffer is empty.
        Falls back to `_latest_b64` if timestamps cannot be parsed.
        """
        with self._cond:
            if not self._ring:
                return None

            if not detection_ts:
                return self._latest_b64

            try:
                t_target = _parse_pipeline_ts(detection_ts)
            except (ValueError, TypeError):
                return self._latest_b64

            best_b64: Optional[str] = None
            best_delta = float("inf")
            for frame_ts, frame_b64 in self._ring:
                try:
                    delta = abs(_parse_pipeline_ts(frame_ts) - t_target)
                    if delta < best_delta:
                        best_delta = delta
                        best_b64 = frame_b64
                except (ValueError, TypeError):
                    continue

            if best_b64 is not None and best_delta <= max_age_sec:
                log.debug("Ring buffer match: camera=%s delta=%.3fs", self._camera_id, best_delta)
                return best_b64

            # Fallback: use latest regardless of age
            return self._latest_b64

    def request_frame_and_wait(self, timeout: float = 3.0) -> Optional[str]:
        """Fallback: send getimage and wait for the very next frame response.

        Used when the ring buffer is empty (e.g. MQTT just connected).
        """
        with self._cond:
            self.request_frame()
            self._cond.wait(timeout=timeout)
            return self._latest_b64


# Registry: camera_id -> _MqttImageSubscriber
_mqtt_subscribers: dict[str, _MqttImageSubscriber] = {}
_mqtt_sub_lock = threading.Lock()


def _get_mqtt_subscriber(camera_id: str) -> _MqttImageSubscriber:
    with _mqtt_sub_lock:
        if camera_id not in _mqtt_subscribers:
            sub = _MqttImageSubscriber(camera_id, _MQTT_HOST, _MQTT_PORT)
            _mqtt_subscribers[camera_id] = sub
        return _mqtt_subscribers[camera_id]


# Set of camera IDs that have MQTT image publishing enabled
# Populated by prewarm_grabbers via RTSP_PREWARM_CAMERAS; overridden by
# MQTT_IMAGE_CAMERAS env var (comma-separated list)
_mqtt_image_cameras: set[str] = set(
    c.strip() for c in os.getenv("MQTT_IMAGE_CAMERAS", "").split(",") if c.strip()
)


def use_mqtt_image(camera_id: str) -> bool:
    """Return True if this camera should use MQTT image topic instead of RTSP."""
    return camera_id in _mqtt_image_cameras




class _FrameGrabber:
    """Background thread that continuously reads an RTSP stream and caches the
    latest frame. Reconnects automatically on stream errors."""

    _RECONNECT_DELAY = 2.0  # seconds between reconnect attempts

    def __init__(self, rtsp_url: str) -> None:
        self._url = rtsp_url
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._read_loop, daemon=True, name=f"grabber-{rtsp_url}")
        self._thread.start()

    def get_latest(self) -> Optional[np.ndarray]:
        """Return a copy of the most recently captured frame, or None."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _read_loop(self) -> None:
        while True:
            cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            connected = cap.isOpened()
            if not connected:
                cap.release()
                time.sleep(self._RECONNECT_DELAY)
                continue

            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # stream error — reconnect
                with self._lock:
                    self._frame = frame

            cap.release()
            time.sleep(self._RECONNECT_DELAY)


# Registry: camera_id -> _FrameGrabber
_grabbers: dict[str, _FrameGrabber] = {}
_grabbers_lock = threading.Lock()


def _get_grabber(camera_id: str) -> _FrameGrabber:
    with _grabbers_lock:
        if camera_id not in _grabbers:
            url = build_rtsp_url(camera_id)
            log.info("Starting persistent RTSP grabber for camera=%s url=%s", camera_id, url)
            _grabbers[camera_id] = _FrameGrabber(url)
        return _grabbers[camera_id]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def crop_bbox(frame: np.ndarray, bbox: dict, padding: int = 10) -> Optional[np.ndarray]:
    """Crop a region from frame using {x, y, width, height} top-left bbox dict."""
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox.get("x", 0)) - padding)
    y1 = max(0, int(bbox.get("y", 0)) - padding)
    x2 = min(w, int(bbox.get("x", 0)) + int(bbox.get("width", 0)) + padding)
    y2 = min(h, int(bbox.get("y", 0)) + int(bbox.get("height", 0)) + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def frame_to_base64_jpeg(image: np.ndarray, quality: int = 80) -> Optional[str]:
    """Encode a numpy image to a base64 JPEG string."""
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def capture_thumbnail(camera_id: str, bbox: Optional[dict], timestamp: str = "") -> Optional[str]:
    """Return a base64 JPEG for camera_id at the detection moment.

    MQTT image path (cameras in MQTT_IMAGE_CAMERAS):
      Looks up the ring-buffered frame whose pipeline timestamp is closest to
      the detection timestamp.  The sscape_adapter embeds the same
      `postdecode_timestamp` in both the image and detection payloads, so the
      closest match is the exact frame being processed when the detection fired.

      Falls back to request_frame_and_wait() if the ring buffer is empty (e.g.
      MQTT just reconnected) — this blocks briefly but only happens once.

    RTSP fallback:
      Used for cameras not in MQTT_IMAGE_CAMERAS.
    """
    if use_mqtt_image(camera_id):
        sub = _get_mqtt_subscriber(camera_id)
        b64 = sub.get_frame_for_timestamp(timestamp, max_age_sec=3.0)
        if b64 is None:
            # Ring buffer is empty — wait for the next heartbeat to deliver a frame
            log.info("Ring buffer empty for camera=%s, waiting for frame", camera_id)
            b64 = sub.request_frame_and_wait(timeout=3.0)
        if b64:
            return b64
        log.warning("No MQTT image for camera=%s — falling back to RTSP", camera_id)

    # RTSP fallback
    grabber = _get_grabber(camera_id)
    frame = grabber.get_latest()
    if frame is None:
        log.warning("No cached RTSP frame for camera=%s", camera_id)
        return None

    crop = frame
    if bbox:
        c = crop_bbox(frame, bbox)
        if c is not None and c.size > 0:
            crop = c

    b64 = frame_to_base64_jpeg(crop)
    if b64 is None:
        log.warning("Failed to encode thumbnail for camera=%s", camera_id)
    return b64


def submit_capture(camera_id: str, bbox: Optional[dict], timestamp: str = ""):
    """Submit a thumbnail capture to the shared thread pool. Returns a Future."""
    return _executor.submit(capture_thumbnail, camera_id, bbox, timestamp)


# ---------------------------------------------------------------------------
# Inline per-camera frame cache
#
# Populated by MQTTAdapter when it receives scenescape/image/camera/+ messages
# (same MQTT connection as detection messages, so image always arrives BEFORE
# the detection on the same connection — guaranteed in-order delivery).
#
# grab_frame_now() reads from this cache synchronously with zero delay.
# ---------------------------------------------------------------------------

# camera_id → (pipeline_timestamp_str, base64_jpeg)
_inline_cache: dict[str, tuple[str, str]] = {}

def notify_frame(camera_id: str, timestamp: str, b64: str) -> None:
    """Store the latest frame for a camera.  Called from the MQTT adapter on
    the same thread that processes detections, so no lock is needed."""
    _inline_cache[camera_id] = (timestamp, b64)
    # Also push into the ring buffer for the heartbeat fallback path
    sub = _mqtt_subscribers.get(camera_id)
    if sub is not None:
        with sub._cond:
            sub._ring.append((timestamp, b64))
            sub._latest_b64 = b64
            sub._cond.notify_all()


def grab_frame_now(camera_id: str, timestamp: str = "") -> Optional[str]:
    """Synchronously return the best available frame for camera_id.

    Priority:
    1. Inline cache (updated by MQTTAdapter image subscription — same connection
       as detection, so image arrives BEFORE detection for the same frame).
    2. Ring buffer (updated by _MqttImageSubscriber heartbeat — fallback when
       the adapter connection is not carrying image messages).
    3. RTSP grabber (last resort).

    This function never blocks and never calls out to the network.
    """
    # 1. Inline cache — best source
    cached = _inline_cache.get(camera_id)
    if cached:
        cache_ts, cache_b64 = cached
        if not timestamp:
            return cache_b64
        # Accept if the cached frame is within 2 seconds of the detection
        try:
            delta = abs(_parse_pipeline_ts(cache_ts) - _parse_pipeline_ts(timestamp))
            if delta <= 2.0:
                log.debug("grab_frame_now: inline cache hit camera=%s delta=%.3fs", camera_id, delta)
                return cache_b64
        except (ValueError, TypeError):
            return cache_b64

    # 2. Ring buffer (separate subscriber + heartbeat)
    sub = _mqtt_subscribers.get(camera_id)
    if sub is not None:
        b64 = sub.get_frame_for_timestamp(timestamp, max_age_sec=3.0)
        if b64:
            log.debug("grab_frame_now: ring buffer hit camera=%s", camera_id)
            return b64

    # 3. RTSP grabber
    if camera_id in _grabbers:
        frame = _grabbers[camera_id].get_latest()
        if frame is not None:
            return frame_to_base64_jpeg(frame)

    log.debug("grab_frame_now: no frame available for camera=%s", camera_id)
    return None


def prewarm_grabbers(camera_ids: list[str]) -> None:
    """Start persistent grabbers and MQTT image subscribers for all cameras
    immediately, so they are ready before the first match event."""
    for cam in camera_ids:
        _get_grabber(cam)
        if use_mqtt_image(cam):
            sub = _get_mqtt_subscriber(cam)
            # Request an initial frame so the cache is warm before first match
            sub.request_frame()
            log.info("Pre-warming MQTT image subscriber for camera=%s", cam)
