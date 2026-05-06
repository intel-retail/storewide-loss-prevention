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
from concurrent.futures import ThreadPoolExecutor
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

class _MqttImageSubscriber:
    """Subscribes to scenescape/image/camera/{camera_id}.

    The DLStreamer sscape_adapter publishes ONE frame per `getimage` command
    (publish_image flag is reset after each publish).  This class sends the
    command and uses a threading.Condition to wait for exactly that response,
    guaranteeing the returned frame matches the detection moment.
    """

    def __init__(self, camera_id: str, mqtt_host: str, mqtt_port: int) -> None:
        import paho.mqtt.client as mqtt  # type: ignore[import]
        self._camera_id = camera_id
        self._host = mqtt_host
        self._port = mqtt_port

        self._latest_b64: Optional[str] = None
        self._cond = threading.Condition(threading.Lock())

        self._image_topic = f"scenescape/image/camera/{camera_id}"
        self._cmd_topic   = f"scenescape/cmd/camera/{camera_id}"

        self._client = mqtt.Client(client_id=f"poi-thumbnail-{camera_id}")
        self._client.on_connect    = self._on_connect
        self._client.on_message    = self._on_message
        self._client.on_disconnect = self._on_disconnect

        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"mqtt-img-{camera_id}"
        )
        self._thread.start()

    def _run(self) -> None:
        while True:
            try:
                self._client.connect(self._host, self._port, keepalive=30)
                self._client.loop_forever()
            except Exception as exc:
                log.warning("MQTT image subscriber disconnected camera=%s: %s", self._camera_id, exc)
            time.sleep(3)

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
        try:
            import json as _json
            data = _json.loads(msg.payload)
            b64 = data.get("image")
            if b64:
                with self._cond:
                    self._latest_b64 = b64
                    self._cond.notify_all()   # wake everyone waiting for a frame
                log.debug("MQTT image received camera=%s len=%d", self._camera_id, len(b64))
        except Exception as exc:
            log.debug("MQTT image parse error camera=%s: %s", self._camera_id, exc)

    def request_frame(self) -> None:
        """Ask the DLStreamer adapter to publish a fresh frame."""
        try:
            self._client.publish(self._cmd_topic, "getimage", qos=0)
            log.debug("Sent getimage for camera=%s", self._camera_id)
        except Exception as exc:
            log.debug("Failed to send getimage camera=%s: %s", self._camera_id, exc)

    def request_frame_and_wait(self, timeout: float = 3.0) -> Optional[str]:
        """Send getimage and block until the pipeline publishes the response.

        Returns the base64 JPEG string, or None on timeout.
        The pipeline processes getimage in the next video frame cycle (~100 ms),
        so the returned frame is guaranteed to be from the detection moment.
        """
        with self._cond:
            self.request_frame()
            # Wait for _on_message to deliver a new frame
            arrived = self._cond.wait(timeout=timeout)
            if not arrived:
                log.warning("Timeout waiting for MQTT image camera=%s", self._camera_id)
                return self._latest_b64   # return whatever we have (may be None)
            return self._latest_b64

    # Legacy helpers kept for prewarm / compatibility
    def get_latest_b64(self, wait_timeout: float = 2.0) -> Optional[str]:
        return self.request_frame_and_wait(timeout=wait_timeout)

    def get_latest_frame(self, wait_timeout: float = 2.0) -> Optional[np.ndarray]:
        b64 = self.get_latest_b64(wait_timeout)
        if b64 is None:
            return None
        try:
            raw = base64.b64decode(b64)
            buf = np.frombuffer(raw, dtype=np.uint8)
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception as exc:
            log.debug("MQTT image decode error camera=%s: %s", self._camera_id, exc)
            return None


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

    MQTT image path (preferred — cameras listed in MQTT_IMAGE_CAMERAS):
      Sends a getimage command to the DLStreamer pipeline and waits for the
      response.  The pipeline captures the current video frame and publishes it
      within ~100 ms (next pipeline cycle).  This frame is from the detection
      moment, not a stale RTSP cache.

    RTSP fallback:
      Used for cameras not configured in MQTT_IMAGE_CAMERAS.
    """
    if use_mqtt_image(camera_id):
        sub = _get_mqtt_subscriber(camera_id)
        b64 = sub.request_frame_and_wait(timeout=3.0)
        if b64 is None:
            log.warning("No MQTT image received for camera=%s — falling back to RTSP", camera_id)
        else:
            return b64

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
