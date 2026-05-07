"""MQTT adapter for SceneScape event ingestion."""

from __future__ import annotations

import json
import logging
import re
import ssl
from typing import Callable, Optional

import paho.mqtt.client as mqtt

from backend.core.config import get_config

log = logging.getLogger("poi.mqtt")

# Primary camera topic: scenescape/data/camera/{camera_id}
# Contains face sub_objects with face-reidentification-retail-0095 embeddings
# — same embedding space as POI enrollment model.
_CAMERA_TOPIC_RE = re.compile(r"scenescape/data/camera/[^/]+$")

# External scene topic: scenescape/external/{scene_id}/person
# Carries global UUID, reid_state, and body-reid embeddings (person-reidentification-retail-0277).
# Body embeddings are a DIFFERENT space from face enrollment — used for monitoring only.
_EXTERNAL_TOPIC_RE = re.compile(r"scenescape/external/[^/]+/person$")
_EXTERNAL_TOPIC = "scenescape/external/+/person"

# Regulated scene topic: scenescape/regulated/scene/{scene_id}  — region entry/exit
_REGULATED_TOPIC_RE = re.compile(r"scenescape/regulated/scene/[^/]+$")
_REGULATED_TOPIC = "scenescape/regulated/scene/+"

# Image topic: scenescape/image/camera/{camera_id}
# Published by sscape_adapter on every getimage command — same processFrame() call as
# the detection, so the image message arrives BEFORE the detection on this same connection.
_IMAGE_TOPIC_RE = re.compile(r"scenescape/image/camera/(?P<camera_id>[^/]+)$")
_IMAGE_TOPIC = "scenescape/image/camera/+"


class MQTTAdapter:
    """Adapter Pattern — wraps paho-mqtt to subscribe to SceneScape events."""

    def __init__(self, on_event: Callable[[str, dict], None], on_region_event: Optional[Callable[[str, dict], None]] = None) -> None:
        self._cfg = get_config()
        self._on_event = on_event
        self._on_region_event = on_region_event
        self._client: Optional[mqtt.Client] = None
        self._running = False

    def start(self) -> None:
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)

        if self._cfg.mqtt_ca_cert:
            self._client.tls_set(
                ca_certs=self._cfg.mqtt_ca_cert,
                cert_reqs=ssl.CERT_REQUIRED,
                tls_version=ssl.PROTOCOL_TLS_CLIENT,
            )
            self._client.tls_insecure_set(True)

        log.info("Connecting to MQTT %s:%d", self._cfg.mqtt_host, self._cfg.mqtt_port)
        self._client.connect(self._cfg.mqtt_host, self._cfg.mqtt_port)
        self._running = True
        self._client.loop_start()

    def stop(self) -> None:
        self._running = False
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            log.info("MQTT disconnected")

    def _on_connect(self, client, userdata, flags, rc, properties=None) -> None:
        if rc == 0:
            # Subscribe to camera topic — face sub_objects have face-reid embeddings
            # (same model/space as enrollment) → primary source for FAISS matching
            camera_topic = self._cfg.mqtt_topic_event
            if camera_topic:
                client.subscribe(camera_topic)
                log.info("MQTT connected, subscribed to camera topic: %s", camera_topic)

            # Subscribe to external topic for monitoring (reid_state, UUIDs)
            client.subscribe(_EXTERNAL_TOPIC)
            log.info("MQTT subscribed to external topic (monitoring): %s", _EXTERNAL_TOPIC)

            # Subscribe to regulated scene topic for region entry/exit tracking
            if self._on_region_event is not None:
                client.subscribe(_REGULATED_TOPIC)
                log.info("MQTT subscribed to regulated scene topic: %s", _REGULATED_TOPIC)

            # Subscribe to image topic — same connection as detections so images
            # arrive BEFORE the corresponding detection message (in-order delivery).
            # The frame is cached in thumbnail._inline_cache and grabbed synchronously
            # by the consumer when it processes the detection — zero timing gap.
            client.subscribe(_IMAGE_TOPIC)
            log.info("MQTT subscribed to image topic: %s", _IMAGE_TOPIC)
        else:
            log.error("MQTT connection failed: rc=%s", rc)

    def _on_message(self, client, userdata, msg) -> None:
        if not self._running:
            return
        try:
            payload = json.loads(msg.payload)

            # Image topic: cache the frame so the detection handler can grab it
            # synchronously.  The image is published by sscape_adapter BEFORE the
            # detection for the same frame (same processFrame() call, same connection),
            # so by the time the detection message arrives here the frame is cached.
            m = _IMAGE_TOPIC_RE.match(msg.topic)
            if m:
                b64 = payload.get("image")
                ts  = payload.get("timestamp", "")
                if b64:
                    from backend.utils.thumbnail import notify_frame
                    notify_frame(m.group("camera_id"), ts, b64)
                return  # image messages are not forwarded to event handlers

            if self._on_region_event is not None and _REGULATED_TOPIC_RE.match(msg.topic):
                self._on_region_event(msg.topic, payload)
            elif _EXTERNAL_TOPIC_RE.match(msg.topic) or _CAMERA_TOPIC_RE.match(msg.topic):
                self._on_event(msg.topic, payload)
        except json.JSONDecodeError:
            log.warning("Invalid JSON on %s", msg.topic)
        except Exception:
            log.exception("Error handling MQTT message on %s", msg.topic)

    def _on_disconnect(self, client, userdata, flags, rc, properties=None) -> None:
        if self._running:
            log.warning("MQTT disconnected (rc=%s), reconnecting...", rc)

    @property
    def is_connected(self) -> bool:
        return self._client is not None and self._client.is_connected()
