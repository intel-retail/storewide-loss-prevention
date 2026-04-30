# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
MQTT-based queue consumer for Behavioral Analysis requests.

Topic ``ba/requests`` carries one message per fresh frame stored by
swlp-service. Each message is fully self-describing:

    {
      "scene_id":        "...",
      "person_id":       "...",
      "region_id":       "...",
      "entry_timestamp": "20260429T113629"
    }

The consumer is **stateless and single-shot**: for each message it fetches
the latest K frames from SeaweedFS for that visit, runs pose extraction +
VLM, and publishes exactly one ``ba/results`` message in response.

Requests are processed serially via an internal ``asyncio.Queue`` so that
multiple messages do not pile up concurrent VLM calls. There is no
per-person worker, no polling loop, and no ``action`` (start/exit) field.
"""

import asyncio
import json
import logging
from typing import Optional

import paho.mqtt.client as mqtt

from config import Settings
from pose_analyzer import PatternResult
from yolo_pipeline import extract_poses

logger = logging.getLogger(__name__)


class BAQueueConsumer:
    """Stateless single-shot consumer of ``ba/requests`` events."""

    def __init__(
        self,
        settings: Settings,
        frame_store=None,
        pose_analyzer=None,
    ) -> None:
        self.settings = settings
        self.request_topic = settings.ba_request_topic
        self.result_topic = settings.ba_result_topic
        self.frame_store = frame_store
        self.pose_analyzer = pose_analyzer
        self.min_frames = settings.min_frames_for_detection

        self.client: Optional[mqtt.Client] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.connected = False
        self._shutdown = asyncio.Event()
        # Bounded internal queue so we apply natural backpressure when BA is
        # slower than the publisher. Old requests are dropped if the queue
        # is full -- the latest frame is always more interesting than stale ones.
        self._queue: Optional[asyncio.Queue] = None
        self._worker_task: Optional[asyncio.Task] = None

    def initialize(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self._queue = asyncio.Queue(maxsize=64)
        self.client = mqtt.Client(client_id="ba-queue-consumer")
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    async def start(self) -> None:
        logger.info(
            "BA queue consumer connecting to MQTT host=%s port=%s",
            self.settings.mqtt_host, self.settings.mqtt_port,
        )
        self.client.connect_async(
            self.settings.mqtt_host, self.settings.mqtt_port, keepalive=60
        )
        self.client.loop_start()
        self._worker_task = asyncio.create_task(self._worker())
        await self._shutdown.wait()

    async def stop(self) -> None:
        self._shutdown.set()
        if self._worker_task:
            self._worker_task.cancel()
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        logger.info("BA queue consumer stopped")

    def publish_result(self, result: dict) -> None:
        if self.client and self.connected:
            self.client.publish(self.result_topic, json.dumps(result), qos=1)
            logger.info(
                "Published BA result person_id=%s status=%s",
                result.get("person_id"), result.get("status"),
            )

    # ---- paho callbacks ------------------------------------------------------

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            self.connected = True
            client.subscribe(self.request_topic, qos=1)
            logger.info(
                "BA queue consumer connected, subscribed to %s",
                self.request_topic,
            )
        else:
            logger.error("BA queue consumer MQTT connect failed, rc=%s", rc)

    def _on_disconnect(self, client, userdata, rc) -> None:
        self.connected = False
        logger.warning("BA queue consumer MQTT disconnected, rc=%s", rc)

    def _on_message(self, client, userdata, msg: mqtt.MQTTMessage) -> None:
        if msg.topic != self.request_topic:
            return
        try:
            payload = json.loads(msg.payload)
        except json.JSONDecodeError:
            logger.error("Invalid JSON in BA request message")
            return

        person_id = payload.get("person_id", "")
        region_id = payload.get("region_id", "")
        entry_timestamp = payload.get("entry_timestamp", "")
        scene_id = payload.get("scene_id", "")
        if not person_id:
            logger.warning("BA message missing person_id, skipping")
            return

        if not self.loop or not self._queue:
            return

        request = (person_id, region_id, entry_timestamp, scene_id)
        # Threadsafe enqueue from the paho thread onto the asyncio queue.
        self.loop.call_soon_threadsafe(self._enqueue, request)

    def _enqueue(self, request: tuple) -> None:
        if self._queue is None:
            return
        try:
            self._queue.put_nowait(request)
        except asyncio.QueueFull:
            # Drop oldest, keep newest -- the latest frame is always more
            # relevant for the analysis than a stale one.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(request)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    # ---- worker --------------------------------------------------------------

    async def _worker(self) -> None:
        """Single serial worker: process one BA request to completion at a time."""
        assert self._queue is not None
        while not self._shutdown.is_set():
            try:
                request = await self._queue.get()
            except asyncio.CancelledError:
                return
            person_id, region_id, entry_timestamp, scene_id = request
            try:
                await self._handle_request(
                    person_id, region_id, entry_timestamp, scene_id
                )
            except Exception:
                logger.exception(
                    "Unhandled error processing BA request person_id=%s",
                    person_id,
                )

    # ---- single request analysis --------------------------------------------

    async def _handle_request(
        self, person_id: str, region_id: str,
        entry_timestamp: str, scene_id: str,
    ) -> None:
        """Fetch latest K frames, run pose+VLM, publish one ba/results."""
        try:
            frames = await self.frame_store.get_frames(
                entity_id=person_id,
                max_frames=self.settings.max_frames_to_fetch,
                max_age_seconds=0,
                region_id=region_id,
                entry_timestamp=entry_timestamp,
                scene_id=scene_id,
            )
        except Exception:
            logger.exception(
                "Frame fetch failed person_id=%s region_id=%s",
                person_id, region_id,
            )
            return

        # frame_store.get_frames returns [(ndarray_bgr, timestamp_ms), ...].
        # extract_poses() consumes the same tuple shape directly.
        frames_available = len(frames)
        if frames_available < self.min_frames:
            logger.debug(
                "Skipping BA request: %d/%d frames",
                frames_available, self.min_frames,
            )
            return

        pose_frames = frames[-self.settings.pose_frames_count:]
        poses = await extract_poses(pose_frames, person_id, self.settings)

        if not poses:
            self.publish_result({
                "person_id": person_id, "region_id": region_id,
                "entry_timestamp": entry_timestamp, "scene_id": scene_id,
                "status": "no_match", "confidence": 0.0,
                "vlm_response": None, "frames_analyzed": frames_available,
            })
            return

        results = self.pose_analyzer.detect_all_patterns(poses)
        matched = [r for r in results if r.matched]
        result = (
            max(matched, key=lambda r: r.confidence)
            if matched
            else results[0] if results
            else PatternResult(
                matched=False, confidence=0.0,
                pattern_id="shelf_to_waist",
                description="No patterns evaluated",
            )
        )

        if not result.matched:
            self.publish_result({
                "person_id": person_id, "region_id": region_id,
                "entry_timestamp": entry_timestamp, "scene_id": scene_id,
                "status": "no_match",
                "confidence": result.confidence,
                "vlm_response": None,
                "frames_analyzed": frames_available,
            })
            return

        logger.warning(
            "Entity %s: pose pattern matched (confidence=%.3f), calling VLM",
            person_id, result.confidence,
        )
        if self.settings.vlm_enabled and self.pose_analyzer.vlm_client:
            result = await self.pose_analyzer.analyze_with_vlm(
                frames=frames,
                pose_result=result,
            )
        vlm_response = None
        if result.vlm_result:
            vlm_response = result.vlm_result.get("reasoning")

        status = "suspicious" if result.vlm_confirmed is True else "no_match"
        if result.vlm_confirmed is not True:
            logger.info(
                "Entity %s: VLM did not confirm (vlm_confirmed=%s)",
                person_id, result.vlm_confirmed,
            )

        self.publish_result({
            "person_id": person_id, "region_id": region_id,
            "entry_timestamp": entry_timestamp, "scene_id": scene_id,
            "status": status,
            "confidence": result.confidence,
            "vlm_response": vlm_response,
            "frames_analyzed": frames_available,
        })
