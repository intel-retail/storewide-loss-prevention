# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
MQTT-based queue consumer for Behavioral Analysis visit lifecycle events.

Topic ``ba/requests`` carries two kinds of messages, distinguished by the
``action`` field:

* ``action == "start"`` -- swlp-service published once when a person enters
  a HIGH_VALUE zone. We spawn a polling worker keyed by
  ``{scene_id}/{person_id}/{region_id}/{entry_timestamp}`` that watches the
  matching prefix in the SeaweedFS ``behavioral-frames`` bucket.

* ``action == "exit"`` -- swlp-service published once when the visit ends.
  We cancel the worker.

Each worker polls the bucket on a fixed cadence
(``settings.visit_poll_interval``). When at least
``settings.min_frames_for_detection`` *new* frames have accumulated since
the last analysis (tracked via a per-worker watermark on the frame
timestamp), the worker runs pose extraction + VLM and publishes one
``ba/results`` message. It keeps polling so that multiple discrete
concealment events in the same visit each produce their own alert.
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


def _visit_key(scene_id: str, person_id: str, region_id: str,
               entry_timestamp: str) -> str:
    return f"{scene_id}/{person_id}/{region_id}/{entry_timestamp}"


class BAQueueConsumer:
    """Consumes visit lifecycle events and runs per-visit polling workers."""

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
        self.poll_interval = settings.visit_poll_interval

        self.client: Optional[mqtt.Client] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.connected = False
        self._shutdown = asyncio.Event()
        # Per-visit worker tasks, keyed by _visit_key().
        self._workers: dict[str, asyncio.Task] = {}
        # Cooperative-stop flags, keyed by _visit_key(). The polling loop
        # checks this each iteration; in-flight pose/VLM calls are NOT
        # cancelled and run to completion.
        self._stop_flags: dict[str, bool] = {}

    def initialize(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.client = mqtt.Client(client_id="ba-queue-consumer")
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    async def start(self) -> None:
        logger.info(
            "BA queue consumer connecting to MQTT",
            extra={"host": self.settings.mqtt_host, "port": self.settings.mqtt_port},
        )
        self.client.connect_async(
            self.settings.mqtt_host, self.settings.mqtt_port, keepalive=60
        )
        self.client.loop_start()
        await self._shutdown.wait()

    async def stop(self) -> None:
        self._shutdown.set()
        # Signal all workers to stop after their current iteration.
        for key in list(self._workers.keys()):
            self._stop_flags[key] = True
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        logger.info("BA queue consumer stopped")

    def publish_result(self, result: dict) -> None:
        if self.client and self.connected:
            self.client.publish(
                self.result_topic, json.dumps(result), qos=1
            )
            logger.info(
                "Published BA result",
                extra={
                    "person_id": result.get("person_id"),
                    "status": result.get("status"),
                },
            )

    # ---- paho callbacks ------------------------------------------------------

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            self.connected = True
            client.subscribe(self.request_topic, qos=1)
            logger.info(
                f"BA queue consumer connected, subscribed to {self.request_topic}"
            )
        else:
            logger.error(f"BA queue consumer MQTT connect failed, rc={rc}")

    def _on_disconnect(self, client, userdata, rc) -> None:
        self.connected = False
        logger.warning(f"BA queue consumer MQTT disconnected, rc={rc}")

    def _on_message(self, client, userdata, msg: mqtt.MQTTMessage) -> None:
        if msg.topic != self.request_topic:
            return
        try:
            payload = json.loads(msg.payload)
        except json.JSONDecodeError:
            logger.error("Invalid JSON in BA request message")
            return

        action = payload.get("action", "start")
        person_id = payload.get("person_id", "")
        region_id = payload.get("region_id", "")
        entry_timestamp = payload.get("entry_timestamp", "")
        scene_id = payload.get("scene_id", "")
        if not person_id:
            logger.warning("BA message missing person_id, skipping")
            return

        key = _visit_key(scene_id, person_id, region_id, entry_timestamp)

        if action == "exit":
            self._stop_worker(key)
            return

        # Default action is "start" (also accepts legacy messages with no action).
        if action not in ("start",):
            logger.debug(f"Ignoring unknown BA action: {action}")
            return

        if not self.loop:
            return
        # Spawn the worker on the asyncio loop.
        self.loop.call_soon_threadsafe(
            self._spawn_worker, key, person_id, region_id, entry_timestamp, scene_id
        )

    # ---- worker management ---------------------------------------------------

    def _spawn_worker(self, key: str, person_id: str, region_id: str,
                      entry_timestamp: str, scene_id: str) -> None:
        existing = self._workers.get(key)
        if existing and not existing.done():
            logger.debug(f"BA worker already running for {key}, skipping spawn")
            return
        self._workers[key] = asyncio.create_task(
            self._run_worker(key, person_id, region_id, entry_timestamp, scene_id)
        )

    def _stop_worker(self, key: str) -> None:
        """Signal the worker to stop after its current iteration.

        We do NOT cancel the task: any in-flight pose extraction / VLM call
        is allowed to finish (and publish its result). Only the polling
        loop is asked to exit, so no further bucket fetches happen.
        """
        if key in self._workers:
            self._stop_flags[key] = True
            logger.info(f"BA worker stop requested for {key}")

    async def _run_worker(self, key: str, person_id: str, region_id: str,
                          entry_timestamp: str, scene_id: str) -> None:
        """Per-visit polling loop.

        Watches the bucket prefix for this (scene/person/region/entry_ts);
        whenever ``min_frames`` new frames have arrived since the last
        analysis, runs pose+VLM and publishes a result. Continues until
        cancelled by an ``exit`` message.
        """
        logger.info(
            f"BA worker started for {key} (poll={self.poll_interval}s, "
            f"min_frames={self.min_frames})"
        )
        watermark_ts = 0  # only frames with timestamp > watermark are "new"
        self._stop_flags[key] = False
        try:
            while True:
                if self._stop_flags.get(key):
                    logger.info(f"BA worker exiting cleanly for {key}")
                    return
                await asyncio.sleep(self.poll_interval)
                if self._stop_flags.get(key):
                    logger.info(f"BA worker exiting cleanly for {key}")
                    return
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
                    logger.exception(f"Frame fetch failed for {key}")
                    continue

                # Pick frames newer than the last analysis.
                new_frames = [(f, ts) for (f, ts) in frames if ts > watermark_ts]
                if len(new_frames) < self.min_frames:
                    continue

                # Analyse the latest batch and bump the watermark to the
                # newest analysed frame so the next cycle waits for the next
                # batch of fresh frames.
                await self._analyze_batch(
                    person_id, region_id, entry_timestamp, scene_id, new_frames
                )
                watermark_ts = max(ts for _, ts in new_frames)
        except asyncio.CancelledError:
            logger.debug(f"BA worker loop cancelled for {key}")
            raise
        except Exception:
            logger.exception(f"BA worker crashed for {key}")
        finally:
            self._workers.pop(key, None)
            self._stop_flags.pop(key, None)

    # ---- single-batch analysis -----------------------------------------------

    async def _analyze_batch(
        self, person_id: str, region_id: str, entry_timestamp: str,
        scene_id: str, frames: list,
    ) -> None:
        """Run pose + VLM on a batch of frames and publish exactly one result."""
        frames_available = len(frames)
        try:
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

            if result.matched:
                logger.warning(
                    f"Entity {person_id}: pose pattern matched "
                    f"(confidence={result.confidence:.3f}), calling VLM"
                )
                if self.settings.vlm_enabled and self.pose_analyzer.vlm_client:
                    result = await self.pose_analyzer.analyze_with_vlm(
                        frames=frames,
                        pose_result=result,
                    )
                vlm_response = None
                if result.vlm_result:
                    vlm_response = result.vlm_result.get("reasoning")

                if result.vlm_confirmed is True:
                    self.publish_result({
                        "person_id": person_id, "region_id": region_id,
                        "entry_timestamp": entry_timestamp, "scene_id": scene_id,
                        "status": "suspicious",
                        "confidence": result.confidence,
                        "vlm_response": vlm_response,
                        "frames_analyzed": frames_available,
                    })
                    return
                # VLM disagreed or failed -> not suspicious for this batch.
                logger.info(
                    f"Entity {person_id}: VLM did not confirm "
                    f"(vlm_confirmed={result.vlm_confirmed})"
                )
                self.publish_result({
                    "person_id": person_id, "region_id": region_id,
                    "entry_timestamp": entry_timestamp, "scene_id": scene_id,
                    "status": "no_match",
                    "confidence": result.confidence,
                    "vlm_response": vlm_response,
                    "frames_analyzed": frames_available,
                })
                return

            self.publish_result({
                "person_id": person_id, "region_id": region_id,
                "entry_timestamp": entry_timestamp, "scene_id": scene_id,
                "status": "no_match",
                "confidence": result.confidence,
                "vlm_response": None,
                "frames_analyzed": frames_available,
            })
        except Exception:
            logger.exception(f"Error analysing batch for {person_id}")
