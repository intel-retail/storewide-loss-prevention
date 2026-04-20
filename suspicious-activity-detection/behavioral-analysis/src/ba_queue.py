# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
MQTT-based queue consumer for Behavioral Analysis requests.

Subscribes to ba/requests, runs pose + VLM analysis, publishes results to ba/results.
"""

import asyncio
import json
import logging
from typing import Optional

import paho.mqtt.client as mqtt

from config import Settings

logger = logging.getLogger(__name__)


class BAQueueConsumer:
    """
    Consumes analysis requests from MQTT, runs pose + VLM analysis,
    and publishes results.

    Request topic (ba/requests) payload:
        {"person_id": str, "region_id": str, "entry_timestamp": str}

    Result topic (ba/results) payload:
        {"person_id": str, "region_id": str, "entry_timestamp": str,
         "status": str, "confidence": float|null, "vlm_response": str|null}
    """

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
        self._processing: set[str] = set()  # dedup in-flight requests

    def initialize(self, loop: asyncio.AbstractEventLoop) -> None:
        """Create MQTT client and configure callbacks."""
        self.loop = loop
        self.client = mqtt.Client(client_id="ba-queue-consumer")
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    async def start(self) -> None:
        """Connect to broker and run the MQTT loop."""
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
        """Disconnect from broker."""
        self._shutdown.set()
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        logger.info("BA queue consumer stopped")

    def publish_result(self, result: dict) -> None:
        """Publish an analysis result to ba/results."""
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
            logger.info(f"BA queue consumer connected, subscribed to {self.request_topic}")
        else:
            logger.error(f"BA queue consumer MQTT connect failed, rc={rc}")

    def _on_disconnect(self, client, userdata, rc) -> None:
        self.connected = False
        logger.warning(f"BA queue consumer MQTT disconnected, rc={rc}")

    def _on_message(self, client, userdata, msg: mqtt.MQTTMessage) -> None:
        """Handle incoming ba/requests — dispatch to async analysis."""
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
            logger.warning("BA request missing person_id, skipping")
            return

        # Dedup: skip if already processing this person+region
        dedup_key = f"{person_id}:{region_id}"
        if dedup_key in self._processing:
            return
        self._processing.add(dedup_key)

        # Dispatch async analysis
        if self.loop:
            asyncio.run_coroutine_threadsafe(
                self._analyze(person_id, region_id, entry_timestamp, dedup_key, scene_id),
                self.loop,
            )

    async def _analyze(
        self, person_id: str, region_id: str, entry_timestamp: str, dedup_key: str,
        scene_id: str = "",
    ) -> None:
        """
        Core analysis pipeline:
        1. Fetch frames from SeaweedFS
        2. If enough frames -> run pose detection on last 10
        3. If suspicious -> call VLM with all frames
        4. Publish result
        """
        try:
            # Step 1: Fetch frames
            frames = await self.frame_store.get_frames(
                entity_id=person_id,
                max_frames=self.settings.max_frames_to_fetch,
                max_age_seconds=0,
                region_id=region_id,
                entry_timestamp=entry_timestamp,
                scene_id=scene_id,
            )

            frames_available = len(frames)

            # Step 2: Not enough frames — stay silent, swlp re-publishes in 1s
            if frames_available < self.min_frames:
                logger.debug(
                    f"Entity {person_id}: {frames_available}/{self.min_frames} frames, need more"
                )
                return

            # Step 3: Run pose on last 10 frames
            last_10 = frames[-10:]
            pose_sequence = self.pose_analyzer.extract_poses(last_10)

            if len(pose_sequence) < self.min_frames:
                logger.debug(
                    f"Entity {person_id}: only {len(pose_sequence)} poses extracted, need more"
                )
                return

            result = self.pose_analyzer.detect_pattern(
                pose_sequence=pose_sequence,
                pattern_id="shelf_to_waist",
            )

            # Step 4: If suspicious -> call VLM with ALL frames
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

                # Only report "suspicious" if VLM confirmed or VLM was not used
                if result.vlm_confirmed is False:
                    # VLM explicitly disagreed — not suspicious
                    logger.info(
                        f"Entity {person_id}: VLM overruled pose match "
                        f"(confidence={result.confidence:.3f})"
                    )
                    self.publish_result({
                        "person_id": person_id,
                        "region_id": region_id,
                        "entry_timestamp": entry_timestamp,
                        "scene_id": scene_id,
                        "status": "no_match",
                        "confidence": result.confidence,
                        "vlm_response": vlm_response,
                        "frames_analyzed": frames_available,
                    })
                else:
                    self.publish_result({
                        "person_id": person_id,
                        "region_id": region_id,
                        "entry_timestamp": entry_timestamp,
                        "scene_id": scene_id,
                        "status": "suspicious",
                        "confidence": result.confidence,
                        "vlm_response": vlm_response,
                        "frames_analyzed": frames_available,
                    })
            else:
                # Step 5: Not suspicious
                logger.info(
                    f"Entity {person_id}: no match (confidence={result.confidence:.3f})"
                )
                self.publish_result({
                    "person_id": person_id,
                    "region_id": region_id,
                    "entry_timestamp": entry_timestamp,
                    "scene_id": scene_id,
                    "status": "no_match",
                    "confidence": result.confidence,
                    "vlm_response": None,
                    "frames_analyzed": frames_available,
                })

        except Exception:
            logger.exception(f"Error analyzing entity {person_id}")
        finally:
            self._processing.discard(dedup_key)
