#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Subscribe to SceneScape region events and save to a JSON file on disk."""

import json
import os
import ssl
import subprocess
import sys
from datetime import datetime, timezone

import paho.mqtt.client as mqtt


def resolve_broker_host():
    """Auto-discover the MQTT broker IP from the Docker container."""
    host = os.environ.get("MQTT_HOST")
    if host:
        return host

    # Try to resolve from Docker
    container_names = ["scenescape-broker-1", "broker"]
    for name in container_names:
        try:
            result = subprocess.run(
                ["docker", "inspect", name, "-f",
                 "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}"],
                capture_output=True, text=True, timeout=5,
            )
            ip = result.stdout.strip()
            if ip:
                print(f"Auto-discovered broker IP from '{name}': {ip}")
                return ip
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

    # Fallback: try the Docker DNS name (works if running inside the network)
    return "broker.scenescape.intel.com"


BROKER = resolve_broker_host()
PORT = int(os.environ.get("MQTT_PORT", "1883"))
CA_CERT = os.environ.get("CA_CERT", "secrets/certs/scenescape-ca.pem")
SCENE_ID = os.environ.get("SCENE_ID", "3bc091c7-e449-46a0-9540-29c499bca18c")
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "region_events.json")

TOPIC = f"scenescape/event/region/{SCENE_ID}/#"


def on_connect(client, userdata, flags, rc, properties=None):
    print(f"Connected (rc={rc})")
    print(f"Subscribed to: {TOPIC}")
    print(f"Writing events to: {os.path.abspath(OUTPUT_FILE)}")
    client.subscribe(TOPIC, 1)


def on_message(client, userdata, msg, properties=None):
    try:
        data = json.loads(msg.payload)
    except json.JSONDecodeError:
        print(f"Non-JSON message on {msg.topic} ({len(msg.payload)} bytes)")
        return

    event = {
        "topic": msg.topic,
        "received_at": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }

    # Print summary
    region_name = data.get("region_name", "?")
    entered = [o.get("id", "?") for o in data.get("entered", [])]
    exited = [e.get("object", e).get("id", "?") for e in data.get("exited", [])]
    counts = data.get("counts", {})
    print(f"[{event['received_at']}] {region_name} | counts={counts} entered={entered} exited={exited}")

    # Append to JSON file (read existing, append, write back)
    events = []
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            try:
                events = json.load(f)
            except json.JSONDecodeError:
                events = []

    events.append(event)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(events, f, indent=2)


def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    if os.path.exists(CA_CERT):
        client.tls_set(ca_certs=CA_CERT, cert_reqs=ssl.CERT_NONE)
        client.tls_insecure_set(True)

    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Connecting to {BROKER}:{PORT}...")
    client.connect(BROKER, PORT, 60)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print(f"\nStopped. Events saved to {os.path.abspath(OUTPUT_FILE)}")
        client.disconnect()


if __name__ == "__main__":
    main()
