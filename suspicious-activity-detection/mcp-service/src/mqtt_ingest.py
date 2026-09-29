"""MQTT alert ingestion for the SAD MCP durable log."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from config import Settings

IngestFn = Callable[..., None]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first(*values: Any, default: str = "") -> str:
    for value in values:
        if value is not None and value != "":
            return str(value)
    return default


def _event_name(alert_type: str, metadata: dict[str, Any], payload: dict[str, Any]) -> str:
    return _first(
        payload.get("event_type"),
        payload.get("event_name"),
        metadata.get("event_type"),
        alert_type.lower(),
        default="report_suspicious_activity",
    ).lower()


def _stable_ref_id(topic: str, alert: dict[str, Any], metadata: dict[str, Any]) -> str:
    explicit = _first(alert.get("alert_id"), metadata.get("alert_id"))
    if explicit:
        return explicit
    raw = json.dumps(alert, sort_keys=True, default=str)
    digest = hashlib.sha1(f"{topic}:{raw}".encode("utf-8")).hexdigest()[:16]
    return f"mqtt-{digest}"


def _source_timestamp_ms(alert: dict[str, Any], metadata: dict[str, Any], payload: dict[str, Any]) -> int | None:
    value = alert.get("timestamp") or metadata.get("timestamp") or payload.get("timestamp")
    if value in (None, ""):
        return None
    try:
        if isinstance(value, (int, float)):
            return int(value * 1000) if value < 100_000_000_000 else int(value)
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp() * 1000)
    except (OverflowError, TypeError, ValueError):
        return None


def normalize_alert(topic: str, alert: dict[str, Any], use_case: str) -> dict[str, Any]:
    metadata = _as_dict(alert.get("metadata"))
    payload = _as_dict(alert.get("payload"))
    evidence = _as_dict(payload.get("evidence"))
    alert_type = _first(alert.get("alert_type"), metadata.get("alert_type"), topic.rsplit("/", 1)[-1])
    event_name = _event_name(alert_type, metadata, payload)
    zone = _first(
        alert.get("region_name"),
        metadata.get("zone_name"),
        metadata.get("zone_id"),
        payload.get("zone"),
        payload.get("zone_id"),
        default="unknown",
    )
    description = _first(
        payload.get("description"),
        payload.get("message"),
        alert.get("description"),
        alert.get("message"),
        default=json.dumps(payload or metadata, sort_keys=True, default=str),
    )
    return {
        "zone": zone,
        "pose": _first(payload.get("pose"), payload.get("action"), event_name),
        "severity": _first(alert.get("alert_level"), metadata.get("severity"), payload.get("severity"), default="unknown").lower(),
        "camera_id": _first(alert.get("camera_id"), metadata.get("camera_id"), payload.get("camera_id"), evidence.get("camera_id"), default="unknown"),
        "object_id": _first(alert.get("object_id"), metadata.get("person_id"), payload.get("object_id"), payload.get("person_id"), default="unknown"),
        "description": description,
        "ref_id": _stable_ref_id(topic, alert, metadata),
        "event_name": event_name,
        "use_case": _first(alert.get("use_case"), metadata.get("use_case"), payload.get("use_case"), use_case),
        "frame": _first(alert.get("frame"), payload.get("frame"), payload.get("frame_uri"), evidence.get("frame"), evidence.get("uri")),
        "station": _first(metadata.get("station"), payload.get("station"), zone),
        "shift": _first(metadata.get("shift"), payload.get("shift"), default="unknown"),
        "ts_ms": _source_timestamp_ms(alert, metadata, payload),
    }


def start_alert_ingest_listener(settings: Settings, ingest: IngestFn) -> None:
    if not settings.mqtt_ingest_enabled:
        print("[SAD MCP] MQTT alert ingest disabled", flush=True)
        return

    def run() -> None:
        import paho.mqtt.client as mqtt

        def on_connect(client: Any, userdata: Any, flags: Any, rc: int) -> None:
            if rc == 0:
                client.subscribe(settings.mqtt_alert_topic, qos=1)
                print(f"[SAD MCP] Subscribed to {settings.mqtt_alert_topic}", flush=True)
            else:
                print(f"[SAD MCP] MQTT connect failed rc={rc}", flush=True)

        def on_message(client: Any, userdata: Any, msg: Any) -> None:
            try:
                alert = json.loads(msg.payload.decode("utf-8"))
                if not isinstance(alert, dict):
                    return
                ingest(**normalize_alert(msg.topic, alert, settings.use_case))
            except Exception as exc:
                print(f"[SAD MCP] Failed to ingest alert from {msg.topic}: {exc}", flush=True)

        backoff = 1
        while True:
            client = mqtt.Client()
            client.on_connect = on_connect
            client.on_message = on_message
            try:
                client.connect(settings.mqtt_host, settings.mqtt_port, keepalive=60)
                backoff = 1
                client.loop_forever()
            except Exception as exc:
                print(f"[SAD MCP] MQTT ingest connection error: {exc}; retrying in {backoff}s", flush=True)
            finally:
                try:
                    client.disconnect()
                except Exception:
                    pass
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)

    threading.Thread(target=run, name="sad-mcp-alert-ingest", daemon=True).start()