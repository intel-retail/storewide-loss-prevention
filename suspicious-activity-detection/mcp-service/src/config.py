"""Env-driven settings for the SAD MCP service."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# Persistent by default so a separate seed/pipeline process and the running MCP
# server share the same durable log (an in-memory log would isolate them).
_DEFAULT_LOG_PATH = str(Path(__file__).resolve().parents[1] / "data" / "sad_log.sqlite")


@dataclass(frozen=True)
class Settings:
    store_id: str
    transport: str  # stdio | sse | streamable-http
    host: str
    port: int
    log_backend: str  # sqlite | jsonl | memory
    log_path: str
    delivery: str  # off | webhook
    webhook_url: str | None
    expose_subscribe: bool
    zone_config_path: str
    use_case: str
    store_timezone: str
    mqtt_ingest_enabled: bool
    mqtt_host: str
    mqtt_port: int
    mqtt_alert_topic: str


def configured_zones(path: str) -> list[str]:
    config_path = Path(path)
    if not config_path.is_file():
        return []

    data = json.loads(config_path.read_text(encoding="utf-8"))
    zones = data.get("zones", {})
    if isinstance(zones, dict):
        return sorted(str(zone) for zone in zones if zone)
    if isinstance(zones, list):
        out: list[str] = []
        for zone in zones:
            if isinstance(zone, str):
                out.append(zone)
            elif isinstance(zone, dict) and zone.get("name"):
                out.append(str(zone["name"]))
        return sorted(out)
    return []


def get_settings() -> Settings:
    log_backend = os.getenv("SAD_LOG_BACKEND", "sqlite")
    log_path = os.getenv("SAD_LOG_PATH", _DEFAULT_LOG_PATH)
    if log_backend != "memory":
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    return Settings(
        store_id=os.getenv("STORE_ID", "store_001"),
        transport=os.getenv("MCP_TRANSPORT", "stdio"),
        host=os.getenv("MCP_HOST", "0.0.0.0"),
        port=int(os.getenv("MCP_PORT", "9000")),
        log_backend=log_backend,
        log_path=log_path,
        delivery=os.getenv("SAD_DELIVERY", "off"),
        webhook_url=os.getenv("SAD_WEBHOOK_URL"),
        expose_subscribe=os.getenv("SAD_EXPOSE_SUBSCRIBE", "false").lower() == "true",
        zone_config_path=os.getenv("SAD_ZONE_CONFIG_PATH", "/app/zone_config.json"),
        use_case=os.getenv("USE_CASE", "retail"),
        store_timezone=os.getenv("SAD_TIMEZONE", "Asia/Kolkata"),
        mqtt_ingest_enabled=os.getenv("SAD_MQTT_INGEST_ENABLED", "true").lower() == "true",
        mqtt_host=os.getenv("SAD_MQTT_HOST", os.getenv("MQTT_HOST", "broker.scenescape.intel.com")),
        mqtt_port=int(os.getenv("SAD_MQTT_PORT", os.getenv("MQTT_PORT", "1883"))),
        mqtt_alert_topic=os.getenv("SAD_MQTT_ALERT_TOPIC", os.getenv("MQTT_ALERT_TOPIC", "alerts/#")),
    )
