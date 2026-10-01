"""Start FastMCP and the MQTT event-ingest listener."""

from __future__ import annotations

from config import get_settings
from mqtt_ingest import start_alert_ingest_listener
from tools import ingest_alert, mcp


def main() -> None:
    s = get_settings()
    start_alert_ingest_listener(s, ingest_alert)
    if s.transport in {"http", "streamable-http"}:
        mcp.run(transport="http", host=s.host, port=s.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
