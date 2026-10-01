"""One-way event-hub delivery for committed SAD envelopes."""

from __future__ import annotations

import time
import urllib.error
import urllib.request

from events import EventEnvelope


def _post(hub_url: str, event: EventEnvelope, timeout_s: float) -> bool:
    request = urllib.request.Request(
        hub_url,
        data=json.dumps({
            "type": "mcp_event",
            "event": json.loads(event.to_json()),
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout_s) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def push_event_to_hub(
    hub_url: str | None,
    event: EventEnvelope,
    retries: int = 3,
    timeout_s: float = 5.0,
) -> bool:
    """Push one already-persisted event envelope to the configured hub."""
    if not hub_url:
        return False
    for attempt in range(retries):
        if _post(hub_url, event, timeout_s):
            return True
        if attempt + 1 < retries:
            time.sleep(0.5 * (2**attempt))
    print(f"[SAD MCP] Event hub delivery failed after {retries} attempts: {hub_url}", flush=True)
    return False