# SAD MCP Service — Integration Guide

The **Suspicious Activity Detection (SAD) MCP service** uses FastMCP for
standard MCP discovery, tool validation, and Streamable HTTP transport. SAD
owns its SQLite event log, MQTT ingestion, and callback delivery.

- **Service name:** `suspicious_activity`
- **MCP framework:** FastMCP 2.x
- **Location:** `storewide-loss-prevention/suspicious-activity-detection/mcp-service`

---

## 1. What this service is

The SAD pipeline (YOLO-pose + VLM behavioral analysis) detects suspicious or
unsafe activity in camera zones. This MCP service makes those detections available
to the agent through the **uniform Service Contract**: the agent describes the
service and reads durable history. Runtime actions are intentionally not exposed;
zone and rule setup is configuration.

The service supports both current SAD scenarios:

- **Retail:** existing suspicious-activity events such as loitering and concealment.
- **Kitchen:** food-safety events from the `kitchen-prep` SceneScape zone, pose
  trigger `floor_to_food_area`, and VLM confirmation that an item was picked from
  the floor and placed back in the food area.

The service directly declares normal FastMCP tools. Its SQLite store preserves
the event envelope, idempotency, ordered history, and replay; its delivery
module handles optional registered callbacks. There is no SDK-specific tool
decorator or protocol layer.

---

## 2. FastMCP server and service-owned infrastructure

FastMCP implements the MCP protocol and exposes the service's Python functions
as standard MCP tools. SAD separately owns the production behavior around those
tools: event persistence, MQTT ingestion, time-window queries, replay, and
optional callback delivery.

```mermaid
flowchart TB
    subgraph SAD["SAD MCP service (this repo)"]
        TOOLS["tools.py<br/>MCP tool declarations"]
        QUERIES["queries.py<br/>pure log queries"]
        EVENTS["events.py<br/>schema + ingest seam"]
        MODELS["models.py<br/>Activity type"]
        CONFIG["config.py<br/>env settings"]
        MAIN["main.py<br/>entrypoint"]
    end
    subgraph SAD_RUNTIME["SAD service runtime"]
      MCP["FastMCP<br/>standard tools/list + tools/call"]
      LOG["SAD-owned SQLite store"]
      DEL["SAD callback dispatcher"]
    end
    AGENT["Central QSR Agent<br/>(Hermes — MCP client)"]

    TOOLS -->|@mcp.tool| MCP
    TOOLS --> QUERIES --> LOG
    EVENTS -->|build envelope| LOG
    MAIN -->|mcp.run| MCP
    LOG --> DEL
    MCP <-->|tools/list, tools/call| AGENT
```

  `describe` is an application-level tool for event schemas and service metadata.
  FastMCP and every compatible client still use standard `tools/list` and
  `tools/call`; clients do not have to call `describe` first.

  **Dependencies** (`pyproject.toml`):

```toml
dependencies = ["fastmcp>=2.14,<3", "paho-mqtt>=1.6,<3"]
```

---

## 3. Package structure

| File | Responsibility |
|---|---|
| `src/tools.py` | **The MCP tools** — creates the FastMCP instance and declares tools using FastMCP's `@mcp.tool`. |
| `src/queries.py` | Internal pure query logic over the durable log. Not exposed to the agent. |
| `src/events.py` | Event type, payload schema, and standard event envelope. |
| `src/store.py` | SAD-owned SQLite log, compatible with the previous event table, plus persistent subscriptions. |
| `src/delivery.py` | Condition matching and retrying HTTP callback delivery. |
| `src/models.py` | `Activity` `TypedDict` — structured output type. |
| `src/config.py` | Env-driven settings (store id, transport, host, port). |
| `src/main.py` | Entrypoint: starts MQTT ingestion and FastMCP. |
| `scripts/replay_events.py` | Replays stored event envelopes; `--deliver` simulates callback delivery. |
| `scripts/demo_e2e.py` | End-to-end demo (no MCP client needed). |
| `tests/test_queries.py` | Unit tests for the query helpers. |

---

## 4. The Service Contract (what the agent sees)

This service exposes describe + read tools. Subscribe/callback and runtime act
tools are disabled for the food-safety sensor contract.

| Capability | This service |
|---|---|
| **describe** | Advertises the `report_suspicious_activity` event schema + all read tools with descriptions. |
| **read tools** | `Get_all_activities`, `Get_activity_by_zone`, `Get_activity_by_zone_timestamp`, `Search_retrospective_frames`, `Get_trend_counts`, `Get_event_count`, `Get_all_zones`. |
| **subscribe** | Disabled by default (`SAD_EXPOSE_SUBSCRIBE=false`). |
| **act tools** | None at runtime. Kitchen zone/rule setup is configuration under `configs/usecase/kitchen/`. |

### Read tools

| Tool | Params | Returns |
|---|---|---|
| `Get_all_activities` | — | `list[Activity]` — every recorded activity |
| `Get_activity_by_zone` | `zone` | activities in that zone |
| `Get_activity_by_zone_timestamp` | `zone`, `start_time?`, `end_time?` | zone activities in an ISO-8601 range; local times use `SAD_TIMEZONE` |
| `Search_retrospective_frames` | `query?`, `start_time?`, `end_time?`, `zone?`, `event_name?`, `use_case?` | matching event records and any attached frame references |
| `Get_trend_counts` | `start_time?`, `end_time?`, `event_name?`, `use_case?` | counts grouped by station and shift |
| `Get_event_count` | `start_time?`, `end_time?`, `zone?`, `event_name?`, `use_case?`, `minimum_severity?` | compact count; `minimum_severity=high` includes high and critical |

Time values use ISO-8601 strings, for example `2026-09-29T13:00:00`. Naive
values are interpreted in the configured `SAD_TIMEZONE`; timestamps with an
explicit UTC offset are honored as supplied. The SQLite backend applies the time
range before its result limit, so older history cannot hide newer matches. The
JSONL backend preserves correctness but must scan its append-only file.
| `Get_all_zones` | — | distinct zones with activity |

Descriptions come from the function **docstrings**; parameter docs from
`Annotated[..., Field(description=...)]` — the standard MCP convention.

### Runtime actions

No runtime action tools are exposed. For kitchen food-safety, aiming the zone,
setting the pose trigger, and enabling VLM confirmation are controlled by these
configuration files:

| File | Kitchen setting |
|---|---|
| `configs/usecase/kitchen/scene-config.yaml` | Scene `kitchen food safety`, zone `kitchen-prep: HIGH_VALUE`. |
| `configs/usecase/kitchen/patterns.yaml` | Pose pattern `floor_to_food_area` with VLM confirmation prompt. |
| `configs/usecase/kitchen/rules.yaml` | Emits `FOOD_SAFETY_VIOLATION` when BA/VLM status is suspicious. |

### The `Activity` shape

```python
class Activity(TypedDict):
    ref_id: str        # source event id (MQTT message id) — idempotent
    ts_ms: int         # epoch milliseconds
  event_name: str    # food_safety_violation, loitering, concealment, ...
  use_case: str      # retail | kitchen
    zone: str
    pose: str          # reach-over, item-conceal, slip, ...
    severity: str      # low | medium | high
    camera_id: str
    object_id: str     # SceneScape cross-camera person id
    description: str    # VLM one-line summary
  frame: str          # SeaweedFS frame/object reference
  station: str        # prep, checkout, etc.
  shift: str          # breakfast, lunch, dinner, overnight, unknown
```

Kitchen food-safety events use the generic event type
`report_suspicious_activity` with payload `event_name="food_safety_violation"`,
`zone="kitchen-prep"`, and `description`/`frame` pointing to the VLM-confirmed
evidence.

---

## 5. How events fan out (SAD pipeline → agent)

The SAD MQTT consumer publishes a violation by calling **one function**,
`ingest_alert(...)`. From there `mcp-service-sdk` does *emit-to-log-first, then
fan out*:

```mermaid
sequenceDiagram
    participant P as SAD pipeline (MQTT alerts, ba results)
    participant I as ingest_alert()
    participant SS as ServiceServer.emit()
    participant LOG as Durable log (SQLite)
    participant DEL as Delivery (fan-out)
    participant A as Agent inbox

    P->>I: violation (zone, pose, severity, ... , ref_id=mqtt_msg_id)
    I->>SS: emit("report_suspicious_activity", payload, ref_id)
    SS->>LOG: append (idempotent on ref_id)
    Note over LOG: written FIRST — nothing lost, replayable
    SS->>DEL: dispatch(event)
    DEL->>A: push to enabled sinks (with retries)
    Note over DEL: EventHub (default) / Webhook / Disabled
```

Key properties:

- **Log first.** The event is persisted to the service's own durable log *before*
  any delivery — so nothing is dropped and the whole run is replayable for
  benchmarking/debugging.
- **Source event time.** When the MQTT alert includes a timestamp, the MCP
  envelope preserves it as `ts_ms`, so agent queries and the Automatic Alerts
  UI refer to the same event time. Missing/invalid source timestamps fall back
  to ingestion time.
- **Idempotent.** `ref_id` = the MQTT message id; a redelivered message is stored
  once, never double-counted.
- **Fan-out sinks** (chosen by config in `mcp-service-sdk`):
  - **EventHub** (default) — de-dupes, fans out, retries; the agent inbox listens here.
  - **Webhook** — HTTP callback for partners / other agent frameworks.
  - **Disabled** — clean benchmark runs (log only, no delivery).
- **Bounded retries.** Failed deliveries are retried with backoff; nothing fails silently.

### Wiring the pipeline (the seam)

```python
from tools import ingest_alert   # from src/tools.py

ingest_alert(
    zone="kitchen-prep",
    pose="floor_to_food_area",
    severity="critical",
    camera_id="lp-camera1",
    object_id="p-1001",
    description="Item picked from floor and placed back in the food area",
    ref_id=mqtt_message_id,     # -> idempotent replay
    event_name="food_safety_violation",
    use_case="kitchen",
    frame="s3://behavioral-frames/.../frame.jpg",
    station="prep",
    shift="lunch",
)
```

Drop this call into the SAD MQTT consumer (`behavioral-analysis/src`) wherever a
result/alert is produced.

---

## 6. How the agent reads and acts

```mermaid
sequenceDiagram
    participant A as Agent (Hermes)
    participant MCP as SAD MCP server
    participant PG as Policy Gate
    participant Q as queries.py + log

    A->>MCP: describe (on connect)
    A->>MCP: read Search_retrospective_frames("floor food area", time window)
    MCP->>Q: query durable log
    Q-->>A: list[Activity]
    A->>MCP: read Get_trend_counts(use_case="kitchen")
    MCP->>Q: aggregate station/shift buckets
    Q-->>A: trend counts
```

The agent connects as a standard MCP client. Because the contract is uniform,
onboarding this service needs **no agent code changes** — it discovers everything
via `describe`.

---

## 7. How it runs

The service runs as a **host process** (no container image to maintain), managed
by the Makefile and started/stopped with the main stack.

| Command | Effect |
|---|---|
| `make up USE_CASE=retail` | Brings up the retail SAD stack and the SAD MCP service. |
| `make up USE_CASE=kitchen` | Brings up the kitchen food-safety stack and the SAD MCP service. |
| `make down` | Stops the stack and the MCP service container. |

Details:

- **Transport:** `streamable-http` in deployment (so the agent can reach it over
  the network); `stdio` locally for CLI use.
- **Endpoint:** `http://localhost:9000/mcp` (override port with `MCP_PORT`).
- **Container:** compose service `sad-mcp-service`, image `intel/sad-mcp:${TAG}`.
- **Log volume:** Docker volume `sad-mcp-data` at `/data/sad_log.sqlite`.

> **Networking note:** the agent reaches this at `http://<host>:9000/mcp`. If Hermes
> runs inside the compose network, use the host address (e.g. `host.docker.internal:9000`)
> rather than a compose service DNS name, since this is a host process, not a container.

---

## 8. Configuration (env vars)

| Var | Default | Meaning |
|---|---|---|
| `STORE_ID` | `store_001` | Store identity stamped on every event. |
| `MCP_TRANSPORT` | `stdio` | `stdio` \| `sse` \| `streamable-http`. Makefile sets `streamable-http`. |
| `MCP_HOST` | `0.0.0.0` | Bind host for HTTP transports. |
| `MCP_PORT` | `9000` | Bind port for HTTP transports. |
| `SAD_EXPOSE_SUBSCRIBE` | `false` | Keep callback subscription disabled for the read-only sensor contract. |
| `SAD_TIMEZONE` | `Asia/Kolkata` | IANA timezone used to interpret ISO datetimes without an explicit offset. |
| `SAD_DELIVERY` | `off` | `off` or `webhook`; when `webhook`, events are pushed after durable-log append. |
| `SAD_WEBHOOK_URL` | — | Hub/webhook endpoint used when `SAD_DELIVERY=webhook`. |
| `SEED_DEMO` | `false` | Optional local demo history seeding; production/default startup uses real events only. |

---

## 9. Develop & verify

```bash
cd mcp-service
python3 -m venv .venv
.venv/bin/pip install -e .        # installs mcp-service-sdk from edge-ai-libraries + this package

# run the server
.venv/bin/sad-mcp                 # stdio; set MCP_TRANSPORT=streamable-http for HTTP

# tests + demo (no MCP client needed)
.venv/bin/python -m pytest -q tests/
.venv/bin/python scripts/demo_e2e.py
```

For a no-install local smoke against the checked-out SDK source:

```bash
PYTHONPATH=src:/home/intel/sachin/oep/edge-ai-libraries/libraries/mcp-service-sdk/src \
  python scripts/demo_e2e.py
```

---

## 10. Versioning

- This service consumes `mcp-service-sdk` from the `edge-ai-libraries` repository subdirectory.
- `make mcp-up` runs `pip install -e` every start, so bumping the pin here is picked up automatically.
- When an internal package registry exists, swap the git URL for `mcp-service-sdk[mcp]==<version>`.

---

## 11. Summary

- The SAD service is a **thin instance** of the generic `mcp-service-sdk` contract.
- It writes only its **schema + query helpers + tool declarations**; the base
  provides the log, fan-out, policy gate, telemetry, and MCP scaffolding.
- Events flow **pipeline → `ingest_alert` → emit → durable log (first) → optional fan-out → agent**,
  idempotent on `ref_id` and fully replayable.
- The agent connects over MCP and discovers describe/read tools via `describe` —
  no agent changes needed to onboard this service.
