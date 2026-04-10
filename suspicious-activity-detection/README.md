# Store-wide Loss Prevention: Suspicious Activity Detection

> **GitHub:** [intel-sandbox/storewide-loss-prevention](https://github.com/intel-sandbox/storewide-loss-prevention)

MQTT-driven loss prevention service for Intel SceneScape retail deployments. The service monitors person behavior across store zones using real-time tracking from SceneScape, manages session state and detection rules, and stores cropped person frames in SeaweedFS. Behavioral analysis (pose detection, VLM confirmation) and advanced rule evaluation are handled by separate external services called conditionally.

## Prerequisites

- **Intel SceneScape** running with cameras configured and regions/zones defined
- **Docker** and **Docker Compose** installed
- SceneScape Docker network (`scenescape_scenescape`) available

## Core Responsibilities

The Store-wide Loss Prevention service owns four responsibilities:

1. **MQTT Subscription & Event Routing** — subscribes to SceneScape topics, parses messages, routes to handlers
2. **Session State Management** — creates/updates/expires PersonSession per tracked person
3. **Business Logic (Detection Rules)** — evaluates region events, fires alerts
4. **Frame Manager** — stores cropped person frames in SeaweedFS for persons in HIGH_VALUE zones

## External Services (called conditionally)

| Service | Purpose | When Called |
|---------|---------|-------------|
| **BehavioralAnalysis Service** | Pose analysis + VLM concealment confirmation | Person enters HIGH_VALUE zone with frames available |
| **Rule Service** | Advanced/configurable rule evaluation | On every region event (if enabled) |

These are separate containers — not part of this service.

## Suspicious Activities Detected

| # | Activity | Trigger | Alert Level |
|---|----------|---------|-------------|
| 1 | Merchandise Concealment | BehavioralAnalysis Service confirms (confidence ≥ 0.80) | WARNING |
| 2 | Checkout Bypass | Visited HIGH_VALUE zone, exited without passing CHECKOUT | WARNING / CRITICAL* |
| 3 | Loitering | > 120 s in a HIGH_VALUE zone (once per zone per session) | WARNING |
| 4 | Repeated Visits | > 3 visits to same HIGH_VALUE zone | WARNING |
| 5 | Restricted Zone Violation | Entered RESTRICTED zone (immediate) | CRITICAL |

\* Escalates to CRITICAL if concealment was also suspected for that person.

## Architecture

```
Cameras → SceneScape (DLStreamer + Controller) → MQTT Bus
    ↓                                               ↓
    ↓  scenescape/data/scene/+/+                     ↓  scenescape/event/region/+/+/+
    ↓  (position, cameras, bbox)                     ↓  (enter/exit with dwell time)
    ↓                                               ↓
Store-wide Loss Prevention
  ├── MQTT Service            — subscribes to scene-data, region-event, and image topics
  ├── Session Manager         — PersonSession lifecycle, consumes SceneScape region events
  ├── Rule Engine             — detection rules → alerts, calls external services conditionally
  ├── Frame Manager           — SeaweedFS (cropped person frames, rolling buffer, evidence)
  ├── Alert Publisher         — MQTT + REST API + structured log
  └── External Service Clients
       ├── BehavioralAnalysis — HTTP client for pose + VLM service
       └── Rule Service       — HTTP client for advanced rule evaluation
```

### MQTT Topics Consumed

| Topic Pattern | Purpose | Data Provided |
|---------------|---------|---------------|
| `scenescape/event/region/{scene_id}/{region_id}/count` | Region entry/exit events | `entered[]`, `exited[]` with `dwell`, `object_id`, `counts` |
| `scenescape/data/camera/{camera_id}` | Per-camera detections | `objects[]` with `id`, `bbox`, `confidence` |
| `scenescape/image/camera/{camera_id}` | Camera frames | Base64 encoded image, `timestamp` |

### Session State (PersonSession)

| Field | Type | Purpose |
|-------|------|---------|
| `object_id` | string | SceneScape's persistent person identifier |
| `first_seen` | datetime | Session creation timestamp |
| `last_seen` | datetime | Last activity timestamp for expiry |
| `visited_checkout` | boolean | Whether the person entered any checkout zone |
| `visited_high_value` | boolean | Whether the person entered any high-value zone |
| `zone_visit_counts` | dict | Count of entries per region_id |
| `current_zones` | dict | Currently occupied zones with entry timestamps |
| `loiter_alerted` | dict | Tracks if a loiter alert has already been triggered per zone |
| `concealment_suspected` | boolean | Set to true if behavioral analysis confirms suspicious behavior |

### SeaweedFS Storage Design

Only cropped person frames for individuals in HIGH_VALUE zones are stored, at 2 fps per person.

```
bucket: loss-prevention-frames
├── {object_id}/
│   ├── {timestamp_1}.jpg    # Cropped person frame
│   ├── {timestamp_2}.jpg
│   └── ...                  # Rolling buffer of last 20 frames (~10 seconds)
└── alerts/
    └── {alert_id}/
        └── evidence/        # Frames sent to analysis, retained for audit
```

- **Rolling buffer**: 20 frames per person (~400 KB)
- **Active retention**: while in HIGH_VALUE zone + 60s after exit
- **Evidence retention**: 24 hours (configurable)

## Quick Start

### Using setup.sh (recommended)

```bash
cd /path/to/storewide-loss-prevention

# Ser env vars
source setup.sh --setenv

# First time — builds images, copies TLS cert, starts containers
source setup.sh --setup

# Subsequent runs — start without rebuilding
source setup.sh --run

# Restart
source setup.sh --restart

# Stop
source setup.sh --stop

# Clean up (remove containers + volumes)
source setup.sh --clean
```

### Manual setup

```bash
cd /path/to/storewide-loss-prevention

# 1. Copy SceneScape TLS certificate
mkdir -p secrets/certs
cp ../../scenescape/secrets/certs/scenescape-ca.pem secrets/certs/

# 2. (Optional) Set SceneScape API credentials for zone auto-discovery
export SCENESCAPE_API_USER=scenectrl
export SCENESCAPE_API_PASSWORD=<password>

# 3. Start services
cd docker
docker compose up -d --build
```

### Environment Variables

Override defaults by exporting before running `setup.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MQTT_HOST` | `broker.scenescape.intel.com` | SceneScape MQTT broker |
| `LP_SERVICE_PORT` | `8082` | Service REST API port |
| `SEAWEEDFS_S3_PORT` | `8333` | SeaweedFS S3-compatible port |
| `BEHAVIORAL_ANALYSIS_URL` | `http://behavioral-analysis-service:8090` | BehavioralAnalysis service URL |
| `RULE_SERVICE_URL` | `http://rule-service:8091` | Rule service URL |
| `SCENESCAPE_API_USER` | *(empty)* | SceneScape API username (enables zone auto-discovery) |
| `SCENESCAPE_API_PASSWORD` | *(empty)* | SceneScape API password |

## Services Started

| Container | Port | Description |
|-----------|------|-------------|
| storewide-loss-prevention | 8082 | FastAPI service (alerts, sessions, health) |
| seaweedfs | 8333 / 9333 | S3-compatible object storage (person frames, evidence) |

All containers join the `scenescape_scenescape` Docker network.

## API Endpoints

### Core

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/api/v1/lp/alerts` | Recent alerts (query: `?alert_type=CONCEALMENT`, `?object_id=42`) |
| GET | `/api/v1/lp/alerts/count` | Total alert count |
| GET | `/api/v1/lp/sessions` | Active person sessions |
| GET | `/api/v1/lp/sessions/count` | Active session count |
| GET | `/api/v1/lp/status` | Service status + statistics (includes zone counts) |

### Zone Management (runtime, no restart needed)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/lp/zones` | List all configured zones |
| PUT | `/api/v1/lp/zones/{region_id}` | Add or update a zone mapping |
| DELETE | `/api/v1/lp/zones/{region_id}` | Remove a zone mapping |
| POST | `/api/v1/lp/zones/discover` | Re-scan SceneScape for new regions |
| GET | `/api/v1/lp/zones/names` | View zone name → type mapping from config |

## Zone Configuration

Zones map SceneScape regions to LP zone types (`HIGH_VALUE`, `CHECKOUT`, `EXIT`, `RESTRICTED`). There are three ways to configure them — **no manual UUID copying needed**.

### Option A: Auto-discovery at startup (recommended)

Set SceneScape API credentials and the service auto-fetches all regions and maps them by name:

```bash
export SCENESCAPE_API_USER=scenectrl
export SCENESCAPE_API_PASSWORD=<password>
source setup.sh --run
```

Name matching rules in `zone_config.json` control the mapping — region names in SceneScape must match exactly:
```json
{
  "scene_name": "Retail",
  "scenescape_api": {
    "base_url": "https://web.scenescape.intel.com",
    "auth_path": "/api/v1/auth",
    "scenes_path": "/api/v1/scenes",
    "regions_path": "/api/v1/regions",
    "verify_ssl": false
  },
  "zones": {
    "jewelry_zone": "HIGH_VALUE",
    "entrance_exit_zone": "EXIT",
    "restricted_office_zone": "RESTRICTED",
    "checkout_zone": "CHECKOUT"
  }
}
```

### Option B: Runtime API (no restart)

```bash
# Add a zone
curl -X PUT http://localhost:8082/api/v1/lp/zones/<region-uuid> \
  -H "Content-Type: application/json" \
  -d '{"name": "Electronics Aisle", "type": "HIGH_VALUE"}'

# List all zones
curl http://localhost:8082/api/v1/lp/zones

# Trigger re-discovery from SceneScape
curl -X POST http://localhost:8082/api/v1/lp/zones/discover

# Remove a zone
curl -X DELETE http://localhost:8082/api/v1/lp/zones/<region-uuid>
```

> **Note:** Runtime API changes (Option B) are in-memory and reset on container restart. Auto-discovery (Option A) re-runs on every startup.

## Project Structure

```
storewide-loss-prevention/
├── setup.sh                    # Setup, run, stop, clean commands
├── README.md
├── docker/
│   ├── docker-compose.yaml     # 2 services: LP service + SeaweedFS
│   ├── Dockerfile
│   └── .env.example
├── secrets/
│   └── certs/
│       └── scenescape-ca.pem   # Copied from SceneScape (auto by setup.sh)
└── src/
    ├── main.py                 # FastAPI app, service wiring
    ├── docker-entrypoint.sh
    ├── pyproject.toml
    ├── api/
    │   ├── __init__.py
    │   └── routes.py           # REST endpoints
    ├── config/
    │   ├── app_config.json     # MQTT, SeaweedFS, external services, rules thresholds
    │   └── zone_config.json    # Region name → zone type mapping
    ├── models/
    │   ├── __init__.py
    │   ├── session.py          # PersonSession, RegionVisit
    │   ├── events.py           # RegionEvent, EventType, ZoneType
    │   └── alerts.py           # Alert, AlertType, AlertLevel
    ├── services/
    │   ├── __init__.py
    │   ├── config.py                      # ConfigService — dynamic zone map + JSON configs
    │   ├── scenescape_client.py           # SceneScape REST API — auth, fetch regions, auto-map
    │   ├── mqtt_service.py                # MQTT subscribe + publish
    │   ├── session_manager.py             # Consumes SceneScape region events → ENTER/EXIT/PERSON_LOST
    │   ├── rule_engine.py                 # Detection rules + external service integration
    │   ├── frame_manager.py               # SeaweedFS CRUD (rolling buffer, evidence)
    │   ├── alert_publisher.py             # MQTT + in-memory ring buffer
    │   ├── behavioral_analysis_client.py  # HTTP client for external BehavioralAnalysis service
    │   └── rule_service_client.py         # HTTP client for external Rule service
    └── tests/
        ├── __init__.py
        ├── test_session_manager.py
        └── test_rule_engine.py
```
