# Store-wide Loss Prevention: Suspicious Activity Detection


MQTT-driven loss prevention service for Intel SceneScape retail deployments. The service monitors person behavior across store zones using real-time tracking from SceneScape, manages session state and detection rules, and stores cropped person frames in SeaweedFS. Behavioral analysis (pose detection, VLM confirmation) and advanced rule evaluation are handled by separate external services called conditionally.

## Prerequisites

- **Docker** (24.0+) and **Docker Compose** (v2.20+)
- ~10 GB disk space for Docker images, OpenVINO models, and sample video
- Scene zip file (`scenescape/webserver/storewide-loss-prevention.zip`)
- Sample video (`scenescape/sample_data/lp-camera1.mp4`)

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
| **Alert Service** | Advanced/configurable rule evaluation | On every region event (if enabled) |

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
       └── Alert Service      — HTTP client for advanced rule evaluation
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

### Prerequisites

| Requirement | Minimum Version |
|-------------|-----------------|
| Docker | 24.0+ |
| Docker Compose | v2.20+ |
| Disk space | ~10 GB (images + models + video) |

### Required Files

Before starting, ensure these files are in place:

| File | Purpose |
|------|---------|
| `scenescape/webserver/storewide-loss-prevention.zip` | Scene map + zone definitions (imported into SceneScape) |
| `scenescape/sample_data/lp-camera1.mp4` | Sample video for the camera replay |
| `configs/zone_config.json` | Zone name → type mapping (e.g., `aisle1` → `HIGH_VALUE`) |

### Start Everything (SceneScape + LP)

```bash
cd suspicious-activity-detection

# Single command — generates secrets, downloads models, builds, and starts all services
make demo
```

`make demo` performs the following steps automatically:
1. Generates TLS certificates, SceneScape secrets, and `docker/.env`
2. Copies sample video into the Docker volume
3. Downloads OpenVINO models (person-detection + re-identification)
4. Initializes Docker volumes with correct permissions
5. Builds the LP and Gradio UI container images
6. Starts all 11 containers (SceneScape + LP)
7. Imports the scene map into SceneScape
8. Tails LP logs to `application.log`

Once running:

| Service | URL | Credentials |
|---------|-----|-------------|
| SceneScape UI | https://localhost | `admin` / password printed by `make demo` |
| Gradio Dashboard | http://localhost:7860 | — |
| LP REST API | http://localhost:8082 | — |
| LP logs | `application.log` | `tail -f application.log` |

### Start SceneScape Only (without LP)

```bash
make run-scenescape
```

### Start LP Only (SceneScape already running)

```bash
make demo-lp
```

### Stop Services

```bash
# Stop everything (SceneScape + LP)
make down

# Stop SceneScape only
make down-scenescape

# Stop LP only
make down-lp
```

### Monitoring

```bash
# Follow live logs for all services
make logs

# Show container status
make status

# Check LP health
curl http://localhost:8082/health

# Check LP service status + zone counts
curl http://localhost:8082/api/v1/lp/status
```

### Clean Up

```bash
# Stop and remove all containers + volumes
make clean

# Also remove generated secrets and .env
make clean-all
```

### Configuration

All environment variables are in `configs/.env.example`. The `make demo` command auto-generates `docker/.env` — you do not need to create it manually.

Key variables you may want to customize in `configs/.env.example` before running:

| Variable | Default | Description |
|----------|---------|-------------|
| `SCENE_ZIP` | `storewide-loss-prevention.zip` | Scene zip filename (in `scenescape/webserver/`) |
| `VIDEO_FILE` | `lp-camera1.mp4` | Video filename (in `scenescape/sample_data/`) |
| `MODELS` | `person-detection-retail-0013,...` | OpenVINO models (comma-separated) |
| `MODEL_PRECISION` | `FP32` | Model precision (`FP32`, `FP16`) |
| `SCENESCAPE_VERSION` | `v2026.0.0` | SceneScape Docker image tag |
| `LP_SERVICE_PORT` | `8082` | LP REST API port |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`) |

Zone mapping is configured in `configs/zone_config.json`:

```json
{
  "scene_name": "storewide loss prevention",
  "camera_name": "lp-camera1",
  "zones": {
    "aisle1": "HIGH_VALUE",
    "aisle2": "CHECKOUT"
  }
}
```

Zone names must match the region names defined in your SceneScape scene.

## Services Started

`make demo` starts the following containers:

| Container | Port | Description |
|-----------|------|-------------|
| `storewide-lp-web-1` | 443 | SceneScape web UI + REST API |
| `storewide-lp-broker-1` | 1883 | MQTT broker (Mosquitto) |
| `storewide-lp-pgserver-1` | 5432 | PostgreSQL database |
| `storewide-lp-scene-1` | — | SceneScape controller (tracking + analytics) |
| `storewide-lp-lp-cams-1` | — | DLStreamer pipeline server (inference) |
| `storewide-lp-lp-video-1` | — | Video replay (FFmpeg → RTSP) |
| `storewide-lp-mediaserver-1` | 8554 | RTSP media server |
| `storewide-lp-ntpserv-1` | — | NTP time sync |
| `storewide-lp-storewide-loss-prevention-1` | 8082 | LP FastAPI service |
| `storewide-lp-gradio-ui-1` | 7860 | Gradio monitoring dashboard |
| `storewide-lp-seaweedfs-1` | 8333 | SeaweedFS object storage (person frames) |

All containers join the `storewide-lp_storewide-lp` Docker network.

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

The LP service auto-discovers zones from SceneScape on startup. Zone names in `configs/zone_config.json` are matched against SceneScape region names:

```json
{
  "scene_name": "storewide loss prevention",
  "zones": {
    "aisle1": "HIGH_VALUE",
    "aisle2": "CHECKOUT"
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
├── README.md
├── Makefile
├── configs/
│   ├── .env.example            # Environment variable template
│   ├── app_config.json
│   ├── s3-config.json
│   └── zone_config.json        # Region name → zone type mapping
├── docker/
│   ├── docker-compose.yaml     # LP services: LP + Gradio UI + SeaweedFS
│   ├── Dockerfile
│   └── .env                    # Auto-generated by init.sh (gitignored)
├── scenescape/
│   ├── docker-compose-scenescape.yaml
│   ├── scripts/
│   │   ├── setup.sh            # Setup, run, stop, clean commands
│   │   ├── init.sh             # Generate secrets, DLStreamer config, .env
│   │   ├── install.sh          # First-time install helper
│   │   └── download_models.sh  # Download OpenVINO models
│   └── ...                     # SceneScape config, secrets, sample_data
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
    │   └── rule_service_client.py         # HTTP client for external Alert service
    └── tests/
        ├── __init__.py
        ├── test_session_manager.py
        └── test_rule_engine.py
```
