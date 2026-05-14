# Get Started

## Overview

POI Re-identification is a real-time retail loss-prevention system that detects enrolled
Persons of Interest across multiple cameras using face re-identification and FAISS vector
search. This guide walks you through deploying and configuring the application.

## Prerequisites

### System Requirements

- System must meet [minimum requirements](./get-started/system-requirements.md).
- Intel® SceneScape must be deployed and running with DLStreamer pipelines configured.

The POI system operates alongside Intel® SceneScape in a distributed architecture:

| Service          | Port  | Purpose                                           |
| ---------------- | ----- | ------------------------------------------------- |
| POI Backend      | 8000  | REST API, MQTT consumer, FAISS matching            |
| POI UI           | 3000  | React operator interface                           |
| Redis            | 6379  | Metadata, events, cache                            |
| Alert Service    | 8001  | Alert fan-out (WebSocket, MQTT, log)               |
| SceneScape       | 8443  | Spatial scene management + DLStreamer pipelines     |

> **Note:** The MCP Server (port 9000) is an optional service for AI-powered analysis via
> Claude Desktop. It is not started by `make up` — start it separately with
> `docker compose up -d mcp-server`.

### Software Dependencies

- **Docker**: [Installation Guide](https://docs.docker.com/get-docker/)
  - Must be configured to run without sudo ([Post-install guide](https://docs.docker.com/engine/install/linux-postinstall/))
- **Git**: [Installation Guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- **Make**: Required for build and deployment commands

### Required Services

Before setting up the POI system, ensure these services are running:

#### 1. Intel® SceneScape

SceneScape provides the upstream inference pipeline (DLStreamer) and spatial scene management:

- Person detection via `person-detection-retail-0013`
- Face detection via `face-detection-retail-0004`
- Face re-identification via `face-reidentification-retail-0095` (256-d embeddings)
- Region tracking via regulated scene events

Refer to the [SceneScape documentation](../../../scenescape/README.md) for setup instructions.

#### 2. MQTT Broker

SceneScape's MQTT broker must be accessible from the POI backend. The default configuration
connects to the broker bundled with SceneScape.

## Quick Start

### Step 1: Clone the Repository

```bash
git clone https://github.com/sainijit/storewide-loss-prevention-retail.git
cd storewide-loss-prevention-retail/person-of-interest
```

### Step 2: Update Submodules

Pull the latest shared dependencies (SceneScape, performance-tools, etc.):

```bash
make update-submodules
```

### Step 3: Place Video Files

Place your camera video files in the `scenescape/sample_data/` directory. The filenames
must match the `video` entries in `configs/zone_config.json`:

```bash
../scenescape/sample_data/Camera_01.mp4
../scenescape/sample_data/Camera_02.mp4
```

> **Note:** Without video files, the cameras will appear offline in both the SceneScape
> and POI UIs. Any MP4 video containing people can be used for testing.

### Step 4: Initialize Environment

All application configuration is centralized in `configs/zone_config.json`. Edit that file
first, then run `make init` to generate `docker/.env`, secrets, and per-camera pipeline
configuration automatically.

```bash
# Edit the configuration file with your camera and scene details
nano configs/zone_config.json

# Initialize environment (generates docker/.env, secrets, pipeline configs)
make init
```

Minimal `configs/zone_config.json` example:

```json
{
  "scene_name": "conference room",
  "scene_zip": "conference-room.zip",
  "cameras": [
    { "name": "Camera_01", "video": "Camera_01.mp4" }
  ],
  "models": "person-detection-retail-0013,face-detection-retail-0004,face-reidentification-retail-0095",
  "scenescape": {
    "registry": "",
    "version": "latest",
    "controller_image": "scenescape-controller",
    "manager_image": "scenescape-manager"
  },
  "store": {
    "name": "Retail",
    "id": "store_001"
  },
  "services": {
    "lp_service_port": 8082,
    "log_level": "INFO"
  },
  "benchmark": {
    "target_latency_ms": 2000,
    "latency_metric": "avg"
  }
}
```

The `zone_config.json` file defines:

- `scene_name`, `scene_zip` for SceneScape scene setup
- `cameras[]` as an array of `{name, video}` camera entries
- `models` as the comma-separated OpenVINO model list
- `scenescape{}` for registry, version, and controller/manager images
- `store{}` for store name and ID
- `services{}` for ports, log level, and SeaweedFS settings
- `benchmark{}` for stream-density benchmark parameters

### Step 5: Build the Application

```bash
# Build POI backend and UI images locally
make build REGISTRY=false
```

### Step 6: Download Models

The OpenVINO face detection and re-identification models are required for both enrollment
and DLStreamer inference:

```bash
make download-models
```

This downloads `face-detection-retail-0004`, `face-reidentification-retail-0095`,
`person-detection-retail-0013`, and `person-reidentification-retail-0277` into the
`models/` directory.

### Step 7: Launch the Application

The POI system connects to SceneScape via a shared Docker network. Create it if it
doesn't exist:

```bash
docker network create storewide-lp 2>/dev/null || true
```

Then start the services:

```bash
# Start SceneScape + POI stack
make up
```

For a complete first-time setup (init + models + build + start all services including
the Storewide LP pipeline), you can use:

```bash
make demo
```

> **Note:** `make demo` starts the full Storewide LP stack (swlp-service,
> behavioral-analysis, gradio-ui) in addition to SceneScape. For the POI system only,
> use `make build REGISTRY=false && make up`.

This launches the following containers:

| Container            | Image                        | Port  |
| -------------------- | ---------------------------- | ----- |
| `poi-backend`        | `person-of-interest-poi-backend` | 8000  |
| `poi-ui`             | `person-of-interest-ui`      | 3000  |
| `poi-redis`          | `redis:8.6.2`                | 6379  |
| `poi-alert-service`  | `intel/alert-service:0.0.1`  | 8001  |

> **Note:** Use `make up` for subsequent starts after the initial build. SceneScape must
> be running (either started by `make up` automatically, or via `make run-scenescape`
> separately). To start the MCP server, run `docker compose up -d mcp-server` separately.

### Step 8: Access the Interface

Open your browser and navigate to:

```text
http://<host-ip>:3000
```

The POI Backend API is available at:

```text
http://<host-ip>:8000/docs
```

### Step 9: Stop Services

```bash
# Stop all services
make down
```

## Advanced Configuration

### Environment Variables

All values in `docker/.env` are auto-generated from `configs/zone_config.json` by
`make init` (`scenescape/scripts/init.sh`). Do **not** edit `docker/.env` directly; update
`configs/zone_config.json` and re-run `make init` instead.

| `zone_config.json` Key | Generated `docker/.env` Values | Description |
| ---------------------- | ------------------------------ | ----------- |
| `scene_name`, `scene_zip` | `SCENE_NAME`, `SCENE_ZIP` | Scene name and scene archive used by SceneScape |
| `cameras[]` | `CAMERA_NAME`, `VIDEO_FILE`, `CAMERA_NAME_2`, `VIDEO_FILE_2` | Camera names and input videos for generated pipelines |
| `models`, `model_precision` | `MODELS`, `MODEL_PRECISION` | OpenVINO model list and precision |
| `scenescape{}` | `SCENESCAPE_REGISTRY`, `SCENESCAPE_VERSION`, image settings | SceneScape image source and version selection |
| `store{}` | `STORE_NAME`, `STORE_ID` | Store metadata used by the stack |
| `services{}` | `LP_SERVICE_PORT`, `LOG_LEVEL`, `SEAWEEDFS_*` | Service ports, logging, and SeaweedFS settings |
| `benchmark{}` | `BENCHMARK_*`, `RESULTS_PATH` | Stream-density benchmark configuration |

`make init` also injects generated secrets, user IDs, and pipeline-config paths into
`docker/.env`.

> **Note:** Benchmark-related environment variables are configured in the `benchmark`
> section of `configs/zone_config.json` and written into `docker/.env` during initialization.

### Running Tests and Generating Coverage Report

1. **Run Tests**

   ```bash
   make test
   ```

2. **Run Tests with Coverage**

   ```bash
   make coverage
   ```

3. **Generate HTML Coverage Report**

   ```bash
   make coverage-html
   ```

   Open `backend/htmlcov/index.html` in your browser to view the report.

### Custom Build Configuration

If using a container registry, set the registry URL before building:

```bash
export REGISTRY=docker.io/username
make build
```

See [Build from Source](./get-started/build-from-source.md) for detailed build options.

## SceneScape Configuration

- Use `make export-scene` to export scene configuration from a running SceneScape instance.
- Store scene zip files referenced by `scene_zip` in the repository's `scenescape/webserver/`
  directory.
- `make init` generates DLStreamer pipeline configuration files per camera from
  `configs/pipeline-config.json` and writes them into
  `scenescape/dlstreamer-pipeline-server/`.

## Next Steps

1. **Explore Features**: Learn about application capabilities in the [How to Use Guide](./how-to-use-application.md)
2. **Troubleshooting**: If you encounter issues, check the [Troubleshooting Guide](./troubleshooting.md)
3. **MQTT Pipeline**: Understand the data flow in the [MQTT Pipeline Design](./mqtt-pipeline-design.md)

<!--hide_directive
:::{toctree}
:hidden:

./get-started/system-requirements
./get-started/build-from-source

:::
hide_directive-->
