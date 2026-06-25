# Setup Intel® SceneScape

SceneScape is the spatial intelligence layer that provides person detection,
tracking, and re-identification for the Suspicious Activity Detection
application. It runs entirely from pre-built Docker images — no source
checkout is required.

This page explains how SceneScape is configured and managed within the
application stack.

## Overview

SceneScape is a **shared component** located at
`storewide-loss-prevention/scenescape/`. It is never run directly from that
directory. Instead, the application's `Makefile` invokes SceneScape targets
with `APP_DIR` pointing to the application directory, so that `init.sh` reads
the correct configs and writes to the correct `docker/.env`.

When you run `make up` from the `suspicious-activity-detection/` directory,
SceneScape is automatically initialized and started alongside the LP services.

## Architecture

SceneScape provides the following services:

| Service | Description |
|---------|-------------|
| **Controller** | Manages scene state, zone definitions, and object tracking across cameras. |
| **DLStreamer Pipeline Server** | Runs person detection and re-identification inference on RTSP camera streams using OpenVINO™. Publishes tracking data via MQTT. |
| **MQTT Broker (Mosquitto)** | Secure message bus for inter-service communication. Uses TLS client certificates for authentication (no username/password). |
| **PostgreSQL** | Stores scene definitions, zone metadata, and controller state. |
| **VDMS** | Vector database for person re-identification descriptor storage. |
| **NTP Server** | Provides time synchronization for all containers. |
| **Media Server** | RTSP relay for camera streams. |
| **Web UI** | SceneScape admin interface for scene management and live visualization. |

## Configuration Files

SceneScape reads its configuration from the application's `configs/` directory:

| File | Purpose |
|------|---------|
| `configs/zone_config.json` | Defines the scene, camera(s), video source, and zone-to-type mappings (e.g., `aisle1` → `HIGH_VALUE`). |
| `configs/pipeline-config.json` | DLStreamer pipeline template with `{{CAMERA_NAME}}` placeholder. Rendered at init time with device-specific settings. |
| `configs/.env.example` | Reference environment variables including `SCENESCAPE_REGISTRY`, `SCENESCAPE_VERSION`, and model/device settings. |
| `configs/res/*.env` | Device resource profiles that control inference device, decode chain, and throughput options. |

## Scene Definition

A scene `.zip` archive must be placed in `scenescape/webserver/` before
running `make up`. The filename must match the
`scene_zip` value in `configs/zone_config.json`:

```json
"scene_name": "storewide loss prevention",
"scene_zip": "storewide-loss-prevention.zip",
"camera_name": "lp-camera1",
```

This archive defines the scene in SceneScape and contains:

- A floor plan image
- Zone polygon definitions (regions of interest)
- Camera calibration data

On first startup, the import script (`scenescape/webserver/scene-import.sh`)
waits for the controller to become healthy, then uploads all `.zip` files
from that directory to create the scene, zones, and camera configuration
inside SceneScape.

## Initialization

The `scenescape/scripts/init.sh` script performs the following steps:

1. **Generates TLS certificates** for MQTT broker authentication (stored in
   `scenescape/secrets/certs/`).
2. **Creates Django secrets** for the SceneScape web UI (stored in
   `scenescape/secrets/django/`).
3. **Generates a random admin password** (`SUPASS`) for the web interface.
4. **Reads `configs/.env.example`** and sources AI-model settings
   (`VLM_MODEL_NAME`, `TARGET_DEVICE`, `DETECT_MODEL`, etc.).
5. **Loads the selected device resource config** (e.g.,
   `configs/res/all-gpu-cpu.env`) to set inference devices and decode chain.
6. **Renders `pipeline-config.json`** with camera name, model paths, and
   device settings into
   `scenescape/dlstreamer-pipeline-server/<app>-pipeline-config.json`.
7. **Writes `docker/.env`** with all resolved variables for Docker Compose.

To run initialization manually (without starting services):

```bash
cd suspicious-activity-detection/
make init
```

Or with a specific device profile:

```bash
make init DEVICE=all-gpu.env
```

## Image Versions

SceneScape container images are controlled by two variables in
`configs/.env.example`:

```bash
# Private registry prefix (leave empty for local images)
SCENESCAPE_REGISTRY=

# Image tag for SceneScape containers (controller, manager, etc.)
# Must match the deployed SceneScape release — see configs/.env.example for the current value
SCENESCAPE_VERSION=
```

- **`SCENESCAPE_REGISTRY`**: Set to a registry URL if pulling from a private
  registry (e.g., `myregistry.com/scenescape/`). Leave empty to use locally
  available images.
- **`SCENESCAPE_VERSION`**: Must match the SceneScape release deployed. This
  tag is applied to the controller and manager images.

## Device Profiles

The DLStreamer pipeline server supports multiple inference device
configurations. Select a profile via the `DEVICE` parameter:

```bash
 DEVICE=all-gpu-cpu.env    # GPU detect + CPU re-id
 DEVICE=all-gpu.env        # All GPU
 DEVICE=all-cpu.env        # All CPU
 DEVICE=all-npu-cpu.env    # NPU detect + CPU re-id
 DEVICE=all-npu.env        # All NPU (default)

 example: make up EVICE=all-gpu-cpu.env
```

Profiles are defined in `configs/res/` and control:

- Video decode chain (VA-API for GPU, software for CPU)
- Inference device for detection and re-identification
- Pre-process backend
- Throughput and batching options

## Prerequisites

Before running SceneScape, ensure the following steps are completed:

### 1. Download Sample Video

Download the sample video defined in `configs/zone_config.json`:

```bash
cd suspicious-activity-detection/
make download-sample-data
```

This downloads the video from the `video_url` and saves it as `video_file`
(e.g., `lp-camera1.mp4`) into `scenescape/sample_data/`.

### 2. Download AI Models

Download the OpenVINO™ detection and re-identification models:

```bash
make download-models
```

This downloads person-detection and person-reidentification models into the
`models/` directory.

### 3. Verify Scene Archive

Ensure the scene `.zip` file exists at `scenescape/webserver/` with the
filename matching `scene_zip` in `configs/zone_config.json`.

## Running the Stack

To start the full stack (SceneScape + LP services + model mounts), run from
the `suspicious-activity-detection/` directory:

```bash
cd suspicious-activity-detection/
make up
```

`make up` automatically performs initialization (`init.sh`), downloads
sample data and models, creates Docker volumes, and starts all services
including SceneScape infrastructure and the LP detection pipeline.

The app-level Docker Compose overlay
([docker/docker-compose.yaml](https://github.com/intel-retail/storewide-loss-prevention/blob/main/suspicious-activity-detection/docker/docker-compose.yaml)) mounts
the detection and re-identification models into the DLStreamer container:

```yaml
volumes:
  - ${MODEL_PATH:-./models}/detect_models:/home/pipeline-server/models/detect:ro
  - ${MODEL_PATH:-./models}/reid_models:/home/pipeline-server/models/reid:ro
```

To stop the full stack:

```bash
make down
```

## Re-Identification (ReID)

SceneScape tracks persons across frames and camera views using a
re-identification model. The ReID descriptor store uses VDMS (Vector Data
Management System).

## MQTT Authentication

The SceneScape MQTT broker (Mosquitto) uses **TLS client certificates** for
authentication — not username/password. Certificates are auto-generated by
`init.sh` into `scenescape/secrets/certs/` and mounted read-only into all
containers that connect to the broker.

Key MQTT settings in `configs/.env.example`:

```bash
MQTT_HOST=broker.scenescape.intel.com    # Broker hostname (resolved via Docker DNS)
MQTT_PORT=1883                           # Broker port
```

> **Note:** There is no `MQTT_USERNAME` or `MQTT_PASSWORD` variable. All MQTT
> connections are authenticated via TLS client certificates. If you see
> connection errors, re-run `make init` to regenerate the certificates.

## Accessing the SceneScape UI

Once the stack is running:

| Service | URL | Credentials |
|---------|-----|-------------|
| SceneScape UI | `https://localhost` | `admin` / password printed by `make up` |

The password is auto-generated and stored in `docker/.env` as `SUPASS`. You
can retrieve it at any time:

```bash
grep SUPASS docker/.env | cut -d= -f2-
```

## Environment Variable Reference

The following variables in `configs/.env.example` are specific to SceneScape
configuration. They are read by `init.sh` and written to `docker/.env`.

### SceneScape Image Versions

| Variable | Description | Default |
|----------|-------------|---------|
| `SCENESCAPE_REGISTRY` | Private registry prefix for SceneScape images. Leave empty for local images. Set to a registry URL (e.g., `myregistry.com/scenescape/`) when pulling from a private registry. | *(empty)* |
| `SCENESCAPE_VERSION` | Image tag for SceneScape containers (controller, manager). Must match the SceneScape release deployed alongside this application. | `v2026.1.0-rc1` |
| `DLSTREAMER_VERSION` | Intel DL Streamer image tag for video analytics pipelines. | `2026.1.0-ubuntu24-rc1.1` |

### Host / Proxy

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST_IP` | External IP of the host machine, used for service discovery. | *(empty)* |
| `http_proxy` | HTTP proxy URL. Required if containers need internet access through a proxy. Leave blank if no proxy is needed. | *(empty)* |
| `https_proxy` | HTTPS proxy URL. Same as above for HTTPS traffic. | *(empty)* |
| `no_proxy` | Comma-separated list of hosts/IPs that bypass the proxy. | `localhost,127.0.0.1` |

> **Note:** Some services (e.g., SeaweedFS) also read the uppercase forms
> `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`. The Docker Compose file
> auto-derives these from the lowercase variables.

### MQTT (SceneScape Broker)

The MQTT broker is provided by the SceneScape stack (Mosquitto).
Authentication uses **TLS client certificates** auto-generated by `init.sh`
into `scenescape/secrets/`. No `MQTT_USERNAME` or `MQTT_PASSWORD` is needed.

| Variable | Description | Default |
|----------|-------------|---------|
| `MQTT_HOST` | Broker hostname (resolved via Docker DNS). | `broker.scenescape.intel.com` |
| `MQTT_PORT` | Broker port. | `1883` |

### SceneScape Secrets (auto-populated)

These are generated automatically by `init.sh` — do not edit them manually.

| Variable | Description |
|----------|-------------|
| `SECRETSDIR` | Path to TLS certificates and Django secrets. Default: `./scenescape/secrets` |
| `SUPASS` | Auto-generated admin password for the SceneScape web UI. |
| `DATABASE_PASSWORD` | Auto-generated PostgreSQL password. |
| `CONTROLLER_AUTH` | Auto-generated auth token for SceneScape controller. |
| `UID` / `GID` | User/group ID for container file permissions. Default: `1000` |

### DLStreamer Pipeline

| Variable | Description | Default |
|----------|-------------|---------|
| `DETECT_MODEL` | Person detection model name. | `yolov8s` |
| `DETECT_MODEL_PRECISION` | Detection model precision. | `FP16` |
| `REID_MODEL` | Person re-identification model name. | `person-reidentification-retail-0277` |
| `REID_MODEL_PRECISION` | ReID model precision. | `FP16` |
| `RENDER_GROUP_ID` | Linux render group GID for GPU access inside containers. Find your host value with `getent group render \| cut -d: -f3`. | `992` |

## Troubleshooting

### Containers fail to start with certificate errors

Re-run initialization to regenerate certificates:

```bash
make init
make up
```

### DLStreamer pipeline not detecting persons

- Verify the video file exists in the sample data volume:

  ```bash
  docker volume inspect storewide-lp_vol-sample-data
  ```

- Check that `configs/zone_config.json` has the correct `video_file` name
  matching the camera name.
- Check DLStreamer logs:

  ```bash
  docker logs storewide-lp-lp-cams-1
  ```

### GPU/NPU device not accessible

Ensure the host exposes the required devices:

- GPU: `/dev/dri` must be accessible
- NPU: `/dev/accel/accel0` must be accessible

Check the render group GID matches your host:

```bash
getent group render | cut -d: -f3
```

Update `RENDER_GROUP_ID` in `configs/.env.example` if it differs from the
default (`992`).
