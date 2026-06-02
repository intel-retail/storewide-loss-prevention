# Get Started

This guide walks you through deploying and running the POI Re-identification
system, including the POI backend, React UI, alert service, and the supporting
SceneScape stack.

Before you begin, review the
[System Requirements](./get-started/system-requirements.md) to ensure your
environment meets the recommended hardware and software prerequisites.

## 1. Clone the Repository

> **Note:** Make sure you are in the `person-of-interest`
> directory before running the commands in this guide.

```bash
git clone https://github.com/intel-retail/storewide-loss-prevention.git
cd storewide-loss-prevention/person-of-interest
```

## 2. Initialize Submodules

Clone the required submodules (for example, `performance-tools`):

```bash
make update-submodules
```

## 3. Provide Required Assets

Before starting, ensure these files are in place:

| File | Purpose |
|------|---------|
| `configs/zone_config.json` | Central configuration: scene, cameras, models, SceneScape images, zones |
| `configs/res/*.env` | Device resource configs for DLStreamer pipeline (GPU, CPU, NPU profiles) |
| `configs/pipeline-config.json` | DLStreamer pipeline template (rendered per camera by `init.sh`) |
| `../scenescape/webserver/conference-room.zip` | Scene map + zone definitions imported into SceneScape |
| `../scenescape/sample_data/Camera_01.mp4` | Sample video used by the camera replay |

> **Note:** The video file names must match the `video` entries in
> `configs/zone_config.json` (for example, `Camera_01.mp4` corresponds to
> camera `Camera_01`).

### `zone_config.json` Reference

All application configuration is centralized in `configs/zone_config.json`.
Edit this file first, then run `make init` to generate `docker/.env`, secrets,
and per-camera pipeline configurations automatically.

Minimal example:

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
    "version": "v2026.1.0-rc1",
    "controller_image": "intel/scenescape-controller",
    "manager_image": "intel/scenescape-manager",
    "dlstreamer_version": "2026.1.0-ubuntu24-rc1.1"
  },
  "store": {
    "name": "Retail",
    "id": "store_001"
  }
}
```

The `zone_config.json` file defines:

- `scene_name`, `scene_zip` for SceneScape scene setup
- `cameras[]` as an array of `{name, video}` camera entries
- `models` as the comma-separated OpenVINO model list
- `scenescape{}` for registry, version, and controller/manager image tags
- `store{}` for store name and ID
- `services{}` for ports, log level, and SeaweedFS settings
- `scenescape_api{}` for SceneScape REST API endpoint settings
- `benchmark{}` for stream-density benchmark parameters

## 4. Initialize Environment

Before running `make init`, set `HOST_IP` to the network-reachable IP address of
this machine. MediaMTX uses it to advertise the correct WebRTC ICE candidates so
browsers can connect to the live camera streams.

### Proxy Configuration (if behind a corporate proxy)

If your environment requires a proxy to reach the internet (for example, to pull
Docker images or download models), export the proxy variables **before** running
any `make` commands:

```bash
# Set proxy (adjust to your network)
export HTTP_PROXY=http://your-proxy-server:port
export HTTPS_PROXY=http://your-proxy-server:port
export NO_PROXY=localhost,127.0.0.1,web.scenescape.intel.com,<host-ip>

# Lowercase variants (used by some tools)
export http_proxy=$HTTP_PROXY
export https_proxy=$HTTPS_PROXY
export no_proxy=$NO_PROXY
```

> **Important:** Add `web.scenescape.intel.com` and your machine's IP to
> `NO_PROXY` so that internal Docker-to-Docker traffic does not go through
> the proxy. Failing to do so may cause SceneScape scene imports or API
> calls to fail with connection errors.
>
> These variables are propagated into Docker containers via the
> `docker-compose.yml` environment sections. If the proxy is not set,
> the variables default to blank strings (you will see harmless Docker
> Compose warnings about unset variables).

```bash
# Set HOST_IP (required for WebRTC camera streams)
export HOST_IP=$(hostname -I | awk '{print $1}')

# Or set a specific IP if the auto-detected one is not correct:
# export HOST_IP=192.168.1.100

# Edit the configuration file with your camera and scene details
nano configs/zone_config.json

# Initialize environment with default device profile (CPU)
make init

# Or select a specific device profile:
make init DEVICE=all-gpu-cpu.env
```

### Device Profiles

The `DEVICE` parameter selects which hardware to use for inference. Device
profiles are defined in `configs/res/` and control the GStreamer decode chain,
inference device, pre-process backend, model precision, and throughput options.

| Profile | Decode | Detection | Re-ID | Precision | Command |
|---------|--------|-----------|-------|-----------|---------|
| `all-cpu.env` (default) | CPU (`avdec_h264`) | CPU | CPU | FP32 | `make init DEVICE=all-cpu.env` |
| `all-gpu-cpu.env` | GPU (`vah264dec`) | GPU | CPU | FP16 | `make init DEVICE=all-gpu-cpu.env` |
| `all-gpu.env` | GPU (`vah264dec`) | GPU | GPU | FP16 | `make init DEVICE=all-gpu.env` |
| `all-npu-cpu.env` | GPU (`vah264dec`) | NPU | CPU | FP16-INT8 | `make init DEVICE=all-npu-cpu.env` |
| `all-npu.env` | GPU (`vah264dec`) | NPU | NPU | FP16-INT8 | `make init DEVICE=all-npu.env` |

> **Note:** GPU profiles require an Intel integrated or discrete GPU with
> VA-API support. NPU profiles require an Intel NPU (available on Meteor Lake
> and later platforms). The `all-cpu.env` profile works on any x86 system.

> **Note:** `HOST_IP` must be exported **before** running `make init`. It is
> written into `docker/.env` during initialization. If it is not set, `make init`
> will print an error and exit. `make up` will also exit with an error if
> `HOST_IP` is missing from `docker/.env`.
>
> After changing `DEVICE`, always re-run `make init` to regenerate the pipeline
> configs with the correct device settings.

`make init` generates the following from `configs/zone_config.json`:

- `docker/.env` — all environment variables for Docker Compose
- SceneScape TLS certificates and secrets
- Per-camera DLStreamer pipeline configuration files

## 5. Pull or Build Images

Pre-built container images are available on Docker Hub. The `docker-compose.yml`
references them directly (`intel/poi-backend:2026.1.0-rc1` and
`intel/poi-ui:2026.1.0-rc1`), so `make up` will pull them automatically if they
are not already present locally.

To explicitly pull before starting:

```bash
docker compose --env-file docker/.env -f docker-compose.yml pull poi-backend ui
```

To build from source instead of using pre-built images:

```bash
make build REGISTRY=false
```

When building locally with `REGISTRY=false`, the images are tagged as `poi-backend`
and `poi-ui`, and the compose file uses them via the `POI_BACKEND_IMAGE` and
`POI_UI_IMAGE` environment variables. See [Build from Source](./get-started/build-from-source.md)
for detailed build options.

## 6. Download Models

The OpenVINO face detection and re-identification models are required for both enrollment
and DLStreamer inference:

```bash
make download-models
```

This downloads `face-detection-retail-0004`, `face-reidentification-retail-0095`,
`person-detection-retail-0013`, and `person-reidentification-retail-0277` in FP32,
FP16, and FP16-INT8 precisions. It also exports `clip-reid-market1501` (body re-ID)
in both FP32 and FP16.

## 7. Launch the Application

```bash
make up
```

> **Note:** `make up` auto-detects the host machine's IP address and writes
> `HOST_IP` to `docker/.env` for WebRTC camera feeds. If accessing the UI from
> a different machine, verify `HOST_IP` in `docker/.env` is set to the correct
> network-reachable IP.

The DLStreamer pipeline runs four inference stages using the device and precision
selected by `DEVICE` during `make init`:

- **Person detection**: `person-detection-retail-0013`
- **Body re-ID**: `clip-reid-market1501`
- **Face detection**: `face-detection-retail-0004`
- **Face re-ID**: `face-reidentification-retail-0095`

The pipeline template (`configs/pipeline-config.json`) is rendered at init time
with the selected device profile settings (decode chain, device, precision,
pre-process backend, throughput options).

For a complete first-time setup (init + models + build + start all services), you
can use:

```bash
make demo
```

`make up` performs the following steps automatically:

1. Detects and cleans stale Docker networks (if present).
2. Starts SceneScape services (manager, controller, broker, DLStreamer, etc.).
3. Polls SceneScape web health (up to 150 seconds).
4. Resolves the SceneScape scene UID for the POI backend.
5. Starts POI services (backend, UI, Redis, alert service).

This launches the following containers:

| Container            | Image                        | Port  |
| -------------------- | ---------------------------- | ----- |
| `poi-backend`        | `poi-backend`                | 8000  |
| `poi-ui`             | `poi-ui`                     | 3000  |
| `poi-redis`          | `redis:8.6.2`                | 6379  |
| `poi-alert-service`  | `intel/alert-service:0.0.1`  | 8001  |

> **Note:** Use `make up` for subsequent starts after the initial setup. SceneScape
> is started automatically by the `up` target.

## 8. View Logs

```bash
make logs
```

## 9. Stop Services

```bash
# Stop everything
make down
```

## 10. Access the Interface

Once running:

| Service | URL | Credentials |
|---------|-----|-------------|
| SceneScape UI | https://localhost | `admin` / password printed by `make init` |
| POI UI | http://\<host-ip\>:3000 | — |
| POI Backend API | http://\<host-ip\>:8000/docs | — |
| POI logs | `make logs` | View all service logs |

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

`make init` also injects generated secrets, user IDs, pipeline-config paths, and
`HOST_IP` into `docker/.env`.

> **Note:** `HOST_IP` must be exported before running `make init` (see Step 4).
> It is **not** sourced from `zone_config.json` — it is read directly from your
> shell environment.

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

## Clean Up

```bash
# Stop and remove all containers + volumes
make clean

# Also remove generated secrets and .env
make clean-all
```

## Benchmarking

For detailed benchmarking instructions, parameters, and result processing, see the
[Benchmarking Guide](./benchmarking.md).

## Next Steps

- Learn more about application capabilities in the [How to Use Guide](./how-to-use-application.md)
- Understand the data flow in the [MQTT Pipeline Design](./mqtt-pipeline-design.md)
- If you encounter issues, check the [Troubleshooting Guide](./troubleshooting.md)

<!--hide_directive
:::{toctree}
:hidden:

./get-started/system-requirements
./get-started/build-from-source

:::
hide_directive-->
