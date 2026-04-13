# SceneScape (Shared Component)

This directory contains the configuration, scripts, and deployment files for [Intel SceneScape](https://github.com/intel/SceneScape) — the spatial intelligence layer used by applications under `storewide-loss-prevention/`.

SceneScape is a **shared component** — it is not run directly. Each application (e.g., `suspicious-activity-detection/`) invokes SceneScape through its own Makefile, passing its app-specific configs.

SceneScape runs entirely from pre-built Docker images; no source checkout is required.

## Usage

SceneScape must be run from an **application directory**, not from this directory directly:

```bash
# From an application directory, e.g.:
cd suspicious-activity-detection/
make run-scenescape      # init + start SceneScape with this app's config
make down-scenescape     # stop SceneScape
make demo                # start full stack (SceneScape + app services)
```

Each application passes its own `APP_DIR` so that `init.sh` reads the correct `configs/` and writes to the correct `docker/.env`.

## How It Works

1. **`scripts/init.sh <app-dir>`** reads `configs/zone_config.json` from the given app directory, generates TLS certificates and Django secrets, produces a DLStreamer pipeline config from the app's `configs/pipeline-config.json`, and writes `<app-dir>/docker/.env`.

2. The generated pipeline config is named `<app-name>-pipeline-config.json` (e.g., `suspicious-activity-detection-pipeline-config.json`) to avoid conflicts when multiple apps share SceneScape.

3. **`scripts/download_models.sh`** downloads OpenVINO models (person-detection, person-reidentification) from the Open Model Zoo into a Docker volume. Models that already exist are skipped.

4. **`docker-compose.yaml`** starts the full SceneScape stack (controller, DLStreamer pipeline server, MQTT broker, database, media server) using pre-built images.

5. The DLStreamer pipeline server runs person detection and re-identification inference on RTSP camera streams, publishing tracking data via MQTT for consuming applications.

## Adding a New Application

To create a new application that uses SceneScape:

1. Create a new directory under `storewide-loss-prevention/` (e.g., `my-new-app/`)
2. Add a `configs/` folder with:
   - `zone_config.json` — scene, camera , zones(resgions) all required configuration   
   - `pipeline-config.json` — DLStreamer pipeline template (use `{{CAMERA_NAME}}` placeholder)
3. Add a `docker/` folder for your app's `docker-compose.yaml`
4. Place your video file(s) in `scenescape/sample_data/` and scene zip(s) in `scenescape/webserver/`
5. Add a `Makefile` that calls SceneScape targets with `APP_DIR`:
   ```makefile
   SCENESCAPE_DIR := ../scenescape
   
   run-scenescape:
   @$(MAKE) -C $(SCENESCAPE_DIR) APP_DIR=$(CURDIR) run
   
   down-scenescape:
   @$(MAKE) -C $(SCENESCAPE_DIR) APP_DIR=$(CURDIR) down
   ```

## Directory Structure

```
scenescape/
├── Makefile                        # SceneScape make targets (requires APP_DIR)
├── docker-compose.yaml             # Docker Compose for SceneScape services
├── scripts/
│   ├── init.sh                     # Generate secrets, pipeline config, .env
│   ├── install.sh                  # First-time install helper
│   └── download_models.sh          # Download OpenVINO models into Docker volume
├── controller/
│   ├── reid-config.json            # Re-identification tracker config
│   └── tracker-config.json         # Object tracker config
├── dlstreamer-pipeline-server/
│   ├── <app>-pipeline-config.json  # Auto-generated (gitignored)
│   ├── model-proc-files/           # Model processing JSON files
│   └── user_scripts/               # GVA Python callbacks (sscape_adapter)
├── mosquitto/
│   └── mosquitto-secure.conf       # MQTT broker configuration
├── sample_data/                    # Sample video files (.mp4, .ts)
├── secrets/                        # Auto-generated TLS certs, Django secrets, auth tokens
│   ├── generate_secrets.sh         # Secret generation script
│   ├── certs/                      # TLS certificates
│   └── django/                     # Django secrets.py
└── webserver/
    ├── scene-import.sh             # Scene auto-import (supports multiple .zip files)
    └── *.zip                       # Scene definition archives
```

## Make Targets

All targets require `APP_DIR` to be set (done automatically when called from an app's Makefile).

| Target              | Description                                              |
| ------------------- | -------------------------------------------------------- |
| `init`              | Generate secrets, pipeline config, and `docker/.env`     |
| `init-sample-data`  | Create and populate the sample data Docker volume        |
| `download-models`   | Download OpenVINO models (skips existing ones)           |
| `init-volumes`      | Create the media Docker volume                           |
| `run`               | Full init + start SceneScape                             |
| `down`              | Stop all SceneScape services                             |
