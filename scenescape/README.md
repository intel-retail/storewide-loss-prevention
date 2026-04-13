# SceneScape

This directory contains the configuration, scripts, and deployment files for [Intel SceneScape](https://github.com/intel/SceneScape) — the spatial intelligence layer used by the Storewide Loss Prevention application.

SceneScape runs entirely from pre-built Docker images; no source checkout is required.

## Quick Start

```bash
# From this directory
make run        # init + start SceneScape
make down       # stop SceneScape
```

Or from the parent project directory:

```bash
make run-scenescape      # start SceneScape only
make down-scenescape     # stop  SceneScape only
make demo                # start full stack (SceneScape + LP services)
```

## Make Targets

| Target              | Description                                              |
| ------------------- | -------------------------------------------------------- |
| `make run`          | Full init (secrets, volumes, models) and start SceneScape |
| `make down`         | Stop all SceneScape services                             |
| `make init`         | Generate secrets, DLStreamer config, and `docker/.env`    |
| `make init-sample-data` | Create and populate the sample data Docker volume    |
| `make download-models`  | Download OpenVINO models (skips existing ones)       |
| `make init-volumes` | Create the media Docker volume                           |
| `make logs`         | Tail SceneScape service logs                             |
| `make status`       | Show running SceneScape containers                       |

## Directory Structure

```
scenescape/
├── Makefile                        # SceneScape make targets
├── docker-compose-scenescape.yaml  # Docker Compose for SceneScape services
├── scripts/
│   ├── init.sh                     # Generate secrets, DLStreamer config, .env
│   ├── install.sh                  # First-time install helper
│   ├── download_models.sh          # Download OpenVINO models into Docker volume│   
├── controller/
│   ├── reid-config.json            # Re-identification tracker config
│   └── tracker-config.json         # Object tracker config
├── dlstreamer-pipeline-server/
│   ├── config.json                 # Auto-generated DLStreamer pipeline config
│   ├── lp-config.json              # DLStreamer pipeline template
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
    ├── scene-import.sh             # Scene auto-import script
    └── *.zip                       # Scene definition archives
```

## How It Works

1. **`init.sh`** reads `configs/zone_config.json` from the parent project, generates TLS certificates and Django secrets, produces the DLStreamer pipeline config from `lp-config.json`, and writes `docker/.env`.

2. **`download_models.sh`** downloads OpenVINO models (person-detection, person-reidentification) from the Open Model Zoo into a Docker volume. Models that already exist are skipped.

3. **`docker-compose-scenescape.yaml`** starts the full SceneScape stack (webserver, controller, DLStreamer pipeline server, MQTT broker, database, media server) using pre-built images.

4. The DLStreamer pipeline server runs person detection and re-identification inference on RTSP camera streams, publishing tracking data via MQTT for the LP service to consume.
