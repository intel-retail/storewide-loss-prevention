# Get Started

This guide walks you through building and running the Store-wide Loss
Prevention reference application, including the swlp-service, the Behavioral
Analysis Service (pose + VLM), and the supporting SceneScape stack.

Before you begin, review the
[System Requirements](./get-started/system-requirements.md) to ensure your
environment meets the recommended hardware and software prerequisites.

## 1. Clone the Repository

> **Note:** Make sure you are in the `suspicious-activity-detection`
> directory before running the commands in this guide.

If you have not already cloned the repository that contains this workload, do
so now:

```bash
git clone https://github.com/intel-retail/storewide-loss-prevention.git
cd storewide-loss-prevention/suspicious-activity-detection
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
| `configs/.env.example` | Reference for all environment variables; copied to `docker/.env` by `init.sh` |
| `configs/zone_config.json` | Zone name → type mapping (for example, `aisle1` → `HIGH_VALUE`) |
| `configs/rules.yaml` | Declarative rules: triggers, conditions, actions, and deduplication scope |
| `../scenescape/webserver/storewide-loss-prevention.zip` | Scene map + zone definitions imported into SceneScape |
| `../scenescape/sample_data/lp-camera1.mp4` | Sample video used by the camera replay |

> **Note:** The video file name must match the camera name defined in the
> SceneScape scene (for example, `lp-camera1.mp4` corresponds to camera
> `lp-camera1`).

## 4. Download Sample Video

Download the sample video defined in `configs/zone_config.json` (`video_url`)
to `../scenescape/sample_data/`:

```bash
make download-sample-data
```

The video is saved with the filename specified by the `video_file` key in
`zone_config.json` (for example, `lp-camera1.mp4`). If the file already
exists, the download is skipped.

## 5. Download AI Models

The first run requires downloading the OpenVINO™ and VLM models:

```bash
make download-models
```

This downloads:

- `Qwen/Qwen2.5-VL-7B-Instruct` (VLM weights for behavioral analysis).
- YOLO pose-estimation and person-detection models.

Models are written to the `suspicious-activity-detection/models/` directory
on the host (for example,
`/home/intel/sachin/storewide-loss-prevention/suspicious-activity-detection/models`)
and shared with each container via a Docker volume. Expected layout:

```
suspicious-activity-detection/models/
├── vlm_models/
│   └── Qwen/Qwen2.5-VL-7B-Instruct/
└── yolo_models/
```

## 6. Run the Sample

### Run Everything (SceneScape + LP)

```bash
make up
```

By default this uses the **NPU detect + CPU re-identification** configuration
(`all-npu-cpu.env`). To select a different device profile, pass the `DEVICE`
parameter:

```bash
# NPU detect + CPU reid (default — recommended for long-running deployments)
make up DEVICE=all-npu-cpu.env

# NPU detect + NPU reid
make up DEVICE=all-npu.env

# GPU detect + CPU reid
make up DEVICE=all-gpu-cpu.env

# All GPU (detect + reid on GPU)
make up DEVICE=all-gpu.env

# All CPU (software decode, detect + reid on CPU)
make up DEVICE=all-cpu.env
```

Device profiles are defined in `configs/res/` and control the GStreamer decode
chain, inference device, pre-process backend, and throughput options. A single
unified pipeline template (`configs/pipeline-config.json`) is rendered at
init time using the selected profile.

### Disable the Gradio UI

The Gradio dashboard (`swlp-suscpicious-ui`) is enabled by default. To run
the stack **without** the UI container (for example, headless deployments or
resource-constrained systems), pass `ENABLE_UI=false`:

```bash
# Start without the Gradio UI
make up ENABLE_UI=false

# Start with the UI (default)
make up ENABLE_UI=true
```

When the UI is disabled, the `swlp-suscpicious-ui` container is not built or
started, and the Gradio dashboard at `http://localhost:7860` is unavailable.
All other services (SceneScape, swlp-service, Behavioral Analysis, OVMS, etc.)
continue to run normally.

`make up` performs the following steps automatically:

1. Sources the selected device resource config (`configs/res/<DEVICE>`).
2. Generates TLS certificates, SceneScape secrets, and `docker/.env`.
3. Renders `pipeline-config.json` with device-specific settings.
4. Copies the sample video into the Docker volume.
5. Initializes Docker volumes with correct permissions.
6. Builds the LP, Behavioral Analysis, and Gradio UI container images.
7. Starts all SceneScape and LP containers.
8. Imports the scene map into SceneScape.

## 7. View Logs

```bash
make logs
```

## 8. Stop Services

```bash
# Stop everything
make down

```

## 9. Access the UI

Once running:

| Service | URL | Credentials |
|---------|-----|-------------|
| SceneScape UI | `https://localhost` | `admin` / password printed by `make up` |
| Gradio Dashboard | `http://localhost:7860` | — |
| LP REST API | `http://localhost:8082` | — |
| LP logs | `make logs` | View all service logs |

From the Gradio dashboard you can observe live alerts, evidence frames, and
session state across all configured cameras.

## 10. Inspect Alerts and Sessions via REST

```bash
# Health check
curl http://localhost:8082/health

# Active sessions
curl http://localhost:8082/api/v1/lp/sessions

# Service status + zone counts
curl http://localhost:8082/api/v1/lp/status
```

## 11. Tune Detection Behavior

Detection thresholds, deduplication scope, and severity escalation are defined
declaratively in `configs/rules.yaml`. Edit the file and restart the
swlp-service to apply changes:

```yaml
variables:
  repeat_visit_threshold: 4
  loiter_threshold_seconds: 5

rules:
  - id: loitering
    trigger:
      event_type: zone_loiter
      zone_type: HIGH_VALUE
    conditions:
      - field: dwell_seconds
        op: gt
        value: ${loiter_threshold_seconds:20}
    actions:
      - type: alert
        params:
          alert_type: LOITERING
          severity: WARNING
          fire_once_per: zone
```

See [How It Works](./how-it-works.md) for a full description of triggers,
conditions, and the rule engine.

## Clean Up

```bash
# Stop and remove all containers + volumes
make clean

# Also remove generated secrets and .env
make clean-all
```

## Next Steps

- Learn more about [How It Works](./how-it-works.md) for a high-level
  architectural overview.
- Experiment with different device profiles (`all-npu-cpu.env`,
  `all-gpu-cpu.env`, `all-cpu.env`) to compare NPU, GPU, and CPU behavior.
- Replace the sample video, zone map, or rules with your own assets by
  updating the `configs/`, `models/`, and `videos` volumes.

<!--hide_directive
:::{toctree}
:hidden:

get-started/system-requirements.md

:::
hide_directive-->
