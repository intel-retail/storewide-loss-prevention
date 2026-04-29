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

## 2. Provide Required Assets

Before starting, ensure these files are in place:

| File | Purpose |
|------|---------|
| `configs/.env.example` | Reference for all environment variables; copied to `docker/.env` by `init.sh` |
| `configs/zone_config.json` | Zone name → type mapping (for example, `aisle1` → `HIGH_VALUE`) |
| `configs/rules.yaml` | Declarative rules: triggers, conditions, actions, and dedup scope |
| `../scenescape/webserver/storewide-loss-prevention.zip` | Scene map + zone definitions imported into SceneScape |
| `../scenescape/sample_data/lp-camera1.mp4` | Sample video used by the camera replay |

## 3. Download AI Models

The first run requires downloading the OpenVINO and VLM models:

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

## 4. Run the Sample

### Run Everything (SceneScape + LP)

```bash
make demo
```

`make demo` performs the following steps automatically:

1. Generates TLS certificates, SceneScape secrets, and `docker/.env`.
2. Copies the sample video into the Docker volume.
3. Initializes Docker volumes with correct permissions.
4. Builds the LP, Behavioral Analysis, and Gradio UI container images.
5. Starts all SceneScape and LP containers.
6. Imports the scene map into SceneScape.
7. Tails LP logs to `application.log`.


## 5 Stop Services

```bash
# Stop everything
make down

```

## 6. Access the UI

Once running:

| Service | URL | Credentials |
|---------|-----|-------------|
| SceneScape UI | https://localhost | `admin` / password printed by `make demo` |
| Gradio Dashboard | http://localhost:7860 | — |
| LP REST API | http://localhost:8082 | — |
| LP logs | `application.log` | `tail -f application.log` |

From the Gradio dashboard you can observe live alerts, evidence frames, and
session state across all configured cameras.

## 7. Inspect Alerts and Sessions via REST

```bash
# Health check
curl http://localhost:8082/health

# Recent alerts (filterable)
curl "http://localhost:8082/api/v1/lp/alerts?alert_type=CONCEALMENT"

# Active sessions
curl http://localhost:8082/api/v1/lp/sessions

# Service status + zone counts
curl http://localhost:8082/api/v1/lp/status
```

## 8. Tune Detection Behavior

Detection thresholds, dedup scope, and severity escalation are defined
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
- Experiment with different `VLM_DEVICE` values to compare CPU, GPU, and NPU
  behavior.
- Replace the sample video, zone map, or rules with your own assets by
  updating the `configs/`, `models/`, and `videos` volumes.

<!--hide_directive
:::{toctree}
:hidden:

get-started/system-requirements.md

:::
hide_directive-->

