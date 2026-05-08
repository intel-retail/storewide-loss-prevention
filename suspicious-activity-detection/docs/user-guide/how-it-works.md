# How It Works

This section describes the overall architecture of the Store-wide Loss
Prevention application and explains the function of each service.

## High-level Architecture

At a high level, the system is composed of several microservices that work
together to ingest camera streams, run AI inference on Intel hardware (CPU,
GPU, NPU), evaluate detection rules against per-person session state, and
expose alerts and evidence to a UI for store operators.

The main services in this deployment are:

- [SceneScape](#scenescape)
- [swlp-service](#swlp-service)
- [Behavioral Analysis Service](#behavioral-analysis-service)
- [Alert Service](#alert-service)
- [Frame Storage (SeaweedFS / MinIO)](#frame-storage-seaweedfs--minio)
- [Gradio UI](#gradio-ui)

The following sections describe each service in more detail.

### SceneScape

Intel® SceneScape provides the upstream tracking pipeline:

- **DLStreamer pipeline:** Person detection (`person-detection-retail-0013`)
  and re-identification (`person-reidentification-retail-0277`) on the camera
  stream.
- **Controller:** Multi-camera fusion, persistent person identity (`object_id`),
  per-region enter/exit events with dwell time.
- **MQTT bus (Mosquitto):** Publishes scene-data, region events, region data,
  and camera-image responses on standard topics.

SceneScape is an upstream dependency; the LP application subscribes to its MQTT
topics rather than running its own detector.

### swlp-service

The **swlp-service** is the core of the LP application. It is purely
MQTT-driven and has no direct dependency on the camera pipeline.

- **MQTT subscription and event routing:** Subscribes to SceneScape topics and
  fans events out to handlers.
- **Session state management:** Creates, updates, and expires a `PersonSession`
  per tracked person. Sessions are kept in memory, keyed by
  `(scene_id, object_id)`.
- **Rule evaluation:** A declarative engine driven by `configs/rules.yaml`
  matches triggers (`zone_entry`, `zone_loiter`, `zone_exit`, `ba_result`) and
  conditions against a flat context dict built from the event and session.
- **Action execution:** Each rule emits one or more actions (`alert`,
  `escalate`). Alerts are deduplicated according to a `fire_once_per` scope
  (`zone`, `session`, or `none`). Escalation actions kick off the Behavioral
  Analysis Orchestrator for HIGH_VALUE-zone visits.
- **Frame management:** While a person is in a HIGH_VALUE zone, cropped
  frames are stored at the configured rate per visit. When an alert fires for
  that visit, the relevant frames are copied to a per-alert evidence prefix.

### Behavioral Analysis Service

The **Behavioral Analysis Service** runs pose detection plus a Vision
Language Model (Qwen2.5-VL) to determine whether a person is concealing
merchandise:

- **Trigger:** swlp-service publishes one `ba/requests` message per stored
  HIGH_VALUE-zone frame.
- **Pose pre-filter:** YOLO pose estimation extracts keypoints to detect
  hand-near-body or pocket-region interactions; non-suspicious frames
  short-circuit and emit a `no_match` result without invoking the VLM.
- **VLM inference:** Suspicious candidates are passed to Qwen2.5-VL on the
  configured device (`VLM_DEVICE`). The model returns a natural-language
  justification and a confidence score.
- **Result emission:** A `ba/results` message is published with status
  (`suspicious`, `no_match`, `no_enough_data`), `confidence`,
  `frames_analyzed`, `last_frame_ts`, and `vlm_response`.

The service is **stateless across requests** — each `ba/requests` message
triggers an independent single-shot analysis using the latest K frames in the
visit's frame bucket.

### Alert Service

The **Alert Service** is the downstream consumer of alerts produced by
swlp-service:

- **Time-window dedup:** Per alert type, configurable in `alert-config.yaml`
  (for example, `CONCEALMENT` window 60 s, `LOITERING` window 120 s). Uses
  `sha1(scene_id, person_id, zone_id)` as the dedup key.
- **Delivery routing:** Each alert type maps to one or more delivery
  channels — typically an MQTT topic per type plus a structured log entry.
- **Persistence:** Recent alerts are retained for retrieval via the LP REST
  API (`/api/v1/lp/alerts`).

This is a complementary layer to the swlp-side `fire_once_per` scope: the
swlp-service prevents bursty re-emission *within* a session; the Alert Service
prevents cross-service duplicates over a *time window*.

### Frame Storage (SeaweedFS / MinIO)

Cropped person frames for individuals in HIGH_VALUE zones are written to an
S3-compatible object store with the following layout:

```
bucket: behavioral-frames
├── {scene_id}/{person_id}/{region_id}/{entry_timestamp}/frames/
│   └── {ts_ms}.jpg          # Per-visit BA-bucket frames
└── alerts/
    └── {person_id}/{alert_id}/frames/
        └── {ts_ms}.jpg      # Frames copied on alert (evidence)
```

- **Per-visit retention:** Frames stay until the visit drains (EXIT received,
  `requests_sent == results_received`, no `suspicious` verdict). Suspicious
  visits' frames are preserved as evidence.
- **Evidence retention:** Per-alert prefix keeps the frames that backed the
  alert; retention is configurable.

### Gradio UI

The **Gradio UI** provides a web-based dashboard:

- Live alert feed with severity, alert type, person, and zone.
- Per-alert evidence frame gallery.
- Active session table.
- Service health and zone-count summary from the LP REST API.

The UI is exposed on port 7860 by default and accessed via a standard web
browser.

## Data and Control Flows

Putting the pieces together:

1. **Tracking** — SceneScape ingests camera streams, runs detection and
   re-identification, and publishes per-person region events and scene data
   to MQTT.
2. **Session updates** — swlp-service consumes those events, creating or
   updating a `PersonSession` per `(scene_id, object_id)`. Configurable
   session flags (for example, `visited_high_value`, `visited_checkout`)
   are set automatically on zone entry.
3. **Rule evaluation** — Each event is mapped to a rule trigger
   (`zone_entry`, `zone_loiter`, `zone_exit`). The rule engine evaluates
   `rules.yaml` conditions against a flat context built from the event and
   session, and returns a list of actions.
4. **Action execution** —
   - `alert` actions build an `Alert` object, apply the configured
     `fire_once_per` dedup, build a `details` payload from YAML, and
     publish to the Alert Service.
   - `escalate` actions invoke a registered escalation service. For
     HIGH_VALUE zones this starts the Behavioral Analysis Orchestrator,
     which publishes `getimage` requests to active cameras at the
     configured frame-capture cadence.
5. **Frame capture** — Camera image replies land in the LP service. While the
   person is in a HIGH_VALUE zone, the frame is stored under the per-visit
   prefix and one `ba/requests` message is published.
6. **Behavioral analysis** — The Behavioral Analysis Service consumes
   `ba/requests`, runs pose pre-filter + VLM, and publishes a `ba/results`
   message.
7. **BA result handling** — swlp-service routes `ba/results` through the
   rule engine (trigger `ba_result`). Suspicious results fire a
   `CONCEALMENT` alert; the evidence frames for that visit (up to
   `last_frame_ts`) are copied to `alerts/{person_id}/{alert_id}/frames/`.
   Non-suspicious results that complete a visit (counts match, EXIT seen)
   trigger frame cleanup for that prefix.
8. **Visualization** — The Gradio UI and the LP REST API surface alerts,
   sessions, and evidence frames to the operator.

This modular architecture allows each component (tracking, rules, behavioral
analysis, alerting, UI) to be developed, deployed, and scaled independently
while sharing common infrastructure (MQTT, frame storage, configuration).

## Suspicious Activities Detected

| # | Activity | Trigger | Default Severity |
|---|----------|---------|------------------|
| 1 | Merchandise Concealment | Behavioral Analysis returns `suspicious` | WARNING |
| 2 | Checkout Bypass | Visited HIGH_VALUE then EXIT without CHECKOUT | WARNING / CRITICAL\* |
| 3 | Loitering | Dwell > `loiter_threshold_seconds` in HIGH_VALUE | WARNING |
| 4 | Repeated Visits | Re-entries to same HIGH_VALUE ≥ `repeat_visit_threshold` | WARNING |
| 5 | Restricted Zone Violation | Entered RESTRICTED zone | CRITICAL |

\* Escalates to CRITICAL when `concealment_suspected` is set on the session.

## Configuration Surfaces

| Concern | File |
|---------|------|
| Detection rules, thresholds, severity, dedup, escalation | `configs/rules.yaml` |
| Session flags (zone-visited / external) | `configs/rules.yaml` (`session_flags:`) |
| Zone name → type mapping | `configs/zone_config.json` |
| Alert routing and time-window dedup | `configs/alert-config.yaml` |
| MQTT, storage, services | `configs/.env.example` / `docker/.env` |
| DLStreamer pipeline template | `configs/pipeline-config.json` |

Most behavior changes are YAML edits — adding a new rule, severity, dedup
scope, or alert type does not require Python changes.
