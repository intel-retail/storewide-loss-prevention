# Release Notes

- [Version 1.1.0](#version-110)
- [Version 1.0.0](#version-100)

## Current Release

### Version 1.1.0

**Release Date**: May 2026

**New**:

- **Detecting suspicious behavior**:

  | Activity | Trigger | Alert Level |
  |---|---|---|
  | Merchandise Concealment | Pose + VLM confirms suspicious behavior | WARNING |
  | Checkout Bypass | Visited HIGH_VALUE zone, exited without CHECKOUT | WARNING / CRITICAL |
  | Loitering | Dwell time exceeds threshold in HIGH_VALUE zone | WARNING |
  | Repeated Visits | Re-entries ≥ threshold to the same HIGH_VALUE zone | WARNING |
  | Restricted Zone Violation | Entered a RESTRICTED zone | CRITICAL |

- **Declarative rule engine**: Rules configuration moved from code to
  declarative `configs/rules.yaml` with variable substitution, session flags,
  and conditional alert severity escalation — no code changes needed to add or
  modify detection rules
- **Behavioral analysis with VLM**: Two-stage pipeline using YOLO pose estimation followed
  by Qwen2.5-VL-7B-Instruct VLM inference via OVMS for concealment detection in
  high-value zones
- **Session flag system**: Auto-set boolean flags (e.g., `visited_high_value`,
  `concealment_suspected`) based on zone visits or external service results, usable as
  rule conditions
- **Checkout bypass detection**: Rule that fires when a person visits a high-value zone
  and exits without passing through checkout, with severity escalated to CRITICAL if
  concealment was detected
- **Dynamic SceneScape configuration**: All environment variables are now auto-generated
  from `configs/zone_config.json` via `make init` — eliminates manual `.env` editing
- **Scene export script**: `make export-scene` exports scene config from a running
  SceneScape instance as an importable zip file
- **Per-camera pipeline configs**: DL Streamer pipeline configs are generated dynamically
  per camera from zone_config.json
- **Benchmark submodule**: Benchmark targets now use performance-tools submodule
  instead of inline scripts.
- **Stream density benchmarking**: Integrated performance-tools submodule with `make benchmark`,
  `make benchmark-stream-density`, and `make consolidate-metrics` targets
- **App-specific controller configs**: Tracker and reid configs moved from SceneScape to
  each app's `configs/` directory (SAD: L2/30)
- **Device resource configs**: Selectable device profiles (`all-gpu-cpu.env`, `all-npu-cpu.env`,
  etc.) via `DEVICE=` parameter with automatic validation

**Fixed**:

- Fixed behavioral analysis timeout when VLM inference exceeds default HTTP timeout
- Fixed rule engine variable substitution not applying environment overrides
- Fixed Gradio UI health check failing on startup race condition

### Version 1.0.0

**Release Date**: April 2026

**Features**:

- Rule-based suspicious activity detection using Intel® SceneScape zone events (entry,
  exit, loiter) with configurable thresholds
- Restricted zone violation alerts — immediate CRITICAL alert on entry to RESTRICTED zones
- Repeated high-value zone visit detection with configurable visit count threshold
- Loitering detection — alerts when dwell time in high-value zones exceeds threshold
- Behavioral analysis service with YOLO pose estimation and VLM-based concealment
  detection using OpenVINO™ Model Server
- Multi-strategy alert delivery via dedicated alert service — MQTT publish, WebSocket,
  and logging with configurable handlers
- Gradio UI for real-time alert monitoring with zone map visualization
- Person session tracking with automatic timeout and zone visit history
- Frame capture from SceneScape cameras with configurable cadence and storage in
  SeaweedFS (S3-compatible object storage)
- Docker Compose deployment with services: swlp-service, behavioral-analysis, alert-service,
  Gradio UI, SeaweedFS, and OVMS-VLM
- Integration with Intel® SceneScape for multi-camera person tracking and zone events

**OpenVINO™ Models Used**:

| Model                                | Purpose                        | Output                    |
| ------------------------------------ | ------------------------------ | ------------------------- |
| `yolov8s`                            | Person detection (DL Streamer) | Person bounding boxes     |
| `person-reidentification-retail-0277`| Person re-identification       | Embedding vector          |
| `yolo26n-pose`                       | Pose estimation                | Skeleton keypoints        |
| `Qwen/Qwen2.5-VL-7B-Instruct`        | Visual language model (VLM)    | Concealment classification|

**HW Used for Validation**:

- Intel® Xeon® Scalable Processor (4th Generation)
- Intel® Arc™ GPU (for VLM inference via OVMS)
- Ubuntu 22.04 LTS

**Known Issues/Limitations**:

- VLM inference latency can be 5–30 seconds per request depending on GPU load; behavioral
  analysis results may lag behind real-time events.
- SeaweedFS frame storage requires sufficient disk space; configure `evidence_retention_hours`
  in `app_config.json` to manage retention.
- SceneScape integration is required for all zone-based detection rules; without SceneScape,
  no suspicious activity detection is possible.
- The `fire_once_per: session` deduplication means a repeated-visit alert will not re-fire
  even if the person continues visiting the zone after the threshold is crossed.
- OVMS VLM service requires GPU with sufficient VRAM for Qwen2.5-VL-7B-Instruct; NPU
  offload is supported via device config but may have reduced throughput.
