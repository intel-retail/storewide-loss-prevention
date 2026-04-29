# Release Notes: Store-wide Loss Prevention

## Version 1.0.0 - May 3, 2026

This is the initial release of the application; it is therefore considered a
preview version.

Store-wide Loss Prevention showcases how a single Intel®-powered edge system
can ingest multi-camera retail video, perform multi-person tracking and
re-identification with SceneScape, evaluate declarative rules against
per-person session state, and confirm suspicious behavior with a Vision
Language Model — all within one integrated dashboard.

It demonstrates how heterogeneous workloads — from CPU-based detection and
re-identification through GPU/NPU-accelerated VLM inference — can coexist
efficiently on one platform without compromising real-time performance.

### New

The initial feature set of the application is now available:

- **Multi-camera person tracking** via SceneScape (DLStreamer pipeline +
  controller, MQTT bus).
- **Per-person session management** with configurable session flags
  (`visited_high_value`, `visited_checkout`, `visited_exit`,
  `concealment_suspected`).
- **Declarative rule engine** driven by `configs/rules.yaml`:
  - Triggers: `zone_entry`, `zone_loiter`, `zone_exit`, `ba_result`.
  - Conditions with comparison operators (`eq`, `gte`, `gt`, etc.) and
    indexed dict lookups (`zone_visit_counts[region_id]`).
  - Actions: `alert` and `escalate`.
  - Generic `fire_once_per` dedup scope (`zone`, `session`, `none`).
  - YAML-defined `details` payload built via `$ctx.<field>` and
    `$param.<field>` substitution.
- **Built-in detection rules:**
  - Merchandise Concealment (VLM-confirmed)
  - Checkout Bypass (with CRITICAL escalation under concealment)
  - Loitering (per-zone dedup)
  - Repeated Visits (per-session dedup)
  - Restricted Zone Violation
- **Behavioral Analysis Service** with YOLO pose pre-filter and Qwen2.5-VL
  inference; CPU, GPU, and NPU device targets supported via `VLM_DEVICE`.
- **Per-visit BA frame lifecycle** with explicit request/result accounting:
  - Frames live under
    `{scene_id}/{person_id}/{region_id}/{entry_ts}/frames/`.
  - On `suspicious` verdict, frames up to `last_frame_ts` are copied to
    `alerts/{person_id}/{alert_id}/frames/` as evidence; originals are
    preserved.
  - On non-suspicious visits where requests match results and EXIT is seen,
    the per-visit prefix is cleaned up.
- **Alert Service** with per-type time-window dedup and pluggable delivery
  (MQTT topic per alert type, structured log).
- **REST API** for alerts, sessions, status, and runtime zone management
  (`/api/v1/lp/zones`).
- **Gradio dashboard** for live alerts, evidence frames, and session state.

### Known Issues

- **Continuous-loop demo videos:** When a sample video is configured to loop
  seamlessly, SceneScape may keep emitting events under the same
  `object_id`. Session state and `fire_once_per` dedup persist across loop
  boundaries, suppressing repeat alerts. Tune `session_timeout_seconds` in
  `configs/rules.yaml` to be shorter than the loop gap, or rely on re-id
  re-assigning a new `object_id` between loops.
- **NPU device errors at startup:** Containers configured with
  `--device=/dev/accel/accel0` may fail with a `gathering device information`
  error on systems without NPU. Workaround: comment out the NPU device
  bindings in `docker/docker-compose.yaml`:

  ```yaml
  # devices:
  #   - /dev/dri
  #   - /dev/accel
  #   - /dev/accel/accel0
  ```

- **Session state is in-memory only:** Restarting the swlp-service container
  drops all active session state. There is no persistence layer in this
  release; per-trip context is forgotten across restarts.
- **Single-instance deployment:** Horizontal scaling of swlp-service is not
  supported in this release because session state is process-local. Multi
  instance deployments would each maintain an independent view of the same
  person.
