# System Requirements

This section lists the hardware, software, and network requirements for
running the Store-wide Loss Prevention application.

## Host Operating System

- Ubuntu 22.04 LTS (recommended and validated).
- Other recent 64-bit Linux distributions may work, but are not fully
  validated.

## Hardware Requirements

- **CPU:**
  - 8 physical cores (16 threads) or more recommended.
  - x86_64 architecture with support for AVX2.

- **System Memory (RAM):**
  - Minimum: 16 GB.
  - Recommended: 32 GB or more for smoother multi-service operation and
    headroom for the VLM.

- **Storage:**
  - Minimum free disk space: 30 GB.
  - Recommended: 60 GB+ to accommodate Docker images, OpenVINO models, the
    VLM weights (Qwen2.5-VL is several GB), sample video, and frame storage
    for behavioral analysis.

- **Graphics / Accelerators:**
  - Required: Intel CPU.
  - Optional (recommended for full experience):
    - Intel integrated or discrete GPU supported by Intel® Graphics Compute
      Runtime — used for person detection, re-identification, pose
      estimation, and VLM inference.
    - Intel NPU supported by the `linux-npu-driver` stack — recommended for
      VLM inference (see [Release Notes](../release-notes.md) for a known
      issue on systems without NPU).

  - The host must expose GPU and NPU devices to Docker, for example:
    - `/dev/dri` (GPU)
    - `/dev/accel/accel0` (NPU)

  - Cameras: at least one RTSP source or a sample video file replayed via
    the bundled `lp-video` container.

## Software Requirements

- **Docker and Container Runtime:**
  - Docker Engine 24.x or newer.
  - Docker Compose v2 (integrated as `docker compose`) or compatible compose
    plugin.
  - Ability to run containers with:
    - Device mappings for GPU/NPU (for the swlp-service, behavioral-analysis,
      and DLStreamer pipeline server).
    - Bind mounts for sample video and generated TLS certificates.

- **Python (for helper scripts and tools):**
  - Python 3.10 or newer recommended.
  - Used primarily for asset preparation scripts (`download_models`) and
    local tooling; application containers include their own Python runtimes.

- **Git and Make:**
  - `git` for cloning the repository.
  - `make` to run provided automation targets (for example, `make demo`,
    `make download-models`, `make clean`).

## AI Models and Workloads

The application bundles several AI workloads, each with its own models and
inputs or outputs:

- **Person Detection (SceneScape DLStreamer pipeline):**
  - **Model:** `person-detection-retail-0013` from Open Model Zoo, converted
    to OpenVINO IR.
  - **Input:** Camera frames (BGR) from the RTSP source or replayed video.
  - **Output:** Per-frame bounding boxes used by SceneScape's tracker.
  - **Target devices:** Intel CPU or GPU via OpenVINO (`DETECTION_DEVICE`).

- **Person Re-Identification (SceneScape DLStreamer pipeline):**
  - **Model:** `person-reidentification-retail-0277` from Open Model Zoo,
    converted to OpenVINO IR.
  - **Input:** Cropped person patches from the detector.
  - **Output:** Embedding vectors used by SceneScape's controller to assign
    persistent `object_id` across cameras and time.
  - **Target devices:** Intel CPU or GPU via OpenVINO (`REID_DEVICE`).

- **Pose Estimation (Behavioral Analysis pre-filter):**
  - **Model:** RTMPose, converted to OpenVINO IR
    (`/models/rtmpose_models/`).
  - **Input:** Cropped person frames from a HIGH_VALUE-zone visit.
  - **Output:** 2D keypoints used to detect hand-near-body or pocket-region
    interactions; non-suspicious frames short-circuit and emit a `no_match`
    result without invoking the VLM.
  - **Target devices:** Intel CPU or GPU via OpenVINO (`POSE_DEVICE`).

- **VLM Concealment Confirmation (Behavioral Analysis):**
  - **Model:** `Qwen/Qwen2.5-VL-7B-Instruct` Vision Language Model
    (`/models/vlm_models/`).
  - **Input:** A small batch of cropped person frames flagged as candidates
    by the pose pre-filter, with a structured prompt.
  - **Output:** A natural-language justification, a `status`
    (`suspicious`, `no_match`, or `no_enough_data`), a `confidence`, and a
    `last_frame_ts`. Published on the `ba/results` MQTT topic.
  - **Target devices:** Intel CPU, GPU, or NPU via OpenVINO (`VLM_DEVICE`).
    NPU is recommended where available.

## Network and Proxy

- **Network Access:**
  - Local network connectivity to access the LP REST API
    (`http://<HOST_IP>:8082`), the Gradio dashboard
    (`http://<HOST_IP>:7860`), and the SceneScape UI (`https://<HOST_IP>`).
  - Optional outbound internet access to download Docker base images,
    OpenVINO models, and Qwen2.5-VL weights (if not pre-cached).

- **Proxy Support (optional):**
  - If your environment uses HTTP/HTTPS proxies, configure:
    - `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY` in the shell before running
      `make`.

## Permissions

- Ability to run Docker as a user in the `docker` group or with `sudo`.
- Sufficient permissions to access device nodes for GPU and NPU (typically
  via membership in groups such as `video` or `render`, or via explicit
  `devices` configuration in Docker Compose).

## Browser Requirements

- Modern web browser (Chrome, Edge, or Firefox) to access the Gradio
  dashboard and SceneScape UI.
- JavaScript enabled.

These requirements are intended for development and evaluation environments.
For any production-like deployment, you should also consider additional
factors such as security hardening, monitoring, backup, retention of
evidence frames in object storage, and resource isolation.
