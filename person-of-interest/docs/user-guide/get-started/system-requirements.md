# System Requirements

This page provides detailed hardware, software, and platform requirements to help you set up
and run the POI Re-identification system efficiently.

## Hardware Requirements

| Component | Minimum                      | Recommended                           |
| --------- | ---------------------------- | ------------------------------------- |
| CPU       | Intel® Core™ i7 (8th gen+)   | Intel® Xeon® Scalable (4th gen+)      |
| RAM       | 16 GB                        | 32 GB                                 |
| Storage   | 50 GB SSD                    | 100 GB NVMe SSD                       |
| GPU       | Not required (CPU inference) | Intel® Arc™ for accelerated inference |
| Network   | 1 Gbps Ethernet              | 10 Gbps Ethernet                      |

## Software Requirements

| Software      | Version       | Purpose                       |
| ------------- | ------------- | ----------------------------- |
| Ubuntu        | 22.04 / 24.04 | Host operating system         |
| Docker        | 24.0+         | Container runtime             |
| Docker Compose| v2.20+        | Multi-container orchestration |
| Python        | 3.10+         | Backend runtime               |
| Git           | 2.30+         | Version control               |
| Make          | 4.3+          | Build automation              |

## Supported Platforms

The POI system has been validated on:

- Intel® Xeon® Scalable Processors (4th and 5th Generation)
- Intel® Core™ Ultra Processors
- Ubuntu 22.04 LTS and 24.04 LTS

## Intel® SceneScape Requirements

The POI system requires Intel® SceneScape with the following DL Streamer models:

| Model                                  | Purpose                | Output               |
| -------------------------------------- | ---------------------  | -------------------- |
| `person-detection-retail-0013`         | Person detection       | Bounding boxes       |
| `face-detection-retail-0004`           | Face detection         | Face bounding boxes  |
| `face-reidentification-retail-0095`    | Face re-identification | 256-d float32 vector |
| `person-reidentification-retail-0277`  | Body re-identification | 256-d float32 vector |

> **Note:** The POI system uses only face embeddings (`face-reidentification-retail-0095`) for
> FAISS matching. Body re-identification embeddings (`person-reidentification-retail-0277`)
> are from a different embedding space and are used only for SceneScape cross-camera tracking.

## Compatibility Notes

- GPU inference acceleration requires Intel® integrated graphics or compatible Intel®
  discrete GPUs with OpenVINO™ support.
- FAISS is configured with `IndexFlatIP` for exact cosine similarity — no GPU acceleration
  is required for the vector search component.
- Redis 8.x is recommended for optimal performance with the event storage and caching layer.

## Validation

- Follow instructions at [Get Started](../get-started.md) to verify the deployment.
