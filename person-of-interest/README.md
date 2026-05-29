# POI Re-identification: Real-Time Retail Loss Prevention

The POI Re-identification system is a real-time retail loss-prevention application that detects enrolled Persons of Interest (POIs) across multiple cameras using Intel® OpenVINO™ face re-identification and FAISS vector search. Integrated with Intel® SceneScape spatial computing, it delivers instant security alerts and supports offline historical investigation to trace suspect movements across all cameras and time ranges.

## Quick Start

```bash
git clone https://github.com/intel-retail/storewide-loss-prevention.git
cd storewide-loss-prevention/person-of-interest

# Initialize submodules
make update-submodules

# Initialize environment and start (pulls images from Docker Hub automatically)
make init
make up
```

For the full setup guide, including device profile selection and building from
source, see [Get Started](./docs/user-guide/get-started.md).

## Documentation

- **Overview**
  - [Overview](./docs/user-guide/index.md): A high-level introduction.
  - [Architecture](./docs/user-guide/index.md#how-it-works): High-level architecture and key components.

- **Getting Started**
  - [Get Started](./docs/user-guide/get-started.md): Step-by-step guide to deploy the application.
  - [System Requirements](./docs/user-guide/get-started/system-requirements.md): Hardware and software requirements.
  - [How to Use the Application](./docs/user-guide/how-to-use-application.md): Explore features and verify functionality.
  - [Troubleshooting](./docs/user-guide/troubleshooting.md): Support and troubleshooting information.

- **Deployment**
  - [Get Started](./docs/user-guide/get-started.md): Step-by-step guide including pre-built Docker Hub images.
  - [Build from Source](./docs/user-guide/get-started/build-from-source.md): Instructions for building from source code.

- **Technical Reference**
  - [MQTT Pipeline Design](./docs/user-guide/mqtt-pipeline-design.md): Deep dive into the MQTT data flow, embedding pipeline, and Redis data model.

- **Benchmarking**
  - [Benchmarking Guide](./docs/user-guide/get-started.md#benchmarking): Single-scene latency, stream density scaling, metrics consolidation, and plotting.

- **API Reference**
  - [API Reference](./docs/user-guide/api-reference.md): Comprehensive reference for the REST API endpoints.

- **Release Notes**
  - [Release Notes](./docs/user-guide/release-notes.md): Information on the latest updates, improvements, and bug fixes.
