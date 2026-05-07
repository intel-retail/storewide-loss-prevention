# Build from Source

This guide provides detailed instructions for building the POI Re-identification application
container images from source code. Whether you are customizing the application or
troubleshooting deployment issues, this guide walks you through the complete build process.

## Overview

The POI Re-identification application consists of multiple components that work together:

- **POI Backend**: FastAPI server for POI enrollment, FAISS matching, MQTT consumption, and
  alert generation.
- **React UI**: TypeScript/React single-page application for the operator interface.
- **Alert Service**: Dedicated alert fan-out microservice (pre-built image).
- **Redis**: In-memory data store for metadata, events, and caching.
- **MCP Server**: FastMCP server exposing AI tools for LLM/VLM integration.

## Step 1: Clone the Repository

```bash
git clone https://github.com/sainijit/storewide-loss-prevention-retail.git
cd storewide-loss-prevention-retail/person-of-interest
```

## Step 2: Initialize Environment

```bash
make init
```

Use `make init` to generate `.env` and pipeline configs from `zone_config.json`. If you
only need to create `.env` from `.env.example`, run `make init-env` instead. See
[Get Started](../get-started.md#step-2-initialize-environment) for required variables.

## Step 3: Build the Docker Images

### Local Build (No Registry)

```bash
make build REGISTRY=false
```

This builds the following images locally:

| Image                              | Dockerfile               | Description          |
| ---------------------------------- | ------------------------ | -------------------- |
| `person-of-interest-poi-backend`   | `backend/Dockerfile`     | Backend API server   |
| `person-of-interest-ui`            | `ui/Dockerfile`          | React UI             |

> **Note:** The MCP server image (`person-of-interest-mcp-server`) is not built by default.
> To build it, run `docker compose build mcp-server` separately.

### Registry Build

To build and tag images for a container registry:

```bash
export REGISTRY=docker.io/username
make build
```

This builds the images and tags them as:

- `docker.io/username/poi-backend:latest`
- `docker.io/username/poi-ui:latest`

### What the Build Does

The `make build` target performs the following:

1. **Pulls SceneScape images** (if available) from the local Docker cache
2. **Builds POI backend** — multi-stage Docker build with OpenVINO™ runtime, FAISS, and
   Python dependencies
3. **Builds POI UI** — multi-stage Node.js build producing a static nginx container
4. **Tags images** for the specified registry (if `REGISTRY` is not `false`)

## Step 4: Launch the Application

```bash
make up
```

Verify all containers are running:

```bash
make status
```

## Available Make Targets

| Target                       | Description                                      |
| ---------------------------- | ------------------------------------------------ |
| `make build`                 | Build POI backend and UI images                  |
| `make up`                    | Start all POI services                           |
| `make down`                  | Stop all services                                |
| `make restart`               | Restart all services                             |
| `make logs`                  | Follow logs from all POI services                |
| `make log-alert`             | Follow alert service logs                        |
| `make init`                  | Generate .env and pipeline configs from zone_config.json |
| `make init-env`              | Create `.env` from `.env.example`                |
| `make demo`                  | All-in-one: init + models + build + start        |
| `make run-scenescape`        | Start SceneScape only                            |
| `make down-scenescape`       | Stop SceneScape only                             |
| `make export-scene`          | Export scene config from running SceneScape       |
| `make download-models`       | Download OpenVINO AI models                      |
| `make benchmark`             | Single-scene latency benchmark                   |
| `make benchmark-stream-density` | Iterative stream density benchmark            |
| `make consolidate-metrics`   | Consolidate benchmark metrics to CSV             |
| `make test`                  | Run backend unit tests                           |
| `make coverage`              | Run tests with coverage report                   |
| `make coverage-html`         | Generate HTML coverage report                    |
| `make update-submodules`     | Update git submodules from remote                |
| `make clean`                 | Stop and remove containers + volumes             |
| `make clean-secrets`         | Remove generated secrets and .env                |
| `make clean-images`          | Remove LP Docker images                          |
| `make clean-all`             | Clean everything including videos                |
| `make status`                | Show service status                              |
| `make help`                  | Show all available targets                       |

## What to Do Next

- [Get Started](../get-started.md): Complete the initial setup and configuration steps
- [System Requirements](./system-requirements.md): Review hardware and software requirements
- [How to Use the Application](../how-to-use-application.md): Learn about the application's features
- [API Reference](../api-reference.md): Explore the available REST API endpoints
- [Troubleshooting](../troubleshooting.md): Find solutions to common deployment issues
