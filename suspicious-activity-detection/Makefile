# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Storewide Loss Prevention - Makefile
#
# Full-stack deployment using pre-built Scenescape Docker images.
# No Scenescape source checkout required.
# Scenescape-specific targets are in scenescape/Makefile.

TAG ?= 2026.1.0
REGISTRY ?= true

SCENESCAPE_DIR := ../scenescape
COMPOSE_SCENESCAPE := docker compose --project-directory . --env-file docker/.env -f $(SCENESCAPE_DIR)/docker-compose.yaml
COMPOSE_LP := docker compose --project-directory . --env-file docker/.env -f docker/docker-compose.yaml
COMPOSE_ALL := docker compose --project-directory . --env-file docker/.env -f $(SCENESCAPE_DIR)/docker-compose.yaml -f docker/docker-compose.yaml
# When stream-density override exists, layer it in
CAMERAS_OVERRIDE := docker/docker-compose.cameras.yaml
COMPOSE_ALL := $(if $(wildcard $(CAMERAS_OVERRIDE)),$(COMPOSE_ALL) -f $(CAMERAS_OVERRIDE),$(COMPOSE_ALL))

# Enable Gradio UI (set to false to skip)
ENABLE_UI ?= true

# Conditionally enable the Gradio UI profile
ENABLE_UI_LOWER := $(shell echo $(ENABLE_UI) | tr A-Z a-z)
ifeq ($(ENABLE_UI_LOWER),true)
  COMPOSE_ALL := $(COMPOSE_ALL) --profile ui
endif

# Also bring up the self-contained search stack as part of `make up` (set false to skip)
ENABLE_SEARCH ?= true
ENABLE_SEARCH_LOWER := $(shell echo $(ENABLE_SEARCH) | tr A-Z a-z)

RESULTS_PATH ?= ./results

export TAG
REGISTRY_URL = intel/

# Normalize REGISTRY to lowercase for case-insensitive comparison
REGISTRY_LOWER := $(shell echo $(REGISTRY) | tr A-Z a-z)

# Auto-shutdown timeout in minutes (0 = disabled)
AUTO_STOP_MINUTES ?= 15

# Device resource config: pass DEVICE=all-gpu-cpu.env, all-gpu.env, all-cpu.env, or all-npu-cpu.env
DEVICE ?= all-npu-cpu.env
RESOURCE_CONFIG := configs/res/$(DEVICE)

# Validate DEVICE before doing anything
.PHONY: validate-device
validate-device:
	@if [ ! -f "$(RESOURCE_CONFIG)" ]; then \
		echo "\033[1;31mERROR: Invalid DEVICE='$(DEVICE)'. File '$(RESOURCE_CONFIG)' not found.\033[0m"; \
		echo "Valid options:"; \
		ls -1 configs/res/*.env 2>/dev/null | xargs -I{} basename {} | sed 's/^/  - /'; \
		exit 1; \
	fi


# ---- Submodules ----

.PHONY: update-submodules
update-submodules:
	@echo "Cloning performance tool repositories"
	cd .. && git submodule deinit -f .
	cd .. && git submodule update --init --remote --recursive
	@echo "Submodules updated (if any present)."

# ---- Models ----

.PHONY: download-models
download-models:
	@echo "Downloading models for Suspicious Activity Detection..."
	bash download_models/model_download.sh

.PHONY: download-sample-data
download-sample-data:
	@echo "Downloading sample video..."
	bash download_models/download_sample_video.sh


# ---- Initialization ----

.PHONY: init
init:
	@$(MAKE) -C $(SCENESCAPE_DIR) APP_DIR=$(CURDIR) RESOURCE_CONFIG=$(RESOURCE_CONFIG) init

# ---- Full Stack (Scenescape + LP) ----

.PHONY: up

up: validate-device
	@$(MAKE) -C $(SCENESCAPE_DIR) APP_DIR=$(CURDIR) RESOURCE_CONFIG=$(RESOURCE_CONFIG) init init-sample-data download-models init-volumes
	@$(MAKE) patch-device
	@mkdir -p $(RESULTS_PATH) $(RESULTS_PATH)/ba $(RESULTS_PATH)/swlp
	@if [ "$(REGISTRY)" = "true" ]; then \
		echo "Pulling LP images from $(REGISTRY_URL)..."; \
		DOCKER_REGISTRY=$(REGISTRY_URL) $(COMPOSE_ALL) pull; \
	else \
		echo "Building LP images..."; \
		BUILD_SERVICES="behavioral-analysis"; \
		if [ "$(ENABLE_UI_LOWER)" = "true" ]; then \
			BUILD_SERVICES="$$BUILD_SERVICES swlp-suscpicious-ui"; \
		fi; \
		$(COMPOSE_ALL) build $$BUILD_SERVICES; \
	fi
	@if [ "$(ENABLE_SEARCH_LOWER)" = "true" ]; then \
		echo "Building recall bridge image..."; \
		$(SEARCH_ENV) $(COMPOSE_FULL) build vss-recall-bridge; \
	fi
	@echo ""
	@echo "=== Starting all services (single docker compose) ==="
	@if [ "$(ENABLE_SEARCH_LOWER)" = "true" ]; then \
		echo "Starting pgserver first and ensuring 'video_summary_db' exists (before pipeline-manager)..."; \
		$(SEARCH_ENV) $(COMPOSE_FULL) up -d pgserver; \
		PG=$$(docker ps --filter name=pgserver --format '{{.Names}}' | head -n1); \
		if [ -n "$$PG" ]; then \
			for i in $$(seq 1 30); do \
				docker exec "$$PG" pg_isready -U scenescape -d scenescape >/dev/null 2>&1 && break; \
				sleep 2; \
			done; \
			docker exec "$$PG" psql -U scenescape -tc "SELECT 1 FROM pg_database WHERE datname='video_summary_db'" 2>/dev/null | grep -q 1 \
				|| docker exec "$$PG" createdb -U scenescape video_summary_db 2>/dev/null \
				&& echo "  ✓ database video_summary_db ready on $$PG" \
				|| echo "  ✓ database video_summary_db already present"; \
		else \
			echo "  ! pgserver container not found — pipeline-manager may need a manual createdb"; \
		fi; \
		$(SEARCH_ENV) $(COMPOSE_FULL) up -d; \
		echo "Ensuring recall bridge is running..."; \
		$(SEARCH_ENV) $(COMPOSE_FULL) up -d vss-recall-bridge 2>/dev/null || true; \
		echo "Connecting recall bridge to the LP network for the unified UI..."; \
		BRIDGE=$$(docker ps --filter name=vss-recall-bridge --format '{{.Names}}' | head -n1); \
		if [ -n "$$BRIDGE" ]; then \
			docker network connect --alias vss-recall-bridge $(LP_NETWORK) "$$BRIDGE" 2>/dev/null \
				&& echo "  ✓ $$BRIDGE attached to $(LP_NETWORK)" \
				|| echo "  ✓ $$BRIDGE already on $(LP_NETWORK)"; \
		fi; \
	else \
		$(COMPOSE_ALL) up -d; \
	fi

	@echo ""
	@echo "Services running:"
	@echo "  Scenescape UI:   https://localhost"
	@echo "  Login:           admin / $$(grep SUPASS docker/.env | cut -d= -f2-)"
	@if [ "$(ENABLE_UI_LOWER)" = "true" ]; then \
		echo "  SAD-DASHBOARD UI:       http://localhost:7860"; \
		echo "  Investigator UI: http://localhost:7860/recall"; \
	else \
		echo "  SAD-DASHBOARD UI:       DISABLED (ENABLE_UI=false)"; \
		echo "  Investigator UI: DISABLED (ENABLE_UI=false)"; \
	fi
	@echo ""
	@echo "To stop: make down"
	@if [ -f /tmp/swlp-auto-stop.pid ]; then \
		kill $$(cat /tmp/swlp-auto-stop.pid) 2>/dev/null || true; \
		rm -f /tmp/swlp-auto-stop.pid; \
	fi
	@if [ "$(AUTO_STOP_MINUTES)" -gt 0 ] 2>/dev/null; then \
		echo ""; \
		echo "=== Auto-shutdown scheduled in $(AUTO_STOP_MINUTES) minutes ==="; \
		( sleep `expr $(AUTO_STOP_MINUTES) \* 60` && cd $(CURDIR) && $(MAKE) down ) &  \
		echo $$! > /tmp/swlp-auto-stop.pid; \
	fi

.PHONY: down
down:
	@BRIDGE=$$(docker ps -a --filter name=vss-recall-bridge --format '{{.Names}}' | head -n1); \
	if [ -n "$$BRIDGE" ]; then \
		docker network disconnect $(LP_NETWORK) "$$BRIDGE" 2>/dev/null || true; \
	fi
	@if [ "$(ENABLE_SEARCH_LOWER)" = "true" ]; then \
		$(SEARCH_ENV) $(COMPOSE_FULL) --profile ui --profile recall-ui down --remove-orphans; \
	else \
		$(COMPOSE_ALL) --profile ui down --remove-orphans; \
	fi
	-docker volume rm storewide-lp_seaweedfs-data 2>/dev/null || true


# Scenescape only (without LP services)
.PHONY: run-scenescape
run-scenescape:
	@$(MAKE) -C $(SCENESCAPE_DIR) APP_DIR=$(CURDIR) run

.PHONY: down-scenescape
down-scenescape:
	@$(MAKE) -C $(SCENESCAPE_DIR) APP_DIR=$(CURDIR) down

.PHONY: export-scene
export-scene:
	@$(MAKE) -C $(SCENESCAPE_DIR) APP_DIR=$(CURDIR) export-scene

.PHONY: logs
logs:
	$(COMPOSE_ALL) logs -f

.PHONY: status
status:
	$(COMPOSE_ALL) ps

# ---- LP-Only (Scenescape must be running separately) ----

.PHONY: run-lp
run-lp:
ifeq ($(REGISTRY_LOWER),true)
	@echo "Pulling LP images from $(REGISTRY_URL)..."
	DOCKER_REGISTRY=$(REGISTRY_URL) $(COMPOSE_ALL) pull  behavioral-analysis swlp-suscpicious-ui
else
	@echo "Building LP images..."
	$(COMPOSE_ALL) build  behavioral-analysis swlp-suscpicious-ui
endif
	@echo "Starting LP services only (SceneScape must be running)..."
	$(COMPOSE_ALL) up -d  behavioral-analysis swlp-suscpicious-ui

.PHONY: down-lp
down-lp:
	$(COMPOSE_LP) down



# ---- Search Stack (VSS search services + recall bridge + investigator UI) ----
# Self-contained: the recall bridge ingests clips from a mounted source file
# (file ingest), so it does NOT need Scenescape or the suspicious-activity stack.
SEARCH_REGISTRY ?= intel/
SEARCH_TAG ?= latest
EMBEDDING_MODEL_NAME ?= CLIP/clip-vit-b-32
SEARCH_ENV := REGISTRY=$(SEARCH_REGISTRY) SEARCH_TAG=$(SEARCH_TAG) EMBEDDING_MODEL_NAME=$(EMBEDDING_MODEL_NAME)
COMPOSE_SEARCH := docker compose --project-directory . -f docker/docker-compose.search.yaml

# Single combined invocation: scenescape + LP (+ cameras override + ui profile) + search.
# Lets `make up`/`make down` manage everything in ONE docker compose command/project.
# The consolidated override disables the search stack's postgres-service and points
# the VSS pipeline-manager at the shared SceneScape `pgserver`.
COMPOSE_FULL := $(COMPOSE_ALL) -f docker/docker-compose.search.yaml -f docker/docker-compose.search-consolidated.yaml

.PHONY: run-search
run-search:
	@echo "=== Starting search stack (VSS search + recall bridge + investigator UI) ==="
	$(SEARCH_ENV) $(COMPOSE_SEARCH) --profile recall-ui up -d --build
	@echo ""
	@echo "Search services running:"
	@echo "  Investigator UI:  http://localhost:$${RECALL_UI_PORT:-7861}/recall"
	@echo "  Recall bridge:    http://localhost:$${RECALL_BRIDGE_PORT:-8090}/health"
	@echo "  Pipeline Manager: http://localhost:$${PM_HOST_PORT:-3001}"
	@echo "  Video Search MS:  http://localhost:$${VS_HOST_PORT:-7890}"
	@echo ""
	@echo "To stop: make down-search"

# Backend-only search stack (no swlp-recall-ui). Used by `make up`, where the
# unified dashboard UI (swlp-suscpicious-ui) serves /recall. The recall bridge is
# attached to the LP network (with its service alias) so the dashboard UI can
# reach it by name within the combined stack.
LP_NETWORK ?= storewide-lp_storewide-lp

.PHONY: down-search
down-search:
	$(SEARCH_ENV) $(COMPOSE_SEARCH) down --remove-orphans

.PHONY: logs-search
logs-search:
	$(SEARCH_ENV) $(COMPOSE_SEARCH) logs -f

.PHONY: status-search
status-search:
	$(SEARCH_ENV) $(COMPOSE_SEARCH) ps



# Patch graph.pbtxt device to match TARGET_DEVICE from configs/.env.example
.PHONY: patch-device
patch-device:
	@TARGET_DEVICE=$$(grep '^TARGET_DEVICE=' configs/.env.example 2>/dev/null | cut -d= -f2- || echo GPU); \
	VLM_MODEL=$$(grep '^VLM_MODEL_NAME=' configs/.env.example 2>/dev/null | cut -d= -f2- || echo Qwen/Qwen2.5-VL-7B-Instruct); \
	GRAPH_FILE=models/vlm_models/$${VLM_MODEL}/graph.pbtxt; \
	if [ -f "$$GRAPH_FILE" ]; then \
		CURRENT=$$(grep -oP '(?<=device: ")[^"]+' "$$GRAPH_FILE" || true); \
		if [ "$$CURRENT" != "$$TARGET_DEVICE" ]; then \
			sed -i "s|device: \"$$CURRENT\"|device: \"$$TARGET_DEVICE\"|" "$$GRAPH_FILE"; \
			echo "  ✓ graph.pbtxt device updated: $$CURRENT → $$TARGET_DEVICE"; \
		else \
			echo "  ✓ graph.pbtxt device already set to $$TARGET_DEVICE"; \
		fi \
	fi

# ---- Stream Density & Benchmarking ----

PERF_TOOLS_DIR := ../performance-tools/benchmark-scripts
PERF_TOOLS_DOCKER_DIR := ../performance-tools/docker
SWLP_STREAM_DENSITY := $(PERF_TOOLS_DIR)/swlp_stream_density.py

# Benchmark image configuration
BENCHMARK_IMAGE ?= retail-benchmark:2026.1.0-rc1
REGISTRY_BENCHMARK ?= intel/retail-benchmark:$(TAG)

.PHONY: fetch-benchmark
fetch-benchmark: ## Pull benchmark image from registry
	@echo "Fetching benchmark image from registry..."
	docker pull $(REGISTRY_BENCHMARK)
	docker tag $(REGISTRY_BENCHMARK) $(BENCHMARK_IMAGE)
	@echo "Benchmark image ready: $(BENCHMARK_IMAGE)"

.PHONY: build-benchmark
build-benchmark: ## Build benchmark Docker image
	@echo "Building benchmark Docker image..."
	@if [ "$(REGISTRY)" = "true" ]; then \
		$(MAKE) fetch-benchmark; \
	else \
		cd $(PERF_TOOLS_DOCKER_DIR) && $(MAKE) build-benchmark; \
	fi
	@echo "Benchmark image built: $(BENCHMARK_IMAGE)"

# Benchmark configuration (override via env or CLI)
BENCHMARK_TARGET_LATENCY_MS ?= 10000
BENCHMARK_LATENCY_METRIC ?= avg
BENCHMARK_SCENE_INCREMENT ?= 1
BENCHMARK_INIT_DURATION ?= 90
BENCHMARK_STABILISE_DURATION ?= 30
BENCHMARK_MAX_ITERATIONS ?= 50
BENCHMARK_MIN_THROUGHPUT_RATIO ?= 0.5

# Short-name aliases
ifdef TARGET_LATENCY_MS
BENCHMARK_TARGET_LATENCY_MS := $(TARGET_LATENCY_MS)
endif
ifdef LATENCY_METRIC
BENCHMARK_LATENCY_METRIC := $(LATENCY_METRIC)
endif
ifdef SCENE_INCREMENT
BENCHMARK_SCENE_INCREMENT := $(SCENE_INCREMENT)
endif

BENCHMARK_VENV := cd $(PERF_TOOLS_DIR) && \
	python3 -m venv venv 2>/dev/null && \
	. venv/bin/activate && \
	pip3 install -q -r requirements.txt && \
	pip3 install -q -e . && \
	cd $(CURDIR) && \
	export PYTHONPATH=$(PERF_TOOLS_DIR):$$PYTHONPATH

.PHONY: benchmark
benchmark: ## Run a single-scene benchmark (1 scene, measure latency)
	@$(MAKE) up ENABLE_UI=false
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║       SWLP Benchmark — Single Scene                             ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo "  Target Latency: $(BENCHMARK_TARGET_LATENCY_MS)ms"
	@echo "  Results Dir:    $(RESULTS_PATH)"
	@echo ""
	mkdir -p $(RESULTS_PATH)
	$(BENCHMARK_VENV) && \
	python3 $(SWLP_STREAM_DENSITY) run $(CURDIR) \
		--single_run \
		--scenes 1 \
		--target_latency_ms $(BENCHMARK_TARGET_LATENCY_MS) \
		--latency_metric $(BENCHMARK_LATENCY_METRIC) \
		--init_duration $(BENCHMARK_INIT_DURATION) \
		--stabilise_duration $(BENCHMARK_STABILISE_DURATION) \
		--results_dir $(CURDIR)/$(RESULTS_PATH); \
	rc=$$?; $(MAKE) down; exit $$rc

.PHONY: benchmark-stream-density
benchmark-stream-density: ## Iteratively increase scenes until latency threshold is reached
	@$(MAKE) up ENABLE_UI=false
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║       SWLP Stream Density — Scene-Based Latency Scaling         ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo "  Target Latency:   $(BENCHMARK_TARGET_LATENCY_MS)ms"
	@echo "  Latency Metric:   $(BENCHMARK_LATENCY_METRIC)"
	@echo "  Min Throughput:   $(BENCHMARK_MIN_THROUGHPUT_RATIO)"
	@echo "  Scene Increment:  +$(BENCHMARK_SCENE_INCREMENT) per iteration"
	@echo "  Init Duration:    $(BENCHMARK_INIT_DURATION)s"
	@echo "  Stabilise:        $(BENCHMARK_STABILISE_DURATION)s"
	@echo "  Max Iterations:   $(BENCHMARK_MAX_ITERATIONS)"
	@echo "  Results Dir:      $(RESULTS_PATH)"
	@echo ""
	mkdir -p $(RESULTS_PATH)
	$(BENCHMARK_VENV) && \
	python3 $(SWLP_STREAM_DENSITY) run $(CURDIR) \
		--target_latency_ms $(BENCHMARK_TARGET_LATENCY_MS) \
		--latency_metric $(BENCHMARK_LATENCY_METRIC) \
		--scene_increment $(BENCHMARK_SCENE_INCREMENT) \
		--init_duration $(BENCHMARK_INIT_DURATION) \
		--stabilise_duration $(BENCHMARK_STABILISE_DURATION) \
		--max_iterations $(BENCHMARK_MAX_ITERATIONS) \
		--min_throughput_ratio $(BENCHMARK_MIN_THROUGHPUT_RATIO) \
		--results_dir $(CURDIR)/$(RESULTS_PATH); \
	rc=$$?; $(MAKE) down; exit $$rc

.PHONY: consolidate-metrics
consolidate-metrics: ## Consolidate benchmark metrics from results/ into a CSV
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║                   Consolidating Metrics                        ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	$(BENCHMARK_VENV) && \
	python3 $(PERF_TOOLS_DIR)/consolidate_multiple_run_of_metrics.py \
		--root_directory $(CURDIR)/$(RESULTS_PATH) \
		--output $(CURDIR)/$(RESULTS_PATH)/consolidated_metrics.csv
	@echo ""
	@echo "Results written to: $(RESULTS_PATH)/consolidated_metrics.csv"

.PHONY: stream-density
stream-density: benchmark-stream-density ## Alias for benchmark-stream-density

.PHONY: generate-stream-density
generate-stream-density:
	@echo "Generating stream-density overrides only..."
	$(BENCHMARK_VENV) && \
	python3 $(SWLP_STREAM_DENSITY) generate $(CURDIR)
	@echo "To start: make demo"

.PHONY: clean-stream-density
clean-stream-density:
	$(BENCHMARK_VENV) && \
	python3 $(SWLP_STREAM_DENSITY) clean $(CURDIR)

.PHONY: down-stream-density
down-stream-density:
	$(BENCHMARK_VENV) && \
	python3 $(SWLP_STREAM_DENSITY) down $(CURDIR)

# ---- Build ----

.PHONY: build
build:
	$(COMPOSE_SCENESCAPE) build

.PHONY: build-lp
build-lp: ## Build LP container images (swlp-service, behavioral-analysis, swlp-suscpicious-ui)
	docker build -t intel/swlp-behavioral-analysis:$(TAG) -f behavioral-analysis/Dockerfile behavioral-analysis/
	docker build -t intel/swlp-service:$(TAG) -f swlp-service/Dockerfile .
	docker build -t intel/swlp-suscpicious-ui:$(TAG) -f ui/Dockerfile ui/

# ---- Cleanup ----

.PHONY: clean
clean:
	-$(COMPOSE_SCENESCAPE) down -v
	-$(COMPOSE_LP) down -v 2>/dev/null || true

.PHONY: clean-secrets
clean-secrets:
	rm -rf $(SCENESCAPE_DIR)/secrets/ca $(SCENESCAPE_DIR)/secrets/certs $(SCENESCAPE_DIR)/secrets/django
	rm -rf $(SCENESCAPE_DIR)/secrets/pgserver $(SCENESCAPE_DIR)/secrets/supass
	rm -f $(SCENESCAPE_DIR)/secrets/controller.auth $(SCENESCAPE_DIR)/secrets/browser.auth $(SCENESCAPE_DIR)/secrets/calibration.auth
	rm -f docker/.env

.PHONY: clean-all
clean-all: clean clean-secrets
	rm -rf $(SAMPLE_DATA_DIR)/*.ts

# ---- Help ----

.PHONY: help
help:
	@echo "Storewide Loss Prevention - Available targets:"
	@echo ""
	@echo "  Full Stack (Scenescape + LP, using pre-built images):"
	@echo "    make run-scenescape  - Single command: init + models + start"
	@echo "    make demo            - Initialize and start everything"
	@echo "    make download-models - Download OpenVINO AI models"
	@echo "    make download-sample-data - Download sample video from CAMERA_URL"
	@echo "    make down            - Stop all services"
	@echo "    make down-scenescape - Stop Scenescape only"
	@echo "    make logs            - Follow logs"
	@echo "    make status          - Show service status"
	@echo ""
	@echo "  LP-Only (Scenescape must be running separately):"
	@echo "    make demo-lp         - Start LP services only"
	@echo "    make down-lp         - Stop LP services"
	@echo ""
	@echo "  Build:"
	@echo "    make build           - Build LP container images"
	@echo ""
	@echo "  Stream Density & Benchmarking:"	@echo "    make fetch-benchmark           - Pull benchmark image from registry"
	@echo "    make build-benchmark           - Build benchmark Docker image"	@echo "    make benchmark                 - Single-scene benchmark (1 scene)"
	@echo "    make benchmark-stream-density  - Increase scenes until latency threshold"
	@echo "    make consolidate-metrics       - Consolidate metrics to CSV"
	@echo "    make stream-density            - Alias for benchmark-stream-density"
	@echo "    make generate-stream-density   - Generate overrides only (no start)"
	@echo "    make clean-stream-density      - Revert to single camera"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean           - Stop and remove containers + volumes"
	@echo "    make clean-secrets   - Remove generated secrets and .env"
	@echo "    make clean-all       - Clean everything including videos"
	@echo ""
	@echo "  Prerequisites:"
	@echo "    - Docker with Compose v2"
	@echo "    - openssl, python3, curl"
	@echo "    - Internet access (to download sample videos)"
	@echo ""
	@echo "  Device Configuration:"
	@echo "    make up DEVICE=all-gpu-cpu.env - GPU decode + GPU detect + CPU reid (default)"
	@echo "    make up DEVICE=all-gpu.env     - GPU decode + GPU detect + GPU reid"
	@echo "    make up DEVICE=all-cpu.env     - CPU decode + CPU detect + CPU reid"
	@echo "    Resource configs: configs/res/all-gpu.env, configs/res/all-cpu.env"
