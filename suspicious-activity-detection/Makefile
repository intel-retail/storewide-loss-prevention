# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Storewide Loss Prevention - Makefile
#
# Full-stack deployment using pre-built SceneScape Docker images.
# No SceneScape source checkout required.
# SceneScape-specific targets are in scenescape/Makefile.

TAG ?= 2026.1.0-rc1
REGISTRY ?= true

SCENESCAPE_DIR := ../scenescape
COMPOSE_SCENESCAPE := docker compose --project-directory . --env-file docker/.env -f $(SCENESCAPE_DIR)/docker-compose.yaml
COMPOSE_LP := docker compose --project-directory . --env-file docker/.env -f docker/docker-compose.yaml
COMPOSE_ALL := docker compose --project-directory . --env-file docker/.env -f $(SCENESCAPE_DIR)/docker-compose.yaml -f docker/docker-compose.yaml
# When stream-density override exists, layer it in
CAMERAS_OVERRIDE := docker/docker-compose.cameras.yaml
COMPOSE_ALL := $(if $(wildcard $(CAMERAS_OVERRIDE)),$(COMPOSE_ALL) -f $(CAMERAS_OVERRIDE),$(COMPOSE_ALL))

RESULTS_PATH ?= ./results

export TAG
REGISTRY_URL = intel/

# Normalize REGISTRY to lowercase for case-insensitive comparison
REGISTRY_LOWER := $(shell echo $(REGISTRY) | tr A-Z a-z)

# Auto-shutdown timeout in minutes (0 = disabled)
AUTO_STOP_MINUTES ?= 20

# Device resource config: pass DEVICE=all-gpu-cpu.env (default), all-gpu.env, or all-cpu.env
DEVICE ?= all-gpu-cpu.env
RESOURCE_CONFIG := configs/res/$(DEVICE)


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


# ---- Initialization ----

.PHONY: init
init:
	@$(MAKE) -C $(SCENESCAPE_DIR) APP_DIR=$(CURDIR) RESOURCE_CONFIG=$(RESOURCE_CONFIG) init

# ---- Full Stack (SceneScape + LP) ----

.PHONY: up

up:
	@$(MAKE) -C $(SCENESCAPE_DIR) APP_DIR=$(CURDIR) RESOURCE_CONFIG=$(RESOURCE_CONFIG) init init-sample-data download-models init-volumes
	@$(MAKE) patch-device
	@mkdir -p $(RESULTS_PATH) $(RESULTS_PATH)/ba $(RESULTS_PATH)/swlp
	@if [ "$(REGISTRY)" = "true" ]; then \
		echo "Pulling LP images from $(REGISTRY_URL)..."; \
		DOCKER_REGISTRY=$(REGISTRY_URL) $(COMPOSE_ALL) pull; \
	else \
		echo "Building LP images..."; \
		$(COMPOSE_ALL) build swlp-service behavioral-analysis swlp-suscpicious-ui; \
	fi
	@echo ""
	@echo "=== Starting all services ==="
	$(COMPOSE_ALL) up -d
	@echo ""
	
	@echo ""
	@echo "Services running:"
	@echo "  SceneScape UI:   https://localhost"
	@echo "  Login:           admin / $$(grep SUPASS docker/.env | cut -d= -f2-)"
	@echo "  Gradio UI:       http://localhost:7860"
	@echo "  LP API:          http://localhost:8082"
	@echo "  Alert Service:   http://localhost:8000"
	@echo "  LP logs:         application.log"
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
	-$(COMPOSE_ALL) down  --remove-orphans
	-docker volume rm storewide-lp_seaweedfs-data 2>/dev/null || true
	@rm -f application.log

# Wipes the corrupt VDMS reid_vector descriptor set + graph metadata that
# SceneScape's first-init creates without dimensions, then bounces the
# controller so it re-creates the schema cleanly. Without this, ReID stays
# stuck in pending_collection/empty-chain forever and re-entries are not
# tracked. Safe to run any time the stack is up.
.PHONY: fix-reid
fix-reid:
	@docker ps --format '{{.Names}}' | grep -q '^storewide-lp-vdms-1$$' || { echo "vdms not running — skipping"; exit 0; }
	@echo "=== Wiping corrupt VDMS reid_vector descriptor + graph ==="
	-@docker stop storewide-lp-scene-1 storewide-lp-swlp-service-1 >/dev/null 2>&1 || true
	@docker exec storewide-lp-vdms-1 sh -c 'rm -rf /vdms/build/db/descriptors/reid_vector /vdms/build/db/graph && mkdir -p /vdms/build/db/descriptors /vdms/build/db/graph'
	@docker restart storewide-lp-vdms-1 >/dev/null
	@sleep 4
	-@docker start storewide-lp-scene-1 >/dev/null 2>&1 || true
	@sleep 6
	-@docker start storewide-lp-swlp-service-1 >/dev/null 2>&1 || true
	@echo "=== ReID schema reset complete ==="

# SceneScape only (without LP services)
.PHONY: run-scenescape
run-scenescape:
	@$(MAKE) -C $(SCENESCAPE_DIR) APP_DIR=$(CURDIR) run

.PHONY: down-scenescape
down-scenescape:
	@$(MAKE) -C $(SCENESCAPE_DIR) APP_DIR=$(CURDIR) down

.PHONY: logs
logs:
	$(COMPOSE_ALL) logs -f

.PHONY: status
status:
	$(COMPOSE_ALL) ps

# ---- LP-Only (SceneScape must be running separately) ----

.PHONY: run-lp
run-lp:
ifeq ($(REGISTRY_LOWER),true)
	@echo "Pulling LP images from $(REGISTRY_URL)..."
	DOCKER_REGISTRY=$(REGISTRY_URL) $(COMPOSE_ALL) pull swlp-service behavioral-analysis swlp-suscpicious-ui
else
	@echo "Building LP images..."
	$(COMPOSE_ALL) build swlp-service behavioral-analysis swlp-suscpicious-ui
endif
	@echo "Starting LP services only (SceneScape must be running)..."
	$(COMPOSE_ALL) up -d swlp-service behavioral-analysis swlp-suscpicious-ui

.PHONY: down-lp
down-lp:
	$(COMPOSE_LP) down



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
BENCHMARK_IMAGE ?= retail-benchmark:latest
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
BENCHMARK_TARGET_LATENCY_MS ?= 2000
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
benchmark:  up ## Run a single-scene benchmark (1 scene, measure latency)
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
benchmark-stream-density: up ## Iteratively increase scenes until latency threshold is reached
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
	@echo "  Full Stack (SceneScape + LP, using pre-built images):"
	@echo "    make run-scenescape  - Single command: init + models + start"
	@echo "    make demo            - Initialize and start everything"
	@echo "    make download-models - Download OpenVINO AI models"
	@echo "    make down            - Stop all services"
	@echo "    make down-scenescape - Stop SceneScape only"
	@echo "    make logs            - Follow logs"
	@echo "    make status          - Show service status"
	@echo ""
	@echo "  LP-Only (SceneScape must be running separately):"
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
