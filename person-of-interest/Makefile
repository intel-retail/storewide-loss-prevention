# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Storewide Loss Prevention - Makefile
#
# Full-stack deployment using pre-built SceneScape Docker images.
# No SceneScape source checkout required.
# SceneScape-specific targets are in scenescape/Makefile.

SCENESCAPE_DIR := ../scenescape

# Device resource config: pass DEVICE=all-cpu.env, all-gpu-cpu.env, all-gpu.env, all-npu-cpu.env, or all-npu.env
# When not passed on the command line, auto-detect from docker/.env (written by 'make init')
# so the NPU overlay is automatically applied on subsequent 'make up' calls (ITEP-92805).
_DEVICE_FROM_ENV := $(shell grep '^RESOURCE_CONFIG=' docker/.env 2>/dev/null | sed 's|.*configs/res/||' | tr -d '[:space:]')
DEVICE ?= $(if $(_DEVICE_FROM_ENV),$(_DEVICE_FROM_ENV),all-gpu.env)
RESOURCE_CONFIG := configs/res/$(DEVICE)

COMPOSE_SCENESCAPE := docker compose --project-directory . --env-file docker/.env -f $(SCENESCAPE_DIR)/docker-compose.yaml -f docker-compose.scenescape-overrides.yml
COMPOSE_LP := docker compose --project-directory . --env-file docker/.env -f docker/docker-compose.yaml
COMPOSE_ALL := docker compose --project-directory . --env-file docker/.env -f $(SCENESCAPE_DIR)/docker-compose.yaml -f docker-compose.scenescape-overrides.yml -f docker/docker-compose.yaml
# When stream-density override exists, layer it in only for density > 1
CAMERAS_OVERRIDE := docker/docker-compose.cameras.yaml
_STREAM_DENSITY_FROM_ENV := $(shell grep '^STREAM_DENSITY=' docker/.env 2>/dev/null | cut -d= -f2 | tr -d '[:space:]')
COMPOSE_ALL := $(if $(and $(wildcard $(CAMERAS_OVERRIDE)),$(filter-out 0 1,$(_STREAM_DENSITY_FROM_ENV))),$(COMPOSE_ALL) -f $(CAMERAS_OVERRIDE),$(COMPOSE_ALL))
# When an NPU device config is selected, layer in the NPU device override (ITEP-92805)
NPU_OVERLAY := docker-compose.npu-overrides.yml
COMPOSE_SCENESCAPE := $(COMPOSE_SCENESCAPE) $(if $(findstring npu,$(DEVICE)),-f $(NPU_OVERLAY),)
COMPOSE_ALL := $(COMPOSE_ALL) $(if $(findstring npu,$(DEVICE)),-f $(NPU_OVERLAY),)

RESULTS_PATH ?= ./results

# ---- Initialization ----

.PHONY: init
init:
	@if [ ! -f "$(RESOURCE_CONFIG)" ]; then \
		echo "\033[1;31mERROR: Invalid DEVICE='$(DEVICE)'. File '$(RESOURCE_CONFIG)' not found.\033[0m"; \
		echo "Valid options: $$(ls -1 configs/res/*.env 2>/dev/null | xargs -I{} basename {} | tr '\n' ' ')"; \
		exit 1; \
	fi
	@if [ -z "$(HOST_IP)" ]; then \
		echo ""; \
		echo "\033[1;31mERROR: HOST_IP is not set.\033[0m"; \
		echo ""; \
		echo "  HOST_IP is required for WebRTC camera streams to work in the Live Alerts UI."; \
		echo "  Set it to the network-reachable IP of this machine before running make init:"; \
		echo ""; \
		echo "    export HOST_IP=\$$(ip route get 1 | awk '{print \$$7; exit}') && make init"; \
		echo ""; \
		exit 1; \
	fi
	@echo "  HOST_IP=$(HOST_IP) (used for WebRTC ICE candidates)"
	@$(MAKE) -C $(SCENESCAPE_DIR) APP_DIR=$(CURDIR) RESOURCE_CONFIG=$(RESOURCE_CONFIG) init download-models init-sample-data init-volumes
	@cp -f .env.example .env
	@# Pre-create bind-mounted directories as the host user so poi-backend
	@# (which runs as HOST_UID:HOST_GID) can write uploads and results.
	@mkdir -p data/uploads data/faiss results

# ---- Full Stack (SceneScape + LP) ----

.PHONY: demo
demo:
	@$(MAKE) -C $(SCENESCAPE_DIR) APP_DIR=$(CURDIR) RESOURCE_CONFIG=$(RESOURCE_CONFIG) init init-sample-data download-models init-volumes
	@echo ""
	@echo "=== Building POI images ==="
	$(COMPOSE_POI) build poi-backend ui
	@echo ""
	@echo "=== Starting all services ==="
	$(MAKE) up
	@echo ""
	@echo "To stop: make down"

.PHONY: lp-down
lp-down:
	-$(COMPOSE_POI) --profile benchmark down
	-$(COMPOSE_SCENESCAPE) down
	@rm -f application.log
	@rm -f $(SCENESCAPE_DIR)/dlstreamer-pipeline-server/person-of-interest-*-[0-9]*-pipeline-config.json 2>/dev/null || true
	@rm -f docker/docker-compose.cameras.yaml 2>/dev/null || true

# SceneScape only (without LP services)
.PHONY: run-scenescape
run-scenescape:
	@$(MAKE) -C $(SCENESCAPE_DIR) APP_DIR=$(CURDIR) run

.PHONY: down-scenescape
down-scenescape:
	@$(MAKE) -C $(SCENESCAPE_DIR) APP_DIR=$(CURDIR) down

.PHONY: export-scene
export-scene:
	@$(MAKE) -C $(SCENESCAPE_DIR) APP_DIR=$(CURDIR) export-scene

.PHONY: import-scene
import-scene:
	@$(MAKE) -C $(SCENESCAPE_DIR) APP_DIR=$(CURDIR) import-scene

.PHONY: lp-logs
lp-logs:
	$(COMPOSE_ALL) logs -f

.PHONY: status
status:
	$(COMPOSE_ALL) ps

# ---- LP-Only (SceneScape must be running separately) ----

.PHONY: demo-lp
demo-lp:
	@echo "Building LP images..."
	$(COMPOSE_ALL) build swlp-service behavioral-analysis gradio-ui
	@echo "Starting LP services only (SceneScape must be running)..."
	$(COMPOSE_LP) up -d

.PHONY: down-lp
down-lp:
	$(COMPOSE_LP) down

# ---- Models ----

.PHONY: download-models
download-models:
	@echo "Downloading models for Person of Interest (FP32 + FP16 for multi-device support)..."
	@bash $(SCENESCAPE_DIR)/scripts/download_models.sh --precisions FP32,FP16,FP16-INT8
	@bash scripts/download_clip_reid.sh

# ---- Stream Density & Benchmarking ----

PERF_TOOLS_DIR := ../performance-tools/benchmark-scripts
SAD_STREAM_DENSITY := $(PERF_TOOLS_DIR)/sad_stream_density.py
POI_STREAM_DENSITY := $(PERF_TOOLS_DIR)/poi_stream_density.py

# Load benchmark vars from docker/.env (if it exists) so values like
# BENCHMARK_TARGET_LATENCY_MS, BENCHMARK_INIT_DURATION etc. are picked up.
# Only BENCHMARK_* lines are extracted to avoid parsing issues with other values.
ifneq (,$(wildcard docker/.env))
$(foreach line,$(shell grep '^BENCHMARK_' docker/.env 2>/dev/null),$(eval $(line)))
endif

# Benchmark configuration (override via env or docker/.env)
BENCHMARK_TARGET_LATENCY_MS ?= 30000
BENCHMARK_LATENCY_METRIC ?= avg
BENCHMARK_SCENE_INCREMENT ?= 1
BENCHMARK_INIT_DURATION ?= 120
BENCHMARK_STABILISE_DURATION ?= 60
BENCHMARK_DURATION ?= 120
BENCHMARK_MAX_ITERATIONS ?= 50
BENCHMARK_MIN_THROUGHPUT_RATIO ?= 0.5
# Max total seconds to wait for an alert from a new camera per iteration.
# The benchmark retries in INIT_DURATION windows until an alert is received
# or this limit is hit.  Default covers ~3 video cycles (~55s each).
# Increase for CPU mode: MAX_ALERT_WAIT=300 make benchmark-stream-density DEVICE=all-cpu.env
BENCHMARK_MAX_ALERT_WAIT ?= 180

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

# Resolve the resource-config path once so it can be forwarded to poi_stream_density.py
BENCHMARK_RESOURCE_CONFIG := $(CURDIR)/$(RESOURCE_CONFIG)

BENCHMARK_VENV := cd $(PERF_TOOLS_DIR) && \
	python3 -m venv venv 2>/dev/null && \
	. venv/bin/activate && \
	pip3 install -q -r requirements.txt && \
	pip3 install -q -e . && \
	cd $(CURDIR)

TAG ?= 2026.1.0
REGISTRY_BENCHMARK ?= intel/retail-benchmark:$(TAG)

.PHONY: fetch-benchmark
fetch-benchmark: ## Pull benchmark metrics-collector image from registry
	@echo "$(BLUE)Fetching benchmark image from registry...$(NC)"
	docker pull $(REGISTRY_BENCHMARK)
	@echo "$(GREEN)Benchmark image ready$(NC)"

.PHONY: build-benchmark
build-benchmark: ## Build benchmark metrics-collector Docker image
	@echo "$(BLUE)Building benchmark Docker image...$(NC)"
	@if [ "$(REGISTRY)" = "true" ]; then \
		$(MAKE) fetch-benchmark; \
	else \
		cd ../performance-tools && $(MAKE) build-benchmark-docker; \
	fi

# Start metrics-collector (benchmark profile)
.PHONY: _start-metrics-collector
_start-metrics-collector:
	@echo "$(BLUE)=== Starting metrics-collector ===$(NC)"
	@if ! docker image inspect $(REGISTRY_BENCHMARK) >/dev/null 2>&1; then \
		echo "$(YELLOW)  metrics-collector image not found locally — attempting pull...$(NC)"; \
		docker pull $(REGISTRY_BENCHMARK) 2>/dev/null || true; \
	fi
	@if docker image inspect $(REGISTRY_BENCHMARK) >/dev/null 2>&1; then \
		$(COMPOSE_POI) --profile benchmark up -d metrics-collector && \
		echo "$(GREEN)  metrics-collector started$(NC)"; \
	else \
		echo "$(YELLOW)  metrics-collector image not available — skipping (run 'make build-benchmark' to build locally)$(NC)"; \
	fi

# Stop metrics-collector
.PHONY: _stop-metrics-collector
_stop-metrics-collector:
	@if docker ps -q -f name=poi-metrics-collector 2>/dev/null | grep -q .; then \
		echo "$(BLUE)=== Stopping metrics-collector ===$(NC)"; \
		echo "  Flushing metrics (10 s grace period)..."; \
		sleep 10; \
		docker stop -t 5 poi-metrics-collector 2>/dev/null || true; \
		$(COMPOSE_POI) --profile benchmark rm -f metrics-collector 2>/dev/null || true; \
	fi

.PHONY: benchmark
benchmark: _benchmark-fresh-start _benchmark-enroll _start-metrics-collector ## Single-scene POI benchmark with auto-enroll
	@$(MAKE) --no-print-directory poi-benchmark BENCHMARK_RESOURCE_CONFIG=$(BENCHMARK_RESOURCE_CONFIG); ret=$$?; \
		$(MAKE) --no-print-directory _stop-metrics-collector; \
		$(MAKE) --no-print-directory _benchmark-cleanup; \
		exit $$ret

.PHONY: benchmark-stream-density
benchmark-stream-density: _benchmark-fresh-start _benchmark-enroll _start-metrics-collector ## Stream density with auto-enroll
	@$(MAKE) --no-print-directory poi-benchmark-stream-density BENCHMARK_RESOURCE_CONFIG=$(BENCHMARK_RESOURCE_CONFIG); ret=$$?; \
		$(MAKE) --no-print-directory _stop-metrics-collector; \
		$(MAKE) --no-print-directory _benchmark-cleanup; \
		$(MAKE) --no-print-directory down; \
		exit $$ret

# Internal: tear down stale services for a clean benchmark
.PHONY: _benchmark-fresh-start
_benchmark-fresh-start:
	@echo "$(BLUE)=== Cleaning up previous benchmark state ===$(NC)"
	@echo "  Device: $(DEVICE)  Resource config: $(RESOURCE_CONFIG)"
	@$(MAKE) --no-print-directory down 2>/dev/null || true
	@echo "$(BLUE)=== Removing POI Compose volumes ===$(NC)"
	@$(COMPOSE_POI) down -v --remove-orphans 2>/dev/null || true
	@echo "$(BLUE)=== Clearing stale results ===$(NC)"
	@mkdir -p $(RESULTS_PATH)
	@if [ "$$(stat -c '%U' $(RESULTS_PATH) 2>/dev/null)" != "$$(id -un)" ]; then \
		docker run --rm -v "$(CURDIR)/$(RESULTS_PATH)":/r debian:bookworm-slim \
			chown -R "$$(id -u):$$(id -g)" /r; \
	fi
	@rm -f $(RESULTS_PATH)/cpu_usage.log $(RESULTS_PATH)/memory_usage.log \
		$(RESULTS_PATH)/disk_bandwidth.log $(RESULTS_PATH)/pcm*.csv \
		$(RESULTS_PATH)/pcm-power.log $(RESULTS_PATH)/qmassa*.json \
		$(RESULTS_PATH)/qmassa_error.log $(RESULTS_PATH)/npu_usage.csv \
		$(RESULTS_PATH)/vlm_application_metrics*.txt \
		$(RESULTS_PATH)/consolidated_metrics.csv 2>/dev/null || true
	@echo "$(BLUE)=== Clearing stale FAISS index ===$(NC)"
	@rm -f data/faiss/poi.index data/faiss/id_map.json
	@echo "$(BLUE)=== Initializing with DEVICE=$(DEVICE) ===$(NC)"
	@$(MAKE) --no-print-directory init

# Internal: ensure services are running, then enroll sample POI before benchmark
.PHONY: _benchmark-enroll
_benchmark-enroll: up
	@echo "$(BLUE)=== Enrolling benchmark POI from sample_data ===$(NC)"
	@for i in 1 2 3 4 5 6 7 8 9 10; do \
		if curl -sf http://localhost:8000/api/v1/status >/dev/null 2>&1; then break; fi; \
		echo "  Waiting for poi-backend..."; sleep 5; \
	done
	@ENROLL_RESP=$$(curl -sf -X POST http://localhost:8000/api/v1/poi \
		-F "images=@sample_data/poi_1.png" \
		-F "severity=high" \
		-F "description=Benchmark POI" 2>&1); \
	POI_ID=$$(echo "$$ENROLL_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('poi_id',''))" 2>/dev/null); \
	if [ -n "$$POI_ID" ]; then \
		echo "  Enrolled POI: $$POI_ID"; \
		echo "$$POI_ID" > .benchmark_poi_id; \
	else \
		echo "$(RED)  Failed to enroll POI: $$ENROLL_RESP$(NC)"; \
		exit 1; \
	fi

# Internal: delete benchmark POI and clean data after benchmark
.PHONY: _benchmark-cleanup
_benchmark-cleanup:
	@echo ""
	@echo "$(BLUE)=== Cleaning up benchmark POI ===$(NC)"
	@if [ -f .benchmark_poi_id ]; then \
		POI_ID=$$(cat .benchmark_poi_id); \
		curl -sf -X DELETE "http://localhost:8000/api/v1/poi/$$POI_ID" >/dev/null 2>&1 && \
			echo "  Deleted POI: $$POI_ID" || \
			echo "  POI $$POI_ID already removed"; \
		rm -f .benchmark_poi_id; \
	fi
	@$(MAKE) --no-print-directory clean-data
	@echo "$(GREEN)  Benchmark cleanup complete$(NC)"

.PHONY: consolidate-metrics
consolidate-metrics: ## Consolidate benchmark metrics from results/ into a CSV
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║                   Consolidating Metrics                        ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	$(BENCHMARK_VENV) && \
	python3 $(PERF_TOOLS_DIR)/parse_qmassa_metrics_to_json.py \
		-d $(CURDIR)/$(RESULTS_PATH) -k qmassa 2>/dev/null || true
	$(BENCHMARK_VENV) && \
	python3 $(PERF_TOOLS_DIR)/consolidate_multiple_run_of_metrics.py \
		--root_directory $(CURDIR)/$(RESULTS_PATH) \
		--output $(CURDIR)/$(RESULTS_PATH)/consolidated_metrics.csv
	@echo ""
	@echo "Results written to: $(RESULTS_PATH)/consolidated_metrics.csv"

.PHONY: plot-metrics
plot-metrics: ## Generate plots from benchmark metrics
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║                   Generating Metrics Plots                     ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	$(BENCHMARK_VENV) && \
	python3 $(PERF_TOOLS_DIR)/usage_graph_plot.py \
		--dir $(CURDIR)/$(RESULTS_PATH)
	@echo ""
	@echo "Plots saved to: $(RESULTS_PATH)/"

.PHONY: stream-density
stream-density: benchmark-stream-density ## Alias for benchmark-stream-density

.PHONY: generate-stream-density
generate-stream-density:
	@echo "Generating stream-density overrides only..."
	$(BENCHMARK_VENV) && \
	python3 $(SAD_STREAM_DENSITY) generate $(CURDIR)
	@echo "To start: make demo"

.PHONY: clean-stream-density
clean-stream-density:
	$(BENCHMARK_VENV) && \
	python3 $(SAD_STREAM_DENSITY) clean $(CURDIR)

.PHONY: down-stream-density
down-stream-density:
	$(BENCHMARK_VENV) && \
	python3 $(SAD_STREAM_DENSITY) down $(CURDIR)

# ---- POI Stream Density & Benchmarking ----

.PHONY: poi-benchmark
poi-benchmark: ## Run a single-scene POI benchmark (detection-to-alert latency)
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║       POI Benchmark — Single Scene                             ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo "  Device:         $(DEVICE)"
	@echo "  Target Latency: $(BENCHMARK_TARGET_LATENCY_MS)ms"
	@echo "  Duration:       $(BENCHMARK_DURATION)s (exits early on alert)"
	@echo "  Results Dir:    $(RESULTS_PATH)"
	@echo ""
	mkdir -p $(RESULTS_PATH)
	@# Fix ownership if results/ was created by a Docker container (root-owned).
	@if [ "$$(stat -c '%U' $(RESULTS_PATH) 2>/dev/null)" != "$$(id -un)" ]; then \
		docker run --rm -v "$(CURDIR)/$(RESULTS_PATH)":/r debian:bookworm-slim \
			chown -R "$(shell id -u):$(shell id -g)" /r && \
		echo "  Fixed ownership of $(RESULTS_PATH)"; \
	fi
	$(BENCHMARK_VENV) && \
	python3 $(POI_STREAM_DENSITY)  \
		--single_run \
		--scenes 1 \
		--benchmark_duration $(BENCHMARK_DURATION) \
		--poi_scripts_dir $(CURDIR)/benchmark \
		--app_dir $(CURDIR) \
		--target_latency_ms $(BENCHMARK_TARGET_LATENCY_MS) \
		--latency_metric $(BENCHMARK_LATENCY_METRIC) \
		--results_dir $(CURDIR)/$(RESULTS_PATH) \
		--resource_config $(BENCHMARK_RESOURCE_CONFIG)
.PHONY: poi-benchmark-stream-density
poi-benchmark-stream-density: ## Iteratively increase scenes until POI latency threshold
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║       POI Stream Density — Detection-to-Alert Scaling          ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo "  Device:           $(DEVICE)"
	@echo "  Target Latency:   $(BENCHMARK_TARGET_LATENCY_MS)ms"
	@echo "  Latency Metric:   $(BENCHMARK_LATENCY_METRIC)"
	@echo "  Scene Increment:  +$(BENCHMARK_SCENE_INCREMENT) per iteration"
	@echo "  Init Duration:    $(BENCHMARK_INIT_DURATION)s"
	@echo "  Stabilise:        $(BENCHMARK_STABILISE_DURATION)s"
	@echo "  Max Iterations:   $(BENCHMARK_MAX_ITERATIONS)"
	@echo "  Results Dir:      $(RESULTS_PATH)"
	@echo ""
	mkdir -p $(RESULTS_PATH)
	@# Fix ownership if results/ was created by a Docker container (root-owned).
	@if [ "$$(stat -c '%U' $(RESULTS_PATH) 2>/dev/null)" != "$$(id -un)" ]; then \
		docker run --rm -v "$(CURDIR)/$(RESULTS_PATH)":/r debian:bookworm-slim \
			chown -R "$(shell id -u):$(shell id -g)" /r && \
		echo "  Fixed ownership of $(RESULTS_PATH)"; \
	fi
	$(BENCHMARK_VENV) && \
	python3 $(POI_STREAM_DENSITY) \
		--poi_scripts_dir $(CURDIR)/benchmark \
		--app_dir $(CURDIR) \
		--target_latency_ms $(BENCHMARK_TARGET_LATENCY_MS) \
		--latency_metric $(BENCHMARK_LATENCY_METRIC) \
		--scene_increment $(BENCHMARK_SCENE_INCREMENT) \
		--init_duration $(BENCHMARK_INIT_DURATION) \
		--stabilise_duration $(BENCHMARK_STABILISE_DURATION) \
		--max_iterations $(BENCHMARK_MAX_ITERATIONS) \
		--max_alert_wait $(BENCHMARK_MAX_ALERT_WAIT) \
		--results_dir $(CURDIR)/$(RESULTS_PATH) \
		--resource_config $(BENCHMARK_RESOURCE_CONFIG)


.PHONY: poi-stream-density
poi-stream-density: poi-benchmark-stream-density ## Alias for poi-benchmark-stream-density

.PHONY: poi-generate-stream-density
poi-generate-stream-density:
	@echo "Generating POI stream-density overrides only..."
	$(BENCHMARK_VENV) && \
	python3 $(POI_STREAM_DENSITY) generate $(CURDIR)
	@echo "To start: make demo"

.PHONY: poi-clean-stream-density
poi-clean-stream-density:
	$(BENCHMARK_VENV) && \
	python3 $(POI_STREAM_DENSITY) clean $(CURDIR)

.PHONY: poi-down-stream-density
poi-down-stream-density:
	$(BENCHMARK_VENV) && \
	python3 $(POI_STREAM_DENSITY) down $(CURDIR)

# ---- Build ----

.PHONY: lp-build
lp-build:
	$(COMPOSE_SCENESCAPE) build

# ---- Cleanup ----

.PHONY: clean
clean:
	@# Flush Redis data while container is still running (best-effort)
	@docker exec poi-redis redis-cli FLUSHALL 2>/dev/null || true
	@# Remove FAISS indexes inside the container (best-effort)
	@docker exec poi-backend rm -f /data/faiss/poi.index /data/faiss/id_map.json 2>/dev/null || true
	@# Stop POI containers first (without -v to avoid shared network conflict)
	-$(COMPOSE_POI) down 2>/dev/null || true
	@# Now attempt full teardown with volumes (may warn about shared network — that's OK)
	-$(COMPOSE_SCENESCAPE) down -v 2>/dev/null || true
	-$(COMPOSE_LP) down -v 2>/dev/null || true
	@# Clean local data dirs as final fallback
	rm -rf data/faiss/* data/uploads/*
	@echo "$(GREEN)=== Clean complete. Redis flushed, FAISS removed, containers stopped. ===$(NC)"

.PHONY: clean-secrets
clean-secrets:
	rm -rf $(SCENESCAPE_DIR)/secrets/ca $(SCENESCAPE_DIR)/secrets/certs $(SCENESCAPE_DIR)/secrets/django
	rm -rf $(SCENESCAPE_DIR)/secrets/pgserver $(SCENESCAPE_DIR)/secrets/supass
	rm -f $(SCENESCAPE_DIR)/secrets/controller.auth $(SCENESCAPE_DIR)/secrets/browser.auth $(SCENESCAPE_DIR)/secrets/calibration.auth
	rm -f docker/.env

.PHONY: clean-images
clean-images:
	-docker rmi swlp-service:latest behavioral-analysis:latest gradio-ui:latest 2>/dev/null || true
	-docker rmi scenescape-manager:latest scenescape-controller:latest 2>/dev/null || true

.PHONY: clean-all
clean-all: clean clean-secrets clean-images
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
	@echo "    make export-scene    - Export scene config from running SceneScape"
	@echo "    make lp-down         - Stop all LP services"
	@echo "    make down-scenescape - Stop SceneScape only"
	@echo "    make lp-logs         - Follow LP logs"
	@echo "    make status          - Show LP service status"
	@echo ""
	@echo "  LP-Only (SceneScape must be running separately):"
	@echo "    make demo-lp         - Start LP services only"
	@echo "    make down-lp         - Stop LP services"
	@echo ""
	@echo "  Stream Density & Benchmarking (SAD):"
	@echo "    make benchmark                 - Single-scene benchmark (1 scene)"
	@echo "    make benchmark-stream-density  - Increase scenes until latency threshold"
	@echo "    make consolidate-metrics       - Consolidate metrics to CSV"
	@echo "    make plot-metrics              - Generate plots from benchmark metrics"
	@echo "    make stream-density            - Alias for benchmark-stream-density"
	@echo "    make generate-stream-density   - Generate overrides only (no start)"
	@echo "    make clean-stream-density      - Revert to single camera"
	@echo "    make down-stream-density       - Stop stream-density services"
	@echo ""
	@echo "  Stream Density & Benchmarking (POI):"
	@echo "    make poi-benchmark                 - Single-scene POI benchmark"
	@echo "    make poi-benchmark-stream-density  - Increase scenes until POI latency threshold"
	@echo "    make poi-stream-density            - Alias for poi-benchmark-stream-density"
	@echo "    make poi-generate-stream-density   - Generate POI overrides only (no start)"
	@echo "    make poi-clean-stream-density      - Revert POI to single camera"
	@echo "    make poi-down-stream-density       - Stop POI stream-density services"
	@echo ""
	@echo "  Build:"
	@echo "    make lp-build        - Build LP container images"
	@echo "    make build-benchmark - Build metrics-collector Docker image"
	@echo "    make fetch-benchmark - Pull metrics-collector image from registry"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean           - Stop and remove containers + volumes"
	@echo "    make clean-secrets   - Remove generated secrets and .env"
	@echo "    make clean-images    - Remove LP Docker images"
	@echo "    make clean-all       - Clean everything including videos"
	@echo ""
	@echo "  Prerequisites:"
	@echo "    - Docker with Compose v2"
	@echo "    - openssl, python3, curl"
	@echo "    - Internet access (to download sample videos)"
	@echo ""
	@echo "  POI System:"
	@echo "    make build           - Build POI backend + UI images"
	@echo "    make up              - Start POI stack (backend, UI, Redis, alert-service)"
	@echo "    make down            - Stop POI stack"
	@echo "    make logs            - Follow POI logs"
	@echo "    make restart         - Restart POI stack"
	@echo "    make init-env        - Create .env from .env.example"
	@echo "    make test            - Run backend tests"
	@echo "    make coverage        - Run tests with coverage"
	@echo "    make coverage-html   - Generate HTML coverage report"
	@echo "    make log-alert       - Tail alert-service logs"
	@echo "    make clean-redis     - Flush all Redis data (detections, POI metadata)"
	@echo "    make clean-faiss     - Remove FAISS index files"
	@echo "    make clean-data      - Flush Redis + remove FAISS (full data reset)"
	@echo "    make update-submodules - Update git submodules from remote"
	@echo "    make help            - Show this help message"

# ========================================================================
# POI Re-identification System
# ========================================================================

COMPOSE_POI := docker compose --env-file docker/.env -f docker-compose.yml
REGISTRY ?= true

BLUE := \033[1;34m
GREEN := \033[1;32m
YELLOW := \033[1;33m
NC := \033[0m

.PHONY: build
build:
	@echo "$(BLUE)=== Using local SceneScape images (skipping registry pull) ===$(NC)"
	-$(COMPOSE_SCENESCAPE) pull 2>/dev/null || echo "SceneScape images already available locally"
	@echo ""
	@echo "$(BLUE)=== Building POI images ===$(NC)"
	$(COMPOSE_POI) build poi-backend ui
	@echo ""
	@if [ "$(REGISTRY)" = "true" ]; then \
		echo "$(BLUE)=== Pulling benchmark metrics-collector image ===$(NC)"; \
		docker pull $(REGISTRY_BENCHMARK) 2>/dev/null && \
			echo "$(GREEN)  Benchmark image ready$(NC)" || \
			echo "$(YELLOW)  Benchmark image not available — run 'make build-benchmark' to build locally$(NC)"; \
	else \
		echo "$(BLUE)=== Skipping benchmark image pull (REGISTRY=false) ===$(NC)"; \
	fi
	@echo "$(GREEN)Build complete.$(NC)"

.PHONY: up
up:
	@if [ ! -f .env ]; then \
		echo "$(BLUE)=== Creating .env from .env.example ===$(NC)"; \
		cp .env.example .env; \
	fi
	@# Remove stale storewide-lp network missing Compose labels so Compose recreates it correctly
	@if docker network inspect storewide-lp >/dev/null 2>&1 && \
	   [ "$$(docker network inspect storewide-lp --format '{{index .Labels "com.docker.compose.network"}}')" != "storewide-lp" ]; then \
		$(COMPOSE_POI) down 2>/dev/null || true; \
		if docker network rm storewide-lp 2>/dev/null; then \
			echo "$(BLUE)=== Removed stale storewide-lp network (missing Compose labels) ===$(NC)"; \
		else \
			echo "$(YELLOW)=== WARNING: storewide-lp network has missing Compose labels but could not be removed (in use by other containers?) ===$(NC)"; \
		fi; \
	fi
	@# Remove orphan DB volumes from a previous deployment to prevent DB password mismatch on first start
	@if ! docker inspect storewide-lp-web-1 >/dev/null 2>&1; then \
		if docker volume inspect storewide-lp_vol-db >/dev/null 2>&1; then \
			echo "$(BLUE)=== Removing orphan DB volumes (fresh deployment with stale volumes) ===$(NC)"; \
			docker volume rm storewide-lp_vol-db storewide-lp_vol-migrations 2>/dev/null && \
				echo "  Removed storewide-lp_vol-db and storewide-lp_vol-migrations" || true; \
		fi; \
		echo "$(BLUE)=== Clearing stale FAISS index (fresh deployment) ===$(NC)"; \
		rm -f data/faiss/poi.index data/faiss/id_map.json; \
	fi
	@# HOST_IP is required for WebRTC ICE candidates (camera streams in the UI)
	@HOST_IP_VAL="$$(grep -E '^HOST_IP=.+' docker/.env 2>/dev/null | cut -d= -f2-)"; \
	if [ -z "$$HOST_IP_VAL" ]; then \
		echo ""; \
		echo "$(RED)ERROR: HOST_IP is not set in docker/.env$(NC)"; \
		echo ""; \
		echo "  HOST_IP is required for WebRTC camera streams to work in the UI."; \
		echo "  Set it before running make init:"; \
		echo ""; \
		echo "    export HOST_IP=\$$(ip route get 1 | awk '{print \$$7; exit}') && make init"; \
		echo ""; \
		echo "  Or for a specific IP:"; \
		echo ""; \
		echo "    export HOST_IP=192.168.1.100 && make init"; \
		echo ""; \
		exit 1; \
	fi
	@$(MAKE) -C $(SCENESCAPE_DIR) APP_DIR=$(CURDIR) RESOURCE_CONFIG=$(RESOURCE_CONFIG) init init-sample-data init-volumes
	@chmod 644 $(SCENESCAPE_DIR)/secrets/certs/*.key $(SCENESCAPE_DIR)/secrets/ca/*.key 2>/dev/null || true
	@echo "$(BLUE)=== Starting SceneScape services ===$(NC)"
	@# CONTROLLER_AUTH must be in the process environment (not just --env-file) for the
	@# environment:-source Docker secret to be injected at container creation time.
	CONTROLLER_AUTH="$$(grep '^CONTROLLER_AUTH=' docker/.env | cut -d= -f2-)" $(COMPOSE_SCENESCAPE) up -d
	@echo "  Waiting for SceneScape web health check..."; \
	attempts=0; \
	while [ $$attempts -lt 15 ]; do \
		status=$$(docker inspect storewide-lp-web-1 --format='{{.State.Health.Status}}' 2>/dev/null || echo "missing"); \
		if [ "$$status" = "healthy" ]; then break; fi; \
		if [ "$$status" = "unhealthy" ]; then \
			if docker logs storewide-lp-pgserver-1 2>&1 | tail -20 | grep -q 'password authentication failed'; then \
				echo "$(BLUE)=== SceneScape DB password mismatch — resetting stale DB volumes ===$(NC)"; \
				$(COMPOSE_SCENESCAPE) stop pgserver web scene scene-import 2>/dev/null || true; \
				docker volume rm storewide-lp_vol-db storewide-lp_vol-migrations 2>/dev/null && \
					echo "  Removed storewide-lp_vol-db and storewide-lp_vol-migrations" || true; \
				CONTROLLER_AUTH="$$(grep '^CONTROLLER_AUTH=' docker/.env | cut -d= -f2-)" $(COMPOSE_SCENESCAPE) up -d; \
			else \
				echo "$(YELLOW)=== SceneScape web unhealthy (not a DB password issue) — check logs with: docker logs storewide-lp-web-1 ===$(NC)"; \
			fi; \
			break; \
		fi; \
		attempts=$$((attempts + 1)); \
		sleep 10; \
	done
	@echo ""
	@echo "$(BLUE)=== Resolving SCENE_UID ===$(NC)"
	@bash scripts/resolve_scene_uid.sh
	@echo ""
	@echo "$(BLUE)=== Starting POI services ===$(NC)"
	$(COMPOSE_POI) up -d poi-backend ui poi-alert-service
	@echo ""
	@echo "System running:"
	@echo "  SceneScape UI:   https://localhost"
	@echo "  Backend API:     http://localhost:8000"
	@echo "  POI UI:          http://localhost:3000"
	@echo "  Alert Service:   http://localhost:8001"
	@echo ""
	@if [ -f docker/.env ]; then \
		SUPASS=$$(grep '^SUPASS=' docker/.env 2>/dev/null | cut -d= -f2-); \
		if [ -n "$$SUPASS" ]; then \
			echo "  SceneScape Login: admin / $$SUPASS"; \
			echo ""; \
		fi; \
	fi
	@echo "To stop: make down"

.PHONY: down
down:
	-$(COMPOSE_POI) --profile benchmark down
	-$(COMPOSE_SCENESCAPE) down

.PHONY: logs
logs:
	$(COMPOSE_POI) logs -f

.PHONY: restart
restart:
	$(COMPOSE_POI) restart

.PHONY: init-env
init-env:
	@if [ -f .env ]; then \
		read -p ".env already exists. Override? [y/N] " confirm; \
		if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
			cp .env.example .env; \
			echo ".env overwritten from .env.example"; \
		else \
			echo ".env left unchanged"; \
		fi; \
	else \
		cp .env.example .env; \
		echo ".env created from .env.example — edit it with your settings"; \
	fi

.PHONY: test
test:
	cd backend && python -m pytest tests/ -v

.PHONY: coverage
coverage:
	cd backend && python -m pytest tests/ --cov=backend --cov-report=term-missing -v

.PHONY: coverage-html
coverage-html:
	cd backend && python -m pytest tests/ --cov=backend --cov-report=html -v
	@echo "Coverage report: backend/htmlcov/index.html"

.PHONY: log-alert
log-alert:
	$(COMPOSE_POI) logs -f alert-service

.PHONY: clean-redis
clean-redis: ## Flush all Redis data (POI metadata, detections, caches)
	@echo "$(BLUE)=== Flushing Redis ===$(NC)"
	@docker exec poi-redis redis-cli FLUSHALL
	@echo "$(GREEN)Redis flushed.$(NC)"

.PHONY: clean-faiss
clean-faiss: ## Remove FAISS index files and reset detection counter
	@echo "$(BLUE)=== Cleaning FAISS indexes ===$(NC)"
	@if docker exec poi-backend rm -f /data/faiss/poi.index /data/faiss/id_map.json 2>/dev/null; then \
		echo "$(GREEN)FAISS indexes removed. Restart poi-backend to take effect.$(NC)"; \
	else \
		echo "  Container not running — removing POI Compose volumes"; \
		$(COMPOSE_POI) down -v >/dev/null 2>&1 || true; \
		echo "$(GREEN)FAISS volume removed. Will be recreated on next start.$(NC)"; \
	fi

.PHONY: clean-data
clean-data: clean-redis clean-faiss ## Flush Redis + remove FAISS (full data reset)
	@echo ""
	@echo "$(GREEN)All POI data cleared. Run 'make restart' to reinitialise.$(NC)"

.PHONY: update-submodules
update-submodules:
	@echo "$(BLUE)Cloning performance tool repositories$(NC)"
	cd .. && git submodule deinit -f .
	cd .. && git submodule update --init --remote --recursive
	@echo "$(GREEN)Submodules updated (if any present).$(NC)"

