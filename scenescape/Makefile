# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# SceneScape - Makefile
#
# Manages SceneScape services: init, volumes, models, start, stop.

SCENESCAPE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ENV_FILE := $(APP_DIR)/docker/.env
CAMERAS_COMPOSE := $(APP_DIR)/docker/docker-compose.cameras.yaml
COMPOSE_SCENESCAPE := docker compose --env-file $(ENV_FILE) -f $(SCENESCAPE_DIR)/docker-compose.yaml $(if $(wildcard $(CAMERAS_COMPOSE)),-f $(CAMERAS_COMPOSE))
SAMPLE_DATA_DIR := $(SCENESCAPE_DIR)/sample_data

# ---- Init ----

.PHONY: init
init:
	@if [ -z "$(APP_DIR)" ]; then \
		echo "ERROR: APP_DIR is not set."; \
		echo "Usage: make run APP_DIR=<path-to-app-dir>"; \
		echo "  <app-dir> must contain configs/ and docker/ subdirectories."; \
		echo "  Example: make run APP_DIR=../suspicious-activity-detection"; \
		exit 1; \
	fi
	@chmod +x $(SCENESCAPE_DIR)/scripts/init.sh
	@$(SCENESCAPE_DIR)/scripts/init.sh $(APP_DIR)

# ---- Sample Data Volume ----

.PHONY: init-sample-data
init-sample-data:
	@echo "Initializing sample data volume..."
	@docker volume create storewide-lp_vol-sample-data 2>/dev/null || true
	@docker run --rm -v storewide-lp_vol-sample-data:/dest alpine:3.23 chown $$(id -u):$$(id -g) /dest
	@if [ -d "$(SAMPLE_DATA_DIR)" ] && [ -n "$$(ls -A $(SAMPLE_DATA_DIR)/*.ts $(SAMPLE_DATA_DIR)/*.mp4 2>/dev/null)" ]; then \
		echo "Copying sample videos to volume..."; \
		docker run --rm \
			-v $(SAMPLE_DATA_DIR):/source:ro \
			-v storewide-lp_vol-sample-data:/dest \
			--user $$(id -u):$$(id -g) \
			alpine:3.23 \
			sh -c "cp -v /source/*.ts /source/*.mp4 /dest/ 2>/dev/null || true"; \
		echo "Sample data volume initialized."; \
	else \
		echo "WARNING: No video files (.ts/.mp4) in $(SAMPLE_DATA_DIR)/"; \
		echo "Run ./scenescape/scripts/init.sh to download sample videos."; \
	fi

# ---- Download Models ----

.PHONY: download-models
download-models:
	@echo "All models are downloaded by the app-level model_download.sh"
	@echo "Run: make download-models from your app directory (e.g. suspicious-activity-detection)"

# ---- Volumes ----

.PHONY: init-volumes
init-volumes:
	@echo "Initializing Docker volumes..."
	@docker volume create storewide-lp_vol-media 2>/dev/null || true
	@docker run --rm -v storewide-lp_vol-media:/data alpine:3.23 sh -c "chown 1000:1000 /data && touch /data/placeholder"
	@docker volume create storewide-lp_vol-models 2>/dev/null || true
	@echo "Volumes initialized."

# ---- Load local SceneScape images from reid-drop ----
# Use after receiving a reid-drop.tar containing updated controller/manager images.
# The loaded images are tagged scenescape-controller:latest / scenescape-manager:latest.
# To use them, set in docker/.env:
#   SCENESCAPE_CONTROLLER_IMAGE=scenescape-controller
#   SCENESCAPE_MANAGER_IMAGE=scenescape-manager
#   SCENESCAPE_VERSION=latest

REID_DROP_DIR ?= $(SCENESCAPE_DIR)/../reid-drop

.PHONY: load-images
load-images:
	@echo "=== Loading SceneScape images from $(REID_DROP_DIR) ==="
	@test -f "$(REID_DROP_DIR)/scenescape-controller.tar" || { echo "ERROR: $(REID_DROP_DIR)/scenescape-controller.tar not found"; exit 1; }
	@test -f "$(REID_DROP_DIR)/scenescape-manager.tar"    || { echo "ERROR: $(REID_DROP_DIR)/scenescape-manager.tar not found"; exit 1; }
	docker load < "$(REID_DROP_DIR)/scenescape-controller.tar"
	docker load < "$(REID_DROP_DIR)/scenescape-manager.tar"
	@echo "=== Images loaded ==="
	@echo "  scenescape-controller:latest"
	@echo "  scenescape-manager:latest"
	@echo ""
	@echo "To use these images, add to docker/.env:"
	@echo "  SCENESCAPE_CONTROLLER_IMAGE=scenescape-controller"
	@echo "  SCENESCAPE_MANAGER_IMAGE=scenescape-manager"
	@echo "  SCENESCAPE_VERSION=latest"

# ---- Start SceneScape ----

.PHONY: run
run: init init-sample-data download-models init-volumes
	@echo ""
	@echo "=== Starting SceneScape ==="
	$(COMPOSE_SCENESCAPE) up -d
	@echo ""
	@echo "=== SceneScape is running ==="
	@echo "  UI:    https://localhost"
	@echo "  Login: admin / $$(grep SUPASS $(ENV_FILE) | cut -d= -f2-)"
	@echo ""
	@echo "To stop: make -C $(SCENESCAPE_DIR) down"

# ---- Stop SceneScape ----

.PHONY: down
down:
	-$(COMPOSE_SCENESCAPE) down

# ---- Logs ----

.PHONY: logs
logs:
	$(COMPOSE_SCENESCAPE) logs -f

# ---- Status ----

.PHONY: status
status:
	$(COMPOSE_SCENESCAPE) ps

# ---- Export Scene ----

.PHONY: export-scene
export-scene:
	@chmod +x $(SCENESCAPE_DIR)/scripts/export-scene.sh
	@$(SCENESCAPE_DIR)/scripts/export-scene.sh $(APP_DIR)

# ---- Import Scene ----

.PHONY: import-scene
import-scene:
	@chmod +x $(SCENESCAPE_DIR)/scripts/import-scene.sh
	@$(SCENESCAPE_DIR)/scripts/import-scene.sh $(APP_DIR)
