# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# SceneScape - Makefile
#
# Manages SceneScape services: init, volumes, models, start, stop.

SCENESCAPE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ENV_FILE := $(APP_DIR)/docker/.env
COMPOSE_SCENESCAPE := docker compose --env-file $(ENV_FILE) -f $(SCENESCAPE_DIR)/docker-compose.yaml
SAMPLE_DATA_DIR := $(SCENESCAPE_DIR)/sample_data

# ---- Init ----

.PHONY: init
init:
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
	@chmod +x $(SCENESCAPE_DIR)/scripts/download_models.sh
	@$(SCENESCAPE_DIR)/scripts/download_models.sh

# ---- Volumes ----

.PHONY: init-volumes
init-volumes:
	@echo "Initializing Docker volumes..."
	@docker volume create storewide-lp_vol-media 2>/dev/null || true
	@docker run --rm -v storewide-lp_vol-media:/data alpine:3.23 sh -c "chown 1000:1000 /data && touch /data/placeholder"
	@echo "Volumes initialized."

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
