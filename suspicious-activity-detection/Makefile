# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Storewide Loss Prevention - Makefile
#
# Full-stack deployment using pre-built SceneScape Docker images.
# No SceneScape source checkout required.

COMPOSE_SCENESCAPE := docker compose -f docker/docker-compose-scenescape.yaml
COMPOSE_LP := docker compose -f docker/docker-compose.yaml
COMPOSE_ALL := docker compose -f docker/docker-compose-scenescape.yaml -f docker/docker-compose.yaml
SAMPLE_DATA_DIR := scenescape/sample_data

# ---- Full Stack (SceneScape + LP) ----

.PHONY: demo
demo: init init-sample-data download-models init-volumes
	@echo ""
	@echo "=== Building LP images ==="
	$(COMPOSE_ALL) build storewide-loss-prevention gradio-ui
	@echo ""
	@echo "=== Starting all services ==="
	$(COMPOSE_ALL) up -d
	@rm -f application.log
	@$(COMPOSE_ALL) logs -f storewide-loss-prevention > application.log 2>&1 &
	@echo ""
	@echo "Services running:"
	@echo "  SceneScape UI:   https://localhost"
	@echo "  Login:           admin / $$(grep SUPASS docker/.env | cut -d= -f2-)"
	@echo "  Gradio UI:       http://localhost:7860"
	@echo "  LP API:          http://localhost:8082"
	@echo "  LP logs:         application.log"
	@echo ""
	@echo "To stop: make down"

# SceneScape only (without LP services)
.PHONY: run-scenescape
run-scenescape: init init-sample-data download-models init-volumes
	@echo ""
	@echo "=== Starting SceneScape ==="
	$(COMPOSE_SCENESCAPE) up -d
	@echo ""
	@echo "=== SceneScape is running ==="
	@echo "  UI:    https://localhost"
	@echo "  Login: admin / $$(grep SUPASS docker/.env | cut -d= -f2-)"
	@echo ""
	@echo "To stop: make down"

.PHONY: init
init:
	@chmod +x scripts/init.sh
	@./scripts/init.sh

.PHONY: init-sample-data
init-sample-data:
	@echo "Initializing sample data volume..."
	@docker volume create storewide-lp_vol-sample-data 2>/dev/null || true
	@docker run --rm -v storewide-lp_vol-sample-data:/dest alpine:3.23 chown $$(id -u):$$(id -g) /dest
	@if [ -d "$(SAMPLE_DATA_DIR)" ] && [ -n "$$(ls -A $(SAMPLE_DATA_DIR)/*.ts $(SAMPLE_DATA_DIR)/*.mp4 2>/dev/null)" ]; then \
		echo "Copying sample videos to volume..."; \
		docker run --rm \
			-v $(CURDIR)/$(SAMPLE_DATA_DIR):/source:ro \
			-v storewide-lp_vol-sample-data:/dest \
			--user $$(id -u):$$(id -g) \
			alpine:3.23 \
			sh -c "cp -v /source/*.ts /source/*.mp4 /dest/ 2>/dev/null || true"; \
		echo "Sample data volume initialized."; \
	else \
		echo "WARNING: No video files (.ts/.mp4) in $(SAMPLE_DATA_DIR)/"; \
		echo "Run ./scripts/init.sh to download sample videos."; \
	fi

.PHONY: download-models
download-models:
	@chmod +x scripts/download_models.sh
	@./scripts/download_models.sh

.PHONY: init-volumes
init-volumes:
	@echo "Initializing Docker volumes..."
	@docker volume create storewide-lp_vol-media 2>/dev/null || true
	@docker run --rm -v storewide-lp_vol-media:/data alpine:3.23 sh -c "chown 1000:1000 /data && touch /data/placeholder"
	@echo "Volumes initialized."

.PHONY: down
down:
	-$(COMPOSE_ALL) down
	@rm -f application.log

.PHONY: down-scenescape
down-scenescape:
	-$(COMPOSE_SCENESCAPE) down

.PHONY: logs
logs:
	$(COMPOSE_ALL) logs -f

.PHONY: status
status:
	$(COMPOSE_ALL) ps

# ---- LP-Only (SceneScape must be running separately) ----

.PHONY: demo-lp
demo-lp:
	@echo "Building LP images..."
	$(COMPOSE_ALL) build storewide-loss-prevention gradio-ui
	@echo "Starting LP services only (SceneScape must be running)..."
	$(COMPOSE_LP) up -d

.PHONY: down-lp
down-lp:
	$(COMPOSE_LP) down

# ---- Build ----

.PHONY: build
build:
	$(COMPOSE_SCENESCAPE) build

# ---- Cleanup ----

.PHONY: clean
clean:
	-$(COMPOSE_SCENESCAPE) down -v
	-$(COMPOSE_LP) down -v 2>/dev/null || true

.PHONY: clean-secrets
clean-secrets:
	rm -rf scenescape/secrets/ca scenescape/secrets/certs scenescape/secrets/django
	rm -rf scenescape/secrets/pgserver scenescape/secrets/supass
	rm -f scenescape/secrets/controller.auth scenescape/secrets/browser.auth scenescape/secrets/calibration.auth
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
	@echo "  Cleanup:"
	@echo "    make clean           - Stop and remove containers + volumes"
	@echo "    make clean-secrets   - Remove generated secrets and .env"
	@echo "    make clean-all       - Clean everything including videos"
	@echo ""
	@echo "  Prerequisites:"
	@echo "    - Docker with Compose v2"
	@echo "    - openssl, python3, curl"
	@echo "    - Internet access (to download sample videos)"
