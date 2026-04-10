# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Storewide Loss Prevention - Makefile
#
# Full-stack deployment using pre-built SceneScape Docker images.
# No SceneScape source checkout required.

COMPOSE_FULL := docker compose -f docker/docker-compose.full.yaml
COMPOSE_LP := docker compose -f docker/docker-compose.yaml
SAMPLE_DATA_DIR := scenescape/sample_data

# ---- Full Stack (SceneScape + LP) ----

.PHONY: demo
demo: init
	@echo "Starting full stack (SceneScape + Loss Prevention)..."
	@$(MAKE) init-sample-data
	$(COMPOSE_FULL) up -d
	@echo ""
	@echo "Services running:"
	@echo "  SceneScape UI:   https://localhost"
	@echo "  Gradio UI:       http://localhost:7860"
	@echo "  LP API:          http://localhost:8082"
	@echo ""
	@echo "To stop: make down"

.PHONY: init
init:
	@chmod +x init.sh
	@./init.sh

.PHONY: init-sample-data
init-sample-data:
	@echo "Initializing sample data volume..."
	@docker volume create scenescape_vol-sample-data 2>/dev/null || true
	@docker run --rm -v scenescape_vol-sample-data:/dest alpine:3.23 chown $$(id -u):$$(id -g) /dest
	@if [ -d "$(SAMPLE_DATA_DIR)" ] && [ -n "$$(ls -A $(SAMPLE_DATA_DIR)/*.ts 2>/dev/null)" ]; then \
		echo "Copying sample videos to volume..."; \
		docker run --rm \
			-v $(CURDIR)/$(SAMPLE_DATA_DIR):/source:ro \
			-v scenescape_vol-sample-data:/dest \
			--user $$(id -u):$$(id -g) \
			alpine:3.23 \
			sh -c "cp -v /source/*.ts /dest/ 2>/dev/null || true"; \
		echo "Sample data volume initialized."; \
	else \
		echo "WARNING: No .ts video files in $(SAMPLE_DATA_DIR)/"; \
		echo "Run ./init.sh to download sample videos."; \
	fi

.PHONY: down
down:
	-$(COMPOSE_FULL) down
	-$(COMPOSE_LP) down 2>/dev/null || true

.PHONY: logs
logs:
	$(COMPOSE_FULL) logs -f

.PHONY: status
status:
	$(COMPOSE_FULL) ps

# ---- LP-Only (SceneScape must be running separately) ----

.PHONY: demo-lp
demo-lp:
	@echo "Starting LP services only (SceneScape must be running)..."
	$(COMPOSE_LP) up -d

.PHONY: down-lp
down-lp:
	$(COMPOSE_LP) down

# ---- Build ----

.PHONY: build
build:
	$(COMPOSE_FULL) build

# ---- Cleanup ----

.PHONY: clean
clean:
	-$(COMPOSE_FULL) down -v
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
	@echo "    make demo            - Initialize and start everything"
	@echo "    make down            - Stop all services"
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
