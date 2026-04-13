# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Storewide Loss Prevention - Makefile
#
# Full-stack deployment using pre-built SceneScape Docker images.
# No SceneScape source checkout required.
# SceneScape-specific targets are in scenescape/Makefile.

SCENESCAPE_DIR := ../scenescape
COMPOSE_SCENESCAPE := docker compose --env-file docker/.env -f $(SCENESCAPE_DIR)/docker-compose.yaml
COMPOSE_LP := docker compose --env-file docker/.env -f docker/docker-compose.yaml
COMPOSE_ALL := docker compose --env-file docker/.env -f $(SCENESCAPE_DIR)/docker-compose.yaml -f docker/docker-compose.yaml

# ---- Full Stack (SceneScape + LP) ----

.PHONY: demo
demo:
	@$(MAKE) -C $(SCENESCAPE_DIR) APP_DIR=$(CURDIR) init init-sample-data download-models init-volumes
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

.PHONY: down
down:
	-$(COMPOSE_ALL) down
	@rm -f application.log

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
	@echo "  Cleanup:"
	@echo "    make clean           - Stop and remove containers + volumes"
	@echo "    make clean-secrets   - Remove generated secrets and .env"
	@echo "    make clean-all       - Clean everything including videos"
	@echo ""
	@echo "  Prerequisites:"
	@echo "    - Docker with Compose v2"
	@echo "    - openssl, python3, curl"
	@echo "    - Internet access (to download sample videos)"
