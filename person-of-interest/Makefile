# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Storewide Loss Prevention - Makefile
#
# Full-stack deployment using pre-built SceneScape Docker images.
# No SceneScape source checkout required.
# SceneScape-specific targets are in scenescape/Makefile.

SCENESCAPE_DIR := ../scenescape
COMPOSE_SCENESCAPE := docker compose --project-directory . --env-file docker/.env -f $(SCENESCAPE_DIR)/docker-compose.yaml
COMPOSE_LP := docker compose --project-directory . --env-file docker/.env -f docker/docker-compose.yaml
COMPOSE_ALL := docker compose --project-directory . --env-file docker/.env -f $(SCENESCAPE_DIR)/docker-compose.yaml -f docker/docker-compose.yaml

# ---- Initialization ----

.PHONY: init
init:
	@$(MAKE) -C $(SCENESCAPE_DIR) APP_DIR=$(CURDIR) init

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
	@echo "To stop: make lp-down"

.PHONY: lp-down
lp-down:
	-$(COMPOSE_ALL) down
	@rm -f application.log

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
	$(COMPOSE_ALL) build storewide-loss-prevention gradio-ui
	@echo "Starting LP services only (SceneScape must be running)..."
	$(COMPOSE_LP) up -d

.PHONY: down-lp
down-lp:
	$(COMPOSE_LP) down

# ---- Build ----

.PHONY: lp-build
lp-build:
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
	@echo "  Build:"
	@echo "    make lp-build        - Build LP container images"
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
	@echo "  POI System:"
	@echo "    make build           - Build POI backend + UI images (REGISTRY=false for local)"
	@echo "    make up              - Start POI stack (backend, UI, Redis, alert-service)"
	@echo "    make down            - Stop POI stack"
	@echo "    make logs            - Follow POI logs"
	@echo "    make restart         - Restart POI stack"
	@echo "    make init-env        - Create .env from .env.example"
	@echo "    make test            - Run backend tests"
	@echo "    make coverage        - Run tests with coverage"
	@echo "    make coverage-html   - Generate HTML coverage report"
	@echo "    make log-alert       - Tail alert-service logs"
	@echo "    make benchmark       - Run end-to-end latency benchmark"
	@echo "    make benchmark-stream-density - Run stream density benchmark"
	@echo "    make update-submodules - Update git submodules from remote"
	@echo "    make help            - Show this help message"

# ========================================================================
# POI Re-identification System
# ========================================================================

COMPOSE_POI := docker compose -f docker-compose.yml
REGISTRY ?= false

BLUE := \033[1;34m
GREEN := \033[1;32m
NC := \033[0m

.PHONY: build
build:
	@echo "$(BLUE)=== Using local SceneScape images (skipping registry pull) ===$(NC)"
	-$(COMPOSE_SCENESCAPE) pull 2>/dev/null || echo "SceneScape images already available locally"
	@echo ""
	@echo "$(BLUE)=== Building POI images ===$(NC)"
ifeq ($(REGISTRY),false)
	$(COMPOSE_POI) build poi-backend ui
else
	$(COMPOSE_POI) build poi-backend ui
	docker tag person-of-interest-poi-backend $(REGISTRY)/poi-backend:latest
	docker tag person-of-interest-ui $(REGISTRY)/poi-ui:latest
endif
	@echo "$(GREEN)Build complete.$(NC)"

.PHONY: up
up:
	@echo "$(BLUE)=== Starting SceneScape services ===$(NC)"
	-$(COMPOSE_SCENESCAPE) up -d --no-recreate
	@echo ""
	@echo "$(BLUE)=== Starting POI services ===$(NC)"
	$(COMPOSE_POI) up -d poi-backend ui poi-alert-service
	@echo ""
	@echo "System running:"
	@echo "  SceneScape UI:   https://localhost:8443"
	@echo "  Backend API:     http://localhost:8000"
	@echo "  POI UI:          http://localhost:3000"
	@echo "  Alert Service:   http://localhost:8001"
	@echo ""
	@echo "To stop: make down"

.PHONY: down
down:
	-$(COMPOSE_POI) down
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

.PHONY: benchmark
benchmark:
	cd backend && python -m backend.benchmark.benchmark

.PHONY: benchmark-stream-density
benchmark-stream-density:
	cd backend && python -m backend.benchmark.stream_density

.PHONY: update-submodules
update-submodules:
	@echo "$(BLUE)Cloning/updating performance tool repositories...$(NC)"
	cd .. && git submodule deinit -f .
	cd .. && git submodule update --init --recursive --remote
	@echo "$(GREEN)Submodules updated successfully.$(NC)"

