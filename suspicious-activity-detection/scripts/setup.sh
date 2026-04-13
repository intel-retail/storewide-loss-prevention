#!/bin/bash

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Color codes for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

export APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
export HOST_IP=$(ip route get 1 2>/dev/null | awk '{print $7}')
if [ -z "$HOST_IP" ]; then
    export HOST_IP="localhost"
fi

# Project name
PROJECT_NAME="storewide-loss-prevention"

# SceneScape certs source
SCENESCAPE_CERTS="${APP_DIR}/../../scenescape/secrets/certs"

# Setting command usage and invalid arguments handling before the actual setup starts
if [ "$#" -eq 0 ] || ([ "$#" -eq 1 ] && [ "$1" = "--help" ]); then
    # If no valid argument is passed, print usage information
    echo -e "-----------------------------------------------------------------"
    echo -e "${YELLOW}USAGE: ${GREEN}source setup.sh ${BLUE}[--setenv | --setup | --run | --restart | --stop | --clean | --help]"
    echo -e "${YELLOW}"
    echo -e "  --setenv:                 Set environment variables without building image or starting any containers"
    echo -e "  --setup:                  Build and run Store-wide Loss Prevention"
    echo -e "  --run:                    Start without building image (if already built)"
    echo -e "  --restart:                Restart Store-wide Loss Prevention"
    echo -e "  --stop:                   Stop Store-wide Loss Prevention"
    echo -e "  --clean:                  Clean up containers, volumes, and logs"
    echo -e "  --help:                   Show this help message${NC}"
    echo -e "-----------------------------------------------------------------"
    echo -e "${CYAN}NOTE: This assumes SceneScape is already running in the same network.${NC}"
    echo -e "-----------------------------------------------------------------"
    return 0

elif [ "$#" -gt 2 ]; then
    echo -e "${RED}ERROR: Too many arguments provided.${NC}"
    echo -e "${YELLOW}Use --help for usage information${NC}"
    return 1

elif [ "$1" != "--help" ] && [ "$1" != "--setenv" ] && [ "$1" != "--run" ] && [ "$1" != "--setup" ] && [ "$1" != "--restart" ] && [ "$1" != "--stop" ] && [ "$1" != "--clean" ]; then
    # Default case for unrecognized option
    echo -e "${RED}Unknown option: $1 ${NC}"
    echo -e "${YELLOW}Use --help for usage information${NC}"
    return 1

elif [ "$1" = "--stop" ] || [ "$1" = "--clean" ]; then
    echo -e "${YELLOW}Stopping Store-wide Loss Prevention ${RED}${PROJECT_NAME} ${YELLOW}... ${NC}"
    
    # Stop log collector process
    if [ -f "${APP_DIR}/.log_collector.pid" ]; then
        kill $(cat "${APP_DIR}/.log_collector.pid") 2>/dev/null
        rm -f "${APP_DIR}/.log_collector.pid"
    fi

    docker compose -f "${APP_DIR}/docker/docker-compose.yaml" -p ${PROJECT_NAME} down 2> /dev/null

    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to stop Store-wide Loss Prevention. ${NC}"
        return 1
    fi
    echo -e "${GREEN}All containers for Store-wide Loss Prevention stopped and removed! ${NC}"

    # Remove application log file
    if [ -f "${APP_DIR}/application.log" ]; then
        rm -f "${APP_DIR}/application.log"
        echo -e "${GREEN}Application log file removed.${NC}"
    fi

    if [ "$1" = "--clean" ]; then
        echo -e "${YELLOW}Removing volumes for Store-wide Loss Prevention ... ${NC}"
        docker volume ls | grep $PROJECT_NAME | awk '{ print $2 }' | xargs docker volume rm 2>/dev/null || true
        echo -e "${GREEN}Cleanup completed successfully. ${NC}"
    fi

    return 0
fi

# Export environment variables required by application (HOST_IP already set above)
export LOG_LEVEL=${LOG_LEVEL:-INFO}
export USER_GROUP_ID=$(id -g)
export VIDEO_GROUP_ID=$(getent group video | awk -F: '{printf "%s\n", $3}' 2>/dev/null || echo "44")
export RENDER_GROUP_ID=$(getent group render | awk -F: '{printf "%s\n", $3}' 2>/dev/null || echo "109")

# Store Configuration (can be overridden by environment variables)
export STORE_NAME=${STORE_NAME:-retail_store_1}
export STORE_ID=${STORE_ID:-store_001}

# MQTT Configuration
export MQTT_HOST=${MQTT_HOST:-broker.scenescape.intel.com}
export MQTT_PORT=${MQTT_PORT:-1883}

# SeaweedFS Configuration
export SEAWEEDFS_S3_PORT=${SEAWEEDFS_S3_PORT:-8333}
export SEAWEEDFS_MASTER_PORT=${SEAWEEDFS_MASTER_PORT:-9333}
export SEAWEEDFS_VOLUME_PORT=${SEAWEEDFS_VOLUME_PORT:-8080}

# External Services
export BEHAVIORAL_ANALYSIS_URL=${BEHAVIORAL_ANALYSIS_URL:-http://behavioral-analysis-service:8090}
export RULE_SERVICE_URL=${RULE_SERVICE_URL:-http://rule-service:8091}

# LP Service Port Configuration
export LP_SERVICE_PORT=${LP_SERVICE_PORT:-8082}

# Get and print the ports of all running services
print_service_endpoints() {
    echo -e
    echo -e "${MAGENTA}======================================================="
    echo -e "SERVICE ENDPOINTS"
    echo -e "=======================================================${NC}"
    
    for CONTAINER_NAME in $(docker ps --format '{{.Names}}' | grep $PROJECT_NAME);
    do
        case "$CONTAINER_NAME" in
            *storewide-loss-prevention*)
                BACKEND_SERVICE_NAME="Store-wide Loss Prevention API"
                PORT=$(docker port "$CONTAINER_NAME" 8082 | cut -d: -f2)
                echo -e "${CYAN}$BACKEND_SERVICE_NAME -> http://$HOST_IP:$PORT/docs${NC}"
                echo -e "${CYAN}  Health  -> http://$HOST_IP:$PORT/health${NC}"
                echo -e "${CYAN}  Alerts  -> http://$HOST_IP:$PORT/api/v1/lp/alerts${NC}"
                echo -e "${CYAN}  Sessions -> http://$HOST_IP:$PORT/api/v1/lp/sessions${NC}"
                ;;
            *seaweedfs*)
                SERVICE_NAME="SeaweedFS S3"
                PORT=$(docker port "$CONTAINER_NAME" 8333 | cut -d: -f2)
                echo -e "${GREEN}$SERVICE_NAME -> http://$HOST_IP:$PORT${NC}"
                ;;
        esac
    done
    echo -e "${MAGENTA}=======================================================${NC}"
    echo -e
}

# Print environment summary
print_env_summary() {
    echo -e "${MAGENTA}======================================================="
    echo -e "ENVIRONMENT CONFIGURATION"
    echo -e "=======================================================${NC}"
    echo -e "${CYAN}Store Name:${NC} $STORE_NAME"
    echo -e "${CYAN}Store ID:${NC} $STORE_ID"
    echo -e "${CYAN}MQTT Broker:${NC} $MQTT_HOST:$MQTT_PORT"
    echo -e "${CYAN}SeaweedFS S3:${NC} localhost:$SEAWEEDFS_S3_PORT"
    echo -e "${CYAN}Behavioral Analysis:${NC} $BEHAVIORAL_ANALYSIS_URL"
    echo -e "${CYAN}Rule Service:${NC} $RULE_SERVICE_URL"
    echo -e "${CYAN}Service Port:${NC} $LP_SERVICE_PORT"
    echo -e "${MAGENTA}=======================================================${NC}"
    echo -e
}

# Exit after setting environment variables if --setenv is passed
if [ "$1" = "--setenv" ]; then
    print_env_summary
    echo -e "${GREEN}Environment variables set successfully${NC}"
    return 0
fi

# Build and run services based on the argument
case "$1" in
    "--setup")
        echo -e "${BLUE}Setting up Store-wide Loss Prevention...${NC}"
        print_env_summary

        # Copy SceneScape TLS cert if not already present
        if [ ! -f "${APP_DIR}/secrets/certs/scenescape-ca.pem" ]; then
            echo -e "${YELLOW}Copying SceneScape TLS certificate...${NC}"
            mkdir -p "${APP_DIR}/secrets/certs"
            if [ -f "${SCENESCAPE_CERTS}/scenescape-ca.pem" ]; then
                cp "${SCENESCAPE_CERTS}/scenescape-ca.pem" "${APP_DIR}/secrets/certs/"
                echo -e "${GREEN}TLS certificate copied.${NC}"
            else
                echo -e "${RED}WARNING: SceneScape CA cert not found at ${SCENESCAPE_CERTS}/scenescape-ca.pem${NC}"
                echo -e "${YELLOW}Copy it manually: mkdir -p ${APP_DIR}/secrets/certs && cp <path-to-ca.pem> ${APP_DIR}/secrets/certs/${NC}"
            fi
        fi

        docker compose -f "${APP_DIR}/docker/docker-compose.yaml" -p ${PROJECT_NAME} up --build -d

        # Start collecting application logs
        echo -e "${YELLOW}Starting application log collection...${NC}"
        docker compose -f "${APP_DIR}/docker/docker-compose.yaml" -p ${PROJECT_NAME} logs -f storewide-loss-prevention > "${APP_DIR}/application.log" 2>&1 &
        echo $! > "${APP_DIR}/.log_collector.pid"
        echo -e "${GREEN}Application logs -> ${APP_DIR}/application.log${NC}"
        ;;
    "--run")
        echo -e "${BLUE}Starting Store-wide Loss Prevention...${NC}"
        print_env_summary
        docker compose -f "${APP_DIR}/docker/docker-compose.yaml" -p ${PROJECT_NAME} up -d

        # Start collecting application logs
        echo -e "${YELLOW}Starting application log collection...${NC}"
        docker compose -f "${APP_DIR}/docker/docker-compose.yaml" -p ${PROJECT_NAME} logs -f storewide-loss-prevention > "${APP_DIR}/application.log" 2>&1 &
        echo $! > "${APP_DIR}/.log_collector.pid"
        echo -e "${GREEN}Application logs -> ${APP_DIR}/application.log${NC}"
        ;;
    "--restart")
        echo -e "${BLUE}Restarting Store-wide Loss Prevention...${NC}"
        docker compose -f "${APP_DIR}/docker/docker-compose.yaml" -p ${PROJECT_NAME} restart
        ;;
esac

# Check if the command was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to execute docker compose command${NC}"
    return 1
fi

# Wait for services to be ready
echo -e "${YELLOW}Waiting for services to start...${NC}"
sleep 5

# Print service endpoints
print_service_endpoints

echo -e "${GREEN}Store-wide Loss Prevention is ready!${NC}"
echo -e "${CYAN}Connected to SceneScape MQTT broker at: $MQTT_HOST:$MQTT_PORT${NC}"