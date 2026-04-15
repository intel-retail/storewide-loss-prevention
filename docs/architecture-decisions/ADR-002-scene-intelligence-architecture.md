# ADR-002: Scene Intelligence Service Architecture

**Status:** Accepted  
**Date:** 2026-04-14  
**Supersedes:** ADR-001 (Rule Engine as Library)

---

## Summary

This document defines the architecture for a **Scene Intelligence Service** — a generic, reusable service that processes SceneScape events, manages entity state, evaluates business rules, and dispatches actions including HTTP callbacks to application-specific services.

The **Storewide Loss Prevention Service** becomes a focused application that handles frame processing and behavioral analysis, receiving coordination signals from Scene Intelligence.

---

## Architecture Overview

```
                              +-----------------------------+
                              |         SceneScape          |
                              |  (Video Analytics Platform) |
                              +-------------+---------------+
                                            |
                                      MQTT  |
                      +---------------------+---------------------+
                      |                                           |
                      | event/region/#               data/camera/+|
                      | (entry/exit)                image/camera/+|
                      v                             (frames+bbox) v
     +-----------------------------+         +-----------------------------+
     |     Scene Intelligence      |         |  Storewide Loss Prevention  |
     |          Service            |         |          Service            |
     |                             |         |                             |
     |  * State Manager (Redis)    |         |  * Tracking Controller      |
     |  * Rule Engine              |  HTTP   |  * Frame Processor          |
     |  * Action Dispatcher        +-------->|  * SeaweedFS Client         |
     |                             | /start  |                             |
     +-------------+---------------+ /stop   +-------------+---------------+
                   |                                       |
                   |                          +------------+------------+
                   |                          |                         |
                   v                          v                         v
     +------------------------+   +---------------------+   +----------------------+
     |     Alert Service      |   |    Alert Service    |   | BehavioralAnalysis   |
     |                        |<--+       Client        |   |       Service        |
     |  * MQTT publish        |   |     (in LP svc)     |   |                      |
     |  * Webhook dispatch    |   |                     |   |  * Pose Detection    |
     |  * Deduplication       |   | POST: CONCEALMENT   |   |  * VLM Inference     |
     +------------------------+   +---------------------+   +----------------------+
                   ^
                   |  POST: CHECKOUT_BYPASS, LOITERING,
                   |        RESTRICTED_ZONE, REPEATED_VISITS
                   |
     +-------------+---------------+
     |     Scene Intelligence      |
     |    (Action Dispatcher)      |
     +-----------------------------+
```

**Connection Summary:**

| From | To | Protocol | Purpose |
|------|-----|----------|---------|
| SceneScape | Scene Intelligence | MQTT | Region entry/exit events |
| SceneScape | Loss Prevention | MQTT | Camera data + frames |
| Scene Intelligence | Loss Prevention | HTTP | /tracking/start, /tracking/stop |
| Scene Intelligence | Alert Service | HTTP | State-based alerts |
| Loss Prevention | BehavioralAnalysis | HTTP | Frame analysis requests |
| Loss Prevention | Alert Service | HTTP | CONCEALMENT alerts |

---


## MQTT Topic Ownership

Each service subscribes to **distinct** SceneScape topics — no overlap.

| Topic Pattern | Subscriber | Data |
|---------------|------------|------|
| `scenescape/event/region/{scene}/{region}/count` | Scene Intelligence | Entry/exit events with dwell time |
| `scenescape/data/camera/{camera_id}` | Storewide Loss Prevention | Bounding boxes per detected person |
| `scenescape/image/camera/{camera_id}` | Storewide Loss Prevention | Camera frame images |

This separation ensures:
- No duplicate message processing
- Clear ownership of each data stream
- Independent scaling per workload type

---

## Service Responsibilities

### 1. Scene Intelligence Service (Generic)

**Purpose:** Process entity tracking events, manage entity state, evaluate business rules, and dispatch actions.

**Responsibilities:**

| Responsibility | Description |
|----------------|-------------|
| **MQTT Subscription** | Subscribe to SceneScape region events |
| **Event Parsing** | Extract entity_id, zone_id, zone_type, dwell from messages |
| **State Management** | Maintain entity state in Redis (zones, flags, history) |
| **Rule Evaluation** | Match events against configured rules |
| **Action Dispatch** | Execute actions: alerts, HTTP callbacks, state updates |
| **API Exposure** | REST API for state queries and external state updates |

**Does NOT do:**
- Frame processing
- ML inference
- Application-specific business logic beyond rule evaluation

**Configuration Files:**

```
scene-intelligence/
├── config/
│   ├── subscriptions.yaml    # MQTT topics to subscribe
│   ├── entity_schema.yaml    # State structure & transitions
│   ├── rules.yaml            # Business rules
│   └── handlers.yaml         # HTTP callback endpoints
```

---

### 2. Storewide Loss Prevention Service (Application-Specific)

**Purpose:** Process camera frames for persons of interest, coordinate behavioral analysis, and publish analysis-based alerts.

**Responsibilities:**

| Responsibility | Description |
|----------------|-------------|
| **MQTT Subscription** | Subscribe to SceneScape camera data and images |
| **Tracking Control** | Maintain set of entity IDs to actively track |
| **Frame Correlation** | Match frames with detection bounding boxes |
| **Frame Processing** | Crop person images from camera frames |
| **Frame Storage** | Store cropped frames to SeaweedFS (rolling buffer) |
| **Analysis Orchestration** | Call BehavioralAnalysis service with frames |
| **Alert Publishing** | Publish CONCEALMENT alerts when VLM confirms suspicious activity |

**Does NOT do:**
- Region event processing (Scene Intelligence does this)
- Entity state management (Scene Intelligence owns Redis)
- State-based rule evaluation (Scene Intelligence does this)

**Tracking Set:**

The service maintains a lightweight in-memory set of entity IDs:

```python
tracking_set: Set[str] = set()

# Populated by HTTP callbacks from Scene Intelligence
# POST /api/v1/tracking/start  → tracking_set.add(entity_id)
# POST /api/v1/tracking/stop   → tracking_set.discard(entity_id)
```

---

### 3. BehavioralAnalysis Service (Generic)

**Purpose:** Perform ML-based pose detection and VLM inference.

**Responsibilities:**

| Responsibility | Description |
|----------------|-------------|
| **Pose Detection** | Extract keypoints from person frames using OpenVINO |
| **Pose History** | Maintain sliding window of poses per entity |
| **Pattern Matching** | Detect configured motion patterns (e.g., shelf-to-waist) |
| **VLM Escalation** | Send frames + prompt to VLM for confirmation |

**Stateless per request:** Does not persist state beyond in-memory pose history with TTL.

---

### 4. Alert Service (Generic)

**Purpose:** Publish alerts to configured destinations.

**Responsibilities:**

| Responsibility | Description |
|----------------|-------------|
| **Alert Publishing** | Send alerts to MQTT, webhooks, logging |
| **Deduplication** | Prevent duplicate alerts within time window |
| **Routing** | Route alerts by type to different destinations |

---

## Data Flow: Complete Example

### Scenario: Person enters high-value zone, exhibits suspicious behavior

```
TIME     SOURCE              ACTION
--------------------------------------------------------------------------------

         +---------------------------------------------------------------+
         |  PHASE 1: Person Enters High-Value Zone                       |
         +---------------------------------------------------------------+

T+0      SceneScape          Publishes region entry event
                             Topic: scenescape/event/region/store1/electronics/count
                             Payload: {
                               "entered": [{"id": "person_42"}],
                               "exited": [],
                               "rate": 1,
                               "timestamp": "2026-04-14T10:00:00Z"
                             }

T+1      Scene Intelligence  Receives MQTT message
                             Parses: entity_id=person_42, zone=electronics, event=entry

T+2      Scene Intelligence  Updates Redis state:
                             HSET entity:person_42 current_zones.electronics 
                               '{"entered_at":"2026-04-14T10:00:00Z","zone_type":"HIGH_VALUE"}'
                             HINCRBY entity:person_42 zone_visit_counts.electronics 1
                             HSET entity:person_42 flags.visited_high_value "true"

T+3      Scene Intelligence  Evaluates rules for event_type="zone_entry"
                             
                             Rule "high_value_entry" matches:
                               trigger: zone_entry where zone_type == HIGH_VALUE
                               actions: 
                                 - http_callback to LP /tracking/start

T+4      Scene Intelligence  Dispatches action:
                             POST http://loss-prevention:8080/api/v1/tracking/start
                             {
                               "entity_id": "person_42",
                               "zone_id": "electronics",
                               "zone_type": "HIGH_VALUE",
                               "timestamp": "2026-04-14T10:00:00Z"
                             }

T+5      Loss Prevention     Receives POST /tracking/start
                             Adds "person_42" to tracking_set
                             Returns 200 OK

         +---------------------------------------------------------------+
         |  PHASE 2: Frame Processing Begins                             |
         +---------------------------------------------------------------+

T+100    SceneScape          Publishes camera detection
                             Topic: scenescape/data/camera/cam_03
                             Payload: {
                               "objects": [
                                 {"id": "person_42", "bbox": [100,200,180,450], "confidence": 0.95}
                               ],
                               "timestamp": "2026-04-14T10:00:01Z"
                             }

T+101    SceneScape          Publishes camera frame
                             Topic: scenescape/image/camera/cam_03
                             Payload: {
                               "image": "<base64>",
                               "timestamp": "2026-04-14T10:00:01Z"
                             }

T+102    Loss Prevention     Receives camera data + image
                             Correlates by timestamp
                             
                             For each detection:
                               Is "person_42" in tracking_set? -> YES
                               
                             Crops frame using bbox [100,200,180,450]
                             Stores to SeaweedFS: person_42/1713088801000.jpg

T+103    Loss Prevention     Calls BehavioralAnalysis:
                             POST http://behavioral:8080/api/v1/analyze
                             {
                               "entity_id": "person_42",
                               "frame": "<base64 cropped>",
                               "pattern_id": "shelf_to_waist"
                             }

T+104    BehavioralAnalysis  Runs pose detection
                             Extracts keypoints
                             Appends to pose_history[person_42]
                             
                             History has 3 frames (need 10)
                             Returns: {"status": "accumulating", "frames_collected": 3}

T+105    Loss Prevention     Receives response
                             status=accumulating -> continue

         ... frames T+200 through T+900 processed similarly ...

         +---------------------------------------------------------------+
         |  PHASE 3: Pattern Detected, VLM Escalation                    |
         +---------------------------------------------------------------+

T+1000   Loss Prevention     Calls BehavioralAnalysis with frame #10

T+1001   BehavioralAnalysis  Has 10 frames in history
                             Evaluates shelf-to-waist pattern
                             
                             Pattern MATCHED:
                               Frames 1-5: wrist above hip level
                               Frames 6-10: wrist near waist
                             
                             Retrieves frames from SeaweedFS
                             Sends to VLM with prompt
                             
                             VLM Response:
                             {
                               "concealment_occurred": true,
                               "confidence": 0.85,
                               "observation": "Person reached to shelf and moved object to pocket"
                             }
                             
                             Returns: {
                               "status": "vlm_complete",
                               "pattern_matched": true,
                               "vlm_result": {...}
                             }

T+1002   Loss Prevention     Receives VLM result
                             confidence >= 0.80 -> publish alert directly
                             
                             POST http://alert-service:8080/api/v1/alerts
                             {
                               "alert_type": "CONCEALMENT",
                               "severity": "WARNING",
                               "entity_id": "person_42",
                               "zone_id": "electronics",
                               "metadata": {
                                 "vlm_confidence": 0.85,
                                 "observation": "Person reached to shelf..."
                               }
                             }

T+1003   Alert Service       Publishes alert to configured destinations
                             - MQTT: alerts/concealment
                             - Webhook: POST to external system

         +---------------------------------------------------------------+
         |  PHASE 4: Person Exits Zone                                   |
         +---------------------------------------------------------------+

T+5000   SceneScape          Publishes region exit event
                             Topic: scenescape/event/region/store1/electronics/count
                             Payload: {
                               "entered": [],
                               "exited": [{"id": "person_42", "dwell": 50}]
                             }

T+5001   Scene Intelligence  Updates state:
                             Remove electronics from current_zones
                             Add electronics to visited_zones

T+5002   Scene Intelligence  Rule "high_value_exit" matches:
                             actions:
                               - http_callback to LP /tracking/stop

T+5003   Scene Intelligence  POST http://loss-prevention:8080/api/v1/tracking/stop
                             { "entity_id": "person_42" }

T+5004   Loss Prevention     Removes "person_42" from tracking_set
                             Stops processing frames for this entity
```

---

## Configuration Files

### Scene Intelligence: rules.yaml

```yaml
rules:
  # TRACKING CONTROL RULES
  
  - id: high_value_zone_entry
    name: Start Tracking on High-Value Entry
    trigger:
      event_type: zone_entry
      condition: "event.zone_type == 'HIGH_VALUE'"
    conditions: []
    actions:
      - type: http_callback
        params:
          method: POST
          url: "${LOSS_PREVENTION_URL}/api/v1/tracking/start"
          body:
            entity_id: "$event.entity_id"
            zone_id: "$event.zone_id"
            zone_type: "$event.zone_type"

  - id: high_value_zone_exit
    name: Stop Tracking on High-Value Exit
    trigger:
      event_type: zone_exit
      condition: "event.zone_type == 'HIGH_VALUE'"
    conditions: []
    actions:
      - type: http_callback
        params:
          method: POST
          url: "${LOSS_PREVENTION_URL}/api/v1/tracking/stop"
          body:
            entity_id: "$event.entity_id"

  # ALERT RULES

  - id: checkout_bypass
    name: Checkout Bypass Detection
    trigger:
      event_type: zone_exit
      condition: "event.zone_type == 'EXIT'"
    conditions:
      - field: "state.flags.visited_high_value"
        op: eq
        value: true
      - field: "state.flags.visited_checkout"
        op: eq
        value: false
    actions:
      - type: alert
        params:
          alert_type: CHECKOUT_BYPASS
          severity: WARNING

  - id: loitering_on_exit
    name: Loitering Detection (On Exit)
    trigger:
      event_type: zone_exit
      condition: "event.zone_type == 'HIGH_VALUE'"
    conditions:
      - field: "event.dwell"
        op: gt
        value: 120
    actions:
      - type: alert
        params:
          alert_type: LOITERING
          severity: WARNING

  - id: restricted_zone_violation
    name: Restricted Zone Entry
    trigger:
      event_type: zone_entry
      condition: "event.zone_type == 'RESTRICTED'"
    conditions: []
    actions:
      - type: alert
        params:
          alert_type: RESTRICTED_ZONE
          severity: CRITICAL

  - id: repeated_visits
    name: Repeated High-Value Zone Visits
    trigger:
      event_type: zone_entry
      condition: "event.zone_type == 'HIGH_VALUE'"
    conditions:
      - field: "state.zone_visit_counts[$event.zone_id]"
        op: gte
        value: 3
    actions:
      - type: alert
        params:
          alert_type: REPEATED_VISITS
          severity: INFO

  # NOTE: CONCEALMENT alerts are NOT handled by Scene Intelligence.
  # The Storewide Loss Prevention service publishes CONCEALMENT alerts
  # directly to Alert Service after VLM confirmation.
```

---

### Scene Intelligence: entity_schema.yaml

```yaml
entity_types:
  person:
    schema:
      current_zones:
        type: map
        key: zone_id
        value:
          entered_at: datetime
          zone_type: string
      
      last_camera: string
      last_seen: datetime
      first_seen: datetime
      
      visited_zones:
        type: list
        item: string
      
      zone_visit_counts:
        type: map
        key: zone_id
        value: integer
      
      flags:
        visited_checkout: boolean
        visited_high_value: boolean
        visited_restricted: boolean
      
      loiter_alerted_zones:
        type: list
        item: string

    transitions:
      - event_type: zone_entry
        updates:
          - field: "current_zones[$event.zone_id]"
            value:
              entered_at: "$event.timestamp"
              zone_type: "$event.zone_type"
          - field: "zone_visit_counts[$event.zone_id]"
            op: increment
          - field: "flags.visited_checkout"
            value: true
            condition: "$event.zone_type == 'CHECKOUT'"
          - field: "flags.visited_high_value"
            value: true
            condition: "$event.zone_type == 'HIGH_VALUE'"

      - event_type: zone_exit
        updates:
          - field: "current_zones[$event.zone_id]"
            op: delete
          - field: "visited_zones"
            op: append
            value: "$event.zone_id"

    ttl_seconds: 3600
```

---

### Storewide Loss Prevention: API Endpoints

```yaml
endpoints:
  - path: /api/v1/tracking/start
    method: POST
    description: Add entity to active tracking set
    request:
      entity_id: string (required)
      zone_id: string
      zone_type: string
    response:
      status: "tracking" | "already_tracking"
    
  - path: /api/v1/tracking/stop
    method: POST
    description: Remove entity from tracking set
    request:
      entity_id: string (required)
    response:
      status: "stopped" | "not_tracking"
    
  - path: /api/v1/tracking/status
    method: GET
    description: Get current tracking status
    response:
      tracking_count: integer
      entities: list[string]

  - path: /health
    method: GET
  - path: /metrics
    method: GET
```

---

## Service Communication Summary

| FROM | TO | PROTOCOL | PURPOSE |
|------|-----|----------|---------|
| SceneScape | Scene Intel | MQTT | Region events |
| SceneScape | Loss Prevention | MQTT | Frames/detections |
| Scene Intel | Loss Prevention | HTTP POST | Start/stop tracking |
| Scene Intel | Alert Service | HTTP POST | State-based alerts |
| Scene Intel | Redis | Redis protocol | State read/write |
| Loss Prevention | Alert Service | HTTP POST | CONCEALMENT alert |
| Loss Prevention | BehavioralAnalysis | HTTP POST | Analyze frames |
| Loss Prevention | SeaweedFS | S3 API | Store frames |
| BehavioralAnalysis | SeaweedFS | S3 API | Retrieve frames |
| BehavioralAnalysis | OVMS | HTTP POST | VLM inference |
| Alert Service | External | MQTT/Webhook | Alert delivery |

---

## Deployment Overview

```yaml
# docker-compose.yaml

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  mosquitto:
    image: eclipse-mosquitto:2
    ports:
      - "1883:1883"
  
  seaweedfs:
    image: chrislusf/seaweedfs
    command: "server -s3"
    ports:
      - "8333:8333"

  scene-intelligence:
    build: ./scene-intelligence
    environment:
      - MQTT_BROKER=mqtt://mosquitto:1883
      - REDIS_URL=redis://redis:6379
      - ALERT_SERVICE_URL=http://alert-service:8080
      - LOSS_PREVENTION_URL=http://loss-prevention:8080
    depends_on:
      - redis
      - mosquitto

  alert-service:
    build: ./alert-service
    environment:
      - MQTT_BROKER=mqtt://mosquitto:1883
      - WEBHOOK_URL=${ALERT_WEBHOOK_URL}
    depends_on:
      - mosquitto

  behavioral-analysis:
    build: ./behavioral-analysis
    environment:
      - SEAWEEDFS_URL=http://seaweedfs:8333
      - VLM_ENDPOINT=http://ovms:8080/v1/chat/completions
    volumes:
      - ./models:/models:ro

  loss-prevention:
    build: ./storewide-loss-prevention
    environment:
      - MQTT_BROKER=mqtt://mosquitto:1883
      - SEAWEEDFS_URL=http://seaweedfs:8333
      - ALERT_SERVICE_URL=http://alert-service:8080
      - BEHAVIORAL_ANALYSIS_URL=http://behavioral-analysis:8080
    depends_on:
      - mosquitto
      - seaweedfs
      - alert-service
      - behavioral-analysis
```

---

## Summary

| Service | Type | MQTT Topics | State | Key Function |
|---------|------|-------------|-------|--------------|
| **Scene Intelligence** | Generic | `event/region/#` | Redis (owns) | Event->State->Rules->Actions |
| **Storewide Loss Prevention** | App-specific | `data/camera/+`, `image/camera/+` | In-memory set | Frames->Store->Analyze->Alert |
| **BehavioralAnalysis** | Generic | None | In-memory pose history | Pose+VLM inference |
| **Alert Service** | Generic | None | Dedup cache | Alert routing |

---

## Alert Ownership

| Alert Type | Triggered By | Published By |
|------------|--------------|-------------|
| CHECKOUT_BYPASS | Region exit + state flags | **Scene Intelligence** |
| LOITERING | Region exit + dwell time | **Scene Intelligence** |
| RESTRICTED_ZONE | Region entry | **Scene Intelligence** |
| REPEATED_VISITS | Region entry + visit count | **Scene Intelligence** |
| CONCEALMENT | VLM analysis result | **Storewide Loss Prevention** |

Scene Intelligence publishes alerts for **state-based rules** (tracking data it owns).  
Loss Prevention publishes alerts for **analysis-based rules** (ML results only it knows).
