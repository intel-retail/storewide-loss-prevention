# ADR-001: Rule Engine as Library vs Service

**Status:** Accepted  
**Date:** 2026-04-13  
**Context:** Loss Prevention System Architecture  

---

## Summary

This document explains why the Rule Engine is implemented as an **in-process library** while BehavioralAnalysis is implemented as a **separate service**. The decision is based on operational complexity, latency requirements, and the nature of the work each component performs.

---

## Decision

| Component | Implementation | Rationale |
|-----------|----------------|-----------|
| Rule Engine | **Library** (in-process) | Simple condition checks, low latency required |
| BehavioralAnalysis | **Service** (separate container) | Complex ML inference, independent scaling |

---

## Context: What Each Component Does

### Rule Engine

The Rule Engine evaluates business rules against entity state to determine if an alert should be fired.

**Example Rule Evaluation:**
```
INPUT:
  - Event: person exited "exit_zone"
  - State: visited_high_value=true, visited_checkout=false

LOGIC:
  if event.zone_type == "EXIT" 
     and state.visited_high_value == true 
     and state.visited_checkout == false:
    return Alert(CHECKOUT_BYPASS)

OUTPUT:
  - Alert type or nothing
```

**Characteristics:**
- Pure condition matching (boolean logic)
- Operates on in-memory state (no I/O)
- Executes in microseconds
- No external dependencies during evaluation
- Deterministic output

### BehavioralAnalysis Service

The BehavioralAnalysis Service performs ML-based pose detection and VLM inference to analyze human behavior.

**Example Analysis Flow:**
```
INPUT:
  - Cropped person frame (15-20 KB JPEG)
  - Entity ID for history lookup

PROCESSING:
  1. Load pose estimation model (OpenVINO)
  2. Run inference on frame → extract 17 keypoints
  3. Append to sliding window history (10 frames)
  4. Match pose sequence against patterns
  5. If pattern detected:
     a. Retrieve 10 historical frames from SeaweedFS
     b. Collect 10 more frames (async, ~5 seconds)
     c. Send 20 frames to VLM with prompt
     d. Parse VLM response

OUTPUT:
  - Pattern match result with confidence
  - VLM analysis (if escalated)
```

**Characteristics:**
- ML model inference (compute-intensive)
- External I/O (SeaweedFS reads, VLM API calls)
- Executes in 50-500ms per call
- GPU/NPU acceleration beneficial
- Maintains internal state (pose history)

---

## Analysis: Why Rule Engine Should NOT Be a Service

### 1. Latency Overhead Outweighs Benefit

**Rule evaluation timing:**
- In-process function call: **<1ms**
- Service call (same host): **5-15ms** (serialization + HTTP + deserialization)
- Service call (network): **10-50ms**

For every region entry/exit event, adding 10ms of latency provides no functional benefit. The Loss Prevention Service processes hundreds of events per second — this overhead accumulates.

```
Events per second: 100
Service call overhead: 10ms
Wasted time per second: 1000ms (1 full second of blocking)
```

### 2. No Independent Scaling Requirement

Rule evaluation is CPU-bound and lightweight. It scales linearly with the host service. There is no scenario where rule evaluation becomes a bottleneck while the rest of the Loss Prevention Service has spare capacity.

**Contrast with BehavioralAnalysis:**
- Pose detection benefits from GPU/NPU
- VLM inference is the slowest operation in the pipeline
- Multiple cameras may need concurrent analysis
- Scaling analysis independently from event processing is valuable

### 3. No Shared State Across Applications

A separate Rule Engine service would be justified if:
- Multiple applications evaluate the same rules
- Rules are managed centrally with a UI
- Rules change frequently without deployment

**Current reality:**
- Single application (Loss Prevention)
- Rules are static configuration
- Changes deploy with the application

### 4. Operational Complexity Without Benefit

Separate service adds:
- Container orchestration
- Health checks and restart policies
- Network failure handling
- Service discovery
- Distributed tracing for debugging
- Version compatibility management

For a function that does `if a and not b: return X`, this is unjustified overhead.

---

## Analysis: Why BehavioralAnalysis SHOULD Be a Service

### 1. Compute-Intensive Operations

| Operation | Duration | Resource |
|-----------|----------|----------|
| Pose estimation inference | 20-50ms | GPU/NPU |
| Pattern matching | <1ms | CPU |
| Frame retrieval (10 frames) | 10-30ms | Network I/O |
| VLM inference | 300-500ms | GPU/API |

A single analysis request can take 400-600ms. Running this in-process would block event handling.

### 2. Independent Scaling

```
Scenario: 10 cameras, peak hour

Events per second (region entry/exit): ~50
Frames requiring analysis: ~20/second (persons in high-value zones)
Analysis duration: ~100ms average (pose only, no VLM)

Event processing capacity needed: 1 instance
Analysis capacity needed: 2-3 instances (or GPU acceleration)
```

Separating analysis allows:
- Horizontal scaling of inference workload
- GPU/NPU resource allocation independent of main service
- Queue-based load leveling during traffic spikes

### 3. Hardware Specialization

BehavioralAnalysis can be deployed on nodes with:
- Intel GPUs (Arc) for accelerated inference
- NPUs for efficient pose estimation
- High memory for model loading

The Loss Prevention Service (event processing) can run on standard CPU nodes.

### 4. Failure Isolation

If VLM inference fails or times out:
- BehavioralAnalysis handles retry/fallback
- Loss Prevention Service continues processing events
- Alerts for non-VLM rules still fire

If rule evaluation fails (in-process), the entire service restarts — but this is acceptable because rule evaluation cannot fail (pure logic on valid state).

### 5. Reusability

BehavioralAnalysis is intentionally generic:
- Accepts any entity ID and frame
- Patterns defined in configuration
- VLM prompts are configurable

This enables reuse in:
- Queue frustration detection (different patterns)
- Workplace safety monitoring (different patterns)
- Customer service gesture recognition

---

## Comparison Table

| Criterion | Rule Engine | BehavioralAnalysis |
|-----------|-------------|---------------------|
| **Operation type** | Boolean logic | ML inference |
| **Execution time** | <1ms | 50-500ms |
| **External I/O** | None | SeaweedFS, VLM API |
| **CPU/GPU needs** | Minimal CPU | GPU/NPU beneficial |
| **Scaling pattern** | Scales with host | Independent scaling |
| **Failure impact** | Fatal (logic error) | Graceful degradation |
| **Reuse potential** | Low (app-specific rules) | High (generic analysis) |
| **State** | Reads session state | Maintains pose history |
| **Latency tolerance** | None (sync path) | Acceptable (async ok) |

---

## Implementation: Rule Engine as Library

### Directory Structure

```
suspicious-activity-detection/
└── src/
    └── services/
        └── rule_engine.py      # In-process rule evaluation
```

### Interface

```python
from dataclasses import dataclass
from typing import Optional
from models.session import PersonSession
from models.events import RegionEvent

@dataclass
class Action:
    type: str           # "alert", "escalate", "log"
    params: dict        # Action-specific parameters

class RuleEngine:
    """In-process rule evaluation engine.
    
    Evaluates business rules against session state and events.
    Returns actions to be executed by the caller.
    """
    
    def __init__(self, rules_config_path: str):
        """Load rules from YAML configuration."""
        self.rules = self._load_rules(rules_config_path)
    
    def evaluate(
        self, 
        event_type: str, 
        event: RegionEvent, 
        session: PersonSession
    ) -> list[Action]:
        """Evaluate all rules for the given event type.
        
        Args:
            event_type: Type of event ("zone_entry", "zone_exit", "loiter_check")
            event: The event data
            session: Current session state for the entity
            
        Returns:
            List of actions to execute (may be empty)
        """
        actions = []
        for rule in self.rules:
            if self._matches(rule, event_type, event, session):
                actions.extend(rule.actions)
        return actions
```

### Rule Configuration

```yaml
# configs/rules.yaml

rules:
  - id: checkout_bypass
    name: Checkout Bypass Detection
    trigger:
      event_type: zone_exit
      zone_type: EXIT
    conditions:
      - field: session.visited_high_value
        op: eq
        value: true
      - field: session.visited_checkout
        op: eq
        value: false
    actions:
      - type: alert
        params:
          alert_type: CHECKOUT_BYPASS
          severity: WARNING

  - id: loitering
    name: Loitering in High-Value Zone
    trigger:
      event_type: zone_exit
      zone_type: HIGH_VALUE
    conditions:
      - field: event.dwell
        op: gt
        value: ${LOITER_THRESHOLD:120}
    actions:
      - type: alert
        params:
          alert_type: LOITERING
          severity: WARNING

  - id: restricted_zone
    name: Restricted Zone Violation
    trigger:
      event_type: zone_entry
      zone_type: RESTRICTED
    conditions: []  # No additional conditions
    actions:
      - type: alert
        params:
          alert_type: RESTRICTED_ZONE
          severity: CRITICAL

  - id: repeated_visits
    name: Repeated High-Value Zone Visits
    trigger:
      event_type: zone_entry
      zone_type: HIGH_VALUE
    conditions:
      - field: session.zone_visit_counts[event.zone_id]
        op: gte
        value: ${REPEAT_VISIT_THRESHOLD:3}
    actions:
      - type: alert
        params:
          alert_type: REPEATED_VISITS
          severity: INFO
```

### Usage in Loss Prevention Service

```python
# handlers/region_handler.py

class RegionEventHandler:
    def __init__(
        self, 
        session_manager: SessionManager,
        rule_engine: RuleEngine,        # Library, not service client
        alert_client: AlertServiceClient,
        behavioral_client: BehavioralAnalysisClient  # Service client
    ):
        self.sessions = session_manager
        self.rules = rule_engine
        self.alerts = alert_client
        self.behavioral = behavioral_client
    
    async def on_region_exit(self, event: RegionExitEvent):
        session = self.sessions.get_or_create(event.object_id)
        
        # Update session state
        session.remove_current_zone(event.zone_id)
        session.zones_visited.append(event.zone_id)
        
        # Evaluate rules - direct function call, no network
        actions = self.rules.evaluate("zone_exit", event, session)
        
        # Execute actions
        await self._execute_actions(actions, event, session)
    
    async def _execute_actions(self, actions: list[Action], event, session):
        for action in actions:
            match action.type:
                case "alert":
                    await self.alerts.publish(
                        alert_type=action.params["alert_type"],
                        severity=action.params["severity"],
                        object_id=session.object_id,
                        metadata={"zone_id": event.zone_id}
                    )
                case "escalate":
                    # This DOES call a service - justified by complexity
                    await self.behavioral.escalate_to_vlm(
                        entity_id=session.object_id,
                        reason=action.params["reason"]
                    )
```

---

## Implementation: BehavioralAnalysis as Service

### Service Interface

```
POST /api/v1/analyze
Content-Type: application/json

{
  "entity_id": "person_abc123",
  "frame": "<base64 encoded JPEG>",
  "pattern_id": "shelf_to_waist"
}

Response:
{
  "status": "accumulating" | "no_match" | "pattern_detected" | "vlm_complete",
  "pattern_id": "shelf_to_waist",
  "confidence": 0.85,
  "vlm_result": {
    "concealment_occurred": true,
    "confidence": 0.82,
    "observation": "Person reached toward shelf and moved hand to waist area"
  }
}
```

### Deployment

```yaml
# docker-compose.yaml

services:
  loss-prevention:
    build: ./src
    depends_on:
      - behavioral-analysis
      - alert-service
    environment:
      - RULES_CONFIG=/app/configs/rules.yaml
      - BEHAVIORAL_SERVICE_URL=http://behavioral-analysis:8080
      - ALERT_SERVICE_URL=http://alert-service:8080

  behavioral-analysis:
    build: ./behavioral-analysis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - POSE_MODEL_PATH=/models/human-pose-estimation-0001
      - VLM_ENDPOINT=http://ovms:8080/v1/chat/completions
    volumes:
      - ./models:/models:ro

  alert-service:
    build: ./alert-service
    environment:
      - MQTT_BROKER=mqtt://mosquitto:1883
      - WEBHOOK_URL=${ALERT_WEBHOOK_URL}
```

---

## Summary

### Rule Engine: Keep It Simple

The Rule Engine's job is to answer: "Given this event and this state, should we fire an alert?"

This is a **pure function** with:
- No I/O
- No ML inference
- No async operations
- Microsecond execution time

Wrapping this in an HTTP API adds latency and complexity with no benefit.

### BehavioralAnalysis: Embrace Complexity

The BehavioralAnalysis Service handles:
- ML model inference (GPU-accelerated)
- State management (pose history)
- External I/O (frame storage, VLM API)
- Long-running operations (500ms+)

This is inherently complex, benefits from independent scaling, and justifies service boundaries.

---

## Decision Outcome

- **Rule Engine**: Implemented as `rule_engine.py` library within Loss Prevention Service
- **Rules**: Defined in `configs/rules.yaml`, loaded at startup
- **BehavioralAnalysis**: Remains a separate service with REST API
- **Alert Service**: Remains a separate service (handles external webhooks, deduplication)

This design minimizes latency for the hot path (event → rule → alert) while maintaining flexibility for compute-intensive analysis.
