# Configurable Session Flags & Service Registry

## Overview

Two architectural changes decouple business logic from hardcoded data models:

1. **Configurable Session Flags** — boolean flags on `PersonSession` (e.g. "has this person visited checkout?") are now defined in `rules.yaml` instead of hardcoded as Python fields.
2. **Service Registry** — escalation services (e.g. behavioral analysis) are registered by name at startup and selected from rules via config, rather than hardcoded `if/elif` chains.

Both changes eliminate the need for code modifications when adding new zone types, tracking flags, or escalation services.

---

## What Changed

### Files Modified

| File | Change |
|------|--------|
| `configs/rules.yaml` | Added `session_flags` and `services` sections |
| `swlp-service/models/session.py` | Replaced `visited_checkout`, `visited_exit`, `visited_high_value`, `concealment_suspected` booleans with generic `flags: Dict[str, bool]`. Added backward-compatible read-only properties. |
| `swlp-service/services/config.py` | Loads `session_flags` and `services` from `rules.yaml`. Added `get_session_flag_defs()` and `get_service_defs()` accessors. |
| `swlp-service/services/rule_adapter.py` | Replaced hardcoded state transitions with config-driven flag loop. Replaced hardcoded BA orchestrator reference with service registry. Updated `_build_context()` to merge `session.flags`. Added `EscalationService` protocol and `register_service()`. |
| `swlp-service/main.py` | Changed from passing `ba_orchestrator=` constructor arg to calling `rule_adapter.register_service("behavioral_analysis", ba_orchestrator)`. |
| `swlp-service/tests/test_rule_engine.py` | Updated `FakeConfig` to provide `get_session_flag_defs()` / `get_service_defs()`. Updated test sessions to use `flags={}` dict instead of keyword booleans. |

---

## 1. Configurable Session Flags

### Before (hardcoded)

```python
# session.py — fixed fields
class PersonSession:
    visited_checkout: bool = False
    visited_high_value: bool = False
    visited_exit: bool = False
    concealment_suspected: bool = False

# rule_adapter.py — hardcoded if/elif per zone type
if event.zone_type == ZoneType.HIGH_VALUE:
    session.visited_high_value = True
elif event.zone_type == ZoneType.CHECKOUT:
    session.visited_checkout = True
elif event.zone_type == ZoneType.EXIT:
    session.visited_exit = True
```

Adding a new zone type (e.g. `PHARMACY`) required changes to **3 files**: `session.py` (new field), `events.py` (enum), `rule_adapter.py` (new elif).

### After (config-driven)

```yaml
# rules.yaml
session_flags:
  visited_high_value:
    trigger: zone_visited
    zone_type: HIGH_VALUE
  visited_checkout:
    trigger: zone_visited
    zone_type: CHECKOUT
  visited_exit:
    trigger: zone_visited
    zone_type: EXIT
  concealment_suspected:
    trigger: external
    source: behavioral_analysis
    field: status
    match_value: suspicious
```

```python
# session.py — generic dict
class PersonSession:
    flags: Dict[str, bool] = field(default_factory=dict)

# rule_adapter.py — config-driven loop
flag_names = self._zone_visited_flags.get(zone_type_str, [])
for flag_name in flag_names:
    if not session.flags.get(flag_name):
        session.flags[flag_name] = True
```

### How to Add a New Flag (NO code change)

**Example: Track whether a person has visited a pharmacy zone.**

1. Add the zone type to `zone_config.json`:
   ```json
   { "zones": { "pharmacy-aisle": "PHARMACY" } }
   ```

2. Add the flag to `rules.yaml`:
   ```yaml
   session_flags:
     visited_pharmacy:
       trigger: zone_visited
       zone_type: PHARMACY
   ```

3. Reference it in a rule:
   ```yaml
   rules:
     - id: pharmacy_alert
       trigger:
         event_type: zone_entry
         zone_type: EXIT
       conditions:
         - field: visited_pharmacy
           op: eq
           value: true
         - field: visited_checkout
           op: eq
           value: false
       actions:
         - type: alert
           params:
             alert_type: CHECKOUT_BYPASS
             severity: WARNING
   ```

**Zero Python code changes required.**

### Flag Trigger Types

| Trigger | When flag is set | Config fields |
|---------|-----------------|---------------|
| `zone_visited` | Person enters any zone of the specified `zone_type` | `zone_type` (string) |
| `external` | External service result matches criteria | `source` (service name), `field` (result field to check), `match_value` (expected value) |

### Backward Compatibility

`PersonSession` retains read-only properties for the well-known flags:

```python
@property
def visited_checkout(self) -> bool:
    return self.flags.get("visited_checkout", False)
```

Existing code that reads `session.visited_checkout` (e.g. API routes) continues to work. The properties are thin wrappers over the generic `flags` dict.

---

## 2. Service Registry

### Before (hardcoded)

```python
# rule_adapter.py constructor
def __init__(self, ..., ba_orchestrator=None):
    self._ba = ba_orchestrator

# Hardcoded service dispatch
elif action.type == "escalate":
    if action.params.get("service") == "behavioral_analysis":
        self._ba.start(...)
```

Adding a new escalation service required modifying the constructor and adding a new `elif`.

### After (registry pattern)

```python
# Protocol definition
class EscalationService(Protocol):
    def start(self, object_id: str, region_id: str, scene_id: str) -> None: ...
    def stop(self, object_id: str, region_id: str) -> None: ...
    def stop_all(self, object_id: str) -> None: ...

# Registry-based dispatch
handler = self._service_registry.get(service_name)
if handler:
    handler.start(event.object_id, event.region_id, event.scene_id)
```

Registration at startup:
```python
# main.py
rule_adapter.register_service("behavioral_analysis", ba_orchestrator)
```

### How to Add a New Escalation Service

1. **Write the service** (implements `EscalationService` protocol):
   ```python
   class FaceRecognitionService:
       def start(self, object_id, region_id, scene_id): ...
       def stop(self, object_id, region_id): ...
       def stop_all(self, object_id): ...
   ```

2. **Register at startup** in `main.py`:
   ```python
   face_svc = FaceRecognitionService(...)
   rule_adapter.register_service("face_recognition", face_svc)
   ```

3. **Wire in rules.yaml** (NO code change to rule_adapter):
   ```yaml
   services:
     face_recognition:
       handler: face_recognition_service

   rules:
     - id: face_check_restricted
       trigger:
         event_type: zone_entry
         zone_type: RESTRICTED
       actions:
         - type: escalate
           params:
             service: face_recognition
   ```

### Service Lifecycle

| Event | What happens |
|-------|-------------|
| Rule matches with `type: escalate` | `handler.start(object_id, region_id, scene_id)` called |
| Person exits zone (EXITED event) | `handler.stop(object_id, region_id)` called for ALL registered services |
| Person lost (PERSON_LOST event) | `handler.stop_all(object_id)` called for ALL registered services |

---

## What is Configurable vs. What Requires Code

| Aspect | Configurable at runtime (YAML only) | Requires code |
|--------|--------------------------------------|---------------|
| New zone types | Yes — `zone_config.json` | No* |
| New session flags (zone_visited) | Yes — `rules.yaml session_flags` | No |
| New session flags (external) | Yes — `rules.yaml session_flags` | No |
| New rule conditions | Yes — `rules.yaml rules` | No |
| New rule triggers | Yes — `rules.yaml rules` | No |
| Changing thresholds | Yes — `rules.yaml variables` | No |
| Enabling/disabling rules | Yes — `rules.yaml enabled` | No |
| Which service a rule invokes | Yes — `rules.yaml` action params | No |
| New escalation service logic | No | Yes (write service + register) |
| New alert types | No | Yes (add to AlertType enum) |
| New condition operators | No | Yes (add to _OPS in engine.py) |
| Alert dedup mechanics | No | Yes (operational plumbing) |

*Note: The `ZoneType` enum in `events.py` still constrains what zone types the session manager can emit. This is a separate change if fully dynamic zone types are needed.

---

## Architecture Diagram

```
rules.yaml
├── session_flags:
│   ├── visited_checkout: {trigger: zone_visited, zone_type: CHECKOUT}
│   ├── visited_pharmacy: {trigger: zone_visited, zone_type: PHARMACY}  ← new, no code change
│   └── concealment_suspected: {trigger: external, source: ba, ...}
├── services:
│   ├── behavioral_analysis: {handler: ba_orchestrator}
│   └── face_recognition: {handler: face_svc}  ← new, register in main.py
└── rules:
    └── conditions reference flag names (visited_pharmacy, etc.)

                    ┌──────────────────────────┐
                    │    ConfigService          │
                    │  get_session_flag_defs()  │
                    │  get_service_defs()       │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │   RuleEngineAdapter       │
                    │                          │
                    │ _zone_visited_flags:      │
                    │  {HIGH_VALUE: [visited_hv]│
                    │   CHECKOUT: [visited_ck]} │
                    │                          │
                    │ _service_registry:        │
                    │  {ba: orchestrator,       │
                    │   face: face_svc}        │
                    └────────────┬─────────────┘
                                 │
               on ENTERED event  │
               ┌─────────────────▼──────────────┐
               │ for flag in zone_visited_flags: │
               │   session.flags[flag] = True    │
               └────────────────────────────────┘

                    ┌───────────────────────────┐
                    │    PersonSession          │
                    │  flags: Dict[str, bool]   │
                    │    {"visited_checkout": T, │
                    │     "visited_pharmacy": T} │
                    │                           │
                    │  @property visited_*      │
                    │  (backward compat)        │
                    └───────────────────────────┘
```

## Test Coverage

All 27 existing tests pass with no modifications to test logic (only fixture updates to use `flags={}` syntax). The tests validate:

- Config-driven session flag setting on zone entry
- Checkout bypass detection using flags
- Concealment severity upgrade using flags
- BA result processing with external flag trigger
- Alert dedup across all alert types
- Session manager zone tracking
