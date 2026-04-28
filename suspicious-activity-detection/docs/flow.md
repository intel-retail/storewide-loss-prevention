# End-to-End Flow: HIGH_VALUE Zone Visit

This document traces a single person's visit to a HIGH_VALUE zone through every
MQTT topic, handler, and service in `swlp-service`.

## MQTT Subscriptions (set up at startup)

| Topic | Handler | Defined in |
|---|---|---|
| `scenescape/data/scene/+/+` | `SessionManager.on_scene_data` | `services/mqtt_service.py` |
| `scenescape/event/region/+/+/+` | `SessionManager.on_region_event` | |
| `scenescape/data/region/+/+` | `SessionManager.on_region_data` | |
| `scenescape/data/camera/+` | `on_camera_image` (closure in `main.py`) | |
| `ba/results` | `BAQueueConsumer._handle_ba_result` | `services/ba_queue.py` |

## MQTT Publications

| Topic | Direction | Publisher |
|---|---|---|
| `scenescape/cmd/camera/{cam}` | swlp → camera | `BehavioralAnalysisOrchestrator` |
| `ba/requests` | swlp → BA service | `BAQueuePublisher` |
| `alerts/{type}` | swlp → UI | `AlertServiceClient` |

---

## Timeline of One Visit

### T=0.0s — Person enters store (not yet in HV zone)

- **IN** `scenescape/data/scene/{scene}/person` (every frame, ~10 Hz)
  - `SessionManager.on_scene_data`
    - Creates `PersonSession`, `reid_state="pending_collection"`
    - Resolves canonical UUID via `previous_ids_chain` (collapses re-id flicker)

### T≈1–3s — Re-id settles

- **IN** `scenescape/data/scene/{scene}/person`
  - `reid_state` flips to `"matched"`
  - Session is now real; visible in `/sessions` API

### T=5s — Person crosses HV zone boundary

- **IN** `scenescape/event/region/{scene}/{region}/count` (one-shot)
  - `SessionManager.on_region_event` with `entered: [...]`
  - `SessionManager._fire_enter`
    - Records `RegionVisit(entry_time)`
    - `session.current_zones[region] = entry_iso`
    - Emits `ENTERED` event
  - `RuleEngineAdapter.on_event`
    - Sets flag `visited_high_value = True`
    - Rule `behavioral_analysis` fires → `escalate` action
    - `ba_orchestrator.start(person, region, scene)`

### T≥5s — Orchestrator loop runs (1 task per visit, 5 Hz)

- **OUT** `scenescape/cmd/camera/{cam}` payload `getimage`
  - Skipped during first 2 s (`ba_initial_delay`)
  - Skipped while `reid_state != "matched"`

### T≥5s — Per-frame camera reply

- **IN** `scenescape/data/camera/{cam}` (JPEG base64)
  - `on_camera_image` (in `main.py`)
    - Guard: `session.reid_state == "matched"` and person in HV zone
    - `frame_mgr.store_person_frame()` writes to:
      - `loss-prevention-frames/{scene}/{oid}/{ts}.jpg` (rolling buffer)
      - `behavioral-frames/{scene}/{oid}/{region}/{entry_ts}/frames/{ts_ms}.jpg`

### T≥5s — Continuous region data (every frame)

- **IN** `scenescape/data/region/{scene}/{region}`
  - `SessionManager.on_region_data`
    - Reads SceneScape's `dwell` for this person
    - Emits `LOITER` event with `dwell_seconds`
  - `RuleEngineAdapter` evaluates `loitering` rule
    - Condition: `dwell_seconds > 20`
    - Once true: `LOITERING` alert fires (deduped per region)

### T=7s — BA request (after `ba_initial_delay`)

- **OUT** `ba/requests`
  ```json
  {"person_id": "...", "region_id": "...",
   "entry_timestamp": "...", "scene_id": "..."}
  ```
  - BA service consumes, reads `behavioral-frames` bucket, runs VLM,
    publishes result.

### T=8s — BA result

- **IN** `ba/results`
  - `BAQueueConsumer._handle_ba_result`
  - If `status == "suspicious"`: fires `CONCEALMENT` alert
  - `session.ba_alerted[region] = True` → orchestrator self-terminates

### T=25s — Loiter threshold crossed

- Triggered by ongoing `region_data` ticks
- **OUT** `alerts/loitering` via `AlertServiceClient`

### T=40s — Person leaves HV zone

- **IN** `scenescape/event/region/{scene}/{region}/count`
  - `SessionManager.on_region_event` with `exited: [...]`
  - `SessionManager._fire_exit`
    - Closes `RegionVisit(exit_time, dwell)`
    - `session.current_zones.pop(region)`
    - Emits `EXITED` event
  - `RuleEngineAdapter.on_event`
    - Iterates `_service_registry` → `ba_orchestrator.stop()` (no-op,
      task self-terminates after `absence_grace`)
    - `_deferred_frame_cleanup` scheduled (currently disabled)

### T≈40–45s — Orchestrator task self-terminates

- `_run()` sees `region_id not in current_zones`
- Waits `absence_grace` (5 s) — survives flicker
- Returns; task removed from `_tasks` dict

### T=130s — Person exits store / absent

- `SessionManager.run_expiry_loop` notices `last_seen > 90 s` ago
- `_expire_session`
  - Closes any open visits → emits `EXITED` defensively
  - Emits `PERSON_LOST` → `ba_orchestrator.stop_all()`
  - Deletes `_sessions[skey]`
  - Prunes `_oid_alias` entries

---

## Key Handlers Per File

| File | Function | Responsibility |
|---|---|---|
| `main.py` | `on_camera_image` | Writes JPEGs to S3 (gated on `reid_state == "matched"`) |
| `services/session_manager.py` | `on_scene_data` | Session lifecycle, `reid_state` promotion |
| `services/session_manager.py` | `on_region_event` | `ENTERED` / `EXITED` |
| `services/session_manager.py` | `on_region_data` | `LOITER` (continuous) |
| `services/rule_adapter.py` | `on_event` | Rule evaluation + escalation |
| `services/ba_orchestrator.py` | `_run` | Per-visit `getimage` + `ba/requests` loop |
| `services/ba_queue.py` | `BAQueueConsumer` | Consumes `ba/results`, fires alerts |

---

## reid_state Gates (Why They Exist)

Frame writes and BA requests are gated on `session.reid_state == "matched"`
to avoid creating ghost folders in the `behavioral-frames` bucket and
spurious VLM calls for short-lived flickering tracks.

Trade-off: the first 1–3 s of footage after zone entry are dropped
(person walking in). Concealment behaviour is observed afterward, so
this is acceptable.
