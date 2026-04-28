# End-to-End Flow: HIGH_VALUE Zone Visit

This document traces a single person's visit to a HIGH_VALUE zone through every
MQTT topic, handler, and service.

## MQTT Subscriptions

| Topic | Subscriber | Handler |
|---|---|---|
| `scenescape/data/scene/+/+` | swlp-service | `SessionManager.on_scene_data` |
| `scenescape/event/region/+/+/+` | swlp-service | `SessionManager.on_region_event` |
| `scenescape/data/region/+/+` | swlp-service | `SessionManager.on_region_data` |
| `scenescape/data/camera/+` | swlp-service | `on_camera_image` (in `main.py`) |
| `ba/requests` | behavioral-analysis | `BAQueueConsumer._on_message` |
| `ba/results` | swlp-service | `RuleEngineAdapter._handle_ba_result` |

## MQTT Publications

| Topic | Direction | Publisher | Cardinality |
|---|---|---|---|
| `scenescape/cmd/camera/{cam}` | swlp → camera | `BAOrchestrator` | every tick (5–10 Hz) |
| `ba/requests` action="start" | swlp → BA | `BAQueuePublisher.publish_start` | once per visit |
| `ba/requests` action="exit"  | swlp → BA | `BAQueuePublisher.publish_exit`  | once per visit |
| `ba/results` | BA → swlp | per-visit polling worker | one per detected event |
| `alerts/{type}` | swlp → UI | `AlertServiceClient` | per alert |

---

## Lifecycle of a Visit

```
T=0.0s   Person enters store
         IN  scenescape/data/scene/{scene}/person
         → SessionManager creates PersonSession
           reid_state = "pending_collection"

T=1–3s   Re-id settles; reid_state -> "matched"

T=5.0s   Person crosses HV zone boundary
         IN  scenescape/event/region/{scene}/{region}/count
         → SessionManager._fire_enter
              session.current_zones[region] = entry_iso
              emits ENTERED event
         → RuleEngineAdapter.on_event
              evaluates rule "behavioral_analysis"
              executes "escalate" → ba_orchestrator.start()

T≥5.0s   BA orchestrator task runs (one per visit)
         OUT scenescape/cmd/camera/{cam}  payload="getimage"   (every 1/fps s)
         IN  scenescape/data/camera/{cam}                       (camera reply)
         → on_camera_image
         → frame_mgr.store_person_frame()
              writes to behavioral-frames bucket
              key: {scene}/{oid}/{region}/{entry_ts}/frames/{ts_ms}.jpg
         OUT ba/requests action="start"                         (ONCE)
              {person_id, region_id, entry_timestamp, scene_id}
         → behavioral-analysis spawns a per-visit polling worker

T≥6.0s   BA polling worker (in behavioral-analysis service)
         every visit_poll_interval (1 s):
            list bucket prefix {scene}/{oid}/{region}/{entry_ts}/frames/
            collect frames newer than internal watermark
            if new_count >= min_frames_for_detection:
              extract poses (YOLO-Pose / RTMPose)
              run pattern detection
              if matched: call VLM
              OUT ba/results
                  {status: "suspicious"|"no_match", confidence,
                   vlm_response, frames_analyzed}
              advance watermark to newest analysed frame

T≥6.0s   Concealment alert (when ba/results status="suspicious")
         IN  ba/results
         → RuleEngineAdapter._handle_ba_result
              evaluates rule "concealment_detected"
              executes "alert" CONCEALMENT
         OUT alerts/concealment

         (Multiple suspicious verdicts per visit are allowed -- the worker
          keeps polling; new concealment events fire additional alerts.)

T≥5.0s   Continuous loiter detection (separate from BA)
         IN  scenescape/data/region/{scene}/{region}
         → SessionManager.on_region_data emits LOITER events with dwell
         → rule "loitering" fires LOITERING alert when dwell > threshold

T=40s    Person leaves HV zone
         IN  scenescape/event/region/{scene}/{region}/count  (exited list)
         → SessionManager._fire_exit
         → RuleEngineAdapter.on_event handles EXITED
              calls ba_orchestrator.stop(object_id, region_id)
              → cancels the visit task; finally-block publishes
         OUT ba/requests action="exit"                          (ONCE)
              {person_id, region_id, entry_timestamp, scene_id}
         → behavioral-analysis cancels the per-visit polling worker

T=130s   Session timeout (last_seen > session_timeout_seconds)
         → SessionManager._expire_session
              emits PERSON_LOST
         → ba_orchestrator.stop_all()
              cancels any remaining tasks; each emits its exit message
```

## Key Files

| File | Function | Responsibility |
|---|---|---|
| swlp-service `main.py` | `on_camera_image` | Writes JPEGs to bucket (gated on `reid_state == "matched"`) |
| swlp-service `services/session_manager.py` | `on_scene_data` / `on_region_event` / `on_region_data` | Session lifecycle, ENTERED/EXITED, LOITER |
| swlp-service `services/rule_adapter.py` | `on_event`, `_handle_ba_result` | Rule evaluation + escalation; multiple BA verdicts per visit allowed |
| swlp-service `services/ba_orchestrator.py` | `_run` | Per-visit `getimage` capture loop; publishes `start`/`exit` lifecycle events |
| swlp-service `services/ba_queue.py` | `BAQueuePublisher` | `publish_start` / `publish_exit` |
| behavioral-analysis `src/ba_queue.py` | `BAQueueConsumer._run_worker` | Per-visit bucket polling + analysis loop |

## Why Polling Lives on the BA Side

The behavioural-analysis service is the only party that knows when "enough new
frames" have arrived to make analysis meaningful. Inverting the cadence to live
there:

* swlp-service publishes only two MQTT messages per visit (start/exit) instead
  of one per second.
* BA decides when to run pose+VLM based on data readiness, not on a timer set
  by an upstream service.
* Multiple discrete concealment events in the same visit each produce their
  own alert without needing time-cooldown logic.

## reid_state Gate

Frame writes are gated on `session.reid_state == "matched"` to avoid creating
ghost folders in the bucket for short-lived flickering tracks. Trade-off:
the first 1–3 s of footage after zone entry are dropped (person walking in).
Concealment is observed afterward, so this is acceptable.
