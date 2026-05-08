# POI Real-Time Matching — Low-Level Design

## 1. System Overview

The POI (Person of Interest) system performs **real-time face re-identification** across multiple cameras in a retail environment. It answers: *"Is this enrolled suspect visible on any camera right now?"*

```
┌─────────────────┐     RTSP      ┌──────────────────┐    MQTT     ┌──────────────┐
│  IP Cameras     │──────────────▶│  DLStreamer       │───────────▶│  MQTT Broker │
│  (Camera_01/02) │               │  (lp-video)       │            │  (Mosquitto) │
└─────────────────┘               │                    │            └──────┬───────┘
                                  │ person-detection   │                   │
                                  │ clip-reid (body)   │                   │ scenescape/data/camera/{id}
                                  │ face-detection     │                   │
                                  │ face-reid (256-d)  │                   ▼
                                  │ gvatrack           │            ┌──────────────┐
                                  └──────────────────┘            │  POI Backend │
                                                                   │  (FastAPI)    │
┌─────────────┐   REST /api/v1/poi   ┌────────────┐               │              │
│  POI UI     │◀────────────────────▶│  Enrollment │◀──────────────│  MQTT        │
│  (React)    │                      │  Service    │               │  Consumer    │
│             │◀─── WebSocket ──────▶│             │               │              │
└─────────────┘                      └─────┬──────┘               └──────┬───────┘
                                           │                              │
                                     ┌─────▼──────┐              ┌───────▼───────┐
                                     │  FAISS     │◀─────────────│  Matching     │
                                     │  IndexFlat │  cosine      │  Service      │
                                     │  IP (256-d)│  search      │  (cache-aside)│
                                     └────────────┘              └───────┬───────┘
                                                                         │ EventBus
                                     ┌────────────┐              ┌───────▼───────┐
                                     │  Redis     │◀─────────────│  Alert        │
                                     │  (metadata,│              │  Service      │
                                     │   cache,   │              │  (observer)   │
                                     │   alerts)  │              └───────┬───────┘
                                     └────────────┘                      │ HTTP POST
                                                                  ┌──────▼───────┐
                                                                  │  Alert       │
                                                                  │  Service     │──▶ WebSocket → UI
                                                                  │  (fan-out)   │──▶ Log
                                                                  └──────────────┘
```

---

## Offline Search Architecture

The historical search API (`POST /api/v1/search`) searches the all-detections index:

### Detection Index Search
Searches the `DetectionIndexRepository` (every face seen by DLStreamer, 7-day retention):
- Generates a query embedding from the uploaded image via OpenVINO
- Uses cosine similarity (IndexFlatIP on L2-normalized vectors)
- Applies configurable threshold (`SEARCH_SIMILARITY_THRESHOLD`, default 0.65)
- Groups results by track/appearance ID, keeping best entry hit per track
- Filters by time range if specified (start_time / end_time)
- Searches exit vectors for matched tracks (rolling exit embeddings)
- Builds grouped appearance cards with entry + exit frames and zone dwells
- Returns sorted by overall similarity (max of entry and exit)

> See **§10. Offline Search Pipeline** for the full low-level design.

### Detection Index Lifecycle
Each tracked person stores one entry embedding (on first detection) with additional
embeddings stored up to `DETECTION_EMBEDDINGS_PER_TRACK` (default 5), spaced by
`DETECTION_EMBEDDING_INTERVAL` seconds (default 10). The `claim_track()` SETNX gate
prevents the same appearance from being stored repeatedly.

## Entry/Exit Frame Architecture

The system captures and stores frames at key moments:

- **Entry frame**: Stored per FAISS ID when a face is first indexed in the detection
  index. Uses key `detection:frame:{faiss_id}`. Immutable — never overwritten.
- **Exit frame**: Rolling frame updated as a tracked person continues to be detected.
  Uses key pattern `track:exit:frame:{track_id}`. Overwritten on each new detection.
- **Track-level frames**: Stored in Redis at `track:frame:{track_id}:entry` and
  `track:frame:{track_id}:last_seen` by the MQTT consumer.
- **Zone frames**: Entry and exit frames per zone dwell, stored alongside region dwell
  records.

The `exit_promoter.py` module handles promoting exit vectors and frames when tracks
are finalized.

## 2. Enrollment Flow

### Sequence

```
User → POST /api/v1/poi (images[], severity, notes)
  │
  ├─ Validate: 1-5 images, non-empty
  │
  ▼
POIService.create_poi()
  │
  ├─ Generate poi_id (uuid4 short hash)
  │
  ├─ For each image:
  │   ├─ OpenCV decode (imdecode)
  │   ├─ Face Detection (face-detection-retail-0004, threshold > 0.5)
  │   │   └─ Select highest-confidence face
  │   ├─ Crop face region
  │   ├─ Resize to 128×128
  │   ├─ Convert to float32 (raw [0,255] — NO /255 scaling)
  │   ├─ Face Re-ID inference (face-reidentification-retail-0095)
  │   │   └─ Output: 256-d float32 vector
  │   ├─ L2-normalize embedding (unit vector)
  │   └─ Save reference image to /data/uploads/{poi_id}/ref_{i}.jpg
  │
  ├─ FAISSRepository.add(poi_id, embeddings[])
  │   ├─ faiss.normalize_L2(vectors)  ← redundant but defensive
  │   ├─ index.add_with_ids(vectors, [next_id, ...])
  │   ├─ Update id_map: {faiss_id → poi_id}
  │   └─ Persist index + id_map to disk
  │
  ├─ Redis: store POI metadata at poi:{poi_id}
  │   └─ {poi_id, severity, notes, reference_images[], status, created_at}
  │
  └─ Return {poi_id, status: "active", ...}
```

### Key Classes

| Class | File | Responsibility |
|-------|------|----------------|
| `POIService` | `service/poi_service.py` | Orchestrates enrollment end-to-end |
| `EmbeddingModelFactory` | `factory/factories.py` | Singleton; loads OpenVINO models, generates 256-d face embeddings |
| `FAISSRepository` | `infrastructure/faiss/repository.py` | CRUD on FAISS IndexFlatIP + IndexIDMap wrapper |
| `RedisPOIRepository` | `infrastructure/redis/repository.py` | POI metadata CRUD in Redis |
| `POI`, `ReferenceImage` | `domain/entities/poi.py` | Domain entities |

### Embedding Model Details

```python
# Face Detection
model: face-detection-retail-0004 (FP32)
input: [1, 3, 300, 300] BGR
output: [1, 1, N, 7] — (batch, label, conf, x1, y1, x2, y2)
threshold: 0.5

# Face Re-Identification
model: face-reidentification-retail-0095 (FP32)
input: [1, 3, 128, 128] BGR, float32 (raw pixel values [0,255] — no /255 scaling)
output: [1, 256] — 256-dimensional embedding
post-processing: L2-normalize → unit vector
```

---

## 3. Real-Time Matching Flow (Online Pipeline)

This is the core online path — happens for every camera frame at ~10 FPS.

### Sequence

```
DLStreamer Pipeline (per frame at 10 FPS)
  │
  ├─ person-detection-retail-0013 (threshold 0.5)
  ├─ clip-reid-market1501 (body re-id, GPU)
  ├─ face-detection-retail-0004 (threshold 0.6)
  ├─ face-reidentification-retail-0095 (face re-id, GPU)
  ├─ gvatrack (short-term-imageless)
  └─ MQTT publish → scenescape/data/camera/{camera_id}
      │
      │  JSON payload:
      │  {
      │    "id": "Camera_01",
      │    "timestamp": "2026-05-06T06:30:00.936Z",
      │    "objects": {
      │      "person": [{
      │        "id": 1,
      │        "confidence": 0.95,
      │        "bounding_box_px": {x, y, width, height},
      │        "metadata": {"reid": {"embedding_vector": "<base64>", "model_name": "..."}},
      │        "sub_objects": {
      │          "face": [{
      │            "confidence": 0.98,
      │            "bounding_box_px": {...},
      │            "metadata": {"reid": {"embedding_vector": "<base64 256-d float32>"}}
      │          }]
      │        }
      │      }]
      │    }
      │  }
      │
      ▼
EventConsumer._handle_camera_event()
  │
  ├─ Parse camera_id from MQTT topic
  ├─ Extract person objects from payload
  ├─ Dedup persons by int_id within single message
  │
  ├─ For each person:
  │   ├─ Scan sub_objects.face[]
  │   ├─ Filter: face.confidence >= 0.80 (FACE_CONFIDENCE_THRESHOLD)
  │   ├─ Decode embedding_vector:
  │   │   └─ base64 → struct.unpack → 256 float32 values
  │   ├─ Select best face by confidence
  │   └─ Build object_id = "cam:{camera_id}:{person_int_id}"
  │
  └─ _run_matching(object_id, embedding, timestamp, camera_id, confidence, bbox)
      │
      ├─ Store movement event in Redis
      │
      ▼
  MatchingService.match_object(object_id, embedding)
      │
      ├─── Cache Check (Redis) ───────────────────────────────┐
      │    key: "object:{object_id}"                          │
      │    ├─ HIT + above threshold → return cached MatchResult│
      │    ├─ HIT + below threshold → evict cache, return None │
      │    └─ MISS → proceed to FAISS                         │
      │                                                        │
      ▼                                                        │
  CosineSimilarityStrategy.match()                             │
      │                                                        │
      ├─ faiss.normalize_L2(query_vector)                      │
      ├─ FAISSRepository.search(vector, top_k=10)              │
      │   └─ IndexFlatIP.search() → (distances[], ids[])       │
      │      Inner product on L2-normed vectors = cosine sim   │
      ├─ Map faiss_ids → poi_ids via id_map                    │
      ├─ Filter: similarity >= threshold (0.60)                │
      ├─ Sort descending by similarity                         │
      └─ Return list[MatchResult]                              │
      │                                                        │
      ├─ Cache best match: Redis SETEX                         │
      │   key: "object:{object_id}"                            │
      │   value: {poi_id, similarity}                          │
      │   TTL: object_cache_ttl (5s)                           │
      └────────────────────────────────────────────────────────┘
      │
      │ MatchResult found
      ▼
  _run_matching (continued)
      │
      ├─ Log: "POI match: poi={id} camera={cam} similarity={score}"
      ├─ Store match metadata in Redis (TTL 1h)
      ├─ Thumbnail capture (async, see §5)
      ├─ Build AlertPayload via AlertService.create_alert_payload()
      ├─ Store movement with poi_id
      └─ EventBus.publish("match_found", MatchFoundEvent)
```

### FAISS Index Architecture

```
IndexIDMap (wrapper)
  └─ IndexFlatIP (inner index)
       │
       ├─ Exhaustive inner-product search
       ├─ O(n) per query — fine for <10K vectors
       ├─ All vectors L2-normalized at insert time
       │   → inner product == cosine similarity
       ├─ Dimension: 256
       └─ Returns: (similarity_score, faiss_id) pairs

ID Mapping:
  IndexIDMap maps external IDs ↔ internal positions
  id_map.json: {"3": "poi-749b9abd", "4": "poi-c1474782"}
```

### Cache-Aside Pattern

```
                    ┌─────────────┐
                    │   Redis     │
     ┌──────────────│   Cache     │──────────────┐
     │  GET         │             │  SETEX        │
     │              └─────────────┘  (5s TTL)     │
     │                                             │
     ▼              ┌─────────────┐               │
  Cache HIT?  ──NO──▶   FAISS    │───match───────▶│
     │              │   Search   │                │
     │YES           └─────────────┘               │
     │                                             │
     ▼                                             │
  Return cached                                    │
  MatchResult                                      │
```

**Purpose**: Avoid repeated FAISS searches for the same tracked person within a short window. The DLStreamer tracker assigns stable `person_int_id` per camera, so `cam:Camera_02:1` stays the same across frames while the person is tracked.

---

## 4. Alert Flow

### Sequence

```
EventBus.publish("match_found", MatchFoundEvent)
  │
  ▼
AlertService._on_match_found(event)      ← Observer pattern
  │
  ├─ Dedup Check (Layer 1 — Backend)
  │   key: "alert:sent:{object_id}:{poi_id}"
  │   TTL: alert_dedup_ttl (60s)
  │   ├─ EXISTS → skip (log at DEBUG)
  │   └─ NOT EXISTS → proceed
  │
  ├─ Dispatch to all AlertStrategy instances:
  │   └─ AlertServiceStrategy.send(alert)
  │       ├─ Build REST payload:
  │       │   {
  │       │     "alert_type": "POI_MATCH",
  │       │     "timestamp": "...",
  │       │     "metadata": {
  │       │       "alert_id", "poi_id", "severity",
  │       │       "camera_id", "similarity_score", "confidence",
  │       │       "bbox": [x1,y1,x2,y2],
  │       │       "thumbnail_path": "/api/v1/thumbnail/{object_id}",
  │       │       "notes", "enrollment_date", "total_previous_matches"
  │       │     }
  │       │   }
  │       └─ POST → http://poi-alert-service:8000/api/v1/alerts
  │
  ├─ If ALL strategies succeed:
  │   ├─ Store alert in Redis: alert:{alert_id} (TTL 7 days)
  │   └─ Mark sent: SETEX alert:sent:{dedup_key} 60 "1"
  │
  ▼
Alert Service (separate container, port 8001)
  │
  ├─ Dedup Check (Layer 2 — Alert Service)
  │   strategy: field_hash(poi_id, camera_id)
  │   window: 60 seconds
  │   in-memory hash store
  │   ├─ Duplicate → drop
  │   └─ New → proceed
  │
  ├─ Route to delivery targets:
  │   ├─ Log delivery (always)
  │   └─ WebSocket broadcast → all connected UI clients
  │
  ▼
POI UI (React) receives WebSocket alert → renders notification
```

### Dedup Layers Summary

| Layer | Location | Key | TTL | Purpose |
|-------|----------|-----|-----|---------|
| Object Cache | MatchingService | `object:{cam:Camera:id}` | 5s | Skip FAISS for same tracked person |
| Backend Alert Dedup | AlertService | `alert:sent:{obj}:{poi}` | 60s | Suppress duplicate alerts same person+POI |
| Alert-Service Dedup | DedupEngine | SHA1(poi_id, camera_id) | 60s | Suppress duplicate alerts same POI+camera |

---

## 5. Thumbnail Capture Flow

```
_run_matching() — on POI match
  │
  ├─ claim_thumbnail(object_id, ttl=30)  ← Redis SETNX, prevents concurrent captures
  │
  └─ submit_capture(camera_id, bbox, timestamp) → ThreadPoolExecutor
      │
      ▼
  capture_thumbnail(camera_id, bbox, timestamp)
      │
      ├─ MQTT Image Path (preferred):
      │   │
      │   │  Background: _MqttImageSubscriber
      │   │  ├─ Subscribes to scenescape/image/camera/{id}
      │   │  ├─ Publishes "getimage" every 0.08s to scenescape/cmd/camera/{id}
      │   │  ├─ Caches frames in OrderedDict ring buffer (60 frames)
      │   │  └─ ~3-5 seconds of frame coverage
      │   │
      │   ├─ Lookup by timestamp:
      │   │   1. Exact match (same timestamp string)
      │   │   2. Nearest timestamp (within buffer, < 1s tolerance)
      │   │   3. Latest frame (last resort)
      │   │
      │   ├─ Decode base64 → OpenCV image
      │   ├─ Crop using bbox [x, y, width, height] + 10px padding
      │   └─ Encode cropped face → JPEG base64
      │
      ├─ RTSP Fallback:
      │   ├─ _FrameGrabber reads from rtsp://mediaserver:8554/{camera_id}
      │   ├─ Persistent background thread, auto-reconnect
      │   └─ Returns latest frame (no timestamp matching)
      │
      └─ Store: Redis SETEX thumbnail:{object_id} 3600 <base64_jpeg>
          └─ Served via GET /api/v1/thumbnail/{object_id}
```

---

## 6. Startup Wiring (Dependency Injection)

```python
# backend/main.py — lifespan()

# Infrastructure
faiss_repo      = FAISSRepository()          # Singleton, loads from disk
poi_repo        = RedisPOIRepository()
cache_repo      = RedisCacheRepository()
event_repo      = RedisEventRepository()
mapping_repo    = RedisEmbeddingMappingRepository()

# Strategy (pluggable via Strategy Pattern)
matching_strat  = CosineSimilarityStrategy(faiss_repo, mapping_repo)
alert_strat     = AlertServiceStrategy(cfg.alert_service_url)

# Services
event_bus       = EventBus()
poi_service     = POIService(poi_repo, faiss_repo, mapping_repo)
matching_svc    = MatchingService(matching_strat, cache_repo)
event_svc       = EventService(event_repo)
alert_svc       = AlertService([alert_strat], event_repo, poi_repo, event_bus)

# Consumers
event_consumer  = EventConsumer(matching_svc, alert_svc, event_svc, event_repo, event_bus)
region_consumer = ScenescapeRegionConsumer(event_repo)
mqtt_adapter    = MQTTAdapter(cfg, event_consumer, region_consumer)

# Route injection
poi_routes.init(poi_service)
search_routes.init(faiss_repo, mapping_repo, event_repo)
thumbnail_routes.init(event_repo)
```

---

## 7. Key Design Patterns

| Pattern | Where | Purpose |
|---------|-------|---------|
| **Clean Architecture** | All layers | API → Service → Domain ← Infrastructure |
| **Strategy** | `MatchingStrategy`, `AlertStrategy` | Pluggable matching algorithms and alert delivery |
| **Observer** | `EventBus` + `AlertService` | Decoupled match → alert dispatch |
| **Cache-Aside** | `MatchingService` + Redis | Avoid repeated FAISS searches for same tracked object |
| **Singleton** | `Config`, `FAISSRepository`, `EmbeddingModelFactory` | Single shared instance |
| **Factory** | `EmbeddingModelFactory` | Lazy model loading, encapsulated inference |
| **Builder** | `POIBuilder` | Fluent POI construction during enrollment |

---

## 8. Data Flow Summary

```
Camera Frame (10 FPS)
    │
    ▼
DLStreamer: Detect Person → Detect Face → Generate 256-d Embedding
    │
    ▼ (MQTT, ~10 msgs/sec per camera)
POI Backend: Extract face embedding from sub_objects
    │
    ├─ confidence < 0.80? → SKIP
    │
    ▼
MatchingService: Cache check → FAISS cosine search
    │
    ├─ similarity < 0.60? → SKIP (no match)
    │
    ▼
AlertService: Dedup check → Build AlertPayload → POST to alert-service
    │
    ├─ dedup hit (60s window)? → SKIP
    │
    ▼
Alert-Service: Dedup check → WebSocket broadcast → UI shows alert
    │
    ▼
Thumbnail: MQTT getimage → ring buffer lookup → bbox crop → Redis store
```

### Latency Budget (typical)

| Stage | Latency |
|-------|---------|
| DLStreamer inference (person+face detect+reid) | ~50-100ms |
| MQTT publish → consumer receive | ~1-5ms |
| Embedding decode + validation | <1ms |
| FAISS search (IndexFlatIP, <10 vectors) | <0.1ms |
| Redis cache check | <1ms |
| Alert dispatch (HTTP POST) | ~5-20ms |
| Thumbnail capture + crop | ~50-200ms |
| **End-to-end (detection → UI alert)** | **~100-300ms** |

---

## 9. Configuration Reference

| Parameter | Env Var | Default | Description |
|-----------|---------|---------|-------------|
| Similarity threshold | `SIMILARITY_THRESHOLD` | 0.6 | Minimum cosine similarity for online match |
| Search threshold | `SEARCH_SIMILARITY_THRESHOLD` | 0.65 | Minimum cosine similarity for offline search |
| Face confidence filter | (hardcoded) | 0.80 | Minimum DLStreamer face detection confidence |
| Object cache TTL | `OBJECT_CACHE_TTL` | 300s | Cache matched object_id → poi_id |
| Alert dedup TTL | `ALERT_DEDUP_TTL` | 300s | Suppress duplicate alerts |
| Alert-service dedup window | config.yaml | 60s | Field-hash dedup at alert-service |
| FAISS dimension | (hardcoded) | 256 | face-reidentification-retail-0095 output |
| FAISS top_k (online) | `SEARCH_TOP_K` | 10 | Max candidates from POI FAISS search |
| FAISS top_k (offline) | `DETECTION_INDEX_TOP_K` | 20 | Max candidates from detection index |
| Track seen TTL | `TRACK_SEEN_TTL` | 600s | Dedup gate for detection index per track |
| Embeddings per track | `DETECTION_EMBEDDINGS_PER_TRACK` | 5 | Max stored embeddings per appearance |
| Embedding interval | `DETECTION_EMBEDDING_INTERVAL` | 10s | Min seconds between stored embeddings |
| Thumbnail buffer | (hardcoded) | 60 frames | Ring buffer for MQTT image frames |
| Thumbnail poll interval | (hardcoded) | 0.08s | getimage command frequency |

---

## 10. Offline Search Pipeline (Historical Investigation)

The offline search API (`POST /api/v1/search`) provides historical person lookup.
Unlike online matching which compares against enrolled POIs only, offline search queries
the **detection index** — a FAISS index containing every face ever seen by any camera
(7-day retention via Redis TTL).

### Architecture

```
                                   ┌─────────────────────────┐
User uploads suspect image ────────▶  POST /api/v1/search    │
                                   └───────────┬─────────────┘
                                               │
                                               ▼
                                   ┌─────────────────────────┐
                                   │  EmbeddingModelFactory   │
                                   │  generate_from_bytes()   │
                                   │  → face detect → crop    │
                                   │  → resize 128×128        │
                                   │  → reid model → 256-d    │
                                   │  → L2-normalize          │
                                   └───────────┬─────────────┘
                                               │ query_vector
                                               ▼
                              ┌──────────────────────────────────┐
                              │     DetectionIndexRepository      │
                              │     (FAISS IndexFlatIP, in-mem)   │
                              │                                    │
                              │  .search(query_vector, top_k=20)  │
                              │  → L2-normalize query              │
                              │  → Inner Product search            │
                              │  → filter by Redis existence       │
                              └───────────────┬──────────────────┘
                                              │ hits: [(faiss_id, similarity)]
                                              ▼
                              ┌──────────────────────────────────┐
                              │  Group by track_id               │
                              │  Keep best entry hit per track   │
                              │  Filter: similarity ≥ threshold  │
                              │  Filter: timestamp in range      │
                              └───────────────┬──────────────────┘
                                              │ best_entry per track
                                              ▼
                              ┌──────────────────────────────────┐
                              │  search_exits(query_vec, tracks) │
                              │  → load exit vectors from Redis  │
                              │  → compute cosine similarity     │
                              │  → return {track_id: exit_sim}   │
                              └───────────────┬──────────────────┘
                                              │
                                              ▼
                              ┌──────────────────────────────────┐
                              │  Build grouped appearances       │
                              │  → entry frame URL               │
                              │  → exit frame URL                │
                              │  → zone dwell information        │
                              │  → sort by max(entry, exit) sim  │
                              └──────────────────────────────────┘
```

### Detection Index Lifecycle

The detection index stores embeddings for **every face** seen by DLStreamer cameras,
providing a searchable history independent of POI enrollment.

```
Camera detection event (face embedding)
    │
    ▼
EventConsumer._handle_camera_event()
    │
    ├── claim_track(object_id) ← Redis SETNX with track_seen_ttl
    │     │
    │     ├── First time (NX succeeds):
    │     │     ├── Create unique appearance_id = "{object_id}@{timestamp}"
    │     │     ├── set_active_appearance(object_id, appearance_id)
    │     │     ├── DetectionIndex.add(vector, camera_id, appearance_id, timestamp, bbox)
    │     │     │     ├── L2-normalize vector
    │     │     │     ├── Add to in-memory FAISS (IndexIDMap)
    │     │     │     ├── Persist metadata → detection:meta:{faiss_id}  (7-day TTL)
    │     │     │     └── Persist vector bytes → detection:vec:{faiss_id}  (7-day TTL)
    │     │     └── store_frame(faiss_id, b64_jpeg) → detection:frame:{faiss_id}
    │     │
    │     └── Already claimed (NX fails):
    │           └── Retrieve appearance_id from get_active_appearance(object_id)
    │
    └── update_exit(appearance_id, vector, camera_id, timestamp, bbox, frame)
          ├── Overwrite detection:exit_vec:{appearance_id}   (rolling)
          ├── Overwrite detection:exit_meta:{appearance_id}  (rolling)
          └── Overwrite detection:exit_frame:{appearance_id} (rolling)
              TTL = track_seen_ttl + 300s (buffer for promoter)
```

### Appearance ID vs Object ID

| Concept | Format | Purpose |
|---------|--------|---------|
| `object_id` | `uuid` or `cam:{camera}:{int_id}` | Stable within one appearance window |
| `appearance_id` | `{object_id}@{unix_timestamp}` | Globally unique per appearance — prevents cross-person contamination when tracker IDs are recycled |

The `@timestamp` suffix ensures that if DLStreamer reuses `person_id=1` for a different
physical person (after the gate expires), the new person gets a new appearance_id.

### Entry vs Exit Vectors

| | Entry | Exit |
|---|---|---|
| **When stored** | First detection (claim_track succeeds) | Every detection (rolling overwrite) |
| **Key pattern** | `detection:vec:{faiss_id}` + FAISS | `detection:exit_vec:{appearance_id}` |
| **Frame** | `detection:frame:{faiss_id}` | `detection:exit_frame:{appearance_id}` |
| **In FAISS** | Yes (permanent, 7-day TTL via Redis check) | No (Redis only, queried at search time) |
| **TTL** | 7 days (appearance_ttl_days × 86400) | track_seen_ttl + 300s |
| **Search method** | `DetectionIndex.search()` (FAISS inner product) | `search_exits()` (direct cosine computation) |

### Exit Promoter

The `promote_exits()` method is called periodically to move exit vectors into the FAISS
index after a person has left (gate expired):

```
Periodic task (every 60s)
    │
    ├── Scan all detection:exit_vec:* keys
    ├── For each track_id:
    │     ├── Check gate: detection:track:seen:{base_object_id}
    │     │     ├── Gate alive → skip (person still in frame)
    │     │     └── Gate expired → person has left
    │     ├── Check already promoted: detection:exit_promoted:{track_id} (NX key)
    │     │     ├── Exists → skip (already promoted)
    │     │     └── Set NX → proceed
    │     ├── Read exit vector bytes
    │     ├── Add to FAISS with role='exit' metadata
    │     └── Copy exit frame to detection:frame:{new_faiss_id}
    │
    └── Return count of promoted vectors
```

This gives the offline search TWO matching opportunities per appearance:
1. The entry embedding (first seen)
2. The exit embedding (last seen before leaving)

### Similarity Scoring

```
For each appearance in search results:
    entry_similarity = FAISS inner product (query vs entry vector)
    exit_similarity  = direct cosine (query vs rolling exit vector)
    overall_similarity = max(entry_similarity, exit_similarity)

Appearances are sorted by overall_similarity descending.
```

### UUID Resolution for Track Keys

SceneScape regulated scene topic provides global UUIDs for tracked persons across
cameras. The MQTT consumer resolves these via IoU matching:

```
ScenescapeRegionConsumer (scenescape/regulated/scene/+)
    │
    ├── Extracts per-person camera_bounds from message
    └── Stores in Redis: uuid_camera_bounds:{camera_id} → {uuid: bbox}

EventConsumer._handle_camera_event()
    │
    ├── Gets person bounding_box_px from detection
    ├── Calls event_repo.get_uuid_for_camera_bbox(camera_id, bbox, iou_threshold=0.3)
    │     ├── Loads all UUIDs with camera_bounds for this camera
    │     ├── Computes IoU between detection bbox and each UUID's camera_bounds
    │     └── Returns UUID with highest IoU (if ≥ 0.3)
    │
    ├── UUID found → object_id = uuid (globally unique, never recycled)
    └── No UUID   → object_id = "cam:{camera_id}:{person_int_id}" (fallback)
```

---

## 11. Online vs Offline Comparison

| Aspect | Online (Real-Time) | Offline (Historical Search) |
|--------|-------------------|-----------------------------|
| **Trigger** | Every MQTT face detection message | User uploads image via API |
| **FAISS Index** | POI index (enrolled suspects only) | Detection index (all faces, 7-day retention) |
| **Index type** | Disk-persisted (`IndexFlatIP` + `IndexIDMap`) | In-memory, rebuilt from Redis on restart |
| **Embedding source** | DLStreamer face sub_object (base64 decoded) | OpenVINO inference on uploaded image |
| **Normalization** | Both enrollment and query L2-normalized | Both stored vectors and query L2-normalized |
| **Similarity metric** | Cosine (Inner Product on L2-normed vectors) | Cosine (same) |
| **Threshold** | `SIMILARITY_THRESHOLD` (default 0.55) | `SEARCH_SIMILARITY_THRESHOLD` (default 0.65) |
| **Top-K** | `SEARCH_TOP_K` (default 10) | `DETECTION_INDEX_TOP_K` (default 20) |
| **Caching** | Cache-Aside (object:{id} → poi, TTL=5s) | None (one-shot query) |
| **Output** | Alert (WebSocket + log) | JSON appearance timeline with frames |
| **Grouping** | Per POI ID | Per track/appearance ID |
| **Frame evidence** | Thumbnail capture from MQTT/RTSP | Entry + exit frames from Redis |
| **Model** | face-reidentification-retail-0095 (256-d) | face-reidentification-retail-0095 (256-d) |
| **Preprocessing** | DLStreamer resize (128×128, raw float32) | OpenVINO resize (128×128, raw float32) |
