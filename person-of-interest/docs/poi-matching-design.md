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

The historical search API (`POST /api/v1/search`) uses a two-stage search pipeline:

### Stage 1: Enrolled POI Index
Searches the enrolled POI FAISS index (`FAISSRepository`) using the query embedding.
- Uses cosine similarity (IndexFlatIP on L2-normalized vectors)
- Applies configurable threshold (`similarity_threshold`, default 0.6)
- Groups hits by POI ID, takes best similarity per POI
- Applies margin check: rejects if `best - second_best < 0.05` (ambiguous match)
- If a POI is identified, returns all recorded events for that POI

### Stage 2: Detection Index (fallback)
If no POI match is found, searches the all-detections index (`DetectionIndexRepository`):
- Contains all face embeddings ever seen by DLStreamer (7-day TTL)
- Groups results by track ID, keeping best entry hit per track
- Searches exit vectors for matched tracks (rolling exit embeddings)
- Builds grouped appearance cards with entry + exit frames and zone dwells
- Returns sorted by overall similarity (max of entry and exit)

### Track Purity Filter
DLStreamer reuses integer track IDs across different physical persons. To prevent
false positives from track ID reuse:
- For each track, counts events per POI from Redis
- Computes purity = our_count / total_count
- Skips tracks with purity < 40% (clearly dominated by another person)

### Multi-Embedding Detection Index
Each tracked person stores up to 5 face embeddings (spaced 10 seconds apart) for
more robust matching. The detection index uses a counter + cooldown approach in
`claim_track()` to avoid storing too many embeddings from the same moment.

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
  │   ├─ Normalize: float32, /255.0  ← pixel values to [0,1]
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
input: [1, 3, 128, 128] BGR, float32, /255.0
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
| Similarity threshold | `SIMILARITY_THRESHOLD` | 0.6 | Minimum cosine similarity for match |
| Face confidence filter | (hardcoded) | 0.80 | Minimum DLStreamer face detection confidence |
| Object cache TTL | `OBJECT_CACHE_TTL` | 300s | Cache matched object_id → poi_id |
| Alert dedup TTL | `ALERT_DEDUP_TTL` | 300s | Suppress duplicate alerts |
| Alert-service dedup window | config.yaml | 60s | Field-hash dedup at alert-service |
| FAISS dimension | (hardcoded) | 256 | face-reidentification-retail-0095 output |
| FAISS top_k | `SEARCH_TOP_K` | 10 | Max candidates from FAISS search |
| Thumbnail buffer | (hardcoded) | 60 frames | Ring buffer for MQTT image frames |
| Thumbnail poll interval | (hardcoded) | 0.08s | getimage command frequency |
