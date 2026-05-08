# POI MQTT Pipeline Design

## Topics Subscribed

| # | Topic | Source Consumer | Purpose |
|---|-------|-----------------|---------|
| 1 | `scenescape/data/camera/+` | `EventConsumer` | Person detection + face reid embeddings (PRIMARY — used for FAISS matching) |
| 2 | `scenescape/regulated/scene/+` | `ScenescapeRegionConsumer` | Scene tracking + region entry/exit |
| 3 | `scenescape/external/+/person` | `EventConsumer` | Global UUID observability + movement event storage (monitoring only — NOT used for FAISS) |

---

## Topic 1: `scenescape/data/camera/{camera_id}`

**Published by:** DLStreamer pipeline server (runs person + face detection + reid models)

**Raw payload structure:**
```json
{
  "timestamp": "2026-04-27T07:42:17.916Z",
  "objects": {
    "person": [
      {
        "id": 1,
        "bounding_box_px": [200, 150, 280, 380],
        "confidence": 0.91,
        "metadata": {
          "reid": {
            "embedding_vector": "AABAP...base64string...1368chars"
          }
        },
        "sub_objects": {
          "face": [
            {
              "id": "face-001",
              "bounding_box_px": [210, 155, 260, 210],
              "metadata": {
                "reid": {
                  "embedding_vector": "AABAP...base64string...1368chars"
                }
              }
            }
          ]
        }
      }
    ],
    "face": []
  }
}
```

**What we extract:**
1. `camera_id` — from the MQTT topic itself
2. `person[].id` — SceneScape tracking ID
3. `person[].sub_objects.face[].metadata.reid.embedding_vector` ← **face embedding (only source used for FAISS)**
4. Embedding decode: `base64.b64decode(str)` → 1024 bytes → `struct.unpack("256f", raw)` → **256-dim float vector**

> **Important:** Body-level embeddings (`person[].metadata.reid.embedding_vector`) come from
> `person-reidentification-retail-0277` — a DIFFERENT embedding space from the face model.
> They are **not** used for FAISS matching. Only face sub-object embeddings from
> `face-reidentification-retail-0095` are sent to FAISS.

**What we do with it:**
```
embedding vector → DetectionIndex.claim_track(track_id, vector, frame)
                   ├── if first time (SETNX gate)
                   │     → store entry in FAISS + detection:frame:{faiss_id}
                   └── always → update rolling exit vector + frame
                → FAISS POI search (top_k=10, threshold=SIMILARITY_THRESHOLD)
                → if match found → create alert → store in Redis
                → always → store MovementEvent in Redis
```

---

## Topic 2: `scenescape/regulated/scene/{scene_id}`

**Published by:** SceneScape scene controller (regulated/smoothed tracking)

**Raw payload structure:**
```json
{
  "timestamp": "2026-04-27T07:42:20.100Z",
  "id": "db68a737-92db-4477-880b-07bc7d658ab9",
  "objects": [
    {
      "id": 1,
      "category": "person",
      "visibility": ["Camera_01"],
      "metadata": {
        "reid": { "embedding_vector": [[0.12, -0.05, "... 256 floats"]] }
      },
      "regions": {
        "entrance-zone": { "entered": "2026-04-27T07:41:55.000Z" },
        "aisle-3":       { "entered": "2026-04-27T07:42:10.000Z" }
      },
      "sub_objects": { "face": [] }
    }
  ]
}
```

**What we extract (stateful diff per person):**
- `current_regions = set(person["regions"].keys())`
- `previous_regions` = in-memory state from last message for same `person.id`
- `current - previous` → **region ENTERED** → `store_region_entry()`
- `previous - current` → **region EXITED** → `store_region_exit()` + compute dwell time
- If person disappears from `objects` entirely → forced exit for all their regions

---

## Redis Data Model

### 1. Movement Events — `event:{object_id}:{timestamp}`

Written every detection frame. TTL = 7 days.

```
KEY:   event:1:2026-04-27T07:42:22.753Z
TYPE:  string (JSON)
TTL:   604800 sec (7 days)

VALUE:
{
  "object_id": 1,
  "timestamp": "2026-04-27T07:42:22.753Z",
  "camera_id": "Camera_02",
  "region":    "Camera_02",
  "poi_id":    null,              ← filled when POI match found
  "embedding_reference": null
}
```

> Currently **3,425+ movement event keys** live from Camera_01 and Camera_02.

---

### 2. POI Registry — `poi:{poi_id}` + `poi:index`

Created when an operator enrolls a POI via the UI.

```
KEY:   poi:{poi_id}              → full POI JSON  (no TTL — permanent)
KEY:   poi:index                 → Redis SET of all poi_ids

VALUE of poi:{poi_id}:
{
  "event_type": "poi_enrollment",
  "poi_id": "poi-a3f2c1b0",
  "enrolled_by": "operator",
  "severity": "high",            ← low | medium | high
  "notes": "Shoplifter suspect",
  "reference_images": [
    {
      "source": "uploaded_image",
      "embedding_id": "emb-0001",
      "vector_dim": 256,
      "image_path": "/data/images/poi-a3f2c1b0.jpg"
    }
  ],
  "status": "active",
  "timestamp": "2026-04-27T08:00:00Z"
}
```

---

### 3. FAISS ID Mappings — `faiss2poi:{int}` + `poi2faiss:{poi_id}`

Created alongside POI enrollment to bridge FAISS integer indices to POI string IDs.

```
KEY:  faiss2poi:{int}            → "poi-a3f2c1b0"   (FAISS internal index → POI ID)
KEY:  poi2faiss:{poi_id}         → SET{0, 1, ...}   (POI ID → all its FAISS vector indices)
```

---

### 4. Alert Cache & Dedup

```
KEY:  alert:sent:{object_id}:{poi_id}  → "1"    TTL=configurable  (dedup: 1 alert per window per person+POI pair)
KEY:  alert:{alert_id}                 → full alert JSON   TTL=7 days
KEY:  alerts:recent                    → LIST of last 1000 alert JSONs  (no TTL)

VALUE of alert:{alert_id}:
{
  "event_type": "poi_match_alert",
  "alert_id": "alert-uuid",
  "timestamp": "2026-04-27T08:05:12Z",
  "poi_id": "poi-a3f2c1b0",
  "severity": "high",
  "status": "New",
  "match": {
    "camera_id": "Camera_02",
    "confidence": 0.91,
    "similarity_score": 0.87,
    "bbox": [200, 150, 280, 380],
    "frame_number": 0,
    "thumbnail_path": "/api/v1/thumbnail/cam:Camera_02:1"
  },
  "poi_metadata": {
    "notes": "Shoplifter suspect",
    "enrollment_date": "2026-04-20T08:00:00Z",
    "total_previous_matches": 3
  }
}
```

---

### 5. Object → POI Cache — `object:{object_id}`

Cache-Aside pattern: avoids hitting FAISS on every video frame for the same tracked person.

```
KEY:  object:{object_id}  → JSON {"poi_id": "poi-a3f2c1b0", "similarity": 0.87}   TTL=configurable
```

Once matched, subsequent frames for the same `object_id` skip FAISS entirely for the cache
TTL period (compose default: 5 seconds).

---

### 6. Region Presence & Dwell — `region:presence:*` + `region:dwell:*`

Written when regions are configured as polygons in the SceneScape scene.

```
KEY:  region:presence:{scene_id}:{region_id}:{object_id}    TTL=1h
VALUE:
{
  "first_seen": "2026-04-27T07:41:55Z",
  "region_name": "entrance-zone",
  "camera_id": "Camera_01"
}

KEY:  region:dwell:{object_id}:{scene_id}:{region_id}:{date}  TTL=7 days
VALUE:
{
  "object_id": 1,
  "scene_id": "db68a737-92db-4477-880b-07bc7d658ab9",
  "region_id": "entrance-zone",
  "region_name": "entrance-zone",
  "exit_time": "2026-04-27T07:42:30Z",
  "dwell_sec": 35.0
}
```

### 7. Detection Index — `detection:*`

Stores per-detection data for the all-detections FAISS index.

```
KEY:  detection:meta:{faiss_id}     → JSON metadata (track_id, camera_id, timestamp, bbox)
KEY:  detection:frame:{faiss_id}    → JPEG binary (entry frame)   TTL=7 days
KEY:  detection:track_count:{track} → integer (embedding count for track)
KEY:  detection:track_last:{track}  → timestamp (last embedding stored)
```

### 8. Track Exit Frames — `track:exit:*`

Rolling exit data, updated on each new detection for an active track.

```
KEY:  track:exit:vector:{track_id}   → float32 bytes (latest exit embedding)
KEY:  track:exit:frame:{track_id}    → JPEG binary (latest exit frame)
KEY:  track:exit:meta:{track_id}     → JSON (timestamp, camera_id, bbox)
```

### 9. Track Frames — `track:frame:*`

```
KEY:  track:frame:{track_id}:entry      → JPEG binary (first frame seen)
KEY:  track:frame:{track_id}:last_seen  → JPEG binary (most recent frame)
```

---

## End-to-End Data Flow

```
Camera Feed
    │
    ▼
DLStreamer Pipeline Server
  (person-detection-retail-0013 +
   face-detection-retail-0004 +
   face-reidentification-retail-0095 +
   person-reidentification-retail-0277)
    │
    ├──► scenescape/data/camera/Camera_01
    └──► scenescape/data/camera/Camera_02
              │
              ▼
       poi-backend: EventConsumer
       ├── decode base64 → 256-dim float vector
       │     (face sub_object embedding only — body reid is NOT used for FAISS)
       ├── MatchingService.match_object(object_id, vector)
       │     ├── Cache hit (object:N)?  → skip FAISS, return cached poi_id
       │     └── Cache miss → FAISS.search(top_k=10, threshold=SIMILARITY_THRESHOLD)
       │                         │
       │                    match found?
       │                    ├── YES → AlertService
       │                    │           ├── dispatch to ALL strategies (must all succeed)
       │                    │           ├── store  alert:{id}  in Redis
       │                    │           ├── push   alerts:recent list
       │                    │           └── set    alert:sent:{obj_id}:{poi_id}  TTL=configurable
       │                    └── NO  → nothing
       └── store event:{object_id}:{timestamp} in Redis  (always, TTL=7d)

SceneScape Scene Controller
    │
    └──► scenescape/regulated/scene/{scene_id}
              │
              ▼
       poi-backend: ScenescapeRegionConsumer
       ├── diff regions per person (stateful in-memory)
       ├── entry detected → store region:presence:* (TTL=1h)
       └── exit  detected → compute dwell_sec
                          → store region:dwell:* (TTL=7d)
                          → delete region:presence:*
```

---

## Redis Key Summary Table

| Key Pattern | Type | TTL | Purpose |
|---|---|---|---|
| `event:{obj_id}:{timestamp}` | string (JSON) | 7 days | Movement event per detection frame |
| `events:poi:{poi_id}` | SET of event keys | 7 days | Index: all events for a matched POI |
| `poi:{poi_id}` | string (JSON) | none | POI metadata (enrolled person) |
| `poi:index` | SET | none | All registered POI IDs |
| `faiss2poi:{int}` | string | none | FAISS index → POI ID mapping |
| `poi2faiss:{poi_id}` | SET | none | POI ID → FAISS indices |
| `object:{obj_id}` | string (JSON) | configurable | Cache-Aside: tracking ID → matched POI + similarity |
| `alert:{alert_id}` | string (JSON) | 7 days | Full alert record |
| `alerts:recent` | LIST | none | Last 1000 alerts (ring buffer) |
| `alert:sent:{obj_id}:{poi_id}` | string | configurable | Dedup flag: alert already fired for this person+POI pair |
| `region:presence:{scene}:{region}:{obj}` | string (JSON) | 1h | Region entry timestamp |
| `region:dwell:{obj}:{scene}:{region}:{date}` | string (JSON) | 7 days | Completed region visit with dwell time |
| `detection:meta:{faiss_id}` | string (JSON) | 7 days | Detection metadata (track, camera, bbox) |
| `detection:frame:{faiss_id}` | binary (JPEG) | 7 days | Entry frame per detection |
| `detection:track_count:{track}` | integer | 7 days | Embedding count per track |
| `track:exit:vector:{track_id}` | binary | 7 days | Latest exit embedding |
| `track:exit:frame:{track_id}` | binary (JPEG) | 7 days | Latest exit frame |
| `track:frame:{track_id}:entry` | binary (JPEG) | 7 days | Track entry frame |
| `track:frame:{track_id}:last_seen` | binary (JPEG) | 7 days | Track last-seen frame |
