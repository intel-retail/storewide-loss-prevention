# VLM-Recall Search With VSS

Status: Draft / Proposal
Component: `storewide-loss-prevention/suspicious-activity-detection`

## 1. Use Case

> "Show me the person in a blue shirt between 2:00-3:00 PM at the entrance camera."

This is hybrid recall over historical footage:

| Query part | Example | Retrieval method |
|------------|---------|------------------|
| Appearance | `person in a blue shirt` | VSS multimodal frame embeddings |
| Time | `2:00-3:00 PM` | VSS `timeFilter` |
| Location | `entrance` / `cam-12` / `aisle-7-cam` | VSS upload `tag` (camera tag) |

A returned result is a short video clip (and its frames) that VSS already indexed, joined back
to the originating camera and any suspicious-activity event by a thin bridge service.

## 2. Approach In One Line

Tag each **camera** once, segment its **RTSP** stream into short MP4 clips, **tag and upload**
every clip from recall-enabled cameras, and let the **stock VSS search stack** own the index. A
small **wrapper/bridge service** does the RTSP -> tag -> VSS plumbing and joins results back to
cameras and clips.

There is **no per-frame aisle derivation**, **no rule/gating engine**, and **no SWLP-owned
vector database** in this design. Location is a camera tag, which VSS supports natively.

## 3. Why VSS Alone Is Enough (Validated)

The full VSS search stack is up and running (confirmed):
`nginx`, `pipeline-manager`, `video-search` (search-ms), `vdms-dataprep`, `vdms-vector-db`,
`multimodal-embedding-serving`, `minio-service`, `postgres-service`.

Stock VSS answers all three parts of the query:

| Query part | How VSS answers it | Native? |
|------------|--------------------|---------|
| `blue shirt` | multimodal frame embedding similarity (VDMS retrieval) | Yes |
| location (`entrance`, `aisle-7-cam`, `cam-12`) | upload **`tags`** filter | Yes (tags are per-video, set at upload) |
| `2:00-3:00 PM` | `timeFilter` on `created_at` | Yes (but `created_at` = upload time, see 6.3) |

Evidence (from the VSS sample app):

- Appearance similarity: `search-ms/server.py`, `search-ms/src/vdms_retriever/retriever.py`.
- Time filter + NL time parsing: `search-ms/server.py` `QueryRequest`, `src/utils/time_filters.py`.
- Tags at upload: `pipeline-manager/.../video.controller.ts`, `video.service.ts`.

Because each camera is tagged once, the per-video tag **is** the location filter. No SceneScape
region math is required for recall.

### 3.1 The Two Real Gaps (what the wrapper solves)

Stock VSS has exactly two limitations for this use case, and the wrapper exists to close them:

1. **VSS ingests MP4 files only — not RTSP.** The wrapper segments each RTSP stream into short,
   streamable MP4 clips and uploads them.
2. **`created_at` is the upload timestamp, not capture time.** The wrapper ingests in
   near-real-time and also records the true capture window in its own mapping table, so an
   investigator's real-world time range maps to the right clips.

## 4. Current Repo Hooks

| Need | Existing hook | Use in recall feature |
|------|---------------|-----------------------|
| Camera identity | SceneScape camera names | Map camera -> tags in `configs/cameras.yaml` |
| API surface | `swlp-service/api/routes.py` owns `/api/v1/lp/*` | Add the recall search proxy endpoint |
| UI | `ui/ui_gradio.py` serves the SWLP dashboard (FastAPI HTML at `GET /`, polls `/api/data`) | Add a "Recall Search" panel that calls the bridge `/recall/search` |

The wrapper continuously ingests clips from recall-enabled cameras and tags them by camera; it
does not gate on detections or run any detector.

## 5. Architecture

```mermaid
flowchart LR
    subgraph Cameras
      R1[RTSP cam-12]
      R2[RTSP entrance]
    end
    R1 --> SEG[Segmenter ffmpeg]
    R2 --> SEG
    SEG --> TAG[Tag Mapper]
    TAG --> UP[VSS Client: POST /manager/videos]
    UP --> EMB[POST /manager/videos/search-embeddings/:id]
    UP --> MAP[(Bridge mapping DB:\nvideoId -> camera, capture_start/end, tags, clip_url)]
    INV[Existing SWLP dashboard ui/ui_gradio.py] --> SAPI[Bridge Search API]
    SAPI --> VSEARCH[POST /manager/search]
    VSEARCH --> SAPI
    SAPI --> MAP
    SAPI --> INV
```

Proposed location for the new service:
`suspicious-activity-detection/vss-recall-bridge/`, added to `docker/docker-compose.yaml` on the
same network as the VSS stack.

## 6. Verified VSS API Contract

All paths are on the pipeline-manager, reached through nginx with the `/manager/` prefix
(default `http://<HOST_IP>:12345`). search-ms (`8000`), data-prep (`7890`), and embedding
serving (`9777`) are internal.

| Step | Method + Path | Body | Returns |
|------|---------------|------|---------|
| Upload clip | `POST /manager/videos` (multipart) | `video` (MP4 file), `name`, `tags` (comma-separated) | `{ "videoId": "..." }` |
| Create embeddings | `POST /manager/videos/search-embeddings/{videoId}` | none | `{ "status": "success" }` |
| Search | `POST /manager/search` | `{ "query", "tags" (comma-separated), "timeFilter": { "start", "end" } }` | results with per-segment `metadata` (`video_id`, `tags`, `created_at`, `segment_start/end`, `relevance_score`) |

### 6.1 Upload constraints

- Accepts **streamable MP4 only** (moov atom before mdat). **No RTSP URL** — file upload only
  (`pipeline-manager/.../video-validator.service.ts`).
- `tags` are accepted at upload and are the only per-video metadata channel -> this is where the
  **camera tag** goes.
- Embedding is **not** automatic; it is triggered by the explicit
  `POST /videos/search-embeddings/{videoId}` call.

### 6.2 Search request fields

`search-ms` `QueryRequest`: `query` (text), `tags` (list), `time_filter` (`{start, end}` ISO).
Tag match = any requested tag present on the result. There is no zone/person/camera field beyond
tags — which is exactly why camera identity is encoded as a tag.

### 6.3 Time handling caveat

`created_at` is set server-side at upload (`new Date().toISOString()` in `video.service.ts`), so
VSS time filtering is against **upload time**. The bridge handles this two ways (use both):

- Ingest in near-real-time so `created_at ≈ capture time`.
- Persist the true `capture_start/capture_end` per `videoId` in the bridge DB and translate the
  investigator's requested window into the corresponding clips, independent of upload time.

## 7. Ingestion (RTSP -> MP4)

One worker per configured camera. Segment with ffmpeg into short clips (e.g. 30-60 s):

```bash
ffmpeg -rtsp_transport tcp -i "$RTSP_URL" \
  -an -c:v copy -f segment -segment_time 60 -reset_timestamps 1 \
  -strftime 1 "/clips/${CAMERA_ID}_%Y%m%dT%H%M%S.mp4"
```

The segment muxer does **not** guarantee a faststart (moov-first) layout, which VSS requires.
After each segment closes, remux before upload:

```bash
ffmpeg -i in.mp4 -c copy -movflags +faststart out.mp4
```

If `-c copy` produces unstreamable output, re-encode with `-movflags +faststart`. Derive the true
capture window from the `strftime` filename (`capture_start = file start`,
`capture_end = start + segment_time`), not from upload time.

### 7.1 File ingest (pre-existing MP4 / backfill)

The wrapper also accepts an **already-recorded file** instead of a live RTSP stream. This is the
same pipeline minus the segmenter: take a file path + a camera id, faststart-remux it, then tag
and upload. Use this for backfill and for demos. A concrete run is shown in
[Section 12](#12-concrete-walkthrough-file-already-on-disk-tagged-cam1).

## 8. Tag Mapper

The tag list is the location channel. Per clip:

```text
tags = [ camera_id,            # e.g. "cam-12"
         area_label,           # e.g. "entrance", "checkout", "aisle-7-cam"
         store_id,             # e.g. "store-001"
         date_bucket ]         # optional "2026-06-17" for coarse filtering
```

Camera -> tags mapping lives in a static config (`configs/cameras.yaml`), so tagging a camera is a
one-time setup, not per-frame work.

## 9. VSS Client + Mapping Store

For each clip:

1. `POST /manager/videos` (multipart: file + `name` + `tags`) -> `videoId`.
2. `POST /manager/videos/search-embeddings/{videoId}` to trigger embedding.
3. Upsert a row in the bridge DB (small Postgres/SQLite):
   `videoId -> { camera_id, area_label, capture_start, capture_end, tags, clip_url }`.

The mapping store lets us (a) translate a real-world time window to the right clips despite VSS
storing upload time, and (b) join VSS results back to a camera and a playable clip URL.

### 9.1 Where the Mapping DB lives

The mapping store is a small, flat, relational table (one row per uploaded clip):
`video_id, camera_id, area_label, capture_start, capture_end, tags, clip_url`. It holds **no
video bytes** — clips live in VSS's MinIO; embeddings live in VDMS. Storage options:

| Option | What it is | Use when |
|--------|-----------|----------|
| **SQLite (MVP default)** | one file in a bridge volume | single host, getting started |
| **Dedicated Postgres database (chosen)** | a **separate** `recall_bridge` database created on VSS's existing `postgres-service` | shared/HA storage without a new container |

We reuse the **process** of VSS's `postgres-service` but **not** its application database. The
bridge owns a dedicated database (`recall_bridge`) so its `clips` table is fully isolated from
VSS's own `video`/`search`/`tag` tables and TypeORM migrations:

```sql
CREATE DATABASE recall_bridge;   -- isolated from VSS's own pipeline-manager DB
```

```bash
# bridge env (points at the SAME server, a SEPARATE database)
RECALL_DB_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres-service:5432/recall_bridge
```

**Never** write the bridge's tables into VSS's own application database — that DB is private to
the pipeline-manager (schema and migrations can change on any VSS upgrade, and
`docker compose down -v` would wipe the mapping rows). Same server, different database.

## 10. Search Proxy + Enrichment

Investigators call the bridge, not VSS directly:

```json
POST /api/v1/lp/recall/search
{
  "query": "person in a blue shirt",
  "cameras": ["entrance", "cam-12"],
  "time_start": "2026-06-17T14:00:00",
  "time_end": "2026-06-17T15:00:00",
  "limit": 20
}
```

The bridge does **not** rely on VSS's own `timeFilter` for the real-world window (see 10.1 for
why). It sends only the appearance + tag query to VSS and applies the time window itself:

```text
POST /manager/search
{
  "query": "person in a blue shirt",
  "tags":  "entrance,cam-12",            # camera/area tags
  "timeFilter": null                      # absolute window resolved by the bridge, not VSS
}
```

Then it enriches each VSS hit by joining `video_id` -> bridge mapping to attach `camera_id`,
`area_label`, true `capture_start/end`, and a playable `clip_url`, and returns the enriched,
sorted results to the UI.

### 10.1 Capture-time filtering in the bridge (Option A)

**Why not VSS's `timeFilter`.** Verified against the running stack: VSS stores `created_at` as the
**upload** timestamp (a result for the demo clip returned `created_at:
2026-06-19T07:58:54.696913+00:00`, matching its upload `createdAt` to the millisecond), and the
public one-off `POST /manager/search` only honors a **relative** `{value, unit}` window — an
absolute `{start, end}` is dropped by `normalizeTimeFilter`. So VSS time filtering answers
"uploaded in this window", not "captured in this window", which is wrong for investigator recall
and for backfilled clips.

**What the bridge does instead.** The mapping DB already stores the **true**
`capture_start/capture_end` per `videoId`. The bridge resolves the time window itself:

1. **Pre-filter by capture time.** Select the `videoId`s whose capture window intersects the
   requested `[time_start, time_end]`. Every axis is **optional** — a NULL predicate is a no-op, so
   the same query handles "time + camera", "camera only", or "no filter at all":
   ```sql
   SELECT video_id FROM clips
   WHERE (:time_start IS NULL OR capture_start < :time_end)
     AND (:time_end   IS NULL OR capture_end   > :time_start)
     AND (:cameras    IS NULL OR camera_id IN (:cameras));
   ```
   When **neither time nor camera** is given, the bridge skips the pre-filter entirely (candidate
   set = everything) and relies purely on VSS's semantic ranking — see the fallback note below.
2. **Appearance search in VSS.** Call `POST /manager/search` with the `query` and the camera/area
   `tags`, and `timeFilter: null`.
3. **Post-filter the hits.** Keep only VSS results whose `video_id` is in the pre-filtered set
   (and optionally re-check overlap), then enrich and sort. When the pre-filter was skipped, this
   step does **enrichment only** (attach camera/area/capture-time/clip_url to each VSS hit).

This gives true capture-time recall, is immune to upload lag, and works for backfill. It needs no
change to VSS — the bridge owns the time axis.

**Fallback when the query has no time (or no camera).** Time and camera are optional hints, so the
bridge degrades gracefully:

| User query | Pre-filter candidate set | Behaviour |
|------------|--------------------------|-----------|
| appearance + camera + time | clips matching camera **AND** time overlap | tightest, fastest, most accurate |
| appearance + camera (no time) | all clips for that camera, any time | post-filtered to that camera |
| appearance only (no time, no camera) | none — pre-filter skipped | pure VSS semantic ranking; post-filter does enrichment only |

```python
candidates = mapping.prefilter(time_start, time_end, cameras)  # None when no time & no camera
hits = vss_search(query, timeFilter=None)
if candidates is not None:
    hits = [h for h in hits if h.video_id in candidates]       # skipped when None
hits = [enrich(h) for h in hits]                               # always runs
```

The no-filter path is where the **top-K caveat** (10.3) bites hardest: with nothing to pre-filter
on, VSS ranks the appearance across the *entire* index and returns only its top-K by score, so over
very large archives a true match could rank below K. A time or camera hint both speeds up recall
and improves accuracy by shrinking the candidate set — but neither is required; the system still
answers a bare appearance query.

### 10.2 In-video position vs. real-world time

There are **two different time axes**, and they answer different questions:

| Axis | Field | Meaning | Example question |
|------|-------|---------|------------------|
| **In-video position** | `segment_start` / `segment_end` (seconds) | where inside a clip the match occurs | "red apple **between 1:00-2:00 of the video**" |
| **Real-world capture time** | bridge `capture_start/end` | when the footage was recorded | "person in blue **between 2-3 PM**" |

VSS returns `segment_start/segment_end` natively on every hit, so **in-video position needs no
bridge logic** — just filter the returned segments. Real-world capture time is what 10.1 solves.

To find an appearance **between minute 1 and 2 of a specific video**, search the phrase and keep
hits whose segment overlaps `[60, 120]` seconds:

```bash
curl -s -X POST http://<HOST_IP>:12345/manager/search/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"red apple"}' \
| python3 -c "import sys,json;
d=json.load(sys.stdin); res=d['results'][0]['results']
for r in res:
    m=r['metadata']
    if m['segment_start'] < 120 and m['segment_end'] > 60:
        print(m['video_id'][:8], f\"{m['segment_start']:.0f}-{m['segment_end']:.0f}s\", round(m['relevance_score'],3))"
```

(On the current demo clip `red apple` peaks at `32-40s` with a strong score and tapers off by
`48s`; nothing overlaps the `60-120s` window, so the filter correctly returns no rows.) The bridge
exposes this as an optional `video_pos_start`/`video_pos_end` (seconds) pair on
`/api/v1/lp/recall/search` that post-filters on `segment_start/segment_end`.

Response shape returned to the UI:

```json
{
  "results": [
    {
      "clip_id": "<videoId>",
      "camera_id": "cam-12",
      "area_label": "entrance",
      "capture_start": "2026-06-17T14:05:12Z",
      "capture_end": "2026-06-17T14:06:12Z",
      "segment_start": 18.0,
      "segment_end": 24.0,
      "score": 0.87,
      "tags": ["cam-12", "entrance", "store-001"],
      "clip_url": "/api/v1/lp/recall/clips/<videoId>"
    }
  ]
}
```

### 10.3 Worked example: how pre-filter and post-filter cooperate

Investigator asks: *"person in a **red jacket** near the **entrance**, **yesterday 14:00-15:00**."*
That query has three axes — **appearance** (semantic, VSS), **camera/area** (structured, mapping
DB), and **real-world time** (structured, mapping DB, because VSS only knows upload time).

**Step 1 — Pre-filter (before VSS).** Narrow the universe to clips that *could* match by camera
and capture time. Note the **overlap** test (`start < end AND end > start`), so a clip spanning
14:58-15:02 is still included:

```sql
SELECT video_id FROM clips
WHERE capture_start < '2026-06-18T15:00:00'
  AND capture_end   > '2026-06-18T14:00:00'
  AND camera_id = 'entrance-cam';
-- => ['vid_A', 'vid_B', 'vid_C']   (3 candidates out of ~10,000 clips that day)
```

**Step 2 — VSS appearance search (`timeFilter: null`).** VSS ranks "red jacket" across its *whole*
index and cannot tell which hits are the right camera/time:

```text
vid_B  0.94   red jacket, entrance, yesterday 14:30      <- correct
vid_Q  0.91   red jacket, aisle-3,  last week            <- wrong cam + time
vid_A  0.88   red jacket, entrance, yesterday 14:05      <- correct
vid_Z  0.85   red jacket, loading-dock, 3 days ago       <- wrong cam + time
```

**Step 3 — Post-filter (after VSS).** Intersect VSS hits with the pre-filtered candidate set, then
enrich each survivor from its mapping row:

```python
candidates = {'vid_A', 'vid_B', 'vid_C'}            # from step 1
hits = [h for h in vss_results if h.video_id in candidates]   # drops vid_Q, vid_Z
for h in hits:                                       # enrich for the dashboard
    row = mapping.get(h.video_id)
    h.camera_id, h.area_label = row.camera_id, row.area_label
    h.clip_url, h.capture_ts  = row.clip_url, row.capture_start
# => 1. Front Entrance 14:30 score 0.94 [play]
#    2. Front Entrance 14:05 score 0.88 [play]
```

**Why split into two filters:**

| | Pre-filter | Post-filter |
|---|---|---|
| **When** | before VSS | after VSS |
| **Filters on** | camera + real-world capture time | intersection with candidates + enrichment |
| **Why needed** | shrinks candidates on axes VSS can't do | drops VSS hits that are semantically right but wrong cam/time; attaches `clip_url`/`area`/`capture_ts` for display |

Mental model: **pre-filter = "which clips are eligible?"**, **VSS = "which eligible clips *look*
right?"**, **post-filter = "keep eligible ∩ look-right, then make it presentable."**

> **Top-K caveat.** Because the bridge passes `timeFilter: null`, VSS searches its whole index and
> returns only its top-K by score (`AGGREGATION_INITIAL_K=1000`). In theory a true match could rank
> below K and never reach the post-filter. Mitigations: raise VSS's K, or adopt the optional
> SWLP-owned vector index (Section 15) keyed by capture time so the time filter and vector search
> happen together. For store-scale recall over a bounded time window the pre-filtered candidate set
> is small, so K=1000 is comfortably sufficient in practice.

## 11. Service Design And Folder Structure

A single deployable service, `suspicious-activity-detection/vss-recall-bridge/`. Two planes share
one config and one mapping DB: an **ingest plane** (RTSP/file -> tag -> upload to VSS) and a
**query plane** (HTTP API -> VSS search -> enrich).

```text
suspicious-activity-detection/
└── vss-recall-bridge/
    ├── app/
    │   ├── main.py              # FastAPI app + background ingest workers (lifespan)
    │   ├── config.py           # env + cameras.yaml loader
    │   ├── models.py           # pydantic request/response + Clip/Mapping types
    │   ├── ingest/
    │   │   ├── segmenter.py    # RTSP -> MP4 segments (1 ffmpeg task per camera)
    │   │   ├── file_ingest.py  # pre-existing MP4 / backfill (Section 7.1)
    │   │   ├── remux.py        # ffmpeg -movflags +faststart
    │   │   ├── tagger.py       # camera -> tags (Section 8)
    │   │   └── uploader.py     # VSS upload + embedding trigger + mapping upsert
    │   ├── query/
    │   │   ├── routes.py       # POST /recall/search, GET /recall/clips/{id}
    │   │   └── enrich.py       # join VSS hits -> mapping rows
    │   ├── clients/
    │   │   └── vss_client.py   # httpx client for /manager/* (upload, embed, search)
    │   └── store/
    │       └── mapping.py      # dedicated recall_bridge DB (PG) / SQLite: videoId -> camera/time/tags/clip_url
    ├── configs/
    │   └── cameras.yaml        # camera id -> rtsp_url, area_label, store_id, enabled
    ├── clips/                  # local segment/remux scratch (gitignored)
    ├── Dockerfile
    ├── requirements.txt
    └── README.md
```

Components map to the folders above:

1. **RTSP Segmenter** (`ingest/segmenter.py`): one ffmpeg worker per camera, faststart remux,
   capture-time bookkeeping.
2. **File Ingest** (`ingest/file_ingest.py`): one-shot ingest of an existing MP4 for a camera id.
3. **Tag Mapper** (`ingest/tagger.py` + `configs/cameras.yaml`): camera -> tags.
4. **VSS Client** (`clients/vss_client.py`): upload + embedding trigger + search + retries.
5. **Mapping Store** (`store/mapping.py`): `videoId` -> camera/time/tags/clip URL. SQLite for the
   MVP; a dedicated `recall_bridge` database on VSS's `postgres-service` for shared/HA (never the
   VSS app DB — see 9.1).
6. **Search Proxy API** (`query/routes.py`): `POST /api/v1/lp/recall/search`,
   `GET /api/v1/lp/recall/clips/{clip_id}`.
7. **Reuse the existing dashboard** (`ui/ui_gradio.py`): add a "Recall Search" panel to the
   current FastAPI HTML dashboard (the one already served at `GET /` that polls `/api/data`).
   No new UI app — a query box, camera/area filter, time range, and a results gallery that calls
   the bridge `/api/v1/lp/recall/search` and links each `clip_url`.
8. **Compose wiring** (`docker/docker-compose.yaml`): bridge service on the VSS network.

### 11.1 `configs/cameras.yaml`

This is the single source of truth for the camera -> tag mapping. Each camera is configured once;
the bridge reads this file at startup (`app/config.py`) to drive both ingestion (which streams to
segment) and tagging (what tags each clip gets).

```yaml
# configs/cameras.yaml
cameras:
  cam1:
    rtsp_url: rtsp://localhost:8554/cam1        # live camera                      
    source_file: null
    area_label: lp-camera1                       # human location label, also a search tag
    store_id: store-001
    enabled: true                                # false = skip this camera entirely
    segment_seconds: 60                          # clip length for RTSP segmenting
    extra_tags: []                               # optional static tags appended to every clip

  cam-2:
    rtsp_url: null                               # null = file-ingest only (no RTSP stream)
    source_file: /data/clips/entrance-backfill.mp4   # pre-recorded MP4 to ingest
    area_label: entrance
    store_id: store-001
    enabled: true
    extra_tags: ["front-of-store"]
```

Field reference:

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `<camera id>` (map key) | string | yes | Stable camera id, e.g. `cam1`. Becomes the first/primary search tag and `camera_id` in the mapping DB. |
| `rtsp_url` | string or `null` | yes | Live RTSP URL. `null` -> this camera is **file-ingest only** (uses `source_file`); the segmenter is not started for it. |
| `source_file` | path or `null` | only if `rtsp_url` is null | Path to a pre-recorded MP4 to ingest (the `cam1` / `lp-camera1.mp4` case). |
| `area_label` | string | yes | Human location label (e.g. `entrance`, `aisle-7-cam`). Added as a tag so investigators can filter by area, not just camera id. |
| `store_id` | string | recommended | Store identifier; added as a tag for multi-store deployments. |
| `enabled` | bool | yes | `false` skips the camera (no segmenting, no ingest). |
| `segment_seconds` | int | no (default 60) | RTSP clip length. Lower = fresher index and finer time granularity, more uploads. Ignored for file ingest. |
| `extra_tags` | list[str] | no | Static tags appended to every clip from this camera. |

The resulting tag list per clip is:
`[<camera id>, area_label, store_id, *extra_tags]` (plus an optional `date_bucket`). For `cam1`
that is `["cam1", "lp-camera1", "store-001"]`, matching the Section 12 walkthrough.

## 12. Concrete Walkthrough: File Already On Disk Tagged `cam1`

Scenario: the video already exists at
`/home/intel/sachin/storewide-loss-prevention/scenescape/sample_data/lp-camera1.mp4` and we tag
it `cam1`. No RTSP, no rules — just tag and ingest, then search.

```mermaid
sequenceDiagram
    participant CLI as file_ingest (cam1)
    participant RX as remux
    participant VSS as VSS pipeline-manager
    participant DB as Mapping DB
    participant U as Investigator
    participant API as Bridge Search API

    CLI->>RX: faststart remux lp-camera1.mp4
    RX-->>CLI: lp-camera1.faststart.mp4
    CLI->>VSS: POST /manager/videos (file, name, tags=cam1,lp-camera1,store-001)
    VSS-->>CLI: { videoId }
    CLI->>VSS: POST /manager/videos/search-embeddings/{videoId}
    VSS-->>CLI: { status: success }
    CLI->>DB: upsert videoId -> {cam1, capture_start/end, tags, clip_url}
    U->>API: search "person in a blue shirt" cameras=[cam1]
    API->>VSS: POST /manager/search { query, tags: "cam1", timeFilter }
    VSS-->>API: hits with video_id + score
    API->>DB: join video_id -> mapping
    API-->>U: enriched results (camera, time, clip_url)
```

### 12.1 Ingest the file

Conceptual CLI (implemented by `app/ingest/file_ingest.py`):

```bash
python -m app.ingest.file_ingest \
  --file /home/intel/sachin/storewide-loss-prevention/scenescape/sample_data/lp-camera1.mp4 \
  --camera cam1
```

What it does, step by step:

1. **Resolve tags** from `cameras.yaml` for `cam1` ->
   `tags = ["cam1", "lp-camera1", "store-001"]`.
2. **Capture window**: a file has no live clock, so use the file mtime (or an explicit
   `--capture-start`) as `capture_start`, and `capture_start + duration` (from `ffprobe`) as
   `capture_end`.
3. **Faststart remux** so VSS accepts it:
   ```bash
   ffmpeg -i lp-camera1.mp4 -c copy -movflags +faststart lp-camera1.faststart.mp4
   ```
4. **Upload** to VSS:
   ```bash
   curl -F "video=@lp-camera1.faststart.mp4" \
        -F "name=lp-camera1" \
        -F "tags=cam1,lp-camera1,store-001" \
        http://<HOST_IP>:12345/manager/videos
   # -> { "videoId": "<id>" }
   ```
5. **Trigger embeddings**:
   ```bash
   curl -X POST http://<HOST_IP>:12345/manager/videos/search-embeddings/<id>
   # -> { "status": "success" }
   ```
6. **Upsert mapping row**:
   `<id> -> { camera_id: cam1, area_label: lp-camera1, capture_start, capture_end,
   tags: [cam1, lp-camera1, store-001], clip_url }`.

### 12.2 Search it back

```bash
curl -X POST http://<HOST_IP>:8080/api/v1/lp/recall/search \
  -H 'Content-Type: application/json' \
  -d '{ "query": "person in a blue shirt", "cameras": ["cam1"], "limit": 20 }'
```

The bridge resolves `cam1` -> tag `cam1`, calls VSS `POST /manager/search` with
`tags: "cam1"`, joins each hit's `video_id` back to the mapping row, and returns:

```json
{
  "results": [
    {
      "clip_id": "<id>",
      "camera_id": "cam1",
      "area_label": "lp-camera1",
      "segment_start": 12.0,
      "segment_end": 18.0,
      "score": 0.83,
      "tags": ["cam1", "lp-camera1", "store-001"],
      "clip_url": "/api/v1/lp/recall/clips/<id>"
    }
  ]
}
```

This is the minimum viable path: one file, one tag, one search. Live RTSP cameras use the same
uploader and mapping store; only the source (segmenter vs file) differs.

## 13. Integration With The SWLP Docker Compose

### 13.0 Prerequisite Environment Variables

Export these before starting the VSS search stack (`source setup.sh --search`). They cover the
registry/tag, service credentials, and the embedding/VLM model selection that the stack has no
defaults for:

```bash
export REGISTRY_URL=intel
export TAG=latest
export MINIO_ROOT_USER=minio
export MINIO_ROOT_PASSWORD=minio_pswd
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=postgres
export RABBITMQ_USER=rabbitmq
export RABBITMQ_PASSWORD=rabbitmq
export MULTIMODAL_EMBEDDING_MODEL="CLIP/clip-vit-b-32"
export TEXT_EMBEDDING_MODEL="QwenText/qwen3-embedding-0.6b"
export VLM_TARGET_DEVICE="GPU"
export VLM_MODEL_NAME="OpenVINO/Phi-3.5-vision-instruct-int8-ov"
export ENABLED_WHISPER_MODELS="tiny.en,small.en,medium.en"
export OD_MODEL_NAME="yolov8l-worldv2"
```

Notes for our integration:

- `MULTIMODAL_EMBEDDING_MODEL` (here `CLIP/clip-vit-b-32`) is the **only model variable strictly
  required** for `--search`; it is the embedding model the recall bridge depends on. The same
  model must be used for indexing and query embedding.
- `MINIO_*` and `POSTGRES_*` are mandatory credentials for the upload-storage and
  pipeline-manager-metadata containers we rely on.
- `TEXT_EMBEDDING_MODEL`, `VLM_*`, `ENABLED_WHISPER_MODELS`, `OD_MODEL_NAME`, and `RABBITMQ_*`
  belong to the summary / unified pipelines and are not exercised by clip upload -> embedding ->
  search. Keep them exported (they are harmless) if you run a combined stack; they can be omitted
  for a pure `--search` bring-up.
- These same credentials must match what the `vss-recall-bridge` uses when it talks to the stack.

When the VSS search stack starts, nine containers come up. Not all are needed for our
integration (clip upload -> embedding -> search). Classification:

| VSS container | Role | Needed for integration? |
|---------------|------|-------------------------|
| `pipeline-manager` | upload + embedding-trigger + search API (`/manager/*`) | **Required** — the bridge talks to this |
| `video-search` (search-ms) | runs the semantic search over VDMS | **Required** — backs `POST /search` |
| `vdms-dataprep` | extracts frames + creates embeddings on upload | **Required** — builds the index |
| `vdms-vector-db` | VDMS vector store | **Required** — holds the vectors |
| `multimodal-embedding-serving` | generates image/text embeddings | **Required** — embedding model |
| `minio-service` | object storage for uploaded videos/frames | **Required** — upload target |
| `postgres-service` | pipeline-manager metadata DB | **Required** — pipeline-manager dependency; the bridge also reuses this server for its own **separate** `recall_bridge` database (see 9.1) |
| `nginx` | reverse proxy exposing the `/manager/` prefix | **Optional** — only if the bridge uses `/manager/` URLs; otherwise call `pipeline-manager:3000` directly on the shared network |
| `vss-singleton-ui` | VSS's own React search UI | **Not needed** — we reuse the SWLP dashboard (`ui/ui_gradio.py`) |

So **7 required, 1 optional (`nginx`), 1 droppable (`vss-singleton-ui`)**.

### 13.1 How to wire them in

The VSS services are defined across
`edge-ai-libraries/sample-applications/video-search-and-summarization/docker/compose.*.yaml`
(notably `compose.base.yaml` and `compose.search.yaml`) on the `vs_network`. Two viable options:

1. **Separate stacks, shared network (recommended).** Keep running the VSS stack via
   `source setup.sh --search`. In the SWLP compose, attach the new `vss-recall-bridge` service
   (and nothing else) to the VSS network so it can reach `pipeline-manager` by name:

   ```yaml
   # suspicious-activity-detection/docker/docker-compose.yaml
   services:
     vss-recall-bridge:
       build:
         context: ../vss-recall-bridge
       image: intel/swlp-vss-recall-bridge:${TAG}
       environment:
         VSS_BASE_URL: http://pipeline-manager:3000      # direct, no nginx
         # or via proxy: http://nginx:80/manager
         # dedicated database on VSS's postgres-service (NOT the VSS app DB)
         RECALL_DB_URL: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres-service:5432/recall_bridge
       volumes:
         - vss-recall-clips:/clips
       networks:
         - storewide-lp
         - vs_network

   networks:
     vs_network:
       external: true        # created by the VSS stack

   volumes:
     vss-recall-clips:
   ```

   This keeps the VSS stack independently upgradeable and avoids copying 7 service definitions
   into the SWLP compose. The bridge is the only new SWLP-owned container.

2. **Single merged compose.** Copy the 7 required VSS service definitions (omit `nginx` and
   `vss-singleton-ui`) into the SWLP compose and bring everything up together. More control, but
   you take on maintaining the VSS service config and its env/model variables
   (`MULTIMODAL_EMBEDDING_MODEL`, MinIO/Postgres creds, etc.). Prefer option 1 unless a single
   `docker compose up` is a hard requirement.

### 13.2 Network reachability

- If `VSS_BASE_URL = http://pipeline-manager:3000`, the bridge skips `nginx` entirely and you do
  **not** need to start `nginx` or `vss-singleton-ui`.
- If you keep the `/manager/` prefix (`http://nginx:80/manager`), include `nginx` but you can
  still drop `vss-singleton-ui`.
- The bridge's own search API (`/api/v1/lp/recall/*`) stays on the SWLP side and is what the
  existing dashboard calls — no VSS UI involved.

## 14. Phased Plan

1. **File-ingest MVP**: `file_ingest.py` + remux + VSS upload/embedding + mapping row + search
   proxy, demoed on `lp-camera1.mp4` tagged `cam1` (Section 12).
2. **RTSP**: segmenter + faststart remux + capture-time bookkeeping for one live camera.
3. **Scale**: all recall-enabled cameras, `cameras.yaml`, time-window translation, and a
   "Recall Search" panel added to the existing `ui/ui_gradio.py` dashboard.
4. **Hardening**: retention/cleanup, multi-camera scale, backpressure.

## 15. Optional Future: SWLP-Owned Vector Index

Camera tagging makes a SWLP-owned vector DB (e.g. Qdrant) **unnecessary** for the current query.
Add one later **only** if a hard requirement appears that stock VSS cannot meet:

- Cross-camera **`person_id`** grouping ("same person across cameras").
- **Person-ROI crops** as the embedded unit (VSS embeds whole frames/clips).
- Custom payload filters beyond tags + time.

In that case SWLP would generate person crops, call VSS Multimodal Embedding Serving for vectors
only, and own the index/payload itself. This is intentionally out of scope for the selected
approach.

## 15.1 Performance & Scale

**Target SLA:** a VLM-recall query returns in **< 2 seconds even at 10M stored vectors**
(investigator workflow — the user must never wait on the system).

**Where the time goes.** The query critical path has three stages, and only one is expensive:

| Stage | Work | Typical cost |
|-------|------|--------------|
| Pre-filter | Postgres `SELECT` of candidate `video_id`s by camera + capture time | ~1–5 ms |
| **Appearance search** | **VDMS vector similarity over the index** | **the bottleneck** |
| Post-filter + enrich | intersect with candidates, join mapping rows | ~5–20 ms |

So the entire 2-second budget is effectively a budget on the **VDMS vector search**. Postgres and
the bridge are negligible.

**The scaling mismatch in the current design.** Today the bridge sends `timeFilter: null` and lets
VSS search the **whole** index (`AGGREGATION_INITIAL_K=1000`). The camera/time pre-filter runs in
Postgres *before* the search, but it does **not** shrink the vector search — VDMS still scores the
full index every query, and we discard non-candidates only *afterward*. At small/medium scale
(thousands–hundreds of thousands of vectors) this is comfortably < 2 s. At **10M vectors** it only
holds if VDMS uses an **ANN index (HNSW/IVF)**; a brute-force/flat index will not meet the SLA.

**What the SLA forces.** To *guarantee* < 2 s at 10M, the camera/time filter must happen **inside**
the vector search, not after it — so only the relevant subset is scored. Two options:

| Option | How it meets < 2 s @ 10M | Trade-off |
|--------|--------------------------|-----------|
| **A. Push the filter into VDMS** | Pass a server-side metadata constraint so VDMS scores only the candidate subset, not all 10M | Requires search-ms/VSS to accept a filter on **capture time** (a field we control); VSS today only filters `created_at` = upload time, the wrong axis |
| **B. SWLP-owned ANN index (promotes §15)** | Own a Qdrant/VDMS index with `capture_time` + `camera_id` in the payload and HNSW ANN; filter + search happen together → sub-second at 10M is standard | No longer "stock VSS, unchanged"; SWLP owns indexing for the recall path |

**Decision.** The thin-bridge design (this document) is correct and meets the SLA at MVP scale and
is the right starting point. The **10M / < 2 s** requirement, however, breaks the "VSS alone is
enough" premise: post-hoc filtering cannot guarantee sub-second latency at that scale. When the
deployment approaches that scale, **Option B (SWLP-owned ANN index keyed by capture time +
camera)** is the recommended path, with Option A as a cheaper alternative **iff** the live VDMS
index is already HNSW and can accept a capture-time payload filter. This means §15 is promoted from
"optional future" to **required at 10M scale**.

**Action before committing to a path:** verify the index type VDMS uses in the running stack
(HNSW vs flat) and whether search-ms can pass a payload/metadata filter on a bridge-controlled
capture-time field. That single check decides A vs B.

## 16. Open Questions

### Ingestion / timing

- Clip length vs latency: shorter clips = fresher index and finer time granularity, but more
  uploads/embeddings. Start at 60 s?
- Rely on near-real-time upload for time accuracy, or always translate via the mapping DB (more
  robust; handles backlog/replay)?
- Does `-c copy` segmenting reliably produce VSS-streamable MP4 after faststart remux for our
  camera codecs, or do some cameras need re-encode?

### Tagging / query semantics

- Camera tag vocabulary: per-camera id, area label, store id — which are required vs optional?
- Normalize user-entered location names (`aisle 7`, `aisle7`, `aisle-7-cam`) to one camera tag?
- Timezone for `between 2:00-3:00 PM`: store-local, browser, or explicit in request?
- If a requested camera/area tag is unknown, error, search all, or ask the user to refine?

### Storage / retention / privacy

- Retention for clips, embeddings, and mapping rows: hours, days, configurable per store? Who
  deletes them and on what schedule?
- Who may use investigator recall, and do we need audit logs per query and viewed clip?
- Are appearance attributes (clothing, etc.) acceptable under product/privacy requirements?

### Operations

- Can current hardware handle continuous segmenting + embedding for all cameras, or do we need
  rate limits / a dedicated host?
- Behavior when VSS, pipeline-manager, or a camera is down: buffer clips, skip, or mark pending?
