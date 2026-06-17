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

The bridge translates this to a VSS call:

```text
POST /manager/search
{
  "query": "person in a blue shirt",
  "tags":  "entrance,cam-12",            # camera/area tags
  "timeFilter": { "start": "...Z", "end": "...Z" }
}
```

Then it enriches each VSS hit by joining `video_id` -> bridge mapping to attach `camera_id`,
`area_label`, true `capture_start/end`, and a playable `clip_url`, and returns the enriched,
sorted results to the UI.

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
    │       └── mapping.py      # SQLite/Postgres: videoId -> camera/time/tags/clip_url
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
5. **Mapping Store** (`store/mapping.py`): `videoId` -> camera/time/tags/clip URL (SQLite first).
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
| `postgres-service` | pipeline-manager metadata DB | **Required** — pipeline-manager dependency |
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
         RECALL_DB_PATH: /data/recall.db
       volumes:
         - vss-recall-clips:/clips
         - vss-recall-db:/data
       networks:
         - storewide-lp
         - vs_network

   networks:
     vs_network:
       external: true        # created by the VSS stack

   volumes:
     vss-recall-clips:
     vss-recall-db:
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
