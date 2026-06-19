# vss-recall-bridge — Implementation Build Spec

Status: Ready to build
Companion to: [vlm-recall.md](vlm-recall.md) (architecture & rationale)
Goal: a coding agent (or developer) can build the **complete** `vss-recall-bridge` service from
this file alone. The design doc explains *why*; this file is the exact *what* and *how*.

> Read order: skim [vlm-recall.md](vlm-recall.md) §5–§10 once for context, then implement strictly
> from this spec. Where they ever disagree, **this spec wins** for implementation details.

---

## 0. What you are building (one paragraph)

A FastAPI service named `vss-recall-bridge` with two planes that share one config and one mapping
DB. The **ingest plane** segments each camera's RTSP stream (or ingests a file) into short MP4
clips, tags each clip with its camera identity, uploads it to the stock VSS pipeline-manager,
triggers embedding, and records a mapping row (`video_id → camera, capture time, tags, clip_url`).
The **query plane** exposes `POST /api/v1/lp/recall/search`: it pre-filters candidate clips by
real-world capture time + camera from the mapping DB, runs an appearance-only search in VSS
(`timeFilter: null`), then post-filters VSS hits to that candidate set and enriches them for the
dashboard. The bridge stores **no video bytes** — clips live in VSS MinIO, vectors in VDMS.

---

## 1. Tech stack & dependencies

- Python 3.11+, FastAPI, Uvicorn, httpx (async), Pydantic v2, PyYAML, SQLAlchemy 2.x (async),
  `aiosqlite` (MVP) + `asyncpg` (Postgres), `python-multipart` (uploads), `ffmpeg` (system binary).

`requirements.txt`:

```
fastapi>=0.110
uvicorn[standard]>=0.29
httpx>=0.27
pydantic>=2.6
pydantic-settings>=2.2
pyyaml>=6.0
sqlalchemy>=2.0
aiosqlite>=0.20
asyncpg>=0.29
python-multipart>=0.0.9
tenacity>=8.2
```

`ffmpeg` must be on `PATH` (installed in the Docker image). No GPU needed in the bridge itself.

---

## 2. Folder structure (create exactly this)

```
suspicious-activity-detection/vss-recall-bridge/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, lifespan starts/stops ingest workers
│   ├── config.py           # Settings (env) + cameras.yaml loader
│   ├── models.py           # Pydantic request/response + DTOs
│   ├── db.py               # SQLAlchemy engine/session + Clip ORM model + init
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── segmenter.py    # RTSP -> MP4 segments (1 asyncio task per camera)
│   │   ├── file_ingest.py  # one-shot ingest of an existing MP4
│   │   ├── remux.py        # ffmpeg -movflags +faststart
│   │   ├── tagger.py       # camera -> tag list
│   │   └── pipeline.py     # upload + embed + mapping upsert (shared by both ingest paths)
│   ├── query/
│   │   ├── __init__.py
│   │   ├── routes.py       # POST /recall/search, GET /recall/clips/{id}
│   │   └── enrich.py       # join VSS hits -> mapping rows
│   ├── clients/
│   │   ├── __init__.py
│   │   └── vss_client.py   # httpx client for /manager/* (upload, embed, search)
│   └── store/
│       ├── __init__.py
│       └── mapping.py      # CRUD: prefilter(), upsert(), get(), set_status()
├── configs/
│   └── cameras.yaml
├── clips/                  # scratch for segment/remux (gitignored, ephemeral)
├── tests/
│   ├── test_tagger.py
│   ├── test_mapping_prefilter.py
│   └── test_search_postfilter.py
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## 3. Configuration contract

### 3.1 Environment (`app/config.py`, pydantic-settings)

| Env var | Required | Default | Meaning |
|---------|----------|---------|---------|
| `VSS_BASE_URL` | yes | `http://pipeline-manager:3000` | Base for `/manager/*` (or `http://nginx:80/manager`). |
| `RECALL_DB_URL` | yes | `sqlite+aiosqlite:///./recall.db` | Mapping DB. Prod: `postgresql+asyncpg://USER:PWD@postgres-service:5432/recall_bridge`. |
| `CAMERAS_CONFIG` | no | `configs/cameras.yaml` | Path to camera config. |
| `CLIPS_DIR` | no | `./clips` | Scratch dir for segmenting/remux. |
| `BRIDGE_API_KEY` | yes | — | Static API key required on every `/api/v1/lp/recall/*` request (see §8). |
| `CLIP_URL_TTL_SECONDS` | no | `300` | Expiry for signed clip URLs. |
| `DEFAULT_SEGMENT_SECONDS` | no | `60` | Fallback when a camera omits `segment_seconds`. |
| `SEARCH_TIMEZONE` | no | `UTC` | Timezone used to resolve NL time windows in queries. |
| `HTTP_TIMEOUT_SECONDS` | no | `30` | httpx timeout to VSS. |

`.env.example` must list all of the above with placeholder values.

### 3.2 `configs/cameras.yaml`

Schema exactly as in [vlm-recall.md](vlm-recall.md) §11.1. Loader rules:
- map key = `camera_id` (string, required).
- `rtsp_url` string → start a segmenter task; `null` → file-ingest only (requires `source_file`).
- `enabled: false` → skip entirely.
- `segment_seconds` default = `DEFAULT_SEGMENT_SECONDS`.
- Validate on startup; **fail fast** with a clear error if a camera has neither `rtsp_url` nor
  `source_file`, or an enabled file-ingest camera points at a missing file.

---

## 4. Data model (mapping DB)

### 4.1 ORM (`app/db.py`) — table `clips`

| Column | Type | Notes |
|--------|------|-------|
| `video_id` | `str` PK | VSS `videoId` returned by upload. |
| `camera_id` | `str` indexed | from cameras.yaml. |
| `area_label` | `str` | human location label. |
| `store_id` | `str` nullable | multi-store tag. |
| `capture_start` | `datetime` (UTC, tz-aware) indexed | real-world clip start. |
| `capture_end` | `datetime` (UTC, tz-aware) indexed | real-world clip end. |
| `tags` | `str` (comma-separated) | exact tag string uploaded to VSS. |
| `clip_url` | `str` nullable | resolved playable URL/key for the clip. |
| `segment_key` | `str` UNIQUE | idempotency key: `f"{camera_id}:{capture_start.isoformat()}"`. |
| `status` | `str` | `uploaded` → `embedded` → `failed`. Default `uploaded`. |
| `created_at` | `datetime` server default | row insert time (bookkeeping only). |

Indexes: `(camera_id, capture_start)`, `(capture_start, capture_end)`, unique `segment_key`.

`init_db()` creates the table if missing. For Postgres, the dedicated database `recall_bridge`
must already exist (`CREATE DATABASE recall_bridge;` — see [vlm-recall.md](vlm-recall.md) §9.1);
the bridge creates only its **table**, never touches VSS's tables.

### 4.2 `app/store/mapping.py` function contracts

```python
async def upsert(session, *, video_id, camera_id, area_label, store_id,
                 capture_start, capture_end, tags, segment_key,
                 status="uploaded") -> None
    # INSERT ... ON CONFLICT(segment_key) DO NOTHING/UPDATE. Idempotent.

async def set_status(session, video_id: str, status: str,
                     clip_url: str | None = None) -> None

async def get(session, video_id: str) -> Clip | None

async def prefilter(session, *, time_start, time_end,
                    cameras: list[str] | None) -> set[str] | None
    # Returns set of video_id whose capture window overlaps [time_start, time_end]
    # AND camera_id in cameras (when given).
    # Returns None when BOTH time_start/time_end AND cameras are None
    # (caller treats None as "no candidate filter" -> enrichment only).
```

`prefilter` SQL (optional predicates — matches [vlm-recall.md](vlm-recall.md) §10.1):

```sql
SELECT video_id FROM clips
WHERE (:time_start IS NULL OR capture_start < :time_end)
  AND (:time_end   IS NULL OR capture_end   > :time_start)
  AND (:cameras    IS NULL OR camera_id IN :cameras)
  AND status = 'embedded';
```

> Note `status = 'embedded'`: never return clips that uploaded but failed embedding — they are
> not searchable and would create phantom candidates.

---

## 5. VSS client (`app/clients/vss_client.py`)

Async httpx client. Verified contract (see [vlm-recall.md](vlm-recall.md) §6). Wrap each call in
`tenacity` retry (3 attempts, exponential backoff) for transient 5xx/connection errors.

```python
class VssClient:
    def __init__(self, base_url: str, timeout: float): ...

    async def upload(self, *, file_path: str, name: str, tags: str) -> str:
        # POST {base}/manager/videos  (multipart: video=<mp4>, name, tags=comma-separated)
        # returns videoId

    async def trigger_embedding(self, video_id: str) -> None:
        # POST {base}/manager/videos/search-embeddings/{video_id}  (no body)
        # expect {"status": "success"}

    async def search(self, *, query: str, tags: str | None) -> list[dict]:
        # POST {base}/manager/search/query   (ONE-OFF immediate endpoint)
        # body: {"query": query, "tags": tags, "timeFilter": null}
        # response nesting: d["results"][0]["results"][i]["metadata"]
        # return the flat list of metadata dicts (video_id, segment_start/end,
        # relevance_score, tags, created_at, ...)
```

**Critical details (do not deviate):**
- Search uses `POST /manager/search/query` (immediate), **not** `/manager/search` (async, returns
  empty results first).
- Always send `"timeFilter": null`. The bridge owns the time axis; VSS `created_at` is upload time.
- Parse defensively: `d.get("results", [{}])[0].get("results", [])`, then each item's
  `["metadata"]`.

---

## 6. Ingest plane

### 6.1 `ingest/remux.py`

```python
async def faststart_remux(src: str, dst: str) -> None
    # ffmpeg -y -i src -c copy -movflags +faststart dst
    # if returncode != 0, retry once with re-encode:
    #   ffmpeg -y -i src -c:v libx264 -movflags +faststart dst
    # raise on second failure.
```

### 6.2 `ingest/segmenter.py`

One asyncio task per RTSP camera. Use ffmpeg segment muxer with `strftime` names so capture time
comes from the **filename**, not upload time:

```bash
ffmpeg -rtsp_transport tcp -i "$RTSP_URL" \
  -an -c:v copy -f segment -segment_time $SEG -reset_timestamps 1 \
  -strftime 1 "$CLIPS_DIR/${CAMERA_ID}_%Y%m%dT%H%M%S.mp4"
```

Implementation:
- Launch ffmpeg via `asyncio.create_subprocess_exec`.
- Watch `CLIPS_DIR` for **completed** segments (a file is "done" when the next one appears, or via
  size-stable polling). Process each finished segment exactly once.
- For each finished segment: parse `capture_start` from the filename, compute
  `capture_end = capture_start + segment_seconds`, then hand off to `pipeline.process_clip(...)`.
- On ffmpeg exit/crash: log, backoff, restart the task (camera may be temporarily down).

### 6.3 `ingest/file_ingest.py`

```python
async def ingest_file(camera, source_file: str) -> str
    # capture_start: file mtime (UTC) unless camera config provides an explicit capture_start.
    # capture_end:   capture_start + (probed duration or segment_seconds).
    # -> pipeline.process_clip(...). Returns video_id.
```

### 6.4 `ingest/tagger.py`

```python
def build_tags(camera) -> str
    # returns comma-separated: camera_id, area_label, store_id, *extra_tags
    # (matches vlm-recall.md §8; omit None/empty).
```

### 6.5 `ingest/pipeline.py` (the heart — make it idempotent & crash-safe)

```python
async def process_clip(*, camera, clip_path, capture_start, capture_end) -> str:
    segment_key = f"{camera.camera_id}:{capture_start.isoformat()}"
    # 1. If a row with this segment_key already exists with status in {uploaded, embedded}:
    #    skip (idempotent restart). 
    # 2. faststart_remux(clip_path -> remuxed_path)
    # 3. tags = build_tags(camera)
    # 4. video_id = await vss.upload(file_path=remuxed_path, name=basename, tags=tags)
    # 5. await mapping.upsert(... status="uploaded", segment_key=segment_key, video_id=...)
    # 6. await vss.trigger_embedding(video_id)
    # 7. await mapping.set_status(video_id, "embedded", clip_url=resolve_clip_url(video_id))
    # 8. delete remuxed_path and original segment (bridge stores no video bytes)
    # On any exception after step 4: set_status(video_id, "failed"); keep the temp file for retry;
    #    log and continue (one bad clip must not kill the camera task).
```

**Ordering rule (resolves the consistency risk):** upload → write mapping row (`uploaded`) →
embed → flip to `embedded`. A crash leaves a recoverable state; `prefilter` only returns
`embedded` rows, so half-ingested clips are invisible until complete. `segment_key` uniqueness
prevents double-upload on restart.

---

## 7. Query plane

### 7.1 Request / response models (`app/models.py`)

```python
class RecallSearchRequest(BaseModel):
    query: str                          # appearance text, required
    cameras: list[str] | None = None    # optional camera_id filter
    time_start: datetime | None = None  # optional real-world window start
    time_end: datetime | None = None    # optional real-world window end
    video_pos_start: float | None = None  # optional in-video seconds filter
    video_pos_end: float | None = None
    limit: int = 20

class RecallHit(BaseModel):
    clip_id: str            # video_id
    camera_id: str
    area_label: str
    capture_start: datetime
    capture_end: datetime
    segment_start: float    # in-video seconds
    segment_end: float
    score: float
    tags: list[str]
    clip_url: str

class RecallSearchResponse(BaseModel):
    results: list[RecallHit]
```

### 7.2 `query/routes.py` — `POST /api/v1/lp/recall/search`

Algorithm (matches [vlm-recall.md](vlm-recall.md) §10.1/§10.3):

```python
candidates = await mapping.prefilter(session,
                time_start=req.time_start, time_end=req.time_end, cameras=req.cameras)
tags = ",".join(req.cameras) if req.cameras else None
raw = await vss.search(query=req.query, tags=tags)     # timeFilter always null
hits = raw
if candidates is not None:
    hits = [m for m in hits if m["video_id"] in candidates]
if req.video_pos_start is not None or req.video_pos_end is not None:
    lo = req.video_pos_start or 0
    hi = req.video_pos_end or 1e9
    hits = [m for m in hits if m["segment_start"] < hi and m["segment_end"] > lo]
results = await enrich(session, hits)                  # join mapping rows -> RecallHit
results.sort(key=lambda h: h.score, reverse=True)
return RecallSearchResponse(results=results[:req.limit])
```

### 7.3 `GET /api/v1/lp/recall/clips/{clip_id}`

Returns a signed/expiring URL or proxies the clip bytes from MinIO (see §8). Never expose raw
unauthenticated MinIO URLs.

### 7.4 `query/enrich.py`

```python
async def enrich(session, hits: list[dict]) -> list[RecallHit]
    # For each VSS metadata dict, mapping.get(video_id); attach camera_id, area_label,
    # capture_start/end, clip_url. Drop hits with no mapping row (orphans) and log them.
```

---

## 8. Security (mandatory — do not skip)

- **API auth:** every `/api/v1/lp/recall/*` request must carry `X-API-Key: $BRIDGE_API_KEY`.
  Reject with 401 otherwise. Implement as a FastAPI dependency.
- **Clip access:** `clip_url` must be a short-lived signed URL (`CLIP_URL_TTL_SECONDS`) or an
  authenticated proxy endpoint on the bridge — never a raw public MinIO link.
- **Input validation:** `limit` capped (e.g. ≤ 100); `query` length-bounded; camera ids validated
  against `cameras.yaml`. Reject unknown cameras with 400 (don't silently search all).
- **No secrets in logs.** Never log API keys, DB passwords, or full signed URLs.
- **Least privilege DB:** the bridge's DB user owns only `recall_bridge`; no rights on VSS's DB.

---

## 9. main.py wiring

```python
@asynccontextmanager
async def lifespan(app):
    settings = get_settings()
    await init_db(settings.recall_db_url)
    cameras = load_cameras(settings.cameras_config)
    app.state.vss = VssClient(settings.vss_base_url, settings.http_timeout_seconds)
    tasks = []
    for cam in cameras:
        if not cam.enabled: continue
        if cam.rtsp_url:
            tasks.append(asyncio.create_task(run_segmenter(cam, ...)))
        elif cam.source_file:
            tasks.append(asyncio.create_task(ingest_file(cam, cam.source_file)))
    app.state.ingest_tasks = tasks
    yield
    for t in tasks: t.cancel()
    await app.state.vss.aclose()

app = FastAPI(lifespan=lifespan)
app.include_router(query_router, prefix="/api/v1/lp/recall",
                   dependencies=[Depends(require_api_key)])

@app.get("/health")
async def health(): return {"status": "ok"}
```

---

## 10. Dockerfile

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY configs ./configs
ENV CLIPS_DIR=/clips
VOLUME ["/clips"]
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

Compose wiring: see [vlm-recall.md](vlm-recall.md) §13.1 (attach to `vs_network`, set
`VSS_BASE_URL`, `RECALL_DB_URL`, `BRIDGE_API_KEY`).

---

## 11. Build order (do it in this sequence)

1. **Skeleton:** `config.py`, `models.py`, `db.py`, `main.py` `/health`. Run, hit `/health`.
2. **VSS client + file ingest:** `vss_client.py`, `remux.py`, `tagger.py`, `pipeline.py`,
   `file_ingest.py`. Ingest one local MP4 ([vlm-recall.md](vlm-recall.md) §12); confirm a `clips`
   row appears with `status='embedded'`.
3. **Query plane:** `mapping.prefilter`, `routes.py`, `enrich.py`. Search the ingested clip; get an
   enriched hit back.
4. **Security:** API-key dependency + signed clip URLs.
5. **RTSP segmenter:** `segmenter.py`; ingest a live stream end-to-end.
6. **Dashboard panel:** add "Recall Search" to `ui/ui_gradio.py` calling the bridge.
7. **Dockerfile + compose:** bring up on `vs_network` against the running VSS stack.

---

## 12. Acceptance criteria (definition of done)

- [ ] `GET /health` → `{"status":"ok"}`.
- [ ] File ingest of `lp-camera1.mp4` tagged `cam1` creates exactly one `clips` row, `status`
      transitions `uploaded → embedded`, temp files deleted.
- [ ] Re-running the same ingest does **not** create a duplicate VSS upload or row (idempotent via
      `segment_key`).
- [ ] `POST /api/v1/lp/recall/search {"query":"red apple"}` returns enriched hits with
      `camera_id`, `area_label`, `capture_start/end`, `clip_url`.
- [ ] Query with `time_start/time_end` outside any clip window → empty results.
- [ ] Query with no time and no camera → pure VSS ranking, enrichment only (no crash).
- [ ] Request without `X-API-Key` → 401.
- [ ] Unknown camera id in `cameras` → 400.
- [ ] `clip_url` is signed/expiring or proxied (not a raw MinIO link).
- [ ] RTSP camera produces clips whose `capture_start` matches the segment filename timestamp.
- [ ] Unit tests: `build_tags`, `prefilter` (overlap + optional predicates), post-filter
      intersection.

---

## 13. Risks resolved by this spec (cross-ref)

| Risk (from review) | Resolution in this spec |
|--------------------|-------------------------|
| Mapping ↔ VSS orphan on crash | §6.5 ordering + `status` lifecycle + `segment_key` idempotency; `prefilter` returns only `embedded`. |
| Retention/deletion across MinIO/VDMS/DB | Out of MVP scope; tracked in [vlm-recall.md](vlm-recall.md) §16. Bridge deletes only its temp files. |
| Unauthenticated footage access | §8 API key + signed/proxied clip URLs. |
| Capture-time accuracy | §6.2 filename timestamp (RTSP) / §6.3 mtime or explicit (file); host needs NTP. |
| Embedding lag / eventual consistency | `prefilter` `status='embedded'` gate; recall is over historical footage. |
| Top-K=1000 ceiling | Pre-filter shrinks candidates; documented in [vlm-recall.md](vlm-recall.md) §10.3. |
