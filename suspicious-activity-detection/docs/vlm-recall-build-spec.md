# vss-recall-bridge — Implementation Build Spec

Status: Ready to build
Companion to: [vlm-recall.md](vlm-recall.md) (architecture & rationale)
Goal: a coding agent (or developer) can build the **complete** `vss-recall-bridge` service from
this file alone. The design doc explains *why*; this file is the exact *what* and *how*.

> Read order: skim [vlm-recall.md](vlm-recall.md) §4–§9 once for context, then implement strictly
> from this spec. Where they ever disagree, **this spec wins** for implementation details.
>
> VSS now ships **R1** (absolute `timeFilter:{start,end}` applied on search — verified A/B) and
> **subset/OR tag matching** (see [vlm-recall.md](vlm-recall.md) §3.1, §3.3). Capture time is
> handled by **near-real-time ingest + query window padding** (§4, §7), so **no new ingest field
> (R2) is required** for live recall; R2 is only needed for bulk backfill of old footage.

---

## 0. What you are building (one paragraph)

A **stateless** FastAPI service named `vss-recall-bridge`. The always-on **ingest plane** segments
each camera's RTSP stream (or ingests a file) into short MP4 clips, tags each clip with its camera
identity, and uploads it to the stock VSS pipeline-manager **near-real-time** (wait for the chunk,
then upload), then triggers embedding. An **optional thin query proxy** exposes
`POST /api/v1/lp/recall/search`: it maps `{cameras, time_start, time_end}` to
`{tags, timeFilter:{start,end}}` (padding the window by one chunk, §7), submits VSS's **stateful**
search, polls for the result, and returns the hits as-is. There is **no mapping DB, no pre/post-
filter, and no enrichment** — VSS applies the tag + time filter server-side and every field the
dashboard needs (`video_id`, `tags`, `created_at`, `segment_start/end`, `score`, `video_url`)
comes straight off the VSS hit. The bridge stores **nothing** — clips live in VSS MinIO, vectors
in VDMS.

---

## 1. Tech stack & dependencies

- Python 3.11+, FastAPI, Uvicorn, httpx (async), Pydantic v2, PyYAML, `python-multipart`
  (uploads), `ffmpeg` (system binary). **No database libraries** — the bridge is stateless.

`requirements.txt`:

```
fastapi>=0.110
uvicorn[standard]>=0.29
httpx>=0.27
pydantic>=2.6
pydantic-settings>=2.2
pyyaml>=6.0
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
│   ├── models.py           # Pydantic request/response DTOs
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── segmenter.py    # RTSP -> MP4 segments (1 asyncio task per camera)
│   │   ├── file_ingest.py  # one-shot ingest of an existing MP4
│   │   ├── remux.py        # ffmpeg -movflags +faststart
│   │   ├── tagger.py       # camera -> tag list
│   │   └── pipeline.py     # remux + upload (near-real-time) + embed (shared by both ingest paths)
│   ├── query/
│   │   ├── __init__.py
│   │   └── routes.py       # optional thin proxy: POST /recall/search -> VSS stateful search
│   └── clients/
│       ├── __init__.py
│       └── vss_client.py   # httpx client for /manager/* (upload, embed, stateful search)
├── configs/
│   └── cameras.yaml
├── clips/                  # scratch for segment/remux (gitignored, ephemeral)
├── tests/
│   ├── test_tagger.py
│   └── test_search_proxy.py
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
| `CAMERAS_CONFIG` | no | `configs/cameras.yaml` | Path to camera config. |
| `CLIPS_DIR` | no | `./clips` | Scratch dir for segmenting/remux. |
| `BRIDGE_API_KEY` | yes | — | Static API key required on every `/api/v1/lp/recall/*` request (see §8). |
| `SEARCH_POLL_SECONDS` | no | `1.0` | Interval between `GET /manager/search/{queryId}` polls. |
| `SEARCH_POLL_TIMEOUT_SECONDS` | no | `30` | Max time to poll a `queryId` before giving up. |
| `DEFAULT_SEGMENT_SECONDS` | no | `60` | Fallback when a camera omits `segment_seconds`. |
| `WINDOW_PAD_SECONDS` | no | `0` | Pad applied to the query window (use one chunk length for the §3.2 interim). |
| `SEARCH_TIMEZONE` | no | `UTC` | Timezone used to resolve NL time windows in queries. |
| `HTTP_TIMEOUT_SECONDS` | no | `30` | httpx timeout to VSS. |

`.env.example` must list all of the above with placeholder values.

### 3.2 `configs/cameras.yaml`

Schema exactly as in [vlm-recall.md](vlm-recall.md) §10.1. Loader rules:
- map key = `camera_id` (string, required).
- `rtsp_url` string → start a segmenter task; `null` → file-ingest only (requires `source_file`).
- `enabled: false` → skip entirely.
- `segment_seconds` default = `DEFAULT_SEGMENT_SECONDS`.
- Validate on startup; **fail fast** with a clear error if a camera has neither `rtsp_url` nor
  `source_file`, or an enabled file-ingest camera points at a missing file.

---

## 4. Capture time (no database, no ingest field)

The bridge keeps **no mapping table** and sends **no capture timestamp** to VSS. Instead it relies
on **near-real-time upload**: the segmenter waits for each chunk to close, then uploads
immediately, so VSS's `created_at` (upload time) tracks the clip's real capture time within one
chunk length. The query proxy absorbs that fixed offset by padding the requested window by one
chunk (`WINDOW_PAD_SECONDS`, §7). Nothing about a clip needs to be persisted by the bridge.

The `strftime` segment filename still carries a timestamp, used only for clip **naming/debug** —
it is not uploaded as metadata.

> Optional future (R2): bulk **backfill/replay** of old footage breaks the upload≈capture
> assumption (`created_at` = the moment you import, off by days). Accurate time filtering for that
> case needs VSS to accept a caller-supplied capture timestamp at ingest (R2,
> [vlm-recall.md](vlm-recall.md) §3.1). Out of scope for the live MVP.

---

## 5. VSS client (`app/clients/vss_client.py`)

Async httpx client. Verified contract (see [vlm-recall.md](vlm-recall.md) §5). Wrap each call in
`tenacity` retry (3 attempts, exponential backoff) for transient 5xx/connection errors.

```python
class VssClient:
    def __init__(self, base_url: str, timeout: float,
                 poll_seconds: float, poll_timeout: float): ...

    async def upload(self, *, file_path: str, name: str, tags: str) -> str:
        # POST {base}/manager/videos  (multipart: video=<mp4>, name, tags=comma-separated)
        # returns videoId

    async def trigger_embedding(self, video_id: str) -> None:
        # POST {base}/manager/videos/search-embeddings/{video_id}  (no body)
        # expect {"status": "success"}

    async def search(self, *, query: str, tags: str | None,
                     time_start: datetime | None,
                     time_end: datetime | None) -> list[dict]:
        # 1. POST {base}/manager/search  (STATEFUL endpoint)
        #    body: {"query": query, "tags": tags,
        #           "timeFilter": {"start": time_start, "end": time_end}}  # omit when both None
        #    -> {"queryId": "..."}
        # 2. poll GET {base}/manager/search/{queryId} every poll_seconds until results ready
        #    (or poll_timeout). Return the flat list of hit metadata dicts
        #    (video_id, segment_start/end, relevance_score, tags, created_at, video_url, ...).
```

**Critical details (do not deviate):**
- Search uses the **stateful** `POST /manager/search` (returns `queryId`) + poll
  `GET /manager/search/{queryId}`. The one-off `POST /manager/search/query` **ignores both tags
  and `timeFilter`** and returns everything — do not use it.
- Send `timeFilter` as an **absolute** `{start, end}` (R1). VSS matches it against `created_at`
  (upload time); near-real-time ingest + the query-side chunk padding (§7) keep that aligned with
  capture time. Omit `timeFilter` when the caller gives no window.
- `tags` is a **subset/OR match**: a clip is returned when it shares **any** requested tag, so a
  clip tagged `aisle1,aisle3` matches a query of just `aisle1` (see
  [vlm-recall.md](vlm-recall.md) §3.3). Send the requested camera/area tags; no need to reproduce a
  clip's full set.
- Parse the polled response defensively for the hit list and each item's metadata.

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

One asyncio task per RTSP camera. Use ffmpeg segment muxer with `strftime` names so each clip is
named by its start time; upload happens as soon as the segment closes (near-real-time), so
`created_at` tracks capture time (§4):

```bash
ffmpeg -rtsp_transport tcp -i "$RTSP_URL" \
  -an -c:v copy -f segment -segment_time $SEG -reset_timestamps 1 \
  -strftime 1 "$CLIPS_DIR/${CAMERA_ID}_%Y%m%dT%H%M%S.mp4"
```

Implementation:
- Launch ffmpeg via `asyncio.create_subprocess_exec`.
- Watch `CLIPS_DIR` for **completed** segments (a file is "done" when the next one appears, or via
  size-stable polling). Process each finished segment exactly once.
- For each finished segment: hand it off to `pipeline.process_clip(...)` immediately (the filename
  timestamp is kept for the clip `name`/debug only).
- On ffmpeg exit/crash: log, backoff, restart the task (camera may be temporarily down).

### 6.3 `ingest/file_ingest.py`

```python
async def ingest_file(camera, source_file: str) -> str
    # one-shot remux + upload + embed of an existing MP4 (demos / backfill).
    # -> pipeline.process_clip(...). Returns video_id.
    # NOTE: created_at = upload time, so time filtering on backfilled *old* footage is only
    # accurate with R2 (see §4). Fine for demos and recently recorded files.
```

### 6.4 `ingest/tagger.py`

```python
def build_tags(camera) -> str
    # returns comma-separated: camera_id, area_label, store_id, *extra_tags
    # (matches vlm-recall.md §7; omit None/empty).
```

### 6.5 `ingest/pipeline.py` (the heart — keep it crash-safe)

```python
async def process_clip(*, camera, clip_path) -> str:
    # 1. faststart_remux(clip_path -> remuxed_path)
    # 2. tags = build_tags(camera)
    # 3. video_id = await vss.upload(file_path=remuxed_path, name=basename, tags=tags)
    # 4. await vss.trigger_embedding(video_id)
    # 5. delete remuxed_path and original segment (bridge stores no video bytes, no DB row)
    # On any exception: log, keep the temp file for retry, and continue
    #    (one bad clip must not kill the camera task). Return video_id.
```

**No local state.** The bridge writes nothing for the clip — tags and the playback URL live in VSS
after upload, and `created_at` (upload time) is the time reference (§4). Because there is no
mapping row, ingest is intentionally simple: remux → upload → embed → delete temp files.

> Idempotency note: without a DB the bridge cannot dedup a re-processed segment on its own. The
> segmenter processes each finished segment exactly once (§6.2), so duplicates only arise on a
> crash mid-upload. If exactly-once matters for your deployment, dedup on VSS side by clip `name`
> (the segment filename) before uploading; otherwise a rare duplicate clip is acceptable and
> harmless to search.

---

## 7. Query plane

### 7.1 Request / response models (`app/models.py`)

```python
class RecallSearchRequest(BaseModel):
    query: str                          # appearance text, required
    cameras: list[str] | None = None    # optional camera_id filter -> tags
    time_start: datetime | None = None  # optional real-world window start
    time_end: datetime | None = None    # optional real-world window end
    video_pos_start: float | None = None  # optional in-video seconds filter
    video_pos_end: float | None = None
    limit: int = 20

class RecallHit(BaseModel):            # 1:1 with a VSS hit, no enrichment
    video_id: str
    tags: list[str]
    capture_time: datetime | None
    segment_start: float    # in-video seconds
    segment_end: float
    score: float
    video_url: str

class RecallSearchResponse(BaseModel):
    results: list[RecallHit]
```

### 7.2 `query/routes.py` — `POST /api/v1/lp/recall/search`

Algorithm (matches [vlm-recall.md](vlm-recall.md) §9). The proxy is a pure mapping onto VSS's
stateful search — no candidate set, no post-filter, no DB:

```python
tags = build_tag_query(req.cameras)          # comma-separated tag string (subset/OR), or None
start, end = pad_window(req.time_start, req.time_end, WINDOW_PAD_SECONDS)
raw = await vss.search(query=req.query, tags=tags,
                       time_start=start, time_end=end)   # absolute timeFilter (R1)
hits = raw
if req.video_pos_start is not None or req.video_pos_end is not None:
    lo = req.video_pos_start or 0
    hi = req.video_pos_end or 1e9
    hits = [m for m in hits if m["segment_start"] < hi and m["segment_end"] > lo]
results = [RecallHit(**to_hit(m)) for m in hits]   # straight field map from VSS metadata
results.sort(key=lambda h: h.score, reverse=True)
return RecallSearchResponse(results=results[:req.limit])
```

The only client-side filter is the optional in-video position (`segment_start/end`), which VSS
returns natively. Camera (tags) and real-world time (`timeFilter`) are applied **server-side** by
VSS.

### 7.3 `GET /api/v1/lp/recall/clips/{clip_id}`

Returns the clip's playback URL. Prefer proxying the bytes (or returning a short-lived signed URL)
rather than exposing the raw MinIO URL VSS reports — see §8.

### 7.4 No enrichment module

There is **no `enrich.py`** and no mapping lookup: every field in `RecallHit` is copied directly
from the VSS hit metadata. `build_tag_query` (cameras -> comma-separated tag string, subset/OR
matched by VSS) and `pad_window` (apply `WINDOW_PAD_SECONDS`) are small pure helpers in `routes.py`.

---

## 8. Security (mandatory — do not skip)

- **API auth:** every `/api/v1/lp/recall/*` request must carry `X-API-Key: $BRIDGE_API_KEY`.
  Reject with 401 otherwise. Implement as a FastAPI dependency.
- **Clip access:** prefer proxying the clip bytes through an authenticated bridge endpoint (or
  returning a short-lived signed URL) rather than handing the raw MinIO URL straight to the
  browser.
- **Input validation:** `limit` capped (e.g. ≤ 100); `query` length-bounded; camera ids validated
  against `cameras.yaml`. Reject unknown cameras with 400 (don't silently search all).
- **No secrets in logs.** Never log API keys or full signed URLs.
- **Least privilege:** the bridge holds no database and no credentials beyond `VSS_BASE_URL` +
  `BRIDGE_API_KEY`; it only talks to the VSS pipeline-manager.

---

## 9. main.py wiring

```python
@asynccontextmanager
async def lifespan(app):
    settings = get_settings()
    cameras = load_cameras(settings.cameras_config)
    app.state.vss = VssClient(settings.vss_base_url, settings.http_timeout_seconds,
                              settings.search_poll_seconds, settings.search_poll_timeout_seconds)
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

Compose wiring: see [vlm-recall.md](vlm-recall.md) §12.1 (attach to `vs_network`, set
`VSS_BASE_URL`, `BRIDGE_API_KEY`). No database service or `RECALL_DB_URL` — the bridge is
stateless.

---

## 11. Build order (do it in this sequence)

1. **Skeleton:** `config.py`, `models.py`, `main.py` `/health`. Run, hit `/health`.
2. **VSS client + file ingest:** `vss_client.py`, `remux.py`, `tagger.py`, `pipeline.py`,
   `file_ingest.py`. Ingest one local MP4 ([vlm-recall.md](vlm-recall.md) §11); confirm the clip
   uploads with its capture time and embedding succeeds.
3. **Query proxy:** `routes.py` (`build_tag_query`, `pad_window`). Search the ingested clip; get
   a VSS hit back, fields mapped straight through.
4. **Security:** API-key dependency + proxied/signed clip access.
5. **RTSP segmenter:** `segmenter.py`; ingest a live stream end-to-end.
6. **Dashboard panel:** add "Recall Search" to `ui/ui_gradio.py` calling the bridge.
7. **Dockerfile + compose:** bring up on `vs_network` against the running VSS stack.

---

## 12. Acceptance criteria (definition of done)

- [ ] `GET /health` → `{"status":"ok"}`.
- [ ] File ingest of `lp-camera1.mp4` tagged `cam1` uploads exactly one clip to VSS, embedding
      returns success, temp files deleted, **no local state written**.
- [ ] `POST /api/v1/lp/recall/search {"query":"red apple"}` returns hits whose fields
      (`video_id`, `tags`, `created_at`, `segment_start/end`, `score`, `video_url`) come
      straight from VSS.
- [ ] Query with `time_start/time_end` (padded by one chunk) outside the clip's window → empty
      results (absolute `timeFilter` applied server-side, R1).
- [ ] Query with no time and no camera → VSS ranking returned as-is (no crash).
- [ ] Multi-tag query uses **subset/OR** matching (a single tag of a multi-tagged clip returns
      that clip).
- [ ] Request without `X-API-Key` → 401.
- [ ] Unknown camera id in `cameras` → 400.
- [ ] Clip playback is proxied/signed, not a raw MinIO link.
- [ ] RTSP camera uploads clips whose capture time matches the segment filename timestamp.
- [ ] Unit tests: `build_tag_query` (comma-separated tag string), `pad_window`, in-video position
      filter.

---

## 13. Risks resolved by this spec (cross-ref)

| Risk (from review) | Resolution in this spec |
|--------------------|-------------------------|
| Mapping ↔ VSS orphan on crash | **Eliminated** — no mapping DB; the bridge holds no state to fall out of sync. |
| Retention/deletion across MinIO/VDMS | Out of MVP scope; tracked in [vlm-recall.md](vlm-recall.md) §15. Bridge deletes only its temp files. |
| Unauthenticated footage access | §8 API key + proxied/signed clip access. |
| Capture-time accuracy | Near-real-time upload makes `created_at` ≈ capture time (within one chunk); query padding (§7) absorbs it. Backfill of old footage needs R2 (§4). Host needs NTP. |
| Embedding lag / eventual consistency | Recall is over historical footage; a just-uploaded clip becomes searchable once VSS finishes embedding. |
| Absolute time filter / multi-tag matching | **Resolved** — R1 (absolute `timeFilter`) and subset/OR tag matching are live and verified; §5 + [vlm-recall.md](vlm-recall.md) §3.1/§3.3. |
