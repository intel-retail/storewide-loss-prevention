# vss-recall-bridge

Stateless bridge between SWLP cameras and the VSS search stack. It segments each
camera's RTSP stream (or ingests a file) into short MP4 clips, tags each clip with its
camera identity, uploads it to the VSS pipeline-manager **near-real-time**, and triggers
embedding. An optional thin query proxy maps `{cameras, time_start, time_end}` onto VSS's
stateful search.

See [../docs/vlm-recall.md](../docs/vlm-recall.md) (architecture) and
[../docs/vlm-recall-build-spec.md](../docs/vlm-recall-build-spec.md) (this implementation).

The bridge holds **no database and no mapping table** — every field an investigator sees
comes straight from VSS.

## Layout

```
app/
  main.py              # FastAPI app + background ingest workers (lifespan)
  config.py            # env settings + scene-config.yaml camera loader
  models.py            # request/response DTOs
  ingest/              # remux, tagger, pipeline, segmenter, file_ingest
  query/routes.py      # POST /recall/search, GET /recall/clips/{id}
  clients/vss_client.py
tests/
```

The camera list is **not** configured in this repo — it is read from
`scene-config.yaml`, mounted at runtime from the consuming application's `configs/`
(path set by the `SCENE_CONFIG` env, default `configs/scene-config.yaml`).

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Bridge settings come from the top-level stack env; for a standalone run set at
# least VSS_BASE_URL (see ../configs/.env.example):
VSS_BASE_URL=http://localhost:3000 \
  uvicorn app.main:app --host 0.0.0.0 --port 8080
curl localhost:8080/health      # {"status":"ok"}
```

`ffmpeg` must be on `PATH`.

## Ingest one file (demo / backfill)

```bash
python -m app.ingest.file_ingest \
  --file /path/to/lp-camera1.mp4 \
  --camera cam1
```

## Search

```bash
curl -X POST http://localhost:8080/api/v1/lp/recall/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"person in a blue shirt","cameras":["cam1"],"limit":20}'
```

Optional fields: `time_start`/`time_end` (ISO-8601 real-world window, applied server-side
by VSS), `video_pos_start`/`video_pos_end` (in-video seconds filter).

## Configuration

All settings come from environment variables (documented in the top-level
[../configs/.env.example](../configs/.env.example)). Cameras are derived from
SceneScape's [../configs/scene-config.yaml](../configs/scene-config.yaml): every camera
under `scenes[].cameras` is ingested from `RTSP_BASE_URL/<camera-name>` and tagged with
its name plus the scene's zone names/types.

## Tests

```bash
pip install pytest
pytest -q
```

## Docker

```bash
docker build -t intel/swlp-vss-recall-bridge .
```

Attach the container to the VSS `vs_network` (external) and set `VSS_BASE_URL`.
See build-spec §10 / design doc §12.1.
