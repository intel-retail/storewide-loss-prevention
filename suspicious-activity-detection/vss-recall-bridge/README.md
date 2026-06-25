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
  config.py            # env settings + cameras.yaml loader
  models.py            # request/response DTOs
  security.py          # X-API-Key dependency
  ingest/              # remux, tagger, pipeline, segmenter, file_ingest
  query/routes.py      # POST /recall/search, GET /recall/clips/{id}
  clients/vss_client.py
configs/cameras.yaml
tests/
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # set BRIDGE_API_KEY and VSS_BASE_URL
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
  -H "X-API-Key: $BRIDGE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"query":"person in a blue shirt","cameras":["cam1"],"limit":20}'
```

Optional fields: `time_start`/`time_end` (ISO-8601 real-world window, applied server-side
by VSS), `video_pos_start`/`video_pos_end` (in-video seconds filter).

## Configuration

All settings come from environment variables (see `.env.example`). Cameras are declared in
`configs/cameras.yaml`; an enabled camera needs either an `rtsp_url` (live) or a
`source_file` (file ingest). Startup fails fast on a misconfigured or missing source.

## Tests

```bash
pip install pytest
pytest -q
```

## Docker

```bash
docker build -t intel/swlp-vss-recall-bridge .
```

Attach the container to the VSS `vs_network` (external) and set `VSS_BASE_URL` +
`BRIDGE_API_KEY`. See build-spec §10 / design doc §12.1.
