# How to Use the POI Re-identification System

This guide helps you verify deployment, access the application features, and interact with
the POI Re-identification components.

## Accessing the Application

### Application URLs

After deployment, access the system at:

- **POI UI**: [http://localhost:3000](http://localhost:3000)
- **Backend API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alert Service**: [http://localhost:8001](http://localhost:8001)

### Important Notes

- **Initialization**: Wait a few seconds after startup for Redis, MQTT connection, and FAISS
  index to initialize.
- **SceneScape Required**: Ensure Intel® SceneScape and DLStreamer pipelines are running
  before starting the POI system.
- **Recommended Browser**: Use Google Chrome for an optimal WebSocket experience.

## Application Overview

The POI Re-identification system provides five main functionalities accessible through the
React UI and REST API.

### Core Features

#### 1. **POI Enrollment**

Register suspected individuals by uploading one or more reference face images.

- **Multi-Image Support**: Upload 1–5 reference images per POI for robust matching
- **Severity Classification**: Assign `low`, `medium`, or `high` severity levels
- **Automatic Embedding**: Face detection and 256-d embedding generation via OpenVINO™
  happens automatically on upload
- **FAISS Indexing**: Embeddings are immediately indexed in FAISS for real-time matching

**How to Enroll a POI:**

1. Navigate to the POI Management page in the UI
2. Click "Add New POI"
3. Upload 1–5 clear face images of the suspect
4. Set the severity level and add descriptive notes
5. Click "Create" — the system generates embeddings and indexes them

#### 2. **Real-Time Alert Monitoring**

Receive instant notifications when an enrolled POI is detected on any camera.

- **WebSocket Push**: Alerts appear in the UI in real-time via WebSocket connection
- **Multi-Camera Coverage**: Detections from all configured cameras trigger alerts
- **Alert Details**: Each alert includes POI identity, camera ID, confidence score,
  timestamp, and captured thumbnail
- **Dedup Protection**: Same person on the same camera is suppressed for a configurable
  window (default 60 seconds)

**Alert Flow:**

```text
Camera → DLStreamer → MQTT → POI Backend → FAISS Match → Alert Service → UI (WebSocket)
```

#### 3. **Historical Search (Offline Investigation)**

Upload a suspect's image and find all appearances across cameras within a time range.

- **Image-Based Query**: Upload a face image to search against stored movement events
- **Time Range Filter**: Specify start and end timestamps to narrow results
- **Two-Stage Search**: First searches enrolled POI index, then falls back to the
  all-detections index for non-enrolled persons
- **Movement Timeline**: Returns a chronological list of cameras and regions visited
- **Entry/Exit Frames**: Each appearance shows entry and exit frames for visual verification
- **Track Grouping**: Results are grouped by track with best entry/exit similarity scores
- **Zone Dwell Tracking**: Shows zone-level entry/exit times and dwell durations with frame
  evidence
- **Thumbnail Evidence**: Captured thumbnails from matched detections

**How to Perform a Historical Search:**

1. Navigate to the Search page in the UI
2. Upload a clear face image of the person to investigate
3. Set the desired time range (start and end timestamps)
4. Click "Search" — the system generates an embedding and searches the event store
5. Review the timeline of appearances with entry/exit frames, camera IDs, zone dwells,
   and similarity scores

#### 4. **Camera Management**

View and manage camera feeds integrated via Intel® SceneScape.

- **Camera List**: Displays all cameras configured in SceneScape
- **Live Status**: Shows which cameras are actively publishing detections
- **SceneScape Proxy**: Camera list is fetched from the SceneScape API transparently

#### 5. **AI-Powered Analysis (MCP Server)**

Optional integration with LLM and VLM models for advanced analysis.

- **Event Summarization**: Use `llm_summarize_events` to generate natural language summaries
  of POI detection patterns
- **Scene Analysis**: Use `vlm_analyze_scene` to analyze surveillance camera frames for
  suspicious activity
- **OpenVINO Inference**: Generate face embeddings on-demand via MCP tools
- **Claude Desktop Integration**: Connect via MCP for interactive AI-assisted investigation

> **Note:** MCP tools require additional configuration. Set `LLM_BASE_URL`, `VLM_BASE_URL`,
> and optionally `MCP_ALLOW_EXTERNAL_AI=true` for non-local AI endpoints.

## Verifying the Deployment

### Check Service Health

```bash
# Verify all containers are running
make status

# Check backend health
curl http://localhost:8000/api/v1/status
```

The status endpoint returns:

```json
{
  "status": "running",
  "faiss_vectors": 5,
  "mqtt_connected": true
}
```

### Test POI Enrollment via API

```bash
curl -X POST http://localhost:8000/api/v1/poi \
  -F "images=@suspect_photo.jpg" \
  -F "severity=high" \
  -F "description=Shoplifting suspect"
```

### Test Historical Search via API

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -F "image=@suspect_photo.jpg" \
  -F "start_time=2026-01-01T00:00:00Z" \
  -F "end_time=2026-12-31T23:59:59Z"
```

## Additional Resources

- **[Troubleshooting Guide](./troubleshooting.md)** — Resolve common deployment and runtime issues
- **[Getting Started](./get-started.md)** — Complete initial setup requirements
- **[MQTT Pipeline Design](./mqtt-pipeline-design.md)** — Understand the data flow and Redis data model
- **[API Reference](./api-reference.md)** — Explore all REST API endpoints
- **[Build from Source](./get-started/build-from-source.md)** — Build and deploy the application manually
