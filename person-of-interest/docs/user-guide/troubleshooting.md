# Troubleshooting

This page provides comprehensive support and troubleshooting information for the POI
Re-identification system. It is divided into the following sections:

- [Common Issues](#common-issues): General troubleshooting steps for resolving issues like
  container failures, MQTT connectivity, and FAISS errors.
- [Troubleshooting Docker Deployments](#troubleshooting-docker-deployments): Steps to address
  problems specific to Docker deployments.
- [SceneScape Integration Issues](#scenescape-integration-issues): Issues related to MQTT
  topics, embedding alignment, and camera connectivity.

If you encounter any problems not addressed here, check the
[GitHub Issues](https://github.com/sainijit/storewide-loss-prevention-retail/issues) board.

## Common Issues

### 1. Containers Not Starting

- **Issue**: The application containers fail to start.
- **Solution**:

  ```bash
  docker ps -a
  docker logs poi-backend
  docker logs poi-alert-service
  ```

  Check the logs for errors. Common causes:
  - Redis not healthy (check `poi-redis` container)
  - Missing `.env` file (run `make init-env`)
  - Port conflicts with other services

### 2. No Alerts Being Generated

- **Issue**: POIs are enrolled but no alerts appear in the UI.
- **Solution**:
  - Verify MQTT connectivity: check `MQTT_HOST` and `MQTT_PORT` in `.env`
  - Verify SceneScape DLStreamer is publishing to `scenescape/data/camera/+`
  - Check the similarity threshold: lower `SIMILARITY_THRESHOLD` if matches are too strict
  - Verify face embeddings are present in MQTT messages (face sub_objects required)
  - Check backend logs: `make logs | grep "poi.consumer"`

### 3. POI Enrollment Fails

- **Issue**: Creating a POI returns an error or no embeddings are generated.
- **Solution**:
  - Ensure uploaded images contain clearly visible faces
  - Check OpenVINO™ model files exist at `MODEL_BASE` path (`/models/intel/`)
  - Verify the `face-detection-retail-0004` and `face-reidentification-retail-0095` models
    are downloaded
  - Check backend logs: `docker logs poi-backend | grep "embedding"`

### 4. Historical Search Returns Empty Results

- **Issue**: Searching with an image returns no matches.
- **Solution**:
  - Ensure movement events exist in Redis (events have a 7-day TTL)
  - Verify the time range covers the period when the person was detected
  - Lower the `SIMILARITY_THRESHOLD` if the query image differs significantly from
    stored embeddings
  - Check FAISS index count: `curl http://localhost:8000/api/v1/status`

### 5. WebSocket Alerts Not Reaching UI

- **Issue**: Backend shows matches in logs but the UI does not display alerts.
- **Solution**:
  - Ensure the alert service is running: `docker logs poi-alert-service`
  - Verify WebSocket connection in the browser developer console
  - Check that `DELIVERY_HANDLERS` includes `websocket` in the alert service config
  - Clear browser cache and reload the UI

### 6. High False Positive Rate

- **Issue**: Too many incorrect POI match alerts.
- **Solution**:
  - Increase `SIMILARITY_THRESHOLD` (e.g., from `0.6` to `0.70` or `0.75`)
  - Ensure POI enrollment images are high quality with clear, frontal face views
  - Increase `FACE_CONFIDENCE_THRESHOLD` in the MQTT consumer (default: `0.80`)
  - Enroll multiple reference images per POI for more robust matching

### 8. Search Returns Wrong Person (Track ID Reuse)

- **Issue**: Historical search returns appearances of a different person mixed with the correct one.
- **Solution**:
  - This is caused by DLStreamer reusing integer track IDs across different physical persons
  - The track purity filter (enabled by default) should catch this — it filters tracks where
    < 40% of events belong to the queried POI
  - If false positives persist, check that the `detection_embeddings_per_track` config is
    set to 5 (more embeddings per track improves matching)
  - Verify face quality: poor lighting or angles produce weak embeddings

### 9. Non-Enrolled Person Not Found in Search

- **Issue**: Searching for a person who is NOT enrolled as a POI returns no results, even
  though the person appears in video.
- **Solution**:
  - Ensure the detection index is populated — check `GET /api/v1/status` for detection
    vector count
  - The detection index stores embeddings from DLStreamer runtime, which uses different
    preprocessing than enrollment. Similarity scores may be lower (0.3-0.5 range)
  - Lower `SIMILARITY_THRESHOLD` if needed (default is 0.6)
  - The system uses a two-stage search: Stage 1 (POI index) uses the same model as
    enrollment and works well. Stage 2 (detection index) has a cross-domain gap

### 10. UI Crashes on Search Results

- **Issue**: The UI crashes with `TypeError: undefined.toLocaleString()` when viewing search results.
- **Solution**:
  - Update to the latest UI code which includes null-safe rendering for search stats
  - Rebuild the UI: `docker compose build ui && docker compose up -d ui`

### 11. FAISS Index Corruption

- **Issue**: Backend crashes on startup with FAISS errors.
- **Solution**:

  ```bash
  # Stop services
  make down

  # Remove FAISS volume and re-enroll POIs
  docker volume rm person-of-interest_faiss-data

  # Restart
  make up
  ```

  > **Note:** This removes all enrolled POIs. Re-enroll after restart.

## Troubleshooting Docker Deployments

### 1. Port Conflicts

- **Issue**: Port conflicts with other running applications.
- **Solution**: Update the `ports` section in `docker-compose.yml`:

  ```yaml
  # Change backend port from 8000 to 8080
  ports:
    - "8080:8000"
  ```

### 2. Docker Network Issues

- **Issue**: POI backend cannot connect to SceneScape MQTT broker.
- **Solution**: Ensure the `storewide-lp` external network exists:

  ```bash
  docker network create storewide-lp
  ```

  Verify SceneScape containers are on the same network.

### 3. Reset Application

Follow these steps to reset the application to the initial state:

```bash
# Stop all services
make down

# Remove all data volumes
docker volume rm person-of-interest_redis-data person-of-interest_faiss-data person-of-interest_upload-data

# Restart fresh
make up
```

## SceneScape Integration Issues

### 1. MQTT Messages Not Received

- **Issue**: Backend logs show no incoming MQTT events.
- **Solution**:
  - Verify MQTT broker is accessible: `mosquitto_sub -h <MQTT_HOST> -t "scenescape/data/camera/#"`
  - Check DLStreamer pipeline is running in SceneScape
  - Ensure camera feeds are active and producing detections
  - Verify `MQTT_CA_CERT` is set if the broker requires TLS

### 2. Embedding Dimension Mismatch

- **Issue**: FAISS search returns unexpected results or errors.
- **Solution**:
  - Verify `FAISS_DIMENSION=256` matches the re-identification model output
  - Ensure both enrollment (OpenVINO™) and runtime (DLStreamer) use
    `face-reidentification-retail-0095` — they must be the same model to share the
    embedding space

### 3. Camera List Empty

- **Issue**: The camera list API returns no cameras.
- **Solution**:
  - Verify `SCENESCAPE_API_URL` and `SCENESCAPE_API_TOKEN` are correctly set in `.env`
  - Test the SceneScape API directly:

    ```bash
    curl -k https://<scenescape-host>/api/v1/cameras \
      -H "Authorization: Token <api-token>"
    ```
