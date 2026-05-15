# Face-ID Application Configuration

import os

# MQTT
MQTT_BROKER = os.getenv("MQTT_BROKER", "broker.scenescape.intel.com")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_CA_CERT = os.getenv("MQTT_CA_CERT", "/run/secrets/certs/scenescape-ca.pem")
SCENE_UID = os.getenv("SCENE_UID", "")
MQTT_TOPIC_FACE = f"scenescape/data/scene/{SCENE_UID}/face"
MQTT_TOPIC_EVENTS = f"scenescape/event/region/{SCENE_UID}/+/objects"

# Redis
REDIS_HOST = os.getenv("REDIS_HOST", "faceid-redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
APPEARANCE_TTL_DAYS = int(os.getenv("APPEARANCE_TTL_DAYS", "7"))
APPEARANCE_TTL_SECS = APPEARANCE_TTL_DAYS * 86400

# FAISS
FAISS_DIMENSION = 256
FAISS_WATCHLIST_INDEX_PATH = os.getenv("FAISS_WATCHLIST_PATH", "/data/faiss/watchlist.index")
FAISS_HISTORY_INDEX_PATH = os.getenv("FAISS_HISTORY_PATH", "/data/faiss/history.index")
FAISS_ID_MAP_PATH = os.getenv("FAISS_ID_MAP_PATH", "/data/faiss/id_map.json")
FAISS_HISTORY_NLIST = 256      # IVF clusters
FAISS_HISTORY_M = 32           # PQ sub-quantizers
FAISS_HISTORY_NPROBE = 16      # clusters to search
FAISS_HISTORY_TRAIN_SIZE = 5000  # min vectors before IVF training

# Thresholds
WATCHLIST_DISTANCE_THRESHOLD = float(os.getenv("WATCHLIST_THRESHOLD", "40.0"))
SEARCH_DISTANCE_THRESHOLD = float(os.getenv("SEARCH_THRESHOLD", "60.0"))
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "50"))

# Ingestion batching
INGEST_BATCH_INTERVAL_SECS = float(os.getenv("INGEST_BATCH_INTERVAL", "5.0"))

# OpenVINO model paths (for enrollment/search embedding generation)
MODEL_BASE = os.getenv("MODEL_BASE", "/models/intel")
DET_MODEL = f"{MODEL_BASE}/face-detection-0204/FP32/face-detection-0204.xml"
LM_MODEL = f"{MODEL_BASE}/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml"
REID_MODEL = f"{MODEL_BASE}/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml"
INFERENCE_DEVICE = os.getenv("INFERENCE_DEVICE", "CPU")

# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "5555"))

# Alerts
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")
ALERT_LOG_MAX = int(os.getenv("ALERT_LOG_MAX", "10000"))

# Maintenance
MAINTENANCE_HOUR = int(os.getenv("MAINTENANCE_HOUR", "3"))  # 3 AM
