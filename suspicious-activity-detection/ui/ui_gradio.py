
import gradio as gr
import pandas as pd
import requests
import json
import os
import time
import threading
import base64
import io
from collections import deque

import paho.mqtt.client as mqtt
from PIL import Image, ImageDraw, ImageFont

# Use Docker service name for container-to-container communication
LP_BASE_URL = os.environ.get("LP_BASE_URL", "http://storewide-loss-prevention:8082")
ZONES_API = f"{LP_BASE_URL}/api/v1/lp/zones"
SESSIONS_API = f"{LP_BASE_URL}/api/v1/lp/sessions?include_pending=true"
ALERTS_API = f"{LP_BASE_URL}/api/v1/lp/alerts"
ZONE_CONFIG = os.environ.get("ZONE_CONFIG", "/app/zone_config.json")

# MQTT config for real-time alerts
MQTT_HOST = os.environ.get("MQTT_HOST", "broker.scenescape.intel.com")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_ALERT_TOPIC = os.environ.get("MQTT_ALERT_TOPIC", "alerts/#")

# MQTT topics for live video with detections
MQTT_CAMERA_TOPIC = os.environ.get("MQTT_CAMERA_TOPIC", "scenescape/image/camera/+")
MQTT_SCENE_TOPIC = os.environ.get("MQTT_SCENE_TOPIC", "scenescape/regulated/scene/+")

MAX_RETRIES = 5
RETRY_DELAY = 3  # seconds

# Thread-safe alert store fed by MQTT
_mqtt_alerts: deque = deque(maxlen=500)
_mqtt_lock = threading.Lock()

# Thread-safe live video frame + detections
_latest_frame = {}      # {camera_id: base64_jpeg}
_latest_detections = []  # list of objects from regulated scene data
_video_lock = threading.Lock()


def _on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe(MQTT_ALERT_TOPIC, qos=1)
        client.subscribe(MQTT_CAMERA_TOPIC, qos=0)
        client.subscribe(MQTT_SCENE_TOPIC, qos=0)
        print(f"[MQTT] Subscribed to {MQTT_ALERT_TOPIC}, {MQTT_CAMERA_TOPIC}, {MQTT_SCENE_TOPIC}")
    else:
        print(f"[MQTT] Connect failed, rc={rc}")


def _on_mqtt_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
    except Exception as e:
        print(f"[MQTT] Failed to parse message on {msg.topic}: {e}")
        return

    if msg.topic.startswith("scenescape/image/camera/"):
        camera_id = msg.topic.split("/")[-1]
        with _video_lock:
            _latest_frame[camera_id] = payload.get("image", "")
        global _frame_dirty
        _frame_dirty = True
    elif msg.topic.startswith("scenescape/regulated/scene/"):
        with _video_lock:
            _latest_detections.clear()
            for obj in payload.get("objects", []):
                _latest_detections.append(obj)
        _frame_dirty = True
    elif msg.topic.startswith("alerts"):
        with _mqtt_lock:
            _mqtt_alerts.appendleft(payload)


def _start_mqtt_listener():
    """Connect to MQTT broker with automatic reconnect on failure."""
    backoff = 1
    while True:
        client = mqtt.Client()
        client.on_connect = _on_mqtt_connect
        client.on_message = _on_mqtt_message
        try:
            client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
            backoff = 1  # reset on successful connect
            client.loop_forever()
        except Exception as e:
            print(f"[MQTT] Connection error: {e} — retrying in {backoff}s")
        finally:
            try:
                client.disconnect()
            except Exception:
                pass
        time.sleep(backoff)
        backoff = min(backoff * 2, 30)  # exponential backoff, cap at 30s


# Start MQTT listener in background thread
_mqtt_thread = threading.Thread(target=_start_mqtt_listener, daemon=True)
_mqtt_thread.start()


def api_get_with_retry(url, retries=MAX_RETRIES, delay=RETRY_DELAY):
    """GET request with retry logic for backend startup delays."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return resp
            if attempt < retries:
                time.sleep(delay)
        except Exception:
            if attempt < retries:
                time.sleep(delay)
            else:
                raise
    return resp  # return last response even if not 200


# Color palette for tracked objects
_BBOX_COLORS = [
    (0, 255, 0), (255, 100, 0), (0, 200, 255), (255, 0, 150),
    (200, 200, 0), (150, 0, 255), (0, 255, 150), (255, 200, 0),
]

# Pre-load font once
try:
    _FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
except Exception:
    _FONT = ImageFont.load_default()


def _get_color(obj_id):
    return _BBOX_COLORS[hash(obj_id) % len(_BBOX_COLORS)]


# Pre-render annotated frame in MQTT thread to avoid work during poll
_rendered_frame = None
_rendered_lock = threading.Lock()
_frame_dirty = True  # Flag: new data arrived since last render


def _render_frame():
    """Render the latest frame with detections. Called lazily on UI poll."""
    global _rendered_frame, _frame_dirty
    # Skip if nothing changed since last render
    if not _frame_dirty:
        return
    _frame_dirty = False

    with _video_lock:
        frame_b64 = _latest_frame.get("lp-camera1", "")
        detections = list(_latest_detections)

    if not frame_b64:
        return

    try:
        img = Image.open(io.BytesIO(base64.b64decode(frame_b64)))
    except Exception:
        return


    # Resize to 640x360 for faster browser transfer
    img = img.resize((640, 360), Image.LANCZOS)
    scale_x = 640 / 1920
    scale_y = 360 / 1080

    draw = ImageDraw.Draw(img)

    for obj in detections:
        bounds = obj.get("camera_bounds", {}).get("lp-camera1")
        if not bounds:
            continue

        try:
            x = float(bounds["x"]) * scale_x
            y = float(bounds["y"]) * scale_y
            w = float(bounds["width"]) * scale_x
            h = float(bounds["height"]) * scale_y
        except (KeyError, TypeError, ValueError):
            continue
        obj_id = obj.get("id", "")
        category = obj.get("category", "object")
        color = _get_color(obj_id)

        # Draw bounding box (2px thick at half res)
        for t in range(2):
            draw.rectangle([x - t, y - t, x + w + t, y + h + t], outline=color)

        # Label
        label = f"{category} {obj_id[:6]}"
        reid_state = obj.get("reid_state", "")
        if reid_state:
            label += f" [{reid_state}]"

        bbox = draw.textbbox((x, y - 18), label, font=_FONT)
        draw.rectangle([bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1], fill=color)
        draw.text((x, y - 18), label, fill=(0, 0, 0), font=_FONT)

    with _rendered_lock:
        _rendered_frame = img


def get_annotated_frame(camera_id="lp-camera1"):
    """Return the pre-rendered annotated frame (renders lazily on demand)."""
    global _frame_dirty
    # Render on demand (driven by UI poll timer), not on every MQTT message.
    # This caps rendering to once per poll interval regardless of frame rate.
    try:
        _render_frame()
    except Exception as e:
        print(f"[UI] Render error: {e}")
    with _rendered_lock:
        frame = _rendered_frame
    if frame is not None:
        return frame
    img = Image.new("RGB", (640, 360), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    text = "Loading......"
    bbox = draw.textbbox((0, 0), text, font=_FONT)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.text(((640 - text_w) / 2, (360 - text_h) / 2), text, fill=(180, 180, 180), font=_FONT)
    return img

def get_scene_name():
    try:
        with open(ZONE_CONFIG, "r") as f:
            config = json.load(f)
        scene_name = config.get("scene_name", "Unknown")
        density = config.get("stream_density", 1)
        # Support scenes array (backward compat)
        scenes = config.get("scenes", [])
        if scenes:
            scene_name = scenes[0].get("scene_name", "Unknown")
            density = len(scenes)
        if density > 1:
            return f"{scene_name} (x{density})"
        return scene_name
    except Exception as e:
        return f"Unknown ({e})"

# Cached data to avoid blanking tables on transient API failures
_cached_zones = pd.DataFrame(columns=["Zone ID", "Name", "Type"])
_cached_sessions = pd.DataFrame(columns=["Person", "Scene", "Zone", "Type", "Visits"])
_cached_alerts = pd.DataFrame(columns=["Alert ID", "Type", "Level", "Person", "Region", "Details", "Timestamp"])
_cached_alert_summary = pd.DataFrame(columns=["Alert Type", "Count"])


def get_zones():
    global _cached_zones
    try:
        resp = api_get_with_retry(url=ZONES_API, retries=2, delay=1)
        if resp.status_code == 200:
            data = resp.json()
            rows = []
            for zone_id, zone in data.items():
                rows.append({
                    "Zone ID": zone_id,
                    "Name": zone.get("name"),
                    "Type": zone.get("type"),
                })
            if rows:
                _cached_zones = pd.DataFrame(rows)
            return _cached_zones
        return _cached_zones
    except Exception:
        return _cached_zones

def get_sessions():
    global _cached_sessions
    try:
        resp = api_get_with_retry(url=SESSIONS_API, retries=2, delay=1)
        if resp.status_code == 200:
            data = resp.json()
            rows = []
            for session in data:
                person_id = session.get("object_id", "")[:8]
                scene_name = session.get("scene_name", "")
                zone_summary = session.get("zone_summary", [])
                if zone_summary:
                    for z in zone_summary:
                        rows.append({
                            "Person": person_id,
                            "Scene": scene_name,
                            "Zone": z.get("zone_name", "?"),
                            "Type": z.get("zone_type", "?"),
                            "Visits": z.get("visit_count", 0),
                        })
            if rows:
                _cached_sessions = pd.DataFrame(rows)
            else:
                _cached_sessions = pd.DataFrame(columns=["Person", "Scene", "Zone", "Type", "Visits"])
            return _cached_sessions
        return _cached_sessions
    except Exception:
        return _cached_sessions

def get_alerts():
    global _cached_alerts
    try:
        # Use MQTT-fed alerts if available, fall back to REST
        with _mqtt_lock:
            data = list(_mqtt_alerts)
        if not data:
            resp = api_get_with_retry(url=ALERTS_API, retries=2, delay=1)
            if resp.status_code == 200:
                data = resp.json()
            else:
                return _cached_alerts
        rows = []
        for alert in data:
            meta = alert.get("metadata", {})
            payload = alert.get("payload", {})
            # Build details from metadata (survives MQTT) excluding known fields
            detail_keys = {k: v for k, v in meta.items()
                          if k not in ("alert_id", "person_id", "zone_id", "zone_name", "severity")}
            if not detail_keys:
                detail_keys = {k: v for k, v in payload.items() if k not in ("severity", "evidence")}
            rows.append({
                "Alert ID": (alert.get("alert_id") or meta.get("alert_id", ""))[:8],
                "Type": alert.get("alert_type", ""),
                "Level": alert.get("alert_level") or meta.get("severity") or payload.get("severity", ""),
                "Person": (alert.get("object_id") or meta.get("person_id", ""))[:8],
                "Region": alert.get("region_name") or meta.get("zone_name", "N/A"),
                "Details": json.dumps(detail_keys) if detail_keys else "{}",
                "Timestamp": alert.get("timestamp", ""),
            })
        if rows:
            _cached_alerts = pd.DataFrame(rows)
        return _cached_alerts
    except Exception:
        return _cached_alerts

def get_alert_summary():
    global _cached_alert_summary
    try:
        with _mqtt_lock:
            data = list(_mqtt_alerts)
        if not data:
            resp = api_get_with_retry(url=ALERTS_API, retries=2, delay=1)
            if resp.status_code == 200:
                data = resp.json()
            else:
                return _cached_alert_summary
        counts = {}
        for alert in data:
            atype = alert.get("alert_type", "UNKNOWN")
            counts[atype] = counts.get(atype, 0) + 1
        rows = [{"Alert Type": k, "Count": v} for k, v in counts.items()]
        if rows:
            _cached_alert_summary = pd.DataFrame(rows)
        return _cached_alert_summary
    except Exception:
        return _cached_alert_summary

def refresh_data():
    return get_annotated_frame(), get_zones(), get_sessions(), get_alerts(), get_alert_summary()

HEADER_HTML = """
<div style="
    position: fixed; top: 0; left: 0; right: 0; z-index: 100;
    background: linear-gradient(135deg, #0071c5 0%, #004a8f 100%);
    width: 100%; height: 52px;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 1.5rem;
    border-bottom: 2px solid #005a9e;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
">
    <div style="display: flex; align-items: center; gap: 0.8rem;">
        <svg width="64" height="28" viewBox="0 0 200 80" xmlns="http://www.w3.org/2000/svg">
            <text x="10" y="55" font-family="Arial, sans-serif" font-size="48" font-weight="bold" fill="white">Intel</text>
        </svg>
        <span style="font-size: 16px; font-weight: 600; color: white; font-family: 'Segoe UI', sans-serif; letter-spacing: 0.3px;">
            Suspicious Activity Detection
        </span>
    </div>
    <span style="font-size: 12px; color: #ffffffaa; font-family: 'Segoe UI', sans-serif;">
        SCENE_NAME_PLACEHOLDER
    </span>
</div>
""".replace("SCENE_NAME_PLACEHOLDER", get_scene_name())

FOOTER_HTML = """
<div style="
    position: fixed; bottom: 0; left: 0; right: 0; z-index: 100;
    background: linear-gradient(135deg, #0071c5 0%, #004a8f 100%);
    width: 100%; color: #ffffffcc;
    text-align: center; padding: 0.4rem; font-size: 12px;
    font-family: 'Segoe UI', sans-serif;
    border-top: 2px solid #005a9e;
    box-shadow: 0 -2px 8px rgba(0,0,0,0.15);
">
    &copy; 2026 Intel Corporation
</div>
"""

CUSTOM_CSS = """
footer, .built-with, .api-link, .settings-link,
div[class*="footer"], a[href*="gradio.app"] { display: none !important; }

.gradio-container {
    padding-top: 52px !important;
    max-width: 100% !important;
    background: #f0f2f5 !important;
}

/* Video panel — dark background */
#video-panel {
    background: #111 !important;
    border-radius: 8px;
    padding: 0.4rem !important;
}
#video-panel img {
    border-radius: 6px;
    width: 100% !important;
    height: auto !important;
}

/* Right sidebar panels */
.panel-card {
    background: white; border-radius: 8px;
    padding: 0.6rem 0.8rem; margin-bottom: 0.5rem;
    border: 1px solid #e0e3e8;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.panel-title {
    font-size: 13px; font-weight: 700; color: #0071c5;
    font-family: 'Segoe UI', sans-serif;
    text-transform: uppercase; letter-spacing: 0.5px;
    margin-bottom: 0.3rem; padding-bottom: 0.25rem;
    border-bottom: 2px solid #0071c5;
}
.live-badge {
    display: inline-block; background: #e53935; color: white;
    font-size: 10px; font-weight: 700; padding: 2px 8px;
    border-radius: 3px; letter-spacing: 1px;
    animation: pulse-red 1.5s infinite;
    margin-bottom: 4px;
}
@keyframes pulse-red {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

/* Compact dataframes */
.gradio-dataframe { font-size: 12px !important; }
.gradio-dataframe th { font-size: 11px !important; padding: 4px 6px !important; }
.gradio-dataframe td { padding: 3px 6px !important; }

/* Alerts section at bottom */
#alerts-row {
    margin-top: 0.3rem;
}
"""

with gr.Blocks(title="Storewide Loss Prevention Dashboard") as demo:
    gr.HTML(HEADER_HTML)

    # ── Top row: Live Video (left) | Zones + Sessions (right) ──
    with gr.Row(equal_height=False):

        # ── LEFT: Live Video Feed ──
        with gr.Column(scale=4, elem_id="video-panel"):
            gr.HTML('<span class="live-badge">&#9679; LIVE</span>')
            live_video = gr.Image(
                label=None, type="pil", interactive=False,
                show_label=False,
            )

        # ── RIGHT: Zones & Person Activity ──
        with gr.Column(scale=5, min_width=320):
            gr.HTML('<div class="panel-card"><div class="panel-title">Zones / Regions</div></div>')
            zones_table = gr.Dataframe(interactive=False, max_height=180)

            gr.HTML('<div class="panel-card" style="margin-top:0.3rem"><div class="panel-title">Person Zone Activity</div></div>')
            sessions_table = gr.Dataframe(interactive=False, max_height=220)

    # ── Bottom row: Alert Summary (left) | All Alerts (right) ──
    with gr.Row(equal_height=False):
        with gr.Column(scale=2, min_width=200):
            gr.HTML('<div class="panel-card"><div class="panel-title">Alert Summary</div></div>')
            alert_summary_table = gr.Dataframe(interactive=False, max_height=120)

        with gr.Column(scale=7, min_width=400):
            gr.HTML('<div class="panel-card"><div class="panel-title">All Alerts</div></div>')
            alerts_table = gr.Dataframe(interactive=False, max_height=250)

    # Auto-poll every 1 second (balanced: responsive UI without overloading backend)
    timer = gr.Timer(1.0)
    timer.tick(
        fn=refresh_data,
        inputs=[],
        outputs=[live_video, zones_table, sessions_table, alerts_table, alert_summary_table],
    )

    demo.load(
        fn=refresh_data,
        inputs=[],
        outputs=[live_video, zones_table, sessions_table, alerts_table, alert_summary_table],
    )

    gr.HTML(FOOTER_HTML)

demo.launch(server_name="0.0.0.0", css=CUSTOM_CSS)
