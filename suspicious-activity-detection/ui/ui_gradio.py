
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
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn


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
_last_camera_ts = 0.0  # throttle: timestamp of last accepted camera frame


def _on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe(MQTT_ALERT_TOPIC, qos=1)
        client.subscribe(MQTT_CAMERA_TOPIC, qos=0)
        client.subscribe(MQTT_SCENE_TOPIC, qos=0)
        print(f"[MQTT] Subscribed to {MQTT_ALERT_TOPIC}, {MQTT_CAMERA_TOPIC}, {MQTT_SCENE_TOPIC}")
    else:
        print(f"[MQTT] Connect failed, rc={rc}")


def _on_mqtt_message(client, userdata, msg):
    global _last_camera_ts

    if msg.topic.startswith("scenescape/image/camera/"):
        # Throttle: skip if last frame was < 180ms ago (match 5 FPS render)
        now = time.monotonic()
        if now - _last_camera_ts < 0.18:
            return
        camera_id = msg.topic.split("/")[-1]
        try:
            payload = json.loads(msg.payload.decode())
        except Exception:
            return
        with _video_lock:
            _latest_frame[camera_id] = payload.get("image", "")
        _last_camera_ts = now
        _frame_event.set()
    elif msg.topic.startswith("scenescape/regulated/scene/"):
        try:
            payload = json.loads(msg.payload.decode())
        except Exception:
            return
        with _video_lock:
            _latest_detections.clear()
            for obj in payload.get("objects", []):
                _latest_detections.append(obj)
        _frame_event.set()
    elif msg.topic.startswith("alerts"):
        try:
            payload = json.loads(msg.payload.decode())
        except Exception:
            return
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
_rendered_jpeg = b""  # Pre-encoded JPEG bytes
_frame_event = threading.Event()  # Signalled when new MQTT data arrives


def _render_frame():
    """Render the latest frame with detections."""
    global _rendered_frame, _rendered_jpeg

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
    img = img.resize((640, 360), Image.BILINEAR)
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

    # Encode JPEG outside lock to minimize lock hold time
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=65)
    jpeg_bytes = buf.getvalue()

    with _rendered_lock:
        _rendered_frame = img
        _rendered_jpeg = jpeg_bytes


# ── Background render thread ──────────────────────────────────────────────────
def _render_loop():
    """Background thread: wait for new frame, render at ~5 FPS max."""
    while True:
        # Block until MQTT delivers a new frame (no CPU spin)
        _frame_event.wait(timeout=1.0)
        _frame_event.clear()
        try:
            _render_frame()
        except Exception as e:
            print(f"[Render] error: {e}")
        time.sleep(0.2)  # ~5 FPS cap


_render_thread = threading.Thread(target=_render_loop, daemon=True)
_render_thread.start()


def _make_loading_jpeg():
    """Generate a 'Loading...' placeholder JPEG."""
    img = Image.new("RGB", (640, 360), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    text = "Loading..."
    bbox = draw.textbbox((0, 0), text, font=_FONT)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((640 - tw) / 2, (360 - th) / 2), text, fill=(180, 180, 180), font=_FONT)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return buf.getvalue()

_loading_jpeg = _make_loading_jpeg()


def _mjpeg_generator():
    """Yield MJPEG multipart frames from the rendered frame buffer."""
    # Stream live frames — show loading placeholder until first real frame arrives
    while True:
        with _rendered_lock:
            jpeg = _rendered_jpeg
        if not jpeg:
            jpeg = _loading_jpeg
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
            + jpeg + b"\r\n"
        )
        time.sleep(0.2)  # ~5 FPS


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
_high_value_seen = False


def _refresh_data_cache():
    """Background thread: refresh zone/session/alert caches every 2s.
    Runs in its own thread so slow HTTP calls never block Gradio's event queue."""
    global _cached_zones, _cached_sessions, _cached_alerts, _cached_alert_summary, _high_value_seen
    while True:
        try:
            # --- Zones ---
            try:
                resp = requests.get(ZONES_API, timeout=3)
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
            except Exception:
                pass

            # --- Sessions ---
            try:
                resp = requests.get(SESSIONS_API, timeout=3)
                if resp.status_code == 200:
                    data = resp.json()
                    rows = []
                    for session in data:
                        person_id = session.get("object_id", "")[:8]
                        scene_name = session.get("scene_name", "")
                        zone_summary = session.get("zone_summary", [])
                        if zone_summary:
                            for z in zone_summary:
                                zone_name = z.get("zone_name", "?")
                                zone_type = z.get("zone_type", "?")
                                visit_count = z.get("visit_count", 0)
                                if zone_type.upper() == "HIGH_VALUE":
                                    _high_value_seen = True
                                rows.append({
                                    "Person": person_id,
                                    "Scene": scene_name,
                                    "Zone": zone_name,
                                    "Type": zone_type,
                                    "Visits": visit_count,
                                })
                    if rows:
                        _cached_sessions = pd.DataFrame(rows)
                    else:
                        _cached_sessions = pd.DataFrame(columns=["Person", "Scene", "Zone", "Type", "Visits"])
            except Exception:
                pass

            # --- Alerts ---
            try:
                with _mqtt_lock:
                    data = list(_mqtt_alerts)
                if not data:
                    resp = requests.get(ALERTS_API, timeout=3)
                    if resp.status_code == 200:
                        data = resp.json()
                if data:
                    rows = []
                    for alert in data:
                        meta = alert.get("metadata", {})
                        payload = alert.get("payload", {})
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
                    # --- Alert Summary ---
                    counts = {}
                    for alert in data:
                        atype = alert.get("alert_type", "UNKNOWN")
                        counts[atype] = counts.get(atype, 0) + 1
                    srows = [{"Alert Type": k, "Count": v} for k, v in counts.items()]
                    if srows:
                        _cached_alert_summary = pd.DataFrame(srows)
            except Exception:
                pass
        except Exception:
            pass
        time.sleep(2.0)


_data_thread = threading.Thread(target=_refresh_data_cache, daemon=True)
_data_thread.start()


def get_zones():
    return _cached_zones

def get_sessions():
    return _cached_sessions

def get_alerts():
    return _cached_alerts

def get_alert_summary():
    return _cached_alert_summary


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
    background: white !important; border-radius: 8px !important;
    padding: 0.6rem 0.8rem !important; margin-bottom: 0.5rem !important;
    border: 1px solid #e0e3e8 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}
.panel-title {
    font-size: 14px !important; font-weight: 700 !important; color: #0071c5 !important;
    font-family: 'Segoe UI', sans-serif !important;
    text-transform: uppercase !important; letter-spacing: 0.5px !important;
    margin-bottom: 0.3rem !important; padding-bottom: 0.25rem !important;
    border-bottom: 2px solid #0071c5 !important;
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
    gr.HTML(f"<style>{CUSTOM_CSS}</style>")
    gr.HTML(HEADER_HTML)

    # ── Top row: Live Video (left) | Zones + Sessions (right) ──
    with gr.Row(equal_height=False):

        # ── LEFT: Live Video Feed (MJPEG stream) ──
        with gr.Column(scale=4, elem_id="video-panel"):
            gr.HTML(
                '<span style="display:inline-block;background:#e53935;color:white;'
                'font-size:10px;font-weight:700;padding:2px 8px;border-radius:3px;'
                'letter-spacing:1px;margin-bottom:4px;">&#9679; LIVE</span>'
            )
            gr.HTML(
                '<div style="position:relative;width:100%;padding-bottom:56.25%;background:#1e1e1e;border-radius:8px;overflow:hidden;">'
                '<img id="mjpeg-feed" src="/mjpeg" style="position:absolute;top:0;left:0;width:100%;height:100%;object-fit:contain;" '
                'alt="Live Video Feed" />'
                '</div>'
                '<script>'
                '(function(){'
                '  function reconnect(){'
                '    var img=document.getElementById("mjpeg-feed");'
                '    if(img){img.src="/mjpeg?t="+Date.now();}'
                '  }'
                '  setTimeout(reconnect, 2000);'
                '  setInterval(function(){'
                '    var img=document.getElementById("mjpeg-feed");'
                '    if(img && (!img.complete || img.naturalWidth===0)){reconnect();}'
                '  }, 5000);'
                '  document.addEventListener("visibilitychange",function(){'
                '    if(!document.hidden){reconnect();}'
                '  });'
                '})();'
                '</script>'
            )

        # ── RIGHT: Zones & Person Activity ──
        with gr.Column(scale=5, min_width=320):
            gr.HTML('<div style="background:white;border-radius:8px;padding:0.6rem 0.8rem;margin-bottom:0.5rem;border:1px solid #e0e3e8;box-shadow:0 1px 3px rgba(0,0,0,0.06);"><div style="font-size:14px;font-weight:700;color:#0071c5;font-family:Segoe UI,sans-serif;text-transform:uppercase;letter-spacing:0.5px;padding-bottom:0.25rem;border-bottom:2px solid #0071c5;">Zones / Regions</div></div>')
            zones_table = gr.Dataframe(interactive=False, max_height=180)

            gr.HTML('<div style="background:white;border-radius:8px;padding:0.6rem 0.8rem;margin-top:0.3rem;margin-bottom:0.5rem;border:1px solid #e0e3e8;box-shadow:0 1px 3px rgba(0,0,0,0.06);"><div style="font-size:14px;font-weight:700;color:#0071c5;font-family:Segoe UI,sans-serif;text-transform:uppercase;letter-spacing:0.5px;padding-bottom:0.25rem;border-bottom:2px solid #0071c5;">Person Zone Activity</div></div>')
            sessions_table = gr.Dataframe(interactive=False, max_height=220)

    # ── Bottom row: Alert Summary (left) | All Alerts (right) ──
    with gr.Row(equal_height=False):
        with gr.Column(scale=2, min_width=200):
            gr.HTML('<div style="background:white;border-radius:8px;padding:0.6rem 0.8rem;margin-bottom:0.5rem;border:1px solid #e0e3e8;box-shadow:0 1px 3px rgba(0,0,0,0.06);"><div style="font-size:14px;font-weight:700;color:#0071c5;font-family:Segoe UI,sans-serif;text-transform:uppercase;letter-spacing:0.5px;padding-bottom:0.25rem;border-bottom:2px solid #0071c5;">Alert Summary</div></div>')
            alert_summary_table = gr.Dataframe(interactive=False, max_height=120)

        with gr.Column(scale=7, min_width=400):
            gr.HTML('<div style="background:white;border-radius:8px;padding:0.6rem 0.8rem;margin-bottom:0.5rem;border:1px solid #e0e3e8;box-shadow:0 1px 3px rgba(0,0,0,0.06);"><div style="font-size:14px;font-weight:700;color:#0071c5;font-family:Segoe UI,sans-serif;text-transform:uppercase;letter-spacing:0.5px;padding-bottom:0.25rem;border-bottom:2px solid #0071c5;">All Alerts</div></div>')
            alerts_table = gr.Dataframe(interactive=False, max_height=250)

    # Data tables poll — video is handled by native MJPEG stream
    data_timer = gr.Timer(2.0)
    data_timer.tick(
        fn=lambda: (get_zones(), get_sessions(), get_alerts(), get_alert_summary()),
        inputs=[],
        outputs=[zones_table, sessions_table, alerts_table, alert_summary_table],
        queue=False,
    )

    demo.load(
        fn=lambda: (get_zones(), get_sessions(), get_alerts(), get_alert_summary()),
        inputs=[],
        outputs=[zones_table, sessions_table, alerts_table, alert_summary_table],
    )

    gr.HTML(FOOTER_HTML)

# Mount Gradio on a FastAPI app with MJPEG endpoint
app = FastAPI()


@app.get("/mjpeg")
def mjpeg_feed():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
