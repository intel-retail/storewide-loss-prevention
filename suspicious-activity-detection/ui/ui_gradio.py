
import pandas as pd
import requests
import json
import os
import time
import threading
import base64
import io
from collections import deque
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
import uvicorn


# Use Docker service name for container-to-container communication
LP_BASE_URL = os.environ.get("LP_BASE_URL", "http://storewide-loss-prevention:8082")
ZONES_API = f"{LP_BASE_URL}/api/v1/lp/zones"
SESSIONS_API = f"{LP_BASE_URL}/api/v1/lp/sessions?include_pending=true"
ALERTS_API = f"{LP_BASE_URL}/api/v1/lp/alerts"
ZONE_CONFIG = os.environ.get("ZONE_CONFIG", "/app/zone_config.json")

# VLM-Recall bridge (investigator search). The API key is held server-side and
# injected by the proxy endpoints below so the browser never sees it.
RECALL_BASE_URL = os.environ.get("RECALL_BASE_URL", "http://vss-recall-bridge:8080")
RECALL_API_KEY = os.environ.get("RECALL_API_KEY", os.environ.get("BRIDGE_API_KEY", ""))
RECALL_SEARCH_API = f"{RECALL_BASE_URL}/api/v1/lp/recall/search"
RECALL_CLIPS_API = f"{RECALL_BASE_URL}/api/v1/lp/recall/clips"

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
_mqtt_alerts: deque = deque(maxlen=800)
_mqtt_lock = threading.Lock()

# Thread-safe live video frame + detections
_latest_frame = {}      # {camera_id: base64_jpeg}
_latest_detections = []  # list of objects from regulated scene data
_video_lock = threading.Lock()
_last_camera_ts = 0.0  # throttle: timestamp of last accepted camera frame


def _fmt_local_ts(iso_str: str) -> str:
    """Convert a UTC ISO timestamp to local time for display."""
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str)
        local_dt = dt.astimezone()  # converts to system local TZ
        return local_dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return iso_str


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
    if not os.path.exists(ZONE_CONFIG):
        return "Suspicious Activity Detection"
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
                        person_id = str(session.get("object_id") or "")[:8]
                        scene_name = str(session.get("scene_name") or "")
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
                        print(f"[DATA] Sessions updated: {len(rows)} rows", flush=True)
                    else:
                        _cached_sessions = pd.DataFrame(columns=["Person", "Scene", "Zone", "Type", "Visits"])
                else:
                    print(f"[DATA] Sessions API returned {resp.status_code}", flush=True)
            except Exception as exc:
                print(f"[DATA] Sessions error: {exc}", flush=True)

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
                            "Timestamp": _fmt_local_ts(alert.get("timestamp", "")),
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


SCENE_NAME = get_scene_name()

# ─── Full HTML dashboard served by FastAPI (no Gradio event system) ───
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Storewide Loss Prevention Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#f0f2f5;color:#222;}
header{position:fixed;top:0;left:0;right:0;z-index:100;height:52px;
 background:linear-gradient(135deg,#0071c5,#004a8f);display:flex;
 align-items:center;justify-content:space-between;padding:0 1.5rem;
 border-bottom:2px solid #005a9e;box-shadow:0 2px 8px rgba(0,0,0,.15);}
header .brand{display:flex;align-items:center;gap:.8rem;}
header .brand span{font-size:16px;font-weight:600;color:#fff;letter-spacing:.3px;}
header .scene{font-size:12px;color:#ffffffaa;}
footer{position:fixed;bottom:0;left:0;right:0;z-index:100;
 background:linear-gradient(135deg,#0071c5,#004a8f);color:#ffffffcc;
 text-align:center;padding:.4rem;font-size:12px;
 border-top:2px solid #005a9e;box-shadow:0 -2px 8px rgba(0,0,0,.15);}
main{padding:60px .8rem 36px;display:grid;
 grid-template-columns:4fr 5fr;grid-template-rows:auto auto;gap:.6rem;
 max-width:1600px;margin:0 auto;}
.video-panel{background:#111;border-radius:8px;padding:.4rem;position:relative;}
.video-panel img{width:100%;border-radius:6px;display:block;}
.live-badge{position:absolute;top:12px;left:12px;background:#e53935;color:#fff;
 font-size:10px;font-weight:700;padding:2px 8px;border-radius:3px;
 letter-spacing:1px;animation:pulse 1.5s infinite;z-index:2;}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.6;}}
.card{background:#fff;border-radius:8px;padding:.6rem .8rem;
 border:1px solid #e0e3e8;box-shadow:0 1px 3px rgba(0,0,0,.06);}
.card+.card{margin-top:.5rem;}
.card-title{font-size:14px;font-weight:700;color:#0071c5;text-transform:uppercase;
 letter-spacing:.5px;padding-bottom:.25rem;margin-bottom:.4rem;
 border-bottom:2px solid #0071c5;}
.right-col{display:flex;flex-direction:column;gap:.5rem;}
.bottom-row{grid-column:1/-1;display:grid;grid-template-columns:2fr 7fr;gap:.6rem;}
table.dt{width:100%;border-collapse:collapse;font-size:12px;}
table.dt th{text-align:left;padding:5px 8px;font-size:11px;font-weight:700;
 color:#333;border-bottom:2px solid #0071c5;background:#f8f9fb;}
table.dt td{padding:4px 8px;border-bottom:1px solid #eee;}
table.dt tr:hover td{background:#f0f6ff;}
.empty{color:#999;padding:12px 8px;font-size:13px;font-style:italic;}
.scroll{overflow-y:auto;}
</style>
</head>
<body>

<header>
 <div class="brand">
  <svg width="64" height="28" viewBox="0 0 200 80"><text x="10" y="55" font-family="Arial,sans-serif" font-size="48" font-weight="bold" fill="white">Intel</text></svg>
  <span>Suspicious Activity Detection</span>
 </div>
 <div style="display:flex;align-items:center;gap:1rem;">
  <a href="/recall" style="font-size:13px;font-weight:600;color:#fff;text-decoration:none;background:#ffffff22;padding:5px 12px;border-radius:4px;border:1px solid #ffffff44;">&#128270; Recall</a>
  <span class="scene">""" + SCENE_NAME + """</span>
 </div>
</header>

<main>
 <!-- LEFT: Video -->
 <div class="video-panel">
  <span class="live-badge">&#9679; LIVE</span>
  <img id="mjpeg-feed" src="/mjpeg" alt="Live Video Feed">
 </div>

 <!-- RIGHT: Zones + Sessions -->
 <div class="right-col">
  <div class="card">
   <div class="card-title">Zones / Regions</div>
   <div id="zones-data" class="scroll" style="max-height:180px">
    <div class="empty">Loading&hellip;</div>
   </div>
  </div>
  <div class="card">
   <div class="card-title">Person Zone Activity</div>
   <div id="sessions-data" class="scroll" style="max-height:220px">
    <div class="empty">Loading&hellip;</div>
   </div>
  </div>
 </div>

 <!-- BOTTOM: Alert Summary + All Alerts -->
 <div class="bottom-row">
  <div class="card">
   <div class="card-title">Alert Summary</div>
   <div id="summary-data" class="scroll" style="max-height:120px">
    <div class="empty">Loading&hellip;</div>
   </div>
  </div>
  <div class="card">
   <div class="card-title">All Alerts</div>
   <div id="alerts-data" class="scroll" style="max-height:250px">
    <div class="empty">Loading&hellip;</div>
   </div>
  </div>
 </div>
</main>

<footer>&copy; 2026 Intel Corporation</footer>

<script>
function T(rows,cols){
 if(!rows||!rows.length) return '<div class="empty">No data</div>';
 var h='<table class="dt"><thead><tr>';
 for(var i=0;i<cols.length;i++) h+='<th>'+cols[i]+'</th>';
 h+='</tr></thead><tbody>';
 for(var r=0;r<rows.length;r++){
  h+='<tr>';
  for(var c=0;c<cols.length;c++){
   var v=rows[r][cols[c]];
   h+='<td>'+(v!==null&&v!==undefined?v:'')+'</td>';
  }
  h+='</tr>';
 }
 return h+'</tbody></table>';
}
function refresh(){
 fetch('/api/data').then(function(r){return r.json()}).then(function(d){
  var e;
  e=document.getElementById('zones-data');
  if(e) e.innerHTML=T(d.zones,['Zone ID','Name','Type']);
  e=document.getElementById('sessions-data');
  if(e) e.innerHTML=T(d.sessions,['Person','Scene','Zone','Type','Visits']);
  e=document.getElementById('alerts-data');
  if(e) e.innerHTML=T(d.alerts,['Alert ID','Type','Level','Person','Region','Details','Timestamp']);
  e=document.getElementById('summary-data');
  if(e) e.innerHTML=T(d.alert_summary,['Alert Type','Count']);
 }).catch(function(err){console.error('[Dashboard]',err)});
}
/* MJPEG reconnect */
function reconn(){var img=document.getElementById('mjpeg-feed');if(img)img.src='/mjpeg?t='+Date.now();}
setInterval(function(){var img=document.getElementById('mjpeg-feed');
 if(img&&(!img.complete||img.naturalWidth===0))reconn();},5000);
document.addEventListener('visibilitychange',function(){if(!document.hidden)reconn();});

setTimeout(refresh,500);
setInterval(refresh,3000);
</script>
</body>
</html>
"""

# ─── FastAPI app with MJPEG, data API, and dashboard ───
app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML


@app.get("/mjpeg")
def mjpeg_feed():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/data")
def api_data():
    """JSON endpoint polled by the dashboard JavaScript every 3 seconds."""
    def _df_to_records(df):
        if df is None or df.empty:
            return []
        return df.to_dict(orient="records")

    return {
        "zones": _df_to_records(_cached_zones),
        "sessions": _df_to_records(_cached_sessions),
        "alerts": _df_to_records(_cached_alerts),
        "alert_summary": _df_to_records(_cached_alert_summary),
    }


# ─── Investigator Recall page + server-side proxy to the VLM-recall bridge ───
RECALL_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Recall &mdash; Storewide Loss Prevention</title>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#f0f2f5;color:#222;}
header{position:fixed;top:0;left:0;right:0;z-index:100;height:52px;
 background:linear-gradient(135deg,#0071c5,#004a8f);display:flex;
 align-items:center;justify-content:space-between;padding:0 1.5rem;
 border-bottom:2px solid #005a9e;box-shadow:0 2px 8px rgba(0,0,0,.15);}
header .brand{display:flex;align-items:center;gap:.8rem;}
header .brand span{font-size:16px;font-weight:600;color:#fff;letter-spacing:.3px;}
header a.back{font-size:13px;font-weight:600;color:#fff;text-decoration:none;
 background:#ffffff22;padding:5px 12px;border-radius:4px;border:1px solid #ffffff44;}
header #status{font-size:12px;color:#ffffffcc;}
main{padding:64px 1rem 24px;display:grid;grid-template-columns:340px 1fr;gap:1rem;
 max-width:1600px;margin:0 auto;}
.card{background:#fff;border-radius:8px;padding:1rem 1.1rem;
 border:1px solid #e0e3e8;box-shadow:0 1px 3px rgba(0,0,0,.06);}
label{display:block;font-size:11px;color:#666;margin:12px 0 5px;
 text-transform:uppercase;letter-spacing:.04em;font-weight:600;}
input,textarea{width:100%;padding:8px 10px;border:1px solid #ccd2da;border-radius:5px;
 font-size:14px;font-family:inherit;background:#fff;color:#222;}
textarea{resize:vertical;min-height:60px;}
input:focus,textarea:focus{outline:none;border-color:#0071c5;}
.row{display:flex;gap:8px;}.row>div{flex:1;}
button{cursor:pointer;border:none;border-radius:5px;font-size:14px;font-weight:600;
 padding:11px;background:#0071c5;color:#fff;width:100%;margin-top:16px;}
button:hover{background:#005a9e;}button:disabled{opacity:.5;cursor:not-allowed;}
.hint{font-size:11px;color:#888;margin-top:4px;}
.title{font-size:14px;font-weight:700;color:#0071c5;text-transform:uppercase;
 letter-spacing:.5px;padding-bottom:.3rem;margin-bottom:.2rem;border-bottom:2px solid #0071c5;}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;margin-top:14px;}
.hit{background:#fff;border:1px solid #e0e3e8;border-radius:8px;overflow:hidden;
 display:flex;flex-direction:column;box-shadow:0 1px 3px rgba(0,0,0,.06);}
.hit video{width:100%;background:#000;aspect-ratio:16/9;display:block;}
.hit .ph{width:100%;aspect-ratio:16/9;background:#11151c;color:#9fb0c3;display:flex;
 align-items:center;justify-content:center;font-size:13px;cursor:pointer;}
.hit .ph:hover{color:#fff;}
.hit .b{padding:10px 12px;font-size:12px;}
.hit .b .meta{color:#666;margin:2px 0;word-break:break-all;}
.score{float:right;font-weight:700;color:#1a9e4b;}
.tags{display:flex;flex-wrap:wrap;gap:5px;margin-top:7px;}
.tag{background:#eef3fb;border:1px solid #d4e0f2;color:#21527d;font-size:11px;padding:2px 8px;border-radius:12px;}
.empty{color:#999;font-size:14px;padding:40px 0;text-align:center;font-style:italic;}
.banner{background:#fdecea;border:1px solid #f5c2c0;color:#a32a26;padding:10px 14px;
 border-radius:6px;font-size:13px;margin-bottom:14px;display:none;}
.banner.show{display:block;}
footer{text-align:center;color:#999;font-size:12px;padding:8px;}
</style>
</head>
<body>
<header>
 <div class="brand">
  <svg width="64" height="28" viewBox="0 0 200 80"><text x="10" y="55" font-family="Arial,sans-serif" font-size="48" font-weight="bold" fill="white">Intel</text></svg>
  <span>&#128270; Recall</span>
 </div>
 <div style="display:flex;align-items:center;gap:1rem;">
  <span id="status">checking&hellip;</span>
  <a class="back" href="/">&#8592; Dashboard</a>
 </div>
</header>
<main>
 <div class="card" style="align-self:start;">
  <div class="title">Search</div>
  <label for="query">Appearance query *</label>
  <textarea id="query" placeholder="e.g. man in red hoodie carrying a black backpack"></textarea>
  <label for="cameras">Cameras (optional)</label>
  <input id="cameras" placeholder="cam1, cam-12" />
  <div class="hint">Comma-separated. Blank = all cameras.</div>
  <div class="row">
   <div><label for="time_start">From (optional)</label><input id="time_start" type="datetime-local" /></div>
  </div>
  <div class="row">
   <div><label for="time_end">To (optional)</label><input id="time_end" type="datetime-local" /></div>
  </div>
  <button id="searchBtn">Search</button>
 </div>
 <div>
  <div id="banner" class="banner"></div>
  <div class="title" id="resultsTitle" style="border:none;">Results</div>
  <div id="grid" class="grid"></div>
  <div id="empty" class="empty">Enter a query and press Search.</div>
 </div>
</main>
<footer>&copy; 2026 Intel Corporation</footer>
<script>
var $=function(id){return document.getElementById(id);};
function showBanner(m){var b=$('banner');b.textContent=m;b.classList.add('show');}
function clearBanner(){$('banner').classList.remove('show');}
function checkHealth(){fetch('/api/recall/health').then(function(r){return r.json();}).then(function(d){
 var s=$('status');if(d.status==='ok'){s.textContent='\\u25CF bridge online';s.style.color='#9affc0';}
 else{s.textContent='\\u25CF bridge error';s.style.color='#ffd27f';}}).catch(function(){
 $('status').textContent='\\u25CF bridge unreachable';$('status').style.color='#ffb3b3';});}
function toIso(v){if(!v)return null;var d=new Date(v);return isNaN(d)?null:d.toISOString();}
function parseCameras(v){var l=(v||'').split(',').map(function(s){return s.trim();}).filter(Boolean);return l.length?l:null;}
function buildBody(){var q=$('query').value.trim();if(!q)throw new Error('Query is required.');
 var b={query:q,limit:20};
 var c=parseCameras($('cameras').value);if(c)b.cameras=c;
 var ts=toIso($('time_start').value),te=toIso($('time_end').value);
 if(ts)b.time_start=ts;if(te)b.time_end=te;return b;}
function fmtTime(t){if(!t)return '\\u2014';var d=new Date(t);return isNaN(d)?t:d.toLocaleString();}
function doSearch(){clearBanner();var btn=$('searchBtn'),grid=$('grid'),empty=$('empty');var body;
 try{body=buildBody();}catch(e){showBanner(e.message);return;}
 btn.disabled=true;btn.textContent='Searching\\u2026';grid.innerHTML='';empty.textContent='Searching\\u2026';empty.style.display='block';
 fetch('/api/recall/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
 .then(function(r){if(r.status===401){showBanner('Bridge rejected the API key (server config).');empty.style.display='none';return null;}
  return r.text().then(function(t){if(!r.ok){showBanner('Search failed ('+r.status+'): '+t);empty.style.display='none';return null;}
  return JSON.parse(t);});})
 .then(function(d){if(d)renderResults(d.results||[]);})
 .catch(function(e){showBanner('Network error: '+e.message);empty.style.display='none';})
 .finally(function(){btn.disabled=false;btn.textContent='Search';});}
function renderResults(results){var grid=$('grid'),empty=$('empty');
 results=results||[];
 $('resultsTitle').textContent='Results ('+results.length+')';grid.innerHTML='';
 if(!results.length){empty.textContent='No matches found.';empty.style.display='block';return;}
 empty.style.display='none';
 results.forEach(function(h){var card=document.createElement('div');card.className='hit';
  var ph=document.createElement('div');ph.className='ph';ph.textContent='\\u25B6 Load clip';
  ph.onclick=function(){loadClip(h.video_id,card,ph,h.segment_start);};
  var tags=(h.tags||[]).map(function(t){return '<span class="tag">'+t+'</span>';}).join('');
  var b=document.createElement('div');b.className='b';
  b.innerHTML='<div><span class="score">'+(h.score||0).toFixed(3)+'</span><strong>'+fmtTime(h.capture_time)+'</strong></div>'+
   '<div class="meta">video: '+h.video_id+'</div>'+
   '<div class="meta">segment: '+(h.segment_start||0).toFixed(1)+'s \\u2013 '+(h.segment_end||0).toFixed(1)+'s</div>'+
   '<div class="tags">'+tags+'</div>';
  card.appendChild(ph);card.appendChild(b);grid.appendChild(card);
  loadClip(h.video_id,card,ph,h.segment_start,false);});}
function loadClip(videoId,card,ph,seekTo,autoplay){ph.textContent='Loading clip\\u2026';ph.onclick=null;
 var video=document.createElement('video');video.controls=true;video.preload='metadata';
 video.src='/api/recall/clips/'+encodeURIComponent(videoId);
 video.onloadedmetadata=function(){if(seekTo){try{video.currentTime=seekTo;}catch(e){}}};
 video.onerror=function(){ph.textContent='Clip load failed';};
 card.replaceChild(video,ph);if(autoplay)video.play().catch(function(){});}
$('searchBtn').addEventListener('click',doSearch);
$('query').addEventListener('keydown',function(e){if(e.key==='Enter'&&(e.ctrlKey||e.metaKey))doSearch();});
checkHealth();
</script>
</body>
</html>
"""


@app.get("/recall", response_class=HTMLResponse)
def recall_page():
    return RECALL_HTML


@app.get("/api/recall/health")
def recall_health():
    """Report whether the recall bridge is reachable (no key exposed)."""
    try:
        r = requests.get(f"{RECALL_BASE_URL}/health", timeout=5)
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as exc:
        return JSONResponse({"status": "unreachable", "detail": str(exc)}, status_code=502)


@app.post("/api/recall/search")
def recall_search(payload: dict):
    """Proxy investigator queries to the bridge, injecting the API key server-side."""
    try:
        r = requests.post(
            RECALL_SEARCH_API,
            json=payload,
            headers={"X-API-Key": RECALL_API_KEY},
            timeout=120,
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)
    return Response(content=r.content, status_code=r.status_code, media_type="application/json")


@app.get("/api/recall/clips/{video_id}")
def recall_clip(video_id: str, request: Request):
    """Stream a proxied clip from the bridge (key injected server-side).

    Forwards the browser's Range header and propagates the upstream 206 +
    Content-Range/Accept-Ranges/Content-Length so the player can seek.
    """
    fwd_headers = {"X-API-Key": RECALL_API_KEY}
    range_header = request.headers.get("range")
    if range_header:
        fwd_headers["Range"] = range_header
    try:
        r = requests.get(
            f"{RECALL_CLIPS_API}/{video_id}",
            headers=fwd_headers,
            timeout=120,
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)
    if r.status_code not in (200, 206):
        return Response(content=r.content, status_code=r.status_code, media_type="application/json")
    passthrough = {}
    for key in ("Content-Range", "Accept-Ranges", "Content-Length", "Content-Type"):
        if key in r.headers:
            passthrough[key] = r.headers[key]
    return Response(
        content=r.content,
        status_code=r.status_code,
        media_type=r.headers.get("Content-Type", "video/mp4"),
        headers=passthrough,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
