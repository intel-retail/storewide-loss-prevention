
import gradio as gr
import pandas as pd
import requests
import json
import os
import time

# Use Docker service name for container-to-container communication
LP_BASE_URL = os.environ.get("LP_BASE_URL", "http://storewide-loss-prevention:8082")
ZONES_API = f"{LP_BASE_URL}/api/v1/lp/zones"
SESSIONS_API = f"{LP_BASE_URL}/api/v1/lp/sessions"
ALERTS_API = f"{LP_BASE_URL}/api/v1/lp/alerts"
ZONE_CONFIG = os.environ.get("ZONE_CONFIG", "/app/zone_config.json")

MAX_RETRIES = 5
RETRY_DELAY = 3  # seconds


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

def get_scene_name():
    try:
        with open(ZONE_CONFIG, "r") as f:
            config = json.load(f)
        return config.get("scene_name", "Unknown")
    except Exception as e:
        return f"Unknown ({e})"

def get_zones():
    try:
        resp = api_get_with_retry(ZONES_API)
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
                return pd.DataFrame(rows)
            return pd.DataFrame(columns=["Zone ID", "Name", "Type"])
        return pd.DataFrame([{"Error": f"API returned {resp.status_code}"}])
    except Exception as e:
        return pd.DataFrame([{"Error": str(e)}])

def get_sessions():
    try:
        resp = api_get_with_retry(SESSIONS_API)
        if resp.status_code == 200:
            data = resp.json()
            rows = []
            for session in data:
                person_id = session.get("object_id", "")[:8]
                zone_summary = session.get("zone_summary", [])
                if zone_summary:
                    for z in zone_summary:
                        rows.append({
                            "Person": person_id,
                            "Zone": z.get("zone_name", "?"),
                            "Type": z.get("zone_type", "?"),
                            "Visits": z.get("visit_count", 0),
                            "Dwell (s)": round(z.get("total_dwell_seconds", 0.0), 1),
                            "Inside": "✔" if z.get("currently_inside") else "",
                        })
            if rows:
                return pd.DataFrame(rows)
            return pd.DataFrame(columns=["Person", "Zone", "Type", "Visits", "Dwell (s)", "Inside"])
        return pd.DataFrame([{"Error": f"API returned {resp.status_code}"}])
    except Exception as e:
        return pd.DataFrame([{"Error": str(e)}])

def get_alerts():
    try:
        resp = api_get_with_retry(ALERTS_API)
        if resp.status_code == 200:
            data = resp.json()
            rows = []
            for alert in data:
                rows.append({
                    "Alert ID": alert.get("alert_id", "")[:8],
                    "Type": alert.get("alert_type"),
                    "Level": alert.get("alert_level"),
                    "Person": alert.get("object_id", "")[:8],
                    "Region": alert.get("region_name", "N/A"),
                    "Details": json.dumps(alert.get("details", {})),
                    "Timestamp": alert.get("timestamp", ""),
                })
            if rows:
                return pd.DataFrame(rows)
            return pd.DataFrame(columns=["Alert ID", "Type", "Level", "Person", "Region", "Details", "Timestamp"])
        return pd.DataFrame([{"Error": f"API returned {resp.status_code}"}])
    except Exception as e:
        return pd.DataFrame([{"Error": str(e)}])

def get_alert_summary():
    try:
        resp = api_get_with_retry(ALERTS_API)
        if resp.status_code == 200:
            data = resp.json()
            counts = {}
            for alert in data:
                atype = alert.get("alert_type", "UNKNOWN")
                counts[atype] = counts.get(atype, 0) + 1
            rows = [{"Alert Type": k, "Count": v} for k, v in counts.items()]
            if rows:
                return pd.DataFrame(rows)
            return pd.DataFrame(columns=["Alert Type", "Count"])
        return pd.DataFrame([{"Error": f"API returned {resp.status_code}"}])
    except Exception as e:
        return pd.DataFrame([{"Error": str(e)}])

def refresh_data():
    return get_zones(), get_sessions(), get_alerts(), get_alert_summary()

HEADER_HTML = """
<div style="
    position: sticky; top: 0; left: 0; right: 0; z-index: 50;
    background: #0071c5; width: 100%; height: 72px;
    display: flex; align-items: center; padding: 0 2rem;
    border-bottom: 1px solid #005a9e;
">
    <div style="display: flex; align-items: center; gap: 1rem;">
        <svg width="89" height="40" viewBox="0 0 200 80" xmlns="http://www.w3.org/2000/svg">
            <text x="10" y="55" font-family="Arial, sans-serif" font-size="48" font-weight="bold" fill="white">intel</text>
        </svg>
        <span style="font-size: 18px; font-weight: 500; color: white; font-family: sans-serif;">
            Storewide Loss Prevention
        </span>
    </div>
</div>
"""

FOOTER_HTML = """
<div style="
    position: sticky; bottom: 0; left: 0; right: 0; width: 100%;
    background: #0071c5; color: white; text-align: center;
    padding: 0 2rem; height: 48px; font-size: 14px; z-index: 10;
    box-shadow: 0 -2px 8px rgba(0,0,0,0.04); border-top: 1px solid #e6e7e8;
    display: flex; align-items: center; justify-content: center;
    font-family: sans-serif; font-weight: 400;
">
    <span>&copy; 2026 Intel Corporation. All rights reserved.</span>
</div>
"""

with gr.Blocks(title="Storewide Loss Prevention Dashboard") as demo:
    gr.HTML("""<style>
        footer, .built-with, .api-link, .settings-link,
        div[class*="footer"], a[href*="gradio.app"] { display: none !important; }
    </style>""")
    gr.HTML(HEADER_HTML)

    gr.Markdown(f"# Scene: {get_scene_name()}")
    gr.Markdown("---")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Zones/Regions")
            zones_table = gr.Dataframe(interactive=False)
        with gr.Column():
            gr.Markdown("## Person Zone Activity")
            sessions_table = gr.Dataframe(interactive=False)

    gr.Markdown("---")
    gr.Markdown("## Alert Summary (by Type)")
    alert_summary_table = gr.Dataframe(interactive=False)

    gr.Markdown("---")
    gr.Markdown("## All Alerts")
    alerts_table = gr.Dataframe(interactive=False)

    # Auto-poll every 2 seconds
    timer = gr.Timer(2)
    timer.tick(
        fn=refresh_data,
        inputs=[],
        outputs=[zones_table, sessions_table, alerts_table, alert_summary_table],
    )

    # Load initial data on page open
    demo.load(
        fn=refresh_data,
        inputs=[],
        outputs=[zones_table, sessions_table, alerts_table, alert_summary_table],
    )

    gr.HTML(FOOTER_HTML)

demo.launch(server_name="0.0.0.0")
