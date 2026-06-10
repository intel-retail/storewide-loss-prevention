#!/usr/bin/env python3
"""
POI Alert Script - Simple action script for performance-tools
Expected arguments: --app_dir, --timeout, --since, --component (optional)
Returns: 0 if alert received, 1 if timeout
"""

import argparse
import sys
import os
import time
import json
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def wait_for_alert_from_camera(camera_id: str, timeout: int, since: datetime) -> bool:
    """Wait for alert from specific camera"""
    elapsed = 0
    since_aware = since.replace(tzinfo=timezone.utc) if since.tzinfo else since
    
    print(f"Waiting up to {timeout}s for alert from {camera_id}...")
    
    while elapsed < timeout:
        sleep_s = min(3, timeout - elapsed)
        time.sleep(sleep_s)
        elapsed += sleep_s
        
        try:
            req = urllib.request.Request("http://localhost:8000/api/v1/alerts?limit=50")
            with urllib.request.urlopen(req, timeout=10) as resp:
                alerts = json.loads(resp.read().decode())
            
            for alert in alerts:
                # Check camera
                alert_camera = alert.get("match", {}).get("camera_id") or alert.get("camera_id", "")
                if alert_camera != camera_id:
                    continue
                
                # Check timestamp
                dispatched_str = alert.get("dispatched_at", "")
                if dispatched_str:
                    try:
                        dispatched = datetime.fromisoformat(dispatched_str.replace('Z', '+00:00'))
                        if dispatched.tzinfo is None:
                            dispatched = dispatched.replace(tzinfo=timezone.utc)
                        
                        if dispatched >= since_aware:
                            print(f"✓ Alert from {camera_id} received after {elapsed}s")
                            return True
                    except Exception:
                        pass
        except Exception as e:
            print(f"  Error checking alerts: {e}")
    
    print(f"✗ No alert from {camera_id} within {timeout}s")
    return False


def wait_for_any_alert(timeout: int, since: datetime) -> bool:
    """Wait for any alert (baseline)"""
    elapsed = 0
    since_aware = since.replace(tzinfo=timezone.utc) if since.tzinfo else since
    
    print(f"Waiting up to {timeout}s for any alert...")
    
    while elapsed < timeout:
        sleep_s = min(3, timeout - elapsed)
        time.sleep(sleep_s)
        elapsed += sleep_s
        
        try:
            req = urllib.request.Request("http://localhost:8000/api/v1/alerts?limit=10")
            with urllib.request.urlopen(req, timeout=10) as resp:
                alerts = json.loads(resp.read().decode())
            
            for alert in alerts:
                dispatched_str = alert.get("dispatched_at", "")
                if dispatched_str:
                    try:
                        dispatched = datetime.fromisoformat(dispatched_str.replace('Z', '+00:00'))
                        if dispatched.tzinfo is None:
                            dispatched = dispatched.replace(tzinfo=timezone.utc)
                        
                        if dispatched >= since_aware:
                            print(f"✓ Alert received after {elapsed}s")
                            return True
                    except Exception:
                        pass
        except Exception:
            pass
    
    print(f"✗ No alert within {timeout}s")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app_dir", required=True)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--since", required=True)
    parser.add_argument("--component", help="Camera name to wait for")
    
    args = parser.parse_args()
    
    since = datetime.fromisoformat(args.since)
    
    if args.component:
        alert_received = wait_for_alert_from_camera(args.component, args.timeout, since)
    else:
        alert_received = wait_for_any_alert(args.timeout, since)
    
    sys.exit(0 if alert_received else 1)


if __name__ == "__main__":
    main()