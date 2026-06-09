#!/usr/bin/env python3
"""
POI Reset Script - Simple action script for performance-tools
Expected arguments: --app_dir
"""

import argparse
import sys
import os
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app_dir", required=True)
    args = parser.parse_args()
    
    print("Resetting POI state...")
    
    # Reset Redis via API
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://localhost:8000/api/v1/alerts",
            method="DELETE"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            print("  Alert dedup cleared")
    except Exception as e:
        print(f"  Could not clear dedup: {e}")
    
    # Clean metrics files
    import glob
    # Do not remove vlm_application_metrics files: poi-backend keeps an open
    # file handler and deleting them causes writes to an unlinked inode.
    patterns = ["vlm_performance_metrics*.txt"]
    for pat in patterns:
        for f in glob.glob(f"/tmp/{pat}"):
            try:
                os.remove(f)
                print(f"  Removed {os.path.basename(f)}")
            except Exception:
                pass
    
    print("Reset complete")
    sys.exit(0)


if __name__ == "__main__":
    main()