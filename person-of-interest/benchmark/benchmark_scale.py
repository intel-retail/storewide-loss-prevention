#!/usr/bin/env python3
"""
POI Scaling Script - action adapter called by performance-tools benchmark.

Delegates all scaling operations to poi_scaling.py which lives in this
(person-of-interest/benchmark) directory.  The old poi_stream_density.py
is no longer imported.

Expected arguments: --app_dir, --num_scenes, --resource_config (optional)
"""

import argparse
import os
import subprocess
import sys

# Ensure this directory is on the path so poi_scaling is importable when
# this script is launched as a subprocess by poi_stream_density_new.py.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import poi_scaling  # noqa: E402 – import after path fix


def main():
    parser = argparse.ArgumentParser(description="POI scaling action script")
    parser.add_argument("--app_dir", required=True)
    parser.add_argument("--num_scenes", type=int)
    parser.add_argument("--resource_config", default="")
    parser.add_argument("--get_new_component", type=int)
    parser.add_argument("--check_healthy", action="store_true")

    args = parser.parse_args()

    if args.check_healthy:
        result = subprocess.run(
            "docker ps --filter name=poi-backend --format '{{.Status}}'",
            shell=True, capture_output=True, text=True,
        )
        sys.exit(0 if "Up" in result.stdout else 1)

    elif args.get_new_component is not None:
        name = poi_scaling.get_new_camera_name(args.app_dir, args.get_new_component)
        if name:
            print(name)
            sys.exit(0)
        sys.exit(1)

    elif args.num_scenes:
        print(f"Scaling POI to {args.num_scenes} scenes...")
        poi_scaling.scale_pipeline_services(
            args.app_dir,
            args.num_scenes,
            resource_config=args.resource_config,
        )
        print("Scale complete")
        sys.exit(0)

    else:
        print("No action specified", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()