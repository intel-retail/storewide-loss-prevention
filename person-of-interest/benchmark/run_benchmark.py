#!/usr/bin/env python3
"""
POI Stream Density Benchmark Entry Point

Simply calls performance-tools which handles everything.
"""

import argparse
import sys
import os

PERF_TOOLS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "performance-tools"
)
sys.path.insert(0, PERF_TOOLS_PATH)

from stream_density_benchmark import StreamDensityBenchmark, BenchmarkConfig


def main():
    parser = argparse.ArgumentParser(description="POI Stream Density Benchmark")
    parser.add_argument("app_dir", help="Path to person-of-interest/")
    parser.add_argument("--target_latency_ms", type=float, default=2000)
    parser.add_argument("--latency_metric", choices=["avg", "max"], default="avg")
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--resource_config", default="")
    
    args = parser.parse_args()
    
    poi_scripts_dir = os.path.join(args.app_dir, "scripts")
    
    config = BenchmarkConfig(
        target_latency_ms=args.target_latency_ms,
        latency_metric=args.latency_metric,
        results_dir=args.results_dir
    )
    
    benchmark = StreamDensityBenchmark(
        config=config,
        poi_scripts_dir=poi_scripts_dir,
        app_dir=args.app_dir,
        resource_config=args.resource_config
    )
    
    results = benchmark.run()
    sys.exit(0 if results["met_target"] else 1)


if __name__ == "__main__":
    main()