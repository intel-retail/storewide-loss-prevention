# Benchmarking and Stream Density

This guide covers the performance benchmarking tools included with the
Store-wide Loss Prevention application. The benchmark framework measures
end-to-end latency as the number of camera/scene pipelines increases,
helping you determine the maximum stream density your hardware can sustain
within a given latency budget.

## Prerequisites

Before running benchmarks, ensure the following setup steps are completed:

```bash
make update-submodules      # clones performance-tools submodule
make download-sample-data   # downloads video to scenescape/sample_data/
make download-models        # downloads models to models/ (detect_models/, reid_models/, vlm_models/)
```

## Quick Start

### Single-Scene Benchmark

Run a one-shot benchmark with a single camera scene to measure baseline
latency:

```bash
make benchmark
```

This target:

1. Builds (or pulls) the benchmark Docker image.
2. Starts the full stack (`make up`).
3. Runs a single-scene measurement against the configured latency target.
4. Tears down the stack on completion.

### Stream Density (Multi-Scene)

Iteratively add camera scenes until end-to-end latency exceeds the target
threshold:

```bash
make benchmark-stream-density
```

At each iteration the framework:

1. Updates `stream_density` in `configs/zone_config.json`.
2. Re-runs `init.sh` to regenerate `.env` and the DL Streamer pipeline config.
3. Generates `docker/docker-compose.cameras.yaml` with additional RTSP
   camera streams (`lp-cams-N`) for each new camera.
4. Restarts `scene-import`, `lp-video` (DL Streamer), and `swlp-service`.
5. Collects latency samples from `swlp-service` docker logs and
   `vlm_application_metrics` files.

SceneScape core services (`web`, `controller`, `broker`), `ovms-vlm`,
`behavioral-analysis`, `seaweedfs`, and `alert-service` remain running
throughout.

The benchmark stops when latency exceeds the target or the maximum iteration
count is reached.

## Configuration

All benchmark parameters can be set via `make` variables or environment
variables:

| Parameter | Make Variable | Default | Description |
|-----------|---------------|---------|-------------|
| Target latency | `BENCHMARK_TARGET_LATENCY_MS` or `TARGET_LATENCY_MS` | `10000` | Latency threshold in milliseconds |
| Latency metric | `BENCHMARK_LATENCY_METRIC` or `LATENCY_METRIC` | `avg` | Which statistic to compare: `avg` or `max` |
| Scene increment | `BENCHMARK_SCENE_INCREMENT` or `SCENE_INCREMENT` | `1` | Number of scenes to add per iteration |
| Init duration | `BENCHMARK_INIT_DURATION` | `90` | Warm-up seconds after service restart |
| Stabilise duration | `BENCHMARK_STABILISE_DURATION` | `30` | Extra wait for the pipeline to stabilise before collecting metrics |
| Max iterations | `BENCHMARK_MAX_ITERATIONS` | `50` | Safety cap on the number of iterations |
| Min throughput ratio | `BENCHMARK_MIN_THROUGHPUT_RATIO` | `0.5` | Minimum ratio of actual-to-expected BA samples (0–1) |
| Results directory | `RESULTS_PATH` | `./results` | Where JSON and CSV results are written |

Example with custom parameters:

```bash
make benchmark-stream-density \
  TARGET_LATENCY_MS=5000 \
  LATENCY_METRIC=max \
  SCENE_INCREMENT=2 \
  BENCHMARK_INIT_DURATION=120 \
  BENCHMARK_STABILISE_DURATION=60
```

Set `REGISTRY=false` to force a local build when running `make benchmark` or
`make benchmark-stream-density`.

### Device Profile

Both `make benchmark` and `make benchmark-stream-density` use the `DEVICE`
parameter to select an inference device profile (same as `make up`). The
profile controls the DL Streamer decode chain, detection device, and
re-identification device:

```bash
make benchmark DEVICE=all-gpu-cpu.env           # GPU detect + CPU re-id
make benchmark-stream-density DEVICE=all-gpu.env # All GPU
make benchmark-stream-density DEVICE=all-cpu.env # All CPU
make benchmark-stream-density DEVICE=all-npu-cpu.env  # NPU detect + CPU re-id (default)
```

Available profiles are in `configs/res/`. See
[SceneScape Setup — Device Profiles](./scenescape-setup.md#device-profiles)
for details.

## Results and Metrics

Results are written to `RESULTS_PATH` (default `./results`) in both JSON and
CSV formats:

```
results/
├── swlp_stream_density_<timestamp>.json
├── swlp_stream_density_<timestamp>.csv
└── consolidated_metrics.csv          # after make consolidate-metrics
```

Each result file contains per-iteration data:

| Field | Description |
|-------|-------------|
| `num_scenes` | Number of camera scenes in this iteration |
| `latency_ms` | Measured latency (avg or max, per config) |
| `passed` | Whether latency was within the target |
| `throughput_ratio` | Actual / expected BA samples |
| `actual_samples` | BA round-trip latency samples collected |
| `samples_per_scene` | Samples per scene during the collection window |
| `memory_percent` | Host memory utilisation |
| `cpu_percent` | Host CPU utilisation |

### Consolidate Metrics

After running one or more benchmarks, consolidate all result files into a
single CSV:

```bash
make consolidate-metrics
```

Output: `results/consolidated_metrics.csv`

### Environment Variables

The script also reads configuration from environment variables (useful in CI):

| Variable | Default | Description |
|----------|---------|-------------|
| `TARGET_LATENCY_MS` | `30000` | Latency threshold |
| `LATENCY_METRIC` | `avg` | `avg` or `max` |
| `SCENE_INCREMENT` | `1` | Scenes per iteration |
| `INIT_DURATION` | `90` | Warm-up seconds |
| `STABILISE_DURATION` | `30` | Collection window |
| `RESULTS_DIR` | `./results` | Output directory |
| `MAX_ITERATIONS` | `50` | Maximum iterations |

## Interpreting Results

A typical stream-density result summary looks like:

```
STREAM DENSITY RESULTS
======================================================================
  Target Latency:  8000ms
  Max Scenes:      3
  Met Target:      Yes
  Best Latency:    1842ms @ 3 scene(s)

Scenes    Latency     Throughput    Mem %     CPU %     Status
------------------------------------------------------------------
1         620         100%          45.2      23.1      ✓ PASS
2         1205        95%           52.8      41.6      ✓ PASS
3         1842        87%           61.3      58.2      ✓ PASS
4         2450        72%           68.1      72.4      ✗ FAIL
======================================================================
```

- **Max Scenes** is the highest scene count that stayed within the latency
  target.
- **Throughput** shows how many BA (behavioral analysis) round-trip samples
  were collected versus the expected count. A ratio below
  `min_throughput_ratio` may indicate pipeline starvation.
- The benchmark exits with code `0` if the target was met, `1` otherwise.

## Troubleshooting

| Issue | Resolution |
|-------|------------|
| `NO DATA – no latency metrics collected` | The pipeline may not have produced detections during the collection window. Increase `BENCHMARK_INIT_DURATION` or `BENCHMARK_STABILISE_DURATION`. |
| Memory threshold exceeded | The host is running low on RAM. Reduce scene count or use a machine with more memory. |
| `init.sh not found` | Ensure the `scenescape` directory is at `../scenescape` relative to `suspicious-activity-detection`. |
| Benchmark image pull fails | Check network connectivity and registry credentials. Use `REGISTRY=false` to build locally. |
| Stale scenes prevent re-import | The framework auto-cleans cloned scenes. If issues persist, run `make clean-stream-density` and retry. |
