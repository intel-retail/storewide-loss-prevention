# VLM Latency — Qwen3-VL-8B-Instruct

## Configuration

| Setting              | Value                                  |
|----------------------|----------------------------------------|
| Model                | Qwen/Qwen3-VL-8B-Instruct              |
| Source model         | Qwen/Qwen3-VL-8B-Instruct              |
| Precision            | int8                                   |
| Target device        | GPU                                    |
| Serving runtime      | OpenVINO Model Server (OVMS)           |
| OVMS image           | openvino/model_server:2026.1-gpu       |
| OVMS pipeline type   | VLM_CB                                 |
| Endpoint             | http://ovms-vlm:8001/v3/chat/completions |
| API                  | OpenAI-compatible chat completions     |
| Frames per call      | 4                                      |
| Max image size       | 512 px (longest side)                  |
| `max_tokens`         | 500                                    |
| `temperature`        | 0.1                                    |
| `vlm_timeout`        | 60 s                                   |
| Pattern under test   | `shelf_to_waist` confirmation          |
| Caller               | `behavioral-analysis` (`vlm_client.analyze`) |
| Measurement          | Wall-clock around `httpx.AsyncClient.post` (request send → response received), logged at INFO from `vlm_client.py` |

## Per-call results

| Run | Latency (ms) | Frames | Prompt tokens | Completion tokens | Total tokens |
|-----|--------------|--------|---------------|-------------------|--------------|
|  1  | 24909        | 4      | 447           | 52                | 499          |
|  2  | 23632        | 4      | 447           | 67                | 514          |
|  3  | 23154        | 4      | 447           | 65                | 512          |
|  4  | 23226        | 4      | 447           | 65                | 512          |
|  5  | 23254        | 4      | 447           | 65                | 512          |
|  6  | 23891        | 4      | 447           | 70                | 517          |
|  7  | 23910        | 4      | 447           | 69                | 516          |
|  8  | 21060        | 4      | 447           | 53                | 500          |
|  9  | 23257        | 4      | 447           | 65                | 512          |
| 10  | 23752        | 4      | 447           | 68                | 515          |
| 11  | 23308        | 4      | 447           | 65                | 512          |
| 12  | 23341        | 4      | 447           | 65                | 512          |
| 13  | 23319        | 4      | 447           | 65                | 512          |
| 14  | 23358        | 4      | 447           | 66                | 513          |

## Aggregate

| Metric          | Value     |
|-----------------|-----------|
| Sample count    | 14        |
| Min             | 21060 ms  |
| Max             | 24909 ms  |
| Mean            | 23384 ms  |
| Median          | 23330 ms  |
| p90             | 23891 ms  |
| Std. deviation  | 812 ms    |
| Mean completion tokens | ~64 |
| Approx. completion throughput | ~2.7 tok/s |

## Comparison vs. Qwen2.5-VL-7B-Instruct (same workload, GPU, int8, 4 frames)

| Metric          | Qwen2.5-VL-7B | Qwen3-VL-8B | Delta        |
|-----------------|---------------|-------------|--------------|
| Median latency  | 25035 ms      | 23330 ms    | **−1705 ms (−6.8%)** |
| Mean latency    | 24095 ms      | 23384 ms    | −711 ms (−3.0%)      |
| p90 latency     | 25782 ms      | 23891 ms    | −1891 ms (−7.3%)     |
| Std. deviation  | 2200 ms       | 812 ms      | −63% (much more stable) |
| Mean prompt tok | 458           | 447         | −11           |
| Mean completion tok | ~55       | ~64         | +9 (more verbose) |

Qwen3-VL is slightly faster despite being a larger 8B model, and notably more
consistent (≈3× lower std. deviation). Image-token compression in Qwen3-VL
explains the small drop in prompt tokens.

## Source

- Log file: `application.log` (lines matching `INFO:vlm_client:VLM call latency=`).
- Logging implemented in `behavioral-analysis/src/vlm_client.py` using `time.perf_counter()` around the `httpx` POST.
