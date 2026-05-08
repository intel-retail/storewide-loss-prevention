# VLM Latency Baseline

## Configuration

| Setting              | Value                                  |
|----------------------|----------------------------------------|
| Model                | Qwen/Qwen2.5-VL-7B-Instruct            |
| Source model         | Qwen/Qwen2.5-VL-7B-Instruct            |
| Precision            | int8                                   |
| Target device        | GPU                                    |
| Serving runtime      | OpenVINO Model Server (OVMS)           |
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
|  1  | 26317        | 4      | 458           | 57                | 515          |
|  2  | 22794        | 4      | 458           | 42                | 500          |
|  3  | 25262        | 4      | 458           | 59                | 517          |
|  4  | 18978        | 4      | 458           | 60                | 518          |
|  5  | 23532        | 4      | 458           | 49                | 507          |
|  6  | 25056        | 4      | 458           | 57                | 515          |
|  7  | 25035        | 4      | 458           | 57                | 515          |
|  8  | 22830        | 4      | 458           | 42                | 500          |
|  9  | 18424        | 4      | 458           | 57                | 515          |
| 10  | 25359        | 4      | 458           | 59                | 517          |
| 11  | 25546        | 4      | 458           | 60                | 518          |
| 12  | 25782        | 4      | 458           | 62                | 520          |
| 13  | 25131        | 4      | 458           | 58                | 516          |
| 14  | 24890        | 4      | 458           | 56                | 514          |
| 15  | 24421        | 4      | 458           | 53                | 511          |
| 16  | 25094        | 4      | 458           | 58                | 516          |
| 17  | 25396        | 4      | 458           | 59                | 517          |
| 18  | 22889        | 4      | 458           | 42                | 500          |
| 19  | 25029        | 4      | 458           | 58                | 516          |

## Aggregate

| Metric          | Value     |
|-----------------|-----------|
| Sample count    | 19        |
| Min             | 18424 ms  |
| Max             | 26317 ms  |
| Mean            | 24095 ms  |
| Median          | 25035 ms  |
| p90             | 25782 ms  |
| Std. deviation  | 2200 ms   |
| Mean completion tokens | ~55 |
| Approx. completion throughput | ~2.3 tok/s |

## Source

- Log file: `application.log` (lines matching `INFO:vlm_client:VLM call latency=`).
- Logging added in `behavioral-analysis/src/vlm_client.py` using `time.perf_counter()` around the `httpx` POST.
