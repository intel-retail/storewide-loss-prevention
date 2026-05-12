# Store-wide Loss Prevention

Store-wide Loss Prevention is an AI-powered retail security solution built on
Intel® SceneScape and Intel® hardware (CPU, GPU, NPU). It uses real-time
multi-camera tracking, pose estimation, and vision-language models to detect
and alert on suspicious in-store behaviors — enabling proactive loss prevention
without intrusive surveillance.

## Use Cases

| # | Use Case | Status | Directory |
|---|----------|--------|-----------|
| 1 | [Suspicious Activity Detection](#suspicious-activity-detection) | Available | `suspicious-activity-detection/` |
| 2 | [Person of Interest (POI)](#person-of-interest) | TBD | — |

---

### Suspicious Activity Detection

MQTT-driven loss prevention service for Intel SceneScape retail deployments.
The service monitors person behavior across store zones using real-time tracking,
manages session state and detection rules, and stores cropped person frames in
SeaweedFS. Behavioral analysis (pose detection + VLM confirmation) runs as a
separate service invoked conditionally when a person enters a high-value zone.

**Detected Activities:**

| # | Activity | Trigger | Alert Level |
|---|----------|---------|-------------|
| 1 | Merchandise Concealment | Behavioral Analysis confirms suspicious pose + VLM | WARNING |
| 2 | Checkout Bypass | Visited HIGH_VALUE zone, exited without CHECKOUT | WARNING / CRITICAL |
| 3 | Loitering | Dwell > threshold in HIGH_VALUE zone | WARNING |
| 4 | Repeated Visits | Re-entries ≥ threshold to same HIGH_VALUE zone | WARNING |
| 5 | Restricted Zone Violation | Entered RESTRICTED zone | CRITICAL |

For full details see the [User Guide](suspicious-activity-detection/docs/user-guide/index.md).

---

### Person of Interest

TBD