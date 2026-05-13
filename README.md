# Store-wide Loss Prevention

A suite of Intel® edge AI applications for retail loss prevention, powered by Intel® SceneScape spatial computing and Intel® OpenVINO™ inference.

## Use Cases

| # | Use Case | Status | Directory |
|---|----------|--------|-----------|
| 1 | [Person of Interest (POI)](#person-of-interest) | Available | `person-of-interest/` |
| 2 | [Suspicious Activity Detection](#suspicious-activity-detection) | Available | `suspicious-activity-detection/` |

---

### Person of Interest

Real-time detection of enrolled suspects across multiple cameras using face re-identification and FAISS vector search, with historical investigation capabilities.

For full details see the [POI User Guide](person-of-interest/docs/user-guide/index.md).

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

For full details see the [SAD User Guide](suspicious-activity-detection/docs/user-guide/index.md).
