# Store-wide Loss Prevention: Suspicious Activity Detection

The Store-wide Loss Prevention (LP) application is a reference workload that
demonstrates how Intel® SceneScape, OpenVINO™, and a Vision Language Model
(VLM) can be combined on a single Intel® platform to detect suspicious in-store
behavior in real time.

It combines several services:

- **swlp-service:** MQTT-driven core that subscribes to SceneScape, manages
  per-person session state, evaluates declarative rules, and emits alerts.
- **Behavioral Analysis Service:** Runs pose detection plus a VLM
  (Qwen2.5-VL) to confirm whether a person is concealing merchandise in a
  HIGH_VALUE zone.
- **Alert Service:** Time-window-based dedup and downstream delivery of
  alerts to MQTT topics, REST consumers, and the dashboard.
- **Frame storage (SeaweedFS / MinIO):** Cropped person frames stored per
  visit; a per-alert evidence prefix is created when an alert fires.
- **Gradio UI:** Dashboard for live alerts, sessions, and evidence frames.

Together, these components illustrate how vision-based AI inference, rule
evaluation, and downstream alerting can be orchestrated, monitored, and
visualized for a retail loss-prevention scenario.

## Supporting Resources

- [Get Started](./get-started.md) – Step-by-step instructions to build and
  run the application using `make` and Docker.
- [System Requirements](./get-started/system-requirements.md) – Hardware,
  software, and network requirements, plus an overview of the AI models used
  by each workload.
- [How It Works](./how-it-works.md) – High-level architecture, service
  responsibilities, and data/control flows.
- [Release Notes](./release-notes.md) – Version history and known issues.

> **Disclaimer:** This application is provided for development and evaluation
> purposes only and is _not_ intended for production loss-prevention or
> surveillance use without further validation, deployment hardening, and
> compliance review.

<!--hide_directive
<div class="component_card_widget">
  <a class="icon_github" href="https://github.com/intel-retail/storewide-loss-prevention">
    GitHub project
  </a>
  <a class="icon_document" href="https://github.com/intel-retail/storewide-loss-prevention/blob/main/suspicious-activity-detection/README.md">
    Readme
  </a>
</div>
hide_directive-->

<!--hide_directive
:::{toctree}
:hidden:

Get Started <get-started.md>
How It Works <how-it-works.md>
Release Notes <release-notes.md>

:::
hide_directive-->
