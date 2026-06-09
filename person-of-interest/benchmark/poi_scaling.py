#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
POI-specific scaling helpers for benchmark stream-density runs.

This module replaces the legacy poi_stream_density.py scaling functions so
the benchmark no longer depends on the performance-tools submodule for
infrastructure operations.  Metrics collection and pass/fail decisions remain
in performance-tools/benchmark-scripts/poi_stream_density_new.py.

Public API used by benchmark_scale.py
--------------------------------------
    scale_pipeline_services(app_dir, num_scenes, wait=90, resource_config="")
    get_new_camera_name(app_dir, num_scenes)
    clean_cameras_override(app_dir)
    set_stream_density(app_dir, density)
    generate_cameras_override(app_dir, num_scenes)
    generate_dlstreamer_config(app_dir, num_scenes)
    reinit_env(app_dir, resource_config="")
    zone_config_path(app_dir)
    docker_compose(app_dir, action)
"""

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# zone_config.json helpers
# ---------------------------------------------------------------------------

def zone_config_path(app_dir: str) -> Path:
    return Path(app_dir) / "configs" / "zone_config.json"


def read_zone_config(app_dir: str) -> dict:
    with open(zone_config_path(app_dir)) as f:
        return json.load(f)


def write_zone_config(app_dir: str, cfg: dict) -> None:
    p = zone_config_path(app_dir)
    bak = p.with_suffix(".json.bak")
    if not bak.exists():
        shutil.copy2(p, bak)
    with open(p, "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info("Updated %s  (stream_density=%s)", p, cfg.get("stream_density"))


def set_stream_density(app_dir: str, density: int) -> None:
    cfg = read_zone_config(app_dir)
    cfg["stream_density"] = density
    write_zone_config(app_dir, cfg)


def read_base_config(app_dir: str) -> dict:
    """Return base camera name and video file from zone_config.json."""
    cfg = read_zone_config(app_dir)
    cameras = cfg.get("cameras", [])
    if cameras:
        camera = cameras[0].get("name", "Camera_01")
        video = cameras[0].get("video_file", "Camera_01.mp4")
    else:
        camera = cfg.get("camera_name", "Camera_01")
        video = cfg.get("video_file", "Camera_01.mp4")
    return {"camera_name": camera, "video_file": video}


# ---------------------------------------------------------------------------
# env-file helpers
# ---------------------------------------------------------------------------

def write_env_var(env_file: str, key: str, value: str) -> None:
    """Write or update KEY=VALUE in an env file."""
    lines: list[str] = []
    found = False
    if os.path.isfile(env_file):
        with open(env_file) as fh:
            lines = fh.readlines()
        for i, line in enumerate(lines):
            if line.startswith(f"{key}=") or line.startswith(f"{key} ="):
                lines[i] = f"{key}={value}\n"
                found = True
                break
    if not found:
        lines.append(f"{key}={value}\n")
    with open(env_file, "w") as fh:
        fh.writelines(lines)
    logger.info("Set %s=%s in %s", key, value, env_file)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def is_npu_device(app_dir: str) -> bool:
    """Return True when the active resource config selects NPU."""
    env_file = os.path.join(app_dir, "docker", ".env")
    if os.path.isfile(env_file):
        with open(env_file) as fh:
            for line in fh:
                if line.startswith("RESOURCE_CONFIG=") and "npu" in line.lower():
                    return True
    return False


# ---------------------------------------------------------------------------
# Docker Compose helpers
# ---------------------------------------------------------------------------

def compose_cmd(app_dir: str) -> str:
    """Build a full compose invocation spanning SceneScape + POI."""
    scenescape_dir = str(Path(app_dir) / ".." / "scenescape")
    scenescape_compose = os.path.join(scenescape_dir, "docker-compose.yaml")
    overrides = os.path.join(app_dir, "docker-compose.scenescape-overrides.yml")
    poi_compose = os.path.join(app_dir, "docker-compose.yml")
    env_file = os.path.join(app_dir, "docker", ".env")

    parts = [
        "docker compose",
        f"--project-directory {shlex.quote(app_dir)}",
        f"--env-file {shlex.quote(env_file)}",
        f"-f {shlex.quote(scenescape_compose)}",
    ]
    if os.path.isfile(overrides):
        parts.append(f"-f {shlex.quote(overrides)}")
    npu_overlay = os.path.join(app_dir, "docker-compose.npu-overrides.yml")
    if os.path.isfile(npu_overlay) and is_npu_device(app_dir):
        parts.append(f"-f {shlex.quote(npu_overlay)}")
    parts.append(f"-f {shlex.quote(poi_compose)}")
    cameras_override = os.path.join(app_dir, "docker", "docker-compose.cameras.yaml")
    if os.path.isfile(cameras_override):
        parts.append(f"-f {shlex.quote(cameras_override)}")
    return " ".join(parts)


def docker_compose(app_dir: str, action: str) -> int:
    """Run a combined compose action (SceneScape + POI)."""
    if "up" in action:
        subprocess.run("docker network create storewide-lp", shell=True, capture_output=True)
    cmd = f"{compose_cmd(app_dir)} {action}"
    logger.info("Running: %s", cmd)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 and "down" not in action:
        logger.warning("docker compose stderr:\n%s", result.stderr[-500:])
    return result.returncode


# ---------------------------------------------------------------------------
# init.sh helper
# ---------------------------------------------------------------------------

def reinit_env(app_dir: str, resource_config: str = "") -> None:
    """Re-run init.sh to regenerate docker/.env with updated config."""
    init_script = Path(app_dir) / ".." / "scenescape" / "scripts" / "init.sh"
    if not init_script.exists():
        logger.warning("init.sh not found at %s — skipping .env regeneration", init_script)
        return

    env = os.environ.copy()
    if resource_config:
        try:
            rel_rc = str(Path(resource_config).relative_to(Path(app_dir)))
        except ValueError:
            rel_rc = resource_config
        env["RESOURCE_CONFIG"] = rel_rc
        logger.info("Re-running init.sh with RESOURCE_CONFIG=%s …", rel_rc)
    else:
        logger.info("Re-running init.sh to update .env …")

    cmd = f"bash {shlex.quote(str(init_script))} {shlex.quote(app_dir)}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        output = (result.stderr + result.stdout)[-500:]
        logger.warning("init.sh returned non-zero:\n%s", output)
    else:
        logger.info("init.sh completed — .env updated")


# ---------------------------------------------------------------------------
# Camera override generation
# ---------------------------------------------------------------------------

def generate_cameras_override(app_dir: str, num_scenes: int) -> None:
    """Generate docker/docker-compose.cameras.yaml for *num_scenes* scenes."""
    override_path = Path(app_dir) / "docker" / "docker-compose.cameras.yaml"
    base = read_base_config(app_dir)
    base_camera = base["camera_name"]
    base_video = base["video_file"]

    scenescape_dir = (Path(app_dir) / ".." / "scenescape").resolve()
    dlstreamer_dir = scenescape_dir / "dlstreamer-pipeline-server"

    base_camera_count = 2
    _is_npu = is_npu_device(app_dir)

    with open(override_path, "w") as f:
        f.write("# Auto-generated by poi_scaling.py — do not edit\n")
        f.write(f"# Stream density: {num_scenes} scenes\n\n")
        f.write("services:\n")

        for i in range(1, num_scenes):
            cam_idx = base_camera_count + i
            cam_name = f"{base_camera}-{cam_idx}"
            cams_svc = f"lp-cams-{cam_idx}"
            video_svc = f"lp-video-{cam_idx}"
            config_name = f"lp-config-{cam_idx}"

            f.write(f"  {cams_svc}:\n")
            f.write(f"    image: linuxserver/ffmpeg:version-8.0-cli\n")
            f.write(f'    command: "-nostdin -re -stream_loop -1 '
                    f'-i /workspace/media/{base_video} '
                    f'-c:v copy -an -f rtsp -rtsp_transport tcp '
                    f'rtsp://mediaserver:8554/{cam_name}"\n')
            f.write(f"    volumes:\n")
            f.write(f"      - vol-sample-data:/workspace/media\n")
            f.write(f"    networks:\n")
            f.write(f"      - storewide-lp\n")
            f.write(f"    depends_on:\n")
            f.write(f"      - mediaserver\n")
            f.write(f'    restart: "no"\n\n')

            f.write(f"  {video_svc}:\n")
            f.write(f"    image: docker.io/intel/dlstreamer-pipeline-server:${{DLSTREAMER_VERSION:-2026.1.0-20260331-weekly-ubuntu24}}\n")
            f.write(f"    networks:\n      storewide-lp:\n")
            f.write(f"    tty: true\n")
            f.write(f"    entrypoint: [\"./run.sh\"]\n")
            f.write(f"    devices:\n      - \"/dev/dri:/dev/dri\"\n")
            if _is_npu:
                f.write(f"      - \"/dev/accel:/dev/accel\"\n")
            f.write(f"    group_add:\n      - \"109\"\n      - \"110\"\n      - \"992\"\n")
            f.write(f"    device_cgroup_rules:\n")
            f.write(f"      - \"c 189:* rmw\"\n      - \"c 209:* rmw\"\n      - \"a 189:* rwm\"\n")
            if _is_npu:
                f.write(f"      - \"c 261:* rmw\"  # Intel NPU accel devices\n")
            f.write(f"    depends_on:\n")
            f.write(f"      broker:\n        condition: service_started\n")
            f.write(f"      ntpserv:\n        condition: service_started\n")
            f.write(f"      {cams_svc}:\n        condition: service_started\n")
            f.write(f"    healthcheck:\n")
            f.write(f'      test: ["CMD", "curl", "-I", "-s", "http://localhost:8080/pipelines"]\n')
            f.write(f"      interval: 10s\n      timeout: 5s\n      retries: 5\n      start_period: 10s\n")
            f.write(f"    environment:\n")
            f.write(f"      - RUN_MODE=EVA\n      - GENICAM=Balluff\n      - GST_DEBUG=1,gencamsrc:2\n")
            f.write(f"      - ADD_UTCTIME_TO_METADATA=true\n      - APPEND_PIPELINE_NAME_TO_PUBLISHER_TOPIC=false\n")
            f.write(f"      - MQTT_HOST=broker.scenescape.intel.com\n      - MQTT_PORT=1883\n")
            f.write(f"      - REST_SERVER_PORT=8080\n")
            f.write(f"      - HTTPS_PROXY=${{HTTPS_PROXY}}\n      - https_proxy=${{https_proxy}}\n")
            f.write(f"      - HTTP_PROXY=${{HTTP_PROXY}}\n      - http_proxy=${{http_proxy}}\n")
            f.write(f"      - NO_PROXY=mediaserver,${{NO_PROXY}}\n      - no_proxy=mediaserver,${{no_proxy}}\n")
            f.write(f"    configs:\n      - source: {config_name}\n        target: /home/pipeline-server/config.json\n")
            f.write(f"    volumes:\n")
            f.write(f"      - ../scenescape/dlstreamer-pipeline-server/user_scripts:/home/pipeline-server/user_scripts\n")
            f.write(f"      - vol-dlstreamer-pipeline-root-{cam_idx}:/var/cache/pipeline_root:uid=1999,gid=1999\n")
            f.write(f"      - vol-sample-data:/home/pipeline-server/videos\n")
            f.write(f"      - vol-models:/home/pipeline-server/models\n")
            f.write(f"    secrets:\n      - source: root-cert\n        target: certs/scenescape-ca.pem\n")
            f.write(f"    restart: always\n    pids_limit: 1000\n\n")

        all_cameras = ["Camera_01", "Camera_02"]
        for i in range(1, num_scenes):
            all_cameras.append(f"{base_camera}-{base_camera_count + i}")
        camera_csv = ",".join(all_cameras)

        f.write(f"  poi-backend:\n    environment:\n")
        f.write(f"      RTSP_PREWARM_CAMERAS: \"{camera_csv}\"\n")
        f.write(f"      MQTT_IMAGE_CAMERAS: \"{camera_csv}\"\n")
        f.write(f"      STREAM_DENSITY: \"{num_scenes}\"\n\n")

        if num_scenes > 1:
            f.write("configs:\n")
            for i in range(1, num_scenes):
                cam_idx = base_camera_count + i
                cam_name = f"{base_camera}-{cam_idx}"
                env_var = f"PIPELINE_CONFIG_{cam_idx}"
                default_path = dlstreamer_dir / f"person-of-interest-{cam_name}-pipeline-config.json"
                f.write(f"  lp-config-{cam_idx}:\n    file: ${{{env_var}:-{default_path}}}\n")
            f.write("\n")

            f.write("volumes:\n")
            for i in range(1, num_scenes):
                f.write(f"  vol-dlstreamer-pipeline-root-{base_camera_count + i}:\n")

    logger.info("Generated cameras override: %s  (%d scenes, %d extra cameras+DLStreamer instances)",
                override_path, num_scenes, max(0, num_scenes - 1))


def clean_cameras_override(app_dir: str) -> None:
    """Remove docker-compose.cameras.yaml and extra DLStreamer pipeline configs."""
    override_path = Path(app_dir) / "docker" / "docker-compose.cameras.yaml"
    if override_path.exists():
        override_path.unlink()
        logger.info("Removed %s", override_path)

    scenescape_dir = Path(app_dir) / ".." / "scenescape"
    dlstreamer_dir = scenescape_dir / "dlstreamer-pipeline-server"
    for cfg_file in dlstreamer_dir.glob("person-of-interest-*-[0-9]*-pipeline-config.json"):
        cfg_file.unlink()
        logger.info("Removed extra pipeline config: %s", cfg_file)


# ---------------------------------------------------------------------------
# DLStreamer config generation
# ---------------------------------------------------------------------------

def generate_dlstreamer_config(app_dir: str, num_scenes: int) -> None:
    """Generate per-camera DLStreamer pipeline configs for scenes > 1."""
    scenescape_dir = Path(app_dir) / ".." / "scenescape"
    dlstreamer_dir = scenescape_dir / "dlstreamer-pipeline-server"
    env_file = os.path.join(app_dir, "docker", ".env")

    base = read_base_config(app_dir)
    base_camera = base["camera_name"]
    base_camera_count = 2

    template_path = dlstreamer_dir / f"person-of-interest-{base_camera}-pipeline-config.json"
    if not template_path.exists():
        logger.warning("Pipeline template not found: %s", template_path)
        return

    with open(template_path) as fh:
        template_cfg = json.load(fh)

    for i in range(1, num_scenes):
        cam_idx = base_camera_count + i
        cam_name = f"{base_camera}-{cam_idx}"
        output_path = dlstreamer_dir / f"person-of-interest-{cam_name}-pipeline-config.json"

        cfg_str = json.dumps(template_cfg).replace(base_camera, cam_name)
        cfg = json.loads(cfg_str)

        if "config" in cfg and "pipelines" in cfg["config"]:
            for pipeline in cfg["config"]["pipelines"]:
                pipeline["name"] = f"reid_{cam_name}"

        with open(output_path, "w") as fh:
            json.dump(cfg, fh, indent=2)
        logger.info("Generated pipeline config: %s", output_path)

        write_env_var(env_file, f"PIPELINE_CONFIG_{cam_idx}", str(output_path.resolve()))

    logger.info("Generated DLStreamer configs for %d total cameras", base_camera_count + num_scenes - 1)


# ---------------------------------------------------------------------------
# Camera name helper
# ---------------------------------------------------------------------------

def get_new_camera_name(app_dir: str, num_scenes: int) -> Optional[str]:
    """Return camera name added in this iteration, or None for baseline."""
    if num_scenes <= 1:
        return None
    base = read_base_config(app_dir)
    base_camera = base["camera_name"]
    cam_idx = 2 + (num_scenes - 1)  # POI always starts with 2 base cameras
    return f"{base_camera}-{cam_idx}"


# ---------------------------------------------------------------------------
# SceneScape REST API helpers
# ---------------------------------------------------------------------------

def _scenescape_get_client(app_dir: str):
    """Authenticate with SceneScape. Returns (base_url, ssl_ctx, token) or (None,None,None)."""
    import ssl
    import urllib.request

    env_file = os.path.join(app_dir, "docker", ".env")
    supass = ""
    if os.path.isfile(env_file):
        for line in open(env_file):
            if line.startswith("SUPASS="):
                supass = line.strip().split("=", 1)[1]
                break
    if not supass:
        logger.warning("Could not read SUPASS from docker/.env — SceneScape API unavailable")
        return None, None, None

    try:
        zone_cfg = read_zone_config(app_dir)
        base_url = zone_cfg.get("scenescape_api", {}).get("base_url", "https://localhost").rstrip("/")
        base_url = base_url + "/api/v1"
    except Exception:
        base_url = "https://localhost/api/v1"

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    import json as _json
    auth_data = _json.dumps({"username": "admin", "password": supass}).encode()
    req = urllib.request.Request(
        f"{base_url}/auth", data=auth_data,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            token = _json.loads(resp.read()).get("token", "")
    except Exception as e:
        logger.warning("SceneScape authentication failed: %s", e)
        return None, None, None

    if not token:
        logger.warning("SceneScape auth returned empty token")
        return None, None, None

    return base_url, ctx, token


def _clone_scene_zip(base_zip_path: str, scene_name: str, camera_name: str) -> bytes:
    """Clone a SceneScape scene ZIP with a new scene/camera name."""
    import io
    import uuid
    import zipfile

    with zipfile.ZipFile(base_zip_path, "r") as zf:
        json_name = None
        base_json = None
        other_files: dict = {}
        for name in zf.namelist():
            data = zf.read(name)
            if name.endswith(".json"):
                json_name = name
                base_json = json.loads(data)
            else:
                other_files[name] = data

    if not json_name or base_json is None:
        raise ValueError(f"No scene JSON found in {base_zip_path}")

    scene_data = json.loads(json.dumps(base_json))
    new_scene_uid = str(uuid.uuid4())
    scene_data["uid"] = new_scene_uid
    scene_data["name"] = scene_name

    for cam in scene_data.get("cameras", []):
        cam["uid"] = camera_name
        cam["name"] = camera_name
        cam["scene"] = new_scene_uid

    for region in scene_data.get("regions", []):
        region["uid"] = str(uuid.uuid4())
        region["scene"] = new_scene_uid

    safe_name = scene_name.replace("/", "_")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf_out:
        zf_out.writestr(f"{safe_name}.json", json.dumps(scene_data))
        for orig_name, data in other_files.items():
            ext = os.path.splitext(orig_name)[1]
            zf_out.writestr(f"{safe_name}{ext}", data)
    return buf.getvalue()


def _scenescape_import_scene(app_dir: str, scene_name: str, camera_name: str) -> tuple:
    """Import scene+camera via SceneScape REST API. Returns (scene_uid, camera_name) or (None,None)."""
    import urllib.error
    import urllib.request

    base_url, ctx, token = _scenescape_get_client(app_dir)
    if not token:
        return None, None

    try:
        zone_cfg = read_zone_config(app_dir)
        scene_zip_name = zone_cfg.get("scene_zip", "conference-room.zip")
    except Exception:
        scene_zip_name = "conference-room.zip"

    zip_path = str(Path(app_dir) / ".." / "scenescape" / "webserver" / scene_zip_name)
    if not Path(zip_path).exists():
        logger.warning("Base scene ZIP not found at %s — falling back to scene-import", zip_path)
        return None, None

    try:
        zip_bytes = _clone_scene_zip(zip_path, scene_name, camera_name)
    except Exception as e:
        logger.warning("Failed to clone scene ZIP: %s", e)
        return None, None

    boundary = "----BenchmarkFormBoundary"
    filename = f"{scene_name.replace(' ', '-')}.zip"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="zipFile"; filename="{filename}"\r\n'
        f"Content-Type: application/zip\r\n\r\n"
    ).encode() + zip_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{base_url}/import-scene/",
        data=body,
        headers={
            "Authorization": f"Token {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
            resp_data = json.loads(resp.read())
            scene_errors = resp_data.get("scene")
            if scene_errors is not None:
                logger.warning("SceneScape import-scene scene error: %s", scene_errors)
                return None, None

            scene_uid = ""
            cameras = resp_data.get("cameras") or []
            for cam_entry in cameras:
                entries = cam_entry if isinstance(cam_entry, list) else [cam_entry]
                for entry in entries:
                    if isinstance(entry, dict) and entry.get("scene"):
                        scene_uid = entry["scene"]
                        break
                if scene_uid:
                    break

            if not scene_uid:
                try:
                    list_req = urllib.request.Request(
                        f"{base_url}/scenes", headers={"Authorization": f"Token {token}"})
                    with urllib.request.urlopen(list_req, context=ctx, timeout=10) as r:
                        for s in json.loads(r.read()).get("results", []):
                            if s.get("name") == scene_name:
                                scene_uid = s["uid"]
                                break
                except Exception:
                    pass

            logger.info("SceneScape scene+camera imported: %s / %s (uid=%s)",
                        scene_name, camera_name, scene_uid)
            return scene_uid or scene_name, camera_name
    except urllib.error.HTTPError as e:
        logger.warning("SceneScape import-scene → HTTP %s: %s", e.code, e.read().decode()[:200])
        return None, None
    except Exception as e:
        logger.warning("SceneScape import-scene → %s", e)
        return None, None


def _delete_cloned_scenes(app_dir: str) -> None:
    """Delete cloned scenes/cameras (those with -N suffix) via SceneScape REST API."""
    import urllib.error
    import urllib.request

    base_url, ctx, token = _scenescape_get_client(app_dir)
    if not token:
        logger.warning("Could not authenticate — skipping scene cleanup")
        return

    auth_header = {"Authorization": f"Token {token}"}

    req = urllib.request.Request(f"{base_url}/scenes", headers=auth_header)
    try:
        with urllib.request.urlopen(req, context=ctx) as resp:
            scenes = json.loads(resp.read()).get("results", [])
    except Exception:
        return

    for scene in scenes:
        name = scene.get("name", "")
        uid = scene.get("uid", "")
        if re.search(r'-\d+$', name):
            logger.info("Deleting cloned scene: %s (%s)", name, uid)
            req = urllib.request.Request(
                f"{base_url}/scene/{uid}", method="DELETE", headers=auth_header)
            try:
                urllib.request.urlopen(req, context=ctx)
            except urllib.error.HTTPError as e:
                logger.warning("  DELETE failed (%s): %s", e.code, e.reason)

    req = urllib.request.Request(f"{base_url}/cameras", headers=auth_header)
    try:
        with urllib.request.urlopen(req, context=ctx) as resp:
            cameras_list = json.loads(resp.read()).get("results", [])
    except Exception:
        cameras_list = []

    for cam in cameras_list:
        cam_name = cam.get("name", "")
        cam_id = cam.get("uid", cam.get("id", cam.get("sensor_id", "")))
        if re.search(r'-\d+$', cam_name) and cam_id:
            logger.info("Deleting orphaned camera: %s (%s)", cam_name, cam_id)
            req = urllib.request.Request(
                f"{base_url}/camera/{cam_id}", method="DELETE", headers=auth_header)
            try:
                urllib.request.urlopen(req, context=ctx)
            except urllib.error.HTTPError as e:
                logger.warning("  DELETE camera failed (%s): %s", e.code, e.reason)


# ---------------------------------------------------------------------------
# Wait helpers
# ---------------------------------------------------------------------------

def _wait_for_web_healthy(timeout: int = 300) -> None:
    candidates = ["storewide-lp-web-1", "scenescape-web-1"]
    for attempt in range(timeout // 5):
        for name in candidates:
            result = subprocess.run(
                f"docker inspect {name} --format '{{{{.State.Health.Status}}}}'",
                shell=True, capture_output=True, text=True)
            status = result.stdout.strip()
            if status == "healthy":
                logger.info("Web container (%s) is healthy (after %ds)", name, attempt * 5)
                return
        if attempt % 6 == 0:
            logger.info("  web status: %s  (waiting…)", status)
        time.sleep(5)
    logger.warning("Web container did not become healthy after %ds — continuing anyway", timeout)


def _wait_for_scene_import_completion(timeout: int = 180) -> None:
    logger.info("Waiting for scene-import to complete (timeout=%ds) …", timeout)
    deadline = time.time() + timeout
    while time.time() < deadline:
        running = subprocess.run(
            "docker ps -q --filter 'name=scene-import' --filter 'status=running'",
            shell=True, capture_output=True, text=True,
        ).stdout.strip()
        if running:
            elapsed = int(timeout - (deadline - time.time()))
            if elapsed % 30 < 6:
                logger.info("  scene-import still running … (%ds elapsed)", elapsed)
            time.sleep(5)
            continue

        exited = subprocess.run(
            "docker ps -aq --filter 'name=scene-import' --filter 'status=exited'",
            shell=True, capture_output=True, text=True,
        ).stdout.strip()
        if exited:
            code_result = subprocess.run(
                f"docker inspect {exited.splitlines()[0]} --format '{{{{.State.ExitCode}}}}'",
                shell=True, capture_output=True, text=True,
            )
            exit_code = code_result.stdout.strip()
            if exit_code == "0":
                logger.info("scene-import completed successfully")
            else:
                logger.warning("scene-import exited with code %s — camera registration may be incomplete", exit_code)
            return
        time.sleep(3)

    logger.warning("scene-import did not complete within %ds", timeout)


def _wait_for_camera_rtsp_ready(camera_name: str, timeout: int = 60) -> bool:
    logger.info("Waiting for RTSP stream camera=%s to be ready (timeout=%ds) …", camera_name, timeout)
    svc_idx = camera_name.split("-")[-1] if "-" in camera_name else "3"
    container_name = f"storewide-lp-lp-cams-{svc_idx}-1"

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            result = subprocess.run(
                "docker exec storewide-lp-mediaserver-1 "
                "wget -qO- 'http://localhost:9997/v3/paths/list'",
                shell=True, capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and camera_name in result.stdout:
                logger.info("RTSP stream camera=%s is ready (confirmed via MediaMTX API)", camera_name)
                return True
        except Exception:
            pass

        running = subprocess.run(
            f"docker inspect {container_name} --format '{{{{.State.Running}}}}'",
            shell=True, capture_output=True, text=True,
        ).stdout.strip()
        if running == "true":
            time.sleep(3)
            logger.info("RTSP container %s is running — stream likely ready", container_name)
            return True

        elapsed = int(timeout - (deadline - time.time()))
        if elapsed % 15 < 4:
            logger.info("  RTSP camera=%s not ready yet … (%ds elapsed)", camera_name, elapsed)
        time.sleep(3)

    logger.warning("RTSP stream camera=%s not confirmed within %ds — DLStreamer may fail to connect",
                   camera_name, timeout)
    return False


def _wait_for_first_detection(timeout: int = 60, poll_interval: int = 3,
                               camera_filter: Optional[str] = None) -> bool:
    camera_label = f" from camera={camera_filter}" if camera_filter else ""
    since = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    deadline = time.time() + timeout
    attempt = 0
    while time.time() < deadline:
        result = subprocess.run(
            f"docker logs --since {since} poi-backend 2>&1",
            shell=True, capture_output=True, text=True)
        output = result.stdout + result.stderr
        is_detection = (
            "POI match" in output or
            "face embedding" in output.lower() or
            "detections" in output.lower() or
            "poi_detections" in output
        )
        if is_detection:
            if camera_filter is None or f"camera={camera_filter}" in output:
                elapsed = int(timeout - (deadline - time.time()))
                logger.info("First detection seen%s after ~%ds — pipeline is warm",
                            camera_label, elapsed)
                return True
        if attempt % 4 == 0:
            remaining = int(deadline - time.time())
            logger.info("  Waiting for first detection%s … (%ds remaining)", camera_label, remaining)
        attempt += 1
        time.sleep(poll_interval)
    logger.warning("No detection%s seen within %ds — proceeding anyway", camera_label, timeout)
    return False


# ---------------------------------------------------------------------------
# Main scaling entry point
# ---------------------------------------------------------------------------

def scale_pipeline_services(app_dir: str, num_scenes: int,
                             wait: int = 90, resource_config: str = "") -> None:
    """Scale the POI video pipeline to *num_scenes* scenes.

    Steps:
      1. Update stream_density in zone_config.json
      2. Generate docker-compose.cameras.yaml
      3. Re-run init.sh to update .env
      4. Write STREAM_DENSITY + BASE_CAMERA_COUNT to docker/.env
      5. Generate per-camera DLStreamer pipeline configs
      6. Bring up new camera services
      7. Wait for web container healthy
      8. Clean stale scenes, register new scene via SceneScape API
      9. Recreate lp-video (DLStreamer)
     10. Wait for first detection (warm-up)
    """
    logger.info("Scaling POI to %d scene(s) …", num_scenes)

    set_stream_density(app_dir, num_scenes)
    generate_cameras_override(app_dir, num_scenes)
    reinit_env(app_dir, resource_config=resource_config)

    env_file = os.path.join(app_dir, "docker", ".env")
    write_env_var(env_file, "STREAM_DENSITY", str(num_scenes))
    write_env_var(env_file, "BASE_CAMERA_COUNT", "2")

    generate_dlstreamer_config(app_dir, num_scenes)

    base_camera_count = 2
    new_cam_services: list[str] = []
    if num_scenes > 1:
        new_cam = get_new_camera_name(app_dir, num_scenes)
        if new_cam:
            cam_idx = int(new_cam.split("-")[-1])
            new_cam_services = [f"lp-cams-{cam_idx}", f"lp-video-{cam_idx}"]

    if new_cam_services:
        logger.info("Removing stale containers for new services: %s …", " ".join(new_cam_services))
        docker_compose(app_dir, f"rm -f {' '.join(new_cam_services)}")

    logger.info("Starting new camera streams …")
    docker_compose(app_dir, "up -d --no-recreate --remove-orphans")

    project = "storewide-lp"
    if num_scenes > 1:
        new_vol_idx = base_camera_count + (num_scenes - 1)
        vol_name = f"{project}_vol-dlstreamer-pipeline-root-{new_vol_idx}"
        logger.info("Initialising volume %s for DLStreamer uid=1999 …", vol_name)
        subprocess.run(
            f"docker run --rm -v {vol_name}:/data alpine sh -c "
            f"'chmod a+rwxt /data && "
            f"mkdir -p /data/user_defined_pipelines && "
            f"chown 1999:1999 /data/user_defined_pipelines'",
            shell=True, capture_output=True, text=True,
        )

    if new_cam_services:
        cam_svc = new_cam_services[0]
        logger.info("Force-starting camera stream service %s …", cam_svc)
        docker_compose(app_dir, f"up -d --force-recreate {cam_svc}")

    _wait_for_web_healthy()
    _delete_cloned_scenes(app_dir)

    if num_scenes > 1:
        new_cam = get_new_camera_name(app_dir, num_scenes)
        zone_cfg = read_zone_config(app_dir)
        base_scene_name = zone_cfg.get("scene_name", "conference room")
        new_scene_name = f"{base_scene_name}-{num_scenes}"
        if new_cam:
            logger.info("Registering scene=%s camera=%s via SceneScape REST API …",
                        new_scene_name, new_cam)
            scene_uid, cam_uid = _scenescape_import_scene(app_dir, new_scene_name, new_cam)
            if not scene_uid:
                logger.warning("SceneScape API import failed — falling back to scene-import sidecar")
                docker_compose(app_dir, "rm -f -s scene-import")
                docker_compose(app_dir, "up -d scene-import")
                _wait_for_scene_import_completion(timeout=180)

    if num_scenes > 1:
        new_cam = get_new_camera_name(app_dir, num_scenes)
        if new_cam:
            _wait_for_camera_rtsp_ready(new_cam, timeout=60)

    logger.info("Recreating DLStreamer container(s) for %d scene(s) …", num_scenes)
    if num_scenes == 1:
        video_services = "lp-video"
    else:
        new_vid_idx = base_camera_count + (num_scenes - 1)
        video_services = f"lp-video lp-video-{new_vid_idx}"
    docker_compose(app_dir, f"up -d --force-recreate --remove-orphans {video_services}")

    new_cam_for_warmup = get_new_camera_name(app_dir, num_scenes)
    _wait_for_first_detection(timeout=wait, poll_interval=3, camera_filter=new_cam_for_warmup)
    logger.info("Pipeline warm — adding 10s stabilisation buffer …")
    time.sleep(10)
