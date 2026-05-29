"""SceneScape REST API adapter."""

from __future__ import annotations

import logging
from typing import Optional

import requests
import urllib3

from backend.core.config import get_config

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

log = logging.getLogger("poi.scenescape")


class ScenescapeAPIAdapter:
    """Adapter Pattern — wraps the SceneScape REST API."""

    def __init__(self) -> None:
        self._cfg = get_config()
        self._base_url = self._cfg.scenescape_api_url.rstrip("/")
        self._token = self._cfg.scenescape_api_token
        self._session = requests.Session()
        self._session.verify = False
        # Bypass any HTTP proxy for internal SceneScape traffic.
        # Python requests does not support wildcard NO_PROXY entries (e.g. *.intel.com),
        # so we set proxies explicitly on the session.
        self._session.proxies = {"http": None, "https": None}
        if not self._token and self._cfg.scenescape_api_user and self._cfg.scenescape_api_password:
            self._token = self._fetch_token(
                self._cfg.scenescape_api_user,
                self._cfg.scenescape_api_password,
            )
        if self._token:
            self._session.headers["Authorization"] = f"Token {self._token}"

    def _fetch_token(self, username: str, password: str) -> str:
        """Obtain an auth token from the SceneScape API using username/password."""
        if not self._base_url:
            return ""
        try:
            resp = self._session.post(
                f"{self._base_url}/api/v1/auth",
                json={"username": username, "password": password},
                timeout=10,
            )
            resp.raise_for_status()
            token = resp.json().get("token", "")
            if token:
                log.info("Obtained SceneScape API token for user '%s'", username)
            return token
        except requests.RequestException:
            log.exception("Failed to obtain SceneScape API token for user '%s'", username)
            return ""

    def _refresh_token(self) -> bool:
        """Re-authenticate and update the session header. Returns True on success."""
        if not (self._cfg.scenescape_api_user and self._cfg.scenescape_api_password):
            return False
        self._token = self._fetch_token(
            self._cfg.scenescape_api_user,
            self._cfg.scenescape_api_password,
        )
        if self._token:
            self._session.headers["Authorization"] = f"Token {self._token}"
            return True
        return False

    def _get_with_retry(self, url: str, timeout: int = 10) -> requests.Response:
        """GET with automatic token refresh on 401."""
        resp = self._session.get(url, timeout=timeout)
        if resp.status_code == 401 and self._refresh_token():
            resp = self._session.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp

    def list_cameras(self) -> list[dict]:
        if not self._base_url:
            log.warning("SceneScape API URL not configured")
            return []
        try:
            resp = self._get_with_retry(f"{self._base_url}/api/v1/cameras")
            data = resp.json()
            if isinstance(data, list):
                return data
            return data.get("results", data.get("cameras", []))
        except requests.RequestException:
            log.exception("Failed to fetch cameras from SceneScape")
            return []

    def get_camera(self, camera_id: str) -> Optional[dict]:
        if not self._base_url:
            return None
        try:
            resp = self._get_with_retry(f"{self._base_url}/api/v1/cameras/{camera_id}")
            return resp.json()
        except requests.RequestException:
            log.exception("Failed to fetch camera %s", camera_id)
            return None
