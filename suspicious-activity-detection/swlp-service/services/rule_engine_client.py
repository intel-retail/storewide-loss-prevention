# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""HTTP client for the Rule Engine Service.

Provides the same evaluate() interface as the local RuleEngine,
so the RuleEngineAdapter works identically in both modes.
"""

from typing import Any, Dict, List, Optional

import aiohttp
import structlog

from .config import ConfigService

logger = structlog.get_logger(__name__)


class Action:
    """Mirrors rule_engine.models.Action for HTTP responses."""

    def __init__(self, type: str, params: Dict[str, Any] = None, rule_id: str = ""):
        self.type = type
        self.params = params or {}
        self.rule_id = rule_id


class RuleEngineClient:
    """HTTP client for the Rule Engine Service.

    Drop-in replacement for the local RuleEngine — same evaluate() signature.
    """

    def __init__(self, config: ConfigService) -> None:
        svc_cfg = config.get_rule_service_config()
        self.base_url = svc_cfg.get("base_url", "http://rule-engine:8091")
        self.timeout = aiohttp.ClientTimeout(
            total=svc_cfg.get("timeout_seconds", 10)
        )
        self._session: Optional[aiohttp.ClientSession] = None
        self._rules_cache: List[dict] = []

        logger.info(
            "RuleEngineClient initialized",
            base_url=self.base_url,
        )

    async def initialize(self) -> None:
        """Create HTTP session and fetch initial rules list."""
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        await self._fetch_rules()

    async def close(self) -> None:
        if self._session:
            await self._session.close()

    async def _fetch_rules(self) -> None:
        """Fetch rules list for inspection methods (rules, get_rule, is_rule_enabled)."""
        try:
            async with self._session.get(f"{self.base_url}/api/v1/rules") as resp:
                if resp.status == 200:
                    self._rules_cache = await resp.json()
                    logger.info("Rules fetched from service", count=len(self._rules_cache))
        except Exception:
            logger.warning("Failed to fetch rules from service, using empty cache")

    @property
    def rules(self) -> List[dict]:
        return list(self._rules_cache)

    def get_rule(self, rule_id: str) -> Optional[dict]:
        for rule in self._rules_cache:
            if rule["id"] == rule_id:
                return dict(rule)
        return None

    def is_rule_enabled(self, rule_id: str) -> bool:
        rule = self.get_rule(rule_id)
        return rule is not None and rule.get("enabled", True)

    async def evaluate(
        self,
        event_type: str,
        zone_type: str,
        context: Dict[str, Any],
    ) -> List[Action]:
        """Call the Rule Engine Service to evaluate rules.

        Same signature as RuleEngine.evaluate() but async and over HTTP.
        """
        payload = {
            "event_type": event_type,
            "zone_type": zone_type,
            "context": context,
        }
        try:
            async with self._session.post(
                f"{self.base_url}/api/v1/evaluate", json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [
                        Action(
                            type=a["type"],
                            params=a.get("params", {}),
                            rule_id=a.get("rule_id", ""),
                        )
                        for a in data
                    ]
                else:
                    logger.error(
                        "Rule engine service error",
                        status=resp.status,
                        body=await resp.text(),
                    )
                    return []
        except Exception:
            logger.exception("Failed to call rule engine service")
            return []
