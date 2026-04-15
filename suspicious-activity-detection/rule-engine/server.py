# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Rule Engine HTTP Service — thin FastAPI wrapper around the generic rule engine.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import structlog
import uvicorn
from fastapi import FastAPI

from rule_engine import RuleEngine
from rule_engine.loader import load_rules

logger = structlog.get_logger(__name__)

app = FastAPI(title="Rule Engine Service", version="0.1.0")

_engine: RuleEngine | None = None


@app.on_event("startup")
def startup() -> None:
    global _engine
    rules_path = Path(os.environ.get("RULES_YAML", "/app/configs/rules.yaml"))

    # Variables for ${var:default} resolution — passed as JSON env var or defaults
    import json
    variables = json.loads(os.environ.get("RULES_VARIABLES", "{}"))

    _engine = RuleEngine(rules_path=rules_path, variables=variables)
    logger.info(
        "Rule Engine Service started",
        rules_path=str(rules_path),
        rules_loaded=len(_engine.rules),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "rules_loaded": len(_engine.rules) if _engine else 0}


@app.post("/api/v1/evaluate")
def evaluate(request: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Evaluate rules against the given event and context.

    Request body:
        {
            "event_type": "zone_entry",
            "zone_type": "RESTRICTED",
            "context": {"visited_checkout": false, ...}
        }

    Returns:
        [{"type": "alert", "params": {...}, "rule_id": "restricted_zone"}, ...]
    """
    actions = _engine.evaluate(
        event_type=request["event_type"],
        zone_type=request["zone_type"],
        context=request.get("context", {}),
    )
    return [
        {"type": a.type, "params": a.params, "rule_id": a.rule_id}
        for a in actions
    ]


@app.get("/api/v1/rules")
def list_rules() -> List[Dict[str, Any]]:
    """List all loaded rules."""
    return _engine.rules if _engine else []


@app.get("/api/v1/rules/{rule_id}")
def get_rule(rule_id: str) -> Dict[str, Any]:
    """Get a single rule by ID."""
    rule = _engine.get_rule(rule_id) if _engine else None
    if rule is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Rule '{rule_id}' not found")
    return rule


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8091)
