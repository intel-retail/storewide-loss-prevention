# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.9.22

COPY rule-engine/pyproject.toml /app/pyproject.toml
COPY rule-engine/server.py /app/server.py
COPY rule-engine/__init__.py rule-engine/engine.py rule-engine/loader.py rule-engine/models.py /app/rule_engine/

RUN uv sync

RUN useradd -m -u 1000 ruleuser && \
    chown -R ruleuser:ruleuser /app

USER ruleuser

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8091

HEALTHCHECK --interval=10s --timeout=5s --retries=3 --start-period=5s \
    CMD curl -f http://localhost:8091/health || exit 1

CMD ["python", "server.py"]
