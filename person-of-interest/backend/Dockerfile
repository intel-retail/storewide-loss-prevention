# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ── Stage 1: Builder ──────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

COPY backend/requirements.txt .
COPY backend/third_party_programs_poi-service.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────
FROM python:3.12-slim

LABEL maintainer="intel"
LABEL description="POI Re-identification Backend"

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

# Install performance-tools for metrics logging
# NOTE: pinned to @main for now (testing). Before production, pin to the exact
# commit/tag matching the performance-tools submodule to avoid version drift
# between this pip-installed copy and the submodule used by Makefile targets.
RUN pip install --no-cache-dir git+https://github.com/intel-retail/performance-tools.git@main#subdirectory=benchmark-scripts

WORKDIR /app

COPY backend/ ./backend/

RUN mkdir -p /data/faiss /data/uploads

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "-m", "backend.main"]
