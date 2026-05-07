# ── Stage 1: Builder ──────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

COPY backend/requirements.txt .
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
RUN pip install --no-cache-dir git+https://github.com/sainijit/performance-tools.git@poi-stream-density#subdirectory=benchmark-scripts

WORKDIR /app

COPY backend/ ./backend/

RUN mkdir -p /data/faiss /data/uploads

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "-m", "backend.main"]
