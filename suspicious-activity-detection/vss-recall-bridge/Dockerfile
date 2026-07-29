# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY third_party_programs_vss-recall-bridge.txt ./
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app


ENV CLIPS_DIR=/clips
VOLUME ["/clips"]
EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
