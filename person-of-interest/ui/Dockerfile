# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ── Stage 1: Build ────────────────────────────────────────
FROM node:20-slim AS builder

WORKDIR /app
COPY ui/third_party_programs_poi-ui-service.txt ./
COPY ui/package.json ui/tsconfig.json ui/vite.config.ts ./
RUN npm install --frozen-lockfile 2>/dev/null || npm install

COPY ui/ ./
RUN npm run build

# ── Stage 2: Serve ───────────────────────────────────────
FROM nginx:1.27

# Apply latest security patches
RUN apt-get update && \
    apt-get upgrade -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/dist /usr/share/nginx/html

# SPA fallback
RUN echo 'server { \
    listen 80; \
    root /usr/share/nginx/html; \
    client_max_body_size 10m; \
    location / { try_files $uri $uri/ /index.html; } \
    location /api/ { proxy_pass http://poi-backend:8000; proxy_read_timeout 300s; proxy_connect_timeout 60s; proxy_send_timeout 120s; } \
    location /uploads/ { proxy_pass http://poi-backend:8000/uploads/; proxy_set_header Host $host; } \
    location /ws/ { proxy_pass http://poi-backend:8000; proxy_http_version 1.1; proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade"; } \
    location /alerts/ws { proxy_pass http://poi-alert-service:8000/api/v1/ws; proxy_http_version 1.1; proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade"; proxy_read_timeout 3600s; } \
}' > /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
