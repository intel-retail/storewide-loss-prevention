# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir pandas requests paho-mqtt Pillow fastapi uvicorn
COPY ui_gradio.py ./
COPY third_party_programs_swlp-ui.txt ./
EXPOSE 7860
CMD ["python", "ui_gradio.py"]