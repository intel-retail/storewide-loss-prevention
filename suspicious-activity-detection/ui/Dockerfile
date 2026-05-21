FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir pandas requests paho-mqtt Pillow fastapi uvicorn
COPY ui_gradio.py ./
EXPOSE 7860
CMD ["python", "ui_gradio.py"]