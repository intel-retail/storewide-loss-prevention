FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Download YOLO model at build time (optional, will auto-download at runtime)
# RUN python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"

# Set Python path
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 8080

# Run the service
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
