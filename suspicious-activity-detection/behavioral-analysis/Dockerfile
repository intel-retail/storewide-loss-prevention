FROM docker.io/intel/dlstreamer:latest

USER root
WORKDIR /app

# Copy only requirements.txt first for dependency caching
COPY requirements.txt .

# Install Python dependencies (cached unless requirements.txt changes)
# - Base image already provides numpy and openvino via apt — skip reinstalling them
# - Clean up afterwards to reduce layer size
RUN pip install --no-cache-dir --break-system-packages --ignore-installed -r requirements.txt \
    && find /usr/local/lib/python3.12/dist-packages -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.12/dist-packages -type d -name 'tests' -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.12/dist-packages -name '*.so' -exec strip --strip-unneeded {} + 2>/dev/null || true \
    && rm -rf /root/.cache /tmp/*

# Now copy actual source and config
COPY src/ ./src/
COPY config/ ./config/

# Extend PYTHONPATH to include our source (gstgva already on PYTHONPATH from base image)
ENV PYTHONPATH=${PYTHONPATH}:/app/src
ENV PYTHONIOENCODING=utf-8

# Expose port
EXPOSE 8080

# Run the service
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
