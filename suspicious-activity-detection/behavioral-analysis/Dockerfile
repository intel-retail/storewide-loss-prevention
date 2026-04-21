FROM docker.io/intel/dlstreamer:latest

USER root
WORKDIR /app

# Copy only pyproject.toml first for dependency caching
COPY pyproject.toml .
# Create minimal src package so pip can resolve the project
RUN mkdir -p src/behavioral_analysis && touch src/behavioral_analysis/__init__.py

# Install Python dependencies (cached unless pyproject.toml changes)
# --ignore-installed: base image has system numpy/openvino via apt (no RECORD),
# pip cannot uninstall them but the versions already satisfy our requirements.
RUN pip install --no-cache-dir --break-system-packages --ignore-installed .

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
