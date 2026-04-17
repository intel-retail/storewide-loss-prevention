FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only pyproject.toml first for dependency caching
COPY pyproject.toml .
# Create minimal src package so pip can resolve the project
RUN mkdir -p src/behavioral_analysis && touch src/behavioral_analysis/__init__.py

# Install Python dependencies (cached unless pyproject.toml changes)
RUN pip install --no-cache-dir .

# Now copy actual source and config
COPY src/ ./src/
COPY config/ ./config/

# Set Python path
ENV PYTHONPATH=/app/src
# Expose port
EXPOSE 8080

# Run the service
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
