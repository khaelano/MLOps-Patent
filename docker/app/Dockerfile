# Multi-stage build for the Patent Anomaly Detection inference server.
# Exposes /health, /predict, and /metrics endpoints via FastAPI + uvicorn.

# ── Stage 1: Builder ───────────────────────────────────────────────────────────
FROM python:3.10-slim as builder

WORKDIR /build

# Install build dependencies (pip, build tools for compiling wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project
COPY . /build/

# Install dependencies into a virtual environment
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -e .

# ── Stage 2: Runtime ───────────────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the project source code
COPY --from=builder /build/patent /app/patent

# Create cache directories for embedder models and HuggingFace downloads
# This enables caching across container replicas via the volume mount in docker-compose.yml
RUN mkdir -p /app/.cache/huggingface && \
    mkdir -p /app/.cache/embed-anything

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=15s --timeout=10s --retries=5 --start-period=120s \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the FastAPI app via uvicorn
ENTRYPOINT ["python", "-m", "uvicorn"]
CMD ["patent.api:app", "--host", "0.0.0.0", "--port", "8000"]
