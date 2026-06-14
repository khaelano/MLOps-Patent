# Multi-stage build for a self-contained Patent Anomaly Detection inference server.
# Embeds the LSHiForest model, drift baseline, ONNX embedder, and source code
# — zero runtime dependencies (no MLflow, no S3, no HuggingFace).

# ── Build args ────────────────────────────────────────────────────────────────
ARG MODEL_SOURCE=./model

# ── Stage 1: Builder ───────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build dependencies (pip, build tools for compiling wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests only — source changes won't invalidate this layer
COPY pyproject.toml uv.lock README.md LICENSE /build/

# Create a minimal package stub so flit_core can discover the project
# (needed by uv export to read metadata; real source overwrites this later)
RUN mkdir -p patent && touch patent/__init__.py

# Install pip and uv
RUN python -m pip install --upgrade pip uv

# Export all dependencies from lockfile to a pip-compatible requirements file
# then install them into the system Python
RUN uv export --no-dev --no-hashes -o /tmp/requirements.txt && \
    uv pip install --system --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Now copy source code — this layer invalidates on source changes
COPY patent/ /build/patent/

# Install the project package (deps already cached, fast)
RUN uv pip install --system --no-cache-dir -e .

# ── Pre-download the ONNX embedder model ─────────────────────────────────────
# The AllMiniLML6V2Q model (~5 MB, 4-bit quantized) is fetched from HuggingFace
# at build time so the container never needs internet access.
# Pass --build-arg HF_TOKEN=$(hf_token) for authenticated requests
# (much higher rate limits — avoids 429 errors on shared CI runners).
ARG HF_TOKEN
COPY docker/app/download_embedder.py /tmp/download_embedder.py
RUN --mount=type=secret,id=hf_token,required=false \
    if [ -f /run/secrets/hf_token ] && [ -s /run/secrets/hf_token ]; then \
        export HF_TOKEN=$(cat /run/secrets/hf_token); \
        echo "HF_TOKEN set (length=${#HF_TOKEN})"; \
    else \
        echo "No HF_TOKEN provided — using anonymous access"; \
    fi; \
    python /tmp/download_embedder.py && \
    rm /tmp/download_embedder.py

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

# ── Embed LSHiForest model + drift baseline ────────────────────────────────
# Downloaded by CI/Makefile before docker build, placed in build context as ./model/
COPY model /app/model/

ENV MODEL_PATH=/app/model/model.lshif \
    MODEL_VERSION_PATH=/app/model/model_version.txt \
    DRIFT_BASELINE_DIR=/app/model/drift_baseline

# Copy the pre-downloaded ONNX embedder cache to appuser's default HF cache path
# (no HF_HOME env var needed — embed-anything finds the default path)
COPY --from=builder /root/.cache/huggingface /home/appuser/.cache/huggingface

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /home/appuser/.cache/huggingface /app

USER appuser

# Health check
HEALTHCHECK --interval=15s --timeout=10s --retries=5 --start-period=30s \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the FastAPI app via uvicorn
ENTRYPOINT ["python", "-m", "uvicorn"]
CMD ["patent.api:app", "--host", "0.0.0.0", "--port", "8000"]
