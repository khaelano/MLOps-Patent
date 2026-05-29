# ── Inference server for the LSHiForest anomaly detection model ────────────
#
# Build (requires model downloaded first):
#   uv run python scripts/download_model.py -o docker/app/model
#   docker build -t ghcr.io/khaelano/mlops-patent:latest -f docker/app/Dockerfile .
#
# Run:
#   docker run -p 8000:8000 ghcr.io/khaelano/mlops-patent:latest

FROM python:3.10-slim

# ── Install runtime Python dependencies ──────────────────────────────────
# No mlflow needed — the model is loaded from a local .lshif file.
# uvicorn without [standard] extras avoids pulling uvloop, httptools,
# websockets, watchfiles (~17 MB).  sympy / mpmath / setuptools are
# onnxruntime tool deps not needed at inference time (~55 MB).
RUN pip install --no-cache-dir \
    fastapi>=0.115.0 \
    "uvicorn>=0.34.0" \
    "embed-anything>=0.7.0" \
    numpy \
    loguru \
    pydantic \
    "prometheus-client>=0.22.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    && pip uninstall -y sympy mpmath setuptools 2>/dev/null; true

# ── Copy the patent package ──────────────────────────────────────────────
COPY patent/ /app/patent/

# ── Copy model artifacts ─────────────────────────────────────────────────
# These must exist before the build (run scripts/download_model.py first).
COPY docker/app/model/ /app/model/

# ── Runtime configuration ────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV LSHIF_MODEL_PATH=/app/model/model.lshif
ENV HF_HOME=/app/.cache

WORKDIR /app

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=5 --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

ENTRYPOINT ["uvicorn", "patent.api:app", "--host", "0.0.0.0", "--port", "8000"]
