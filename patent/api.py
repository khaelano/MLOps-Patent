"""FastAPI inference server for the LSHiForest anomaly detection model.

Fetches the latest **Production** model from the MLflow Model Registry at
startup and serves anomaly scores for submitted text.

Configuration (environment variables)
-------------------------------------
``MLFLOW_TRACKING_URI``
    MLflow tracking server URI (default ``http://127.0.0.1:5000``).
``MLFLOW_MODEL_NAME``
    Registered model name to pull from the registry (default ``patent-lshiforest``).
``EMBEDDER_SPEC``
    Embedder spec ``<protocol>:<model>`` (default ``embed-anything-onnx:AllMiniLML6V2Q``).

Endpoints
---------
``GET /health``
    Liveness / readiness probe.  Returns model version and embedder info.
``GET /metrics``
    Prometheus metrics endpoint (drift gauges, score distribution histogram).
``POST /predict``
    Accepts ``{\"texts\": [\"title abstract\", ...]}`` and returns anomaly scores.
"""

from __future__ import annotations

from collections import deque
from contextlib import asynccontextmanager
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from loguru import logger
import numpy as np
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from patent.dataset.embedders import get_embedder
from patent.lshiforest import LSHiForest, rescale_scores

# ── Configuration (env vars with defaults) ──────────────────────────────────

MODEL_PATH: str | None = os.getenv("MODEL_PATH")
MODEL_VERSION_PATH: str | None = os.getenv("MODEL_VERSION_PATH")
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_MODEL_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "patent-lshiforest")
EMBEDDER_SPEC: str = os.getenv("EMBEDDER_SPEC", "embed-anything-onnx:AllMiniLML6V2Q")

# ── Score buffer for drift detection ───────────────────────────────────────
_SCORE_BUFFER_SIZE = int(os.getenv("DRIFT_SCORE_BUFFER_SIZE", "10000"))
_score_buffer: deque[float] = deque(maxlen=_SCORE_BUFFER_SIZE)

# ── GitHub Actions dispatch (CT pipeline trigger) ──────────────────────────
_GITHUB_DISPATCH_URL: str | None = os.getenv("GITHUB_DISPATCH_URL")
_GITHUB_DISPATCH_TOKEN: str | None = os.getenv("GITHUB_DISPATCH_TOKEN")
_GITHUB_DISPATCH_EVENT: str = os.getenv("GITHUB_DISPATCH_EVENT", "drift-detected")

# ── Global state — populated during lifespan startup ────────────────────────

_model: LSHiForest | None = None
_embedder: Any = None
_model_version: str | None = None


# ── Lifespan ────────────────────────────────────────────────────────────────


def _load_model() -> tuple[LSHiForest, str]:
    """Load the Production model from a local path or MLflow registry.

    When ``MODEL_PATH`` is set (embedded image), loads directly from disk.
    Otherwise falls back to querying the MLflow Model Registry.
    """
    if MODEL_PATH:
        logger.info(f"Loading model from {MODEL_PATH}")
        model = LSHiForest.load(MODEL_PATH)
        version = "embedded"
        if MODEL_VERSION_PATH and os.path.exists(MODEL_VERSION_PATH):
            version = Path(MODEL_VERSION_PATH).read_text().strip()
        return model, version
    return _load_model_from_registry()


def _load_model_from_registry() -> tuple[LSHiForest, str]:
    """Download the latest Production model artifact and deserialise it.

    Returns
    -------
    (LSHiForest, version_string)
    """
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    prod_versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])

    if not prod_versions:
        raise RuntimeError(
            f"No Production version found for model '{MLFLOW_MODEL_NAME}'. "
            "Train and register a model first."
        )

    prod = prod_versions[0]
    version = str(prod.version)
    run_id = prod.run_id
    logger.info(f"Found {MLFLOW_MODEL_NAME} v{version} (run {run_id}) in Production")

    artifact_uri = f"runs:/{run_id}/model.lshif"
    with TemporaryDirectory() as tmpdir:
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=artifact_uri,
            dst_path=tmpdir,
        )
        model_path = str(local_path) if isinstance(local_path, str) else str(local_path)
        logger.info(f"Loading model from {model_path}")
        model = LSHiForest.load(model_path)
        return model, version


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: connect to MLflow, load embedder, fetch Production model.

    Shutdown: release embedder resources.
    """
    global _model, _embedder, _model_version

    # Log embedder cache configuration
    hf_home = os.environ.get("HF_HOME", "<not set>")
    logger.info(f"HuggingFace cache directory: {hf_home}")

    logger.info(f"Loading embedder: {EMBEDDER_SPEC}")
    _embedder = get_embedder(EMBEDDER_SPEC)
    logger.info(f"Embedder loaded (dim={_embedder.embedding_dim})")

    logger.info(f"Loading model '{MLFLOW_MODEL_NAME}' ...")
    _model, _model_version = _load_model()
    logger.success(
        f"Model v{_model_version} loaded ({_model.n_trees} trees, family={_model.family_name})"
    )

    yield

    if _embedder is not None:
        _embedder.stop_pool()
    logger.info("Inference server shutdown complete")


# ── Application ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Patent Anomaly Detection API",
    description="Score paper novelty via LSHiForest anomaly detection",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ──────────────────────────────────────────────


class PredictRequest(BaseModel):
    texts: list[str]


class PredictResponse(BaseModel):
    scores: list[float]
    rescaled_scores: list[float]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str | None
    embedder: str


# ── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health() -> dict[str, Any]:
    """Liveness check — returns model and embedder metadata."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ok",
        "model_name": MLFLOW_MODEL_NAME,
        "model_version": _model_version,
        "embedder": EMBEDDER_SPEC,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    """Score a batch of texts for anomaly.

    Each text should be a paper title (the same format used during
    training).  Returns raw LSHiForest anomaly scores and percentile-
    rescaled scores (both ∈ [0, 1]).
    """
    from patent.monitoring.metrics import (
        INFERENCE_TIME,
        PREDICT_ERRORS,
        PREDICT_INFLIGHT,
        PREDICT_LATENCY,
        PREDICT_PAYLOAD_SIZE,
        PREDICT_REQUESTS,
        SCORE_ROLLING_MEAN,
    )

    # ── Payload size (kB) ──────────────────────────────────────────────
    payload_bytes = len(req.model_dump_json().encode("utf-8"))
    PREDICT_PAYLOAD_SIZE.observe(payload_bytes / 1024.0)

    t0 = time.perf_counter()
    PREDICT_REQUESTS.inc()
    PREDICT_INFLIGHT.inc()

    if _model is None or _embedder is None:
        PREDICT_ERRORS.inc()
        PREDICT_INFLIGHT.dec()
        raise HTTPException(status_code=503, detail="Model or embedder not loaded")

    if not req.texts:
        PREDICT_ERRORS.inc()
        PREDICT_INFLIGHT.dec()
        raise HTTPException(status_code=400, detail="No texts provided")

    # ── Embed ───────────────────────────────────────────────────────────
    t_infer = time.perf_counter()
    try:
        embeddings = _embedder.encode(req.texts, show_progress=False)
    except Exception as exc:
        PREDICT_ERRORS.inc()
        PREDICT_INFLIGHT.dec()
        logger.error(f"Embedding failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}")

    X = np.asarray(embeddings, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    # ── Score ───────────────────────────────────────────────────────────
    try:
        raw_scores = _model.score(X)
        rescaled = rescale_scores(raw_scores)
    except Exception as exc:
        PREDICT_ERRORS.inc()
        PREDICT_INFLIGHT.dec()
        logger.error(f"Scoring failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}")

    infer_elapsed = time.perf_counter() - t_infer
    INFERENCE_TIME.observe(infer_elapsed)

    # ── Record scores for drift monitoring ────────────────────────────────
    for s in raw_scores:
        _score_buffer.append(float(s))

    # Update rolling mean from the current buffer
    if _score_buffer:
        SCORE_ROLLING_MEAN.set(float(np.mean(list(_score_buffer))))

    PREDICT_LATENCY.observe(time.perf_counter() - t0)
    PREDICT_INFLIGHT.dec()

    return PredictResponse(
        scores=raw_scores.tolist(),
        rescaled_scores=rescaled.tolist(),
        model_version=_model_version or "unknown",
    )


@app.post("/dispatch")
async def dispatch_github() -> dict[str, Any]:
    """Relay a Grafana drift alert to the GitHub Actions Model CT pipeline.

    Receives the Grafana webhook payload (ignored — we only need it as a
    trigger signal), then sends a properly formatted ``repository_dispatch``
    request to the GitHub API.  The CT workflow listens for event type
    ``drift-detected``.

    Returns the GitHub API status code and body for debugging.
    """
    import requests as http_requests

    if not _GITHUB_DISPATCH_URL or not _GITHUB_DISPATCH_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="GITHUB_DISPATCH_URL and GITHUB_DISPATCH_TOKEN must be set",
        )

    try:
        resp = http_requests.post(
            _GITHUB_DISPATCH_URL,
            json={"event_type": _GITHUB_DISPATCH_EVENT},
            headers={
                "Authorization": f"Bearer {_GITHUB_DISPATCH_TOKEN}",
                "Accept": "application/vnd.github+json",
            },
            timeout=15,
        )
        return {"status": resp.status_code, "ok": resp.ok}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/metrics")
async def metrics() -> Response:
    """Expose Prometheus metrics for drift monitoring.

    Returns the plain-text Prometheus exposition format including all
    registered drift gauges, histograms, and model info.
    """
    from patent.monitoring import METRICS_REGISTRY

    # ── Opportunistically update drift metrics from buffered scores ─────────
    try:
        _maybe_update_drift_metrics()
    except Exception:
        logger.exception("Failed to update drift metrics — continuing anyway")

    return Response(
        content=generate_latest(METRICS_REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )


def _maybe_update_drift_metrics() -> None:
    """Run a drift check using buffered scores, if enough data is available.

    Score-distribution drift: KS test, Wasserstein distance, mean shift,
    and energy distance against the saved training baseline.

    Also samples process memory and CPU usage via *psutil* so the
    Grafana dashboard can show resource utilisation.
    """
    # ── Update process resource gauges ────────────────────────────────────
    try:
        import psutil

        from patent.monitoring.metrics import PROCESS_CPU_PERCENT, PROCESS_MEMORY_BYTES

        global _proc
        if "_proc" not in globals():
            _proc = psutil.Process()
            _proc.cpu_percent()  # prime the first measurement (returns 0)
        PROCESS_MEMORY_BYTES.set(_proc.memory_info().rss)
        PROCESS_CPU_PERCENT.set(_proc.cpu_percent())
    except Exception:
        pass

    if len(_score_buffer) < 100:
        return

    from patent.monitoring import (
        load_drift_baseline,
        update_drift_metrics,
    )
    from patent.monitoring.drift_detector import detect_drift

    baseline = load_drift_baseline(model_version=_model_version)

    new_scores = np.array(list(_score_buffer), dtype=np.float64)

    if baseline is not None:
        result = detect_drift(baseline.scores, new_scores)
        ks_stat = result["ks_statistic"]
        ks_pval = result["ks_pvalue"]
        mean_shift_val = result["mean_shift"]
        energy_dist = result["energy_distance"]
        drift_detected = result["drift_detected"]
    else:
        ks_stat = 0.0
        ks_pval = 1.0
        mean_shift_val = 0.0
        energy_dist = 0.0
        drift_detected = False

    update_drift_metrics(
        ks_statistic=ks_stat,
        ks_pvalue=ks_pval,
        mean_shift=mean_shift_val,
        energy_distance=energy_dist,
        drift_detected=drift_detected,
        n_samples=len(new_scores),
        scores=new_scores,
        model_version=_model_version,
        model_name=MLFLOW_MODEL_NAME,
    )

    # Clear the buffer so the next drift check only sees new scores.
    # Without this, old drifted scores persist in the 10k-entry deque
    # and contaminate subsequent checks even after normal data resumes.
    _score_buffer.clear()
