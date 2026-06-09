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
from tempfile import TemporaryDirectory
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from loguru import logger
import mlflow
import numpy as np
from pydantic import BaseModel

from patent.dataset.embedders import get_embedder
from patent.lshiforest import LSHiForest, rescale_scores

# ── Prometheus metrics support (optional) ──────────────────────────────────
_METRICS_AVAILABLE = False
CONTENT_TYPE_LATEST: str = ""
generate_latest: Any = None

try:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    _METRICS_AVAILABLE = True
except ImportError:
    pass

# ── Configuration (env vars with defaults) ──────────────────────────────────

MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_MODEL_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "patent-lshiforest")
EMBEDDER_SPEC: str = os.getenv("EMBEDDER_SPEC", "embed-anything-onnx:AllMiniLML6V2Q")

# ── Score buffer for drift detection ───────────────────────────────────────
_SCORE_BUFFER_SIZE = int(os.getenv("DRIFT_SCORE_BUFFER_SIZE", "10000"))
_score_buffer: deque[float] = deque(maxlen=_SCORE_BUFFER_SIZE)

# ── Global state — populated during lifespan startup ────────────────────────

_model: LSHiForest | None = None
_embedder: Any = None
_model_version: str | None = None


# ── Lifespan ────────────────────────────────────────────────────────────────


def _load_model_from_registry() -> tuple[LSHiForest, str]:
    """Download the latest Production model artifact and deserialise it.

    Returns
    -------
    (LSHiForest, version_string)
    """
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

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

    # Log embedder cache configuration
    hf_home = os.environ.get("HF_HOME", "<not set>")
    logger.info(f"HuggingFace cache directory: {hf_home}")

    logger.info(f"Loading embedder: {EMBEDDER_SPEC}")
    _embedder = get_embedder(EMBEDDER_SPEC)
    logger.info(f"Embedder loaded (dim={_embedder.embedding_dim})")

    logger.info(f"Loading model '{MLFLOW_MODEL_NAME}' from MLflow Registry ...")
    _model, _model_version = _load_model_from_registry()
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

    Each text should be a concatenation of title and abstract (the same
    format used during training).  Returns raw LSHiForest anomaly scores
    and percentile-rescaled scores (both ∈ [0, 1]).
    """
    if _model is None or _embedder is None:
        raise HTTPException(status_code=503, detail="Model or embedder not loaded")

    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    # ── Embed ───────────────────────────────────────────────────────────
    try:
        embeddings = _embedder.encode(req.texts, show_progress=False)
    except Exception as exc:
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
        logger.error(f"Scoring failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}")

    # ── Record scores for drift monitoring ────────────────────────────────
    for s in raw_scores:
        _score_buffer.append(float(s))

    return PredictResponse(
        scores=raw_scores.tolist(),
        rescaled_scores=rescaled.tolist(),
        model_version=_model_version or "unknown",
    )


@app.get("/metrics")
async def metrics() -> Response:
    """Expose Prometheus metrics for drift monitoring.

    Returns the plain-text Prometheus exposition format including all
    registered drift gauges, histograms, and model info.
    """
    if not _METRICS_AVAILABLE:
        raise HTTPException(status_code=501, detail="prometheus_client not installed")

    from patent.monitoring import METRICS_REGISTRY

    # ── Opportunistically update drift metrics from buffered scores ─────────
    _maybe_update_drift_metrics()

    return Response(
        content=generate_latest(METRICS_REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )


def _maybe_update_drift_metrics() -> None:
    """Run a drift check using buffered scores, if enough data is available."""
    if not _METRICS_AVAILABLE or len(_score_buffer) < 100:
        return

    from patent.monitoring import (
        compute_drift_metrics,
        load_drift_baseline,
        update_drift_metrics,
    )

    baseline = load_drift_baseline()

    new_scores = np.array(list(_score_buffer), dtype=np.float64)
    # Dummy embeddings — real embedding tracking would require buffering
    # embeddings too, but for a score-only drift check this suffices
    dummy_embeddings = np.zeros((len(new_scores), _embedder.embedding_dim), dtype=np.float32)

    report = compute_drift_metrics(
        new_embeddings=dummy_embeddings,
        new_scores=new_scores,
        baseline=baseline,
    )

    update_drift_metrics(
        ks_statistic=report.ks_statistic,
        ks_pvalue=report.ks_pvalue,
        mean_shift=report.mean_shift,
        emb_shift=report.embedding_mean_shift,
        n_samples=report.n_samples,
        scores=new_scores,
        model_version=_model_version,
        model_name=MLFLOW_MODEL_NAME,
    )
