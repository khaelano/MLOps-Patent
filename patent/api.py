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
``POST /predict``
    Accepts ``{"texts": ["title abstract", ...]}`` and returns anomaly scores.
``GET /metrics``
    Prometheus metrics (auto-instrumented HTTP + custom application metrics).
``GET /drift``
    Dedicated endpoint for application-level Prometheus metrics.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from tempfile import TemporaryDirectory
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from loguru import logger
import mlflow
import numpy as np
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from patent.dataset.embedders import get_embedder
from patent.lshiforest import LSHiForest, rescale_scores
from patent.monitoring.metrics import (
    DATA_EMBEDDING_DIM,
    METRICS_REGISTRY,
    MODEL_INFO,
)

MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_MODEL_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "patent-lshiforest")
EMBEDDER_SPEC: str = os.getenv("EMBEDDER_SPEC", "embed-anything-onnx:AllMiniLML6V2Q")

_model: LSHiForest | None = None
_embedder: Any = None
_model_version: str | None = None

PREDICTION_COUNT = Counter(
    "patent_predictions_total",
    "Total number of predictions served.",
    registry=METRICS_REGISTRY,
)

PREDICTION_SCORE_HIST = Histogram(
    "patent_prediction_scores",
    "Distribution of raw anomaly scores from predictions.",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=METRICS_REGISTRY,
)


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

    logger.info(f"Loading embedder: {EMBEDDER_SPEC}")
    _embedder = get_embedder(EMBEDDER_SPEC)
    logger.info(f"Embedder loaded (dim={_embedder.embedding_dim})")

    logger.info(f"Loading model '{MLFLOW_MODEL_NAME}' from MLflow Registry ...")
    _model, _model_version = _load_model_from_registry()
    logger.success(
        f"Model v{_model_version} loaded ({_model.n_trees} trees, family={_model.family_name})"
    )

    MODEL_INFO.labels(
        model_version=_model_version or "unknown",
        model_name=MLFLOW_MODEL_NAME,
    ).set(1)
    DATA_EMBEDDING_DIM.set(_embedder.embedding_dim)

    yield

    if _embedder is not None:
        _embedder.stop_pool()
    logger.info("Inference server shutdown complete")


app = FastAPI(
    title="Patent Anomaly Detection API",
    description="Score paper novelty via LSHiForest anomaly detection",
    version="0.1.0",
    lifespan=lifespan,
)

Instrumentator(
    excluded_handlers=["/metrics", "/health"],
    registry=METRICS_REGISTRY,
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=True)


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

    try:
        embeddings = _embedder.encode(req.texts, show_progress=False)
    except Exception as exc:
        logger.error(f"Embedding failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}")

    X = np.asarray(embeddings, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    try:
        raw_scores = _model.score(X)
        rescaled = rescale_scores(raw_scores)
    except Exception as exc:
        logger.error(f"Scoring failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}")

    PREDICTION_COUNT.inc(len(req.texts))
    for val in raw_scores:
        if np.isfinite(val):
            PREDICTION_SCORE_HIST.observe(float(val))

    return PredictResponse(
        scores=raw_scores.tolist(),
        rescaled_scores=rescaled.tolist(),
        model_version=_model_version or "unknown",
    )


@app.get("/drift")
async def drift_metrics() -> Response:
    """Expose custom drift and model metrics in Prometheus text format.

    This endpoint contains application-level metrics (drift gauges, model
    info, prediction counters) scoped to the dedicated ``METRICS_REGISTRY``,
    separate from the auto-instrumented HTTP metrics.
    """
    return Response(
        content=generate_latest(METRICS_REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )
