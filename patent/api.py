"""FastAPI inference server for the LSHiForest anomaly detection model.

Two operating modes, selected by environment variable:

**Docker / local mode** (``LSHIF_MODEL_PATH`` is set)
    Loads the model directly from a ``.lshif`` file on disk.  The embedder
    is loaded at startup.  No MLflow connectivity required.

**Registry mode** (``LSHIF_MODEL_PATH`` is *not* set — the default)
    Fetches the latest **Production** model from the MLflow Model Registry
    at startup and serves anomaly scores for submitted text.

Configuration (environment variables)
-------------------------------------
``LSHIF_MODEL_PATH``
    Path to a ``.lshif`` file to load at startup (Docker / local mode).
``LSHIF_MODEL_VERSION``
    Version label to report in ``/health`` when using local mode
    (default ``"local"``).
``MLFLOW_TRACKING_URI``
    MLflow tracking server URI (default ``http://127.0.0.1:5000``).
    Only used in registry mode.
``MLFLOW_MODEL_NAME``
    Registered model name (default ``patent-lshiforest``).  Only used in
    registry mode.
``EMBEDDER_SPEC``
    Embedder spec ``<protocol>:<model>`` (default
    ``embed-anything-onnx:AllMiniLML6V2Q``).

Endpoints
---------
``GET /health``
    Liveness / readiness probe.  Returns model version and embedder info.
``GET /ping``
    Minimal liveness probe (backward-compatible with MLflow's ``/ping``).
``POST /predict``
    Accepts ``{"texts": ["title abstract", ...]}`` and returns anomaly scores.
``GET /metrics``
    Prometheus scraping endpoint.  Exposed automatically by
    ``prometheus_fastapi_instrumentator``.

Metrics
-------
* ``http_requests_total``              – total requests (throughput)
* ``http_request_duration_seconds``    – inference latency histogram
* ``http_requests_inprogress``         – concurrent requests gauge
* ``patent_predictions_total``         – total texts scored
* ``patent_anomaly_score``             – raw anomaly score histogram
* ``patent_anomaly_score_rescaled``    – percentile-rescaled score histogram
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from loguru import logger
import numpy as np
from pydantic import BaseModel

from patent.dataset.embedders import get_embedder
from patent.lshiforest import LSHiForest, rescale_scores

# ── Configuration (env vars with defaults) ──────────────────────────────────

LSHIF_MODEL_PATH: str | None = os.getenv("LSHIF_MODEL_PATH")
LSHIF_MODEL_VERSION: str = os.getenv("LSHIF_MODEL_VERSION", "local")

MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_MODEL_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "patent-lshiforest")
EMBEDDER_SPEC: str = os.getenv("EMBEDDER_SPEC", "embed-anything-onnx:AllMiniLML6V2Q")

# ── Global state — populated during lifespan startup ────────────────────────

_model: LSHiForest | None = None
_embedder: Any = None
_model_version: str | None = None

# ── Prometheus metrics (lazily initialised) ────────────────────────────────

_PREDICTION_COUNT: Any = None
_PREDICTION_SCORE_HIST: Any = None
_PREDICTION_RESCALED_HIST: Any = None
_metrics_logger = logging.getLogger(__name__)


def _init_prometheus_metrics() -> None:
    """Create custom Prometheus metric objects (idempotent, no-op if unavailable)."""
    global _PREDICTION_COUNT, _PREDICTION_SCORE_HIST, _PREDICTION_RESCALED_HIST

    if _PREDICTION_COUNT is not None:
        return

    try:
        from prometheus_client import Counter, Histogram  # type: ignore[import-untyped]
    except ImportError:
        _metrics_logger.debug("prometheus_client not installed — custom metrics disabled")
        return

    _PREDICTION_COUNT = Counter(
        "patent_predictions_total",
        "Total number of predictions served (individual texts)",
    )
    _PREDICTION_SCORE_HIST = Histogram(
        "patent_anomaly_score",
        "Distribution of raw LSHiForest anomaly scores ∈ [0, 1]",
        buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    )
    _PREDICTION_RESCALED_HIST = Histogram(
        "patent_anomaly_score_rescaled",
        "Distribution of percentile-rescaled anomaly scores ∈ [0, 1]",
        buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    )


def _record_prediction_metrics(raw_scores: np.ndarray, rescaled: np.ndarray) -> None:
    """Record per-prediction metrics (no-op when prometheus_client is absent)."""
    if _PREDICTION_COUNT is not None:
        _PREDICTION_COUNT.inc(len(raw_scores))
    if _PREDICTION_SCORE_HIST is not None:
        for s in raw_scores:
            _PREDICTION_SCORE_HIST.observe(float(s))
    if _PREDICTION_RESCALED_HIST is not None:
        for s in rescaled:
            _PREDICTION_RESCALED_HIST.observe(float(s))


# ── Model loading ──────────────────────────────────────────────────────────


def _load_model_from_local(path_str: str) -> tuple[LSHiForest, str]:
    """Load an LSHiForest model from a ``.lshif`` file on disk.

    If a ``version.json`` file exists alongside the model, its ``"version"``
    field is used as the version label.  Otherwise ``LSHIF_MODEL_VERSION``
    (env, default ``"local"``) is used.
    """
    model_path = Path(path_str)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {path_str}")

    # Look for version metadata written by scripts/download_model.py
    version_meta = model_path.parent / "version.json"
    version = LSHIF_MODEL_VERSION
    if version_meta.exists():
        try:
            import json

            meta = json.loads(version_meta.read_text())
            version = meta.get("version", LSHIF_MODEL_VERSION)
            logger.info(f"Model version from {version_meta}: v{version}")
        except Exception:
            logger.warning(f"Could not parse {version_meta}, using default version")

    logger.info(f"Loading model from {model_path}")
    model = LSHiForest.load(str(model_path))
    return model, version


def _load_model_from_registry() -> tuple[LSHiForest, str]:
    """Download the latest Production model artifact and deserialise it."""
    from tempfile import TemporaryDirectory

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
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


# ── Lifespan ────────────────────────────────────────────────────────────────


async def _lifespan(app: FastAPI) -> Any:
    """Startup: load embedder and model.  Shutdown: release embedder resources."""
    global _model, _embedder, _model_version

    logger.info(f"Loading embedder: {EMBEDDER_SPEC}")
    _embedder = get_embedder(EMBEDDER_SPEC)
    logger.info(f"Embedder loaded (dim={_embedder.embedding_dim})")

    if LSHIF_MODEL_PATH:
        logger.info("Local mode — loading model from LSHIF_MODEL_PATH")
        _model, _model_version = _load_model_from_local(LSHIF_MODEL_PATH)
    else:
        logger.info(f"Registry mode — loading model '{MLFLOW_MODEL_NAME}' from MLflow Registry")
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
    lifespan=_lifespan,
)


# ── Wire Prometheus instrumentation ─────────────────────────────────────────


def _attach_prometheus(app: FastAPI) -> None:
    """Attach ``prometheus_fastapi_instrumentator`` to *app* if available."""
    try:
        from prometheus_fastapi_instrumentator import (
            Instrumentator,  # type: ignore[import-untyped]  # noqa: E501
        )
    except ImportError:
        logger.debug("prometheus_fastapi_instrumentator not installed — /metrics disabled")
        return

    # Route through Any to work around incomplete upstream type stubs.
    from typing import Any as _Any

    _inst: _Any = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=False,
        should_instrument_requests_inprogress=True,
        should_round_latency_decimals=True,
    )
    _inst.instrument(app)  # 7.x API: instrument() wraps the app; add() is for callbacks
    _inst.expose(app, endpoint="/metrics", include_in_schema=False)
    logger.info("Prometheus /metrics endpoint enabled")


_attach_prometheus(app)

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
    """Liveness / readiness check — returns model and embedder metadata."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ok",
        "model_name": MLFLOW_MODEL_NAME,
        "model_version": _model_version,
        "embedder": EMBEDDER_SPEC,
    }


@app.get("/ping")
async def ping() -> dict[str, str]:
    """Minimal liveness probe (MLflow ``/ping`` compat)."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


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

    # ── Record Prometheus metrics ───────────────────────────────────────
    _init_prometheus_metrics()
    _record_prediction_metrics(raw_scores, rescaled)

    return PredictResponse(
        scores=raw_scores.tolist(),
        rescaled_scores=rescaled.tolist(),
        model_version=_model_version or "unknown",
    )
