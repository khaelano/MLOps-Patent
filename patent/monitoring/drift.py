"""Data and prediction drift detection for the LSHiForest anomaly model.

The drift baseline is captured at training time (or at the first drift check) and
persisted to disk so that subsequent drift checks always compare against the
same reference.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

from loguru import logger
import numpy as np

from patent.config import INTERIM_DATA_DIR

# ── Baseline storage paths ────────────────────────────────────────────────────

_DRIFT_BASELINE_DIR = os.getenv("DRIFT_BASELINE_DIR")
BASELINE_DIR = (
    Path(_DRIFT_BASELINE_DIR) if _DRIFT_BASELINE_DIR else INTERIM_DATA_DIR / "drift_baseline"
)
EMBEDDING_BASELINE_PATH = BASELINE_DIR / "embedding_baseline.npz"
SCORE_BASELINE_PATH = BASELINE_DIR / "score_baseline.npz"
BASELINE_META_PATH = BASELINE_DIR / "baseline_meta.json"

# ── Dataclasses ───────────────────────────────────────────────────────────────


class Baseline:
    """Reference baseline captured from the Production model's training data.

    Attributes
    ----------
    embeddings : np.ndarray
        Reference embeddings (n_samples, embedding_dim) float32.
    scores : np.ndarray
        Reference anomaly scores (n_samples,) float64.
    model_version : str | None
        MLflow model version this baseline corresponds to.
    timestamp : float
        Unix timestamp when the baseline was captured.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        scores: np.ndarray,
        model_version: str | None = None,
        timestamp: float | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.scores = scores
        self.model_version = model_version
        self.timestamp = timestamp or time.time()

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def n_samples(self) -> int:
        return int(self.embeddings.shape[0])


# ── Baseline persistence ──────────────────────────────────────────────────────


def save_drift_baseline(
    embeddings: np.ndarray,
    scores: np.ndarray,
    model_version: str | None = None,
) -> Baseline:
    """Persist a drift baseline to disk.

    Saves embeddings (as compressed NPZ), scores, and metadata so that
    future drift checks compare against the same reference.

    Parameters
    ----------
    embeddings : ndarray of shape (n, d)
        Float32 embeddings from the training/production dataset.
    scores : ndarray of shape (n,)
        Float64 anomaly scores from the Production model.
    model_version : str | None
        MLflow model version string.

    Returns
    -------
    Baseline
        The persisted baseline object.
    """
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    baseline = Baseline(embeddings=embeddings, scores=scores, model_version=model_version)

    logger.info(
        f"Saving drift baseline: {baseline.n_samples} samples, "
        f"{baseline.embedding_dim}d, model v{baseline.model_version}"
    )

    np.savez_compressed(EMBEDDING_BASELINE_PATH, embeddings=embeddings)
    np.savez_compressed(SCORE_BASELINE_PATH, scores=scores)

    _save_baseline_meta(
        {
            "model_version": baseline.model_version,
            "n_samples": baseline.n_samples,
            "embedding_dim": baseline.embedding_dim,
            "timestamp": baseline.timestamp,
        }
    )

    logger.success(f"Drift baseline saved to {BASELINE_DIR}")
    return baseline


def load_drift_baseline(model_version: str | None = None) -> Baseline | None:
    """Load a previously saved drift baseline or return ``None``.

    *model_version* is accepted for API compatibility with callers that pass
    the current model version.  Currently ignored — the baseline is loaded
    from the on-disk NPZ files regardless of version.
    """
    if not EMBEDDING_BASELINE_PATH.exists() or not SCORE_BASELINE_PATH.exists():
        logger.info("No existing drift baseline found.")
        return None

    logger.info(f"Loading drift baseline from {BASELINE_DIR}")

    try:
        emb_data = np.load(EMBEDDING_BASELINE_PATH)
        score_data = np.load(SCORE_BASELINE_PATH)
        meta = _load_baseline_meta()
    except Exception:
        logger.exception("Failed to load drift baseline")
        return None

    return Baseline(
        embeddings=emb_data["embeddings"],
        scores=score_data["scores"],
        model_version=meta.get("model_version"),
        timestamp=meta.get("timestamp"),
    )


def _save_baseline_meta(data: dict[str, Any]) -> None:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_META_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _load_baseline_meta() -> dict[str, Any]:
    if BASELINE_META_PATH.exists():
        with open(BASELINE_META_PATH, "r") as f:
            return dict(json.load(f))
    return {}


# ── Drift computation ─────────────────────────────────────────────────────────
