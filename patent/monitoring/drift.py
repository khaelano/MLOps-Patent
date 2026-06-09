"""Data and prediction drift detection for the LSHiForest anomaly model.

Two complementary drift signals are computed:

1. **Score drift** — new anomaly scores are compared against baseline scores
   using the two-sample KS test, Wasserstein distance, and mean-shift analysis.
2. **Embedding drift** — new text embeddings are compared against baseline
   embeddings via per-dimension mean shift in units of baseline standard
   deviation.

A baseline is captured at training time (or at the first drift check) and
persisted to disk so that subsequent drift checks always compare against the
same reference.
"""

from __future__ import annotations

import json
import time
from typing import Any

from loguru import logger
import numpy as np

from patent.config import INTERIM_DATA_DIR

# ── Baseline storage paths ────────────────────────────────────────────────────

BASELINE_DIR = INTERIM_DATA_DIR / "drift_baseline"
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


class DriftReport:
    """Container for the results of a drift computation.

    Attributes
    ----------
    drift_detected : bool
        Whether drift was detected in the score distribution.
    ks_statistic : float
        Two-sample KS test statistic.
    ks_pvalue : float
        p-value of the KS test.
    wasserstein_distance : float
        Earth Mover's Distance between score distributions.
    mean_shift : float
        Absolute difference in mean anomaly scores.
    mean_shift_relative : float
        Mean shift normalised by baseline std deviation.
    embedding_mean_shift : float
        Average per-dimension absolute shift of embeddings (in baseline std units).
    n_samples : int
        Number of new samples in this check.
    timestamp : float
        Unix timestamp of the drift check.
    """

    def __init__(
        self,
        *,
        drift_detected: bool = False,
        drift_signals: list[str] | None = None,
        ks_statistic: float = 0.0,
        ks_pvalue: float = 1.0,
        wasserstein_distance: float = 0.0,
        mean_shift: float = 0.0,
        mean_shift_relative: float = 0.0,
        embedding_mean_shift: float = 0.0,
        n_samples: int = 0,
        timestamp: float | None = None,
    ) -> None:
        self.drift_detected = drift_detected
        self.drift_signals = drift_signals or []
        self.ks_statistic = ks_statistic
        self.ks_pvalue = ks_pvalue
        self.wasserstein_distance = wasserstein_distance
        self.mean_shift = mean_shift
        self.mean_shift_relative = mean_shift_relative
        self.embedding_mean_shift = embedding_mean_shift
        self.n_samples = n_samples
        self.timestamp = timestamp or time.time()


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


def load_drift_baseline() -> Baseline | None:
    """Load a previously saved drift baseline or return ``None``."""
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


def compute_drift_metrics(
    new_embeddings: np.ndarray,
    new_scores: np.ndarray,
    baseline: Baseline | None = None,
) -> DriftReport:
    """Compute drift metrics between new data and the baseline.

    When *baseline* is ``None`` (first run), the provided data becomes
    the new baseline and no drift is detected.

    Parameters
    ----------
    new_embeddings : ndarray of shape (n, d)
        Float32 embeddings from the current batch of predictions.
    new_scores : ndarray of shape (n,)
        Float64 anomaly scores from the current batch.
    baseline : Baseline | None
        Previously saved baseline.  If ``None``, this call saves the
        provided data as the new baseline.

    Returns
    -------
    DriftReport
    """
    from patent.monitoring.drift_detector import detect_drift

    n_samples = len(new_scores)

    if baseline is None:
        # First run — save as baseline, no drift
        save_drift_baseline(new_embeddings, new_scores)
        logger.info("Initial baseline captured — no drift check performed.")
        return DriftReport(
            drift_detected=False,
            n_samples=n_samples,
            embedding_mean_shift=0.0,
        )

    # ── Score drift ───────────────────────────────────────────────────────
    drift_result = detect_drift(baseline.scores, new_scores)

    # ── Embedding drift ───────────────────────────────────────────────────
    emb_shift = _compute_embedding_mean_shift(baseline.embeddings, new_embeddings)

    report = DriftReport(
        drift_detected=drift_result["drift_detected"],
        drift_signals=drift_result.get("drift_signals", []),
        ks_statistic=drift_result["ks_statistic"],
        ks_pvalue=drift_result["ks_pvalue"],
        wasserstein_distance=drift_result["wasserstein_distance"],
        mean_shift=drift_result["mean_shift"],
        mean_shift_relative=drift_result["mean_shift_relative"],
        embedding_mean_shift=emb_shift,
        n_samples=n_samples,
    )

    if report.drift_detected:
        logger.warning(
            f"Drift detected: KS={report.ks_statistic:.4f} (p={report.ks_pvalue:.4f}), "
            f"Wasserstein={report.wasserstein_distance:.4f}, "
            f"emb_shift={report.embedding_mean_shift:.4f}"
        )

    return report


def _compute_embedding_mean_shift(
    baseline_embeddings: np.ndarray,
    new_embeddings: np.ndarray,
) -> float:
    """Compute average per-dimension mean shift in baseline std units."""
    base_mean = baseline_embeddings.mean(axis=0, dtype=np.float64)
    base_std = baseline_embeddings.std(axis=0, dtype=np.float64)
    new_mean = new_embeddings.mean(axis=0, dtype=np.float64)

    # Avoid division by zero for constant dimensions
    safe_std = np.where(base_std > 1e-8, base_std, 1.0)
    per_dim_shift = np.abs(new_mean - base_mean) / safe_std

    return float(np.mean(per_dim_shift))
