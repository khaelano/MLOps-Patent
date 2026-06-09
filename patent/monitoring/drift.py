"""Data and prediction drift detection for the LSHiForest anomaly model.

Two complementary drift signals are computed:

1. **Score drift** — score new data with the Production model and compare the
   resulting anomaly-score distribution against a stored baseline via the
   two-sample Kolmogorov–Smirnov test and mean shift.

2. **Embedding drift** — compare per-dimension mean of new embeddings against
   a baseline computed from training data.  This detects shifts in the
   semantic space of paper titles before they affect the model.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any

from loguru import logger
import numpy as np

from patent.config import INTERIM_DATA_DIR

BASELINE_DIR = INTERIM_DATA_DIR / "drift_baseline"
BASELINE_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_BASELINE_PATH = BASELINE_DIR / "embedding_baseline.npz"
SCORE_BASELINE_PATH = BASELINE_DIR / "score_baseline.npz"
BASELINE_META_PATH = BASELINE_DIR / "baseline_meta.json"


@dataclass
class Baseline:
    """Stored baseline statistics for drift comparison.

    Attributes
    ----------
    embedding_mean : ndarray (d,)
        Per-dimension mean of training embeddings.
    embedding_std : ndarray (d,)
        Per-dimension standard deviation of training embeddings.
    score_ecdf : tuple of (values, cumulative_probabilities)
        Empirical CDF of training scores (for KS test).
    n_samples : int
        Number of samples in the baseline.
    timestamp : float
        Unix timestamp when baseline was computed.
    model_version : str | None
        MLflow model version that produced this baseline.
    """

    embedding_mean: np.ndarray
    embedding_std: np.ndarray
    score_ecdf: tuple[np.ndarray, np.ndarray]
    n_samples: int
    timestamp: float
    model_version: str | None = None


@dataclass
class DriftReport:
    """Result of a drift check.

    Attributes
    ----------
    score_ks_statistic : float
        KS test statistic for score distribution (0 = identical, 1 = disjoint).
    score_ks_pvalue : float
        KS test p-value.  Low values suggest significant distributional shift.
    score_mean_shift : float
        Difference in mean anomaly score (new – baseline).
    embedding_mean_shift : float
        Average per-dimension absolute mean shift, relative to baseline std.
    n_new_samples : int
        Number of new samples checked.
    checked_at : float
        Unix timestamp of this check.
    """

    score_ks_statistic: float
    score_ks_pvalue: float
    score_mean_shift: float
    embedding_mean_shift: float
    n_new_samples: int
    checked_at: float


def save_drift_baseline(
    embeddings: np.ndarray,
    scores: np.ndarray,
    model_version: str | None = None,
) -> Baseline:
    """Compute and persist drift baseline statistics from training data.

    Parameters
    ----------
    embeddings : ndarray (n, d)
        Training-set embeddings.
    scores : ndarray (n,)
        Anomaly scores from the Production model on the training set.
    model_version : str | None
        MLflow model version for provenance tracking.
    """
    n, d = embeddings.shape
    if n < 100:
        raise ValueError(f"Need at least 100 samples for a stable baseline, got {n}")

    e_mean = np.mean(embeddings, axis=0).astype(np.float32)
    e_std = np.std(embeddings, axis=0).astype(np.float32)
    e_std = np.clip(e_std, 1e-8, None)  # avoid division by zero

    finite = scores[np.isfinite(scores)]
    score_sorted = np.sort(finite)
    cdf = np.arange(1, len(score_sorted) + 1) / len(score_sorted)

    np.savez_compressed(EMBEDDING_BASELINE_PATH, mean=e_mean, std=e_std)
    np.savez_compressed(SCORE_BASELINE_PATH, values=score_sorted, cdf=cdf)

    baseline = Baseline(
        embedding_mean=e_mean,
        embedding_std=e_std,
        score_ecdf=(score_sorted, cdf),
        n_samples=n,
        timestamp=time.time(),
        model_version=model_version,
    )

    _save_baseline_meta(baseline)
    logger.success(f"Drift baseline saved ({n} samples, dim={d}, model_v={model_version})")
    return baseline


def load_drift_baseline() -> Baseline | None:
    """Load the stored drift baseline, or ``None`` if not yet created."""
    if not EMBEDDING_BASELINE_PATH.exists() or not SCORE_BASELINE_PATH.exists():
        logger.warning("No drift baseline found — run training first.")
        return None

    emb = np.load(EMBEDDING_BASELINE_PATH)
    scr = np.load(SCORE_BASELINE_PATH)
    meta = _load_baseline_meta()

    return Baseline(
        embedding_mean=emb["mean"],
        embedding_std=emb["std"],
        score_ecdf=(scr["values"], scr["cdf"]),
        n_samples=meta.get("n_samples", 0),
        timestamp=meta.get("timestamp", 0.0),
        model_version=meta.get("model_version"),
    )


def _save_baseline_meta(baseline: Baseline) -> None:
    BASELINE_META_PATH.write_text(
        json.dumps(
            {
                "n_samples": baseline.n_samples,
                "timestamp": baseline.timestamp,
                "model_version": baseline.model_version,
            },
            indent=2,
        )
    )


def _load_baseline_meta() -> dict[str, Any]:
    if BASELINE_META_PATH.exists():
        return json.loads(BASELINE_META_PATH.read_text())
    return {}


def compute_drift_metrics(
    new_embeddings: np.ndarray,
    new_scores: np.ndarray,
    baseline: Baseline | None = None,
) -> DriftReport:
    """Compute drift metrics by comparing new data against the baseline.

    Parameters
    ----------
    new_embeddings : ndarray (n, d)
        Embeddings from the newly-acquired data.
    new_scores : ndarray (n,)
        Anomaly scores of the new data scored by the Production model.
    baseline : Baseline | None
        Pre-loaded baseline.  If ``None``, loads from disk.

    Returns
    -------
    DriftReport
    """
    if baseline is None:
        baseline = load_drift_baseline()

    if baseline is None:
        return DriftReport(
            score_ks_statistic=0.0,
            score_ks_pvalue=1.0,
            score_mean_shift=0.0,
            embedding_mean_shift=0.0,
            n_new_samples=len(new_scores),
            checked_at=time.time(),
        )

    from scipy.stats import ks_2samp

    finite_new = new_scores[np.isfinite(new_scores)]
    baseline_scores = baseline.score_ecdf[0]

    if len(finite_new) >= 10 and len(baseline_scores) >= 10:
        ks_stat, ks_pval = ks_2samp(finite_new, baseline_scores)
        mean_shift = float(np.mean(finite_new) - np.mean(baseline_scores))
    else:
        ks_stat, ks_pval, mean_shift = 0.0, 1.0, 0.0

    new_mean = np.mean(new_embeddings, axis=0)
    per_dim_shift = np.abs(new_mean - baseline.embedding_mean) / baseline.embedding_std
    emb_shift = float(np.mean(per_dim_shift))

    report = DriftReport(
        score_ks_statistic=float(ks_stat),
        score_ks_pvalue=float(ks_pval),
        score_mean_shift=mean_shift,
        embedding_mean_shift=emb_shift,
        n_new_samples=len(new_scores),
        checked_at=time.time(),
    )

    logger.info(
        f"Drift check: KS={report.score_ks_statistic:.4f} "
        f"(p={report.score_ks_pvalue:.4f}), "
        f"score_Δμ={report.score_mean_shift:.4f}, "
        f"emb_shift={report.embedding_mean_shift:.4f}σ, "
        f"n={report.n_new_samples}"
    )

    return report
