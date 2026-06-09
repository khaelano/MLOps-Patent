"""Prometheus metrics for drift monitoring.

Exposes gauges that Grafana dashboards can query to visualise
data and prediction drift over time.

Metrics exported
----------------
patent_drift_score_ks_statistic
    Two-sample KS test statistic comparing new anomaly scores against baseline
    (0 = identical distributions, 1 = completely disjoint).
patent_drift_score_ks_pvalue
    P-value of the two-sample KS test for anomaly score distribution drift.
patent_drift_score_mean_shift
    Difference in mean anomaly score (new – baseline).
patent_drift_embedding_mean_shift
    Average per-dimension absolute mean shift of embeddings,
    in units of baseline standard deviation.
patent_drift_last_checked_timestamp_seconds
    Unix timestamp of the most recent drift check.
patent_drift_new_samples_total
    Number of samples in the most recent drift check.
patent_drift_score_distribution
    Distribution of anomaly scores from the most recent drift check
    (histogram with 11 buckets from 0.0 to 1.0).
patent_model_info
    Metadata about the currently deployed Production model
    (1 = active, with labels for model_version and model_name).
patent_data_total_rows
    Total number of rows in the training/embedding data.
patent_data_embedding_dim
    Dimensionality of the embedding vectors.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

# ── Lazy-load Prometheus client ───────────────────────────────────────────────

_METRICS_AVAILABLE: bool
try:
    from prometheus_client import CollectorRegistry, Gauge, Histogram  # noqa: F401

    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False

# ── Metric objects (None when prometheus_client is unavailable) ──────────────

METRICS_REGISTRY: Any = CollectorRegistry() if _METRICS_AVAILABLE else None  # type: ignore[possibly-unbound]

DRIFT_GAUGE: Any = (
    Gauge(  # type: ignore[possibly-unbound]
        "patent_drift_score_ks_statistic",
        "Two-sample KS test statistic comparing new anomaly scores against baseline "
        "(0 = identical distributions, 1 = completely disjoint)",
        registry=METRICS_REGISTRY,
    )
    if _METRICS_AVAILABLE
    else None
)

DRIFT_GAUGE_PVALUE: Any = (
    Gauge(  # type: ignore[possibly-unbound]
        "patent_drift_score_ks_pvalue",
        "P-value of the two-sample KS test for anomaly score distribution drift",
        registry=METRICS_REGISTRY,
    )
    if _METRICS_AVAILABLE
    else None
)

DRIFT_GAUGE_MEAN_SHIFT: Any = (
    Gauge(  # type: ignore[possibly-unbound]
        "patent_drift_score_mean_shift",
        "Difference in mean anomaly score (new – baseline)",
        registry=METRICS_REGISTRY,
    )
    if _METRICS_AVAILABLE
    else None
)

DRIFT_GAUGE_EMB_SHIFT: Any = (
    Gauge(  # type: ignore[possibly-unbound]
        "patent_drift_embedding_mean_shift",
        "Average per-dimension absolute mean shift of embeddings, "
        "in units of baseline standard deviation",
        registry=METRICS_REGISTRY,
    )
    if _METRICS_AVAILABLE
    else None
)

DRIFT_GAUGE_LAST_CHECKED: Any = (
    Gauge(  # type: ignore[possibly-unbound]
        "patent_drift_last_checked_timestamp_seconds",
        "Unix timestamp of the most recent drift check",
        registry=METRICS_REGISTRY,
    )
    if _METRICS_AVAILABLE
    else None
)

DRIFT_GAUGE_N_SAMPLES: Any = (
    Gauge(  # type: ignore[possibly-unbound]
        "patent_drift_new_samples_total",
        "Number of samples in the most recent drift check",
        registry=METRICS_REGISTRY,
    )
    if _METRICS_AVAILABLE
    else None
)

# ── Score distribution histogram ──────────────────────────────────────────────

_DRIFT_SCORE_BUCKETS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

DRIFT_SCORE_DISTRIBUTION: Any = (
    Histogram(  # type: ignore[possibly-unbound]
        "patent_drift_score_distribution",
        "Distribution of anomaly scores from the most recent drift check",
        buckets=_DRIFT_SCORE_BUCKETS,
        registry=METRICS_REGISTRY,
    )
    if _METRICS_AVAILABLE
    else None
)

# ── Model info ────────────────────────────────────────────────────────────────

MODEL_INFO: Any = (
    Gauge(  # type: ignore[possibly-unbound]
        "patent_model_info",
        "Metadata about the currently deployed Production model (1 = active)",
        labelnames=["model_version", "model_name"],
        registry=METRICS_REGISTRY,
    )
    if _METRICS_AVAILABLE
    else None
)

# ── Data info ─────────────────────────────────────────────────────────────────

DATA_TOTAL_ROWS: Any = (
    Gauge(  # type: ignore[possibly-unbound]
        "patent_data_total_rows",
        "Total number of rows in the training/embedding data",
        registry=METRICS_REGISTRY,
    )
    if _METRICS_AVAILABLE
    else None
)

DATA_EMBEDDING_DIM: Any = (
    Gauge(  # type: ignore[possibly-unbound]
        "patent_data_embedding_dim",
        "Dimensionality of the embedding vectors",
        registry=METRICS_REGISTRY,
    )
    if _METRICS_AVAILABLE
    else None
)


# ── Public update function ────────────────────────────────────────────────────


def update_drift_metrics(
    *,
    ks_statistic: float,
    ks_pvalue: float,
    mean_shift: float,
    emb_shift: float,
    n_samples: int,
    scores: np.ndarray | None = None,
    model_version: str | None = None,
    model_name: str = "patent-lshiforest",
    embedding_dim: int | None = None,
    total_rows: int | None = None,
) -> None:
    """Update all Prometheus drift metrics with the latest values.

    Call after each drift check to push values into the Prometheus endpoint.

    Parameters
    ----------
    ks_statistic : float
    ks_pvalue : float
    mean_shift : float
    emb_shift : float
    n_samples : int
    scores : ndarray | None
        The raw anomaly scores to record in the distribution histogram.
    model_version : str | None
    model_name : str
    embedding_dim : int | None
    total_rows : int | None
    """
    if not _METRICS_AVAILABLE:
        return

    DRIFT_GAUGE.set(ks_statistic)
    DRIFT_GAUGE_PVALUE.set(ks_pvalue)
    DRIFT_GAUGE_MEAN_SHIFT.set(mean_shift)
    DRIFT_GAUGE_EMB_SHIFT.set(emb_shift)
    DRIFT_GAUGE_LAST_CHECKED.set(time.time())
    DRIFT_GAUGE_N_SAMPLES.set(n_samples)

    if scores is not None and DRIFT_SCORE_DISTRIBUTION is not None:
        finite = scores[np.isfinite(scores)]
        for s in finite:
            DRIFT_SCORE_DISTRIBUTION.observe(float(s))

    if model_version and MODEL_INFO is not None:
        MODEL_INFO.labels(model_version=model_version, model_name=model_name).set(1)

    if total_rows is not None and DATA_TOTAL_ROWS is not None:
        DATA_TOTAL_ROWS.set(total_rows)

    if embedding_dim is not None and DATA_EMBEDDING_DIM is not None:
        DATA_EMBEDDING_DIM.set(embedding_dim)
