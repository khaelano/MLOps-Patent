"""Prometheus metrics for drift monitoring.

Exposes gauges that Grafana dashboards can query to visualise
data and prediction drift over time.
"""

from __future__ import annotations

import numpy as np
from prometheus_client import CollectorRegistry, Gauge, Histogram

METRICS_REGISTRY = CollectorRegistry()

DRIFT_GAUGE = Gauge(
    "patent_drift_score_ks_statistic",
    "Two-sample KS test statistic comparing new anomaly scores against baseline"
    " (0 = identical distributions, 1 = completely disjoint).",
    registry=METRICS_REGISTRY,
)

DRIFT_GAUGE_PVALUE = Gauge(
    "patent_drift_score_ks_pvalue",
    "P-value of the two-sample KS test for anomaly score distribution drift.",
    registry=METRICS_REGISTRY,
)

DRIFT_GAUGE_MEAN_SHIFT = Gauge(
    "patent_drift_score_mean_shift",
    "Difference in mean anomaly score (new – baseline).",
    registry=METRICS_REGISTRY,
)

DRIFT_GAUGE_EMB_SHIFT = Gauge(
    "patent_drift_embedding_mean_shift",
    "Average per-dimension absolute mean shift of embeddings, in units of"
    " baseline standard deviation.",
    registry=METRICS_REGISTRY,
)

DRIFT_GAUGE_LAST_CHECKED = Gauge(
    "patent_drift_last_checked_timestamp_seconds",
    "Unix timestamp of the most recent drift check.",
    registry=METRICS_REGISTRY,
)

DRIFT_GAUGE_N_SAMPLES = Gauge(
    "patent_drift_new_samples_total",
    "Number of samples in the most recent drift check.",
    registry=METRICS_REGISTRY,
)

DRIFT_SCORE_DISTRIBUTION = Histogram(
    "patent_drift_score_distribution",
    "Distribution of anomaly scores from the most recent drift check.",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=METRICS_REGISTRY,
)

MODEL_INFO = Gauge(
    "patent_model_info",
    "Metadata about the currently deployed Production model (1 = active).",
    labelnames=["model_version", "model_name"],
    registry=METRICS_REGISTRY,
)

DATA_TOTAL_ROWS = Gauge(
    "patent_data_total_rows",
    "Total number of processed rows in the training dataset.",
    registry=METRICS_REGISTRY,
)

DATA_EMBEDDING_DIM = Gauge(
    "patent_data_embedding_dimension",
    "Dimensionality of the embeddings.",
    registry=METRICS_REGISTRY,
)


def update_drift_metrics(
    *,
    ks_statistic: float = 0.0,
    ks_pvalue: float = 1.0,
    mean_shift: float = 0.0,
    emb_shift: float = 0.0,
    n_samples: int = 0,
    scores: "np.ndarray | None" = None,
    model_version: str | None = None,
    model_name: str = "patent-lshiforest",
    embedding_dim: int | None = None,
    total_rows: int | None = None,
) -> None:
    """Push drift-related metrics into the Prometheus gauges.

    Parameters
    ----------
    ks_statistic : float
        KS test statistic for score distribution.
    ks_pvalue : float
        KS test p-value.
    mean_shift : float
        Mean anomaly score shift.
    emb_shift : float
        Per-dimension embedding mean shift.
    n_samples: int
        Number of samples checked.
    scores : ndarray | None
        Score array for sampling into the distribution histogram.
    model_version : str | None
        Current deployed model version.
    model_name : str
        Registered model name.
    embedding_dim : int | None
        Dimensionality of current embeddings.
    total_rows : int | None
        Total training rows.
    """
    import time as _time

    DRIFT_GAUGE.set(ks_statistic)
    DRIFT_GAUGE_PVALUE.set(ks_pvalue)
    DRIFT_GAUGE_MEAN_SHIFT.set(mean_shift)
    DRIFT_GAUGE_EMB_SHIFT.set(emb_shift)
    DRIFT_GAUGE_LAST_CHECKED.set(_time.time())
    DRIFT_GAUGE_N_SAMPLES.set(n_samples)

    if scores is not None and len(scores) > 0:
        finite = scores[np.isfinite(scores)]
        if len(finite) > 10_000:
            rng = np.random.default_rng(42)
            finite = rng.choice(finite, size=10_000, replace=False)
        for val in finite:
            DRIFT_SCORE_DISTRIBUTION.observe(float(val))

    if model_version:
        MODEL_INFO._metrics.clear()
        MODEL_INFO.labels(model_version=model_version, model_name=model_name).set(1)

    if embedding_dim is not None:
        DATA_EMBEDDING_DIM.set(embedding_dim)

    if total_rows is not None:
        DATA_TOTAL_ROWS.set(total_rows)
