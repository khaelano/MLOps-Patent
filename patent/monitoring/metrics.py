"""Prometheus metrics for drift monitoring and server performance.

Exposes gauges, counters, and histograms that Grafana dashboards can
query to visualise drift, throughput, latency, and resource usage.

Metrics exported
----------------
patent_drift_score_ks_statistic
    Two-sample KS test statistic comparing new anomaly scores against baseline
    (0 = identical distributions, 1 = completely disjoint).
patent_drift_score_ks_pvalue
    P-value of the two-sample KS test for anomaly score distribution drift.
patent_drift_score_mean_shift
    Difference in mean anomaly score (new – baseline).
patent_drift_score_energy_distance
    Energy Distance between new anomaly scores and baseline
    (0 = identical distributions, higher = more drift).
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

import numpy as np
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

# ── Metric registry ──────────────────────────────────────────────────────────

METRICS_REGISTRY = CollectorRegistry()

# ── Drift gauges ──────────────────────────────────────────────────────────────

DRIFT_GAUGE = Gauge(
    "patent_drift_score_ks_statistic",
    "Two-sample KS test statistic comparing new anomaly scores against baseline "
    "(0 = identical distributions, 1 = completely disjoint)",
    registry=METRICS_REGISTRY,
)

DRIFT_GAUGE_PVALUE = Gauge(
    "patent_drift_score_ks_pvalue",
    "P-value of the two-sample KS test for anomaly score distribution drift",
    registry=METRICS_REGISTRY,
)

DRIFT_GAUGE_MEAN_SHIFT = Gauge(
    "patent_drift_score_mean_shift",
    "Difference in mean anomaly score (new – baseline)",
    registry=METRICS_REGISTRY,
)

DRIFT_GAUGE_LAST_CHECKED = Gauge(
    "patent_drift_last_checked_timestamp_seconds",
    "Unix timestamp of the most recent drift check",
    registry=METRICS_REGISTRY,
)

DRIFT_GAUGE_N_SAMPLES = Gauge(
    "patent_drift_new_samples_total",
    "Number of samples in the most recent drift check",
    registry=METRICS_REGISTRY,
)

# ── Energy Distance gauge ──────────────────────────────────────────────────

DRIFT_GAUGE_ENERGY = Gauge(
    "patent_drift_score_energy_distance",
    "Energy Distance between new anomaly scores and baseline "
    "(0 = identical distributions, higher = more drift)",
    registry=METRICS_REGISTRY,
)

# ── Score distribution histogram ──────────────────────────────────────────────

_DRIFT_SCORE_BUCKETS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

DRIFT_SCORE_DISTRIBUTION = Histogram(
    "patent_drift_score_distribution",
    "Distribution of anomaly scores from the most recent drift check",
    buckets=_DRIFT_SCORE_BUCKETS,
    registry=METRICS_REGISTRY,
)

# ── Drift detection flag ───────────────────────────────────────────────────

DRIFT_DETECTED = Gauge(
    "patent_drift_detected",
    "Whether drift was detected in the most recent check (1 = drift, 0 = no drift)",
    registry=METRICS_REGISTRY,
)

# ── Model info ────────────────────────────────────────────────────────────────

MODEL_INFO = Gauge(
    "patent_model_info",
    "Metadata about the currently deployed Production model (1 = active)",
    labelnames=["model_version", "model_name"],
    registry=METRICS_REGISTRY,
)

# ── Data info ─────────────────────────────────────────────────────────────────

DATA_TOTAL_ROWS = Gauge(
    "patent_data_total_rows",
    "Total number of rows in the training/embedding data",
    registry=METRICS_REGISTRY,
)

DATA_EMBEDDING_DIM = Gauge(
    "patent_data_embedding_dim",
    "Dimensionality of the embedding vectors",
    registry=METRICS_REGISTRY,
)


# ── Performance counters ────────────────────────────────────────────────────

PREDICT_REQUESTS = Counter(
    "patent_predict_requests_total",
    "Total number of /predict requests served",
    registry=METRICS_REGISTRY,
)

PREDICT_ERRORS = Counter(
    "patent_predict_errors_total",
    "Total number of /predict requests that returned an error (4xx/5xx)",
    registry=METRICS_REGISTRY,
)

PREDICT_INFLIGHT = Gauge(
    "patent_predict_requests_inflight",
    "Number of /predict requests currently being processed",
    registry=METRICS_REGISTRY,
)

# ── Performance histograms ──────────────────────────────────────────────────

_PAYLOAD_BUCKETS = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]

PREDICT_PAYLOAD_SIZE = Histogram(
    "patent_predict_payload_size_kb",
    "Size of /predict request payload in kilobytes",
    buckets=_PAYLOAD_BUCKETS,
    registry=METRICS_REGISTRY,
)

_LATENCY_BUCKETS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

PREDICT_LATENCY = Histogram(
    "patent_predict_latency_seconds",
    "End-to-end request latency for /predict (wall clock)",
    buckets=_LATENCY_BUCKETS,
    registry=METRICS_REGISTRY,
)

INFERENCE_TIME = Histogram(
    "patent_inference_time_seconds",
    "Time spent inside the embed + score pipeline (no HTTP overhead)",
    buckets=_LATENCY_BUCKETS,
    registry=METRICS_REGISTRY,
)

# ── Score rolling mean ──────────────────────────────────────────────────────

SCORE_ROLLING_MEAN = Gauge(
    "patent_score_rolling_mean",
    "Rolling mean of anomaly scores over the last ~60 seconds",
    registry=METRICS_REGISTRY,
)

# ── Process resource gauges (updated by psutil) ─────────────────────────────

PROCESS_MEMORY_BYTES = Gauge(
    "patent_process_memory_bytes",
    "Resident set size (RSS) of the inference server process in bytes",
    registry=METRICS_REGISTRY,
)

PROCESS_CPU_PERCENT = Gauge(
    "patent_process_cpu_percent",
    "CPU utilisation of the inference server process (0–100 %)",
    registry=METRICS_REGISTRY,
)


# ── Public update function ────────────────────────────────────────────────────


def update_drift_metrics(
    *,
    ks_statistic: float,
    ks_pvalue: float,
    mean_shift: float,
    energy_distance: float,
    n_samples: int,
    drift_detected: bool = False,
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
    energy_distance : float
        Energy Distance between the two score distributions.
    n_samples : int
    drift_detected : bool
        Whether the consensus drift detector flagged this check (1 = drift).
    scores : ndarray | None
        The raw anomaly scores to record in the distribution histogram.
    model_version : str | None
    model_name : str
    embedding_dim : int | None
    total_rows : int | None
    """
    DRIFT_GAUGE.set(ks_statistic)
    DRIFT_GAUGE_PVALUE.set(ks_pvalue)
    DRIFT_GAUGE_MEAN_SHIFT.set(mean_shift)
    DRIFT_GAUGE_ENERGY.set(energy_distance)
    DRIFT_GAUGE_LAST_CHECKED.set(time.time())
    DRIFT_GAUGE_N_SAMPLES.set(n_samples)
    DRIFT_DETECTED.set(1 if drift_detected else 0)

    if scores is not None:
        finite = scores[np.isfinite(scores)]
        for s in finite:
            DRIFT_SCORE_DISTRIBUTION.observe(float(s))

    if model_version:
        MODEL_INFO.labels(model_version=model_version, model_name=model_name).set(1)

    if total_rows is not None:
        DATA_TOTAL_ROWS.set(total_rows)

    if embedding_dim is not None:
        DATA_EMBEDDING_DIM.set(embedding_dim)
