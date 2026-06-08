"""Deteksi Data Drift dengan KS test, Wasserstein distance, dan perbandingan statistik."""

from __future__ import annotations

from typing import Any

from loguru import logger
import numpy as np


def detect_drift(
    baseline_scores: np.ndarray,
    new_scores: np.ndarray,
    *,
    ks_threshold: float = 0.05,
    wasserstein_threshold: float = 0.15,
    mean_shift_threshold: float = 0.10,
) -> dict[str, Any]:
    """Deteksi drift antara dua distribusi skor anomaly.

    Parameters
    ----------
    baseline_scores : np.ndarray
        Skor dari model baseline (referensi).
    new_scores : np.ndarray
        Skor dari data baru.
    ks_threshold : float
        p-value threshold untuk KS test.  p < ks_threshold → drift terdeteksi.
    wasserstein_threshold : float
        Threshold untuk Wasserstein distance (0–1).  Jarak > threshold → drift.
    mean_shift_threshold : float
        Threshold untuk perubahan rata-rata absolut.  |Δmean| > threshold → drift.

    Returns
    -------
    dict
        Dictionary dengan keys:
        - ``drift_detected``: bool — apakah drift terdeteksi
        - ``ks_statistic``, ``ks_pvalue``: hasil KS test
        - ``wasserstein_distance``: Earth Mover's Distance
        - ``mean_shift``: perubahan rata-rata absolut
        - ``std_shift``: perubahan standar deviasi
        - ``baseline_stats``, ``new_stats``: statistik deskriptif
        - ``triggers``: list of str — metrik mana yang memicu drift
    """
    from scipy.stats import ks_2samp, wasserstein_distance

    base_finite = baseline_scores[np.isfinite(baseline_scores)]
    new_finite = new_scores[np.isfinite(new_scores)]

    if len(base_finite) < 10 or len(new_finite) < 10:
        logger.warning("Too few finite scores for drift detection")
        return {
            "drift_detected": False,
            "error": "insufficient data",
        }

    baseline_stats = {
        "n": int(len(base_finite)),
        "mean": float(np.mean(base_finite)),
        "median": float(np.median(base_finite)),
        "std": float(np.std(base_finite)),
        "min": float(np.min(base_finite)),
        "max": float(np.max(base_finite)),
    }
    new_stats = {
        "n": int(len(new_finite)),
        "mean": float(np.mean(new_finite)),
        "median": float(np.median(new_finite)),
        "std": float(np.std(new_finite)),
        "min": float(np.min(new_finite)),
        "max": float(np.max(new_finite)),
    }

    ks_stat, ks_pvalue = ks_2samp(base_finite, new_finite)

    w_dist = float(wasserstein_distance(base_finite, new_finite))

    mean_shift = abs(new_stats["mean"] - baseline_stats["mean"])
    std_shift = abs(new_stats["std"] - baseline_stats["std"])

    triggers: list[str] = []
    if ks_pvalue < ks_threshold:
        triggers.append(f"KS test (p={ks_pvalue:.4f} < {ks_threshold})")
    if w_dist > wasserstein_threshold:
        triggers.append(f"Wasserstein distance (d={w_dist:.4f} > {wasserstein_threshold})")
    if mean_shift > mean_shift_threshold:
        triggers.append(f"Mean shift (|Δ|={mean_shift:.4f} > {mean_shift_threshold})")

    drift_detected = len(triggers) > 0

    result: dict[str, Any] = {
        "drift_detected": drift_detected,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "wasserstein_distance": w_dist,
        "mean_shift": mean_shift,
        "std_shift": std_shift,
        "baseline_stats": baseline_stats,
        "new_stats": new_stats,
        "triggers": triggers,
        "thresholds": {
            "ks_pvalue": ks_threshold,
            "wasserstein": wasserstein_threshold,
            "mean_shift": mean_shift_threshold,
        },
    }

    if drift_detected:
        logger.warning(
            f"DRIFT DETECTED: {len(triggers)} trigger(s) — "
            f"KS p={ks_pvalue:.4f}, W={w_dist:.4f}, Δmean={mean_shift:.4f}"
        )
    else:
        logger.info(
            f"No drift detected — KS p={ks_pvalue:.4f}, W={w_dist:.4f}, Δmean={mean_shift:.4f}"
        )

    return result


def compare_score_distributions(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    bins: int = 20,
) -> dict[str, Any]:
    """Bandingkan distribusi skor dalam bentuk histogram.

    Returns
    -------
    dict
        Keys: ``reference_hist``, ``current_hist``, ``bin_edges``,
        ``kl_divergence`` (symmetrised), ``js_distance``.
    """
    from scipy.spatial.distance import jensenshannon
    from scipy.special import rel_entr

    ref_finite = reference_scores[np.isfinite(reference_scores)]
    cur_finite = current_scores[np.isfinite(current_scores)]

    if len(ref_finite) < 10 or len(cur_finite) < 10:
        return {"error": "insufficient data"}

    # Shared bin edges across [0, 1] range
    bin_edges = np.linspace(0, 1, bins + 1)
    ref_hist, _ = np.histogram(ref_finite, bins=bin_edges, density=True)
    cur_hist, _ = np.histogram(cur_finite, bins=bin_edges, density=True)

    # Avoid zero probabilities
    epsilon = 1e-10
    ref_hist = ref_hist + epsilon
    cur_hist = cur_hist + epsilon
    ref_hist /= ref_hist.sum()
    cur_hist /= cur_hist.sum()

    # Symmetrised KL divergence
    kl_pq = float(np.sum(rel_entr(ref_hist, cur_hist)))
    kl_qp = float(np.sum(rel_entr(cur_hist, ref_hist)))
    kl_symmetric = (kl_pq + kl_qp) / 2

    # Jensen-Shannon distance
    js_dist = float(jensenshannon(ref_hist, cur_hist))

    return {
        "reference_hist": ref_hist.tolist(),
        "current_hist": cur_hist.tolist(),
        "bin_edges": bin_edges.tolist(),
        "kl_divergence_symmetric": kl_symmetric,
        "js_distance": js_dist,
    }
