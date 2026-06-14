"""Data drift detection via KS test, Wasserstein distance, and distribution comparisons."""

from __future__ import annotations

from typing import Any

from loguru import logger
import numpy as np
from scipy import stats


def detect_drift(
    baseline_scores: np.ndarray,
    new_scores: np.ndarray,
    ks_threshold: float = 0.05,
    wasserstein_threshold: float = 0.1,
    mean_shift_threshold: float = 0.05,
    energy_threshold: float = 0.15,
) -> dict[str, Any]:
    """Detect whether *new_scores* have drifted relative to *baseline_scores*.

    Four complementary tests are run:
    1. **Two-sample KS test** — distribution-level shift (threshold on p-value).
    2. **Wasserstein (Earth Mover's) distance** — how much mass must move.
    3. **Mean shift** — simple first-moment comparison (normalised by baseline std).
    4. **Energy distance** — distributional divergence sensitive to variance
       and spread shifts the other three may miss.

    Drift is flagged when ≥2 of the 4 tests agree.

    Parameters
    ----------
    baseline_scores : ndarray
        Reference anomaly scores from the Production model's training data.
    new_scores : ndarray
        Anomaly scores from the latest batch of predictions.
    ks_threshold : float
        p-value threshold below which the KS test flags drift (default 0.05).
    wasserstein_threshold : float
        Wasserstein distance threshold above which drift is flagged (default 0.1).
    mean_shift_threshold : float
        Mean-shift threshold in units of baseline std deviation (default 0.05).
    energy_threshold : float
        Energy distance threshold above which drift is flagged (default 0.15).

    Returns
    -------
    dict with keys: ``drift_detected`` (bool), ``ks_statistic``, ``ks_pvalue``,
    ``wasserstein_distance``, ``mean_shift``, ``mean_shift_relative``,
    ``energy_distance``, ``baseline_mean``, ``baseline_std``, ``new_mean``, ``new_std``.
    """
    # ── Keep only finite values ──────────────────────────────────────────
    base_finite = baseline_scores[np.isfinite(baseline_scores)]
    new_finite = new_scores[np.isfinite(new_scores)]

    if len(base_finite) < 2 or len(new_finite) < 2:
        logger.warning(
            f"Too few finite scores for drift detection "
            f"(baseline={len(base_finite)}, new={len(new_finite)})"
        )
        return {
            "drift_detected": False,
            "ks_statistic": 0.0,
            "ks_pvalue": 1.0,
            "wasserstein_distance": 0.0,
            "energy_distance": 0.0,
            "mean_shift": 0.0,
            "mean_shift_relative": 0.0,
            "baseline_mean": float(np.mean(base_finite)) if len(base_finite) > 0 else 0.0,
            "baseline_std": float(np.std(base_finite)) if len(base_finite) > 0 else 0.0,
            "new_mean": float(np.mean(new_finite)) if len(new_finite) > 0 else 0.0,
            "new_std": float(np.std(new_finite)) if len(new_finite) > 0 else 0.0,
        }

    base_mean = float(np.mean(base_finite))
    base_std = float(np.std(base_finite))
    new_mean = float(np.mean(new_finite))
    new_std = float(np.std(new_finite))

    # ── Two-sample KS test ───────────────────────────────────────────────
    ks_stat, ks_pvalue = stats.ks_2samp(base_finite, new_finite)
    ks_stat = float(ks_stat)
    ks_pvalue = float(ks_pvalue)

    # ── Wasserstein distance (Earth Mover's Distance) ────────────────────
    wasserstein_dist = float(stats.wasserstein_distance(base_finite, new_finite))

    # ── Energy distance ─────────────────────────────────────────────────
    energy_dist = float(stats.energy_distance(base_finite, new_finite))

    # ── Mean shift (absolute and relative) ───────────────────────────────
    mean_shift = new_mean - base_mean
    mean_shift_relative = abs(mean_shift) / base_std if base_std > 0 else 0.0

    # ── Decision logic ───────────────────────────────────────────────────
    flags: list[str] = []
    if ks_pvalue < ks_threshold:
        flags.append(f"KS test rejected (p={ks_pvalue:.4f} < {ks_threshold})")
    if wasserstein_dist > wasserstein_threshold:
        flags.append(
            f"Wasserstein distance exceeded ({wasserstein_dist:.4f} > {wasserstein_threshold})"
        )
    if mean_shift_relative > mean_shift_threshold:
        flags.append(f"Mean shift exceeded ({mean_shift_relative:.4f} > {mean_shift_threshold})")
    if energy_dist > energy_threshold:
        flags.append(f"Energy distance exceeded ({energy_dist:.4f} > {energy_threshold})")

    drift_detected = len(flags) >= 2  # require at least 2 signals to agree

    if drift_detected:
        logger.warning(f"DRIFT DETECTED: {'; '.join(flags)}")
    else:
        logger.info(
            f"No drift detected (KS p={ks_pvalue:.4f}, "
            f"Wasserstein={wasserstein_dist:.4f}, mean_shift_rel={mean_shift_relative:.4f}, "
            f"energy_dist={energy_dist:.4f})"
        )

    return {
        "drift_detected": drift_detected,
        "drift_signals": flags,
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pvalue,
        "wasserstein_distance": wasserstein_dist,
        "energy_distance": energy_dist,
        "mean_shift": mean_shift,
        "mean_shift_relative": mean_shift_relative,
        "baseline_mean": base_mean,
        "baseline_std": base_std,
        "new_mean": new_mean,
        "new_std": new_std,
    }
