import contextlib
from itertools import combinations
from pathlib import Path
import shutil
import tempfile
from typing import Any, Literal

from loguru import logger
import numpy as np
from scipy.stats import spearmanr

from patent.modeling.lsh_iforest import LSHIForest
from patent.utils import mute_logging


def calculate_stability_metrics(scores_path_a, scores_path_b, top_k=1000):
    """
    Efficiently compares two novelty score sets.
    """
    logger.info("Loading scores for stability check...")
    scores_a = np.load(scores_path_a, mmap_mode="r")
    scores_b = np.load(scores_path_b, mmap_mode="r")

    logger.info("Calculating Spearman Rank Correlation...")
    corr, _ = spearmanr(scores_a, scores_b)

    logger.info(f"Calculating Jaccard Similarity for Top-{top_k}...")
    top_indices_a = np.argpartition(scores_a, top_k)[:top_k]
    top_indices_b = np.argpartition(scores_b, top_k)[:top_k]

    set_a = set(top_indices_a)
    set_b = set(top_indices_b)

    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    jaccard = intersection / union

    return {"spearman_correlation": corr, "jaccard_similarity": jaccard, "top_k": top_k}


def _spearman_sampled(scores_a, scores_b, sample_size=100_000, seed=42) -> float:
    """Compute Spearman on a random subset for efficiency."""
    np.random.seed(seed)
    n = len(scores_a)
    if n <= sample_size:
        idx = np.arange(n)
    else:
        idx = np.random.choice(n, size=sample_size, replace=False)

    mask = np.isfinite(scores_a[idx]) & np.isfinite(scores_b[idx])
    if np.sum(mask) < 10:
        return np.nan

    corr, _ = spearmanr(scores_a[idx][mask], scores_b[idx][mask])
    return float(corr)


def _jaccard_topk(scores_a, scores_b, top_k: int) -> float:
    """Compute Jaccard similarity of top-k indices (highest scores = most anomalous)."""
    valid_mask = np.isfinite(scores_a) & np.isfinite(scores_b)
    valid_idx = np.where(valid_mask)[0]
    n_valid = len(valid_idx)

    if n_valid == 0:
        return 1.0

    actual_k = min(top_k, n_valid)

    top_a_local = np.argsort(-scores_a[valid_idx], kind="stable")[:actual_k]
    top_b_local = np.argsort(-scores_b[valid_idx], kind="stable")[:actual_k]

    top_a = valid_idx[top_a_local]
    top_b = valid_idx[top_b_local]

    intersection = len(np.intersect1d(top_a, top_b, assume_unique=True))
    union = len(np.union1d(top_a, top_b))
    return intersection / union if union > 0 else 1.0


def calculate_stability_metrics_n(
    score_paths: list[str],
    model_names: list[str] | None = None,
    top_k: int = 1000,
    spearman_sample_size: int = 100_000,
    aggregation: Literal["mean", "median", "min"] = "mean",
) -> dict:
    if model_names is None:
        model_names = [f"model_{i}" for i in range(len(score_paths))]

    assert len(score_paths) == len(model_names), "Paths and names must match"
    assert len(score_paths) >= 2, "Need at least 2 models to compare"

    logger.info(f"Loading {len(score_paths)} score files for stability analysis...")

    scores_list = []
    base_shape = None
    for path, name in zip(score_paths, model_names):
        scores = np.load(path, mmap_mode="r")
        if base_shape is None:
            base_shape = scores.shape
        else:
            assert scores.shape == base_shape, f"Shape mismatch: {name} vs baseline"
        scores_list.append(scores)
        logger.debug(f"Loaded {name}: shape={scores.shape}, dtype={scores.dtype}")

    n_models = len(score_paths)
    pairwise_spearman = {}
    pairwise_jaccard = {}

    logger.info(
        f"Computing pairwise metrics for {n_models} models ({n_models * (n_models - 1) // 2} pairs)..."
    )

    for i, j in combinations(range(n_models), 2):
        name_a, name_b = model_names[i], model_names[j]
        scores_a, scores_b = scores_list[i], scores_list[j]

        corr = _spearman_sampled(scores_a, scores_b, sample_size=spearman_sample_size)
        pairwise_spearman[(name_a, name_b)] = corr

        jacc = _jaccard_topk(scores_a, scores_b, top_k=top_k)
        pairwise_jaccard[(name_a, name_b)] = jacc

        logger.debug(f"{name_a} vs {name_b}: Spearman={corr:.4f}, Jaccard@{top_k}={jacc:.4f}")

    spearman_vals = [v for v in pairwise_spearman.values() if not np.isnan(v)]
    jaccard_vals = list(pairwise_jaccard.values())

    def _aggregate(vals, method: str):
        if not vals:
            return np.nan
        arr = np.array(vals)
        if method == "mean":
            return float(np.mean(arr))
        elif method == "median":
            return float(np.median(arr))
        elif method == "min":
            return float(np.min(arr))
        else:
            raise ValueError(f"Unknown aggregation: {method}")

    summary = {
        "n_models": n_models,
        "n_pairs": len(pairwise_spearman),
        "top_k": top_k,
        "spearman_aggregated": _aggregate(spearman_vals, aggregation),
        "spearman_std": float(np.std(spearman_vals)) if spearman_vals else np.nan,
        "spearman_min": float(np.min(spearman_vals)) if spearman_vals else np.nan,
        "spearman_max": float(np.max(spearman_vals)) if spearman_vals else np.nan,
        "jaccard_aggregated": _aggregate(jaccard_vals, aggregation),
        "jaccard_std": float(np.std(jaccard_vals)) if jaccard_vals else np.nan,
        "jaccard_min": float(np.min(jaccard_vals)) if jaccard_vals else np.nan,
        "jaccard_max": float(np.max(jaccard_vals)) if jaccard_vals else np.nan,
    }

    name_to_idx = {name: i for i, name in enumerate(model_names)}
    spearman_matrix = np.full((n_models, n_models), np.nan)
    jaccard_matrix = np.full((n_models, n_models), np.nan)

    for (name_a, name_b), corr in pairwise_spearman.items():
        i, j = name_to_idx[name_a], name_to_idx[name_b]
        spearman_matrix[i, j] = spearman_matrix[j, i] = corr

    for (name_a, name_b), jacc in pairwise_jaccard.items():
        i, j = name_to_idx[name_a], name_to_idx[name_b]
        jaccard_matrix[i, j] = jaccard_matrix[j, i] = jacc

    logger.success(
        f"Stability complete: {n_models} models, "
        f"Spearman={summary['spearman_aggregated']:.4f}±{summary['spearman_std']:.4f}, "
        f"Jaccard@{top_k}={summary['jaccard_aggregated']:.4f}±{summary['jaccard_std']:.4f}"
    )

    return {
        "pairwise_spearman": pairwise_spearman,
        "pairwise_jaccard": pairwise_jaccard,
        "summary": summary,
        "spearman_matrix": spearman_matrix.tolist(),
        "jaccard_matrix": jaccard_matrix.tolist(),
        "model_names": model_names,
    }


def evaluate_params(
    embeddings_path: list[str] | list[Path], num_trees: int, max_depth: int
) -> dict[str, Any]:
    seeds = [234, 223, 342, 122, 89]
    temp_dir = Path(tempfile.mkdtemp())

    score_paths = []
    model_names = []

    with contextlib.nullcontext():
        with mute_logging():
            for k in seeds:
                baseline_path = temp_dir / f"depths_{k}.npy"

                model = LSHIForest(num_trees=num_trees, max_depth=max_depth, seed=k)
                model.build_forest(
                    embeddings_paths=embeddings_path,
                    baseline_output_path=baseline_path,
                )
                score_paths.append(baseline_path)
                model_names.append(f"lshif_{k}")

    metrics = calculate_stability_metrics_n(
        score_paths=score_paths,
        model_names=model_names,
    )

    shutil.rmtree(temp_dir)

    return metrics
