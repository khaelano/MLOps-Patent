from itertools import combinations
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Literal

from loguru import logger
import numpy as np
from scipy.stats import kurtosis, pearsonr, skew, spearmanr

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


def convert_embeddings_to_memmap(
    embeddings_paths: list[str],
    output_path: str | Path,
    column: str = "embedding",
    chunk_size: int = 200_000,
) -> tuple[int, int]:
    import pyarrow.parquet as pq

    embedding_dim = None
    total_rows = 0
    for pq_path in embeddings_paths:
        pq_file = pq.ParquetFile(pq_path)
        total_rows += pq_file.metadata.num_rows
        if embedding_dim is None:
            first_batch = next(pq_file.iter_batches(batch_size=1, columns=[column]))
            embedding_dim = len(first_batch.column(0)[0])

    mmap = np.memmap(
        str(output_path), dtype=np.float32, mode="w+", shape=(total_rows, embedding_dim)
    )
    row_offset = 0
    for pq_path in embeddings_paths:
        pq_file = pq.ParquetFile(pq_path)
        for batch in pq_file.iter_batches(batch_size=chunk_size, columns=[column]):
            flat = batch.column(0).flatten().to_numpy(zero_copy_only=False)
            arr = flat.reshape(-1, embedding_dim).astype(np.float32, copy=False)
            end = row_offset + len(arr)
            mmap[row_offset:end] = arr
            row_offset = end
    mmap.flush()
    del mmap
    return embedding_dim, total_rows


def score_memmap_chunked(
    model: Any,
    embeddings_mmap: np.ndarray,
    total_rows: int,
    chunk_size: int = 100_000,
) -> np.ndarray:
    scores = np.empty(total_rows, dtype=np.float32)
    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        batch = np.array(embeddings_mmap[start:end])
        scores[start:end] = model.score(batch)
    return scores


def _train_single_seed(args):
    """Build and save one seed model (runs in multiprocessing worker)."""
    seed, mmap_path, total_rows, embedding_dim, num_trees, max_depth, output_path = args
    from patent.modeling.lsh_iforest import LSHIForest
    from patent.utils import mute_logging

    embeddings = np.memmap(
        mmap_path, dtype=np.float32, mode="r", shape=(total_rows, embedding_dim)
    )
    with mute_logging():
        model = LSHIForest(num_trees=num_trees, max_depth=max_depth, seed=seed)
        model.build_forest_from_embeddings(embeddings, baseline_output_path=output_path)
    return output_path


def evaluate_params(
    embeddings_paths: list[str] | list[Path],
    num_trees: int,
    max_depth: int,
    n_workers: int | None = None,
) -> dict[str, Any]:
    import concurrent.futures
    import os

    seeds = [234, 223, 342, 122, 89]
    temp_dir = Path(tempfile.mkdtemp())

    if isinstance(embeddings_paths, (str, Path)):
        embeddings_paths = [str(embeddings_paths)]
    else:
        embeddings_paths = [str(p) for p in embeddings_paths]

    logger.info("Pre-loading embeddings into shared memmap...")
    mmap_path = temp_dir / "shared_embeddings.mmap"
    embedding_dim, total_rows = convert_embeddings_to_memmap(embeddings_paths, mmap_path)

    if n_workers is None:
        n_workers = min(len(seeds), max(1, (os.cpu_count() or 1) - 1))

    score_paths = []
    model_names = []

    if n_workers > 1:
        task_args = [
            (
                seed,
                str(mmap_path),
                total_rows,
                embedding_dim,
                num_trees,
                max_depth,
                str(temp_dir / f"depths_{seed}.npy"),
            )
            for seed in seeds
        ]
        with mute_logging():
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
                results = list(pool.map(_train_single_seed, task_args))
        for seed, path in zip(seeds, results):
            score_paths.append(Path(path))
            model_names.append(f"lshif_{seed}")
    else:
        embeddings = np.memmap(
            mmap_path, dtype=np.float32, mode="r", shape=(total_rows, embedding_dim)
        )
        with mute_logging():
            for k in seeds:
                baseline_path = temp_dir / f"depths_{k}.npy"
                model = LSHIForest(num_trees=num_trees, max_depth=max_depth, seed=k)
                model.build_forest_from_embeddings(embeddings, baseline_output_path=baseline_path)
                score_paths.append(baseline_path)
                model_names.append(f"lshif_{k}")

    metrics = calculate_stability_metrics_n(
        score_paths=score_paths,
        model_names=model_names,
    )

    shutil.rmtree(temp_dir)

    return metrics


def analyze_score_distribution(scores: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(scores)
    if np.sum(valid) == 0:
        logger.warning("No finite scores found for distribution analysis")
        return {"n_samples": 0, "n_finite": 0}

    finite = scores[valid]
    n_total = len(scores)
    n_finite = len(finite)
    if n_finite < n_total:
        logger.warning(
            f"Excluded {n_total - n_finite} non-finite scores from distribution analysis"
        )

    percentiles = [10, 25, 50, 75, 90, 95, 99]
    pct_values = np.percentile(finite, percentiles)

    thresholds = [0.5, 0.7, 0.9]
    pct_above = {
        f"pct_above_{t:.1f}".replace(".", "_"): float(np.mean(finite > t) * 100)
        for t in thresholds
    }

    result = {
        "n_samples": n_total,
        "n_finite": n_finite,
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "skewness": float(skew(finite)),
        "kurtosis": float(kurtosis(finite)),
        **{f"percentile_{p}": float(v) for p, v in zip(percentiles, pct_values)},
        **pct_above,
    }

    logger.success(
        f"Score distribution: mean={result['mean']:.4f}, median={result['median']:.4f}, "
        f"skew={result['skewness']:.2f}, "
        f">0.5={result.get('pct_above_0_5', 0):.1f}%, "
        f">0.7={result.get('pct_above_0_7', 0):.1f}%, "
        f">0.9={result.get('pct_above_0_9', 0):.1f}%"
    )

    return result


def distance_to_centroid_correlation(
    embeddings_paths: list[str] | list[Path],
    scores: np.ndarray,
    chunk_size: int = 200_000,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    if isinstance(embeddings_paths, (str, Path)):
        embeddings_paths = [str(embeddings_paths)]
    else:
        embeddings_paths = [str(p) for p in embeddings_paths]

    valid_mask = np.isfinite(scores)
    if np.sum(valid_mask) < 10:
        logger.warning("Too few finite scores for centroid correlation")
        return {}

    total_rows, embedding_dim = _infer_embedding_shape(embeddings_paths)
    if total_rows == 0:
        return {}

    centroid = np.zeros(embedding_dim, dtype=np.float64)
    centroid_n = 0
    for pq_path in embeddings_paths:
        pq_file = pq.ParquetFile(pq_path)
        for batch in pq_file.iter_batches(batch_size=chunk_size, columns=["embedding"]):
            col = batch.column("embedding")
            flat = col.flatten().to_numpy(zero_copy_only=False)
            emb = flat.reshape(-1, embedding_dim).astype(np.float64)
            centroid += emb.sum(axis=0)
            centroid_n += emb.shape[0]

    if centroid_n == 0:
        return {}

    centroid /= centroid_n

    distances = np.empty(total_rows, dtype=np.float64)
    row_offset = 0
    for pq_path in embeddings_paths:
        pq_file = pq.ParquetFile(pq_path)
        for batch in pq_file.iter_batches(batch_size=chunk_size, columns=["embedding"]):
            col = batch.column("embedding")
            flat = col.flatten().to_numpy(zero_copy_only=False)
            emb = flat.reshape(-1, embedding_dim).astype(np.float64)
            n = emb.shape[0]
            diff = emb - centroid
            dists = np.linalg.norm(diff, axis=1)
            distances[row_offset : row_offset + n] = dists
            row_offset += n

    mask = valid_mask & np.isfinite(distances)
    if np.sum(mask) < 10:
        logger.warning("Too few valid pairs for centroid correlation")
        return {
            "mean_distance": float(np.mean(distances)),
            "std_distance": float(np.std(distances)),
        }

    spearman_corr, _ = spearmanr(distances[mask], scores[mask])
    pearson_corr, _ = pearsonr(distances[mask], scores[mask])

    result = {
        "spearman_correlation": float(spearman_corr),
        "pearson_correlation": float(pearson_corr),
        "mean_distance": float(np.mean(distances)),
        "std_distance": float(np.std(distances)),
        "centroid_norm": float(np.linalg.norm(centroid)),
    }

    logger.success(
        f"Centroid correlation: Spearman={result['spearman_correlation']:.4f}, "
        f"Pearson={result['pearson_correlation']:.4f}, "
        f"mean_dist={result['mean_distance']:.4f}±{result['std_distance']:.4f}"
    )

    return result


def _infer_embedding_shape(embeddings_paths: list[str]) -> tuple[int, int]:
    import pyarrow.parquet as pq

    first_file = pq.ParquetFile(embeddings_paths[0])
    first_batch = next(first_file.iter_batches(batch_size=1, columns=["embedding"]))
    embedding_dim = len(first_batch.column(0)[0].as_py())
    total_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in embeddings_paths)
    return total_rows, embedding_dim


def evaluate_subsampling_stability(
    embeddings_paths: list[str] | list[Path],
    num_trees: int,
    max_depth: int,
    n_splits: int = 5,
    subsample_ratio: float = 0.8,
    top_k: int = 1000,
    spearman_sample_size: int = 100_000,
    seed: int = 42,
) -> dict[str, Any]:
    if not (0 < subsample_ratio < 1):
        raise ValueError(f"subsample_ratio must be in (0, 1), got {subsample_ratio}")

    if isinstance(embeddings_paths, (str, Path)):
        embeddings_paths = [str(embeddings_paths)]
    else:
        embeddings_paths = [str(p) for p in embeddings_paths]

    total_rows, _ = _infer_embedding_shape(embeddings_paths)
    subsample_size = int(total_rows * subsample_ratio)

    if subsample_size < 10:
        raise ValueError(f"Subsample size {subsample_size} too small for stability evaluation")

    rng = np.random.default_rng(seed)
    temp_dir = Path(tempfile.mkdtemp())

    score_paths = []
    model_names = []

    with mute_logging():
        for split_idx in range(n_splits):
            split_seed = int(rng.integers(0, 2**31 - 1))
            subset_indices = rng.choice(total_rows, size=subsample_size, replace=False)

            baseline_path = temp_dir / f"depths_split{split_idx}.npy"

            model = LSHIForest(num_trees=num_trees, max_depth=max_depth, seed=split_seed)
            model._build_forest(embeddings_paths, column="embedding")

            subset_rows_int = int(subset_indices.size)
            forest_mmap = model.forest_mmap
            if forest_mmap is None or forest_mmap.size == 0:
                logger.error(f"Split {split_idx}: forest is empty, skipping")
                continue

            subset_mmap_path = temp_dir / f"forest_split{split_idx}.dat"
            subset_mmap = np.memmap(
                subset_mmap_path,
                dtype=model._hash_dtype,
                mode="w+",
                shape=(model.meta.num_trees, subset_rows_int),
            )
            for t in range(model.meta.num_trees):
                subset_mmap[t] = forest_mmap[t, subset_indices]
            subset_mmap.flush()

            model.forest_mmap = subset_mmap
            model.meta.num_rows = subset_rows_int
            model.meta.is_sorted = True

            model._calculate_baseline(baseline_output_path=baseline_path)

            score_paths.append(str(baseline_path))
            model_names.append(f"subsample_{split_idx}")

    if len(score_paths) < 2:
        shutil.rmtree(temp_dir)
        raise RuntimeError(f"Only {len(score_paths)} successful splits (need >= 2)")

    metrics = calculate_stability_metrics_n(
        score_paths=score_paths,
        model_names=model_names,
        top_k=top_k,
        spearman_sample_size=spearman_sample_size,
        aggregation="mean",
    )

    metrics["subsample_ratio"] = subsample_ratio
    metrics["n_splits"] = n_splits

    shutil.rmtree(temp_dir)

    return metrics


def export_top_anomalies(
    scores: np.ndarray,
    metadata: Any,
    output_path: str | Path,
    top_k: int = 100,
) -> list[dict]:
    import pandas as pd

    valid_mask = np.isfinite(scores)
    valid_idx = np.where(valid_mask)[0]
    n_valid = len(valid_idx)

    if n_valid == 0:
        logger.warning("No finite scores, cannot export top anomalies")
        return []

    actual_k = min(top_k, n_valid)
    top_local = np.argsort(-scores[valid_idx], kind="stable")[:actual_k]
    top_indices = valid_idx[top_local]

    columns = ["id", "title", "categories", "update_date"]
    available = [c for c in columns if c in metadata.columns]

    records = []
    for idx in top_indices:
        record = {"anomaly_score": float(scores[idx])}
        for col in available:
            val = metadata.iloc[idx][col]
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val)
            elif isinstance(val, (np.ndarray,)):
                val = val.tolist()
            elif pd.isna(val):
                val = None
            record[col] = val
        records.append(record)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2, default=str)

    logger.success(f"Exported top {actual_k} anomalies to {output_path}")

    return records
