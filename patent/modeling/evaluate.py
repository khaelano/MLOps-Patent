from itertools import combinations
import json
from pathlib import Path
import shutil
from typing import Any, Literal

from loguru import logger
import numpy as np
from scipy.stats import kurtosis, pearsonr, skew, spearmanr

from patent.config import CHUNK_SIZE, project_tempdir
from patent.lshiforest import LSHiForest
from patent.utils import convert_parquet_to_memmap, mute_logging


def _spearman_sampled(scores_a, scores_b, sample_size=100_000, seed=42) -> float:
    """Compute Spearman on a random subset for efficiency."""
    rng = np.random.default_rng(seed)
    n = len(scores_a)
    if n <= sample_size:
        idx = np.arange(n)
    else:
        idx = rng.choice(n, size=sample_size, replace=False)

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
    chunk_size: int = CHUNK_SIZE,
) -> tuple[int, int]:
    return convert_parquet_to_memmap(embeddings_paths, output_path, column)


def _train_single_seed(args):
    """Build and save one seed model (runs in multiprocessing worker)."""
    seed, mmap_path, total_rows, embedding_dim, num_trees, max_depth, output_path = args
    from patent.lshiforest import LSHiForest
    from patent.utils import mute_logging

    embeddings = np.memmap(
        mmap_path, dtype=np.float32, mode="r", shape=(total_rows, embedding_dim)
    )
    with mute_logging():
        model = LSHiForest(n_trees=num_trees, max_depth=max_depth, seed=seed)
        model.fit(embeddings)
        scores = model.score_chunked(embeddings, total_rows)
        np.save(output_path, scores)
    return output_path


def evaluate_params(
    embeddings_paths: list[str] | list[Path],
    num_trees: int,
    max_depth: int,
    n_workers: int | None = None,
    *,
    shared_mmap: tuple[str, int, int] | None = None,
) -> dict[str, Any]:
    """Seed-based stability: train *num_seeds* models and compare scores.

    Parameters
    ----------
    shared_mmap : (path, total_rows, embedding_dim) | None
        When provided, reuse an existing memmap instead of converting
        *embeddings_paths* from scratch.  The caller retains ownership
        of the memmap file.
    """
    import concurrent.futures
    import os

    seeds = [234, 223, 342, 122, 89]
    temp_dir = project_tempdir()

    if shared_mmap is not None:
        mmap_path_str, total_rows, embedding_dim = shared_mmap
        mmap_path = Path(mmap_path_str)
    else:
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
            score_paths.append(str(path))
            model_names.append(f"lshif_{seed}")
    else:
        embeddings = np.memmap(
            str(mmap_path), dtype=np.float32, mode="r", shape=(total_rows, embedding_dim)
        )
        with mute_logging():
            for k in seeds:
                baseline_path = temp_dir / f"depths_{k}.npy"
                model = LSHiForest(n_trees=num_trees, max_depth=max_depth, seed=k)
                model.fit(embeddings)
                scores = model.score_chunked(embeddings, total_rows)
                np.save(baseline_path, scores)
                score_paths.append(str(baseline_path))
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
    chunk_size: int = CHUNK_SIZE,
    mmap_path: str | None = None,
) -> dict[str, Any]:
    if isinstance(embeddings_paths, (str, Path)):
        embeddings_paths = [str(embeddings_paths)]
    else:
        embeddings_paths = [str(p) for p in embeddings_paths]

    valid_mask = np.isfinite(scores)
    if np.sum(valid_mask) < 10:
        logger.warning("Too few finite scores for centroid correlation")
        return {}

    if mmap_path is not None:
        _, embedding_dim = _infer_embedding_shape(embeddings_paths)
        file_size = Path(mmap_path).stat().st_size
        total_rows = file_size // (embedding_dim * 4)
        embeddings = np.memmap(
            mmap_path,
            dtype=np.float32,
            mode="r",
            shape=(total_rows, embedding_dim),
        )
    else:
        total_rows, embedding_dim = _infer_embedding_shape(embeddings_paths)
        if total_rows == 0:
            return {}

        embed_temp_dir = project_tempdir()
        mmap_path = str(embed_temp_dir / "centroid.mmap")
        try:
            from patent.utils import convert_parquet_to_memmap

            convert_parquet_to_memmap(embeddings_paths, mmap_path, column="embedding")
            embeddings = np.memmap(
                mmap_path, dtype=np.float32, mode="r", shape=(total_rows, embedding_dim)
            )
        except Exception:
            shutil.rmtree(embed_temp_dir, ignore_errors=True)
            raise

    centroid = np.zeros(embedding_dim, dtype=np.float64)
    centroid_n = 0
    distances = np.empty(total_rows, dtype=np.float32)

    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        batch = np.array(embeddings[start:end], dtype=np.float64)
        n = batch.shape[0]
        centroid += batch.sum(axis=0)
        centroid_n += n

    if centroid_n == 0:
        if mmap_path is None:
            shutil.rmtree(embed_temp_dir, ignore_errors=True)
        return {}

    centroid /= centroid_n
    centroid_f32 = centroid.astype(np.float32)

    row_offset = 0
    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        batch = np.asarray(embeddings[start:end])
        n = batch.shape[0]
        diff = batch - centroid_f32
        dists = np.linalg.norm(diff, axis=1)
        distances[row_offset : row_offset + n] = dists
        row_offset += n

    if mmap_path is None:
        shutil.rmtree(embed_temp_dir, ignore_errors=True)

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


def _train_subsample_split(args):
    """Build and score one subsample split (runs in multiprocessing worker)."""
    (
        split_idx,
        mmap_path,
        total_rows,
        embedding_dim,
        subset_indices,
        subsample_size,
        num_trees,
        max_depth,
        split_seed,
        output_path,
    ) = args
    from patent.lshiforest import LSHiForest
    from patent.utils import mute_logging

    embeddings = np.memmap(
        mmap_path, dtype=np.float32, mode="r", shape=(total_rows, embedding_dim)
    )

    # Materialise subset into a contiguous array
    subset_data = np.empty((subsample_size, embedding_dim), dtype=np.float32)
    chunk = CHUNK_SIZE
    for start in range(0, subsample_size, chunk):
        end = min(start + chunk, subsample_size)
        subset_data[start:end] = embeddings[subset_indices[start:end]]

    with mute_logging():
        model = LSHiForest(n_trees=num_trees, max_depth=max_depth, seed=split_seed)
        model.fit(subset_data)
        scores = model.score_chunked(subset_data, subsample_size)
        np.save(output_path, scores)
    return output_path


def evaluate_subsampling_stability(
    embeddings_paths: list[str] | list[Path],
    num_trees: int,
    max_depth: int,
    n_splits: int = 5,
    subsample_ratio: float = 0.8,
    top_k: int = 1000,
    spearman_sample_size: int = 100_000,
    seed: int = 42,
    n_workers: int | None = None,
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
    temp_dir = project_tempdir()

    score_paths = []
    model_names = []

    mmap_path = temp_dir / "stability_embeddings.mmap"
    embedding_dim, _ = convert_embeddings_to_memmap(embeddings_paths, mmap_path)

    # Pre-generate split configs
    split_configs: list[dict] = []
    for split_idx in range(n_splits):
        split_seed = int(rng.integers(0, 2**31 - 1))
        subset_indices = rng.choice(total_rows, size=subsample_size, replace=False)
        baseline_path = temp_dir / f"depths_split{split_idx}.npy"
        split_configs.append(
            {
                "split_idx": split_idx,
                "split_seed": split_seed,
                "subset_indices": subset_indices,
                "output_path": str(baseline_path),
            }
        )

    import concurrent.futures
    import os

    if n_workers is None:
        n_workers = min(n_splits, max(1, (os.cpu_count() or 1) - 1))

    if n_workers > 1:
        task_args = [
            (
                cfg["split_idx"],
                str(mmap_path),
                total_rows,
                embedding_dim,
                cfg["subset_indices"],
                subsample_size,
                num_trees,
                max_depth,
                cfg["split_seed"],
                cfg["output_path"],
            )
            for cfg in split_configs
        ]
        with mute_logging():
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
                results = list(pool.map(_train_subsample_split, task_args))
        for cfg, path in zip(split_configs, results):
            score_paths.append(str(path))
            model_names.append(f"subsample_{cfg['split_idx']}")
    else:
        embeddings = np.memmap(
            str(mmap_path), dtype=np.float32, mode="r", shape=(total_rows, embedding_dim)
        )
        with mute_logging():
            for cfg in split_configs:
                # Materialise the subset as a writable memmap
                subset_mmap_path = temp_dir / f"subset_{cfg['split_idx']}.mmap"
                subset_data = np.memmap(
                    str(subset_mmap_path),
                    dtype=np.float32,
                    mode="w+",
                    shape=(subsample_size, embedding_dim),
                )
                chunk = CHUNK_SIZE
                for start in range(0, subsample_size, chunk):
                    end = min(start + chunk, subsample_size)
                    subset_data[start:end] = embeddings[cfg["subset_indices"][start:end]]

                model = LSHiForest(n_trees=num_trees, max_depth=max_depth, seed=cfg["split_seed"])
                model.fit(subset_data)
                scores_data = model.score_chunked(subset_data, subsample_size)
                np.save(cfg["output_path"], scores_data)

                del subset_data

                score_paths.append(cfg["output_path"])
                model_names.append(f"subsample_{cfg['split_idx']}")

        del embeddings

    if len(score_paths) < 2:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Only {len(score_paths)} successful splits (need >= 2)")

    metrics = calculate_stability_metrics_n(
        score_paths=score_paths,
        model_names=model_names,
        top_k=top_k,
        spearman_sample_size=spearman_sample_size,
        aggregation="mean",
    )

    shutil.rmtree(temp_dir, ignore_errors=True)

    metrics["subsample_ratio"] = subsample_ratio
    metrics["n_splits"] = n_splits

    return metrics


def _export_anomalies(
    scores: np.ndarray,
    metadata: Any,
    output_path: str | Path,
    k: int = 100,
    *,
    ascending: bool = False,
    label: str = "anomalies",
) -> list[dict]:
    """Export *k* records sorted by anomaly score.

    Parameters
    ----------
    scores : shape (n,)  float32 or float64
    metadata : pd.DataFrame
    output_path : str or Path
    k : int
        Number of records to export.
    ascending : bool
        True → lowest scores first (least anomalous / "normal").
        False → highest scores first (most anomalous).
    label : str
        Human-readable label for the log message.
    """
    import pandas as pd

    valid_mask = np.isfinite(scores)
    valid_idx = np.where(valid_mask)[0]
    n_valid = len(valid_idx)

    if n_valid == 0:
        logger.warning(f"No finite scores, cannot export {label}")
        return []

    actual_k = min(k, n_valid)
    if ascending:
        local = np.argsort(scores[valid_idx], kind="stable")[:actual_k]
    else:
        local = np.argsort(-scores[valid_idx], kind="stable")[:actual_k]
    indices = valid_idx[local]

    columns = ["id", "title", "categories", "update_date"]
    available = [c for c in columns if c in metadata.columns]

    records: list[dict[str, Any]] = []
    for idx in indices:
        record: dict[str, Any] = {"anomaly_score": float(scores[idx])}
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

    logger.success(f"Exported {label}: {actual_k} records → {output_path}")

    return records


def export_top_anomalies(
    scores: np.ndarray,
    metadata: Any,
    output_path: str | Path,
    top_k: int = 100,
) -> list[dict]:
    return _export_anomalies(
        scores, metadata, output_path, k=top_k, ascending=False, label="top anomalies"
    )


def export_bottom_anomalies(
    scores: np.ndarray,
    metadata: Any,
    output_path: str | Path,
    bottom_k: int = 100,
) -> list[dict]:
    """Export the *bottom_k* least anomalous records (most "normal" papers)."""
    return _export_anomalies(
        scores, metadata, output_path, k=bottom_k, ascending=True, label="bottom anomalies"
    )
