from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

from loguru import logger
import mlflow
import numpy as np
import pyarrow.parquet as pq

from patent.config import CHUNK_SIZE
from patent.modeling.evaluate import (
    analyze_score_distribution,
    convert_embeddings_to_memmap,
    distance_to_centroid_correlation,
    evaluate_params,
    evaluate_subsampling_stability,
    export_top_anomalies,
    score_memmap_chunked,
)
from patent.modeling.lsh_iforest import LSHIForest
from patent.utils import flatten_dict, load_parquet_metadata


def parquet_to_memmap(
    pq_paths: str | list[str],
    mmap_path: str,
    column: str = "embedding",
    chunk_size: int = 200_000,
) -> tuple[int, str]:
    """Utility to convert a column in one or multiple parquet files into a memory-mapped numpy array."""
    if isinstance(pq_paths, (str, Path)):
        pq_paths = [str(pq_paths)]
    else:
        pq_paths = [str(p) for p in pq_paths]

    if not pq_paths:
        raise ValueError("No parquet paths provided.")

    total_rows = 0
    embedding_dim = None

    for pq_path in pq_paths:
        pq_file = pq.ParquetFile(pq_path)
        total_rows += pq_file.metadata.num_rows

    try:
        first_pq_file = pq.ParquetFile(pq_paths[0])
        first_batch = next(first_pq_file.iter_batches(batch_size=1, columns=[column]))
        embedding_dim = len(first_batch.column(0)[0])
    except StopIteration:
        raise ValueError("Parquet file is empty.")
    except Exception as e:
        raise ValueError(f"Failed to infer embedding dimension: {e}")

    logger.info(f"Inferred embedding dimension: {embedding_dim} over {total_rows} total rows")

    mmap_file = np.memmap(mmap_path, dtype="float32", mode="w+", shape=(total_rows, embedding_dim))

    curr_idx = 0
    for pq_path in pq_paths:
        logger.debug(f"Processing {pq_path} to memmap...")
        pq_file = pq.ParquetFile(pq_path)

        for batch in pq_file.iter_batches(batch_size=chunk_size, columns=[column]):
            flat_arr = batch.column(0).flatten().to_numpy()

            arr = flat_arr.reshape(-1, embedding_dim)

            if arr.shape[1] != embedding_dim:
                raise ValueError(
                    f"Dimension mismatch at row {curr_idx}. "
                    f"Expected {embedding_dim}, got {arr.shape[1]}"
                )

            slice_end = curr_idx + len(arr)
            mmap_file[curr_idx:slice_end] = arr

            curr_idx = slice_end

    mmap_file.flush()

    if curr_idx != total_rows:
        logger.warning(
            f"Processed {curr_idx} rows, but metadata says {total_rows}. "
            "Check for nulls or corrupted rows."
        )

    return (embedding_dim, mmap_path)


def process_embeddings(
    parquet_path: str | list[str],
    output_path: str,
) -> tuple[int, str]:
    logger.info(f"Generating embeddings memmap from {parquet_path}...")
    start_time = time.perf_counter()

    num_dim, mmap_path = parquet_to_memmap(parquet_path, output_path, chunk_size=CHUNK_SIZE)

    end_time = time.perf_counter() - start_time
    logger.success(f"Embeddings memmap generated successfully in {end_time:.2f}s.")
    logger.success(f"Embeddings memmap path is {output_path}.")

    return num_dim, mmap_path


def train_model(
    embeddings_dir: str | Path,
    output_dir: str | Path,
    model_params: dict[str, Any] = {},
    mlflow_context: dict[str, Any] | None = None,
) -> str:
    logger.info(f"Training model from embeddings in {embeddings_dir}")
    embeddings_dir = Path(embeddings_dir)
    start_time = time.perf_counter()

    output_dir = Path(output_dir)
    model_path = str((output_dir / "model.lshif"))
    baseline_path = str((output_dir / "baseline_depth.npy"))
    embeddings_paths = [str(p) for p in embeddings_dir.glob("*.parquet")]
    if not embeddings_paths:
        raise FileNotFoundError(f"No .parquet files found in {embeddings_dir}")

    mlflow_run = (
        mlflow.start_run(**mlflow_context) if mlflow_context is not None else nullcontext()
    )
    with mlflow_run:
        model = LSHIForest(**model_params, chunk_size=CHUNK_SIZE)
        model.build_forest(embeddings_paths=embeddings_paths, baseline_output_path=baseline_path)
        model.save_model(model_path)

        end_time = time.perf_counter() - start_time
        logger.success(f"Model trained successfully in {end_time:.2f}s.")
        logger.success(f"Model path is {model_path}.")

    return str(output_dir)


def evaluate_model(
    model_path: str | Path,
    embeddings_dir: str | Path,
    output_path: str | Path = "evaluation.json",
    mlflow_context: dict[str, Any] | None = None,
    top_k: int = 100,
    do_subsampling: bool = False,
    subsample_splits: int = 5,
    n_workers: int | None = None,
) -> dict[str, Any]:
    logger.info(f"Evaluating model {model_path}")
    embeddings_dir = Path(embeddings_dir)
    start_time = time.perf_counter()

    model = LSHIForest.load_model(model_path)
    num_trees = model.meta.num_trees
    max_depth = model.meta.max_depth

    embeddings_paths = [str(p) for p in sorted(embeddings_dir.glob("*.parquet"))]
    if not embeddings_paths:
        raise FileNotFoundError(f"No .parquet files found in {embeddings_dir}")

    mlflow_run = (
        mlflow.start_run(**mlflow_context) if mlflow_context is not None else nullcontext()
    )
    with mlflow_run:
        eval_result: dict[str, Any] = {}
        embed_temp_dir = Path(tempfile.mkdtemp())

        try:
            logger.info("Running seed-based stability evaluation...")
            seed_stability = evaluate_params(
                [Path(p) for p in embeddings_paths],
                num_trees=num_trees,
                max_depth=max_depth,
                n_workers=n_workers,
            )
            eval_result["stability"] = seed_stability["summary"]

            logger.info("Converting embeddings to shared memmap...")
            mmap_path = embed_temp_dir / "embeddings.mmap"
            embedding_dim, total_rows = convert_embeddings_to_memmap(embeddings_paths, mmap_path)

            logger.info("Loading metadata for evaluation...")
            metadata = load_parquet_metadata(embeddings_paths)

            if total_rows == 0:
                logger.warning("No embeddings found, skipping remaining evaluations")
            else:
                logger.info("Computing anomaly scores via chunked memmap scoring...")
                embeddings_mmap = np.memmap(
                    mmap_path, dtype=np.float32, mode="r", shape=(total_rows, embedding_dim)
                )
                scores = score_memmap_chunked(model, embeddings_mmap, total_rows)

                logger.info("Analyzing score distribution...")
                eval_result["score_distribution"] = analyze_score_distribution(scores)

                logger.info("Computing distance-to-centroid correlation...")
                eval_result["centroid_correlation"] = distance_to_centroid_correlation(
                    embeddings_paths, scores
                )

                logger.info(f"Exporting top {top_k} anomalies...")
                top_path = Path(output_path).parent / "top_anomalies.json"
                export_top_anomalies(scores, metadata, top_path, top_k=top_k)
                eval_result["top_anomalies_path"] = str(top_path)

            if do_subsampling:
                logger.info(f"Running subsampling stability ({subsample_splits} splits)...")
                subsample_stability = evaluate_subsampling_stability(
                    [Path(p) for p in embeddings_paths],
                    num_trees=num_trees,
                    max_depth=max_depth,
                    n_splits=subsample_splits,
                )
                eval_result["subsampling_stability"] = subsample_stability["summary"]

        finally:
            shutil.rmtree(embed_temp_dir, ignore_errors=True)

        flattened_eval = flatten_dict(eval_result)
        with open(output_path, "w") as f:
            json.dump(flattened_eval, f, indent=2)

        if mlflow_context:
            mlflow.log_metrics(
                {k: v for k, v in flattened_eval.items() if isinstance(v, (int, float))}
            )
            mlflow.log_artifact(str(output_path))
            top_path = Path(output_path).parent / "top_anomalies.json"
            if top_path.exists():
                mlflow.log_artifact(str(top_path))

    end_time = time.perf_counter() - start_time
    logger.success(f"Model evaluated in {end_time:.2f}s.")
    logger.success(f"Evaluation metrics saved at {output_path}.")

    return eval_result
