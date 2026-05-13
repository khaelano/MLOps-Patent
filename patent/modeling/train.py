from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import time
from typing import Any

from loguru import logger
import mlflow
import numpy as np

from patent.config import CHUNK_SIZE, project_tempdir
from patent.lshiforest import LSHiForest
from patent.modeling.evaluate import (
    analyze_score_distribution,
    convert_embeddings_to_memmap,
    distance_to_centroid_correlation,
    evaluate_params,
    evaluate_subsampling_stability,
    export_bottom_anomalies,
    export_top_anomalies,
)
from patent.utils import convert_parquet_to_memmap, flatten_dict, load_parquet_metadata


def parquet_to_memmap(
    pq_paths: str | list[str],
    mmap_path: str,
    column: str = "embedding",
) -> tuple[int, str]:
    """Utility to convert a column in one or multiple parquet files into a memory-mapped numpy array."""
    if isinstance(pq_paths, (str, Path)):
        pq_paths = [str(pq_paths)]
    else:
        pq_paths = [str(p) for p in pq_paths]

    if not pq_paths:
        raise ValueError("No parquet paths provided.")

    embedding_dim, _ = convert_parquet_to_memmap(pq_paths, str(mmap_path), column)
    logger.info(f"Inferred embedding dimension: {embedding_dim}")
    return embedding_dim, str(mmap_path)


def process_embeddings(
    parquet_path: str | list[str],
    output_path: str,
) -> tuple[int, str]:
    logger.info(f"Generating embeddings memmap from {parquet_path}...")
    start_time = time.perf_counter()

    num_dim, mmap_path = parquet_to_memmap(parquet_path, output_path)

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

    # ── Convert parquet → memmap ONCE, reuse for fit + baseline scoring ──
    embed_temp_dir = project_tempdir()
    mmap_path = embed_temp_dir / "embeddings.mmap"
    try:
        logger.info("Converting Parquet embeddings to memory-mapped array...")
        embedding_dim, total_rows = convert_parquet_to_memmap(embeddings_paths, str(mmap_path))
        if total_rows == 0 or embedding_dim == 0:
            raise ValueError("No embeddings found in provided files")
        embeddings_mmap = np.memmap(
            str(mmap_path),
            dtype=np.float32,
            mode="r",
            shape=(total_rows, embedding_dim),
        )

        mlflow_run = (
            mlflow.start_run(**mlflow_context) if mlflow_context is not None else nullcontext()
        )
        with mlflow_run:
            # Fit directly from the shared memmap (only tiny subsamples are materialised)
            model = LSHiForest(**model_params)
            model.fit(embeddings_mmap)
            model.save(model_path)

            # Baseline scoring from the same memmap (no second conversion)
            logger.info("Computing baseline anomaly scores...")
            baseline_scores = model.score_chunked(
                embeddings_mmap, total_rows, chunk_size=CHUNK_SIZE
            )
            np.save(baseline_path, baseline_scores)
            logger.success(f"Baseline scores saved to {baseline_path}")

            end_time = time.perf_counter() - start_time
            logger.success(f"Model trained successfully in {end_time:.2f}s.")
            logger.success(f"Model path is {model_path}.")
    finally:
        shutil.rmtree(embed_temp_dir, ignore_errors=True)

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

    model = LSHiForest.load(model_path)
    num_trees = model.n_trees
    max_depth = model.max_depth

    embeddings_paths = [str(p) for p in sorted(embeddings_dir.glob("*.parquet"))]
    if not embeddings_paths:
        raise FileNotFoundError(f"No .parquet files found in {embeddings_dir}")

    mlflow_run = (
        mlflow.start_run(**mlflow_context) if mlflow_context is not None else nullcontext()
    )
    with mlflow_run:
        eval_result: dict[str, Any] = {}
        embed_temp_dir = project_tempdir()

        try:
            # ── Create the memmap ONCE, share with evaluate_params ──
            logger.info("Converting embeddings to shared memmap...")
            mmap_path = embed_temp_dir / "embeddings.mmap"
            embedding_dim, total_rows = convert_embeddings_to_memmap(embeddings_paths, mmap_path)

            logger.info("Running seed-based stability evaluation...")
            seed_stability = evaluate_params(
                [Path(p) for p in embeddings_paths],
                num_trees=num_trees,
                max_depth=max_depth,
                n_workers=n_workers,
                shared_mmap=(str(mmap_path), total_rows, embedding_dim),
            )
            eval_result["stability"] = seed_stability["summary"]

            logger.info("Loading metadata for evaluation...")
            metadata = load_parquet_metadata(embeddings_paths)

            if total_rows == 0:
                logger.warning("No embeddings found, skipping remaining evaluations")
            else:
                logger.info("Computing anomaly scores via chunked memmap scoring...")
                embeddings_mmap = np.memmap(
                    mmap_path, dtype=np.float32, mode="r", shape=(total_rows, embedding_dim)
                )
                scores = model.score_chunked(embeddings_mmap, total_rows)

                logger.info("Analyzing score distribution...")
                eval_result["score_distribution"] = analyze_score_distribution(scores)

                logger.info("Computing distance-to-centroid correlation...")
                eval_result["centroid_correlation"] = distance_to_centroid_correlation(
                    embeddings_paths,
                    scores,
                    mmap_path=str(mmap_path),
                )

                logger.info(f"Exporting top {top_k} anomalies...")
                top_path = Path(output_path).parent / "top_anomalies.json"
                export_top_anomalies(scores, metadata, top_path, top_k=top_k)
                eval_result["top_anomalies_path"] = str(top_path)

                logger.info(f"Exporting bottom {top_k} anomalies...")
                bottom_path = Path(output_path).parent / "bottom_anomalies.json"
                export_bottom_anomalies(scores, metadata, bottom_path, bottom_k=top_k)
                eval_result["bottom_anomalies_path"] = str(bottom_path)

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
            bottom_path = Path(output_path).parent / "bottom_anomalies.json"
            if bottom_path.exists():
                mlflow.log_artifact(str(bottom_path))

    end_time = time.perf_counter() - start_time
    logger.success(f"Model evaluated in {end_time:.2f}s.")
    logger.success(f"Evaluation metrics saved at {output_path}.")

    return eval_result
