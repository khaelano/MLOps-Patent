from contextlib import nullcontext
import json
from pathlib import Path
import time
from typing import Any

from loguru import logger
import mlflow
import numpy as np
import pyarrow.parquet as pq

from patent.config import CHUNK_SIZE
from patent.modeling.evaluate import evaluate_params
from patent.modeling.lsh_iforest import LSHIForest
from patent.utils import flatten_dict


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
) -> dict[str, Any]:
    logger.info(f"Evaluating model {model_path}")
    embeddings_dir = Path(embeddings_dir)
    start_time = time.perf_counter()

    model = LSHIForest.load_model(model_path)
    num_trees = model.meta.num_trees
    max_depth = model.meta.max_depth

    embeddings_paths = [f for f in embeddings_dir.iterdir() if f.is_file()]
    eval = evaluate_params(embeddings_paths, num_trees=num_trees, max_depth=max_depth)
    flattened_eval = flatten_dict(eval["summary"])

    with open(output_path, "w") as f:
        json.dump(flattened_eval, f)

    if mlflow_context:
        with mlflow.start_run(**mlflow_context):
            mlflow.log_metrics(flattened_eval)
            mlflow.log_artifact(str(output_path))

    end_time = time.perf_counter() - start_time
    logger.success(f"Model evaluated in {end_time:.2f}s.")
    logger.success(f"Evaluation metrics saved at {output_path}.")

    return eval
