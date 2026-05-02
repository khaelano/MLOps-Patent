from contextlib import nullcontext
import json
from pathlib import Path
import time
from typing import Any

from loguru import logger
import mlflow

from patent.config import CHUNK_SIZE
from patent.modeling.evaluate import evaluate_params
from patent.modeling.lsh_iforest import LSHIForest
from patent.utils import flatten_dict, parquet_to_memmap


def process_embeddings(
    parquet_path: str,
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
    embeddings_dim: int,
    embeddings_path: str | Path,
    output_dir: str | Path,
    model_params: dict[str, Any] = {},
    mlflow_context: dict[str, Any] | None = None,
) -> str:
    logger.info(f"Training model from embeddings in {embeddings_path}")
    start_time = time.perf_counter()

    output_dir = Path(output_dir)
    model_path = str((output_dir / "model.lshif"))
    baseline_path = str((output_dir / "baseline_depth.npy"))

    mlflow_run = (
        mlflow.start_run(**mlflow_context) if mlflow_context is not None else nullcontext()
    )
    with mlflow_run:
        model = LSHIForest(**model_params, chunk_size=CHUNK_SIZE)
        model.build_forest(
            embeddings_dim=embeddings_dim,
            embeddings_path=embeddings_path,
            baseline_output_path=baseline_path,
        )
        model.save_model(model_path)

        end_time = time.perf_counter() - start_time
        logger.success(f"Model trained successfully in {end_time:.2f}s.")
        logger.success(f"Model path is {model_path}.")

    return str(output_dir)


def evaluate_model(
    model_path: str | Path,
    embeddings_path: str | Path,
    output_path: str | Path = "evaluation.json",
    mlflow_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    logger.info(f"Evaluating model {model_path}")
    start_time = time.perf_counter()

    model = LSHIForest.load_model(model_path)
    num_trees = model.meta.num_trees
    max_depth = model.meta.max_depth

    eval = evaluate_params(embeddings_path, num_trees=num_trees, max_depth=max_depth)
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
