from contextlib import nullcontext
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

from loguru import logger
import mlflow
import numpy as np

from patent.config import CHUNK_SIZE, PROJ_ROOT, project_tempdir
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


def _log_pyfunc_model(model_path: str, model_params: dict[str, Any]) -> int | None:
    """Log a pyfunc-flavour model, register it, and return the version number.

    Called *inside* an active MLflow run.  Uses ``registered_model_name``
    to create a new model version immediately (MLflow 3.x behaviour — the
    model is stored as a standalone entity, not as a run artifact).

    Returns the registered model version number, or ``None`` on failure.
    """
    from patent.modeling.pyfunc_model import DEFAULT_EMBEDDER_SPEC, LSHiForestPyfuncModel

    embedder_spec = model_params.get("embedder_spec", DEFAULT_EMBEDDER_SPEC)
    logger.info(f"Logging pyfunc model (embedder={embedder_spec})")

    try:
        model_info = mlflow.pyfunc.log_model(
            name="pyfunc_model",
            python_model=LSHiForestPyfuncModel(embedder_spec=embedder_spec),
            artifacts={"model.lshif": model_path},
            code_paths=[str(PROJ_ROOT / "patent")],
            registered_model_name="patent-lshiforest",
            pip_requirements=[
                "embed-anything>=0.7.0",
                "loguru>=0.7",
                "numpy>=1.24",
                "pandas>=2.0",
                "prometheus-client>=0.22.0",
                "prometheus-fastapi-instrumentator>=7.0.0",
            ],
        )
        version: Any = getattr(model_info, "registered_model_version", None)
        logger.success(f"Pyfunc model logged and registered as version {version}")
        return int(version) if version is not None else None
    except Exception:
        logger.exception("Failed to log pyfunc model")
        return None


def train_model(
    embeddings_dir: str | Path,
    output_dir: str | Path,
    model_params: dict[str, Any] = {},
    mlflow_context: dict[str, Any] | None = None,
    *,
    top_k: int = 100,
    do_subsampling: bool = False,
    n_workers: int | None = None,
) -> dict[str, Any]:
    """Train an LSHiForest model and run evaluation inline.

    When *mlflow_context* is provided (e.g. ``{"experiment_name": "my-exp"}``),
    the model, training metrics, evaluation metrics, and anomaly exports are all
    logged to a single MLflow run.  When *mlflow_context* is ``None``, MLflow is
    skipped entirely.

    Returns a dict with keys ``output_dir``, ``eval_result``, ``run_id``,
    and ``pyfunc_version`` (each is ``None`` when MLflow is not active).
    """
    logger.info(f"Training model from embeddings in {embeddings_dir}")
    embeddings_dir = Path(embeddings_dir)
    start_time = time.perf_counter()

    output_dir = Path(output_dir)
    model_path = str((output_dir / "model.lshif"))
    eval_path = str((output_dir / "evaluation.json"))
    top_path = str((output_dir / "top_anomalies.json"))
    bottom_path = str((output_dir / "bottom_anomalies.json"))
    embeddings_paths = [str(p) for p in embeddings_dir.glob("*.parquet")]
    if not embeddings_paths:
        raise FileNotFoundError(f"No .parquet files found in {embeddings_dir}")

    # ── Convert parquet → memmap ONCE, reuse for fit + scoring + evaluation ──
    embed_temp_dir = project_tempdir()
    mmap_path = embed_temp_dir / "embeddings.mmap"
    eval_result: dict[str, Any] = {}
    run_id: str | None = None
    pyfunc_version: int | None = None

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

        # ── MLflow: set_experiment before start_run ─────────────────────
        if mlflow_context is not None:
            experiment_name = mlflow_context.pop("experiment_name", None)
            if experiment_name:
                mlflow.set_experiment(experiment_name)
            mlflow_run: Any = mlflow.start_run(**mlflow_context)
            using_mlflow = True
        else:
            mlflow_run = nullcontext()
            using_mlflow = False

        with mlflow_run:
            # ── Log model parameters ──
            if using_mlflow:
                mlflow.log_params(model_params)
                mlflow.log_param("embedding_dim", embedding_dim)
                mlflow.log_param("total_rows", total_rows)
                mlflow.log_param("top_k", top_k)
                active_run = mlflow.active_run()
                if active_run is not None:
                    run_id = active_run.info.run_id

            # ── Fit ──
            t_fit = time.perf_counter()
            model = LSHiForest(**model_params)
            model.fit(embeddings_mmap)
            fit_time = time.perf_counter() - t_fit
            logger.success(f"Model fit in {fit_time:.2f}s")

            model.save(model_path)
            if using_mlflow:
                mlflow.log_artifact(model_path)
                # ── Also log as a pyfunc model for ``mlflow models build-docker`` ──
                pyfunc_version = _log_pyfunc_model(model_path, model_params)

            # ── Baseline scoring ──
            t_score = time.perf_counter()
            logger.info("Computing baseline anomaly scores...")
            baseline_scores = model.score_chunked(
                embeddings_mmap, total_rows, chunk_size=CHUNK_SIZE
            )
            baseline_time = time.perf_counter() - t_score
            logger.success(f"Baseline scoring in {baseline_time:.2f}s")

            if using_mlflow:
                total_time = time.perf_counter() - start_time
                mlflow.log_metrics(
                    {
                        "train/fit_time_s": fit_time,
                        "train/baseline_scoring_time_s": baseline_time,
                        "train/total_time_s": total_time,
                    }
                )

            # ── Evaluation (inline, reusing the same memmap) ──
            metadata = load_parquet_metadata(embeddings_paths)

            # Seed-based stability
            logger.info("Running seed-based stability evaluation...")
            t_stab = time.perf_counter()
            seed_stability = evaluate_params(
                [Path(p) for p in embeddings_paths],
                num_trees=model.n_trees,
                max_depth=model.max_depth,
                n_workers=n_workers,
                shared_mmap=(str(mmap_path), total_rows, embedding_dim),
            )
            stab_time = time.perf_counter() - t_stab
            eval_result["stability"] = seed_stability["summary"]

            # Score distribution
            eval_result["score_distribution"] = analyze_score_distribution(baseline_scores)

            # Distance-to-centroid correlation
            logger.info("Computing distance-to-centroid correlation...")
            eval_result["centroid_correlation"] = distance_to_centroid_correlation(
                embeddings_paths,
                baseline_scores,
                mmap_path=str(mmap_path),
            )

            # Export top / bottom anomalies
            logger.info(f"Exporting top {top_k} anomalies...")
            export_top_anomalies(baseline_scores, metadata, top_path, top_k=top_k)
            eval_result["top_anomalies_path"] = top_path

            logger.info(f"Exporting bottom {top_k} anomalies...")
            export_bottom_anomalies(baseline_scores, metadata, bottom_path, bottom_k=top_k)
            eval_result["bottom_anomalies_path"] = bottom_path

            # Subsampling stability (optional — expensive)
            if do_subsampling:
                logger.info("Running subsampling stability (5 splits)...")
                subsample_stability = evaluate_subsampling_stability(
                    [Path(p) for p in embeddings_paths],
                    num_trees=model.n_trees,
                    max_depth=model.max_depth,
                    n_splits=5,
                )
                eval_result["subsampling_stability"] = subsample_stability["summary"]

            # ── Persist & log evaluation ──
            flattened_eval = flatten_dict(eval_result)
            with open(eval_path, "w") as f:
                json.dump(flattened_eval, f, indent=2)
            logger.success(f"Evaluation saved to {eval_path}")

            if using_mlflow:
                # Log numeric eval metrics
                mlflow.log_metrics(
                    {k: v for k, v in flattened_eval.items() if isinstance(v, (int, float))}
                )
                # Log eval-specific timing
                mlflow.log_metric("evaluation/seed_stability_time_s", stab_time)
                # Log eval artifacts
                mlflow.log_artifact(eval_path)
                if os.path.exists(top_path):
                    mlflow.log_artifact(top_path)
                if os.path.exists(bottom_path):
                    mlflow.log_artifact(bottom_path)

    finally:
        shutil.rmtree(embed_temp_dir, ignore_errors=True)

    return {
        "output_dir": str(output_dir),
        "eval_result": eval_result,
        "run_id": run_id,
        "pyfunc_version": pyfunc_version,
    }


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

    if mlflow_context is not None:
        experiment_name = mlflow_context.pop("experiment_name", None)
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        mlflow_run: Any = mlflow.start_run(**mlflow_context)
        using_mlflow = True
    else:
        mlflow_run = nullcontext()
        using_mlflow = False

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

        if using_mlflow:
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
