"""Continuous training pipeline — orchestrates data updates, incremental processing,
retraining, evaluation, and model promotion.

Two trigger modes:

* ``weekly`` — fetches new arXiv data via OAI-PMH, processes only new sources,
  retrains on the full dataset, and promotes if evaluation metrics improve.

* ``drift`` — same workflow as ``weekly`` but intended to be triggered
  by a drift-detection alert (e.g. Prometheus → Alertmanager → webhook).

Usage (CLI)::

    patent pipeline continuous --trigger weekly
    patent pipeline continuous --trigger drift --dry-run
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess
from typing import Any, Literal

from loguru import logger

from patent.config import (
    INTERIM_DATA_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from patent.dataset.ingest import fetch_oai_updates
from patent.modeling.registry import register_from_run
from patent.modeling.train import train_model
from patent.monitoring.drift import save_drift_baseline
from patent.utils import get_last_update_date, set_last_update_date

TriggerMode = Literal["weekly", "drift"]


def _find_new_update_dirs() -> list[Path]:
    """Return update subdirectories whose data has NOT yet been processed.

    Compares ``data/raw/updates/`` subdirs against existing ``data/processed/``
    parquet files.  A subdir is considered "processed" when a corresponding
    parquet exists in ``data/processed/``.
    """
    updates_dir = RAW_DATA_DIR / "updates"
    if not updates_dir.exists():
        return []

    processed_names = {p.stem for p in PROCESSED_DATA_DIR.glob("*.parquet")}

    new_dirs: list[Path] = []
    for subdir in sorted(updates_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if subdir.name not in processed_names:
            new_dirs.append(subdir)

    return new_dirs


def _process_single_source(
    raw_path: Path,
    *,
    embedder_spec: str = "embed-anything-onnx:AllMiniLML6V2Q",
    batch_size: int = 50_000,
) -> Path:
    """Reserialize → clean → embed one raw source, return the processed parquet path."""
    from patent.cli import clean_data, embed_data, reserialize_data

    stem_name = raw_path.name
    out_name = (
        f"{stem_name}.parquet"
        if raw_path.is_dir()
        else Path(stem_name).with_suffix(".parquet").name
    )

    serialized_path = INTERIM_DATA_DIR / "serialized" / out_name
    cleaned_path = INTERIM_DATA_DIR / "cleaned" / out_name
    processed_path = PROCESSED_DATA_DIR / out_name

    is_json = raw_path.suffix == ".zst" or ".json" in raw_path.suffixes

    if not serialized_path.exists():
        logger.info(f"  [reserialize] {raw_path}")
        reserialize_data(file_path=raw_path, output_path=serialized_path, is_json=is_json)
    else:
        logger.info(f"  [skip] {serialized_path.name} already exists")

    if not cleaned_path.exists():
        logger.info(f"  [clean] {serialized_path.name}")
        clean_data(file_path=serialized_path, output_path=cleaned_path)
    else:
        logger.info(f"  [skip] {cleaned_path.name} already exists")

    if not processed_path.exists():
        logger.info(f"  [embed] {cleaned_path.name}")
        embed_data(
            file_path=cleaned_path,
            output_path=processed_path,
            embedder_spec=embedder_spec,
            batch_size=batch_size,
        )
    else:
        logger.info(f"  [skip] {processed_path.name} already exists")

    return processed_path


def _dvc_commit(paths: list[Path], message: str, *, dry_run: bool = False) -> None:
    """Add *paths* to DVC tracking and commit.

    Uses ``dvc add`` on each path, then ``dvc commit``.
    Skips if DVC is not available or *dry_run* is set.
    """
    if dry_run:
        logger.info(f"[dry-run] Would DVC add + commit: {[p.name for p in paths]}")
        return

    try:
        for p in paths:
            if not p.exists():
                continue
            target = str(p)
            # If path is inside data/raw/ (a DVC dep in dvc.yaml), dvc add
            # on individual files within it creates conflicting .dvc files.
            # Use dvc add on data/raw/ itself to capture all changes.
            if str(RAW_DATA_DIR) in str(p.resolve()) and str(p.resolve()) != str(
                RAW_DATA_DIR.resolve()
            ):
                target = str(RAW_DATA_DIR)
            subprocess.run(["dvc", "add", target], check=True, capture_output=True)
        subprocess.run(
            ["dvc", "commit"],
            check=True,
            capture_output=True,
        )
        logger.success(f"DVC commit: {message}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"DVC operation skipped: {e}")


def _update_drift_baseline(
    model_path: str,
    embeddings_dir: Path,
    model_version: str | None,
    *,
    dry_run: bool = False,
) -> None:
    """Score all processed data with the new model and save a fresh drift baseline."""
    from patent.config import CHUNK_SIZE, project_tempdir
    from patent.lshiforest import LSHiForest
    from patent.utils import convert_parquet_to_memmap, get_vectors_from_files

    if dry_run:
        logger.info("[dry-run] Would update drift baseline with new model")
        return

    parquet_paths = sorted(embeddings_dir.glob("*.parquet"))
    if not parquet_paths:
        logger.warning("No processed data to baseline; skipping drift baseline update")
        return

    logger.info("Updating drift baseline with new Production model...")
    model = LSHiForest.load(model_path)

    import shutil

    import numpy as np

    tmpdir = project_tempdir()
    try:
        mmap_path = tmpdir / "drift_embeddings.mmap"
        embedding_dim, total_rows = convert_parquet_to_memmap(
            [str(p) for p in parquet_paths], str(mmap_path)
        )
        mmap = np.memmap(
            str(mmap_path), dtype=np.float32, mode="r", shape=(total_rows, embedding_dim)
        )
        scores = model.score_chunked(mmap, total_rows, chunk_size=CHUNK_SIZE)
        embeddings = get_vectors_from_files([str(p) for p in parquet_paths])
        save_drift_baseline(embeddings, scores, model_version=model_version)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_continuous_pipeline(
    trigger: TriggerMode = "weekly",
    *,
    embedder_spec: str = "embed-anything-onnx:AllMiniLML6V2Q",
    mlflow_experiment: str | None = None,
    top_k: int = 100,
    n_workers: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute the continuous training pipeline end-to-end.

    Parameters
    ----------
    trigger : "weekly" | "drift"
        Trigger mode (affects log messages and metrics tags).
    embedder_spec : str
        Embedder spec ``"<protocol>:<model>"``.
    mlflow_experiment : str | None
        MLflow experiment name for this run.
    top_k : int
        Number of top anomalies to export.
    n_workers : int | None
        Parallel workers for model scoring.
    dry_run : bool
        If True, log steps but don't execute side-effects (DVC, MLflow, etc.).

    Returns
    -------
    dict with keys: ``trigger``, ``new_sources_processed``, ``new_parquet_count``,
    ``trained``, ``promoted``, ``run_id``, ``model_version``.
    """
    result: dict[str, Any] = {
        "trigger": trigger,
        "new_sources_processed": 0,
        "new_parquet_count": 0,
        "trained": False,
        "promoted": False,
        "run_id": None,
        "model_version": None,
    }

    run_ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    logger.info(f"=== Continuous Training Pipeline ({trigger}) — {run_ts} ===")

    # 1. Fetch new data
    logger.info("Step 1/5: Fetching new arXiv data via OAI-PMH...")

    last_date = get_last_update_date()
    if not last_date:
        logger.error(
            "No last_update.txt found. Run 'patent data init' first or set from_date manually."
        )
        return result

    from datetime import timedelta

    from_date = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    to_date = datetime.now().strftime("%Y-%m-%d")

    if from_date > to_date:
        logger.info(f"No new data to fetch (from={from_date} > to={to_date}). Skipping.")
    elif dry_run:
        logger.info(f"[dry-run] Would fetch updates {from_date} → {to_date}")
    else:
        try:
            fetch_oai_updates(RAW_DATA_DIR / "updates", from_date, to_date)
            set_last_update_date(to_date)
            logger.success(f"OAI-PMH updates fetched: {from_date} → {to_date}")
        except Exception as e:
            logger.error(f"Failed to fetch updates: {e}")
            return result

    # 2. Process new sources incrementally
    logger.info("Step 2/5: Processing new data sources...")

    new_dirs = _find_new_update_dirs()
    snapshot_file = RAW_DATA_DIR / "arxiv-metadata-oai-snapshot.json.zst"

    sources_to_process: list[Path] = []
    for subdir in new_dirs:
        stem = f"{subdir.name}.parquet"
        if not (PROCESSED_DATA_DIR / stem).exists():
            sources_to_process.append(subdir)

    if snapshot_file.exists():
        stem = "arxiv-metadata-oai-snapshot.parquet"
        if not (PROCESSED_DATA_DIR / stem).exists():
            sources_to_process.insert(0, snapshot_file)

    if not sources_to_process:
        logger.info("All sources already processed. Skipping data processing.")
    else:
        logger.info(f"Processing {len(sources_to_process)} source(s) incrementally...")
        for source in sources_to_process:
            try:
                _process_single_source(source, embedder_spec=embedder_spec)
                result["new_sources_processed"] += 1
            except Exception as e:
                logger.error(f"Failed to process {source}: {e}")
                return result

    processed_files = sorted(PROCESSED_DATA_DIR.glob("*.parquet"))
    result["new_parquet_count"] = len(processed_files)
    logger.info(f"Total processed parquets available: {len(processed_files)}")

    # 3. DVC commit data
    logger.info("Step 3/5: Committing data to DVC...")

    data_paths = [
        RAW_DATA_DIR / "updates",
        INTERIM_DATA_DIR / "serialized",
        INTERIM_DATA_DIR / "cleaned",
        PROCESSED_DATA_DIR,
        RAW_DATA_DIR / "last_update.txt",
    ]
    _dvc_commit(
        [p for p in data_paths if p.exists()],
        f"[{trigger}] data update {run_ts}",
        dry_run=dry_run,
    )

    # 4. Train + evaluate
    logger.info("Step 4/5: Training model on full dataset...")

    output_dir = MODELS_DIR / f"continuous_{run_ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        logger.info(
            f"[dry-run] Would train on {len(processed_files)} parquets, output to {output_dir}"
        )
        result["trained"] = True
        return result

    ctx = {"experiment_name": mlflow_experiment} if mlflow_experiment else None

    try:
        train_result = train_model(
            embeddings_dir=PROCESSED_DATA_DIR,
            output_dir=output_dir,
            model_params={},
            mlflow_context=ctx,
            top_k=top_k,
            n_workers=n_workers,
        )
        result["trained"] = True
        result["run_id"] = train_result["run_id"]
        result["pyfunc_version"] = train_result["pyfunc_version"]
        logger.success(
            f"Training complete. MLflow run_id={train_result['run_id']}, "
            f"pyfunc_v={train_result['pyfunc_version']}"
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return result

    # 5. Register + promote (if better)
    logger.info("Step 5/5: Evaluating against Production and promoting if better...")

    if train_result["run_id"] and train_result["pyfunc_version"]:
        try:
            reg_result = register_from_run(
                run_id=train_result["run_id"],
                pyfunc_version=train_result["pyfunc_version"],
                metric_key="stability/jaccard_aggregated",
            )
            result["promoted"] = reg_result["promoted_to_production"]
            result["model_version"] = reg_result["version"]

            if reg_result["promoted_to_production"]:
                logger.success(
                    f"Model v{reg_result['version']} promoted to Production! "
                    f"(metric={reg_result['metric_key']}={reg_result['metric_value']:.4f})"
                )
                _update_drift_baseline(
                    str(output_dir / "model.lshif"),
                    PROCESSED_DATA_DIR,
                    model_version=reg_result["version"],
                )
            else:
                logger.info(
                    f"Model not promoted (new metric ≤ production). "
                    f"Previous: v{reg_result['previous_prod_version']} "
                    f"({reg_result['previous_metric_value']})"
                )
        except Exception as e:
            logger.error(f"Registration/promotion failed: {e}")
    else:
        logger.warning("No MLflow run_id — skipping registration. Did training succeed?")

    _dvc_commit(
        [output_dir],
        f"[{trigger}] model update {run_ts}",
        dry_run=dry_run,
    )

    logger.success(f"=== Continuous Training Pipeline ({trigger}) complete ===")
    return result
