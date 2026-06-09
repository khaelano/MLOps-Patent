from pathlib import Path
from typing import cast

from loguru import logger
import pandas as pd
import typer

from patent.config import (
    CHUNK_SIZE,
    INTERIM_DATA_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from patent.dataset.ingest import (
    download_kaggle_snapshot,
    extract_latest_update,
    fetch_oai_updates,
)
from patent.dataset.preprocess import (
    clean_df,
    embed,
    parse_oai_xml_directory,
    parse_oai_xml_file,
    parse_snapshot_json_file,
)
from patent.utils import get_last_update_date, set_last_update_date

app = typer.Typer(help="MLOps Patent Pipeline CLI", no_args_is_help=True)

data_app = typer.Typer(help="Data ingestion and preprocessing commands", no_args_is_help=True)
app.add_typer(data_app, name="data")


@data_app.command("init")
def init_data(
    output_path: Path = typer.Option(
        RAW_DATA_DIR / "arxiv-metadata-oai-snapshot.json.zst",
        "--output-path",
        "-o",
        help="File path for metadata storage",
    ),
):
    """Bootstrap the dataset by downloading the Kaggle snapshot and partitioning it."""

    snapshot_file = download_kaggle_snapshot(output_path)
    last_update_date = extract_latest_update(snapshot_file)
    set_last_update_date(last_update_date)


@data_app.command("update")
def update_data(
    output_path: Path = typer.Option(
        RAW_DATA_DIR / "updates", "--output-path", "-o", help="Updates directory"
    ),
    from_date: str = typer.Option(
        None,
        help="Start date (YYYY-MM-DD). Defaults to the day after the last update date in metadata.",
    ),
    to_date: str = typer.Option(None, help="End date (YYYY-MM-DD). Defaults to today."),
):
    """Fetch incremental updates from arXiv using the OAI-PMH interface."""
    from datetime import datetime, timedelta

    if not from_date:
        last_date = get_last_update_date()
        if not last_date:
            typer.secho(
                "Error: No last update date found. Please run 'init' first or provide --from-date.",
                err=True,
                fg="red",
            )
            raise typer.Exit(1)
        from_date = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )

    if not to_date:
        to_date = datetime.now().strftime("%Y-%m-%d")

    fetch_oai_updates(output_path, from_date, to_date)
    set_last_update_date(to_date)


@data_app.command("reserialize")
def reserialize_data(
    file_path: Path = typer.Argument(..., help="Path to the raw XML/JSON file or directory"),
    output_path: Path = typer.Option(
        None,
        help="Path to save the parsed parquet file. Defaults to data/interim/serialized/[filename].parquet",
    ),
    is_json: bool = typer.Option(False, "--json", help="Parse as JSON instead of XML"),
):
    """Parse a raw XML (or JSON) file/directory to a DataFrame and serialize it as Parquet in the interim folder."""

    if not file_path.exists():
        logger.error(f"Path not found: {file_path}")
        raise typer.Exit(1)

    if not output_path:
        # Strip .zst before replacing the data suffix for clean output naming
        stem_name = file_path.name
        if stem_name.endswith(".zst"):
            stem_name = stem_name[:-4]
        out_name = (
            f"{stem_name}.parquet"
            if file_path.is_dir()
            else Path(stem_name).with_suffix(".parquet").name
        )
        output_path = INTERIM_DATA_DIR / "serialized" / out_name

    if is_json:
        parse_snapshot_json_file(file_path, output_path)
    else:
        if file_path.is_dir():
            parse_oai_xml_directory(file_path, output_path)
        else:
            parse_oai_xml_file(file_path, output_path)

    logger.info(f"Successfully serialized parsed data to {output_path}")


@data_app.command("clean")
def clean_data(
    file_path: Path = typer.Argument(..., help="Path to the input corpus Parquet"),
    output_path: Path = typer.Option(
        None,
        help="Path to dump cleaned artifact. Defaults to data/interim/cleaned/[filename].parquet",
    ),
):
    """Read input artifact (parquet), apply text cleaning, drop missing, and serialize cleaned payload."""

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise typer.Exit(1)

    if not output_path:
        output_path = INTERIM_DATA_DIR / "cleaned" / file_path.with_suffix(".parquet").name

    df = pd.read_parquet(file_path)
    df_cleaned = clean_df(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_parquet(output_path, index=False)
    logger.info(f"Successfully serialized cleaned data to {output_path}")


@data_app.command("embed")
def embed_data(
    file_path: Path = typer.Argument(..., help="Path to the cleaned corpus Parquet"),
    output_path: Path = typer.Option(
        None,
        help="Path to dump embedded artifact. Defaults to data/processed/[filename].parquet",
    ),
    embedder_spec: str = typer.Option(
        "embed-anything-onnx:AllMiniLML6V2Q",
        "--embedder",
        help="Embedder spec: '<protocol>:<model>' (e.g. 'embed-anything-onnx:AllMiniLML6V2Q')",
    ),
    batch_size: int = typer.Option(CHUNK_SIZE, help="Row count per chunk to process sequentially"),
):
    """Read a cleaned Parquet artifact sequentially in chunks using PyArrow to
    minimize memory footprint.  Generates text embeddings for titles using
    a pluggable embedder (default: embed-anything-onnx), and streams the
    resulting features iteratively into a new Parquet file.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from patent.dataset.embedders import get_embedder

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise typer.Exit(1)

    if not output_path:
        output_path = PROCESSED_DATA_DIR / file_path.name

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing '{file_path}' in chunks of {batch_size} rows...")
    parquet_file = pq.ParquetFile(file_path)
    writer = None

    logger.info(f"Loading embedder: {embedder_spec}")
    try:
        embedder = get_embedder(embedder_spec)
    except Exception as e:
        logger.error(f"Failed to load embedder: {e}")
        raise typer.Exit(1)

    try:
        for i, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
            logger.info(f"Processing chunk {i + 1}...")
            df_chunk = batch.to_pandas()

            df_embedded = embed(df_chunk, embedder)

            table = pa.Table.from_pandas(df_embedded)

            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)

            writer.write_table(table)
    finally:
        embedder.stop_pool()

    if writer:
        writer.close()

    logger.info(f"Successfully serialized chunked feature embeddings to {output_path}")


model_app = typer.Typer(help="Model training and evaluation commands", no_args_is_help=True)
app.add_typer(model_app, name="model")


@model_app.command("train")
def train_cmd(
    data: Path = typer.Argument(
        PROCESSED_DATA_DIR,
        help="Directory containing .parquet embeddings (default: data/processed/)",
    ),
    output: Path = typer.Argument(
        MODELS_DIR,
        help="Output directory for the model (default: models/)",
    ),
    num_trees: int = typer.Option(200, "--num-trees", "-t", help="Number of isolation trees"),
    max_depth: int = typer.Option(21, "--max-depth", "-m", help="Maximum tree depth"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    lsh_family: str = typer.Option("l2", "--lsh-family", "-f", help="LSH family: l2, angle"),
    eta: float = typer.Option(
        0.0, "--eta", help="Granularity: 0=local anomalies, 1=global anomalies"
    ),
    params: Path = typer.Option(
        None, "--params", "-p", help="JSON file with additional model params"
    ),
    mlflow_experiment: str = typer.Option(
        None, "--mlflow-experiment", help="MLflow experiment name"
    ),
    top_k: int = typer.Option(100, "--top-k", "-k", help="Number of top anomalies to export"),
    do_subsampling: bool = typer.Option(
        False, "--do-subsampling", help="Enable bootstrap subsampling stability (expensive)"
    ),
    n_workers: int = typer.Option(
        None,
        "--n-workers",
        "-w",
        help="Number of parallel workers for evaluation (default: auto)",
    ),
):
    """Train an LSHiForest model on processed embeddings, then evaluate inline.

    When --mlflow-experiment is provided, model params, training metrics,
    evaluation metrics, and anomaly exports are all logged to a single
    MLflow run.
    """
    import json

    from patent.modeling.train import train_model

    model_cfg = {
        "n_trees": num_trees,
        "max_depth": max_depth,
        "seed": seed,
        "family": lsh_family,
        "eta": eta,
    }
    if params and params.exists():
        with open(params, "r") as f:
            model_cfg.update(json.load(f))

    ctx = {"experiment_name": mlflow_experiment} if mlflow_experiment else None

    output.mkdir(parents=True, exist_ok=True)

    result = train_model(
        embeddings_dir=data,
        output_dir=output,
        model_params=model_cfg,
        mlflow_context=ctx,
        top_k=top_k,
        do_subsampling=do_subsampling,
        n_workers=n_workers,
    )

    if result["run_id"]:
        logger.success(f"MLflow run ID: {result['run_id']}")
        if result.get("pyfunc_version"):
            logger.success(f"Pyfunc version: {result['pyfunc_version']}")
        logger.info(f"To register this model: patent model register --run-id {result['run_id']}")


@model_app.command("evaluate")
def evaluate_cmd(
    model: Path = typer.Argument(
        MODELS_DIR / "model.lshif",
        help="Path to the trained .lshif model file",
    ),
    data: Path = typer.Argument(
        PROCESSED_DATA_DIR,
        help="Directory containing .parquet embeddings (default: data/processed/)",
    ),
    output: Path = typer.Argument(
        MODELS_DIR / "evaluation.json",
        help="Output path for evaluation metrics",
    ),
    top_k: int = typer.Option(100, "--top-k", "-k", help="Number of top anomalies to export"),
    do_subsampling: bool = typer.Option(
        False, "--do-subsampling", help="Enable bootstrap subsampling stability (expensive)"
    ),
    subsample_splits: int = typer.Option(
        5, "--subsample-splits", help="Number of bootstrap splits for subsampling"
    ),
    n_workers: int = typer.Option(
        None,
        "--n-workers",
        "-w",
        help="Number of parallel workers for seed stability (default: auto)",
    ),
    mlflow_experiment: str = typer.Option(
        None, "--mlflow-experiment", help="MLflow experiment name"
    ),
):
    """Evaluate model stability, score distribution, centroid correlation, and export top anomalies."""
    from patent.modeling.train import evaluate_model

    ctx = {"experiment_name": mlflow_experiment} if mlflow_experiment else None

    output.parent.mkdir(parents=True, exist_ok=True)

    evaluate_model(
        model_path=model,
        embeddings_dir=data,
        output_path=output,
        mlflow_context=ctx,
        top_k=top_k,
        do_subsampling=do_subsampling,
        subsample_splits=subsample_splits,
        n_workers=n_workers,
    )


@model_app.command("register")
def register_cmd(
    run_id: str = typer.Argument(..., help="MLflow run ID from a completed 'model train' run"),
    model_name: str = typer.Option(
        "patent-lshiforest",
        "--model-name",
        "-n",
        help="Registered model name in the MLflow Model Registry",
    ),
    metric_key: str = typer.Option(
        "stability/jaccard_aggregated",
        "--metric-key",
        "-m",
        help="Evaluation metric used to decide if the new model is better",
    ),
    pyfunc_version: int | None = typer.Option(
        None,
        "--pyfunc-version",
        "-v",
        help="Pyfunc model version auto-registered during training",
    ),
):
    """Register a trained model to the MLflow Model Registry.

    Compares the new model's evaluation metrics against the latest
    Production version.  The model is auto-registered during training;
    this step only compares metrics and handles promotion.
    """
    from patent.modeling.registry import register_from_run

    register_from_run(
        run_id=run_id,
        model_name=model_name,
        metric_key=metric_key,
        pyfunc_version=pyfunc_version,
    )


@app.command("pipeline")
def pipeline_cmd(
    raw: Path = typer.Option(
        None, "--raw", "-r", help="Raw XML/JSON file or directory to start from"
    ),
    skip_init: bool = typer.Option(False, "--skip-init", help="Skip data download step"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-run steps even if outputs exist"),
):
    """Run the full pipeline: reserialize → clean → embed → train+evaluate."""
    snapshot_file = RAW_DATA_DIR / "arxiv-metadata-oai-snapshot.json.zst"
    updates_dir = RAW_DATA_DIR / "updates"

    if not skip_init and not snapshot_file.exists():
        logger.info("No snapshot found. Run 'data init' first or use --skip-init.")
        raise typer.Exit(1)

    sources = []
    if raw:
        sources.append((raw, ".json" in raw.suffixes))
    else:
        if snapshot_file.exists():
            sources.append((snapshot_file, True))
        if updates_dir.exists():
            for subdir in sorted(updates_dir.iterdir()):
                if subdir.is_dir():
                    sources.append((subdir, False))

    if not sources:
        logger.error("No raw data sources found.")
        raise typer.Exit(1)

    serialized_dir = INTERIM_DATA_DIR / "serialized"
    cleaned_dir = INTERIM_DATA_DIR / "cleaned"

    for raw_path, is_json in sources:
        # Strip .zst (and underlying .json/.xml) to derive a clean stem for output naming
        stem_name = raw_path.name
        if stem_name.endswith(".zst"):
            stem_name = stem_name[:-4]
        out_name = (
            f"{stem_name}.parquet"
            if raw_path.is_dir()
            else Path(stem_name).with_suffix(".parquet").name
        )
        serialized_path = serialized_dir / out_name
        cleaned_path = cleaned_dir / out_name
        processed_path = PROCESSED_DATA_DIR / out_name

        if force or not serialized_path.exists():
            logger.info(f"--- Reserialize: {raw_path} ---")
            reserialize_data(file_path=raw_path, output_path=serialized_path, is_json=is_json)

        if force or not cleaned_path.exists():
            logger.info(f"--- Clean: {serialized_path} ---")
            clean_data(file_path=serialized_path, output_path=cleaned_path)

        if force or not processed_path.exists():
            logger.info(f"--- Embed: {cleaned_path} ---")
            embed_data(file_path=cleaned_path, output_path=processed_path)

    model_path = MODELS_DIR / "model.lshif"
    if force or not model_path.exists():
        logger.info("--- Train + Evaluate ---")
        train_cmd(data=PROCESSED_DATA_DIR, output=MODELS_DIR)

    logger.success("Pipeline completed.")


@app.command("continuous")
def continuous_cmd(
    trigger: str = typer.Option(
        "weekly",
        "--trigger",
        "-t",
        help="Trigger mode: 'weekly' (scheduled) or 'drift' (alert-driven)",
    ),
    embedder: str = typer.Option(
        "embed-anything-onnx:AllMiniLML6V2Q",
        "--embedder",
        help="Embedder spec: '<protocol>:<model>'",
    ),
    mlflow_experiment: str = typer.Option(
        None, "--mlflow-experiment", help="MLflow experiment name"
    ),
    top_k: int = typer.Option(100, "--top-k", "-k", help="Number of top anomalies to export"),
    n_workers: int = typer.Option(
        None,
        "--n-workers",
        "-w",
        help="Number of parallel workers for scoring (default: auto)",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Simulate the pipeline without side-effects"
    ),
):
    """Continuous training pipeline: fetch → process → train → evaluate → promote.

    Supports two trigger modes:

    * ``--trigger weekly`` — fetches new arXiv data, processes incrementally,
      retrains, and promotes if evaluation metrics improve against Production.

    * ``--trigger drift`` — same workflow, intended to be invoked by a drift
      alert from Prometheus/Alertmanager.

    Run with ``--dry-run`` to see what would happen without any side-effects.
    """
    from patent.pipeline.continuous import TriggerMode, run_continuous_pipeline

    if trigger not in ("weekly", "drift"):
        typer.secho(f"Invalid trigger: {trigger!r}. Use 'weekly' or 'drift'.", fg="red")
        raise typer.Exit(1)

    result = run_continuous_pipeline(
        trigger=cast(TriggerMode, trigger),
        embedder_spec=embedder,
        mlflow_experiment=mlflow_experiment,
        top_k=top_k,
        n_workers=n_workers,
        dry_run=dry_run,
    )

    if not result["trained"] and not dry_run:
        typer.secho("Pipeline did not complete training. Check logs above.", fg="yellow")
        raise typer.Exit(1)

    if result["promoted"]:
        typer.secho(
            f"Model v{result['model_version']} promoted to Production!",
            fg="green",
        )
    elif result["trained"]:
        typer.secho(
            f"Training completed but model was not promoted (run_id={result['run_id']}).",
            fg="yellow",
        )


@app.command("simulate-drift")
def simulate_drift_cmd(
    pvalue: float = typer.Option(0.001, help="Fake KS p-value to push (low = more drift)"),
    ks_statistic: float = typer.Option(0.7, help="Fake KS statistic to push (high = more drift)"),
    mean_shift: float = typer.Option(0.25, help="Fake mean score shift to push"),
):
    """Push artificial drift metrics into Prometheus gauges to test alerting.

    Sets the Prometheus drift gauges to values that would trigger
    ``PatentDriftHighKS``, ``PatentDriftPValueLow``, and ``PatentDriftMeanShift``
    alerts.  Use this to verify that Prometheus → Alertmanager → retrain
    chain works end-to-end.

    After running, check the ``/drift`` endpoint and Grafana dashboards.
    """
    import numpy as np

    from patent.monitoring.metrics import update_drift_metrics

    fake_scores = np.array([0.1, 0.9, 0.5, 0.95, 0.2, 0.8], dtype=np.float32)

    update_drift_metrics(
        ks_statistic=ks_statistic,
        ks_pvalue=pvalue,
        mean_shift=mean_shift,
        emb_shift=1.5,
        n_samples=1000,
        scores=fake_scores,
        model_version="simulated",
        embedding_dim=384,
        total_rows=5000,
    )

    typer.secho(
        f"Fake drift pushed: KS={ks_statistic:.3f}  p={pvalue:.4f}  Δμ={mean_shift:.3f}",
        fg="red",
    )
    typer.secho(
        "Check /drift endpoint and Grafana. Alerts should fire within 5 minutes.",
        fg="yellow",
    )


@app.command("drift-check")
def drift_check_cmd(
    data_dir: Path = typer.Option(
        PROCESSED_DATA_DIR,
        "--data-dir",
        "-d",
        help="Directory containing processed .parquet embeddings",
    ),
    embedder_spec: str = typer.Option(
        "embed-anything-onnx:AllMiniLML6V2Q",
        "--embedder",
        help="Embedder spec: '<protocol>:<model>'",
    ),
    model_path: Path | None = typer.Option(
        None,
        "--model-path",
        "-m",
        help="Path to local .lshif model file (skips MLflow Registry lookup)",
    ),
):
    """Check for data drift by comparing current data against the stored baseline.

    Loads the model from MLflow Registry (or a local .lshif file via
    ``--model-path``), scores a random sample of recent data, and compares
    the anomaly-score distribution against the baseline.  Updates Prometheus
    drift gauges.
    """
    import os
    import shutil
    from tempfile import TemporaryDirectory

    import mlflow
    from mlflow.tracking import MlflowClient
    import numpy as np
    import pyarrow.parquet as pq

    from patent.config import project_tempdir
    from patent.lshiforest import LSHiForest
    from patent.monitoring.drift import compute_drift_metrics, load_drift_baseline
    from patent.monitoring.metrics import update_drift_metrics
    from patent.utils import convert_parquet_to_memmap

    baseline = load_drift_baseline()
    if baseline is None:
        typer.secho(
            "No drift baseline found. Run training first to establish a baseline.",
            fg="yellow",
        )
        raise typer.Exit(1)

    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        typer.secho(f"No .parquet files found in {data_dir}", fg="red")
        raise typer.Exit(1)

    # Sample a subset for efficiency (up to ~50k rows)
    total_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in parquet_files)
    sample_size = min(50_000, total_rows)

    logger.info(f"Total rows: {total_rows:,}, sampling {sample_size:,} for drift check")

    tmpdir = project_tempdir()
    try:
        mmap_path = tmpdir / "drift_sample.mmap"
        embedding_dim, _ = convert_parquet_to_memmap(
            [str(p) for p in parquet_files], str(mmap_path)
        )
        mmap = np.memmap(
            str(mmap_path), dtype=np.float32, mode="r", shape=(total_rows, embedding_dim)
        )

        rng = np.random.default_rng(42)
        indices = rng.choice(total_rows, size=sample_size, replace=False)
        indices = np.sort(indices)
        sample_embeddings = np.array(mmap[indices], dtype=np.float32)

        # Load model (local file or MLflow Registry)
        if model_path is not None:
            if not model_path.exists():
                typer.secho(f"Model file not found: {model_path}", fg="red")
                raise typer.Exit(1)
            model = LSHiForest.load(str(model_path))
            model_version = "local"
            logger.info(f"Loaded local model from {model_path}")
        else:
            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
            mlflow.set_tracking_uri(mlflow_uri)

            client = MlflowClient()
            prod_versions = client.get_latest_versions("patent-lshiforest", stages=["Production"])

            if not prod_versions:
                typer.secho(
                    "No Production model found in MLflow Registry. "
                    "Use --model-path to specify a local model file.",
                    fg="red",
                )
                raise typer.Exit(1)

            prod = prod_versions[0]
            model_version = str(prod.version)
            artifact_uri = f"runs:/{prod.run_id}/model.lshif"

            with TemporaryDirectory() as model_tmp:
                local_path = mlflow.artifacts.download_artifacts(
                    artifact_uri=artifact_uri, dst_path=model_tmp
                )
                model = LSHiForest.load(str(local_path))

        sample_scores = model.score_chunked(mmap, total_rows)

        # Compute drift
        report = compute_drift_metrics(sample_embeddings, sample_scores, baseline=baseline)

        # Update Prometheus metrics
        update_drift_metrics(
            ks_statistic=report.score_ks_statistic,
            ks_pvalue=report.score_ks_pvalue,
            mean_shift=report.score_mean_shift,
            emb_shift=report.embedding_mean_shift,
            n_samples=report.n_new_samples,
            scores=sample_scores,
            model_version=model_version,
            embedding_dim=embedding_dim,
            total_rows=total_rows,
        )

        typer.secho(
            f"Drift check complete: KS={report.score_ks_statistic:.4f} "
            f"(p={report.score_ks_pvalue:.4f}), "
            f"score_Δμ={report.score_mean_shift:.4f}, "
            f"emb_shift={report.embedding_mean_shift:.4f}σ",
            fg="green" if report.score_ks_statistic < 0.1 else "yellow",
        )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    app()
