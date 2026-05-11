from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from patent.config import INTERIM_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from patent.dataset.ingest import (
    download_kaggle_snapshot,
    extract_latest_update,
    fetch_oai_updates,
)
from patent.dataset.preprocess import (
    clean_df,
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
        RAW_DATA_DIR / "arxiv-metadata-oai-snapshot.json",
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
        out_name = (
            f"{file_path.name}.parquet"
            if file_path.is_dir()
            else file_path.with_suffix(".parquet").name
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
    model_name: str = typer.Option("all-MiniLM-L6-v2", help="SentenceTransformer model name"),
    batch_size: int = typer.Option(50000, help="Row count per chunk to process sequentially"),
):
    """
    Read a cleaned Parquet artifact sequentially in chunks using PyArrow to minimize memory footprint.
    Generates text embeddings for titles using a SentenceTransformer model (with multiprocessing support),
    and streams the resulting features iteratively into a new Parquet file.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from patent.dataset.preprocess import embed

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise typer.Exit(1)

    if not output_path:
        output_path = PROCESSED_DATA_DIR / file_path.name

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing '{file_path}' in chunks of {batch_size} rows...")
    parquet_file = pq.ParquetFile(file_path)
    writer = None

    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading SentenceTransformer model ('{model_name}')...")
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer model: {e}")
        raise typer.Exit(1)

    pool = model.start_multi_process_pool()

    try:
        for i, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
            logger.info(f"Processing chunk {i + 1}...")
            df_chunk = batch.to_pandas()

            # Pass the loaded model and pool to embed
            df_embedded = embed(df_chunk, model, pool=pool)

            table = pa.Table.from_pandas(df_embedded)

            # Initialize the writer with the schema from the first processed chunk
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)

            writer.write_table(table)
    finally:
        model.stop_multi_process_pool(pool)

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
    num_trees: int = typer.Option(50, "--num-trees", "-t", help="Number of isolation trees"),
    max_depth: int = typer.Option(16, "--max-depth", "-m", help="Maximum tree depth"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    params: Path = typer.Option(
        None, "--params", "-p", help="JSON file with additional model params"
    ),
    mlflow_experiment: str = typer.Option(
        None, "--mlflow-experiment", help="MLflow experiment name"
    ),
):
    """Train an LSHiForest model on processed embeddings."""
    import json

    from patent.modeling.train import train_model

    model_cfg = {"num_trees": num_trees, "max_depth": max_depth, "seed": seed}
    if params and params.exists():
        with open(params, "r") as f:
            model_cfg.update(json.load(f))

    ctx = {"experiment_name": mlflow_experiment} if mlflow_experiment else None

    output.mkdir(parents=True, exist_ok=True)

    train_model(
        embeddings_dir=data,
        output_dir=output,
        model_params=model_cfg,
        mlflow_context=ctx,
    )


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


@app.command("pipeline")
def pipeline_cmd(
    raw: Path = typer.Option(
        None, "--raw", "-r", help="Raw XML/JSON file or directory to start from"
    ),
    skip_init: bool = typer.Option(False, "--skip-init", help="Skip data download step"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-run steps even if outputs exist"),
):
    """Run the full pipeline: reserialize → clean → embed → train → evaluate."""
    snapshot_file = RAW_DATA_DIR / "arxiv-metadata-oai-snapshot.json"
    updates_dir = RAW_DATA_DIR / "updates"

    if not skip_init and not snapshot_file.exists():
        logger.info("No snapshot found. Run 'data init' first or use --skip-init.")
        raise typer.Exit(1)

    sources = []
    if raw:
        sources.append((raw, raw.suffix == ".json"))
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
        out_name = (
            f"{raw_path.name}.parquet"
            if raw_path.is_dir()
            else raw_path.with_suffix(".parquet").name
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
        logger.info("--- Train ---")
        train_cmd(data=PROCESSED_DATA_DIR, output=MODELS_DIR)

    eval_path = MODELS_DIR / "evaluation.json"
    if force or not eval_path.exists():
        logger.info("--- Evaluate ---")
        evaluate_cmd(model=model_path, data=PROCESSED_DATA_DIR, output=eval_path)

    logger.success("Pipeline completed.")


if __name__ == "__main__":
    app()
