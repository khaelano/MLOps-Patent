from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from patent.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from patent.dataset.ingest import (
    download_kaggle_snapshot,
    extract_latest_update,
    fetch_oai_updates,
)
from patent.dataset.preprocess import clean_df, parse_oai_xml_file, parse_snapshot_json_file
from patent.utils import get_last_update_date, set_last_update_date

app = typer.Typer(help="MLOps Patent Pipeline CLI", no_args_is_help=True)

data_app = typer.Typer(help="Data ingestion and preprocessing commands", no_args_is_help=True)
app.add_typer(data_app, name="data")


@data_app.command("init")
def init_data(
    output_path: Path = typer.Option(
        RAW_DATA_DIR / "arxiv-metadata-oai-snapshot.json", "--output-path", "-o", help="File path for metadata storage"
    ),
):
    """Bootstrap the dataset by downloading the Kaggle snapshot and partitioning it."""

    snapshot_file = download_kaggle_snapshot(output_path)
    last_update_date = extract_latest_update(snapshot_file)
    set_last_update_date(last_update_date)


@data_app.command("update")
def update_data(
    output_path: Path = typer.Option(RAW_DATA_DIR / "updates", "--output-path", "-o", help="Updates directory"),
    from_date: str = typer.Option(
        None, help="Start date (YYYY-MM-DD). Defaults to the day after the last update date in metadata."
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
        from_date = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

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
        df = parse_snapshot_json_file(file_path)
    else:
        if file_path.is_dir():
            dfs = []
            for xml_file in file_path.glob("**/*.xml"):
                dfs.append(parse_oai_xml_file(xml_file))
            df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        else:
            df = parse_oai_xml_file(file_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
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


if __name__ == "__main__":
    app()
