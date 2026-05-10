from contextlib import contextmanager
import io
import json
from pathlib import Path
import re
import time
import tracemalloc
import xml.etree.ElementTree as ET

from loguru import logger
import pyarrow as pa
import pyarrow.parquet as pq

_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("title", pa.string()),
        pa.field("categories", pa.string()),
        pa.field("update_date", pa.string()),
    ]
)


@contextmanager
def _benchmark(name: str):
    tracemalloc.start()
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logger.info(f"[Benchmark] {name}: {elapsed:.2f}s, peak memory {peak / 1024**2:.2f} MB")


def _iter_json_records(file_path: Path):
    """Yield dict records from a newline-delimited JSON file (Kaggle snapshot)."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                yield {
                    "id": data.get("id"),
                    "title": data.get("title"),
                    "categories": data.get("categories"),
                    "update_date": data.get("update_date"),
                }
            except json.JSONDecodeError:
                continue


def _iter_xml_records(file_path: Path):
    """Yield dict records from a raw OAI-PMH XML file using a streaming parser."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    content = re.sub(r"<\?xml.*?\?>", "", content)
    wrapped = f"<root>{content}</root>"
    buf = io.BytesIO(wrapped.encode("utf-8"))

    for _event, elem in ET.iterparse(buf, events=("end",)):
        tag_name = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

        if tag_name != "record":
            continue

        id_txt = title_txt = cats_txt = date_txt = ""

        for child in elem.iter():
            ctag = child.tag.split("}")[-1]

            if ctag == "identifier" and not id_txt:
                raw_id = child.text or ""
                id_txt = raw_id.split(":")[-1] if ":" in raw_id else raw_id
            elif ctag == "id" and not id_txt:
                raw_id = child.text or ""
                id_txt = raw_id.split(":")[-1] if ":" in raw_id else raw_id
            elif ctag == "title" and not title_txt:
                title_txt = child.text or ""
            elif ctag == "categories" and not cats_txt:
                cats_txt = child.text or ""
            elif ctag == "datestamp" and not date_txt:
                date_txt = child.text or ""
            elif ctag == "updated" and not date_txt:
                date_txt = child.text or ""

        if id_txt and title_txt:
            yield {
                "id": id_txt,
                "title": title_txt,
                "categories": cats_txt,
                "update_date": date_txt,
            }

        elem.clear()


def _write_records(records_iter, output_path: Path, batch_size: int = 100_000) -> int:
    """Stream record dicts into a Parquet file in batches.

    Returns the total number of records written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    count = 0
    batch = []

    for record in records_iter:
        batch.append(record)
        count += 1
        if len(batch) >= batch_size:
            table = pa.Table.from_pylist(batch, schema=_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(str(output_path), _SCHEMA)
            writer.write_table(table)
            batch.clear()

    if batch:
        table = pa.Table.from_pylist(batch, schema=_SCHEMA)
        if writer is None:
            writer = pq.ParquetWriter(str(output_path), _SCHEMA)
        writer.write_table(table)

    if writer is None:
        table = pa.Table.from_pylist([], schema=_SCHEMA)
        writer = pq.ParquetWriter(str(output_path), _SCHEMA)
        writer.write_table(table)

    writer.close()
    return count


def parse_snapshot_json_file(file_path: Path, output_path: Path, batch_size: int = 100_000):
    """Stream-parse a newline-delimited JSON snapshot into a Parquet file."""
    logger.info(f"Parsing JSON metadata from {file_path}")
    with _benchmark("parse_snapshot_json_file"):
        count = _write_records(_iter_json_records(file_path), output_path, batch_size=batch_size)
    logger.info(f"Successfully serialized {count} records to {output_path}")


def parse_oai_xml_file(file_path: Path, output_path: Path, batch_size: int = 100_000):
    """Stream-parse an OAI-PMH XML file into a Parquet file."""
    logger.info(f"Parsing OAI-PMH XML from {file_path}")
    with _benchmark("parse_oai_xml_file"):
        count = _write_records(_iter_xml_records(file_path), output_path, batch_size=batch_size)
    logger.info(f"Successfully serialized {count} records to {output_path}")


def parse_oai_xml_directory(dir_path: Path, output_path: Path, batch_size: int = 100_000):
    """Stream-parse all XML files in a directory into a single Parquet file."""
    xml_files = sorted(dir_path.glob("**/*.xml"))
    logger.info(f"Parsing {len(xml_files)} OAI-PMH XML files from {dir_path}")

    def _iter_all():
        for xml_file in xml_files:
            yield from _iter_xml_records(xml_file)

    with _benchmark("parse_oai_xml_directory"):
        count = _write_records(_iter_all(), output_path, batch_size=batch_size)
    logger.info(
        f"Successfully serialized {count} records from {len(xml_files)} files to {output_path}"
    )


def clean_df(df):
    """Apply text cleaning to a DataFrame, drop missing, and return cleaned payload."""

    def clean_text(text: str) -> str:
        """Applies LaTeX stripping, whitespace removal, and lowercasing."""
        if not isinstance(text, str):
            return ""
        text = re.sub(r"\$.*?\$", "", text)
        text = text.replace("\n", " ").replace("\t", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower()

    initial_count = len(df)
    logger.info(f"Starting text cleaning for {initial_count} rows...")

    with _benchmark("clean_df"):
        df["title"] = df["title"].apply(clean_text)
        df = df[df["title"].str.strip() != ""]
        df = df.dropna(subset=["id", "title"])

    logger.info(f"Dropped {initial_count - len(df)} invalid/empty rows. Remaining: {len(df)}.")

    return df


def embed(df, model, pool=None):
    """Generate SentenceTransformers embeddings for titles and return processed DataFrame."""

    logger.info(f"Encoding {len(df)} titles...")
    with _benchmark("embed"):
        if pool is not None:
            embeddings = model.encode(df["title"].tolist(), pool=pool, show_progress_bar=True)
        else:
            embeddings = model.encode(df["title"].tolist(), show_progress_bar=True)

    logger.info("Successfully generated embeddings. Assigning to dataframe...")
    df["embedding"] = embeddings.tolist()

    return df
