from contextlib import contextmanager
import io
from pathlib import Path
import time
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from patent.config import CHUNK_SIZE, RAW_DATA_DIR


@contextmanager
def open_maybe_zst(path: Path, mode: str = "r", encoding: str = "utf-8"):
    """Open a file for text reading, transparently decompressing ``.zst`` files.

    Yields a text-mode file-like object regardless of whether the underlying
    file is compressed.
    """
    if path.suffix == ".zst":
        import zstandard as zstd

        with open(path, "rb") as fh:
            dctx = zstd.ZstdDecompressor()
            reader = dctx.stream_reader(fh)
            yield io.TextIOWrapper(reader, encoding=encoding)
    else:
        with open(path, mode, encoding=encoding) as fh:
            yield fh


LAST_UPDATE_FILE = RAW_DATA_DIR / "last_update.txt"


@contextmanager
def mute_logging(module_name=""):
    logger.disable(module_name)
    try:
        yield
    finally:
        logger.enable(module_name)


def flatten_dict(d, parent_key="", sep="/") -> dict[str, Any]:
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def format_mb(bytes_size) -> str:
    return f"{byte_to_mbyte(bytes_size):.2f}"


def byte_to_mbyte(bytes_size) -> float:
    return bytes_size / (1024 * 1024)


def get_last_update_date() -> str | None:
    if LAST_UPDATE_FILE.exists():
        with open(LAST_UPDATE_FILE, "r") as f:
            return f.read().strip()
    return None


def set_last_update_date(date_str: str):
    LAST_UPDATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LAST_UPDATE_FILE, "w") as f:
        f.write(date_str)


def convert_parquet_to_memmap(
    pq_paths: list[str] | list[Path],
    output_path: str | Path,
    column: str = "embedding",
) -> tuple[int, int]:
    """Stream parquet embedding columns to a float32 memmap.

    Reads one row group at a time (memory-safe, ~3 GB peak per 1M-row
    row group). Arrow's C++ reader parallelises page decompression
    internally via its global thread pool, giving ~4-8× speedup over
    single-threaded iter_batches.
    """

    embedding_dim = None
    total_rows = 0
    start_total = time.perf_counter()

    for p in pq_paths:
        pf = pq.ParquetFile(p)
        rows = pf.metadata.num_rows
        rgs = pf.metadata.num_row_groups
        logger.info(f"Parquet metadata [cyan]{Path(p).name}[/]: {rows:,} rows, {rgs} row group(s)")
        total_rows += rows
        if embedding_dim is None:
            first_batch = next(pf.iter_batches(batch_size=1, columns=[column]))
            embedding_dim = len(first_batch.column(0)[0])

    assert embedding_dim is not None
    assert total_rows > 0
    total_float32 = total_rows * embedding_dim
    total_mib = byte_to_mbyte(total_float32 * 4)
    logger.info(
        f"Allocating memmap: {total_rows:,} rows × {embedding_dim} = "
        f"{total_float32:,} floats ({total_mib:.1f} MiB) → {output_path}"
    )

    mmap = np.memmap(
        str(output_path),
        dtype=np.float32,
        mode="w+",
        shape=(total_rows, embedding_dim),
    )

    row_offset = 0
    files_processed = 0
    for p in pq_paths:
        pf = pq.ParquetFile(p)
        n_rgs = pf.metadata.num_row_groups
        file_rows = 0
        t_file = time.perf_counter()

        for rg_idx in range(n_rgs):
            t_rg = time.perf_counter()
            table = pf.read_row_group(rg_idx, columns=[column])
            rg_time = time.perf_counter() - t_rg

            col = table.column(0).combine_chunks()
            if hasattr(col, "values"):
                flat = col.values
            else:
                flat_arrs = col.flatten()
                if isinstance(flat_arrs, list):
                    import pyarrow as pa

                    flat = pa.concat_arrays(flat_arrs)
                else:
                    flat = flat_arrs
            arr = flat.to_numpy(zero_copy_only=False)
            if arr.ndim == 1:
                arr = arr.reshape(-1, embedding_dim)
            n = len(arr)
            mmap[row_offset : row_offset + n] = arr

            pct = (row_offset + n) / total_rows * 100
            if n_rgs < 100:
                logger.debug(
                    f"  RG {rg_idx + 1:3d}/{n_rgs}: {n:>7,} rows "
                    f"(decomp {rg_time:.3f}s, cumulative {pct:.1f}%)"
                )
            row_offset += n
            file_rows += n

        t_file_elapsed = time.perf_counter() - t_file
        files_processed += 1
        logger.info(
            f"Parquet done [cyan]{Path(p).name}[/]: "
            f"{file_rows:,} rows in {t_file_elapsed:.2f}s "
            f"({file_rows / t_file_elapsed:,.0f} rows/s) "
            f"[{files_processed}/{len(pq_paths)} files]"
        )

    mmap.flush()
    del mmap

    elapsed = time.perf_counter() - start_total
    rate = total_rows / elapsed if elapsed > 0 else 0
    logger.info(
        f"Parquet → memmap complete in {elapsed:.2f}s ({rate:,.0f} rows/s, {total_mib:.1f} MiB)"
    )
    return embedding_dim, total_rows


def get_vectors_from_files(file_paths: list[str], target_dtype=np.float32) -> np.ndarray:
    """Load embeddings from Parquet using memory-efficient batching."""
    embeddings_blocks = []

    for f in file_paths:
        logger.debug(f"Loading vectors from {f}")
        try:
            parquet_file = pq.ParquetFile(f)

            # Process in batches iteratively to prevent out-of-memory errors
            for batch in parquet_file.iter_batches(batch_size=CHUNK_SIZE, columns=["embedding"]):
                col = batch.column("embedding")

                # Determine dimensionality
                dim = (
                    col.type.list_size
                    if hasattr(col.type, "list_size")
                    else len(col.flatten()) // len(col)
                )

                # Flatten batch array, cast to NumPy, and reshape
                emb_np = col.flatten().to_numpy(zero_copy_only=False).reshape(-1, dim)

                if emb_np.dtype != target_dtype:
                    emb_np = emb_np.astype(target_dtype)

                embeddings_blocks.append(emb_np)

            logger.debug(f"Successfully processed {f}")

        except Exception as e:
            logger.error(f"Failed to process {f}: {e}")

    if not embeddings_blocks:
        logger.warning("No embeddings found")
        return np.array([])

    result = np.vstack(embeddings_blocks)
    logger.info(f"Loaded {result.shape[0]} embeddings of dimension {result.shape[1]}")
    return result


def load_parquet_metadata(file_paths: list[str]) -> pd.DataFrame:
    metadata_columns = ["id", "title", "categories", "update_date"]
    blocks = []

    for f in file_paths:
        logger.debug(f"Loading metadata from {f}")
        try:
            parquet_file = pq.ParquetFile(f)
            available = [c for c in metadata_columns if c in parquet_file.schema.names]
            if not available:
                logger.warning(f"No metadata columns found in {f}")
                continue

            for batch in parquet_file.iter_batches(batch_size=CHUNK_SIZE, columns=available):
                blocks.append(batch.to_pandas())

        except Exception as e:
            logger.error(f"Failed to load metadata from {f}: {e}")

    if not blocks:
        logger.warning("No metadata found")
        return pd.DataFrame()

    result = pd.concat(blocks, ignore_index=True)
    logger.info(f"Loaded metadata for {len(result)} rows")
    return result
