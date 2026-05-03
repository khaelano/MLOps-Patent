from contextlib import contextmanager
from typing import Any

from loguru import logger
import numpy as np
import pyarrow.parquet as pq

from patent.config import RAW_DATA_DIR

LAST_UPDATE_FILE = RAW_DATA_DIR / "last_update.txt"


@contextmanager
def mute_logging(module_name=""):
    logger.disable(module_name)
    try:
        yield
    finally:
        logger.enable(module_name)


def flatten_dict(d, parent_key="", sep="/") -> dict[str, Any]:
    items = []
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


def get_vectors_from_files(file_paths: list[str], target_dtype=np.float32) -> np.ndarray:
    """
    Load embeddings from Parquet using memory-efficient batching.
    """
    embeddings_blocks = []

    for f in file_paths:
        logger.debug(f"Loading vectors from {f}")
        try:
            parquet_file = pq.ParquetFile(f)

            # Process in batches iteratively to prevent out-of-memory errors
            for batch in parquet_file.iter_batches(batch_size=50000, columns=["embedding"]):
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
