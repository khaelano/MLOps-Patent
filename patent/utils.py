import gc

from loguru import logger
import numpy as np
import pyarrow.parquet as pq

from patent.config import RAW_DATA_DIR

LAST_UPDATE_FILE = RAW_DATA_DIR / "last_update.txt"


def flatten_dict(d, parent_key="", sep="/"):
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


def parquet_to_memmap(
    pq_path: str,
    mmap_path: str,
    column: str = "embedding",
    chunk_size: int = 200_000,
) -> tuple[int, str]:
    """Utility to convert a column in a parquet file into a memory-mapped numpy array."""
    pq_file = pq.ParquetFile(pq_path)
    num_rows = pq_file.metadata.num_rows

    try:
        first_batch = next(pq_file.iter_batches(batch_size=1, columns=[column]))
        first_embedding = first_batch.column(0).to_pylist()[0]

        if isinstance(first_embedding, (list, tuple)):
            embedding_dim = len(first_embedding)
        else:
            raise ValueError(f"Expected list/array for embedding, got {type(first_embedding)}")

    except StopIteration:
        raise ValueError("Parquet file is empty.")
    except Exception as e:
        raise ValueError(f"Failed to infer embedding dimension: {e}")

    logger.info(f"Inferred embedding dimension: {embedding_dim} from {pq_path}")

    mmap_file = np.memmap(mmap_path, dtype="float32", mode="w+", shape=(num_rows, embedding_dim))

    curr_idx = 0
    for batch in pq_file.iter_batches(batch_size=chunk_size, columns=[column]):
        embeddings_list = batch.column(0).to_pylist()

        arr = np.array(embeddings_list, dtype=np.float32)

        if arr.shape[1] != embedding_dim:
            raise ValueError(
                f"Dimension mismatch at row {curr_idx}. "
                f"Expected {embedding_dim}, got {arr.shape[1]}"
            )

        slice_end = curr_idx + len(arr)

        if slice_end > num_rows:
            logger.warning(f"Batch exceeds expected row count. Truncating at {num_rows}.")
            slice_end = num_rows
            arr = arr[: slice_end - curr_idx]

        mmap_file[curr_idx:slice_end] = arr
        mmap_file.flush()

        curr_idx = slice_end

        del arr, embeddings_list
        gc.collect()

    if curr_idx != num_rows:
        logger.warning(
            f"Processed {curr_idx} rows, but parquet metadata says {num_rows}. "
            "Check for nulls or corrupted rows."
        )

    return (embedding_dim, mmap_path)
