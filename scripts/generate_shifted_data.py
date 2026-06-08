#!/usr/bin/env python3
"""Generator data shifted untuk simulasi Data Drift."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger


def _load_embeddings(parquet_path: Path, max_samples: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Load embeddings and metadata from a processed parquet file."""
    pf = pq.ParquetFile(parquet_path)
    first_batch = next(pf.iter_batches(batch_size=1, columns=["embedding"]))
    embedding_dim = len(first_batch.column(0)[0].as_py())

    total_rows = pf.metadata.num_rows
    n_samples = min(max_samples, total_rows)

    # Read metadata
    meta_cols = [c for c in ["id", "title", "categories", "update_date"] if c in pf.schema.names]
    meta_batches = []
    emb_arrays = []

    rows_read = 0
    for batch in pf.iter_batches(batch_size=min(n_samples, 10_000)):
        if rows_read >= n_samples:
            break
        n = min(len(batch), n_samples - rows_read)
        if n < len(batch):
            batch = batch.slice(0, n)

        # Extract embeddings
        emb_col = batch.column("embedding")
        # RecordBatch.column() returns Array (may be ListArray or ChunkedArray)
        if hasattr(emb_col, "combine_chunks"):
            emb_col = emb_col.combine_chunks()
        # ListArray.flatten() gives the flat values array
        if hasattr(emb_col, "flatten"):
            flat_values = emb_col.flatten().to_numpy(zero_copy_only=False)
        else:
            flat_values = emb_col.values.to_numpy(zero_copy_only=False)
        emb_flat = flat_values.reshape(-1, embedding_dim)
        emb_arrays.append(emb_flat)

        # Extract metadata
        if meta_cols:
            meta_batches.append(batch.select(meta_cols).to_pandas())

        rows_read += n

    embeddings = np.vstack(emb_arrays) if emb_arrays else np.empty((0, embedding_dim))
    metadata = pd.concat(meta_batches, ignore_index=True) if meta_batches else pd.DataFrame()

    logger.info(f"Loaded {len(embeddings)} embeddings (dim={embedding_dim})")
    return embeddings, metadata


def apply_mean_shift(embeddings: np.ndarray, strength: float, seed: int = 42) -> np.ndarray:
    """Shift embeddings by a random direction proportional to *strength*."""
    rng = np.random.default_rng(seed)
    n, d = embeddings.shape

    # Compute mean of original embeddings
    original_mean = embeddings.mean(axis=0)

    # Generate a random direction and apply shift
    direction = rng.normal(0, 1, size=d)
    direction /= np.linalg.norm(direction)

    shift_vector = direction * strength * np.linalg.norm(original_mean)
    shifted = embeddings + shift_vector

    logger.info(
        f"Mean shift applied: strength={strength}, "
        f"|shift|={np.linalg.norm(shift_vector):.4f}, "
        f"|original_mean|={np.linalg.norm(original_mean):.4f}"
    )
    return shifted.astype(np.float32)


def apply_noise_injection(
    embeddings: np.ndarray, strength: float, seed: int = 42
) -> np.ndarray:
    """Add Gaussian noise scaled by the embedding norm and *strength*."""
    rng = np.random.default_rng(seed)
    n, d = embeddings.shape

    # Scale noise relative to each vector's norm
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    noise = rng.normal(0, strength, size=(n, d))
    noise = noise * np.mean(norms)

    shifted = embeddings + noise

    snr = np.mean(norms) / (strength * np.mean(norms) + 1e-10)
    logger.info(f"Noise injected: strength={strength}, SNR≈{snr:.2f}")
    return shifted.astype(np.float32)


def apply_distribution_shift(
    embeddings: np.ndarray, strength: float, seed: int = 42
) -> np.ndarray:
    """Apply non-linear transformation to shift the distribution."""
    rng = np.random.default_rng(seed)
    n, d = embeddings.shape

    # Select a random subset of dimensions to transform
    n_dims_to_shift = max(1, int(d * 0.3))
    dims = rng.choice(d, size=n_dims_to_shift, replace=False)

    shifted = embeddings.copy()
    for dim in dims:
        col = shifted[:, dim]
        # Apply tanh-based non-linear transformation
        shifted[:, dim] = np.tanh(col * (1 + strength)) * np.std(col) + np.mean(col)

    logger.info(f"Distribution shift applied: {n_dims_to_shift}/{d} dims, strength={strength}")
    return shifted.astype(np.float32)


def _save_shifted_parquet(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save shifted embeddings + metadata as a Parquet file."""
    n = len(embeddings)

    # Ensure metadata has same number of rows
    if len(metadata) < n:
        # Pad with synthetic records
        extra = pd.DataFrame({
            "id": [f"shifted_{i}" for i in range(len(metadata), n)],
            "title": [f"Shifted Paper {i}" for i in range(len(metadata), n)],
            "categories": ["cs.AI"] * (n - len(metadata)),
            "update_date": ["2026-06-08"] * (n - len(metadata)),
        })
        metadata = pd.concat([metadata, extra], ignore_index=True)
    elif len(metadata) > n:
        metadata = metadata.iloc[:n]

    # Build PyArrow table
    emb_list = [embeddings[i].tolist() for i in range(n)]
    table = pa.table({
        **{col: metadata[col].values for col in metadata.columns},
        "embedding": pa.array(emb_list, type=pa.list_(pa.float32())),
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)
    logger.success(f"Shifted data saved to {output_path} ({n} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate shifted data for CT simulation"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to input processed Parquet file",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Path for shifted output Parquet",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["mean_shift", "noise_injection", "distribution_shift"],
        default="mean_shift",
        help="Type of shift to apply (default: mean_shift)",
    )
    parser.add_argument(
        "--shift-strength", "-s",
        type=float,
        default=0.5,
        help="Shift strength (0.1=subtle, 1.0=strong, default: 0.5)",
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=500,
        help="Number of samples to generate (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    embeddings, metadata = _load_embeddings(args.input, args.num_samples)
    if len(embeddings) == 0:
        logger.error("No embeddings loaded")
        sys.exit(1)

    shift_fns = {
        "mean_shift": apply_mean_shift,
        "noise_injection": apply_noise_injection,
        "distribution_shift": apply_distribution_shift,
    }

    logger.info(f"Applying {args.mode} (strength={args.shift_strength})...")
    shifted_embeddings = shift_fns[args.mode](embeddings, args.shift_strength, args.seed)

    diff = shifted_embeddings - embeddings
    l2_diff = np.linalg.norm(diff, axis=1)
    cosine_sim = np.sum(embeddings * shifted_embeddings, axis=1) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(shifted_embeddings, axis=1) + 1e-10
    )
    logger.info(
        f"Shift statistics: "
        f"|Δ|_mean={l2_diff.mean():.4f}, "
        f"cos_sim_mean={cosine_sim.mean():.4f}, "
        f"cos_sim_min={cosine_sim.min():.4f}"
    )

    _save_shifted_parquet(shifted_embeddings, metadata, args.output)


if __name__ == "__main__":
    main()
