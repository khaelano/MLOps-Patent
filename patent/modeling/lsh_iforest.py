from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import tracemalloc
from typing import Any

from loguru import logger
import mlflow
from mlflow.models import infer_signature
import numpy as np
import pyarrow.parquet as pq
import zstandard

from patent.utils import byte_to_mbyte


class LSHIFWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        from patent.modeling.lsh_iforest import LSHIForest

        model_path: str = context.artifacts["lshif_file"]
        logger.info(f"Loading LSHiForest model from artifact: {model_path}")
        self.model: LSHIForest = LSHIForest.load_model(model_path)
        logger.success("Model successfully loaded into MLflow context.")

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: np.ndarray,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        logger.debug(f"Predict called with input shape: {model_input.shape}")
        return self.model.score(model_input)


@dataclass
class LSHIFMeta:
    embedding_dim: int | None
    num_rows: int | None
    num_trees: int = 50
    max_depth: int = 16
    seed: int = 42
    c_n: float | None = None
    is_sorted: bool = False
    hash_bits: int = 64
    bucket_bits: int = 4
    format_version: int = 1


class LSHIForest:
    HEADER_SIZE: int = 1024

    def __init__(
        self,
        num_trees: int = 50,
        max_depth: int = 16,
        chunk_size: int = 100_000,
        seed: int = 42,
    ) -> None:
        if max_depth > 16:
            err_msg = "max_depth cannot exceed 16 when using 32-bit hashes with 2-bit buckets."
            logger.error(err_msg)
            raise ValueError(err_msg)

        self.chunk_size: int = chunk_size
        self.meta: LSHIFMeta = LSHIFMeta(
            embedding_dim=None,
            num_rows=None,
            num_trees=num_trees,
            max_depth=max_depth,
            seed=seed,
            hash_bits=32,
            bucket_bits=2,
            format_version=2,
        )

        self.model_path: Path = Path(tempfile.mkdtemp()) / "model.lshif"
        self.forest_mmap: np.memmap | None = None
        self.projections: list[np.ndarray] | None = None
        self._tempfile: Path | None = None

        logger.debug(
            f"Initialized LSHIForest: num_trees={num_trees}, "
            f"max_depth={max_depth}, chunk_size={chunk_size}, seed={seed}"
        )

    def __del__(self) -> None:
        if self._tempfile and os.path.exists(self._tempfile):
            os.unlink(self._tempfile)

    @property
    def _hash_dtype(self) -> type:
        return np.uint32 if self.meta.hash_bits == 32 else np.uint64

    def _dump_meta(self, output_path: Path | str) -> None:
        meta_str = json.dumps(asdict(self._loaded_meta()))
        if len(meta_str) > self.HEADER_SIZE:
            err_msg = "Metadata too large for header"
            logger.error(err_msg)
            raise ValueError(err_msg)

        header = meta_str.ljust(self.HEADER_SIZE).encode("utf-8")
        with open(output_path, "r+b") as f:
            f.write(header)
        logger.debug(f"Metadata dumped to header at {output_path}")

    def _loaded_meta(self) -> LSHIFMeta:
        if self.meta.embedding_dim is None or self.meta.num_rows is None:
            err_msg = "Model not loaded or built"
            logger.error(err_msg)
            raise RuntimeError(err_msg)
        return self.meta

    def _get_hyperplanes(self, tree_idx: int) -> np.ndarray:
        rng = np.random.default_rng(self.meta.seed + tree_idx)
        dim = self._loaded_meta().embedding_dim
        assert dim is not None
        return rng.standard_normal((dim, self.meta.hash_bits)).astype("float32")

    def _compute_simhash(self, vectors: np.ndarray, projection_matrix: np.ndarray) -> np.ndarray:
        projected = np.dot(vectors, projection_matrix)
        return np.packbits(projected > 0, axis=1).view(self._hash_dtype).reshape(-1)

    def _generate_signatures_for_tree(
        self, embeddings_mmap: np.memmap, projection_matrix: np.ndarray, chunk_size: int
    ) -> np.ndarray:
        meta = self._loaded_meta()
        assert meta.num_rows is not None

        signatures = np.zeros(meta.num_rows, dtype=self._hash_dtype)
        curr_idx = 0

        while curr_idx < meta.num_rows:
            slice_end = min(curr_idx + chunk_size, meta.num_rows)
            chunk = embeddings_mmap[curr_idx:slice_end]
            signatures[curr_idx:slice_end] = self._compute_simhash(chunk, projection_matrix)
            curr_idx = slice_end

        return signatures

    def _build_single_tree_sorted(
        self, sorted_sigs: np.ndarray, sorted_idx: np.ndarray
    ) -> np.ndarray:
        """
        Compute isolation path lengths on already-sorted signatures.

        Operates in O(n) per depth via linear neighbour comparison instead of
        O(n log n) np.unique calls.  Results are mapped back to original order
        via *sorted_idx*.
        """
        meta = self._loaded_meta()
        n = len(sorted_sigs)
        isolated_depth = np.full(n, meta.max_depth, dtype=np.float32)

        for depth in range(1, meta.max_depth):
            still_active = isolated_depth == meta.max_depth
            if not np.any(still_active):
                break

            shift = self._hash_dtype(meta.hash_bits - (depth * meta.bucket_bits))
            prefixes = sorted_sigs >> shift

            neighbor_eq = prefixes[:-1] == prefixes[1:]

            isolation = np.ones(n, dtype=bool)
            isolation[1:] &= ~neighbor_eq
            isolation[:-1] &= ~neighbor_eq

            newly_isolated = isolation & still_active
            isolated_depth[newly_isolated] = float(depth)

        path_lengths = np.empty(n, dtype=np.float32)
        path_lengths[sorted_idx] = isolated_depth
        return path_lengths

    def _calc_mmap_row(
        self,
        embeddings_path: str | Path,
        embedding_dim: int,
        dtype_size: int = 4,
    ) -> int:
        embeddings_path = Path(embeddings_path)

        if not embeddings_path.exists():
            err_msg = f"File not found: {embeddings_path}"
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        file_size_bytes = os.path.getsize(embeddings_path)
        bytes_per_row = embedding_dim * dtype_size

        if bytes_per_row == 0:
            err_msg = "bytes_per_row cannot be zero (check embedding_dim)"
            logger.error(err_msg)
            raise ValueError(err_msg)

        if file_size_bytes % bytes_per_row != 0:
            err_msg = (
                f"File size ({file_size_bytes} bytes) doesn't perfectly divide "
                f"by {bytes_per_row}. Is {embeddings_path} a valid float32 mmap file?"
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

        inferred_rows = file_size_bytes // bytes_per_row
        logger.debug(f"Inferred {inferred_rows} rows from file size {file_size_bytes} bytes.")
        return inferred_rows

    def _c_factor(self, n: int) -> float:
        if n <= 1:
            return 0.0
        return float(2 * (np.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n))

    def _calculate_baseline(
        self,
        baseline_output_path: str | Path | None = None,
    ) -> None:
        """
        Pass 2: Calculates path depths by reading one tree at a time from the built forest.
        This allows baseline calculation after a Row-First stream like build_from_pq.
        """
        logger.info("Starting Pass 2: Baseline depth calculation...")
        calc_start_time = time.perf_counter()

        meta = self._loaded_meta()
        assert meta.num_rows is not None
        assert self.forest_mmap is not None

        signatures_mmap = self.forest_mmap

        total_depths = np.zeros(meta.num_rows, dtype=np.float32)

        tracemalloc.start()
        for tree_idx in range(meta.num_trees):
            logger.debug(f"Calculating depths for tree {tree_idx + 1}/{meta.num_trees}...")

            tree_sigs = signatures_mmap[tree_idx, :].copy()
            sorted_idx = np.argsort(tree_sigs)
            sorted_sigs = tree_sigs[sorted_idx]
            del tree_sigs

            path_lengths = self._build_single_tree_sorted(sorted_sigs, sorted_idx)
            total_depths += path_lengths

            signatures_mmap[tree_idx, :] = sorted_sigs
            signatures_mmap.flush()
            del sorted_sigs, sorted_idx, path_lengths

        tracemalloc.stop()

        baseline_depths = total_depths / meta.num_trees
        c_n = self._c_factor(meta.num_rows)
        self.meta.c_n = c_n
        self.meta.is_sorted = True

        self._dump_meta(self.model_path)

        if baseline_output_path is not None:
            logger.debug(f"Dumping baseline scores at {baseline_output_path}.")
            np.save(baseline_output_path, baseline_depths)

        build_time = time.perf_counter() - calc_start_time
        logger.success(f"Baseline calculation complete in {build_time:.2f}s. c_n={c_n:.4f}")

        if mlflow.active_run():
            mlflow.log_metric("baseline_calc_time_s", build_time)
            mlflow.log_metric("c_n", c_n)
            mlflow.log_artifact(str(baseline_output_path), artifact_path="baselines")

    @classmethod
    def load_model(cls, model_path_str: str | Path = "output.lshif") -> LSHIForest:
        model_path = Path(model_path_str)
        logger.info(f"Loading LSHiForest model from {model_path}...")

        with open(model_path, "rb") as f:
            header_bytes = f.read(cls.HEADER_SIZE)

        meta_dict = json.loads(header_bytes.decode("utf-8").strip())
        meta = LSHIFMeta(**meta_dict)

        instance = cls()
        instance.meta = meta

        loaded_meta = instance._loaded_meta()
        assert loaded_meta.num_rows is not None
        assert loaded_meta.num_trees is not None

        if meta.format_version == 2:
            with open(model_path, "rb") as f:
                f.seek(cls.HEADER_SIZE)
                comp_size = np.frombuffer(f.read(4), dtype=np.uint32)[0]
                compressed = f.read(comp_size)

            raw = zstandard.ZstdDecompressor().decompress(compressed)
            flat = np.frombuffer(raw, dtype=np.uint32)
            forest = cls._delta_decode(flat, loaded_meta.num_trees, loaded_meta.num_rows)

            v2_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".lshif")
            v2_temp_path = Path(v2_temp.name)
            v2_temp.close()

            mmap_file = np.memmap(
                str(v2_temp_path),
                dtype=np.uint32,
                mode="w+",
                shape=(loaded_meta.num_trees, loaded_meta.num_rows),
            )
            mmap_file[:] = forest
            mmap_file.flush()
            instance.forest_mmap = mmap_file
            instance._tempfile = v2_temp_path
            logger.info(f"Decompressed model to temp file: {v2_temp_path}")
        else:
            instance.forest_mmap = np.memmap(
                str(model_path),
                dtype=instance._hash_dtype,
                mode="r",
                offset=cls.HEADER_SIZE,
                shape=(loaded_meta.num_trees, loaded_meta.num_rows),
            )

        instance.projections = [instance._get_hyperplanes(i) for i in range(loaded_meta.num_trees)]

        logger.success(
            f"Model loaded. Trees: {loaded_meta.num_trees}, Rows: {loaded_meta.num_rows}"
        )
        return instance

    def _delta_encode(self) -> np.ndarray:
        assert self.meta.hash_bits == 32
        assert self.forest_mmap is not None
        first_vals = self.forest_mmap[:, 0].copy()
        deltas = np.diff(self.forest_mmap, axis=1).ravel()
        return np.concatenate([first_vals, deltas]).astype(np.uint32, copy=False)

    @staticmethod
    def _delta_decode(flat: np.ndarray, num_trees: int, num_rows: int) -> np.ndarray:
        first_vals = flat[:num_trees]
        deltas = flat[num_trees:].reshape(num_trees, num_rows - 1)
        forest = np.empty((num_trees, num_rows), dtype=np.uint32)
        forest[:, 0] = first_vals
        forest[:, 1:] = first_vals[:, None] + np.cumsum(deltas, axis=1)
        return forest

    def save_model(
        self, output_path_str: str | Path = "output.lshif", compress: bool = True
    ) -> None:
        output_path = Path(output_path_str)
        logger.info(f"Saving LSHiForest model to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        meta = self._loaded_meta()

        if (
            compress
            and meta.is_sorted
            and meta.hash_bits == 32
            and meta.num_rows
            and meta.num_rows > 0
        ):
            meta.format_version = 2
            flat = self._delta_encode()
            compressed = zstandard.ZstdCompressor(level=3).compress(flat.tobytes())
            with open(output_path, "wb") as f:
                header = json.dumps(asdict(meta)).ljust(self.HEADER_SIZE).encode("utf-8")
                f.write(header)
                f.write(np.uint32(len(compressed)).tobytes())
                f.write(compressed)
            logger.info(f"Compressed forest: {len(compressed)} bytes (delta+zstd)")
        else:
            meta.format_version = 1
            if output_path.resolve() != self.model_path.resolve():
                shutil.copy2(self.model_path, output_path)
                with open(output_path, "r+b") as f:
                    header = json.dumps(asdict(meta)).ljust(self.HEADER_SIZE).encode("utf-8")
                    f.write(header)

        if mlflow.active_run():
            logger.debug("Logging model to MLflow...")
            input_example = np.random.randn(30, 384).astype("float32")
            output_example = self.score(input_example)
            mlflow.pyfunc.log_model(
                name="lshiforest",
                python_model=LSHIFWrapper(),
                artifacts={"lshif_file": str(output_path)},
                input_example=input_example,
                signature=infer_signature(input_example, output_example),
            )

        logger.success("Model saved successfully.")

    def _build_forest(
        self,
        pq_paths: list[str] | list[Path],
        column: str = "embedding",
        chunk_size: int = 100_000,
    ):
        """Streams parquet files directly into SimHash signatures without intermediate float storage."""
        logger.info("Starting forest build from Parquet streams...")
        forest_build_start_time = time.perf_counter()

        if mlflow.active_run():
            mlflow.log_params(
                {
                    "seed": self.meta.seed,
                    "max_depth": self.meta.max_depth,
                    "num_trees": self.meta.num_trees,
                }
            )

        total_rows = 0
        for pq_path in pq_paths:
            pq_file = pq.ParquetFile(pq_path)
            total_rows += pq_file.metadata.num_rows

        first_pq = pq.ParquetFile(pq_paths[0])
        first_batch = next(first_pq.iter_batches(batch_size=1, columns=[column]))
        embedding_dim = len(first_batch.column(0)[0])

        self.meta.embedding_dim = embedding_dim
        self.meta.num_rows = total_rows

        projections_list = [self._get_hyperplanes(i) for i in range(self.meta.num_trees)]
        self.projections = projections_list

        signatures = np.memmap(
            self.model_path,
            dtype=self._hash_dtype,
            mode="w+",
            offset=self.HEADER_SIZE,
            shape=(self.meta.num_trees, total_rows),
        )

        curr_idx = 0
        batch_idx = 0
        tracemalloc.start()
        for pq_path in pq_paths:
            logger.info(f"Streaming {pq_path} into LSHiForest...")
            pq_file = pq.ParquetFile(pq_path)

            for batch in pq_file.iter_batches(batch_size=chunk_size, columns=[column]):
                tracemalloc.reset_peak()
                batch_start_time = time.perf_counter()

                flat_arr = batch.column(0).flatten().to_numpy()
                arr = flat_arr.reshape(-1, embedding_dim)  # Shape: (chunk_size, 384)

                slice_end = curr_idx + len(arr)

                for tree_idx in range(self.meta.num_trees):
                    packed_signature = self._compute_simhash(arr, projections_list[tree_idx])
                    signatures[tree_idx, curr_idx:slice_end] = packed_signature

                curr_idx = slice_end

                if mlflow.active_run():
                    peak_mem, _ = tracemalloc.get_traced_memory()
                    mlflow.log_metrics(
                        {
                            "batch_build_time_s": time.perf_counter() - batch_start_time,
                            "batch_build_peak_memory_mb": byte_to_mbyte(peak_mem),
                        },
                        step=batch_idx,
                    )
                batch_idx += 1

        signatures.flush()
        tracemalloc.stop()

        self.forest_mmap = signatures

        build_time = time.perf_counter() - forest_build_start_time
        logger.success(f"Streaming and forest construction complete in {build_time:.2f}s.")

        if mlflow.active_run():
            mlflow.log_metric("forest_build_time_s", build_time)

    def build_forest(
        self,
        embeddings_paths: list[str] | list[Path],
        baseline_output_path: str | Path | None = None,
        column: str = "embedding",
        chunk_size: int = 100_000,
    ):
        self._build_forest(embeddings_paths, column, chunk_size)
        self._calculate_baseline(baseline_output_path)

    def build_forest_from_embeddings(
        self,
        embeddings: np.ndarray,
        baseline_output_path: str | Path | None = None,
        chunk_size: int = 100_000,
    ):
        total_rows, embedding_dim = embeddings.shape

        self.meta.embedding_dim = embedding_dim
        self.meta.num_rows = total_rows

        projections_list = [self._get_hyperplanes(i) for i in range(self.meta.num_trees)]
        self.projections = projections_list

        signatures = np.memmap(
            self.model_path,
            dtype=self._hash_dtype,
            mode="w+",
            offset=self.HEADER_SIZE,
            shape=(self.meta.num_trees, total_rows),
        )

        curr_idx = 0
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            batch = embeddings[start:end]
            if batch.dtype != np.float32:
                batch = batch.astype(np.float32)

            slice_end = curr_idx + batch.shape[0]
            for tree_idx in range(self.meta.num_trees):
                packed_signature = self._compute_simhash(batch, projections_list[tree_idx])
                signatures[tree_idx, curr_idx:slice_end] = packed_signature
            curr_idx = slice_end

        signatures.flush()
        self.forest_mmap = signatures

        if baseline_output_path is not None:
            self._calculate_baseline(baseline_output_path)

    def score(self, new_embeddings: np.ndarray, normalize: bool = True) -> np.ndarray:
        if self.forest_mmap is None or self.projections is None:
            err_msg = "Database signatures are not loaded."
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        meta = self._loaded_meta()
        if not meta.is_sorted:
            err_msg = "Forest signatures are not sorted. Cannot run binary search. Run build process to enable scoring."
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        assert meta.num_rows is not None

        vectors = np.atleast_2d(new_embeddings)
        num_queries = vectors.shape[0]
        logger.debug(f"Scoring {num_queries} queried vectors...")

        score_start_time = time.perf_counter()
        tracemalloc.start()
        tracemalloc.reset_peak()

        total_depths = np.zeros(num_queries, dtype=np.float32)
        isolation_depths = np.empty(num_queries, dtype=np.float32)
        active_mask = np.empty(num_queries, dtype=bool)

        for tree_idx in range(meta.num_trees):
            new_sigs = self._compute_simhash(vectors, self.projections[tree_idx])
            tree_db_sigs = self.forest_mmap[tree_idx, :]
            idx = np.searchsorted(tree_db_sigs, new_sigs)

            valid_right = idx < meta.num_rows
            valid_left = idx > 0

            idx_right = np.where(valid_right, idx, meta.num_rows - 1)
            idx_left = np.where(valid_left, idx - 1, 0)

            right_sigs = tree_db_sigs[idx_right]
            left_sigs = tree_db_sigs[idx_left]

            sentinel = ~self._hash_dtype(0)
            diff_right = np.where(valid_right, new_sigs ^ right_sigs, sentinel)
            diff_left = np.where(valid_left, new_sigs ^ left_sigs, sentinel)

            min_diff = np.minimum(diff_right, diff_left)

            isolation_depths.fill(meta.max_depth)
            active_mask.fill(True)

            for depth in range(1, meta.max_depth):
                if not np.any(active_mask):
                    break

                shift = self._hash_dtype(meta.hash_bits - (depth * meta.bucket_bits))
                isolated = (min_diff[active_mask] >> shift) > 0

                if np.any(isolated):
                    active_indices = np.where(active_mask)[0]
                    newly_isolated_indices = active_indices[isolated]
                    isolation_depths[newly_isolated_indices] = depth
                    active_mask[newly_isolated_indices] = False

            total_depths += isolation_depths

        mean_depths = total_depths / meta.num_trees

        peak_mem, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        score_time = time.perf_counter() - score_start_time

        avg_time_per_query = score_time / num_queries
        avg_mem_per_query = byte_to_mbyte(peak_mem) / num_queries

        logger.info(
            f"Scored {num_queries} vectors in {score_time:.4f}s "
            f"(Avg: {avg_time_per_query:.5f}s/query). "
            f"Batch Peak memory: {byte_to_mbyte(peak_mem):.2f} MB "
            f"(Avg: {avg_mem_per_query:.4f} MB/query)"
        )

        if mlflow.active_run():
            mlflow.log_metrics(
                {
                    "score_time_s": score_time,
                    "score_peak_memory_mb": byte_to_mbyte(peak_mem),
                    "avg_score_time_per_query_s": avg_time_per_query,
                }
            )

        if normalize:
            if meta.c_n is None or meta.c_n <= 0:
                raise RuntimeError(
                    "Normalization requested but c_n is not set. "
                    "Run calculate_baseline() first or load a model with c_n."
                )
            scores = np.power(2.0, -mean_depths / meta.c_n)
            return np.clip(scores, 0.0, 1.0)

        return mean_depths
