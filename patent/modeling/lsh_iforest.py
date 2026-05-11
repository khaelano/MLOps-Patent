from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
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

from patent.modeling.lsh_families import (
    LSHFamily,
    get_lsh_family,
)
from patent.utils import byte_to_mbyte


@dataclass
class LSHIFMeta:
    embedding_dim: int | None = None
    num_rows: int | None = None
    num_trees: int = 50
    max_depth: int = 16
    seed: int = 42
    lsh_family: str = "angle"
    lsh_family_kwargs: dict[str, Any] | None = None
    format_version: int = 3
    tree_offsets: list[int] | None = None
    tree_sizes: list[int] | None = None
    tree_mus: list[float] | None = None
    tree_branchings: list[int] | None = None


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
        return self.model.score(model_input)


class LSHIForest:
    HEADER_SIZE: int = 2048
    _MIN_SAMPLE_LOG2: int = 6
    _MAX_SAMPLE_LOG2: int = 10

    def __init__(
        self,
        num_trees: int = 50,
        max_depth: int = 16,
        chunk_size: int = 100_000,
        seed: int = 42,
        lsh_family: str = "angle",
        **family_kwargs: Any,
    ) -> None:
        self.chunk_size: int = chunk_size
        self.family: LSHFamily = get_lsh_family(lsh_family, **family_kwargs)
        self.family_name: str = lsh_family

        self.meta = LSHIFMeta(
            num_trees=num_trees,
            max_depth=max_depth,
            seed=seed,
            lsh_family=lsh_family,
            lsh_family_kwargs=family_kwargs if family_kwargs else None,
        )

        self.model_path: Path = Path(tempfile.mkdtemp()) / "model.lshif"
        self.forest_mmap: np.memmap | None = None
        self._forest_dtype: np.dtype | None = None
        self.projections: list = []
        self._tempfile: Path | None = None
        self._bytes_per_path: int = 0
        self._tree_offsets: list[int] = []
        self._tree_sizes: list[int] = []
        self._tree_mus: list[float] = []
        self._tree_branchings: list[int] = []
        self._baseline_embeddings_path: str | None = None

        logger.debug(
            f"Initialized LSHiForest: family={lsh_family}, num_trees={num_trees}, "
            f"max_depth={max_depth}, seed={seed}"
        )

    def __del__(self) -> None:
        if hasattr(self, "_tempfile") and self._tempfile and os.path.exists(self._tempfile):
            os.unlink(self._tempfile)

    # ── metadata helpers ─────────────────────────────────────────────

    def _loaded_meta(self) -> LSHIFMeta:
        if self.meta.embedding_dim is None or self.meta.num_rows is None:
            raise RuntimeError("Model not loaded or built")
        return self.meta

    def _dump_meta(self, output_path: Path | str) -> None:
        meta = self._loaded_meta()
        meta_dict: dict[str, Any] = {
            "embedding_dim": meta.embedding_dim,
            "num_rows": meta.num_rows,
            "num_trees": meta.num_trees,
            "max_depth": meta.max_depth,
            "seed": meta.seed,
            "lsh_family": meta.lsh_family,
            "lsh_family_kwargs": meta.lsh_family_kwargs,
            "format_version": meta.format_version,
            "tree_offsets": self._tree_offsets,
            "tree_sizes": self._tree_sizes,
            "tree_mus": self._tree_mus,
            "tree_branchings": self._tree_branchings,
        }
        meta_str = json.dumps(meta_dict)
        if len(meta_str) > self.HEADER_SIZE:
            raise ValueError("Metadata too large for header")
        header = meta_str.ljust(self.HEADER_SIZE).encode("utf-8")
        with open(output_path, "r+b") as f:
            f.write(header)
        logger.debug(f"Metadata dumped to header at {output_path}")

    # ── variable subsampling ─────────────────────────────────────────

    def _draw_subsample_indices(self, total_rows: int, tree_idx: int) -> np.ndarray:
        rng = np.random.default_rng(self.meta.seed + tree_idx + 10000)
        log2_size = rng.integers(self._MIN_SAMPLE_LOG2, self._MAX_SAMPLE_LOG2 + 1)
        sample_size = min(1 << log2_size, total_rows)
        return rng.choice(total_rows, size=sample_size, replace=False)

    # ── per-tree construction ────────────────────────────────────────

    def _estimate_branching(self, hash_vals: np.ndarray) -> int:
        unique_counts = [len(np.unique(hash_vals[i])) for i in range(hash_vals.shape[0])]
        return max(2, int(np.ceil(np.mean(unique_counts))))

    def _build_single_tree(
        self, subsample: np.ndarray, tree_idx: int
    ) -> tuple[np.ndarray, float, int]:
        projections = self.projections[tree_idx]
        hashes = self.family.compute_hashes(subsample, projections)
        paths = self.family.encode_paths(hashes)
        sorted_paths = np.sort(paths)
        v = self._estimate_branching(hashes)
        mu = self.family.mu(len(subsample), v)
        return sorted_paths, mu, v

    # ── forest mmap helpers ──────────────────────────────────────────

    def _get_path_dtype(self) -> np.dtype:
        if self.family_name == "angle":
            return np.dtype(np.uint16) if self.meta.max_depth <= 16 else np.dtype(np.uint32)
        return np.dtype((np.void, self.meta.max_depth))

    def _init_forest_mmap(self, total_paths: int) -> np.memmap:
        path_dt = self._get_path_dtype()
        self._bytes_per_path = path_dt.itemsize
        total_bytes = total_paths * self._bytes_per_path
        mmap = np.memmap(
            str(self.model_path),
            dtype=np.uint8,
            mode="w+",
            offset=self.HEADER_SIZE,
            shape=(total_bytes,),
        )
        self._forest_dtype = path_dt
        self.forest_mmap = mmap
        return mmap

    def _write_tree_paths(self, byte_offset: int, sorted_paths: np.ndarray) -> int:
        assert self.forest_mmap is not None
        path_bytes = sorted_paths.view(np.uint8).reshape(-1)
        end = byte_offset + len(path_bytes)
        self.forest_mmap[byte_offset:end] = path_bytes
        return end

    def _get_tree_paths(self, tree_idx: int) -> np.ndarray:
        assert self.forest_mmap is not None
        assert self._forest_dtype is not None
        start = self._tree_offsets[tree_idx]
        end = start + self._tree_sizes[tree_idx] * self._bytes_per_path
        raw = self.forest_mmap[start:end]
        return raw.view(self._forest_dtype)

    def _ensure_mmap_capacity(self, byte_offset: int) -> None:
        assert self.forest_mmap is not None
        if byte_offset <= self.forest_mmap.shape[0]:
            return
        new_shape = int(byte_offset * 1.5)
        self.forest_mmap.flush()
        self.forest_mmap = np.memmap(
            str(self.model_path),
            dtype=np.uint8,
            mode="r+",
            offset=self.HEADER_SIZE,
            shape=(new_shape,),
        )

    # ── forest build from parquet ────────────────────────────────────

    def _collect_subsample_from_parquet(
        self,
        pq_paths: list[Path],
        indices: np.ndarray,
        embedding_dim: int,
        column: str,
    ) -> np.ndarray:
        rows: list[np.ndarray] = []
        collected = 0

        for pq_path in pq_paths:
            pq_file = pq.ParquetFile(pq_path)
            for batch in pq_file.iter_batches(batch_size=self.chunk_size, columns=[column]):
                batch_start = collected
                batch_end = collected + batch.num_rows
                local_indices = sorted(
                    [idx - batch_start for idx in indices if batch_start <= idx < batch_end]
                )
                if local_indices:
                    flat_arr = batch.column(0).flatten().to_numpy(zero_copy_only=False)
                    arr = flat_arr.reshape(-1, embedding_dim)
                    rows.append(arr[local_indices].astype(np.float32, copy=False))
                collected = batch_end

        if not rows:
            raise RuntimeError("No subsample rows collected from parquet")
        return np.concatenate(rows, axis=0).astype(np.float32)

    def _build_forest_pq(
        self,
        pq_paths: list[str] | list[Path],
        column: str = "embedding",
    ) -> None:
        logger.info("Starting LSHiForest build from Parquet streams...")
        t_start = time.perf_counter()

        pq_paths = [Path(p) for p in pq_paths]

        total_rows = 0
        for pq_path in pq_paths:
            pq_file = pq.ParquetFile(pq_path)
            total_rows += pq_file.metadata.num_rows

        first_pq = pq.ParquetFile(pq_paths[0])
        first_batch = next(first_pq.iter_batches(batch_size=1, columns=[column]))
        embedding_dim = len(first_batch.column(0)[0])

        self.meta.embedding_dim = embedding_dim
        self.meta.num_rows = total_rows
        self._baseline_embeddings_path = str(pq_paths[0].parent) if pq_paths else None

        if mlflow.active_run():
            mlflow.log_params(
                {
                    "seed": self.meta.seed,
                    "max_depth": self.meta.max_depth,
                    "num_trees": self.meta.num_trees,
                    "lsh_family": self.family_name,
                    "total_rows": total_rows,
                    "embedding_dim": embedding_dim,
                }
            )

        self.projections = [
            self.family.generate_projections(
                embedding_dim, self.meta.max_depth, self.meta.seed + i
            )
            for i in range(self.meta.num_trees)
        ]

        self._tree_offsets = []
        self._tree_sizes = []
        self._tree_mus = []
        self._tree_branchings = []

        total_paths_est = self.meta.num_trees * (1 << self._MAX_SAMPLE_LOG2)
        self._init_forest_mmap(total_paths_est)

        byte_offset = 0
        tracemalloc.start()

        for tree_idx in range(self.meta.num_trees):
            logger.debug(f"Building tree {tree_idx + 1}/{self.meta.num_trees}...")
            tracemalloc.reset_peak()

            indices = self._draw_subsample_indices(total_rows, tree_idx)
            subsample = self._collect_subsample_from_parquet(
                pq_paths, indices, embedding_dim, column
            )
            if len(subsample) != len(indices):
                raise RuntimeError(
                    f"Tree {tree_idx}: expected {len(indices)} rows, got {len(subsample)}"
                )

            sorted_paths, mu, v = self._build_single_tree(subsample, tree_idx)
            self._tree_offsets.append(byte_offset)
            self._tree_sizes.append(len(sorted_paths))
            self._tree_mus.append(mu)
            self._tree_branchings.append(v)

            byte_offset = self._write_tree_paths(byte_offset, sorted_paths)
            self._ensure_mmap_capacity(byte_offset)

            del subsample, sorted_paths

        tracemalloc.stop()
        assert self.forest_mmap is not None
        self.forest_mmap.flush()

        delta = time.perf_counter() - t_start
        logger.success(
            f"Forest built in {delta:.2f}s. {self.meta.num_trees} trees, "
            f"{sum(self._tree_sizes)} total paths, "
            f"avg μ={np.mean(self._tree_mus):.4f}"
        )

        if mlflow.active_run():
            mlflow.log_metrics(
                {
                    "forest_build_time_s": delta,
                    "avg_mu": float(np.mean(self._tree_mus)),
                    "avg_branching": float(np.mean(self._tree_branchings)),
                    "total_paths": sum(self._tree_sizes),
                }
            )

        self._dump_meta(self.model_path)

    # ── forest build from in-memory array ────────────────────────────

    def _build_forest_array(self, embeddings: np.ndarray) -> None:
        logger.info("Starting LSHiForest build from in-memory array...")
        t_start = time.perf_counter()

        total_rows, embedding_dim = embeddings.shape
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        self.meta.embedding_dim = embedding_dim
        self.meta.num_rows = total_rows

        self.projections = [
            self.family.generate_projections(
                embedding_dim, self.meta.max_depth, self.meta.seed + i
            )
            for i in range(self.meta.num_trees)
        ]

        self._tree_offsets = []
        self._tree_sizes = []
        self._tree_mus = []
        self._tree_branchings = []

        total_paths_est = self.meta.num_trees * (1 << self._MAX_SAMPLE_LOG2)
        self._init_forest_mmap(total_paths_est)

        byte_offset = 0
        tracemalloc.start()

        for tree_idx in range(self.meta.num_trees):
            logger.debug(f"Building tree {tree_idx + 1}/{self.meta.num_trees}...")
            tracemalloc.reset_peak()

            indices = self._draw_subsample_indices(total_rows, tree_idx)
            subsample = embeddings[indices].astype(np.float32, copy=False).copy()

            sorted_paths, mu, v = self._build_single_tree(subsample, tree_idx)
            self._tree_offsets.append(byte_offset)
            self._tree_sizes.append(len(sorted_paths))
            self._tree_mus.append(mu)
            self._tree_branchings.append(v)

            byte_offset = self._write_tree_paths(byte_offset, sorted_paths)
            self._ensure_mmap_capacity(byte_offset)

            del subsample, sorted_paths

        tracemalloc.stop()
        assert self.forest_mmap is not None
        self.forest_mmap.flush()

        delta = time.perf_counter() - t_start
        logger.success(
            f"Forest built in {delta:.2f}s. {self.meta.num_trees} trees, "
            f"{sum(self._tree_sizes)} total paths, "
            f"avg μ={np.mean(self._tree_mus):.4f}"
        )

        self._dump_meta(self.model_path)

    # ── public build API ─────────────────────────────────────────────

    def build_forest(
        self,
        embeddings_paths: list[str] | list[Path],
        baseline_output_path: str | Path | None = None,
        column: str = "embedding",
        chunk_size: int = 100_000,
    ) -> None:
        self.chunk_size = chunk_size
        self._build_forest_pq(embeddings_paths, column)
        if baseline_output_path is not None:
            self._calculate_baseline_pq(embeddings_paths, baseline_output_path, column)

    def build_forest_from_embeddings(
        self,
        embeddings: np.ndarray,
        baseline_output_path: str | Path | None = None,
        chunk_size: int = 100_000,
    ) -> None:
        self.chunk_size = chunk_size
        self._build_forest_array(embeddings)
        if baseline_output_path is not None:
            self._calculate_baseline_array(embeddings, baseline_output_path)

    # ── baseline depth calculation (in-memory) ───────────────────────

    def _calculate_baseline_array(
        self,
        embeddings: np.ndarray,
        output_path: str | Path,
    ) -> None:
        logger.info("Computing baseline anomaly depths...")
        t_start = time.perf_counter()

        total_rows = embeddings.shape[0]
        meta = self._loaded_meta()

        all_depths = np.zeros(total_rows, dtype=np.float64)

        for start in range(0, total_rows, self.chunk_size):
            end = min(start + self.chunk_size, total_rows)
            chunk = np.array(embeddings[start:end], dtype=np.float32)
            chunk_depths = np.zeros(end - start, dtype=np.float64)

            for tree_idx in range(meta.num_trees):
                tree_paths = self._get_tree_paths(tree_idx)
                if self.family_name == "angle":
                    t_depths = self._score_angle_chunk(tree_idx, tree_paths, chunk)
                else:
                    t_depths = self._score_l2_chunk(tree_idx, tree_paths, chunk)
                chunk_depths += t_depths

            all_depths[start:end] = chunk_depths / meta.num_trees
            del chunk, chunk_depths

        np.save(str(output_path), all_depths.astype(np.float32))

        delta = time.perf_counter() - t_start
        logger.success(f"Baseline calculation complete in {delta:.2f}s.")

        if mlflow.active_run():
            mlflow.log_metric("baseline_calc_time_s", delta)
            mlflow.log_artifact(str(output_path), artifact_path="baselines")

    # ── baseline depth calculation (parquet streaming) ───────────────

    def _calculate_baseline_pq(
        self,
        pq_paths: list[str] | list[Path],
        output_path: str | Path,
        column: str = "embedding",
    ) -> None:
        logger.info("Computing baseline anomaly depths from Parquet...")
        t_start = time.perf_counter()

        meta = self._loaded_meta()
        assert meta.embedding_dim is not None
        assert meta.num_rows is not None

        all_depths = np.zeros(meta.num_rows, dtype=np.float64)
        row_offset = 0

        for pq_path in map(Path, pq_paths):
            pq_file = pq.ParquetFile(pq_path)
            for batch in pq_file.iter_batches(batch_size=self.chunk_size, columns=[column]):
                flat_arr = batch.column(0).flatten().to_numpy(zero_copy_only=False)
                chunk = flat_arr.reshape(-1, meta.embedding_dim).astype(np.float32)
                n = chunk.shape[0]
                chunk_depths = np.zeros(n, dtype=np.float64)

                for tree_idx in range(meta.num_trees):
                    tree_paths = self._get_tree_paths(tree_idx)
                    if self.family_name == "angle":
                        t_depths = self._score_angle_chunk(tree_idx, tree_paths, chunk)
                    else:
                        t_depths = self._score_l2_chunk(tree_idx, tree_paths, chunk)
                    chunk_depths += t_depths

                all_depths[row_offset : row_offset + n] = chunk_depths / meta.num_trees
                row_offset += n
                del chunk, chunk_depths

        np.save(str(output_path), all_depths.astype(np.float32))

        delta = time.perf_counter() - t_start
        logger.success(f"Baseline calculation complete in {delta:.2f}s.")

        if mlflow.active_run():
            mlflow.log_metric("baseline_calc_time_s", delta)
            mlflow.log_artifact(str(output_path), artifact_path="baselines")

    # ── scoring: angle-based ─────────────────────────────────────────

    def _score_angle_chunk(
        self,
        tree_idx: int,
        tree_paths: np.ndarray,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        projections = self.projections[tree_idx]
        hashes = self.family.compute_hashes(embeddings, projections)
        query_paths = self.family.encode_paths(hashes)

        n_queries = len(query_paths)
        H = self.meta.max_depth
        idx = np.searchsorted(tree_paths, query_paths)

        tree_len = len(tree_paths)
        left_idx = np.clip(idx - 1, 0, tree_len - 1)
        right_idx = np.clip(idx, 0, tree_len - 1)

        qp_u64 = query_paths.astype(np.uint64)
        lp_u64 = tree_paths[left_idx].astype(np.uint64)
        rp_u64 = tree_paths[right_idx].astype(np.uint64)

        xor_left = qp_u64 ^ lp_u64
        xor_right = qp_u64 ^ rp_u64

        cp_left = np.full(n_queries, H, dtype=np.int32)
        cp_right = np.full(n_queries, H, dtype=np.int32)
        mask_l = xor_left > 0
        mask_r = xor_right > 0
        if np.any(mask_l):
            cp_left[mask_l] = (
                H - 1 - np.floor(np.log2(xor_left[mask_l].astype(np.float64))).astype(np.int32)
            )
        if np.any(mask_r):
            cp_right[mask_r] = (
                H - 1 - np.floor(np.log2(xor_right[mask_r].astype(np.float64))).astype(np.int32)
            )

        depths = np.maximum(cp_left, cp_right).astype(np.float64)

        full_left = cp_left == H
        full_right = cp_right == H
        full_match = full_left | full_right
        if np.any(full_match):
            for qi in np.where(full_match)[0]:
                qp = query_paths[qi]
                li = idx[qi] - 1
                left_count = 0
                while li >= 0 and tree_paths[li] == qp:
                    left_count += 1
                    li -= 1
                ri = idx[qi]
                right_count = 0
                while ri < tree_len and tree_paths[ri] == qp:
                    right_count += 1
                    ri += 1
                leaf_size = left_count + right_count
                if leaf_size > 1:
                    depths[qi] += self.family.mu(leaf_size, self._tree_branchings[tree_idx])

        return depths

    # ── scoring: ℓ₂-based ────────────────────────────────────────────

    def _score_l2_chunk(
        self,
        tree_idx: int,
        tree_paths: np.ndarray,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        projections = self.projections[tree_idx]
        hashes = self.family.compute_hashes(embeddings, projections)
        query_paths = self.family.encode_paths(hashes)

        n_queries = len(query_paths)
        H = self.meta.max_depth
        idx = np.searchsorted(tree_paths, query_paths)

        tree_len = len(tree_paths)
        left_idx = np.clip(idx - 1, 0, tree_len - 1)
        right_idx = np.clip(idx, 0, tree_len - 1)

        q_uint8 = (hashes.astype(np.int16) + 128).astype(np.uint8).T

        lp_raw = tree_paths[left_idx]
        rp_raw = tree_paths[right_idx]

        left_hv = np.frombuffer(lp_raw.tobytes(), dtype=np.uint8).reshape(n_queries, H)
        right_hv = np.frombuffer(rp_raw.tobytes(), dtype=np.uint8).reshape(n_queries, H)

        cp_left = self.family.common_prefix_depth(
            q_uint8[0] if n_queries == 1 else q_uint8,
            left_hv if n_queries > 1 else left_hv.reshape(1, -1),
        )
        cp_right = self.family.common_prefix_depth(
            q_uint8[0] if n_queries == 1 else q_uint8,
            right_hv if n_queries > 1 else right_hv.reshape(1, -1),
        )

        depths = np.maximum(cp_left, cp_right).astype(np.float64)

        full_match = (cp_left == H) | (cp_right == H)
        if np.any(full_match):
            for qi in np.where(full_match)[0]:
                qp = query_paths[qi]
                li = idx[qi] - 1
                left_count = 0
                while li >= 0 and tree_paths[li] == qp:
                    left_count += 1
                    li -= 1
                ri = idx[qi]
                right_count = 0
                while ri < tree_len and tree_paths[ri] == qp:
                    right_count += 1
                    ri += 1
                leaf_size = left_count + right_count
                if leaf_size > 1:
                    depths[qi] += self.family.mu(leaf_size, self._tree_branchings[tree_idx])

        return depths

    # ── public scoring ───────────────────────────────────────────────

    def score(self, new_embeddings: np.ndarray, normalize: bool = True) -> np.ndarray:
        if self.forest_mmap is None or not self.projections:
            raise RuntimeError("Forest not built or loaded. Run build_forest() first.")

        meta = self._loaded_meta()
        vectors = np.atleast_2d(new_embeddings).astype(np.float32)
        num_queries = vectors.shape[0]

        logger.debug(f"Scoring {num_queries} queries...")
        score_start = time.perf_counter()
        tracemalloc.start()
        tracemalloc.reset_peak()

        if normalize:
            accumulated = np.zeros(num_queries, dtype=np.float64)
            for tree_idx in range(meta.num_trees):
                tree_paths = self._get_tree_paths(tree_idx)
                if self.family_name == "angle":
                    depths = self._score_angle_chunk(tree_idx, tree_paths, vectors)
                else:
                    depths = self._score_l2_chunk(tree_idx, tree_paths, vectors)
                mu = self._tree_mus[tree_idx]
                accumulated += np.power(2.0, -depths / mu)
            scores = accumulated / meta.num_trees
        else:
            total_depths = np.zeros(num_queries, dtype=np.float64)
            for tree_idx in range(meta.num_trees):
                tree_paths = self._get_tree_paths(tree_idx)
                if self.family_name == "angle":
                    depths = self._score_angle_chunk(tree_idx, tree_paths, vectors)
                else:
                    depths = self._score_l2_chunk(tree_idx, tree_paths, vectors)
                total_depths += depths
            scores = total_depths / meta.num_trees

        peak_mem, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        score_time = time.perf_counter() - score_start

        logger.info(
            f"Scored {num_queries} vectors in {score_time:.4f}s "
            f"(Avg: {score_time / num_queries:.5f}s/query). "
            f"Peak memory: {byte_to_mbyte(peak_mem):.2f} MB"
        )

        if mlflow.active_run():
            mlflow.log_metrics(
                {
                    "score_time_s": score_time,
                    "score_peak_memory_mb": byte_to_mbyte(peak_mem),
                    "avg_score_time_per_query_s": score_time / num_queries,
                }
            )

        if normalize:
            return np.clip(scores, 0.0, 1.0)
        return scores

    # ── serialisation ────────────────────────────────────────────────

    def _meta_as_dict(self) -> dict[str, Any]:
        return {
            "embedding_dim": self.meta.embedding_dim,
            "num_rows": self.meta.num_rows,
            "num_trees": self.meta.num_trees,
            "max_depth": self.meta.max_depth,
            "seed": self.meta.seed,
            "lsh_family": self.family_name,
            "lsh_family_kwargs": self.meta.lsh_family_kwargs,
            "format_version": 3,
            "tree_offsets": self._tree_offsets,
            "tree_sizes": self._tree_sizes,
            "tree_mus": self._tree_mus,
            "tree_branchings": self._tree_branchings,
        }

    def save_model(
        self, output_path_str: str | Path = "output.lshif", compress: bool = True
    ) -> None:
        output_path = Path(output_path_str)
        logger.info(f"Saving LSHiForest model to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._dump_meta(self.model_path)

        with open(self.model_path, "rb") as src:
            src.seek(self.HEADER_SIZE)
            forest_bytes = src.read()

        header_json = json.dumps(self._meta_as_dict())
        header = header_json.ljust(self.HEADER_SIZE).encode("utf-8")

        if compress:
            compressed = zstandard.ZstdCompressor(level=3).compress(forest_bytes)
            with open(output_path, "wb") as dst:
                dst.write(header)
                dst.write(np.uint32(len(compressed)).tobytes())
                dst.write(compressed)
            logger.info(f"Compressed model: {len(compressed)} bytes (zstd)")
        else:
            with open(output_path, "wb") as dst:
                dst.write(header)
                dst.write(forest_bytes)

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

    @classmethod
    def load_model(cls, model_path_str: str | Path) -> LSHIForest:
        model_path = Path(model_path_str)
        logger.info(f"Loading LSHiForest model from {model_path}...")

        with open(model_path, "rb") as f:
            header_bytes = f.read(cls.HEADER_SIZE)

        meta_dict = json.loads(header_bytes.decode("utf-8").strip())
        format_version = meta_dict.get("format_version", 1)

        if format_version < 3:
            raise ValueError(
                f"Model format v{format_version} not supported in this version. "
                "Rebuild the model with the current code."
            )

        lsh_family = meta_dict.get("lsh_family", "angle")
        lsh_kwargs = meta_dict.get("lsh_family_kwargs") or {}

        instance = cls(
            num_trees=meta_dict["num_trees"],
            max_depth=meta_dict["max_depth"],
            seed=meta_dict["seed"],
            lsh_family=lsh_family,
            **lsh_kwargs,
        )

        instance.meta.embedding_dim = meta_dict["embedding_dim"]
        instance.meta.num_rows = meta_dict["num_rows"]
        instance.meta.format_version = format_version

        instance._tree_offsets = meta_dict.get("tree_offsets", [])
        instance._tree_sizes = meta_dict.get("tree_sizes", [])
        instance._tree_mus = meta_dict.get("tree_mus", [])
        instance._tree_branchings = meta_dict.get("tree_branchings", [])

        with open(model_path, "rb") as f:
            f.seek(cls.HEADER_SIZE)
            comp_size = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            compressed = f.read(comp_size)

        raw = zstandard.ZstdDecompressor().decompress(compressed)

        path_dt = instance._get_path_dtype()
        instance._bytes_per_path = path_dt.itemsize

        temp_dir = Path(tempfile.mkdtemp())
        temp_path = temp_dir / "forest.lshif"
        instance._tempfile = temp_path

        forest_len = len(raw)

        mmap_file = np.memmap(
            str(temp_path),
            dtype=np.uint8,
            mode="w+",
            offset=cls.HEADER_SIZE,
            shape=(forest_len,),
        )
        mmap_file[:] = np.frombuffer(raw, dtype=np.uint8)
        mmap_file.flush()

        header_json = json.dumps(instance._meta_as_dict())
        header = header_json.ljust(cls.HEADER_SIZE).encode("utf-8")
        with open(temp_path, "r+b") as f:
            f.write(header)

        instance.forest_mmap = mmap_file
        instance._forest_dtype = path_dt
        instance.model_path = temp_path

        instance.projections = [
            instance.family.generate_projections(
                instance.meta.embedding_dim,
                instance.meta.max_depth,
                instance.meta.seed + i,
            )
            for i in range(instance.meta.num_trees)
        ]

        logger.success(
            f"Model loaded. Family={lsh_family}, Trees={instance.meta.num_trees}, "
            f"Rows={instance.meta.num_rows}"
        )
        return instance
