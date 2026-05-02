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


class LSHIForest:
    HEADER_SIZE: int = 1024

    def __init__(
        self,
        num_trees: int = 50,
        max_depth: int = 16,
        chunk_size: int = 200_000,
        seed: int = 42,
    ) -> None:
        if max_depth > 16:
            err_msg = "max_depth cannot exceed 16 when using 64-bit hashes with 4-bit buckets."
            logger.error(err_msg)
            raise ValueError(err_msg)

        self.chunk_size: int = chunk_size
        self.meta: LSHIFMeta = LSHIFMeta(
            embedding_dim=None,
            num_rows=None,
            num_trees=num_trees,
            max_depth=max_depth,
            seed=seed,
        )

        self.model_path: Path = Path(tempfile.mkdtemp()) / "model.lshif"
        self.forest_mmap: np.memmap | None = None
        self.projections: list[np.ndarray] | None = None

        logger.debug(
            f"Initialized LSHIForest: num_trees={num_trees}, "
            f"max_depth={max_depth}, chunk_size={chunk_size}, seed={seed}"
        )

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
        return rng.standard_normal((dim, 64)).astype("float32")

    def _compute_simhash(self, vectors: np.ndarray, projection_matrix: np.ndarray) -> np.ndarray:
        projected = np.dot(vectors, projection_matrix)
        return np.packbits(projected > 0, axis=1).view(np.uint64).reshape(-1)

    def _generate_signatures_for_tree(
        self, embeddings_mmap: np.memmap, projection_matrix: np.ndarray, chunk_size: int
    ) -> np.ndarray:
        meta = self._loaded_meta()
        assert meta.num_rows is not None

        signatures = np.zeros(meta.num_rows, dtype=np.uint64)
        curr_idx = 0

        while curr_idx < meta.num_rows:
            slice_end = min(curr_idx + chunk_size, meta.num_rows)
            chunk = embeddings_mmap[curr_idx:slice_end]
            signatures[curr_idx:slice_end] = self._compute_simhash(chunk, projection_matrix)
            curr_idx = slice_end

        return signatures

    def _build_single_tree(self, signatures: np.ndarray) -> np.ndarray:
        meta = self._loaded_meta()
        n = len(signatures)
        path_lengths = np.full(n, meta.max_depth, dtype=np.float32)
        active_mask = np.ones(n, dtype=bool)

        for depth in range(1, meta.max_depth):
            if not np.any(active_mask):
                break

            shift = np.uint64(64 - (depth * 4))
            active_sigs = signatures[active_mask]
            prefixes = active_sigs >> shift

            _, inverse, counts = np.unique(prefixes, return_inverse=True, return_counts=True)
            isolated_mask = counts[inverse] == 1

            if np.any(isolated_mask):
                active_indices = np.where(active_mask)[0]
                newly_isolated = active_indices[isolated_mask]
                path_lengths[newly_isolated] = depth
                active_mask[newly_isolated] = False

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

        instance.forest_mmap = np.memmap(
            str(model_path),
            dtype="uint64",
            mode="r",
            offset=instance.HEADER_SIZE,
            shape=(loaded_meta.num_trees, loaded_meta.num_rows),
        )
        instance.projections = [instance._get_hyperplanes(i) for i in range(loaded_meta.num_trees)]

        logger.success(
            f"Model loaded. Trees: {loaded_meta.num_trees}, Rows: {loaded_meta.num_rows}"
        )
        return instance

    def save_model(self, output_path_str: str | Path = "output.lshif") -> None:
        output_path = Path(output_path_str)
        logger.info(f"Saving LSHiForest model to {output_path}...")

        if output_path.resolve() != self.model_path.resolve():
            shutil.copy2(self.model_path, output_path)

        if mlflow.active_run():
            logger.debug("Logging model to MLflow...")

            input_example = np.random.randn(3, 384).astype("float32")
            output_example = self.score(input_example)

            mlflow.pyfunc.log_model(
                name="lshiforest",
                python_model=LSHIFWrapper(),
                artifacts={"lshif_file": str(output_path)},
                input_example=input_example,
                signature=infer_signature(input_example, output_example),
            )

        logger.success("Model saved successfully.")

    def build_forest(
        self,
        embeddings_path: str | Path,
        embeddings_dim: int,
        baseline_output_path: str | Path | None = "baseline_depths.npy",
        resume: bool = True,
    ) -> None:
        logger.info(f"Starting forest build from embeddings: {embeddings_path}")
        forest_build_start_time = time.perf_counter()

        self.meta.embedding_dim = embeddings_dim
        self.meta.num_rows = self._calc_mmap_row(embeddings_path, embeddings_dim)

        loaded_meta = self._loaded_meta()
        assert loaded_meta.num_rows is not None
        assert loaded_meta.embedding_dim is not None

        if mlflow.active_run():
            mlflow.log_params(
                {
                    "seed": loaded_meta.seed,
                    "max_depth": loaded_meta.max_depth,
                    "num_trees": loaded_meta.num_trees,
                }
            )

        checkpoint_path = Path(tempfile.gettempdir()) / "forest.resume"
        depths_path = Path(tempfile.gettempdir()) / "forest_depths.npy"
        model_output_path = str(self.model_path)

        start_tree = 0
        if not os.path.exists(model_output_path):
            logger.debug(f"Initializing empty forest map at {model_output_path}")
            with open(model_output_path, "wb") as f:
                f.write(b" " * self.HEADER_SIZE)

        if resume and depths_path.exists():
            total_depths = np.load(depths_path)
            logger.info("Resuming baseline depth accumulator from disk.")
        else:
            total_depths = np.zeros(loaded_meta.num_rows, dtype=np.float32)

        if resume and os.path.exists(checkpoint_path) and os.path.exists(model_output_path):
            with open(checkpoint_path, "r") as f:
                content = f.read().strip()
                if content:
                    start_tree = int(content)
                    logger.info(f"Found checkpoint! Resuming from Tree {start_tree + 1}...")

        if start_tree >= loaded_meta.num_trees:
            logger.info("Forest is already fully built according to the checkpoint.")
            return

        embeddings_mmap = np.memmap(
            embeddings_path,
            dtype="float32",
            mode="r",
            shape=(loaded_meta.num_rows, loaded_meta.embedding_dim),
        )
        signatures_mmap = np.memmap(
            model_output_path,
            dtype="uint64",
            mode="r+",
            offset=self.HEADER_SIZE,
            shape=(loaded_meta.num_trees, loaded_meta.num_rows),
        )

        tracemalloc.start()
        for tree_idx in range(start_tree, loaded_meta.num_trees):
            tracemalloc.reset_peak()
            logger.debug(f"Building signatures for tree {tree_idx + 1}/{loaded_meta.num_trees}...")
            tree_build_start_time = time.perf_counter()
            projection_matrix = self._get_hyperplanes(tree_idx)
            signatures = self._generate_signatures_for_tree(
                embeddings_mmap, projection_matrix, self.chunk_size
            )

            path_lengths = self._build_single_tree(signatures)
            total_depths += path_lengths

            signatures.sort()
            signatures_mmap[tree_idx] = signatures
            signatures_mmap.flush()

            with open(checkpoint_path, "w") as f:
                f.write(str(tree_idx + 1))
            np.save(depths_path, total_depths)

            if mlflow.active_run():
                peak_mem, curr_mem = tracemalloc.get_traced_memory()
                mlflow.log_metrics(
                    {
                        "tree_build_time_s": time.perf_counter() - tree_build_start_time,
                        "tree_build_peak_memory_mb": byte_to_mbyte(peak_mem),
                    },
                    step=tree_idx,
                )

        tracemalloc.stop()

        self.forest_mmap = signatures_mmap
        self.projections = [self._get_hyperplanes(i) for i in range(loaded_meta.num_trees)]

        baseline_depths = total_depths / loaded_meta.num_trees
        c_n = float(np.mean(baseline_depths))
        self.meta.c_n = c_n
        self._dump_meta(self.model_path)

        if baseline_output_path is not None:
            logger.debug(f"Dumping baseline scores at {baseline_output_path}.")
            np.save(baseline_output_path, baseline_depths)

        for p in [checkpoint_path, depths_path]:
            if p.exists():
                p.unlink()

        build_time = time.perf_counter() - forest_build_start_time
        logger.success(f"Forest & baseline complete in {build_time:.2f}s. c_n={c_n:.4f}")

        if mlflow.active_run():
            mlflow.log_metric("forest_build_time_s", build_time)
            mlflow.log_metric("c_n", c_n)
            mlflow.log_artifact(str(baseline_output_path), artifact_path="baselines")

    def score(self, new_embeddings: np.ndarray, normalize: bool = True) -> np.ndarray:
        if self.forest_mmap is None or self.projections is None:
            err_msg = "Database signatures are not loaded."
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        meta = self._loaded_meta()
        assert meta.num_rows is not None

        vectors = np.atleast_2d(new_embeddings)
        num_queries = vectors.shape[0]
        logger.debug(f"Scoring {num_queries} queried vectors...")

        total_depths = np.zeros(num_queries, dtype=np.float32)

        for tree_idx in range(meta.num_trees):
            new_sigs = self._compute_simhash(vectors, self.projections[tree_idx])
            tree_db_sigs = self.forest_mmap[tree_idx]

            isolation_depths = np.full(num_queries, meta.max_depth, dtype=np.float32)
            active_mask = np.ones(num_queries, dtype=bool)

            for depth in range(1, meta.max_depth):
                if not np.any(active_mask):
                    break

                shift = np.uint64(64 - (depth * 4))
                active_sigs = new_sigs[active_mask]
                new_prefixes = active_sigs >> shift
                search_targets = new_prefixes << shift

                idx = np.searchsorted(tree_db_sigs, search_targets)
                valid_idx = idx < meta.num_rows
                safe_idx = np.clip(idx, 0, meta.num_rows - 1)
                exists = valid_idx & ((tree_db_sigs[safe_idx] >> shift) == new_prefixes)
                isolated_mask = ~exists

                if np.any(isolated_mask):
                    active_indices = np.where(active_mask)[0]
                    newly_isolated_indices = active_indices[isolated_mask]
                    isolation_depths[newly_isolated_indices] = depth
                    active_mask[newly_isolated_indices] = False

            total_depths += isolation_depths

        mean_depths = total_depths / meta.num_trees

        if normalize:
            if meta.c_n is None or meta.c_n <= 0:
                raise RuntimeError(
                    "Normalization requested but c_n is not set. "
                    "Run calculate_baseline() first or load a model with c_n."
                )
            scores = np.power(2.0, -mean_depths / meta.c_n)
            return np.clip(scores, 0.0, 1.0)

        return mean_depths
