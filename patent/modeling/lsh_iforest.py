from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import tracemalloc
from typing import Any

import mlflow
import numpy as np


class LSHIFWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from patent.modeling.lsh_iforest import LSHIForest

        model_path = context.artifacts["lshif_file"]
        self.model = LSHIForest.load_model(model_path)

    def predict(self, context, model_input: np.ndarray, params: dict[str, Any] | None = None):
        data = model_input
        if data.ndim == 1 or data.shape[0] == 1:
            return self.model.score(data)
        else:
            return [self.model.score(row) for row in data]
        pass


@dataclass
class LSHIFMeta:
    embedding_dim: int | None
    num_rows: int | None
    num_trees: int = 50
    max_depth: int = 16
    seed: int = 42


class LSHIForest:
    HEADER_SIZE = 1024

    def __init__(
        self,
        num_trees: int = 50,
        max_depth: int = 16,
        chunk_size=200_000,
        seed=42,
    ):
        if max_depth > 16:
            raise ValueError(
                "max_depth cannot exceed 16 when using 64-bit hashes with 4-bit buckets."
            )

        self.chunk_size = chunk_size
        self.meta = LSHIFMeta(
            embedding_dim=None,
            num_rows=None,
            num_trees=num_trees,
            max_depth=max_depth,
            seed=seed,
        )

        self.model_path: Path = Path(tempfile.gettempdir()) / "forest.memmap"
        self.forest_mmap: np.memmap | None = None
        self.projections: list[np.ndarray] | None = None

    def _dump_meta(self, output_path: Path):
        meta_str = json.dumps(asdict(self._loaded_meta()))
        if len(meta_str) > self.HEADER_SIZE:
            raise ValueError("Metadata too large for header")

        header = meta_str.ljust(self.HEADER_SIZE).encode("utf-8")
        with open(output_path, "r+b") as f:
            f.write(header)

    def _loaded_meta(self):
        if self.meta.embedding_dim is None or self.meta.num_rows is None:
            raise RuntimeError("Model not loaded or built")
        return self.meta

    def _get_hyperplanes(self, tree_idx: int) -> np.ndarray:
        np.random.seed(self.meta.seed + tree_idx)
        return np.random.standard_normal((self._loaded_meta().embedding_dim, 64)).astype("float32")

    def _compute_simhash(self, vectors: np.ndarray, projection_matrix: np.ndarray) -> np.ndarray:
        projected = np.dot(vectors, projection_matrix)
        return np.packbits(projected > 0, axis=1).view(np.uint64).reshape(-1)

    def _generate_signatures_for_tree(
        self, embeddings_mmap: np.memmap, projection_matrix: np.ndarray, chunk_size: int
    ) -> np.ndarray:
        if self.meta.num_rows is None:
            raise ValueError(
                "Number of rows isn't known. Please train or load the database first."
            )

        signatures = np.zeros(self.meta.num_rows, dtype=np.uint64)
        curr_idx = 0

        while curr_idx < self.meta.num_rows:
            slice_end = min(curr_idx + chunk_size, self.meta.num_rows)
            chunk = embeddings_mmap[curr_idx:slice_end]
            signatures[curr_idx:slice_end] = self._compute_simhash(chunk, projection_matrix)
            curr_idx = slice_end

        return signatures

    def _build_single_tree(self, signatures: np.ndarray) -> np.ndarray:
        if self.meta.num_rows is None:
            raise ValueError(
                "Number of rows isn't known. Please train or load the database first."
            )

        path_lengths = np.zeros(self.meta.num_rows, dtype=np.uint8)
        initial_indices = np.arange(self.meta.num_rows, dtype=np.int64)
        stack: list[tuple[np.ndarray, int]] = [(initial_indices, 1)]

        while stack:
            indices, depth = stack.pop()

            if len(indices) == 1:
                path_lengths[indices[0]] = depth
                continue

            if depth >= self.meta.max_depth:
                for idx in indices:
                    path_lengths[idx] = depth
                continue

            shift = np.uint64(64 - (depth * 4))
            bit_mask = np.uint64(0xF)

            bucket_ids = (signatures[indices] >> shift) & bit_mask

            buckets = defaultdict(list)
            for i, bucket_id in enumerate(bucket_ids):
                buckets[bucket_id].append(indices[i])

            for bucket_indices in buckets.values():
                stack.append((np.array(bucket_indices, dtype=np.int64), depth + 1))

        return path_lengths

    def _calc_mmap_row(
        self,
        embeddings_path: str | Path,
        embedding_dim: int,
        dtype_size: int = 4,
    ) -> int:
        embeddings_path = Path(embeddings_path)

        if not embeddings_path.exists():
            raise FileNotFoundError(f"File not found: {embeddings_path}")

        file_size_bytes = os.path.getsize(embeddings_path)
        bytes_per_row = embedding_dim * dtype_size

        if bytes_per_row == 0:
            raise ValueError("bytes_per_row cannot be zero (check embedding_dim)")

        if file_size_bytes % bytes_per_row != 0:
            raise ValueError(
                f"File size ({file_size_bytes} bytes) doesn't perfectly divide "
                f"by {bytes_per_row}. Is {embeddings_path} a valid float32 mmap file?"
            )

        return file_size_bytes // bytes_per_row

    @classmethod
    def load_model(cls, model_path_str: str = "output.lshif"):
        model_path = Path(model_path_str).resolve()
        with open(model_path, "rb") as f:
            header_bytes = f.read(cls.HEADER_SIZE)
        meta = json.loads(header_bytes.decode("utf-8").strip())
        meta = LSHIFMeta(**meta)

        instance = cls()
        instance.meta = meta
        instance.forest_mmap = np.memmap(
            str(model_path),
            dtype="uint64",
            mode="r",
            offset=instance.HEADER_SIZE,
            shape=(instance._loaded_meta().num_trees, instance._loaded_meta().num_rows),
        )
        instance.projections = [
            instance._get_hyperplanes(i) for i in range(instance.meta.num_trees)
        ]

        return instance

    def save_model(self, output_path_sstr: str = "output.lshif"):
        output_path = Path(output_path_sstr)
        if output_path.resolve() != self.model_path.resolve():
            shutil.copy2(self.model_path, output_path)

        if mlflow.active_run():
            mlflow.log_artifact(str(output_path), artifact_path="lshif_model")

    def build_forest(
        self,
        embeddings_path: str,
        embeddings_dim: int,
        resume: bool = True,
    ):
        forest_build_start_time = time.perf_counter()
        self.meta.embedding_dim = embeddings_dim
        self.meta.num_rows = self._calc_mmap_row(embeddings_path, embeddings_dim)

        if mlflow.active_run():
            mlflow.log_params(
                {
                    "seed": self.meta.seed,
                    "max_depth": self.meta.max_depth,
                    "num_trees": self.meta.num_trees,
                }
            )

        checkpoint_path = Path(tempfile.gettempdir()) / "forest.resume"
        model_output_path = str(self.model_path)

        start_tree = 0
        if not os.path.exists(model_output_path):
            with open(model_output_path, "wb") as f:
                f.write(b" " * self.HEADER_SIZE)

        if resume and os.path.exists(checkpoint_path) and os.path.exists(model_output_path):
            with open(checkpoint_path, "r") as f:
                content = f.read().strip()
                if content:
                    start_tree = int(content)
                    print(f"Found checkpoint! Resuming generation from Tree {start_tree + 1}...")

        if start_tree >= self.meta.num_trees:
            print("Forest is already fully built according to the checkpoint.")
            return

        if start_tree == 0:
            print(f"Generating new forest signatures for {self.meta.num_trees} trees...")

        embeddings_mmap = np.memmap(
            embeddings_path,
            dtype="float32",
            mode="r",
            shape=(self._loaded_meta().num_rows, self._loaded_meta().embedding_dim),
        )
        signatures_mmap = np.memmap(
            model_output_path,
            dtype="uint64",
            mode="r+",
            offset=self.HEADER_SIZE,
            shape=(self.meta.num_trees, self.meta.num_rows),
        )

        for tree_idx in range(start_tree, self.meta.num_trees):
            tree_build_start_time = time.perf_counter()
            projection_matrix = self._get_hyperplanes(tree_idx)
            signatures = self._generate_signatures_for_tree(
                embeddings_mmap, projection_matrix, self.chunk_size
            )

            signatures.sort()

            signatures_mmap[tree_idx] = signatures
            signatures_mmap.flush()

            with open(checkpoint_path, "w") as f:
                f.write(str(tree_idx + 1))

            if mlflow.active_run():
                peak_mem, curr_mem = tracemalloc.get_traced_memory()
                mlflow.log_metrics(
                    {
                        "trees_built": tree_idx + 1,
                        "time_taken": tree_build_start_time - time.perf_counter(),
                        "peak_memory_usage": peak_mem,
                    },
                    step=tree_idx,
                )

        self.forest_mmap = signatures_mmap
        self._dump_meta(self.model_path)

        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        if mlflow.active_run():
            mlflow.log_metric("build_time", time.perf_counter() - forest_build_start_time)

    def calculate_baseline(
        self,
        signatures_path: str,
        output_path: str = "novelty_scores.npy",
    ) -> np.ndarray:
        print("Calculating baseline novelty scores...")

        if self.meta.num_rows is None:
            raise ValueError(
                "Number of rows isn't known. Please train or load the database first."
            )

        signatures_mmap = np.memmap(
            signatures_path,
            dtype="uint64",
            mode="r",
            offset=self.HEADER_SIZE,
            shape=(self.meta.num_trees, self.meta.num_rows),
        )

        total_path_lengths = np.zeros(self.meta.num_rows, dtype=np.float32)

        for tree_idx in range(self.meta.num_trees):
            print(f"Scoring Tree {tree_idx + 1}/{self.meta.num_trees}")
            signatures = signatures_mmap[tree_idx]
            path_lengths = self._build_single_tree(signatures)
            total_path_lengths += path_lengths

        average_path_lengths = total_path_lengths / self.meta.num_trees
        np.save(output_path, average_path_lengths)

        if mlflow.active_run():
            mlflow.log_metric("mean_baseline_path_length", float(np.mean(average_path_lengths)))
            mlflow.log_artifact(output_path, artifact_path="baselines")

        print(f"Baseline complete. Average baseline path lengths saved to {output_path}")
        return average_path_lengths

    def score(self, new_embedding: np.ndarray) -> float:
        if self.forest_mmap is None or self.projections is None:
            raise RuntimeError("Database signatures are not loaded.")

        total_depth = 0.0
        vector = new_embedding.reshape(1, -1)

        for tree_idx in range(self.meta.num_trees):
            new_sig = self._compute_simhash(vector, self.projections[tree_idx])[0]
            tree_db_sigs = self.forest_mmap[tree_idx]
            isolation_depth = self.meta.max_depth

            for depth in range(1, self.meta.max_depth):
                shift = np.uint64(64 - (depth * 4))
                new_prefix = new_sig >> shift

                search_target = new_prefix << shift
                idx = np.searchsorted(tree_db_sigs, search_target)

                exists = idx < self.meta.num_rows and (tree_db_sigs[idx] >> shift) == new_prefix

                if not exists:
                    isolation_depth = depth
                    break

            total_depth += isolation_depth

        return total_depth / self.meta.num_trees
