from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class LSHFamily(ABC):
    @property
    @abstractmethod
    def path_dtype(self) -> np.dtype: ...

    @abstractmethod
    def value_range(self, max_depth: int) -> tuple[int, int]: ...

    @abstractmethod
    def generate_projections(self, embedding_dim: int, max_depth: int, seed: int) -> Any: ...

    @abstractmethod
    def compute_hashes(self, embeddings: np.ndarray, projections: Any) -> np.ndarray: ...

    @abstractmethod
    def encode_paths(self, hash_vals: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def common_prefix_depth(self, query_hv: np.ndarray, neighbor_hv: np.ndarray) -> np.ndarray: ...

    def mu(self, size: int, branching: int) -> float:
        if branching <= 1:
            return 0.0
        if size <= 1:
            return 0.0
        if 1 < size <= branching:
            return 1.0
        return (np.log(size) + np.log(branching - 1) + 0.5772156649015329) / np.log(
            branching
        ) - 0.5

    def height_limit(self, sample_size: int, branching: int) -> int:
        if branching <= 1 or sample_size <= 1:
            return 1
        gamma = 0.5772156649015329
        raw = (2.0 * np.log(sample_size) + gamma - np.log(2.0)) / np.log(branching) + 1.0
        return max(2, int(np.ceil(raw)))


class AngleLSHFamily(LSHFamily):
    def __init__(self) -> None:
        self._path_dtype = np.dtype(np.uint16)

    @property
    def path_dtype(self) -> np.dtype:
        return self._path_dtype

    def value_range(self, max_depth: int) -> tuple[int, int]:
        return (1, 2)

    def generate_projections(self, embedding_dim: int, max_depth: int, seed: int) -> Any:
        rng = np.random.default_rng(seed)
        return rng.standard_normal((max_depth, embedding_dim)).astype(np.float32)

    def compute_hashes(self, embeddings: np.ndarray, projections: Any) -> np.ndarray:
        raw = np.dot(projections, embeddings.T)
        return (raw > 0).astype(np.uint8)

    def encode_paths(self, hash_vals: np.ndarray) -> np.ndarray:
        max_depth, n_samples = hash_vals.shape
        if max_depth > 16:
            self._path_dtype = np.dtype(np.uint32)
        path_ints = np.zeros(n_samples, dtype=self._path_dtype)
        path_u64 = path_ints.astype(np.uint64)
        for i in range(max_depth):
            path_u64 = (path_u64 << 1) | hash_vals[i].astype(np.uint64)
        return path_u64.astype(self._path_dtype)

    def common_prefix_depth(self, query_hv: np.ndarray, neighbor_hv: np.ndarray) -> np.ndarray:
        max_depth = len(query_hv)
        diff = neighbor_hv != query_hv
        first_diff = np.argmax(diff, axis=1)
        all_same = ~diff.any(axis=1)
        return np.where(all_same, max_depth, first_diff)


class L2LSHFamily(LSHFamily):
    def __init__(self, bucket_width: float = 4.0) -> None:
        self.bucket_width = float(bucket_width)

    @property
    def path_dtype(self) -> np.dtype:
        raise RuntimeError("path_dtype requires max_depth; use _make_path_dtype(max_depth)")

    def _make_path_dtype(self, max_depth: int) -> np.dtype:
        return np.dtype((np.void, max_depth))

    def value_range(self, max_depth: int) -> tuple[int, int]:
        return (-128, 127)

    def generate_projections(self, embedding_dim: int, max_depth: int, seed: int) -> Any:
        rng = np.random.default_rng(seed)
        omega = rng.standard_normal((max_depth, embedding_dim)).astype(np.float32)
        bias = rng.uniform(0.0, self.bucket_width, size=max_depth).astype(np.float32)
        return np.asarray([self.bucket_width], dtype=np.float32), omega, bias

    def compute_hashes(self, embeddings: np.ndarray, projections: Any) -> np.ndarray:
        W_arr, omega, bias = projections
        W = float(W_arr[0])
        raw = np.dot(omega, embeddings.T)
        raw += bias[:, None]
        hashes = np.floor(raw / W).astype(np.int32)
        return np.clip(hashes, -128, 127).astype(np.int8)

    def encode_paths(self, hash_vals: np.ndarray) -> np.ndarray:
        max_depth, n_samples = hash_vals.shape
        as_uint8 = (hash_vals.astype(np.int16) + 128).astype(np.uint8)
        void_dt = np.dtype((np.void, max_depth))
        col_major = np.ascontiguousarray(as_uint8.T)
        return col_major.view(void_dt).reshape(-1)

    def common_prefix_depth(self, query_hv: np.ndarray, neighbor_hv: np.ndarray) -> np.ndarray:
        max_depth = len(query_hv)
        diff = neighbor_hv != query_hv
        first_diff = np.argmax(diff, axis=1)
        all_same = ~diff.any(axis=1)
        return np.where(all_same, max_depth, first_diff)


_FAMILY_REGISTRY: dict[str, type[LSHFamily]] = {
    "angle": AngleLSHFamily,
    "l2": L2LSHFamily,
}


def get_lsh_family(name: str, **kwargs: Any) -> LSHFamily:
    cls = _FAMILY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown LSH family: {name}. Choose from {list(_FAMILY_REGISTRY)}")
    return cls(**kwargs)
