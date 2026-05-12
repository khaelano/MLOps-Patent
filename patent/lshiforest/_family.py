"""LSH family abstractions for LSHiForest.

Each family defines how to project vectors onto hash keys.
AngleFamily: concatenated random hyperplanes (multi-fork).
L2Family: p-stable Gaussian projection + uniform offset (variable v).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any

import numpy as np

_GAMMA: float = 0.5772156649015329  # Euler-Mascheroni constant


class LSHFamily(ABC):
    """Base class for locality-sensitive hash families."""

    @abstractmethod
    def generate(self, embedding_dim: int, n_levels: int, seed: int) -> Any:
        """Generate parameters for *n_levels* independent hash levels.

        Returns an opaque object consumed by :meth:`hash_batch`.
        """
        ...

    @abstractmethod
    def hash_batch(self, X: np.ndarray, params: Any) -> np.ndarray:
        """Hash a batch of vectors.

        Parameters
        ----------
        X : np.ndarray of shape (n, d), float32
        params : output of :meth:`generate`

        Returns
        -------
        np.ndarray of shape (n, n_levels), int64
        """
        ...

    @abstractmethod
    def hash_batch_trees(self, X: np.ndarray, params_list: list[Any]) -> np.ndarray:
        """Hash *X* against multiple independent tree parameter sets.

        Stacks projections into a single BLAS call for throughput.

        Returns
        -------
        np.ndarray of shape (n_queries, n_trees, n_levels), int64
        """
        ...

    @property
    @abstractmethod
    def MAX_BRANCH(self) -> int:  # noqa: N802
        """Maximum possible branching factor for flat-array allocation."""
        ...

    # ------------------------------------------------------------------
    # Shared formulas (from the paper, §3.2 / eq. 1 and §4.1)
    # ------------------------------------------------------------------

    @staticmethod
    def height_limit(subsample_size: int, v: float = 2.0) -> float:
        """Theoretical upper bound on digital-trie height.

        .. math::
           E(H) \\approx \\frac{2 \\ln \\psi}{\\ln v}
                    + \\frac{\\gamma - \\ln 2}{\\ln v} + 1
        """
        if subsample_size <= v:
            return float(math.ceil(math.log2(max(subsample_size, 2))))
        ln_v = math.log(v)
        h = (2.0 * math.log(subsample_size) + (_GAMMA - math.log(2))) / ln_v + 1.0
        return h

    @staticmethod
    def mu(subsample_size: int, v: float) -> float:
        """Reference path length – average successful search in PATRICIA trie.

        .. math::
           \\mu(\\psi) =
           \\begin{cases}
             \\frac{\\ln \\psi + \\ln(v-1) + \\gamma}{\\ln v} - \\frac12, & \\psi > v \\\\
             1, & 1 < \\psi \\leq v \\\\
             0, & \\text{otherwise}
           \\end{cases}
        """
        if subsample_size <= 1:
            return 0.0
        if subsample_size <= v:
            return 1.0
        ln_v = math.log(v)
        return (math.log(subsample_size) + math.log(v - 1) + _GAMMA) / ln_v - 0.5


class AngleFamily(LSHFamily):
    """Angle-based LSH with *alpha*-bit concatenation per level.

    Each tree level uses *alpha* independent random hyperplanes whose
    binary outputs are concatenated into a ``2^alpha``-way integer key.
    Without concatenation (alpha=1) the tree is always full binary and
    cannot isolate anything — every query finds both children.  With
    alpha ≥ 3, most keys have no children → real early-termination
    isolation occurs.

    Parameters
    ----------
    alpha : int
        Number of hyperplanes concatenated per tree level (default 3,
        giving 2³ = 8 possible keys per level).
    """

    V: float = 2.0

    def __init__(self, alpha: int = 3) -> None:
        self.alpha = alpha

    @property
    def MAX_BRANCH(self) -> int:  # noqa: N802
        return 2**self.alpha

    def generate(self, embedding_dim: int, n_levels: int, seed: int) -> np.ndarray:
        """Generate ``n_levels * alpha`` random projection vectors."""
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n_levels * self.alpha, embedding_dim)).astype(np.float32)

    def hash_batch(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Hash *X* → ``(n, n_levels)`` int64 keys in [0, 2^alpha)."""
        # params: (n_levels * alpha, d)
        dots = X @ params.T  # (n, n_levels * alpha)
        n, total = dots.shape
        n_levels = total // self.alpha
        bits = dots >= 0.0  # (n, n_levels * alpha)
        # Reshape to (n, n_levels, alpha) and pack bits → integer key
        bits_reshaped = bits.reshape(n, n_levels, self.alpha)
        powers = 2 ** np.arange(self.alpha, dtype=np.int64)
        return (bits_reshaped.astype(np.int64) * powers).sum(axis=2)

    def hash_batch_trees(self, X: np.ndarray, params_list: list[np.ndarray]) -> np.ndarray:
        """Hash *X* against multiple trees in one stacked matmul.

        Returns ``(n_queries, n_trees, n_levels)`` int64.
        """
        n_trees = len(params_list)
        total_proj, d = params_list[0].shape
        n_levels = total_proj // self.alpha
        # Stack: (T, L*alpha, d) → (T*L*alpha, d)
        stacked = np.stack(params_list, axis=0).reshape(n_trees * total_proj, d)
        dots = X @ stacked.T  # (n, T*L*alpha)
        bits = dots >= 0.0
        bits_reshaped = bits.reshape(X.shape[0], n_trees, n_levels, self.alpha)
        powers = 2 ** np.arange(self.alpha, dtype=np.int64)
        return (bits_reshaped.astype(np.int64) * powers).sum(axis=3)


class L2Family(LSHFamily):
    """ℓ₂ p-stable LSH: :math:`f(x) = \\lfloor (\\omega\\cdot x + \\omega_0) / W \\rfloor`.

    * ωᵢ ∼ 𝒩(0, 1), ω₀ ∼ U[0, W)
    * Variable branching factor **v** (estimated from the trie).
    * W controls bucket granularity (default 1.0).
    * *buckets* bounds hash values to [0, buckets) via a modulo.
    """

    def __init__(self, W: float = 1.0, buckets: int = 16) -> None:
        self.W = W
        self.buckets = buckets

    @property
    def MAX_BRANCH(self) -> int:  # noqa: N802
        return self.buckets

    def generate(
        self, embedding_dim: int, n_levels: int, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        projections = rng.standard_normal((n_levels, embedding_dim)).astype(np.float32)
        offsets = rng.uniform(0.0, self.W, size=n_levels).astype(np.float32)
        return (projections, offsets)

    def hash_batch(self, X: np.ndarray, params: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        projections, offsets = params
        dots = X @ projections.T
        shifted = dots + offsets
        raw = np.floor(shifted / self.W).astype(np.int64)
        return raw % self.buckets

    def hash_batch_trees(
        self, X: np.ndarray, params_list: list[tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        """Hash *X* against multiple trees in one stacked matmul.

        Returns ``(n_queries, n_trees, H)`` int64.
        """
        n_trees = len(params_list)
        H, d = params_list[0][0].shape

        stacked_proj = np.stack([p[0] for p in params_list], axis=0).reshape(n_trees * H, d)
        stacked_offsets = np.stack([p[1] for p in params_list], axis=0).reshape(-1)

        dots = X @ stacked_proj.T
        shifted = dots + stacked_offsets
        raw = np.floor(shifted / self.W).astype(np.int64)
        hashed = raw % self.buckets
        return hashed.reshape(X.shape[0], n_trees, H)
