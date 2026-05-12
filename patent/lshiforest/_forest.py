"""LSHiForest ensemble — training, scoring, save/load.

Implements the LSHiForest algorithm from:
  Xiang et al., "LSHiForest: A Generic Framework for Fast Tree
  Isolation Based Ensemble Anomaly Analysis", ICDM 2017.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

from patent.lshiforest._family import AngleFamily, L2Family, LSHFamily
from patent.lshiforest._scoring import score_tree
from patent.lshiforest._serialize import load_forest, save_forest
from patent.lshiforest._trie import build_trie

_logger = logging.getLogger(__name__)


def _make_family(name: str) -> LSHFamily:
    """Factory for LSH family objects."""
    name_lower = name.lower()
    if name_lower == "angle":
        return AngleFamily()
    elif name_lower in ("l2", "l2sh"):
        return L2Family()
    else:
        raise ValueError(f"Unknown LSH family: {name!r}")


class LSHiForest:
    """Locality-Sensitive Hashing Isolation Forest.

    Parameters
    ----------
    n_trees : int
        Number of isolation trees in the ensemble (default 200).
    max_depth : int
        Maximum tree depth (default 21).
    seed : int
        Master random seed (default 42).
    family : str
        LSH family name: ``"angle"`` or ``"l2"``.
    eta : float
        Granularity parameter ∈ [0, 1].  0.0 for local / patch-level
        anomalies (default, recommended for diverse text embeddings);
        1.0 for global anomalies only.
    """

    def __init__(
        self,
        n_trees: int = 200,
        max_depth: int = 21,
        seed: int = 42,
        family: str = "l2",
        eta: float = 0.0,
        num_trees: int | None = None,  # alias for n_trees
    ) -> None:
        if num_trees is not None:
            n_trees = num_trees
        self._n_trees = n_trees
        self._max_depth = max_depth
        self._seed = seed
        self._eta = eta
        self._family = _make_family(family)
        self._family_name = family.lower()

        # Populated during fit
        self._trees: list[dict[str, Any]] = []
        self._tree_params: list[dict[str, Any]] = []
        self._subsample_sizes_list: list[int] = []
        self._tree_mus_list: list[float] = []
        self._fitted: bool = False
        self._n_hashes: int = max(max_depth * 2, 42)  # hash functions per tree
        self._tempfile: Path | None = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def n_trees(self) -> int:
        return self._n_trees

    @property
    def max_depth(self) -> int:
        return self._max_depth

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def family_name(self) -> str:
        return self._family_name

    @property
    def _subsample_sizes(self) -> list[int]:
        return self._subsample_sizes_list

    @property
    def _tree_mus(self) -> list[float]:
        return self._tree_mus_list

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> None:
        """Build the forest on *X*.

        Parameters
        ----------
        X : np.ndarray of shape (n, d), float32
        """
        n, embedding_dim = X.shape
        if n == 0:
            raise ValueError("Cannot fit on empty dataset")

        rng = np.random.default_rng(self._seed)

        self._trees = []
        self._tree_params = []
        self._subsample_sizes_list = []
        self._tree_mus_list = []

        # ── trace helpers ──────────────────────────────────────────
        log_every = max(1, self._n_trees // 10 or 1)
        t_start = time.perf_counter()

        _logger.info(
            "LSHiForest.fit | n=%d d=%d trees=%d max_depth=%d family=%s",
            n,
            embedding_dim,
            self._n_trees,
            self._max_depth,
            self._family_name,
        )

        for i in range(self._n_trees):
            # Variable subsample size: 2^k for k ∼ U[6, 10]
            k = int(rng.integers(6, 11))  # [6, 10] inclusive
            subsample_size = min(2**k, n)
            self._subsample_sizes_list.append(subsample_size)

            # Draw subsample
            indices = rng.choice(n, size=subsample_size, replace=False)
            X_sub = X[indices]

            # Generate hash functions for this tree
            tree_seed = int(rng.integers(0, 2**31 - 1))
            params = self._family.generate(embedding_dim, self._n_hashes, tree_seed)

            # Compute height limit
            hl_theory = self._family.height_limit(subsample_size, 2.0)  # initial v=2 estimate
            height_limit = min(int(hl_theory) + 1, self._max_depth)

            # Hash the subsample
            hash_values = self._family.hash_batch(X_sub, params)
            max_branch = self._family.MAX_BRANCH

            # Build trie
            trie = build_trie(hash_values, subsample_size, height_limit, max_branch)
            v_est = trie["v"]

            # Recompute height limit with estimated v, then recompute trie
            hl_refined = self._family.height_limit(subsample_size, v_est)
            height_limit = min(int(hl_refined) + 1, self._max_depth)
            if height_limit != trie["height_limit"]:
                trie = build_trie(hash_values, subsample_size, height_limit, max_branch)
                v_est = trie["v"]

            # Compute μ normalization
            mu = self._family.mu(subsample_size, v_est)

            self._trees.append(trie)
            self._tree_params.append(
                {
                    "projections": params if isinstance(params, np.ndarray) else params[0],
                    "offsets": params[1] if isinstance(params, tuple) else None,
                    "v": v_est,
                    "mu": mu,
                    "subsample_size": subsample_size,
                    "seed": tree_seed,
                }
            )
            self._tree_mus_list.append(mu)

            # ── trace ───────────────────────────────────────────
            if (i + 1) % log_every == 0 or i == self._n_trees - 1:
                elapsed = time.perf_counter() - t_start
                _logger.info(
                    "LSHiForest.fit | tree %d/%d  size=%d  v=%.1f  mu=%.2f  "
                    "trees/s=%.1f  elapsed=%.1fs",
                    i + 1,
                    self._n_trees,
                    subsample_size,
                    v_est,
                    mu,
                    (i + 1) / elapsed if elapsed > 0 else 0,
                    elapsed,
                )
            else:
                _logger.debug(
                    "LSHiForest.fit | tree %d/%d  size=%d  v=%.1f  mu=%.2f",
                    i + 1,
                    self._n_trees,
                    subsample_size,
                    v_est,
                    mu,
                )

        self._fitted = True
        _logger.info(
            "LSHiForest.fit | DONE  %d trees in %.1fs  (%.1f trees/s)",
            self._n_trees,
            time.perf_counter() - t_start,
            self._n_trees / (time.perf_counter() - t_start),
        )

    def fit_parquet(self, paths: list[str]) -> None:
        """Load embeddings from Parquet via a temporary memmap and fit.

        Uses memmap so that :meth:`fit` only materialises the small
        subsamples (~64–1024 rows each) rather than the entire dataset.

        Parameters
        ----------
        paths : list of str
            Paths to Parquet files with an ``"embedding"`` column.
        """
        import shutil

        from patent.config import project_tempdir
        from patent.utils import convert_parquet_to_memmap

        tmpdir = project_tempdir()
        mmap_path = tmpdir / "fit_data.mmap"
        try:
            embedding_dim, total_rows = convert_parquet_to_memmap(paths, str(mmap_path))
            if total_rows == 0 or embedding_dim == 0:
                raise ValueError("No embeddings found in provided files")
            X = np.memmap(
                str(mmap_path),
                dtype=np.float32,
                mode="r",
                shape=(total_rows, embedding_dim),
            )
            self.fit(X)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        X: np.ndarray,
        normalize: bool = True,
        n_workers: int | None = None,
        tree_batch: int = 10,
    ) -> np.ndarray:
        """Compute anomaly scores for *X*.

        Memory-safe incremental accumulation — avoids allocating a
        (n_queries, n_trees) matrix.  Hashes *tree_batch* trees at once
        via a single stacked BLAS matmul to reduce per-tree overhead.

        When *n_workers* > 1, scores multiple tree-batches in parallel
        using a thread pool (numpy releases the GIL during BLAS and
        most array operations).

        Parameters
        ----------
        X : np.ndarray of shape (n, d), float32
        normalize : bool
            If True (default), return scores ∈ (0, 1] via 2^(-h/μ).
            If False, return raw average path lengths.
        n_workers : int | None
            Number of threads for parallel tree scoring (default 1).
        tree_batch : int
            Number of trees to hash simultaneously (default 10).

        Returns
        -------
        np.ndarray of shape (n,) float64
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() before score().")

        n_queries = X.shape[0]
        n_trees = len(self._trees)

        if n_workers is not None and n_workers > 1:
            return self._score_parallel(
                X, normalize=normalize, n_workers=n_workers, tree_batch=tree_batch
            )

        _logger.debug(
            "LSHiForest.score | queries=%d  trees=%d  tree_batch=%d  normalize=%s",
            n_queries,
            n_trees,
            tree_batch,
            normalize,
        )

        # Pre-pack params for batched hashing
        proj_list: list = []
        for p in self._tree_params:
            if p["offsets"] is not None:
                proj_list.append((p["projections"], p["offsets"]))
            else:
                proj_list.append(p["projections"])

        n_batches = (n_trees + tree_batch - 1) // tree_batch
        log_every = max(1, n_batches // 5 or 1)
        t_start = time.perf_counter()

        if normalize:
            scores = np.zeros(n_queries, dtype=np.float64)
            for bi, batch_start in enumerate(range(0, n_trees, tree_batch)):
                batch_end = min(batch_start + tree_batch, n_trees)
                batch_indices = range(batch_start, batch_end)
                batch_params = [proj_list[i] for i in batch_indices]
                batch_size = batch_end - batch_start

                batch_hashes = self._family.hash_batch_trees(X, batch_params)

                for j, i in enumerate(batch_indices):
                    query_hashes = batch_hashes[:, j, :]
                    tree_depths = score_tree(
                        self._trees[i],
                        query_hashes,
                        mu=self._tree_params[i]["mu"],
                        v=self._tree_params[i]["v"],
                        eta=self._eta,
                    )
                    mu_i = self._tree_mus_list[i]
                    safe_ratio = np.clip(tree_depths / mu_i, 0.0, 50.0)
                    scores += 2.0 ** (-safe_ratio)

                # ── trace ───────────────────────────────────────
                if (bi + 1) % log_every == 0 or bi == n_batches - 1:
                    elapsed = time.perf_counter() - t_start
                    _logger.info(
                        "LSHiForest.score | batch %d/%d  trees %d-%d (%d)  "
                        "elapsed=%.1fs  throughput=%.0f q/s",
                        bi + 1,
                        n_batches,
                        batch_start + 1,
                        batch_end,
                        batch_size,
                        elapsed,
                        n_queries * (batch_end) / elapsed if elapsed > 0 else 0,
                    )
            return scores / n_trees
        else:
            raw_sum = np.zeros(n_queries, dtype=np.float64)
            for bi, batch_start in enumerate(range(0, n_trees, tree_batch)):
                batch_end = min(batch_start + tree_batch, n_trees)
                batch_indices = range(batch_start, batch_end)
                batch_params = [proj_list[i] for i in batch_indices]
                batch_size = batch_end - batch_start

                batch_hashes = self._family.hash_batch_trees(X, batch_params)

                for j, i in enumerate(batch_indices):
                    query_hashes = batch_hashes[:, j, :]
                    tree_depths = score_tree(
                        self._trees[i],
                        query_hashes,
                        mu=self._tree_params[i]["mu"],
                        v=self._tree_params[i]["v"],
                        eta=self._eta,
                    )
                    raw_sum += tree_depths

                if (bi + 1) % log_every == 0 or bi == n_batches - 1:
                    elapsed = time.perf_counter() - t_start
                    _logger.info(
                        "LSHiForest.score | batch %d/%d  trees %d-%d (%d)  "
                        "elapsed=%.1fs  throughput=%.0f q/s",
                        bi + 1,
                        n_batches,
                        batch_start + 1,
                        batch_end,
                        batch_size,
                        elapsed,
                        n_queries * (batch_end) / elapsed if elapsed > 0 else 0,
                    )
            return raw_sum / n_trees

    def _score_parallel(
        self,
        X: np.ndarray,
        normalize: bool,
        n_workers: int,
        tree_batch: int,
    ) -> np.ndarray:
        """Parallel tree scoring via thread pool.

        Each worker processes one tree-batch (a group of *tree_batch*
        trees hashed together).  Results are summed into the output.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        n_queries = X.shape[0]
        n_trees = len(self._trees)

        _logger.debug(
            "LSHiForest._score_parallel | queries=%d  trees=%d  "
            "workers=%d  tree_batch=%d  normalize=%s",
            n_queries,
            n_trees,
            n_workers,
            tree_batch,
            normalize,
        )

        # Pre-pack params
        proj_list: list = []
        for p in self._tree_params:
            if p["offsets"] is not None:
                proj_list.append((p["projections"], p["offsets"]))
            else:
                proj_list.append(p["projections"])

        if normalize:
            scores = np.zeros(n_queries, dtype=np.float64)
        else:
            scores = np.zeros(n_queries, dtype=np.float64)

        def _process_batch(batch_start: int, batch_end: int) -> np.ndarray:
            """Process one tree-batch, return partial sums."""
            local_sum = np.zeros(n_queries, dtype=np.float64)
            batch_indices = range(batch_start, batch_end)
            batch_params = [proj_list[i] for i in batch_indices]

            batch_hashes = self._family.hash_batch_trees(X, batch_params)

            for j, i in enumerate(batch_indices):
                query_hashes = batch_hashes[:, j, :]
                tree_depths = score_tree(
                    self._trees[i],
                    query_hashes,
                    mu=self._tree_params[i]["mu"],
                    v=self._tree_params[i]["v"],
                    eta=self._eta,
                )
                if normalize:
                    mu_i = self._tree_mus_list[i]
                    safe_ratio = np.clip(tree_depths / mu_i, 0.0, 50.0)
                    local_sum += 2.0 ** (-safe_ratio)
                else:
                    local_sum += tree_depths
            return local_sum

        # Build batch list and submit
        batch_specs: list[tuple[int, int]] = []
        for bs in range(0, n_trees, tree_batch):
            be = min(bs + tree_batch, n_trees)
            batch_specs.append((bs, be))

        n_batches = len(batch_specs)
        log_every = max(1, n_batches // 5 or 1)
        t_start = time.perf_counter()
        completed = 0

        _logger.info(
            "LSHiForest._score_parallel | submitting %d batches to %d workers",
            n_batches,
            n_workers,
        )

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_map = {
                executor.submit(_process_batch, bs, be): (bs, be) for bs, be in batch_specs
            }
            for fut in as_completed(future_map):
                bs, be = future_map[fut]
                scores += fut.result()
                completed += 1
                # ── trace ───────────────────────────────────────
                if completed % log_every == 0 or completed == n_batches:
                    elapsed = time.perf_counter() - t_start
                    trees_done = min(completed * tree_batch, n_trees)
                    _logger.info(
                        "LSHiForest._score_parallel | batch %d/%d  trees ~%d/%d  elapsed=%.1fs",
                        completed,
                        n_batches,
                        trees_done,
                        n_trees,
                        elapsed,
                    )

        if normalize:
            return scores / n_trees
        return scores / n_trees

    def score_chunked(
        self,
        mmap: np.ndarray,
        total_rows: int,
        chunk_size: int = 100_000,
        n_workers: int | None = None,
        tree_batch: int = 10,
    ) -> np.ndarray:
        """Memory-safe chunked scoring over a memmap.

        Parameters
        ----------
        mmap : np.ndarray (memmap)
            Read-only memmap of shape (total_rows, embedding_dim).
        total_rows : int
            Number of rows in the memmap.
        chunk_size : int
            Rows per chunk (default 100k).
        n_workers : int | None
            Number of threads for parallel tree scoring.
        tree_batch : int
            Number of trees to hash simultaneously.

        Returns
        -------
        np.ndarray of shape (total_rows,) float32
        """
        n_chunks = (total_rows + chunk_size - 1) // chunk_size
        log_every = max(1, n_chunks // 10 or 1)
        t_start = time.perf_counter()

        _logger.info(
            "LSHiForest.score_chunked | rows=%d  chunk_size=%d  chunks=%d  "
            "workers=%s  tree_batch=%d",
            total_rows,
            chunk_size,
            n_chunks,
            n_workers,
            tree_batch,
        )

        scores = np.empty(total_rows, dtype=np.float32)
        for ci, start in enumerate(range(0, total_rows, chunk_size)):
            end = min(start + chunk_size, total_rows)
            batch = np.asarray(mmap[start:end])
            scores[start:end] = self.score(batch, n_workers=n_workers, tree_batch=tree_batch)

            # ── trace ───────────────────────────────────────────
            if (ci + 1) % log_every == 0 or ci == n_chunks - 1:
                elapsed = time.perf_counter() - t_start
                _logger.info(
                    "LSHiForest.score_chunked | chunk %d/%d  "
                    "rows %d-%d (%d)  elapsed=%.1fs  "
                    "throughput=%.0f q/s",
                    ci + 1,
                    n_chunks,
                    start,
                    end,
                    end - start,
                    elapsed,
                    (end) / elapsed if elapsed > 0 else 0,
                )

        return scores

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the fitted model to *path*."""
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted model.")

        model_params = {
            "n_trees": self._n_trees,
            "max_depth": self._max_depth,
            "n_hashes": self._n_hashes,
            "seed": self._seed,
            "family": self._family_name,
            "eta": self._eta,
            "max_branch": self._family.MAX_BRANCH,
        }
        save_forest(path, self._trees, self._tree_params, model_params)

    @classmethod
    def load(cls, path: str | Path) -> "LSHiForest":
        """Load a fitted model from *path*."""
        trees, tree_params, model_params = load_forest(path)

        model = cls.__new__(cls)
        model._n_trees = model_params["n_trees"]
        model._max_depth = model_params["max_depth"]
        model._seed = model_params["seed"]
        model._eta = model_params.get("eta", 0.0)
        family_name = model_params["family"]
        model._family = _make_family(family_name)
        model._family_name = family_name
        model._n_hashes = model_params.get("n_hashes", model_params["max_depth"])
        model._trees = trees
        model._tree_params = tree_params
        model._subsample_sizes_list = [tp["subsample_size"] for tp in tree_params]
        model._tree_mus_list = [tp["mu"] for tp in tree_params]
        model._fitted = True
        model._tempfile = None

        # Regenerate flat arrays if missing (backward-compat with old saves)
        max_branch = model_params.get("max_branch", model._family.MAX_BRANCH)
        from patent.lshiforest._trie import _flatten_trie

        for tree in model._trees:
            if "flat" not in tree:
                tree["flat"] = _flatten_trie(tree["tree"], max_branch)
                tree["max_branch"] = max_branch

        return model
