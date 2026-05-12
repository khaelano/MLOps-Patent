"""Path-length computation for LSH isolation trees (Algorithm 4).

Computes the combined compressed/uncompressed path length for each
query point, then normalises via the reference μ for the final
anomaly score.

Provides two scoring paths:
- ``score_tree`` — auto-selects vectorised (flat-array) scoring when
  available, falls back to per-row recursive Python traversal.
- ``_score_tree_flat`` — fully vectorised traversal using flat numpy
  trie arrays.  This is the fast path for production.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from patent.lshiforest._trie import trie_leaf_mu

# ---------------------------------------------------------------------------
# Vectorised leaf-μ (operates on arrays to avoid Python loops)
# ---------------------------------------------------------------------------

_GAMMA: float = 0.5772156649015329


def _leaf_mu_vec(sizes: np.ndarray, v: float) -> np.ndarray:
    """Vectorised μ correction for leaf nodes."""
    out = np.zeros(len(sizes), dtype=np.float64)
    mask_gt_v = sizes > v
    mask_le_v = (sizes > 1) & ~mask_gt_v
    ln_v = np.log(v)
    out[mask_gt_v] = (
        np.log(sizes[mask_gt_v].astype(np.float64)) + np.log(v - 1) + _GAMMA
    ) / ln_v - 0.5
    out[mask_le_v] = 1.0
    # sizes <= 1 stay 0.0 (default)
    return out


def _safe_ratio_vec(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Vectorised safe ratio."""
    out = np.empty_like(numer, dtype=np.float64)
    valid = denom > 0
    out[valid] = numer[valid].astype(np.float64) / denom[valid].astype(np.float64)
    out[~valid] = np.where(numer[~valid] > 0, numer[~valid].astype(np.float64), 0.0)
    return out


# ---------------------------------------------------------------------------
# Public entry point – auto-selects best implementation
# ---------------------------------------------------------------------------


def score_tree(
    trie: dict[str, Any],
    query_hashes: np.ndarray,
    mu: float,
    v: float,
    eta: float = 1.0,
    depth_limit: int | None = None,
) -> np.ndarray:
    """Score a batch of queries against a single isolation tree.

    Auto-selects vectorised flat-array scoring when the trie contains
    ``"flat"`` arrays, otherwise falls back to per-row recursive Python
    traversal.

    Parameters
    ----------
    trie : dict
        A trie dict as returned by :func:`patent.lshiforest._trie.build_trie`.
    query_hashes : np.ndarray of shape (n_queries, H), int64
        Pre-computed hash keys for each query at every level.
    mu : float
        Reference normalisation constant for this tree.
    v : float
        Branching factor estimate (used for leaf μ).
    eta : float
        Granularity parameter ∈ [0, 1].  1.0 = finest isolation
        (global anomalies); < 1.0 helps for local anomalies.
    depth_limit : int | None
        If set, caps the compressed depth during traversal.

    Returns
    -------
    np.ndarray of shape (n_queries,)  float64
        Raw path lengths (before 2^(-h/μ) normalisation).
    """
    flat = trie.get("flat")
    if flat is not None:
        return _score_tree_flat(flat, query_hashes, v=v, eta=eta, depth_limit=depth_limit)
    return _score_tree_recursive(trie, query_hashes, v=v, eta=eta, depth_limit=depth_limit)


# ---------------------------------------------------------------------------
# Vectorised flat-array scoring (fast path)
# ---------------------------------------------------------------------------


def _score_tree_flat(
    flat: dict[str, np.ndarray],
    query_hashes: np.ndarray,
    v: float,
    eta: float = 1.0,
    depth_limit: int | None = None,
) -> np.ndarray:
    """Score all queries through a flat-array trie in one vectorised pass.

    Algorithm: maintain an *active mask* of queries still traversing.
    Each iteration processes all active internal nodes, routes via
    numpy index-array lookups, and terminates queries that hit a leaf,
    missing child, or depth limit.
    """
    n_queries = len(query_hashes)
    n_hashes = query_hashes.shape[1]
    use_pow = eta != 1.0

    # Unpack flat arrays
    node_type = flat["node_type"]  # (N,) 0=leaf, 1=internal
    node_hash_idx = flat["node_hash_idx"]  # (N,)
    node_size = flat["node_size"]  # (N,)
    child_map = flat["child_map"]  # (N, B)
    max_branch = child_map.shape[1]

    # Working arrays, pre-allocated once
    final_depth = np.empty(n_queries, dtype=np.float64)
    c_depth = np.zeros(n_queries, dtype=np.int32)
    cur_node = np.zeros(n_queries, dtype=np.int32)  # all start at root (idx 0)
    active = np.ones(n_queries, dtype=bool)

    # Loop until all queries terminate
    while active.any():
        act_idx = np.flatnonzero(active)  # linear indices of active queries
        nodes = cur_node[act_idx]
        cd = c_depth[act_idx]

        # --- Determine node types for active queries ---
        types = node_type[nodes]
        at_internal = types == 1
        at_leaf = ~at_internal

        # --- Process leaf nodes ---
        if at_leaf.any():
            leaf_idx = act_idx[at_leaf]
            leaf_nodes = nodes[at_leaf]
            leaf_cd = cd[at_leaf]
            hi = node_hash_idx[leaf_nodes]
            ratio = _safe_ratio_vec(hi, leaf_cd)
            if use_pow:
                base = leaf_cd.astype(np.float64) * (ratio**eta)
            else:
                base = leaf_cd.astype(np.float64) * ratio
            sizes = node_size[leaf_nodes]
            final_depth[leaf_idx] = base + _leaf_mu_vec(sizes, v)
            active[leaf_idx] = False

        if not active.any():
            break

        # --- Process internal nodes ---
        # Refresh after leaf terminations
        act_idx = np.flatnonzero(active)
        nodes = cur_node[act_idx]
        cd = c_depth[act_idx]

        # Check depth_limit
        if depth_limit is not None:
            over_limit = cd > depth_limit
            if over_limit.any():
                limit_idx = act_idx[over_limit]
                limit_nodes = nodes[over_limit]
                limit_cd = cd[over_limit]
                hi = node_hash_idx[limit_nodes]
                ratio = _safe_ratio_vec(hi, limit_cd)
                if use_pow:
                    final_depth[limit_idx] = limit_cd.astype(np.float64) * (ratio**eta)
                else:
                    final_depth[limit_idx] = limit_cd.astype(np.float64) * ratio
                active[limit_idx] = False
                if not active.any():
                    break
                act_idx = np.flatnonzero(active)
                nodes = cur_node[act_idx]
                cd = c_depth[act_idx]

        # Check for ran-out-of-hash-keys (hash_idx >= n_hashes)
        hi = node_hash_idx[nodes]
        out_of_keys = hi >= n_hashes
        if out_of_keys.any():
            ook_idx = act_idx[out_of_keys]
            ook_nodes = nodes[out_of_keys]
            ook_cd = cd[out_of_keys]
            hi_ook = node_hash_idx[ook_nodes]
            ratio = _safe_ratio_vec(hi_ook, ook_cd)
            if use_pow:
                base = ook_cd.astype(np.float64) * (ratio**eta)
            else:
                base = ook_cd.astype(np.float64) * ratio
            sizes = node_size[ook_nodes]
            final_depth[ook_idx] = base + _leaf_mu_vec(sizes, v)
            active[ook_idx] = False
            if not active.any():
                break
            act_idx = np.flatnonzero(active)
            nodes = cur_node[act_idx]
            cd = c_depth[act_idx]
            hi = node_hash_idx[nodes]

        # --- Route: look up hash key for each query at its node's hash_idx ---
        keys = query_hashes[act_idx, hi]  # advanced indexing, element-wise
        # Clamp keys to valid range
        keys = np.clip(keys, 0, max_branch - 1)
        next_nodes = child_map[nodes, keys]  # (num_active,) → child index or -1

        # --- Check for missing children (early termination) ---
        missing = next_nodes == -1
        if missing.any():
            miss_idx = act_idx[missing]
            miss_cd = cd[missing] + 1
            miss_hi = hi[missing] + 1
            ratio = _safe_ratio_vec(miss_hi, miss_cd)
            if use_pow:
                final_depth[miss_idx] = miss_cd.astype(np.float64) * (ratio**eta)
            else:
                final_depth[miss_idx] = miss_cd.astype(np.float64) * ratio
            active[miss_idx] = False
            # Keep only non-missing for continuation
            keep = ~missing
            act_idx = act_idx[keep]
            nodes = nodes[keep]
            cd = cd[keep]
            next_nodes = next_nodes[keep]

        if len(act_idx) == 0:
            break

        # --- Advance: increment depth, move to next node ---
        c_depth[act_idx] = cd + 1
        cur_node[act_idx] = next_nodes

    return final_depth


# ---------------------------------------------------------------------------
# Per-row recursive Python traversal (fallback / backward-compat)
# ---------------------------------------------------------------------------


def _score_tree_recursive(
    trie: dict[str, Any],
    query_hashes: np.ndarray,
    v: float,
    eta: float = 1.0,
    depth_limit: int | None = None,
) -> np.ndarray:
    """Per-row recursive scoring — used when flat arrays are unavailable."""
    n_queries = len(query_hashes)
    depths = np.empty(n_queries, dtype=np.float64)
    root = trie["tree"]
    hv_len = query_hashes.shape[1]
    use_pow = eta != 1.0

    for i in range(n_queries):
        h = _path_length(
            root,
            query_hashes[i],
            c_depth=0,
            eta=eta,
            depth_limit=depth_limit,
            v=v,
            hv_len=hv_len,
            use_pow=use_pow,
        )
        depths[i] = h

    return depths


def _path_length(
    node: dict[str, Any],
    query_hv: np.ndarray,
    c_depth: int,
    eta: float,
    depth_limit: int | None,
    v: float,
    hv_len: int,
    use_pow: bool,
) -> float:
    """Recurse through one tree for a single query (Algorithm 4).

    Optimised hot-path:
    - ``hv_len`` is pre-computed ``len(query_hv)`` to avoid repeated calls.
    - ``use_pow`` skips ``ratio ** eta`` when eta == 1.0.
    """
    hash_idx = node["hash_idx"]

    # Depth limit → terminate
    if depth_limit is not None and c_depth > depth_limit:
        ratio = _safe_ratio_scalar(hash_idx, c_depth)
        if use_pow:
            return c_depth * (ratio**eta)
        return c_depth * ratio

    if node["type"] == "leaf":
        ratio = _safe_ratio_scalar(hash_idx, c_depth)
        if use_pow:
            base = c_depth * (ratio**eta)
        else:
            base = c_depth * ratio
        return base + trie_leaf_mu(node["size"], v)

    # Internal node – hash and route
    if hash_idx >= hv_len:
        ratio = _safe_ratio_scalar(hash_idx, c_depth)
        if use_pow:
            base = c_depth * (ratio**eta)
        else:
            base = c_depth * ratio
        return base + trie_leaf_mu(node["size"], v)

    key = int(query_hv[hash_idx])
    children = node["children"]

    if key in children:
        return _path_length(
            children[key],
            query_hv,
            c_depth + 1,
            eta,
            depth_limit,
            v,
            hv_len,
            use_pow,
        )
    # Early termination — no matching child
    new_c = c_depth + 1
    ratio = _safe_ratio_scalar(hash_idx + 1, new_c)
    if use_pow:
        return new_c * (ratio**eta)
    return new_c * ratio


def _safe_ratio_scalar(numerator: int, denominator: int) -> float:
    """Scalar ratio clamped for well-behaved exponentiation."""
    if denominator <= 0:
        return float(numerator) if numerator > 0 else 0.0
    return numerator / denominator


# ---------------------------------------------------------------------------
# Ensemble-level normalisation
# ---------------------------------------------------------------------------


def normalize_scores(raw_depths: np.ndarray, mus: list[float]) -> np.ndarray:
    """Convert raw per-tree path lengths to anomaly scores ∈ (0, 1].

    .. math::
       AS_x = \\frac{1}{t} \\sum_{i=1}^{t} 2^{-h_i(x) / \\mu_i}

    Parameters
    ----------
    raw_depths : (n_queries, n_trees) float64
    mus : list of per-tree reference μ values.

    Returns
    -------
    (n_queries,) float32 anomaly scores.
    """
    n_queries, n_trees = raw_depths.shape
    mus_arr = np.array(mus, dtype=np.float64).reshape(1, n_trees)
    safe_ratio = np.clip(raw_depths / mus_arr, 0.0, 50.0)
    scores = np.mean(2.0 ** (-safe_ratio), axis=1)
    return scores.astype(np.float64)


def rescale_scores(scores: np.ndarray) -> np.ndarray:
    """Rescale scores to [0, 1] via percentile ranking.

    LSHiForest scores on high-dimensional embeddings are naturally
    compressed — all vectors are near-equidistant so no single point
    can be isolated dramatically faster than others.  This function
    maps the empirical CDF to a uniform [0,1] scale so that:

    * The highest-scoring point maps to ≈1.0
    * The median maps to ≈0.5
    * Scores spread across the full [0,1] interval

    Parameters
    ----------
    scores : np.ndarray of shape (n,)  float64
        Raw anomaly scores from :meth:`LSHiForest.score`.

    Returns
    -------
    np.ndarray of shape (n,)  float64  — rescaled scores ∈ [0, 1].
    """
    # Rank via numpy argsort (avoids scipy dependency)
    n = len(scores)
    order = np.argsort(scores)
    ranks = np.empty(n, dtype=np.float64)
    # Handle ties: assign average rank to equal values
    i = 0
    while i < n:
        j = i
        while j < n and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-based average rank for ties
        ranks[order[i:j]] = avg_rank
        i = j
    return (ranks - 1.0) / (n - 1.0)
