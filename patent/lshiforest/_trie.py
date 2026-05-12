"""Recursive digital-trie construction with PATRICIA compression.

Each node is a plain dict so the tree is easy to inspect and serialize.
Leaf nodes record the subsample size that fell into the bucket.

Also provides a *flat-array* representation for vectorised traversal
during scoring (replaces per-query recursive Python calls with numpy
index-array operations).
"""

from __future__ import annotations

from typing import Any

import numpy as np

_GAMMA: float = 0.5772156649015329  # Euler-Mascheroni constant


def build_trie(
    hash_values: np.ndarray,
    subsample_size: int,
    height_limit: int,
    max_branch: int = 2,
) -> dict[str, Any]:
    """Build a PATRICIA-compressed digital trie from pre-computed hash values.

    Parameters
    ----------
    hash_values : np.ndarray of shape (n, H), int64
        Per-row hash keys for every level (0 .. H-1).  *H* should be
        at least the height limit.
    subsample_size : int
        The number of points in the subsample (n).
    height_limit : int
        Maximum uncompressed depth.
    max_branch : int
        Maximum branching factor (2 for Angle, 16 for L2 with buckets=16).

    Returns
    -------
    dict with keys ``tree`` *(the root node dict)*, ``flat`` *(flat arrays)*,
    ``v`` *(estimated branching factor)*, ``height_limit``.
    """
    n, _ = hash_values.shape
    root, child_counts = _build_recurse(hash_values, 0, n, hash_idx=0, height_limit=height_limit)

    # Estimate branching factor v = average children across internal nodes
    if child_counts:
        v = sum(child_counts) / len(child_counts)
    else:
        v = 2.0
    v = float(max(v, 2.0))

    # Build flat-array representation for vectorised scoring
    flat = _flatten_trie(root, max_branch)

    return {
        "tree": root,
        "flat": flat,
        "v": v,
        "height_limit": height_limit,
        "max_branch": max_branch,
    }


def _flatten_trie(
    root: dict[str, Any],
    max_branch: int,
) -> dict[str, np.ndarray]:
    """Convert a dict-based trie to flat numpy arrays.

    Returns dict with keys:
    - ``node_type``: (N,) uint8  (0=leaf, 1=internal)
    - ``node_hash_idx``: (N,) int32
    - ``node_size``: (N,) int32
    - ``child_map``: (N, max_branch) int32  (-1 = no child)
    """
    # First pass: count nodes and assign indices via DFS
    nodes: list[dict[str, Any]] = []
    _collect_nodes_dfs(root, nodes)

    num_nodes = len(nodes)
    node_type = np.empty(num_nodes, dtype=np.uint8)
    node_hash_idx = np.empty(num_nodes, dtype=np.int32)
    node_size = np.empty(num_nodes, dtype=np.int32)
    child_map = np.full((num_nodes, max_branch), -1, dtype=np.int32)

    for i, node in enumerate(nodes):
        node_type[i] = 0 if node["type"] == "leaf" else 1
        node_hash_idx[i] = node["hash_idx"]
        node_size[i] = node["size"]
        if node["type"] == "internal":
            for key, child_node in node["children"].items():
                child_idx = child_node["_flat_idx"]
                if 0 <= int(key) < max_branch:
                    child_map[i, int(key)] = child_idx

    return {
        "node_type": node_type,
        "node_hash_idx": node_hash_idx,
        "node_size": node_size,
        "child_map": child_map,
    }


def _collect_nodes_dfs(
    node: dict[str, Any],
    out: list[dict[str, Any]],
) -> None:
    """DFS walk assigning ``_flat_idx`` to each node."""
    node["_flat_idx"] = len(out)
    out.append(node)
    if node.get("type") == "internal":
        for child in node.get("children", {}).values():
            _collect_nodes_dfs(child, out)


def _build_recurse(
    hv: np.ndarray,
    start: int,
    end: int,
    hash_idx: int,
    height_limit: int,
) -> tuple[dict[str, Any], list[int]]:
    """Recursively build one sub-trie and collect child-count statistics.

    Parameters
    ----------
    hv : (n, H) int64 array of hash values.
    start, end : row slice [start:end) of the *current* node's points.
    hash_idx : which hash-function index we are using *at this node*.
    height_limit : cap on ``hash_idx`` (actual depth may exceed it
        because PATRICIA compression skips levels).

    Returns
    -------
    node : dict
    child_counts : list of ints (one per internal node in the
        sub-tree) recording how many children each internal node had.
    """
    n = end - start
    max_hash = hv.shape[1]

    # ------------------------------------------------------------------
    # Leaf / height-limit checks
    # ------------------------------------------------------------------
    if n == 0:
        return {"type": "leaf", "hash_idx": hash_idx, "size": 0}, []
    if n == 1 or hash_idx >= height_limit:
        return {"type": "leaf", "hash_idx": hash_idx, "size": n}, []

    # ------------------------------------------------------------------
    # Compute hash keys for the current level
    # ------------------------------------------------------------------
    if hash_idx >= max_hash:
        return {"type": "leaf", "hash_idx": hash_idx, "size": n}, []

    keys = hv[start:end, hash_idx]
    unique_keys = np.unique(keys)

    if len(unique_keys) == 1:
        # PATRICIA compression – single branch ⇒ skip this level
        return _build_recurse(hv, start, end, hash_idx + 1, height_limit)

    # ------------------------------------------------------------------
    # Internal node – partition and recurse
    # ------------------------------------------------------------------
    children: dict[int, dict[str, Any]] = {}
    all_child_counts: list[int] = [len(unique_keys)]

    # Sort indices by key for efficient contiguous slicing
    order = np.argsort(keys)
    sorted_keys = keys[order]
    # Map back so the original start..end window is correctly addressed
    global_order = np.arange(start, end)[order]

    pos = 0
    while pos < n:
        key = int(sorted_keys[pos])
        # Find run end
        run_end = pos
        while run_end < n and sorted_keys[run_end] == key:
            run_end += 1
        run_size = run_end - pos

        # Extract the contiguous slice of rows (via original indices)
        rows = global_order[pos:run_end]
        # Build a local contiguous view of hash values
        sub_hv = hv[rows]

        child_node, sub_counts = _build_recurse(sub_hv, 0, run_size, hash_idx + 1, height_limit)
        children[key] = child_node
        all_child_counts.extend(sub_counts)
        pos = run_end

    return {
        "type": "internal",
        "hash_idx": hash_idx,
        "children": children,
        "size": n,
    }, all_child_counts


def trie_leaf_mu(leaf_size: int, v: float) -> float:
    """μ for a single leaf's point count (used in path-length formula)."""
    if leaf_size <= 1:
        return 0.0
    if leaf_size <= v:
        return 1.0
    import math

    ln_v = math.log(v)
    return (math.log(leaf_size) + math.log(v - 1) + _GAMMA) / ln_v - 0.5
