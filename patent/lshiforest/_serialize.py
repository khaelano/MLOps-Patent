"""Serialization for LSHiForest models.

Uses NPZ archives for numerical data (projection matrices) and
msgpack for structured metadata (tree dicts, params).  No pickle.
"""

from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Any, Literal

import numpy as np


def save_forest(
    path: str | Path,
    trees: list[dict[str, Any]],
    tree_params: list[dict[str, Any]],
    model_params: dict[str, Any],
) -> None:
    """Save a complete LSHiForest ensemble.

    Parameters
    ----------
    path : Path
        Output ``.lshif`` file path (a zip-based NPZ).
    trees : list of dict
        Trie dicts from ``build_trie``, one per tree.
    tree_params : list of dict
        Per-tree metadata with keys: ``projections`` (np.ndarray),
        ``offsets`` (np.ndarray | None), ``v`` (float), ``mu`` (float),
        ``subsample_size`` (int), ``seed`` (int).
    model_params : dict
        Top-level model params: ``n_trees``, ``max_depth``, ``seed``,
        ``family``, ``eta``.
    """
    npz_data: dict[str, np.ndarray] = {}
    meta_trees: list[dict[str, Any]] = []

    for i, (tree, tp) in enumerate(zip(trees, tree_params)):
        prefix = f"tree_{i}"
        npz_data[f"{prefix}_proj"] = tp["projections"]
        if tp.get("offsets") is not None:
            npz_data[f"{prefix}_offsets"] = tp["offsets"]

        # Serialize trie structure to JSON-safe dict, storing hash keys
        # as lists of ints (JSON doesn't allow int keys).
        meta_trees.append(
            {
                "tree": _serialize_node(tree["tree"]),
                "v": float(tree["v"]),
                "mu": float(tp["mu"]),
                "subsample_size": int(tp["subsample_size"]),
                "seed": int(tp.get("seed", 0)),
            }
        )

    meta_json = json.dumps(
        {
            "model": model_params,
            "trees": meta_trees,
        }
    )
    npz_data["meta"] = np.array(meta_json, dtype="S")

    # Use np.savez_compressed to get a zip but with numpy compression
    with open(path, "wb") as f:
        _save_npz(f, npz_data)


def _save_npz(f, arrays: dict[str, np.ndarray]) -> None:
    """Write NPZ-compatible file (zip of .npy entries)."""
    import io
    import zipfile

    with zipfile.ZipFile(f, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, arr in arrays.items():
            buf = io.BytesIO()
            _write_npy(buf, arr)
            zf.writestr(f"{name}.npy", buf.getvalue())


def _write_npy(buf, arr: np.ndarray) -> None:
    """Write a single .npy file to a byte buffer."""
    dtype = np.dtype(arr.dtype)
    if dtype.hasobject:
        raise ValueError("Object arrays are not supported")

    # Header
    header_dict = {"descr": dtype.str, "fortran_order": False, "shape": arr.shape}
    header_bytes = json.dumps(header_dict).encode("ascii")
    # Pad to 16-byte alignment
    header_len = len(header_bytes)
    if header_len % 16 != 0:
        padding = 16 - (header_len % 16)
    else:
        padding = 0
    total_header = header_bytes + b" " * padding + b"\n"

    buf.write(b"\x93NUMPY")
    buf.write(struct.pack("<B", 1))  # major version
    buf.write(struct.pack("<B", 0))  # minor version
    buf.write(struct.pack("<H", len(total_header)))
    buf.write(total_header)
    buf.write(arr.tobytes())


def load_forest(
    path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load a complete LSHiForest ensemble.

    Returns
    -------
    trees : list of trie dicts
    tree_params : list of per-tree param dicts
    model_params : top-level model params dict
    """
    npz_data = _load_npz(path)

    meta_raw = npz_data["meta"]
    if isinstance(meta_raw, np.ndarray):
        meta_json = (
            meta_raw.item().decode("utf-8") if meta_raw.dtype.kind == "S" else str(meta_raw.item())
        )
    else:
        meta_json = meta_raw.decode("utf-8")

    meta = json.loads(meta_json)
    model_params = meta["model"]
    meta_trees = meta["trees"]

    trees: list[dict[str, Any]] = []
    tree_params: list[dict[str, Any]] = []

    for i, mt in enumerate(meta_trees):
        prefix = f"tree_{i}"
        proj = npz_data[f"{prefix}_proj"]
        offsets_key = f"{prefix}_offsets"
        offsets = npz_data.get(offsets_key)

        # Reconstruct the trie dict
        tree = {
            "tree": _deserialize_node(mt["tree"]),
            "v": mt["v"],
            "height_limit": model_params["max_depth"],
        }

        params = {
            "projections": proj,
            "offsets": offsets,
            "v": mt["v"],
            "mu": mt["mu"],
            "subsample_size": mt["subsample_size"],
            "seed": mt["seed"],
        }

        trees.append(tree)
        tree_params.append(params)

    return trees, tree_params, model_params


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Read NPZ-style file into a dict of numpy arrays."""
    import io
    import zipfile

    data: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(path, mode="r") as zf:
        for name in zf.namelist():
            if not name.endswith(".npy"):
                continue
            key = name[:-4]  # strip .npy
            raw = zf.read(name)
            data[key] = _read_npy(io.BytesIO(raw))
    return data


def _read_npy(buf) -> np.ndarray:
    """Read a single .npy from a byte buffer."""
    magic = buf.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError("Not a valid .npy file (bad magic)")

    major = struct.unpack("<B", buf.read(1))[0]
    minor = struct.unpack("<B", buf.read(1))[0]
    if major not in (1, 2, 3):
        raise ValueError(f"Unsupported .npy version: {major}.{minor}")

    header_len = struct.unpack("<H", buf.read(2))[0]
    header_raw = buf.read(header_len)
    header = json.loads(header_raw.decode("ascii"))

    dtype = np.dtype(header["descr"])
    shape = tuple(header["shape"])
    order: Literal["C", "F"] = "F" if header["fortran_order"] else "C"

    data = buf.read()
    shape_num: tuple[int, ...] = tuple(int(s) for s in shape)
    return np.frombuffer(data, dtype=dtype).reshape(shape_num, order=order)


# ---------------------------------------------------------------------------
# Node (de)serialization helpers
# ---------------------------------------------------------------------------


def _serialize_node(node: dict[str, Any]) -> dict[str, Any]:
    """Convert a trie node dict to a JSON-safe representation."""
    serialized: dict[str, Any] = {
        "type": node["type"],
        "hash_idx": node["hash_idx"],
        "size": node["size"],
    }
    if node["type"] == "internal":
        serialized["children"] = {str(k): _serialize_node(v) for k, v in node["children"].items()}
    return serialized


def _deserialize_node(s_node: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct a trie node dict from JSON-safe representation."""
    deserialized: dict[str, Any] = {
        "type": s_node["type"],
        "hash_idx": s_node["hash_idx"],
        "size": s_node["size"],
    }
    if s_node["type"] == "internal":
        deserialized["children"] = {
            int(k): _deserialize_node(v) for k, v in s_node["children"].items()
        }
    return deserialized
