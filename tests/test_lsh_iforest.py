from pathlib import Path
import shutil
import tempfile

import numpy as np
import pytest

from patent.modeling.lsh_iforest import LSHIForest


def test_lshiforest_init():
    model = LSHIForest(num_trees=5, max_depth=16)
    assert model.meta.hash_bits == 32
    assert model.meta.bucket_bits == 2
    assert model._hash_dtype == np.uint32
    assert model.meta.num_trees == 5
    assert model.meta.max_depth == 16


def test_simhash_32bit():
    model = LSHIForest(num_trees=1, max_depth=8)
    model.meta.embedding_dim = 128
    model.meta.num_rows = 100

    proj = model._get_hyperplanes(0)
    assert proj.shape == (128, 32)
    assert proj.dtype == np.float32

    vectors = np.random.randn(10, 128).astype(np.float32)
    sigs = model._compute_simhash(vectors, proj)
    assert sigs.shape == (10,)
    assert sigs.dtype == np.uint32
    assert np.all(sigs < 2**32)


def test_build_single_tree_sorted():
    model = LSHIForest(num_trees=1, max_depth=8)
    model.meta.embedding_dim = 128
    model.meta.num_rows = 100

    rng = np.random.default_rng(42)
    sigs = rng.integers(0, 2**32, size=100, dtype=np.uint32)
    sorted_idx = np.argsort(sigs)
    sorted_sigs = sigs[sorted_idx]

    path_lengths = model._build_single_tree_sorted(sorted_sigs, sorted_idx)
    assert path_lengths.shape == (100,)
    assert path_lengths.dtype == np.float32
    assert np.all(path_lengths >= 1.0)
    assert np.all(path_lengths <= 8.0)


def test_delta_round_trip():
    model = LSHIForest(num_trees=3, max_depth=8)
    model.meta.embedding_dim = 128
    model.meta.num_rows = 100
    model.meta.num_trees = 3

    tmpdir = Path(tempfile.mkdtemp())
    mmap_path = tmpdir / "forest.lshif"
    original = np.random.default_rng(42).integers(
        0, 2**32, size=(3, 100), dtype=np.uint32
    )
    original.sort(axis=1)
    mmap = np.memmap(str(mmap_path), dtype=np.uint32, mode="w+", shape=(3, 100))
    mmap[:] = original
    model.forest_mmap = mmap

    encoded = model._delta_encode()
    decoded = LSHIForest._delta_decode(encoded, 3, 100)
    assert np.array_equal(decoded, original)
    shutil.rmtree(tmpdir)


def test_compress_round_trip():
    model = LSHIForest(num_trees=3, max_depth=8)
    model.meta.embedding_dim = 128
    model.meta.num_rows = 100
    model.meta.num_trees = 3
    model.meta.is_sorted = True

    rng = np.random.default_rng(42)
    data = rng.integers(0, 2**32, size=(3, 100), dtype=np.uint32)
    data.sort(axis=1)
    model.projections = [model._get_hyperplanes(i) for i in range(3)]

    tmpdir = Path(tempfile.mkdtemp())
    mmap_path = tmpdir / "forest.lshif"
    model.model_path = mmap_path
    model.model_path.parent.mkdir(parents=True, exist_ok=True)
    mmap = np.memmap(str(mmap_path), dtype=np.uint32, mode="w+", offset=1024, shape=(3, 100))
    mmap[:] = data
    model.forest_mmap = mmap

    queries = rng.standard_normal((10, 128)).astype(np.float32)
    model._calculate_baseline(tmpdir / "baseline.npy")
    scores_before = model.score(queries)

    out_path = tmpdir / "compressed.lshif"
    model.save_model(out_path, compress=True)
    assert out_path.stat().st_size > 0

    loaded = LSHIForest.load_model(out_path)
    scores_after = loaded.score(queries)

    diff = np.max(np.abs(scores_before - scores_after))
    assert diff < 1e-6, f"Scores differ after compress round-trip: {diff}"

    loaded._tempfile.unlink(missing_ok=True)
    shutil.rmtree(tmpdir)
