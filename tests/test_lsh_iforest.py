from pathlib import Path
import shutil
import tempfile

import numpy as np
import pytest

from patent.modeling.lsh_iforest import LSHIForest


@pytest.fixture
def embeddings() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((500, 128)).astype(np.float32)


class TestLSHIForestAngle:
    def test_init_defaults(self):
        model = LSHIForest(num_trees=10, max_depth=12, seed=42, lsh_family="angle")
        assert model.family_name == "angle"
        assert model.meta.num_trees == 10
        assert model.meta.max_depth == 12
        assert model.meta.seed == 42

    def test_build_and_score(self, embeddings):
        model = LSHIForest(num_trees=10, max_depth=12, seed=42, lsh_family="angle")
        model.build_forest_from_embeddings(embeddings)

        scores = model.score(embeddings[:20])
        assert scores.shape == (20,)
        assert scores.dtype == np.float64
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)
        assert 0.0 < scores.mean() < 1.0

    def test_baseline_output(self, embeddings):
        tmpdir = Path(tempfile.mkdtemp())
        baseline_path = tmpdir / "baseline.npy"

        model = LSHIForest(num_trees=10, max_depth=12, seed=42, lsh_family="angle")
        model.build_forest_from_embeddings(embeddings, baseline_output_path=baseline_path)

        baseline = np.load(baseline_path)
        assert baseline.shape == (500,)
        assert baseline.dtype == np.float32
        shutil.rmtree(tmpdir)

    def test_save_load_round_trip(self, embeddings):
        model = LSHIForest(num_trees=10, max_depth=12, seed=42, lsh_family="angle")
        model.build_forest_from_embeddings(embeddings)

        tmpdir = Path(tempfile.mkdtemp())
        mpath = tmpdir / "model.lshif"
        model.save_model(mpath)

        scores_before = model.score(embeddings[:10])

        loaded = LSHIForest.load_model(mpath)
        scores_after = loaded.score(embeddings[:10])

        diff = np.max(np.abs(scores_before - scores_after))
        assert diff < 1e-6, f"Scores differ after save/load: {diff}"

        loaded._tempfile.unlink(missing_ok=True)
        shutil.rmtree(tmpdir)

    def test_subsample_sizes(self, embeddings):
        model = LSHIForest(num_trees=20, max_depth=8, seed=42, lsh_family="angle")
        model.build_forest_from_embeddings(embeddings)

        sizes = model._tree_sizes
        assert len(sizes) == 20
        assert all(64 <= s <= 1024 for s in sizes), f"Sizes out of range: {sizes}"
        assert len(set(sizes)) > 1, "Expected variable subsample sizes"

    def test_tree_mus_set(self, embeddings):
        model = LSHIForest(num_trees=10, max_depth=8, seed=42, lsh_family="angle")
        model.build_forest_from_embeddings(embeddings)

        assert len(model._tree_mus) == 10
        assert all(mu > 0 for mu in model._tree_mus)

    def test_reproducibility(self, embeddings):
        m1 = LSHIForest(num_trees=10, max_depth=8, seed=42, lsh_family="angle")
        m1.build_forest_from_embeddings(embeddings)

        m2 = LSHIForest(num_trees=10, max_depth=8, seed=42, lsh_family="angle")
        m2.build_forest_from_embeddings(embeddings)

        s1 = m1.score(embeddings[:5])
        s2 = m2.score(embeddings[:5])
        assert np.allclose(s1, s2), "Scores differ across same-seed runs"

    def test_different_seeds_diverge(self, embeddings):
        m1 = LSHIForest(num_trees=10, max_depth=8, seed=42, lsh_family="angle")
        m1.build_forest_from_embeddings(embeddings)

        m2 = LSHIForest(num_trees=10, max_depth=8, seed=99, lsh_family="angle")
        m2.build_forest_from_embeddings(embeddings)

        s1 = m1.score(embeddings[:5])
        s2 = m2.score(embeddings[:5])
        assert not np.allclose(s1, s2), "Scores should differ across seeds"

    def test_anomalies_score_higher(self, embeddings):
        model = LSHIForest(num_trees=10, max_depth=8, seed=42, lsh_family="angle")
        model.build_forest_from_embeddings(embeddings)
        scores = model.score(embeddings)

        normal = embeddings
        anomaly = normal + np.random.default_rng(7).standard_normal(128).astype(np.float32) * 3.0

        normal_score = model.score(normal[:100])
        anomaly_score = model.score(anomaly[:100])

        assert np.median(anomaly_score) > np.median(normal_score), (
            "Anomalies should score higher than normal points"
        )

    def test_score_normalize_flag(self, embeddings):
        model = LSHIForest(num_trees=10, max_depth=8, seed=42, lsh_family="angle")
        model.build_forest_from_embeddings(embeddings)

        scores_norm = model.score(embeddings[:10], normalize=True)
        scores_raw = model.score(embeddings[:10], normalize=False)

        assert np.all(scores_norm >= 0) and np.all(scores_norm <= 1)
        assert np.all(scores_raw >= 0)
        assert not np.allclose(scores_norm, scores_raw)


class TestLSHIForestL2:
    def test_build_and_score(self, embeddings):
        model = LSHIForest(num_trees=10, max_depth=12, seed=42, lsh_family="l2")
        model.build_forest_from_embeddings(embeddings)

        scores = model.score(embeddings[:20])
        assert scores.shape == (20,)
        assert scores.dtype == np.float64
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_save_load_round_trip(self, embeddings):
        model = LSHIForest(num_trees=10, max_depth=12, seed=42, lsh_family="l2")
        model.build_forest_from_embeddings(embeddings)

        tmpdir = Path(tempfile.mkdtemp())
        mpath = tmpdir / "model_l2.lshif"
        model.save_model(mpath)

        scores_before = model.score(embeddings[:10])

        loaded = LSHIForest.load_model(mpath)
        scores_after = loaded.score(embeddings[:10])

        diff = np.max(np.abs(scores_before - scores_after))
        assert diff < 1e-6, f"Scores differ after save/load (L2): {diff}"

        loaded._tempfile.unlink(missing_ok=True)
        shutil.rmtree(tmpdir)

    def test_different_families_diverge(self, embeddings):
        ma = LSHIForest(num_trees=10, max_depth=8, seed=42, lsh_family="angle")
        ma.build_forest_from_embeddings(embeddings)

        ml = LSHIForest(num_trees=10, max_depth=8, seed=42, lsh_family="l2")
        ml.build_forest_from_embeddings(embeddings)

        sa = ma.score(embeddings[:20])
        sl = ml.score(embeddings[:20])

        corr = np.corrcoef(sa, sl)[0, 1]
        assert corr > 0.3, f"Different families should have some correlation, got {corr:.4f}"
        assert not np.allclose(sa, sl), "Different families should produce different scores"


class TestLSHIForestEdgeCases:
    def test_small_dataset(self):
        rng = np.random.default_rng(42)
        tiny = rng.standard_normal((60, 32)).astype(np.float32)

        model = LSHIForest(num_trees=5, max_depth=6, seed=42, lsh_family="angle")
        model.build_forest_from_embeddings(tiny)
        scores = model.score(tiny)
        assert scores.shape == (60,)
        assert np.all(np.isfinite(scores))

    def test_single_query(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((200, 64)).astype(np.float32)

        model = LSHIForest(num_trees=5, max_depth=8, seed=42, lsh_family="angle")
        model.build_forest_from_embeddings(data)

        score = model.score(data[:1])
        assert score.shape == (1,)
        assert np.isfinite(score[0])

    def test_empty_build_error(self):
        model = LSHIForest(num_trees=5, max_depth=8, seed=42)
        with pytest.raises(RuntimeError):
            model.score(np.random.randn(5, 64).astype(np.float32))

    def test_unknown_family(self):
        with pytest.raises(ValueError, match="Unknown LSH family"):
            LSHIForest(num_trees=5, max_depth=8, lsh_family="nonexistent")
