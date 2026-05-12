from pathlib import Path
import shutil
import tempfile

import numpy as np
import pytest

from patent.lshiforest import LSHiForest


@pytest.fixture
def embeddings() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((500, 128)).astype(np.float32)


class TestLSHiForestAngle:
    def test_init_defaults(self):
        model = LSHiForest(n_trees=10, max_depth=12, seed=42, family="angle")
        assert model.family_name == "angle"
        assert model.n_trees == 10
        assert model.max_depth == 12
        assert model.seed == 42

    def test_build_and_score(self, embeddings):
        model = LSHiForest(n_trees=10, max_depth=12, seed=42, family="angle")
        model.fit(embeddings)

        scores = model.score(embeddings[:20])
        assert scores.shape == (20,)
        assert scores.dtype == np.float64
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)
        assert 0.0 < scores.mean() < 1.0

    def test_reproducibility(self, embeddings):
        m1 = LSHiForest(n_trees=10, max_depth=8, seed=42, family="angle")
        m1.fit(embeddings)

        m2 = LSHiForest(n_trees=10, max_depth=8, seed=42, family="angle")
        m2.fit(embeddings)

        s1 = m1.score(embeddings[:5])
        s2 = m2.score(embeddings[:5])
        assert np.allclose(s1, s2), "Scores differ across same-seed runs"

    def test_different_seeds_diverge(self, embeddings):
        m1 = LSHiForest(n_trees=10, max_depth=8, seed=42, family="angle")
        m1.fit(embeddings)

        m2 = LSHiForest(n_trees=10, max_depth=8, seed=99, family="angle")
        m2.fit(embeddings)

        s1 = m1.score(embeddings[:5])
        s2 = m2.score(embeddings[:5])
        assert not np.allclose(s1, s2), "Scores should differ across seeds"

    def test_anomalies_score_higher(self, embeddings):
        model = LSHiForest(n_trees=10, max_depth=8, seed=42, family="angle")
        model.fit(embeddings)

        normal = embeddings
        anomaly = normal + np.random.default_rng(7).standard_normal(128).astype(np.float32) * 3.0

        normal_score = model.score(normal[:100])
        anomaly_score = model.score(anomaly[:100])

        assert np.median(anomaly_score) > np.median(normal_score), (
            "Anomalies should score higher than normal points"
        )

    def test_score_normalize_flag(self, embeddings):
        model = LSHiForest(n_trees=10, max_depth=8, seed=42, family="angle")
        model.fit(embeddings)

        scores_norm = model.score(embeddings[:10], normalize=True)
        scores_raw = model.score(embeddings[:10], normalize=False)

        assert np.all(scores_norm >= 0) and np.all(scores_norm <= 1)
        assert np.all(scores_raw >= 0)
        assert not np.allclose(scores_norm, scores_raw)

    def test_save_load_round_trip(self, embeddings):
        model = LSHiForest(n_trees=10, max_depth=12, seed=42, family="angle")
        model.fit(embeddings)

        tmpdir = Path(tempfile.mkdtemp())
        mpath = tmpdir / "model.lshif"
        model.save(mpath)

        scores_before = model.score(embeddings[:10])

        loaded = LSHiForest.load(mpath)
        scores_after = loaded.score(embeddings[:10])

        diff = np.max(np.abs(scores_before - scores_after))
        assert diff < 1e-6, f"Scores differ after save/load: {diff}"

        if loaded._tempfile is not None:
            loaded._tempfile.unlink(missing_ok=True)
        shutil.rmtree(tmpdir)

    def test_subsample_sizes(self, embeddings):
        model = LSHiForest(n_trees=20, max_depth=8, seed=42, family="angle")
        model.fit(embeddings)

        sizes = model._subsample_sizes
        assert len(sizes) == 20
        assert all(64 <= s <= 1024 for s in sizes), f"Sizes out of range: {sizes}"
        assert len(set(sizes)) > 1, "Expected variable subsample sizes"

    def test_tree_mus_set(self, embeddings):
        model = LSHiForest(n_trees=10, max_depth=8, seed=42, family="angle")
        model.fit(embeddings)

        assert len(model._tree_mus) == 10
        assert all(mu > 0 for mu in model._tree_mus)


class TestLSHiForestL2:
    def test_build_and_score(self, embeddings):
        model = LSHiForest(n_trees=10, max_depth=12, seed=42, family="l2")
        model.fit(embeddings)

        scores = model.score(embeddings[:20])
        assert scores.shape == (20,)
        assert scores.dtype == np.float64
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_save_load_round_trip(self, embeddings):
        model = LSHiForest(n_trees=10, max_depth=12, seed=42, family="l2")
        model.fit(embeddings)

        tmpdir = Path(tempfile.mkdtemp())
        mpath = tmpdir / "model_l2.lshif"
        model.save(mpath)

        scores_before = model.score(embeddings[:10])

        loaded = LSHiForest.load(mpath)
        scores_after = loaded.score(embeddings[:10])

        diff = np.max(np.abs(scores_before - scores_after))
        assert diff < 1e-6, f"Scores differ after save/load (L2): {diff}"

        if loaded._tempfile is not None:
            loaded._tempfile.unlink(missing_ok=True)
        shutil.rmtree(tmpdir)

    def test_different_families_diverge(self, embeddings):
        ma = LSHiForest(n_trees=10, max_depth=8, seed=42, family="angle")
        ma.fit(embeddings)

        ml = LSHiForest(n_trees=10, max_depth=8, seed=42, family="l2")
        ml.fit(embeddings)

        sa = ma.score(embeddings[:20])
        sl = ml.score(embeddings[:20])

        # Different families should produce different scores (not identical)
        assert not np.allclose(sa, sl), "Different families should produce different scores"


class TestLSHiForestEdgeCases:
    def test_small_dataset(self):
        rng = np.random.default_rng(42)
        tiny = rng.standard_normal((60, 32)).astype(np.float32)

        model = LSHiForest(n_trees=5, max_depth=6, seed=42, family="angle")
        model.fit(tiny)
        scores = model.score(tiny)
        assert scores.shape == (60,)
        assert np.all(np.isfinite(scores))

    def test_single_query(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((200, 64)).astype(np.float32)

        model = LSHiForest(n_trees=5, max_depth=8, seed=42, family="angle")
        model.fit(data)

        score = model.score(data[:1])
        assert score.shape == (1,)
        assert np.isfinite(score[0])

    def test_empty_build_error(self):
        model = LSHiForest(n_trees=5, max_depth=8, seed=42)
        with pytest.raises(RuntimeError):
            model.score(np.random.randn(5, 64).astype(np.float32))

    def test_unknown_family(self):
        with pytest.raises(ValueError, match="Unknown LSH family"):
            LSHiForest(n_trees=5, max_depth=8, family="nonexistent")

    def test_full_batch_scoring_depths_in_range(self, embeddings):
        """Regression: raw depths must be in [0, max_depth] range."""
        model = LSHiForest(n_trees=10, max_depth=12, seed=42, family="angle")
        model.fit(embeddings)
        raw = model.score(embeddings, normalize=False)
        assert np.all(raw >= 0), f"Negative depths found: min={raw.min()}"
        assert np.all(raw <= 20), f"Depths exceed expected max: max={raw.max()}"

    def test_full_batch_scoring_depths_l2(self):
        """L2: raw depths must be in [0, max_depth] range for full batch."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((800, 128)).astype(np.float32)
        model = LSHiForest(n_trees=10, max_depth=12, seed=42, family="l2")
        model.fit(data)
        raw = model.score(data, normalize=False)
        assert np.all(raw >= 0), f"Negative depths: min={raw.min()}"
        assert np.all(raw <= 20), f"Depths out of range: max={raw.max()}"
