"""End-to-end simulation tests for the continuous training pipeline.

Uses the sample parquet data in ``data/sample/embeddings/`` to simulate
the full pipeline without touching external services.

Run with::

    python -m pytest tests/test_continuous_training.py -v
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from patent.config import DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from patent.modeling.train import train_model
from patent.monitoring.drift import (
    compute_drift_metrics,
    load_drift_baseline,
    save_drift_baseline,
    BASELINE_DIR,
)
from patent.monitoring.metrics import (
    DRIFT_GAUGE,
    DRIFT_GAUGE_EMB_SHIFT,
    DRIFT_GAUGE_MEAN_SHIFT,
    DRIFT_GAUGE_N_SAMPLES,
    DRIFT_GAUGE_PVALUE,
    MODEL_INFO,
    update_drift_metrics,
)
from patent.pipeline.continuous import (
    _find_new_update_dirs,
    _process_single_source,
    _update_drift_baseline,
)


SAMPLE_PARQUET = DATA_DIR / "sample" / "embeddings" / "processed.parquet"


@pytest.fixture(autouse=True)
def _skip_if_no_sample():
    """Skip all tests if sample data doesn't exist."""
    if not SAMPLE_PARQUET.exists():
        pytest.skip(f"Sample data not found at {SAMPLE_PARQUET}")




def _prepare_simulated_env(
    tmp_path: Path,
) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    """Set up a simulated data environment with an existing and a "new" parquet.

    Returns (processed_dir, new_parquet_path, old_df, new_df).
    """
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    df_full = pd.read_parquet(SAMPLE_PARQUET)

    # Split: 60% "old" (already processed), 40% "new" (just arrived)
    split_idx = int(len(df_full) * 0.6)
    old_df = df_full.iloc[:split_idx].reset_index(drop=True)
    new_df = df_full.iloc[split_idx:].reset_index(drop=True)

    old_path = processed_dir / "old_data.parquet"
    new_path = processed_dir / "new_data.parquet"
    old_df.to_parquet(old_path, index=False)
    # Don't write new_path yet — simulate it arriving later

    return processed_dir, new_path, old_df, new_df




class TestDriftDetection:
    """Unit tests for drift detection without external services."""

    def test_save_and_load_baseline(self, tmp_path):
        """Save a drift baseline and reload it correctly."""
        EMBEDDING_BASELINE_PATH = BASELINE_DIR / "embedding_baseline.npz"
        SCORE_BASELINE_PATH = BASELINE_DIR / "score_baseline.npz"

        # Monkey-patch paths to use tmp_path
        with (
            mock.patch.object(
                __import__("patent.monitoring.drift", fromlist=["EMBEDDING_BASELINE_PATH"]),
                "EMBEDDING_BASELINE_PATH",
                tmp_path / "embedding_baseline.npz",
            ),
            mock.patch.object(
                __import__("patent.monitoring.drift", fromlist=["SCORE_BASELINE_PATH"]),
                "SCORE_BASELINE_PATH",
                tmp_path / "score_baseline.npz",
            ),
            mock.patch.object(
                __import__("patent.monitoring.drift", fromlist=["BASELINE_META_PATH"]),
                "BASELINE_META_PATH",
                tmp_path / "baseline_meta.json",
            ),
        ):
            # Create dummy data
            rng = np.random.default_rng(42)
            embeddings = rng.normal(0, 1, (200, 64)).astype(np.float32)
            scores = rng.random(200).astype(np.float32)

            baseline = save_drift_baseline(embeddings, scores, model_version="1")

            assert baseline.n_samples == 200
            assert baseline.embedding_mean.shape == (64,)
            assert baseline.embedding_std.shape == (64,)
            assert baseline.model_version == "1"

            # Reload
            loaded = load_drift_baseline()
            assert loaded is not None
            assert loaded.n_samples == 200
            np.testing.assert_array_almost_equal(
                loaded.embedding_mean, baseline.embedding_mean
            )

    def test_compute_drift_no_change(self, tmp_path):
        """Drift check against identical distribution should yield low KS."""
        rng = np.random.default_rng(123)
        embeddings = rng.normal(0, 1, (500, 128)).astype(np.float32)
        scores = rng.random(500).astype(np.float32)

        baseline = save_drift_baseline(embeddings, scores, model_version="1")

        # Same data → KS should be ~0
        report = compute_drift_metrics(embeddings, scores, baseline=baseline)
        assert report.score_ks_statistic < 0.05
        assert abs(report.score_mean_shift) < 0.01

    def test_compute_drift_with_shift(self, tmp_path):
        """Drift check against shifted distribution should yield high KS."""
        rng = np.random.default_rng(456)
        embeddings_a = rng.normal(0, 1, (500, 128)).astype(np.float32)
        scores_a = rng.random(500).astype(np.float32)

        baseline = save_drift_baseline(embeddings_a, scores_a)

        # Shifted scores (mean +0.3)
        scores_b = (rng.random(500) + 0.3).astype(np.float32)
        # Shifted embeddings (mean +1.0)
        embeddings_b = rng.normal(1.0, 1, (500, 128)).astype(np.float32)

        report = compute_drift_metrics(embeddings_b, scores_b, baseline=baseline)
        # Should detect significant score distribution drift
        assert report.score_ks_statistic > 0.2
        assert report.score_mean_shift > 0.2
        # Embedding shift should be ~1.0/std
        assert report.embedding_mean_shift > 0.5

    def test_drift_without_baseline(self, tmp_path):
        """Drift check with no baseline returns neutral report."""
        embeddings = np.random.default_rng(1).normal(0, 1, (100, 64)).astype(np.float32)
        scores = np.random.default_rng(2).random(100).astype(np.float32)

        # Ensure no baseline exists on disk by monkey-patching to tmp_path
        with (
            mock.patch(
                "patent.monitoring.drift.EMBEDDING_BASELINE_PATH",
                tmp_path / "nonexistent_emb.npz",
            ),
            mock.patch(
                "patent.monitoring.drift.SCORE_BASELINE_PATH",
                tmp_path / "nonexistent_scr.npz",
            ),
            mock.patch(
                "patent.monitoring.drift.BASELINE_META_PATH",
                tmp_path / "nonexistent_meta.json",
            ),
        ):
            from patent.monitoring.drift import compute_drift_metrics

            report = compute_drift_metrics(embeddings, scores)
        assert report.score_ks_statistic == 0.0
        assert report.score_ks_pvalue == 1.0


class TestPrometheusMetrics:
    """Verify Prometheus metric gauges update correctly."""

    def test_update_drift_metrics(self):
        """Gauges should reflect the values passed to update_drift_metrics."""
        scores = np.array([0.1, 0.5, 0.9, 1.0, 0.75, 0.0], dtype=np.float32)

        update_drift_metrics(
            ks_statistic=0.15,
            ks_pvalue=0.003,
            mean_shift=0.05,
            emb_shift=0.42,
            n_samples=1000,
            scores=scores,
            model_version="3",
            embedding_dim=384,
            total_rows=5000,
        )

        assert DRIFT_GAUGE._value.get() == 0.15
        assert DRIFT_GAUGE_PVALUE._value.get() == 0.003
        assert DRIFT_GAUGE_MEAN_SHIFT._value.get() == 0.05
        assert DRIFT_GAUGE_EMB_SHIFT._value.get() == 0.42
        assert DRIFT_GAUGE_N_SAMPLES._value.get() == 1000


class TestContinuousPipeline:
    """Integration tests for the continuous training pipeline components."""

    def test_find_new_update_dirs_empty(self, tmp_path):
        """_find_new_update_dirs returns empty when no new data exists."""
        with (
            mock.patch.object(
                __import__(
                    "patent.pipeline.continuous", fromlist=["RAW_DATA_DIR"]
                ),
                "RAW_DATA_DIR",
                tmp_path / "raw",
            ),
            mock.patch.object(
                __import__(
                    "patent.pipeline.continuous", fromlist=["PROCESSED_DATA_DIR"]
                ),
                "PROCESSED_DATA_DIR",
                tmp_path / "processed",
            ),
        ):
            (tmp_path / "processed").mkdir()
            new_dirs = _find_new_update_dirs()
            assert new_dirs == []

    def test_process_single_source(self, tmp_path):
        """_process_single_source creates reserialized → cleaned → embedded parquets."""
        # Create a minimal XML source
        xml_content = """<?xml version="1.0"?>
        <OAI-PMH>
          <ListRecords>
            <record>
              <header><identifier>oai:arXiv.org:0704.0001</identifier>
              <datestamp>2008-01-01</datestamp></header>
              <metadata>
                <arXivRaw>
                  <id>0704.0001</id>
                  <title>Test Paper Title</title>
                  <categories>cs.AI</categories>
                  <updated>2008-01-01</updated>
                </arXivRaw>
              </metadata>
            </record>
          </ListRecords>
        </OAI-PMH>"""
        xml_path = tmp_path / "test_update" / "page_1.xml"
        xml_path.parent.mkdir(parents=True)
        xml_path.write_text(xml_content)

        with (
            mock.patch.object(
                __import__(
                    "patent.pipeline.continuous", fromlist=["INTERIM_DATA_DIR"]
                ),
                "INTERIM_DATA_DIR",
                tmp_path / "interim",
            ),
            mock.patch.object(
                __import__(
                    "patent.pipeline.continuous", fromlist=["PROCESSED_DATA_DIR"]
                ),
                "PROCESSED_DATA_DIR",
                tmp_path / "processed",
            ),
        ):
            interim = tmp_path / "interim"
            processed = tmp_path / "processed"
            for d in [
                interim / "serialized",
                interim / "cleaned",
                processed,
            ]:
                d.mkdir(parents=True, exist_ok=True)

            result = _process_single_source(xml_path.parent)
            assert result.exists()
            assert result.suffix == ".parquet"


class TestFullSimulation:
    """Full end-to-end simulation using sample data."""

    def test_end_to_end_simulation(self, tmp_path):
        """Simulate: existing data + new data → train → evaluate → drift baseline.

        This test runs the full training workflow on sample data to verify
        that all components work together.  No external services are required.
        """
        # ── Setup simulated environment ──────────────────────────────────
        processed_dir, new_path, old_df, new_df = _prepare_simulated_env(tmp_path)

        output_dir = tmp_path / "models"
        output_dir.mkdir()

        # ── Step 1: Train on "existing" data only ────────────────────────
        v1_dir = output_dir / "v1"
        v1_dir.mkdir(parents=True, exist_ok=True)
        result_old = train_model(
            embeddings_dir=processed_dir,
            output_dir=v1_dir,
            model_params={"n_trees": 10, "max_depth": 12, "seed": 42},
        )
        model_path_old = v1_dir / "model.lshif"
        assert model_path_old.exists(), f"Model not created at {model_path_old}"
        assert "stability" in result_old["eval_result"]

        # ── Step 2: "New data arrives" ────────────────────────────────────
        new_df.to_parquet(new_path, index=False)
        all_parquets = sorted(processed_dir.glob("*.parquet"))
        assert len(all_parquets) == 2

        # ── Step 3: Train on full (old + new) data ───────────────────────
        v2_dir = output_dir / "v2"
        v2_dir.mkdir(parents=True, exist_ok=True)
        result_full = train_model(
            embeddings_dir=processed_dir,
            output_dir=v2_dir,
            model_params={"n_trees": 10, "max_depth": 12, "seed": 42},
        )
        model_path_full = v2_dir / "model.lshif"
        assert model_path_full.exists()

        # ── Step 4: Evaluation results are present ────────────────────────
        eval_result = result_full["eval_result"]
        assert "stability" in eval_result
        assert "score_distribution" in eval_result
        assert "centroid_correlation" in eval_result

        eval_path = v2_dir / "evaluation.json"
        assert eval_path.exists()
        eval_data = json.loads(eval_path.read_text())
        assert "stability/jaccard_aggregated" in eval_data

        # ── Step 5: Drift baseline + drift check ─────────────────────────
        # Save baseline from model v2
        all_scores = np.array(
            [
                eval_data.get(f"score_distribution/{k}", 0.0)
                for k in ["mean", "median", "std"]
                if eval_data.get(f"score_distribution/{k}") is not None
            ]
        )
        # Actually let's just score a few rows
        from patent.lshiforest import LSHiForest

        model = LSHiForest.load(str(model_path_full))
        # Load a small batch
        small_df = pd.concat([old_df.head(100), new_df.head(100)])
        embeddings_list = small_df["embedding"].tolist()
        X_sample = np.array(embeddings_list, dtype=np.float32)
        scores_sample = model.score(X_sample)

        baseline = save_drift_baseline(
            X_sample, scores_sample, model_version="v2"
        )

        # Check drift with same data (should show no drift)
        report_same = compute_drift_metrics(X_sample, scores_sample, baseline=baseline)
        assert report_same.score_ks_statistic < 0.05  # same distribution
        assert abs(report_same.score_mean_shift) < 0.01

        # Check drift with slightly perturbed embeddings
        rng = np.random.default_rng(99)
        perturbed = X_sample + rng.normal(0, 0.001, X_sample.shape).astype(np.float32)
        scores_perturbed = model.score(perturbed)
        report = compute_drift_metrics(perturbed, scores_perturbed, baseline=baseline)
        # Drift metrics should be finite and reasonable
        assert np.isfinite(report.score_ks_statistic)
        assert np.isfinite(report.embedding_mean_shift)

        # ── Step 6: Prometheus metrics update ────────────────────────────
        update_drift_metrics(
            ks_statistic=report.score_ks_statistic,
            ks_pvalue=report.score_ks_pvalue,
            mean_shift=report.score_mean_shift,
            emb_shift=report.embedding_mean_shift,
            n_samples=report.n_new_samples,
            scores=scores_perturbed,
            model_version="v2",
            embedding_dim=X_sample.shape[1],
            total_rows=len(X_sample),
        )
