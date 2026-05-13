import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pyarrow.parquet as pq
import pytest

from patent.config import DATA_DIR
from patent.modeling.evaluate import (
    analyze_score_distribution,
    distance_to_centroid_correlation,
    evaluate_subsampling_stability,
    export_top_anomalies,
)
from patent.modeling.train import evaluate_model, train_model
from patent.utils import load_parquet_metadata, get_vectors_from_files


def sample_parquet_path():
    return str(DATA_DIR / "sample" / "embeddings" / "processed.parquet")


class TestScoreDistribution:
    def test_basic_distribution(self):
        scores = np.random.default_rng(42).uniform(0, 1, 1000).astype(np.float32)
        result = analyze_score_distribution(scores)

        assert result["n_samples"] == 1000
        assert result["n_finite"] == 1000
        assert 0 <= result["mean"] <= 1
        assert 0 <= result["median"] <= 1
        assert result["std"] > 0
        assert result["min"] >= 0
        assert result["max"] <= 1
        assert "skewness" in result
        assert "kurtosis" in result
        assert "percentile_50" in result
        assert "percentile_90" in result
        assert "percentile_95" in result
        assert "percentile_99" in result
        assert "pct_above_0_5" in result
        assert "pct_above_0_7" in result
        assert "pct_above_0_9" in result
        assert 0 <= result["pct_above_0_5"] <= 100
        assert 0 <= result["pct_above_0_9"] <= 100

    def test_all_high_scores(self):
        scores = np.full(500, 0.85, dtype=np.float32)
        result = analyze_score_distribution(scores)

        assert result["n_samples"] == 500
        assert result["mean"] == pytest.approx(0.85)
        assert result["std"] == pytest.approx(0.0, abs=1e-6)
        assert result["pct_above_0_7"] == pytest.approx(100.0)
        assert result["pct_above_0_9"] == pytest.approx(0.0)

    def test_non_finite_scores(self):
        scores = np.array([0.5, np.nan, np.inf, -np.inf, 0.8], dtype=np.float64)
        result = analyze_score_distribution(scores)

        assert result["n_samples"] == 5
        assert result["n_finite"] == 2
        assert result["mean"] == pytest.approx(0.65)

    def test_all_non_finite(self):
        scores = np.array([np.nan, np.inf], dtype=np.float64)
        result = analyze_score_distribution(scores)

        assert result["n_samples"] == 0
        assert result["n_finite"] == 0


class TestDistanceCentroidCorrelation:
    def test_basic_correlation(self):
        path = sample_parquet_path()
        total_rows = pq.ParquetFile(path).metadata.num_rows
        scores = np.random.default_rng(42).uniform(0, 1, total_rows).astype(np.float32)
        result = distance_to_centroid_correlation([path], scores)

        assert "spearman_correlation" in result
        assert "pearson_correlation" in result
        assert "mean_distance" in result
        assert "std_distance" in result
        assert result["std_distance"] >= 0
        assert result["mean_distance"] >= 0
        assert -1 <= result["spearman_correlation"] <= 1
        assert -1 <= result["pearson_correlation"] <= 1

    def test_empty_paths(self):
        scores = np.array([0.5])
        result = distance_to_centroid_correlation([], scores)
        assert result == {}

    def test_too_few_finite(self):
        scores = np.array([np.nan, np.inf], dtype=np.float64)
        path = sample_parquet_path()
        result = distance_to_centroid_correlation([path], scores)
        assert result == {}


class TestExportTopAnomalies:
    def test_basic_export(self):
        import pandas as pd

        scores = np.random.default_rng(42).uniform(0, 1, 100).astype(np.float32)
        metadata = pd.DataFrame(
            {
                "id": [f"id_{i}" for i in range(100)],
                "title": [f"Title {i}" for i in range(100)],
                "categories": ["cat.A"] * 100,
                "update_date": ["2024-01-15"] * 100,
            }
        )

        output = tempfile.mkdtemp()
        output_path = Path(output) / "top.json"
        records = export_top_anomalies(scores, metadata, output_path, top_k=10)

        assert len(records) == 10
        assert output_path.exists()

        with open(output_path) as f:
            saved = json.load(f)

        assert len(saved) == 10
        for record in saved:
            assert "anomaly_score" in record
            assert "id" in record
            assert "title" in record
            assert record["anomaly_score"] >= 0

        shutil.rmtree(output)

    def test_sorted_descending(self):
        import pandas as pd

        scores = np.array([0.1, 0.9, 0.5, 0.3, 0.8], dtype=np.float32)
        metadata = pd.DataFrame(
            {
                "id": ["a", "b", "c", "d", "e"],
                "title": ["A", "B", "C", "D", "E"],
            }
        )

        output = tempfile.mkdtemp()
        output_path = Path(output) / "top.json"
        records = export_top_anomalies(scores, metadata, output_path, top_k=3)

        assert records[0]["id"] == "b"
        assert records[0]["anomaly_score"] == pytest.approx(0.9)
        assert records[1]["id"] == "e"
        assert records[2]["id"] == "c"

        shutil.rmtree(output)

    def test_top_k_exceeds_data(self):
        import pandas as pd

        scores = np.array([0.3, 0.7], dtype=np.float32)
        metadata = pd.DataFrame({"id": ["x", "y"], "title": ["X", "Y"]})

        output = tempfile.mkdtemp()
        output_path = Path(output) / "top.json"
        records = export_top_anomalies(scores, metadata, output_path, top_k=100)

        assert len(records) == 2
        shutil.rmtree(output)


class TestSubsamplingStability:
    def test_basic_subsampling(self):
        path = sample_parquet_path()
        result = evaluate_subsampling_stability(
            [path],
            num_trees=5,
            max_depth=8,
            n_splits=3,
            subsample_ratio=0.8,
            top_k=100,
            seed=42,
        )

        summary = result["summary"]
        assert "spearman_aggregated" in summary
        assert "jaccard_aggregated" in summary
        assert result["subsample_ratio"] == 0.8
        assert result["n_splits"] == 3

    def test_invalid_ratio(self):
        path = sample_parquet_path()
        with pytest.raises(ValueError, match="subsample_ratio"):
            evaluate_subsampling_stability(
                [path], num_trees=5, max_depth=8, subsample_ratio=1.5
            )


class TestFullEvaluateModel:
    def test_full_evaluation(self):
        temp_path = Path(tempfile.mkdtemp())
        embeddings_dir = Path(DATA_DIR) / "sample" / "embeddings"

        train_result = train_model(embeddings_dir, temp_path)
        model_file = Path(train_result["output_dir"]) / "model.lshif"
        assert model_file.exists()

        result = evaluate_model(
            model_path=model_file,
            embeddings_dir=embeddings_dir,
            output_path=temp_path / "eval.json",
            top_k=5,
        )

        assert "stability" in result
        assert "score_distribution" in result
        assert "centroid_correlation" in result
        assert "top_anomalies_path" in result

        assert result["score_distribution"]["n_samples"] > 0
        assert "spearman_correlation" in result["centroid_correlation"]

        top_path = Path(result["top_anomalies_path"])
        assert top_path.exists()
        with open(top_path) as f:
            top_anomalies = json.load(f)
        assert len(top_anomalies) <= 5
        assert "anomaly_score" in top_anomalies[0]
        assert "title" in top_anomalies[0]

        eval_json = temp_path / "eval.json"
        assert eval_json.exists()

        shutil.rmtree(temp_path)

    def test_full_evaluation_with_subsampling(self):
        temp_path = Path(tempfile.mkdtemp())
        embeddings_dir = Path(DATA_DIR) / "sample" / "embeddings"

        train_result = train_model(
            embeddings_dir,
            temp_path,
            model_params={"num_trees": 5, "max_depth": 8},
        )
        model_file = Path(train_result["output_dir"]) / "model.lshif"

        result = evaluate_model(
            model_path=model_file,
            embeddings_dir=embeddings_dir,
            output_path=temp_path / "eval.json",
            top_k=3,
            do_subsampling=True,
            subsample_splits=2,
        )

        assert "subsampling_stability" in result
        assert "spearman_aggregated" in result["subsampling_stability"]

        shutil.rmtree(temp_path)
