"""Tests that exercise the MLflow integration paths in train_model, evaluate_model,
and the model registry — using the file-based tracking backend so no external
MLflow server is required."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import mlflow
from mlflow.tracking import MlflowClient
import pytest

from patent.config import DATA_DIR
from patent.modeling.registry import register_from_run
from patent.modeling.train import evaluate_model, train_model


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_mlflow():
    """Give every test a private MLflow tracking URI so they don't interfere."""
    tmp = tempfile.mkdtemp(prefix="mlflow-test-")
    uri = Path(tmp).as_uri()
    mlflow.set_tracking_uri(uri)
    yield
    mlflow.set_tracking_uri("")  # reset
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_embeddings_dir() -> Path:
    return Path(DATA_DIR) / "sample" / "embeddings"


# ── train_model ────────────────────────────────────────────────────────────────


class TestTrainModelMlflow:
    def test_train_with_experiment_name_logs_params_and_metrics(
        self, sample_embeddings_dir: Path
    ):
        """train_model must set_experiment, start_run, log params/metrics/artifacts."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            result = train_model(
                sample_embeddings_dir,
                temp_dir,
                model_params={"n_trees": 3, "max_depth": 8, "seed": 1},
                mlflow_context={"experiment_name": "ci-test-train"},
                top_k=5,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Return value
        assert result["output_dir"] == str(temp_dir)
        assert "stability" in result["eval_result"]
        assert result["run_id"] is not None

        # MLflow run exists with correct data
        client = MlflowClient()
        run = client.get_run(result["run_id"])
        assert run is not None

        # Experiment was auto-created with the right name
        exp = client.get_experiment_by_name("ci-test-train")
        assert exp is not None
        assert run.info.experiment_id == exp.experiment_id

        # Params
        params = run.data.params
        assert params["n_trees"] == "3"
        assert params["max_depth"] == "8"
        assert params["seed"] == "1"

        # Training metrics
        metrics = run.data.metrics
        assert "train/fit_time_s" in metrics
        assert "train/baseline_scoring_time_s" in metrics
        assert "train/total_time_s" in metrics

        # Evaluation metrics (flattened, so stability subtrees are separated by /)
        assert "stability/spearman_aggregated" in metrics or any(
            k.startswith("stability") for k in metrics
        )

        # Artifacts
        artifacts = [a.path for a in client.list_artifacts(result["run_id"])]
        assert "model.lshif" in artifacts

    def test_train_without_mlflow_skips_tracking(self, sample_embeddings_dir: Path):
        """Without mlflow_context, no run is created."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            result = train_model(
                sample_embeddings_dir,
                temp_dir,
                model_params={"n_trees": 2, "max_depth": 6},
                mlflow_context=None,  # ← explicitly None
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        assert result["run_id"] is None
        assert "stability" in result["eval_result"]

    def test_train_with_run_id_resumes_run(self, sample_embeddings_dir: Path):
        """Passing a run_id in mlflow_context resumes the existing run."""
        # Create a run first
        mlflow.set_experiment("ci-test-resume")
        with mlflow.start_run() as pre_run:
            pre_run_id = pre_run.info.run_id
            mlflow.log_param("pre_existing", "yes")

        temp_dir = Path(tempfile.mkdtemp())
        try:
            result = train_model(
                sample_embeddings_dir,
                temp_dir,
                model_params={"n_trees": 2, "max_depth": 6},
                mlflow_context={"run_id": pre_run_id, "experiment_name": "ci-test-resume"},
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        assert result["run_id"] == pre_run_id
        # The pre-existing param should still be there
        client = MlflowClient()
        run = client.get_run(pre_run_id)
        assert run.data.params["pre_existing"] == "yes"
        # And new params were added
        assert "n_trees" in run.data.params


# ── evaluate_model ─────────────────────────────────────────────────────────────


class TestEvaluateModelMlflow:
    def test_evaluate_with_experiment_name_logs_metrics(
        self, sample_embeddings_dir: Path
    ):
        """evaluate_model must set_experiment, start_run, and log eval metrics."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Train a tiny model first (without MLflow) so we have a .lshif file
            train_result = train_model(
                sample_embeddings_dir,
                temp_dir,
                model_params={"n_trees": 3, "max_depth": 8},
                mlflow_context=None,
            )
            model_file = Path(train_result["output_dir"]) / "model.lshif"

            # Now evaluate with MLflow
            eval_output = temp_dir / "eval.json"
            eval_result = evaluate_model(
                model_path=model_file,
                embeddings_dir=sample_embeddings_dir,
                output_path=eval_output,
                mlflow_context={"experiment_name": "ci-test-eval"},
                top_k=5,
            )

            assert "stability" in eval_result
            assert eval_output.exists()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Verify the MLflow run has the flattened metrics
        client = MlflowClient()
        exp = client.get_experiment_by_name("ci-test-eval")
        assert exp is not None
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=1,
        )
        assert len(runs) == 1
        metrics = runs[0].data.metrics
        assert any(k.startswith("stability") for k in metrics)


# ── Model registry ─────────────────────────────────────────────────────────────


class TestModelRegistry:
    @patch("patent.modeling.registry.MlflowClient")
    def test_register_from_run_creates_version_and_promotes(self, MockClient):
        """Happy-path: no existing Production version → registers + promotes."""
        mock_client = MockClient.return_value

        # Simulate: run exists with a jaccard metric
        mock_client.get_run.return_value.data.metrics = {
            "stability/jaccard_aggregated": 0.45,
        }

        # No existing Production versions
        mock_client.get_latest_versions.return_value = []

        # create_model_version returns a mock
        mock_version = mock_client.create_model_version.return_value
        mock_version.version = "3"

        result = register_from_run(
            run_id="fake-run-id",
            model_name="patent-lshiforest",
            metric_key="stability/jaccard_aggregated",
        )

        assert result["model_name"] == "patent-lshiforest"
        assert result["version"] == "3"
        assert result["metric_value"] == 0.45
        assert result["promoted_to_production"] is True
        assert result["previous_prod_version"] is None

        # Was promoted to Production
        mock_client.transition_model_version_stage.assert_any_call(
            name="patent-lshiforest", version="3", stage="Production"
        )

    @patch("patent.modeling.registry.MlflowClient")
    def test_register_worse_model_not_promoted(self, MockClient):
        """When new metric ≤ Production metric, register but don't promote."""
        mock_client = MockClient.return_value

        mock_client.get_run.return_value.data.metrics = {
            "stability/jaccard_aggregated": 0.30,
        }

        # Existing Production version with better metric
        prod_version = MagicMock()
        prod_version.version = "2"
        prod_version.run_id = "prod-run"
        mock_client.get_latest_versions.return_value = [prod_version]

        # The Production run's metrics
        mock_prod_run = MagicMock()
        mock_prod_run.data.metrics = {"stability/jaccard_aggregated": 0.50}
        mock_client.get_run.side_effect = [
            mock_client.get_run.return_value,  # first call: new run
            mock_prod_run,  # second call: production run
        ]

        mock_version = mock_client.create_model_version.return_value
        mock_version.version = "3"

        result = register_from_run(run_id="fake-run-id")

        assert result["promoted_to_production"] is False
        assert result["previous_prod_version"] == "2"
        assert result["previous_metric_value"] == 0.50

        # Should NOT have been promoted to Production
        prod_calls = [
            c
            for c in mock_client.transition_model_version_stage.call_args_list
            if c.kwargs.get("stage") == "Production"
            or (len(c.args) >= 3 and c.args[2] == "Production")
        ]
        assert len(prod_calls) == 0

    @patch("patent.modeling.registry.MlflowClient")
    def test_register_better_model_promotes_and_archives(self, MockClient):
        """When new metric improves, promote and archive old."""
        mock_client = MockClient.return_value

        mock_client.get_run.return_value.data.metrics = {
            "stability/jaccard_aggregated": 0.72,
        }

        prod_version = MagicMock()
        prod_version.version = "1"
        prod_version.run_id = "old-run"
        mock_client.get_latest_versions.return_value = [prod_version]

        mock_prod_run = MagicMock()
        mock_prod_run.data.metrics = {"stability/jaccard_aggregated": 0.55}
        mock_client.get_run.side_effect = [
            mock_client.get_run.return_value,
            mock_prod_run,
        ]

        mock_version = mock_client.create_model_version.return_value
        mock_version.version = "2"

        result = register_from_run(run_id="fake-run-id")

        assert result["promoted_to_production"] is True

        # Should promote new to Production
        mock_client.transition_model_version_stage.assert_any_call(
            name="patent-lshiforest", version="2", stage="Production"
        )
        # Should archive old
        mock_client.transition_model_version_stage.assert_any_call(
            name="patent-lshiforest", version="1", stage="Archived"
        )

    @patch("patent.modeling.registry.MlflowClient")
    def test_register_missing_metric_warns_and_promotes(self, MockClient):
        """When metric key is missing from the run, warn and promote anyway."""
        mock_client = MockClient.return_value

        # Run has no jaccard metric
        mock_client.get_run.return_value.data.metrics = {"other_metric": 0.5}
        mock_client.get_latest_versions.return_value = []

        mock_version = mock_client.create_model_version.return_value
        mock_version.version = "1"

        result = register_from_run(run_id="fake-run-id")

        assert result["metric_value"] == 0.0  # fallback
        assert result["promoted_to_production"] is True  # promotes default


# ── Round-trip: train → register (integration-light) ──────────────────────────


class TestTrainRegisterRoundTrip:
    def test_train_then_register_sequence(self, sample_embeddings_dir: Path):
        """End-to-end: train with MLflow, get run_id, attempt register."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # 1. Train with MLflow
            result = train_model(
                sample_embeddings_dir,
                temp_dir,
                model_params={"n_trees": 3, "max_depth": 8, "seed": 42},
                mlflow_context={"experiment_name": "ci-test-roundtrip"},
                top_k=5,
            )
            assert result["run_id"] is not None

            # 2. Verify the run has the model artifact
            client = MlflowClient()
            artifacts = [a.path for a in client.list_artifacts(result["run_id"])]
            assert "model.lshif" in artifacts

            # 3. Verify registration would be possible (the artifact URI is valid)
            # We can't fully register without a server, but verify the run is sound.
            run_data = client.get_run(result["run_id"]).data
            assert "stability/jaccard_aggregated" in run_data.metrics or any(
                k.startswith("stability/") for k in run_data.metrics
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
