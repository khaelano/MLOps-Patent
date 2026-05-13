from pathlib import Path
import shutil
import tempfile

import pytest

from patent.config import DATA_DIR
from patent.modeling.train import evaluate_model, train_model


def test_pipeline():
    """Train an LSHiForest on the sample embeddings and evaluate it."""
    temp_path = Path(tempfile.mkdtemp())
    embeddings_dir = Path(DATA_DIR) / "sample" / "embeddings"
    assert embeddings_dir.exists(), f"Sample data not found at {embeddings_dir}"

    result = train_model(embeddings_dir, temp_path)
    model_file = Path(result["output_dir"]) / "model.lshif"
    assert model_file.exists()
    assert "stability" in result["eval_result"]

    shutil.rmtree(temp_path)


@pytest.mark.skip(reason="Requires running MLflow server at MLFLOW_TRACKING_URI")
def test_pipeline_mlflow():
    import os

    import mlflow

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    with mlflow.start_run(run_name="test_run") as run:
        mlflow_context = {"run_id": run.info.run_id}

    temp_path = Path(tempfile.mkdtemp())
    embeddings_dir = Path(DATA_DIR) / "sample" / "embeddings"

    result = train_model(
        embeddings_dir,
        temp_path,
        mlflow_context=mlflow_context,
        model_params={"num_trees": 10, "max_depth": 12},
    )
    assert "stability" in result["eval_result"]

    shutil.rmtree(temp_path)
