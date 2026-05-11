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

    train_output_path = train_model(embeddings_dir, temp_path)
    model_file = Path(train_output_path) / "model.lshif"
    assert model_file.exists()

    evaluation = evaluate_model(
        model_path=model_file,
        embeddings_dir=embeddings_dir,
        output_path=temp_path / "eval.json",
    )
    assert "stability" in evaluation

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

    train_output_path = train_model(
        embeddings_dir,
        temp_path,
        mlflow_context=mlflow_context,
        model_params={"num_trees": 10, "max_depth": 12},
    )
    evaluation = evaluate_model(
        model_path=Path(train_output_path) / "model.lshif",
        embeddings_dir=embeddings_dir,
        output_path=temp_path / "eval.json",
        mlflow_context=mlflow_context,
    )
    assert "stability" in evaluation

    shutil.rmtree(temp_path)
