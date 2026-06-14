import argparse
import json
import os
from pathlib import Path

from loguru import logger
import mlflow

from patent.config import MODELS_DIR, PROCESSED_DATA_DIR
from patent.modeling.train import train_model


def main():
    parser = argparse.ArgumentParser(description="Train LSHiForest model from processed data.")
    parser.add_argument("--params", type=str, help="Path to params.json (optional)", default=None)
    args = parser.parse_args()

    model_params = {}
    mlflow_params = {}

    if args.params:
        params_path = Path(args.params)
        if params_path.exists():
            with open(params_path, "r") as f:
                config = json.load(f)
                model_params = config.get("model_params", {})
                mlflow_params = config.get("mlflow_params", {})
                logger.info(f"Loaded params from {args.params}")
        else:
            logger.warning(f"Provided params file {args.params} does not exist. Using defaults.")

    parquet_files = list(PROCESSED_DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        logger.error(f"No processed parquet files found in {PROCESSED_DATA_DIR}")
        return

    parquet_files = sorted(parquet_files)
    logger.info(f"Targeting {len(parquet_files)} parquet file(s) for training.")

    # ── MLflow context: env var overrides params.json ──────────────────
    mlflow_context = None
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")

    if mlflow_params:
        mlflow_context = {}
        if experiment_name:
            mlflow_context["experiment_name"] = experiment_name
            mlflow.set_experiment(experiment_name)
        elif "experiment_name" in mlflow_params:
            mlflow.set_experiment(mlflow_params["experiment_name"])

        for k in ["run_id", "experiment_id", "run_name", "description", "tags"]:
            if k in mlflow_params:
                mlflow_context[k] = mlflow_params[k]

        if not mlflow_context:
            mlflow_context = None
    elif experiment_name:
        mlflow_context = {"experiment_name": experiment_name}
        mlflow.set_experiment(experiment_name)

    logger.info("Initializing model training phase...")
    result = train_model(
        embeddings_dir=PROCESSED_DATA_DIR,
        output_dir=MODELS_DIR,
        model_params=model_params,
        mlflow_context=mlflow_context,
    )

    # ── Emit machine-readable output for CI / dvc repro to capture ──
    if result.get("run_id"):
        print(f"MLFLOW_RUN_ID={result['run_id']}")
    if result.get("pyfunc_version"):
        print(f"MLFLOW_PYFUNC_VERSION={result['pyfunc_version']}")

    logger.success(f"Training pipeline completed. Artifacts saved in {MODELS_DIR}")


if __name__ == "__main__":
    main()
