import argparse
from datetime import datetime
import json
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

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = MODELS_DIR / f"run_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created run directory: {run_dir}")

    parquet_files = list(PROCESSED_DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        logger.error(f"No processed parquet files found in {PROCESSED_DATA_DIR}")
        return

    parquet_files = sorted(parquet_files)
    logger.info(f"Targeting {len(parquet_files)} parquet file(s) for training.")

    mlflow_context = None
    if mlflow_params:
        mlflow_context = {}
        if "experiment_name" in mlflow_params:
            mlflow.set_experiment(mlflow_params["experiment_name"])

        for k in ["run_id", "experiment_id", "run_name", "description", "tags"]:
            if k in mlflow_params:
                mlflow_context[k] = mlflow_params[k]

        if not mlflow_context:
            mlflow_context = None

    logger.info("Initializing model training phase...")
    train_model(
        embeddings_dir=PROCESSED_DATA_DIR,
        output_dir=run_dir,
        model_params=model_params,
        mlflow_context=mlflow_context,
    )

    logger.success(f"Training pipeline completed successfully. Artifacts saved in {run_dir}")


if __name__ == "__main__":
    main()
