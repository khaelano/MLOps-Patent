from pathlib import Path

import joblib
from loguru import logger
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
import typer

from patent.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")
    # -----------------------------------------


def load_pca_model(model_path: str = "models/pca_model.joblib") -> IncrementalPCA:
    """
    Load a trained PCA model.
    """
    logger.info(f"Loading PCA model from {model_path}")
    return joblib.load(model_path)


if __name__ == "__main__":
    app()
