import joblib
from loguru import logger
import numpy as np
from sklearn.decomposition import IncrementalPCA


def train_pca_model(
    embeddings: np.ndarray,
    n_components: int = 128,
    batch_size: int = 50000,
    model_save_path: str = "models/pca_model.joblib",
    sample_size: int = 200000,
) -> IncrementalPCA:
    """
    Train an IncrementalPCA model from a full numpy array of embeddings, optionally using a sample for speed.
    """
    logger.info(f"Training PCA model with {n_components} components")
    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    if sample_size and sample_size < embeddings.shape[0]:
        logger.info(
            f"Sampling {sample_size} out of {embeddings.shape[0]} embeddings for PCA training"
        )
        rng = np.random.default_rng(42)
        idx = rng.choice(embeddings.shape[0], size=sample_size, replace=False)
        train_emb = embeddings[idx]
    else:
        train_emb = embeddings

    logger.debug(f"Fitting PCA on {train_emb.shape[0]} samples")
    pca.fit(train_emb)

    logger.info(f"Saving trained PCA model to {model_save_path}")
    joblib.dump(pca, model_save_path)
    return pca
