from loguru import logger
import mlflow
import numpy as np


def test_registry_inference(model_uri: str = "models:/LSH-IForest/2"):
    mlflow.set_tracking_uri("http://localhost:5000")
    logger.info(f"Loading model from registry: {model_uri}")

    model = mlflow.pyfunc.load_model(model_uri=model_uri)

    logger.success("Model loaded successfully.")

    batch_size = 5
    embedding_dim = 384

    # Random float32 array (standard normal distribution)
    dummy_embeddings = np.random.randn(batch_size, embedding_dim).astype(np.float32)

    logger.info(f"Generated dummy embeddings with shape: {dummy_embeddings.shape}")

    logger.info("Running inference...")
    predictions = model.predict(dummy_embeddings)

    logger.info("=== Inference Results ===")
    for i, score in enumerate(predictions):
        status = "ANOMALOUS" if score > 0.6 else "NORMAL"
        logger.info(f"Sample {i + 1}: Score={score:.4f} | Status={status}")

    print("\nRaw Scores:", predictions)

    return predictions


if __name__ == "__main__":
    test_registry_inference(model_uri="models:/LSH-IForest/latest")
