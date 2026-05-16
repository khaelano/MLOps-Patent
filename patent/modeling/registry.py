"""MLflow Model Registry helpers for the LSHiForest anomaly model.

Compares evaluation metrics against the latest Production version and
promotes new versions when they improve.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
import mlflow
from mlflow.tracking import MlflowClient


def register_from_run(
    run_id: str,
    model_name: str = "patent-lshiforest",
    metric_key: str = "stability/jaccard_aggregated",
) -> dict[str, Any]:
    """Register a completed MLflow run's model and promote if better.

    Always creates a new registered model version from the model artifact
    logged during training.  Then compares *metric_key* against the latest
    Production version's value.  If the new value is higher, the new version
    is transitioned to **Production** and the previous Production version is
    archived.

    Parameters
    ----------
    run_id : str
        MLflow run ID from a completed ``model train`` invocation.
    model_name : str
        Name used in the MLflow Model Registry.
    metric_key : str
        Flattened evaluation metric key logged during training
        (default ``"stability/jaccard_aggregated"``).

    Returns
    -------
    dict
        Summary with keys ``model_name``, ``version``, ``metric_value``,
        ``promoted_to_production``, ``previous_prod_version``, and
        ``previous_metric_value``.
    """
    client = MlflowClient()
    artifact_uri = f"runs:/{run_id}/model.lshif"

    # ── Retrieve the new model's evaluation metric from the run ──────────
    run = client.get_run(run_id)
    run_metrics = run.data.metrics
    new_metric = run_metrics.get(metric_key)
    if new_metric is None:
        available = sorted(run_metrics.keys())
        logger.warning(
            f"Metric '{metric_key}' not found in run {run_id}. Available metrics: {available}"
        )
        new_metric = 0.0

    logger.info(
        f"Registering model '{model_name}' from run {run_id} ({metric_key}={new_metric:.4f})"
    )

    # ── Register the new version ─────────────────────────────────────────
    try:
        result = mlflow.register_model(model_uri=artifact_uri, name=model_name)
    except Exception:
        # mlflow.register_model may raise if the model name doesn't exist yet;
        # MlflowClient.create_model_version + create_registered_model
        # handles the first-registration edge-case.
        logger.info("Model name may not exist yet — creating via MlflowClient...")
        try:
            client.create_registered_model(model_name)
        except Exception:
            # Already exists (race) — ignore
            pass
        result = client.create_model_version(
            name=model_name,
            source=artifact_uri,
            run_id=run_id,
        )

    new_version = result.version
    logger.success(f"Registered {model_name} version {new_version}")

    # ── Compare with latest Production version ───────────────────────────
    latest_prod = client.get_latest_versions(model_name, stages=["Production"])
    promoted = False
    previous_version: str | None = None
    previous_metric: float | None = None

    if latest_prod:
        prod = latest_prod[0]
        previous_version = prod.version
        # Fetch the metrics from the production version's run
        prod_run = client.get_run(prod.run_id)
        prod_metrics = prod_run.data.metrics
        previous_metric = prod_metrics.get(metric_key)

        if previous_metric is None:
            logger.warning(
                f"Production version {prod.version} has no '{metric_key}' metric. "
                "Promoting new version by default."
            )
            promoted = True
        elif new_metric > previous_metric:
            logger.info(
                f"New version improves {metric_key}: {previous_metric:.4f} → {new_metric:.4f}"
            )
            promoted = True
        else:
            logger.info(
                f"New version does NOT improve {metric_key}: "
                f"{new_metric:.4f} ≤ {previous_metric:.4f} "
                f"(production v{prod.version})"
            )
    else:
        # No Production version exists — promote the first one
        logger.info("No existing Production version — promoting first version.")
        promoted = True

    if promoted:
        client.transition_model_version_stage(
            name=model_name,
            version=new_version,
            stage="Production",
        )
        logger.success(f"Promoted {model_name} v{new_version} to Production")
        # Archive the previous production version if it existed
        if previous_version:
            client.transition_model_version_stage(
                name=model_name,
                version=previous_version,
                stage="Archived",
            )
            logger.info(f"Archived previous Production {model_name} v{previous_version}")

    return {
        "model_name": model_name,
        "version": new_version,
        "metric_key": metric_key,
        "metric_value": new_metric,
        "promoted_to_production": promoted,
        "previous_prod_version": previous_version,
        "previous_metric_value": previous_metric,
    }
