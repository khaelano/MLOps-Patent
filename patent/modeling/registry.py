"""MLflow Model Registry helpers for the LSHiForest anomaly model.

Compares evaluation metrics against the latest Production version and
promotes new versions when they improve.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from mlflow.tracking import MlflowClient


def register_from_run(
    run_id: str,
    model_name: str = "patent-lshiforest",
    metric_key: str = "stability/jaccard_aggregated",
    pyfunc_version: int | None = None,
) -> dict[str, Any]:
    """Compare evaluation metrics and promote the registered pyfunc model if better.

    The model must already be registered via ``mlflow.pyfunc.log_model`` with
    ``registered_model_name`` (which creates the version automatically in
    MLflow 3.x).  This function compares *metric_key* against the latest
    Production version and transitions stages accordingly.

    Parameters
    ----------
    run_id : str
        MLflow run ID from a completed ``model train`` invocation (used to
        read evaluation metrics).
    model_name : str
        Name used in the MLflow Model Registry.
    metric_key : str
        Flattened evaluation metric key logged during training
        (default ``\"stability/jaccard_aggregated\"``).
    pyfunc_version : int | None
        The version number that was auto-created by ``log_model``.  If
        ``None`` the latest non-Production version is used as a fallback.

    Returns
    -------
    dict
        Summary with keys ``model_name``, ``version``, ``metric_value``,
        ``promoted_to_production``, ``previous_prod_version``, and
        ``previous_metric_value``.
    """
    client = MlflowClient()

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

    # ── Resolve the version to compare ────────────────────────────────────
    if pyfunc_version is not None:
        new_version = str(pyfunc_version)
        logger.info(
            f"Using pyfunc model v{new_version} for '{model_name}' ({metric_key}={new_metric:.4f})"
        )
    else:
        # Fallback: find the latest version that isn't Production
        all_versions = client.search_model_versions(f"name='{model_name}'")
        non_prod = [v for v in all_versions if v.current_stage != "Production"]
        if non_prod:
            new_version = non_prod[-1].version
        else:
            logger.error(f"No non-Production version found for '{model_name}'")
            return {
                "model_name": model_name,
                "version": "0",
                "metric_key": metric_key,
                "metric_value": new_metric,
                "promoted_to_production": False,
                "previous_prod_version": None,
                "previous_metric_value": None,
            }
        logger.warning(f"No pyfunc_version provided — using latest v{new_version}")

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
