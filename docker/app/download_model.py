#!/usr/bin/env python3
"""Download model.lshif + drift_baseline from MLflow to a target directory.

Requires the following environment variables:
    MLFLOW_TRACKING_URI   — MLflow tracking server URI
    AWS_ACCESS_KEY_ID     — S3 access key (for artifact store)
    AWS_SECRET_ACCESS_KEY — S3 secret key
    AWS_ENDPOINT_URL_S3   — S3 endpoint (e.g. Cloudflare R2)

Usage:
    python scripts/download_model.py \\
        --model-name patent-lshiforest \\
        --stage Production \\
        --output-dir /app/model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def download_model(
    output_dir: str,
    model_name: str = "patent-lshiforest",
    stage: str | None = "Production",
    version: str | None = None,
) -> str:
    """Download a model version and its drift baseline from the MLflow Registry.

    Parameters
    ----------
    output_dir : str
        Directory to write the model files into.
    model_name : str
        Registered model name in the MLflow Model Registry.
    stage : str | None
        Model stage to fetch (e.g. ``"Production"``, ``"Staging"``).
        Ignored when *version* is set.
    version : str | None
        Specific model version number.  Overrides *stage*.

    Returns
    -------
    str
        The downloaded model version string.
    """
    client = MlflowClient()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # ── Resolve the model version ───────────────────────────────────────
    if version:
        mv = client.get_model_version(model_name, version)
        print(f"Resolved {model_name} v{version}")
    else:
        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            raise RuntimeError(
                f"No '{stage}' version found for model '{model_name}'. "
                "Train and register a model first."
            )
        mv = versions[0]
        print(f"Resolved {model_name} {stage} → v{mv.version}")

    run_id = mv.run_id
    print(f"Run ID: {run_id}")

    # ── Download model.lshif ──────────────────────────────────────────
    model_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="model.lshif",
        dst_path=str(output),
    )
    print(f"Model downloaded to {model_path}")

    # ── Download drift baseline ─────────────────────────────────────────
    try:
        baseline_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="drift_baseline",
            dst_path=str(output),
        )
        print(f"Baseline downloaded to {baseline_path}")
    except Exception:
        print("No drift_baseline artifact found — drift detection will start "
              "gracefully with no baseline on first deploy.")

    # ── Write version file ──────────────────────────────────────────────
    (output / "model_version.txt").write_text(str(mv.version))
    print(f"Model version {mv.version} written to {output / 'model_version.txt'}")

    return str(mv.version)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download model + baseline from MLflow Model Registry"
    )
    parser.add_argument(
        "--model-name",
        default="patent-lshiforest",
        help="Registered model name (default: patent-lshiforest)",
    )
    parser.add_argument(
        "--version",
        default="Production",
        help="Model version or stage (default: Production). "
             "Numeric values are treated as version numbers; "
             "string values as stage names.",
    )
    parser.add_argument(
        "--output-dir",
        default="/app/model",
        help="Output directory for model files (default: /app/model)",
    )
    args = parser.parse_args()

    # Auto-detect: all-digits = version number, otherwise = stage name
    version: str | None = None
    stage: str | None = None
    if args.version.isdigit():
        version = args.version
    else:
        stage = args.version

    download_model(
        output_dir=args.output_dir,
        model_name=args.model_name,
        stage=stage,
        version=version,
    )


if __name__ == "__main__":
    main()
