#!/usr/bin/env python3
"""Download the latest Production LSHiForest model artifact from the MLflow Registry.

Writes the ``.lshif`` file and a version metadata JSON to an output directory.
Used by ``make docker-build`` to stage model artifacts before the Docker build.

Usage::

    uv run python scripts/download_model.py -o docker/app/model

Output (written to *output_dir*):

    model.lshif          The LSHiForest model file
    version.json         ``{"version": "7", "run_id": "abc123", "name": "patent-lshiforest"}``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def download(output_dir: str | Path, model_name: str | None = None) -> dict[str, str]:
    """Download the latest Production model and return version metadata.

    Parameters
    ----------
    output_dir : str or Path
        Directory where ``model.lshif`` and ``version.json`` are written.
    model_name : str or None
        Registered model name (default from ``MLFLOW_MODEL_NAME`` env var
        or ``"patent-lshiforest"``).

    Returns
    -------
    dict
        ``{"version": str, "run_id": str, "name": str}``
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    model_name = model_name or os.getenv("MLFLOW_MODEL_NAME", "patent-lshiforest")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not prod_versions:
        print(
            f"ERROR: No Production version found for model '{model_name}'.\n"
            "Train and register a model first.",
            file=sys.stderr,
        )
        sys.exit(1)

    prod = prod_versions[0]
    version = str(prod.version)
    run_id = prod.run_id

    print(f"Found {model_name} v{version} (run {run_id}) in Production")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = str(output_dir / "model.lshif")

    artifact_uri = f"runs:/{run_id}/model.lshif"
    print(f"Downloading artifact {artifact_uri} → {dest}")

    # download_artifacts expects a *directory* for dst_path; it places the
    # artifact inside using its relative path.  Download into a temp dir
    # and move the file to the expected name.
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=artifact_uri,
            dst_path=tmpdir,
        )
        actual = str(local_path) if isinstance(local_path, str) else str(local_path)

        # Remove any previous download to avoid shutil.move conflicts
        if Path(dest).exists():
            Path(dest).unlink()

        import shutil

        shutil.move(actual, dest)

    meta: dict[str, str] = {
        "version": version,
        "run_id": run_id,
        "name": model_name,
    }
    meta_path = output_dir / "version.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Model downloaded → {dest}")
    print(f"Metadata written → {meta_path}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Production LSHiForest model from MLflow Registry"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="docker/app/model",
        help="Output directory (default: docker/app/model)",
    )
    parser.add_argument(
        "-n",
        "--model-name",
        default=None,
        help="Registered model name (default from MLFLOW_MODEL_NAME env or patent-lshiforest)",
    )
    args = parser.parse_args()

    meta = download(args.output, args.model_name)
    print(f"\nModel v{meta['version']} ready for Docker build.")


if __name__ == "__main__":
    main()
