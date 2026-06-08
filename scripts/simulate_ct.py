#!/usr/bin/env python3
"""Simulasi Continuous Training closed-loop: deteksi drift → retrain → evaluasi → promosi."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from patent.config import CHUNK_SIZE, PROJ_ROOT, project_tempdir
from patent.lshiforest import LSHiForest, rescale_scores
from patent.modeling.evaluate import analyze_score_distribution
from patent.utils import convert_parquet_to_memmap, load_parquet_metadata


def _find_parquet_files(data_dir: Path) -> list[Path]:
    """Find all .parquet files in data_dir, excluding shifted data."""
    files = sorted(data_dir.glob("*.parquet"))
    return [f for f in files if "shifted" not in f.name.lower()]


def _load_or_train_baseline(
    data_dir: Path,
    output_dir: Path,
    skip_mlflow: bool = True,
    mlflow_experiment: str | None = None,
) -> tuple[LSHiForest, np.ndarray, Any, str]:
    """Load existing model or train a baseline if none exists."""
    model_path = output_dir / "baseline_model.lshif"
    scores_path = output_dir / "baseline_scores.npy"

    pq_files = _find_parquet_files(data_dir)
    if not pq_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    pq_paths = [str(p) for p in pq_files]
    metadata = load_parquet_metadata(pq_paths)

    if model_path.exists() and scores_path.exists():
        logger.info(f"Loading existing baseline model from {model_path}")
        model = LSHiForest.load(str(model_path))
        baseline_scores = np.load(str(scores_path))
        logger.info(f"Loaded baseline: {model.n_trees} trees, {len(baseline_scores)} scores")
        return model, baseline_scores, metadata, "loaded"

    logger.info("Training baseline model...")
    t0 = time.perf_counter()

    import shutil
    tmpdir = project_tempdir()
    try:
        mmap_path = str(tmpdir / "baseline.mmap")
        embedding_dim, total_rows = convert_parquet_to_memmap(pq_paths, mmap_path)
        embeddings = np.memmap(
            mmap_path, dtype=np.float32, mode="r", shape=(total_rows, embedding_dim)
        )

        model = LSHiForest(n_trees=100, max_depth=18, seed=42)
        model.fit(embeddings)
        fit_time = time.perf_counter() - t0
        logger.success(f"Baseline model trained in {fit_time:.2f}s")

        # Score
        t_score = time.perf_counter()
        baseline_scores = model.score_chunked(embeddings, total_rows, chunk_size=CHUNK_SIZE)
        score_time = time.perf_counter() - t_score
        logger.success(f"Baseline scoring in {score_time:.2f}s")

        # Save
        model.save(str(model_path))
        np.save(str(scores_path), baseline_scores)
        logger.info(f"Baseline saved to {output_dir}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return model, baseline_scores, metadata, "trained"


def _simulate_data_drift(
    data_dir: Path,
    output_dir: Path,
    baseline_model: LSHiForest,
    baseline_scores: np.ndarray,
    shift_strength: float = 0.6,
) -> dict[str, Any]:
    """Generate shifted data and detect drift against baseline."""
    from scripts.generate_shifted_data import (
        _load_embeddings,
        _save_shifted_parquet,
        apply_mean_shift,
    )
    from patent.monitoring.drift_detector import compare_score_distributions, detect_drift

    logger.info(f"Simulating data drift (strength={shift_strength})...")

    pq_files = _find_parquet_files(data_dir)
    if not pq_files:
        raise FileNotFoundError("No parquet files for shift generation")

    # Load a subset of original embeddings
    embeddings, metadata = _load_embeddings(pq_files[0], max_samples=500)
    shifted_embeddings = apply_mean_shift(embeddings, strength=shift_strength)

    # Save shifted data
    shifted_path = output_dir / "shifted_data.parquet"
    _save_shifted_parquet(shifted_embeddings, metadata, shifted_path)

    # Score shifted data with baseline model
    logger.info("Scoring shifted data with baseline model...")
    shifted_scores = baseline_model.score(shifted_embeddings)

    drift_result = detect_drift(baseline_scores[: len(shifted_scores)], shifted_scores)
    dist_comparison = compare_score_distributions(
        baseline_scores[: len(shifted_scores)], shifted_scores
    )

    baseline_dist = analyze_score_distribution(baseline_scores)
    shifted_dist = analyze_score_distribution(shifted_scores)

    result = {
        "shift_strength": shift_strength,
        "num_samples": len(shifted_embeddings),
        "shifted_data_path": str(shifted_path),
        "drift_detection": drift_result,
        "distribution_comparison": dist_comparison,
        "baseline_score_distribution": baseline_dist,
        "shifted_score_distribution": shifted_dist,
        "baseline_mean_score": float(np.mean(baseline_scores[np.isfinite(baseline_scores)])),
        "shifted_mean_score": float(np.mean(shifted_scores[np.isfinite(shifted_scores)])),
    }

    if drift_result.get("drift_detected"):
        logger.warning("DRIFT DETECTED! Model baseline tidak cocok dengan data baru.")
    else:
        logger.info("No significant drift detected.")

    return result


def _retrain_model(
    data_dir: Path,
    output_dir: Path,
    mlflow_uri: str | None = None,
    mlflow_experiment: str | None = None,
) -> dict[str, Any]:
    """Retrain model with all available data (including shifted)."""
    logger.info("Retraining model with updated dataset...")

    # Run the CLI train command
    import subprocess

    cmd = [
        sys.executable,
        str(PROJ_ROOT / "patent" / "cli.py"),
        "model", "train",
        str(data_dir),
        str(output_dir),
        "--num-trees", "100",
        "--max-depth", "18",
        "--seed", "123",
    ]

    if mlflow_experiment:
        cmd += ["--mlflow-experiment", mlflow_experiment]

    env = os.environ.copy()
    if mlflow_uri:
        env["MLFLOW_TRACKING_URI"] = mlflow_uri

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    logger.info("STDOUT:\n" + result.stdout[-2000:])
    if result.stderr:
        logger.info("STDERR:\n" + result.stderr[-1000:])

    if result.returncode != 0:
        logger.error(f"Training failed with exit code {result.returncode}")
        return {"success": False, "error": result.stderr}

    # Parse run_id
    import re
    run_id_match = re.search(r"MLflow run ID: ([a-f0-9]{32})", result.stdout)
    pyfunc_match = re.search(r"Pyfunc version: (\d+)", result.stdout)

    return {
        "success": True,
        "run_id": run_id_match.group(1) if run_id_match else None,
        "pyfunc_version": int(pyfunc_match.group(1)) if pyfunc_match else None,
    }


def _compare_and_promote(
    output_dir: Path,
    baseline_model: LSHiForest,
    baseline_scores: np.ndarray,
    new_model: LSHiForest,
    new_scores: np.ndarray,
) -> dict[str, Any]:
    """Compare new model against baseline; decide whether to promote."""
    from patent.modeling.evaluate import analyze_score_distribution
    from patent.monitoring.drift_detector import compare_score_distributions

    logger.info("Comparing new model vs baseline...")

    baseline_dist = analyze_score_distribution(baseline_scores)
    new_dist = analyze_score_distribution(new_scores)

    # Compare top anomalies overlap
    top_k = 100
    baseline_top = set(np.argsort(-baseline_scores)[:top_k])
    new_top = set(np.argsort(-new_scores)[:top_k])

    jaccard_top = (
        len(baseline_top & new_top) / len(baseline_top | new_top)
        if len(baseline_top | new_top) > 0
        else 1.0
    )

    # Compare distributions
    dist_comp = compare_score_distributions(baseline_scores, new_scores)

    # Decision criteria
    new_mean = new_dist.get("mean", 0)
    baseline_mean = baseline_dist.get("mean", 0)
    new_std = new_dist.get("std", 0)
    baseline_std = baseline_dist.get("std", 0)

    # A good model should have:
    # - Lower or comparable mean score (less anomalous overall)
    # - Higher standard deviation (better separation)
    # - Good Jaccard overlap (consistency)

    improvement_score = 0
    reasons = []

    if new_std > baseline_std * 0.95:
        improvement_score += 1
        reasons.append(f"STD improved or stable ({baseline_std:.4f} → {new_std:.4f})")

    if jaccard_top > 0.3:
        improvement_score += 1
        reasons.append(f"Jaccard@{top_k} overlap OK ({jaccard_top:.3f})")

    if new_mean < baseline_mean * 1.1:
        improvement_score += 1
        reasons.append(f"Mean score stable ({baseline_mean:.4f} → {new_mean:.4f})")

    promoted = improvement_score >= 2

    if promoted:
        # Save as new production model
        new_model_path = output_dir / "production_model.lshif"
        new_model.save(str(new_model_path))
        logger.success(f"Model promoted to Production → {new_model_path}")
    else:
        logger.info("New model NOT promoted — did not meet improvement criteria")

    return {
        "promoted_to_production": promoted,
        "improvement_score": improvement_score,
        "reasons": reasons,
        "baseline_mean": baseline_mean,
        "new_mean": new_mean,
        "baseline_std": baseline_std,
        "new_std": new_std,
        "jaccard_top_k": jaccard_top,
        "distribution_comparison": dist_comp,
    }


def _generate_report(
    output_dir: Path,
    baseline_info: dict[str, Any],
    drift_result: dict[str, Any],
    retrain_result: dict[str, Any],
    comparison_result: dict[str, Any],
    elapsed_s: float,
) -> Path:
    """Generate a comprehensive JSON report."""
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "simulation_duration_s": round(elapsed_s, 2),
        "baseline": baseline_info,
        "drift_detection": drift_result,
        "retraining": retrain_result,
        "comparative_evaluation": comparison_result,
        "trigger": {
            "type": "data_drift" if drift_result.get("drift_detection", {}).get("drift_detected")
            else "manual",
            "detected_at": datetime.now(timezone.utc).isoformat(),
        },
        "outcome": {
            "promoted": comparison_result.get("promoted_to_production", False),
            "summary": (
                "✅ Model baru DIPROMOSIKAN ke Production"
                if comparison_result.get("promoted_to_production")
                else "❌ Model baru TIDAK dipromosikan — metrik tidak membaik"
            ),
        },
    }

    report_path = output_dir / "ct_simulation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.success(f"Report saved to {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulasi Continuous Training closed-loop (end-to-end)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJ_ROOT / "data" / "processed",
        help="Directory containing processed .parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJ_ROOT / "models" / "ct-simulation",
        help="Output directory for models and reports",
    )
    parser.add_argument(
        "--shift-strength",
        type=float,
        default=0.6,
        help="Data shift strength for drift simulation (default: 0.6)",
    )
    parser.add_argument(
        "--skip-mlflow",
        action="store_true",
        default=True,
        help="Skip MLflow tracking (default: True for local simulation)",
    )
    parser.add_argument(
        "--mlflow-uri",
        default=None,
        help="MLflow tracking URI (required if --skip-mlflow not set)",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="ct-simulation",
        help="MLflow experiment name (default: ct-simulation)",
    )
    parser.add_argument(
        "--skip-retrain",
        action="store_true",
        help="Skip actual retraining (only detect drift)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    total_start = time.perf_counter()

    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  SIMULASI CONTINUOUS TRAINING (CLOSED-LOOP MLOps)")
    print("=" * 70)
    print(f"  Data directory:  {data_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Shift strength:  {args.shift_strength}")
    print(f"  MLflow:          {'enabled' if not args.skip_mlflow else 'disabled'}")
    print("=" * 70 + "\n")

    # ═════════════════════════════════════════════════════════════════════
    # 1. LOAD BASELINE MODEL
    # ═════════════════════════════════════════════════════════════════════
    print("─" * 70)
    print("  STEP 1: Memuat / Melatih Model Baseline")
    print("─" * 70)

    baseline_model, baseline_scores, metadata, baseline_source = _load_or_train_baseline(
        data_dir, output_dir,
        skip_mlflow=args.skip_mlflow,
        mlflow_experiment=args.mlflow_experiment if not args.skip_mlflow else None,
    )

    baseline_info = {
        "source": baseline_source,
        "n_trees": baseline_model.n_trees,
        "max_depth": baseline_model.max_depth,
        "family": baseline_model.family_name,
        "n_scores": len(baseline_scores),
        "mean_score": float(np.mean(baseline_scores[np.isfinite(baseline_scores)])),
        "std_score": float(np.std(baseline_scores[np.isfinite(baseline_scores)])),
    }
    print(f"  Baseline: {baseline_info['n_trees']} trees, "
          f"μ={baseline_info['mean_score']:.4f}, σ={baseline_info['std_score']:.4f}\n")

    # ═════════════════════════════════════════════════════════════════════
    # 2. SIMULATE DATA DRIFT
    # ═════════════════════════════════════════════════════════════════════
    print("─" * 70)
    print("  STEP 2: Simulasi Data Drift & Deteksi")
    print("─" * 70)

    drift_result = _simulate_data_drift(
        data_dir, output_dir,
        baseline_model, baseline_scores,
        shift_strength=args.shift_strength,
    )

    drift_detected = drift_result.get("drift_detection", {}).get("drift_detected", False)
    ks_pvalue = drift_result.get("drift_detection", {}).get("ks_pvalue", "N/A")
    w_dist = drift_result.get("drift_detection", {}).get("wasserstein_distance", "N/A")

    print(f"  Drift Detected:  {'⚠️  YA' if drift_detected else '✅ TIDAK'}")
    print(f"  KS test p-value: {ks_pvalue}")
    print(f"  Wasserstein d:   {w_dist}")
    print(f"  Baseline μ:      {drift_result['baseline_mean_score']:.4f}")
    print(f"  Shifted μ:       {drift_result['shifted_mean_score']:.4f}")
    print()

    # ═════════════════════════════════════════════════════════════════════
    # 3. RETRAIN (if drift detected or forced)
    # ═════════════════════════════════════════════════════════════════════
    retrain_result: dict[str, Any] = {"success": False, "skipped": False}

    if args.skip_retrain:
        print("─" * 70)
        print("  STEP 3: Retraining — DILEWATI (--skip-retrain)")
        print("─" * 70 + "\n")
        retrain_result["skipped"] = True
    elif drift_detected:
        print("─" * 70)
        print("  STEP 3: Retraining Model dengan Data Terbaru")
        print("─" * 70)

        retrain_result = _retrain_model(
            data_dir, output_dir,
            mlflow_uri=args.mlflow_uri,
            mlflow_experiment=args.mlflow_experiment if not args.skip_mlflow else None,
        )
    else:
        print("─" * 70)
        print("  STEP 3: Retraining — DILEWATI (tidak ada drift)")
        print("─" * 70 + "\n")
        retrain_result["skipped"] = True

    # ═════════════════════════════════════════════════════════════════════
    # 4. COMPARE & PROMOTE
    # ═════════════════════════════════════════════════════════════════════
    comparison_result: dict[str, Any] = {"skipped": True}

    if retrain_result.get("success"):
        print("─" * 70)
        print("  STEP 4: Evaluasi Komparatif — Model Baru vs Baseline")
        print("─" * 70)

        # Load new model and compute scores
        new_model_path = output_dir / "model.lshif"
        if new_model_path.exists():
            new_model = LSHiForest.load(str(new_model_path))

            # Score new model on ALL data
            import shutil
            tmpdir = project_tempdir()
            try:
                pq_files = _find_parquet_files(data_dir)
                pq_paths = [str(p) for p in pq_files]

                mmap_path = str(tmpdir / "compare.mmap")
                embedding_dim, total_rows = convert_parquet_to_memmap(pq_paths, mmap_path)
                embeddings = np.memmap(
                    mmap_path, dtype=np.float32, mode="r", shape=(total_rows, embedding_dim)
                )
                new_scores = new_model.score_chunked(embeddings, total_rows, chunk_size=CHUNK_SIZE)

                comparison_result = _compare_and_promote(
                    output_dir, baseline_model, baseline_scores, new_model, new_scores
                )

                print(f"  Promoted:        {'✅ YA' if comparison_result['promoted_to_production'] else '❌ TIDAK'}")
                for reason in comparison_result.get("reasons", []):
                    print(f"  ✓ {reason}")
                print(f"  Jaccard@{100}:    {comparison_result['jaccard_top_k']:.4f}")
                print()
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            logger.warning("New model not found — skipping comparison")
    else:
        print("─" * 70)
        print("  STEP 4: Evaluasi Komparatif — DILEWATI (retraining gagal/dilewati)")
        print("─" * 70 + "\n")

    # ═════════════════════════════════════════════════════════════════════
    # 5. GENERATE REPORT
    # ═════════════════════════════════════════════════════════════════════
    elapsed = time.perf_counter() - total_start

    print("=" * 70)
    print("  SIMULASI SELESAI")
    print("=" * 70)
    print(f"  Durasi:          {elapsed:.1f}s")
    print(f"  Drift terdeteksi: {'⚠️  YA' if drift_detected else '✅ TIDAK'}")
    print(f"  Model baru:       {'✅ Dipromosikan' if comparison_result.get('promoted_to_production') else '❌ Tidak dipromosikan'}")
    print("=" * 70 + "\n")

    report_path = _generate_report(
        output_dir, baseline_info, drift_result, retrain_result, comparison_result, elapsed
    )

    # Also print summary for easy copying
    print("\n" + json.dumps({
        "drift_detected": drift_detected,
        "promoted": comparison_result.get("promoted_to_production", False),
        "baseline_mean": baseline_info["mean_score"],
        "shifted_mean": drift_result.get("shifted_mean_score", 0),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
