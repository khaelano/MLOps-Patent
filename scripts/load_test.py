#!/usr/bin/env python3
"""Load-test the inference API with realistic patent texts.

Generates traffic against the inference-lb service so you can observe
latency, throughput, and anomaly-score distributions on the Grafana dashboard
(``MLOps — Patent Anomaly Detection``).

Usage (from the *host* while services are running)::

    # Default: 10 concurrent workers, 30 seconds, targeting localhost:8000
    python scripts/load_test.py

    # Custom concurrency, duration, and target URL
    python scripts/load_test.py --workers 20 --duration 60 --url http://localhost:8000

    # Use real texts from a processed parquet file
    python scripts/load_test.py --data data/processed/arxiv-metadata-oai-snapshot.parquet

Requirements (already in pyproject.toml): ``requests``, ``pandas``, ``pyarrow``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any


# ── Synthetic fallback texts used when no --data file is provided ──────────
_FALLBACK_TEXTS: list[str] = [
    (
        "Deep Learning Approaches for Natural Language Processing "
        "This paper presents a comprehensive survey of deep learning methods "
        "applied to natural language processing tasks including sentiment "
        "analysis, machine translation, and question answering. We compare "
        "transformer-based architectures with recurrent and convolutional "
        "neural networks across several benchmark datasets."
    ),
    (
        "Quantum Computing: A New Paradigm for Optimization Problems "
        "We investigate the application of quantum annealing and variational "
        "quantum algorithms to combinatorial optimization problems. Our "
        "results demonstrate significant speedup compared to classical "
        "simulated annealing on MAX-CUT and graph partitioning benchmarks."
    ),
    (
        "Federated Learning with Differential Privacy Guarantees "
        "This work proposes a federated learning framework that provides "
        "formal differential privacy guarantees while maintaining model "
        "utility. We introduce adaptive clipping and noise-calibration "
        "strategies that reduce the privacy-utility trade-off by 15% "
        "compared to previous approaches."
    ),
    (
        "Graph Neural Networks for Molecular Property Prediction "
        "We develop a novel graph neural network architecture specifically "
        "designed for predicting quantum mechanical properties of molecules. "
        "Our message-passing scheme incorporates bond angles and dihedral "
        "torsion information, achieving state-of-the-art results on the "
        "QM9 and MD17 benchmarks."
    ),
    (
        "Reinforcement Learning for Autonomous Navigation in Dynamic "
        "Environments. This paper addresses the challenge of autonomous "
        "robot navigation in crowded dynamic environments using deep "
        "reinforcement learning. We propose a multi-agent training scheme "
        "with curriculum learning that enables safe and efficient navigation "
        "among moving pedestrians."
    ),
    (
        "Efficient Transformers via Sparse Attention Mechanisms "
        "We introduce a family of sparse attention patterns that reduce the "
        "quadratic complexity of standard transformers to O(n log n) while "
        "preserving model quality. Our approach combines local window "
        "attention with randomly sampled global connections."
    ),
    (
        "Adversarial Robustness of Vision Transformers "
        "This study empirically evaluates the adversarial robustness of "
        "Vision Transformer (ViT) architectures compared to convolutional "
        "neural networks. We find that ViTs demonstrate inherent robustness "
        "to certain perturbation types but remain vulnerable to carefully "
        "crafted adversarial patches."
    ),
    (
        "Causal Discovery from Observational Data using Neural Networks "
        "We propose a score-based method for causal structure learning that "
        "leverages neural networks to model complex non-linear relationships. "
        "Our approach outperforms constraint-based methods on datasets with "
        "non-Gaussian noise and non-linear dependencies."
    ),
    (
        "Self-Supervised Learning for Medical Image Segmentation "
        "This paper presents a self-supervised pre-training strategy for "
        "medical image segmentation that leverages unlabeled CT and MRI "
        "scans. Our contrastive learning framework learns anatomical "
        "representations that transfer effectively to downstream segmentation "
        "tasks with limited annotated data."
    ),
    (
        "Blockchain-Based Federated Learning with Verifiable Aggregation "
        "We design a blockchain-based protocol for federated learning that "
        "provides verifiable model aggregation without requiring trust in "
        "a central server. Smart contracts enforce correct aggregation, and "
        "zero-knowledge proofs ensure privacy of individual model updates."
    ),
]


def _load_texts_from_parquet(path: str, max_samples: int = 500) -> list[str]:
    """Extract *max_samples* random texts from a processed parquet file.

    Expects columns ``title`` and ``abstract``; concatenates them as
    ``f\"{title} {abstract}\"``, matching the training pipeline format.
    """
    import pandas as pd

    df = pd.read_parquet(path)
    n = min(len(df), max_samples)

    if "title" in df.columns and "abstract" in df.columns:
        sampled = df.sample(n=n, random_state=42)
        texts = (sampled["title"].fillna("") + " " + sampled["abstract"].fillna("")).tolist()
    elif "texts" in df.columns:
        sampled = df.sample(n=n, random_state=42)
        texts = sampled["texts"].fillna("").tolist()
    else:
        # Fallback: use any available text-like column
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if text_cols:
            sampled = df.sample(n=n, random_state=42)
            texts = sampled[text_cols[0]].fillna("").astype(str).tolist()
        else:
            print(f"[warn] No text columns found in {path}, using fallback texts")
            return _FALLBACK_TEXTS.copy()

    print(f"[info] Loaded {len(texts)} texts from {path}")
    return texts


def _send_batch(
    url: str,
    texts: list[str],
    timeout: float = 60.0,
) -> tuple[float, int, bool]:
    """Send one batch prediction request.  Returns (latency_s, num_texts, success)."""
    import requests

    start = time.perf_counter()
    try:
        resp = requests.post(
            url,
            json={"texts": texts},
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
        latency = time.perf_counter() - start
        success = resp.status_code == 200
        if not success:
            print(f"[warn] HTTP {resp.status_code}: {resp.text[:200]}")
        return latency, len(texts), success
    except requests.exceptions.RequestException as exc:
        latency = time.perf_counter() - start
        print(f"[warn] Request failed: {exc}")
        return latency, len(texts), False


def run_load_test(
    url: str,
    texts: list[str],
    *,
    workers: int = 10,
    duration: float = 30.0,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Continuously send prediction requests for *duration* seconds.

    Parameters
    ----------
    url : str
        Target endpoint (e.g. ``http://localhost:8000/predict``).
    texts : list[str]
        Pool of input texts sampled randomly for each batch.
    workers : int
        Number of concurrent threads.
    duration : float
        How long to run (seconds).
    batch_size : int
        Number of texts per batch.

    Returns
    -------
    dict
        Summary with keys ``total_requests``, ``total_texts``, ``success_rate``,
        ``latency_p50``, ``latency_p95``, ``latency_p99``, ``throughput_rps``.
    """
    latencies: list[float] = []
    total_texts_scored = 0
    total_requests = 0
    successes = 0
    deadline = time.perf_counter() + duration

    print(f"\n{'='*70}")
    print(f"  Load Test: {workers} workers, {duration}s, batch_size={batch_size}")
    print(f"  Target: {url}")
    print(f"  Text pool: {len(texts)} samples")
    print(f"{'='*70}\n")

    def _worker() -> tuple[float, int, bool]:
        return _send_batch(url, random.sample(texts, min(batch_size, len(texts))))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures: list[concurrent.futures.Future] = []
        # Pre-fill the pipeline
        for _ in range(workers * 2):
            futures.append(executor.submit(_worker))

        reported = time.perf_counter()
        while time.perf_counter() < deadline:
            # Submit a new future for each completed one to keep workers busy
            done_futures: set[concurrent.futures.Future] = set()
            for f in futures:
                if f.done():
                    latency, n_texts, ok = f.result()
                    latencies.append(latency)
                    total_texts_scored += n_texts
                    total_requests += 1
                    if ok:
                        successes += 1
                    done_futures.add(f)

            futures = [f for f in futures if f not in done_futures]

            # Refill the queue
            while len(futures) < workers * 2 and time.perf_counter() < deadline:
                futures.append(executor.submit(_worker))

            # Progress report every 10 seconds
            now = time.perf_counter()
            if now - reported >= 10:
                elapsed = now - (deadline - duration)
                rps = total_requests / elapsed if elapsed > 0 else 0
                print(
                    f"  [{elapsed:5.0f}s]  requests={total_requests:5d}  "
                    f"texts={total_texts_scored:6d}  rps={rps:5.1f}  "
                    f"ok={successes}/{total_requests}"
                )
                reported = now

            time.sleep(0.05)

        # Drain remaining futures
        for f in futures:
            latency, n_texts, ok = f.result()
            latencies.append(latency)
            total_texts_scored += n_texts
            total_requests += 1
            if ok:
                successes += 1

    elapsed = time.perf_counter() - (deadline - duration)

    if not latencies:
        print("[error] No requests completed — is the service running?")
        sys.exit(1)

    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[int(len(latencies_sorted) * 0.50)]
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
    p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]
    avg = statistics.mean(latencies)
    rps = total_requests / elapsed if elapsed > 0 else 0
    tps = total_texts_scored / elapsed if elapsed > 0 else 0

    summary = {
        "total_requests": total_requests,
        "total_texts": total_texts_scored,
        "success_rate": successes / total_requests if total_requests else 0,
        "elapsed_s": round(elapsed, 2),
        "latency_avg_s": round(avg, 4),
        "latency_p50_s": round(p50, 4),
        "latency_p95_s": round(p95, 4),
        "latency_p99_s": round(p99, 4),
        "throughput_req_per_sec": round(rps, 1),
        "throughput_texts_per_sec": round(tps, 1),
    }

    print(f"\n{'='*70}")
    print("  Results")
    print(f"{'='*70}")
    print(f"  Duration:          {elapsed:.1f}s")
    print(f"  Total requests:    {total_requests}")
    print(f"  Total texts:       {total_texts_scored}")
    print(f"  Success rate:      {summary['success_rate']:.2%}")
    print(f"  Throughput:        {rps:.1f} req/s  ({tps:.1f} texts/s)")
    print(f"  Latency (avg):     {avg*1000:.1f} ms")
    print(f"  Latency (p50):     {p50*1000:.1f} ms")
    print(f"  Latency (p95):     {p95*1000:.1f} ms")
    print(f"  Latency (p99):     {p99*1000:.1f} ms")
    print(f"\n  View dashboard:    http://localhost:3000")
    print(f"{'='*70}\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load-test the inference API and generate Prometheus metrics"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000/predict",
        help="Inference endpoint URL (default: http://localhost:8000/predict)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of concurrent worker threads (default: 10)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Test duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Texts per batch (default: 4)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to a processed .parquet file with 'title'+'abstract' or 'texts' column",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save summary JSON to this file (optional)",
    )
    args = parser.parse_args()

    # ── Load text pool ──────────────────────────────────────────────────
    if args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"[error] Data file not found: {args.data}")
            sys.exit(1)
        texts = _load_texts_from_parquet(str(data_path))
    else:
        print("[info] No --data provided, using built-in synthetic texts")
        texts = _FALLBACK_TEXTS.copy()

    # ── Run load test ───────────────────────────────────────────────────
    summary = run_load_test(
        url=args.url,
        texts=texts,
        workers=args.workers,
        duration=args.duration,
        batch_size=args.batch_size,
    )

    # ── Persist summary if requested ────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(summary, indent=2))
        print(f"Summary written to {output_path}")


if __name__ == "__main__":
    main()
