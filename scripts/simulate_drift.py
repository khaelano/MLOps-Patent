#!/usr/bin/env python3
"""Simulate data drift by sending non-academic text to the inference API.

Sends random nonsense / non-technical text in batches to skew the
anomaly score distribution.  This is designed to trigger the Grafana
drift alert when the KS test p-value drops below 0.05.

The script works by:
1. Sending normal-looking paper-like text first to establish a scoring baseline
2. Then flooding the API with random words / garbage text
3. The garbage text will have very different embedding vectors → different
   anomaly scores → distribution shift detected by drift checks

Usage:
    python scripts/simulate_drift.py --url http://localhost:8000 --batches 50
"""

from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor

import requests

# ── Word pools ────────────────────────────────────────────────────────────────

# Pool A: Realistic paper-like text (similar to training distribution)
NORMAL_TEXTS = [
    "A novel approach to gradient descent optimization in deep neural networks",
    "Quantum entanglement for secure communication protocols",
    "Transformer architectures for natural language understanding tasks",
    "Probabilistic graphical models for causal inference with applications",
    "Deep reinforcement learning with continuous action spaces for robotics",
    "Attention mechanisms in sequence-to-sequence learning for translation",
    "Federated learning with differential privacy guarantees",
    "Graph neural networks for molecular property prediction",
    "Bayesian optimization for hyperparameter tuning in large-scale models",
    "Self-supervised contrastive learning for visual representation",
    "Stochastic gradient descent with momentum acceleration and variance reduction",
    "Generative adversarial networks for high-fidelity image synthesis",
    "Meta-learning for few-shot classification with optimization-based methods",
    "Neural ordinary differential equations for continuous-depth models",
    "Variational autoencoders with disentangled latent representations",
    "Sparse coding and dictionary learning for signal processing applications",
    "Kernel methods for nonlinear dimensionality reduction and manifold learning",
    "Information-theoretic bounds for generalization in overparameterized models",
    "Adversarial robustness through certified defenses and Lipschitz regularization",
    "Diffusion probabilistic models for high-quality unconditional image generation",
]

# Pool B: Random garbage text (designed to be far from training distribution)
NONSENSE_TEXTS = [
    "asdfghjkl qwertyuiop zxcvbnm 1234567890",
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod",
    "xyzzy plugh plover nothing happens twice in the same river",
    "foo bar baz qux quux corge grault garply waldo fred plugh",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "random words that make no sense whatsoever in any known language",
    "glorp zibber flonk nogwump shizzle wibble fizzbang",
    "the quick brown fox jumps over the lazy dog repeatedly forever",
    "",
    "!@#$%%^&*()_+{}[]|\\:\";'<>?,./~`",
    "test test test test test test test test test test test test test test test",
    "error error error error error error error error error error error error",
    "null null null null null null null null null null null null null null",
    "undefined is not a function undefined is not a function",
    "segmentation fault core dumped memory access violation",
    "0xDEADBEEF 0xCAFEBABE 0x8BADF00D 0xBAADF00D",
    "TODO: fix this later FIXME: this is broken HACK: works for now",
    "ping pong ping pong ping pong ping pong ping pong ping pong ping pong",
    "aaaaabbbbbcccccdddddeeeeefffffggggghhhhhiiiiijjjjj",
    "..... ..... ..... ..... ..... ..... ..... ..... .....",
]


def make_batch(size: int, nonsense_ratio: float = 0.8) -> list[str]:
    """Create a batch of *size* texts with the given ratio of nonsense."""
    n_nonsense = int(size * nonsense_ratio)
    n_normal = size - n_nonsense

    batch = random.choices(NONSENSE_TEXTS, k=n_nonsense)
    batch += random.choices(NORMAL_TEXTS, k=n_normal)
    random.shuffle(batch)
    return batch


def send_batch(url: str, texts: list[str]) -> dict:
    """Send one batch of texts to the /predict endpoint."""
    resp = requests.post(
        f"{url}/predict",
        json={"texts": texts},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(
        description="Simulate data drift by sending non-academic text to the inference API"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Inference API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=50,
        help="Number of batches to send (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Texts per batch (default: 32)",
    )
    parser.add_argument(
        "--nonsense-ratio",
        type=float,
        default=0.8,
        help="Ratio of nonsense to normal text (default: 0.8)",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=5,
        help="Number of warmup batches with all-normal text (default: 5)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent worker threads (default: 4)",
    )
    args = parser.parse_args()

    url = args.url.rstrip("/")

    # ── Health check ─────────────────────────────────────────────────────
    print(f"Checking health at {url}/health ...")
    try:
        health = requests.get(f"{url}/health", timeout=10).json()
        print(f"  Status: {health['status']}")
        print(f"  Model: {health['model_name']} v{health['model_version']}")
        print(f"  Embedder: {health['embedder']}")
    except Exception as e:
        print(f"  ERROR: Cannot reach inference API: {e}")
        return 1

    print(f"\nSending {args.batches} batches ({args.batch_size} texts each) ...")
    print(f"  Nonsense ratio: {args.nonsense_ratio}")
    print(f"  Workers: {args.workers}")
    print()

    # ── Warmup: establish baseline with normal text ──────────────────────
    print(f"Phase 1: Warmup — {args.warmup_batches} batches of normal text")
    for i in range(args.warmup_batches):
        batch = random.choices(NORMAL_TEXTS, k=args.batch_size)
        try:
            result = send_batch(url, batch)
            scores = result["scores"]
            print(
                f"  Warmup {i + 1}/{args.warmup_batches}: "
                f"mean={sum(scores)/len(scores):.4f}, "
                f"max={max(scores):.4f}, min={min(scores):.4f}"
            )
        except Exception as e:
            print(f"  Warmup {i + 1} FAILED: {e}")
        time.sleep(0.5)

    # ── Drift simulation: flood with nonsense ────────────────────────────
    print(f"\nPhase 2: Drift simulation — {args.batches} batches")
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for i in range(args.batches):
            batch = make_batch(args.batch_size, nonsense_ratio=args.nonsense_ratio)
            futures.append(pool.submit(send_batch, url, batch))

        for i, future in enumerate(futures):
            try:
                result = future.result()
                scores = result["scores"]
                mean_score = sum(scores) / len(scores)
                print(
                    f"  Batch {i + 1}/{args.batches}: "
                    f"mean={mean_score:.4f}, max={max(scores):.4f}, "
                    f"min={min(scores):.4f}"
                )
            except Exception as e:
                print(f"  Batch {i + 1} FAILED: {e}")

    elapsed = time.perf_counter() - t0
    print(f"\nDone! {args.batches} batches in {elapsed:.1f}s")

    # ── Check metrics endpoint ───────────────────────────────────────────
    print(f"\nChecking /metrics endpoint ...")
    try:
        resp = requests.get(f"{url}/metrics", timeout=10)
        if resp.status_code == 200:
            for line in resp.text.split("\n"):
                if line.startswith("patent_drift_score_ks_"):
                    print(f"  {line}")
        else:
            print(f"  Status: {resp.status_code}")
    except Exception as e:
        print(f"  Metrics check failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
