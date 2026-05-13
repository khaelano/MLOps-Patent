"""
Profiling harness for MLOps-Patent pipeline stages.

Profiles data preprocessing, embedding, training, and evaluation stages
using psutil for system-level resource monitoring and time.perf_counter
for wall-clock precision.

Usage:
    uv run python pipelines/profile.py [--output reports/figures/profile_report.md]
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from loguru import logger as _logger
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from patent.config import RAW_DATA_DIR, project_tempdir
from patent.dataset.embedders import get_embedder
from patent.dataset.preprocess import clean_df, parse_oai_xml_directory
from patent.lshiforest import LSHiForest
from patent.modeling.evaluate import (
    analyze_score_distribution,
    evaluate_params,
    export_top_anomalies,
)
from patent.utils import (
    convert_parquet_to_memmap,
    load_parquet_metadata,
    mute_logging,
)

# -- ensure the project root is on sys.path -----------------------------------
PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

# ---------------------------------------------------------------------------
# Profiling data structures
# ---------------------------------------------------------------------------


@dataclass
class IntervalSample:
    """A single resource snapshot."""

    elapsed_s: float
    cpu_pct: float  # process CPU %
    rss_mib: float  # resident set size in MiB
    vms_mib: float  # virtual memory size in MiB


@dataclass
class StageProfile:
    """Aggregate metrics for one pipeline stage."""

    stage: str
    wall_clock_s: float
    cpu_mean_pct: float
    cpu_peak_pct: float
    rss_peak_mib: float
    rss_mean_mib: float
    vms_peak_mib: float
    samples: list[IntervalSample] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Resource sampling
# ---------------------------------------------------------------------------


def _mb(b: int) -> float:
    """Bytes → MiB."""
    return b / (1024 * 1024)


class ResourceProbe:
    """Background CPU + memory sampler tied to *proc*."""

    def __init__(self, proc: psutil.Process, interval: float = 0.1) -> None:
        self._proc = proc
        self._interval = interval
        self._samples: list[IntervalSample] = []
        self._start: float | None = None

    def start(self) -> None:
        self._start = time.perf_counter()
        # Prime psutil's CPU measurement
        self._proc.cpu_percent(interval=None)

    def sample(self) -> IntervalSample:
        """Take one snapshot (call from the main thread periodically)."""
        assert self._start is not None
        cpu = self._proc.cpu_percent(interval=None)
        mem = self._proc.memory_info()
        s = IntervalSample(
            elapsed_s=time.perf_counter() - self._start,
            cpu_pct=cpu,
            rss_mib=_mb(mem.rss),
            vms_mib=_mb(mem.vms),
        )
        self._samples.append(s)
        return s

    def stop(self) -> list[IntervalSample]:
        self.sample()  # final snapshot
        return self._samples


# Module-level slot so callers can retrieve the last StageProfile without
# relying on dynamic function-attribute assignment (which type-checkers reject).
_last_profile: StageProfile | None = None


@contextmanager
def profiled_stage(stage_name: str, interval: float = 0.5):
    """Context manager that samples resources at *interval* seconds.

    Yields a *ResourceProbe*.  The caller should call ``probe.sample()``
    at key checkpoints inside the block in addition to the automatic
    periodic sampling done by the manager thread.
    """
    proc = psutil.Process(os.getpid())
    probe = ResourceProbe(proc)
    probe.start()

    t0 = time.perf_counter()
    _logger.info(f"[PROFILE] {stage_name} started")

    # We'll use a simple inline sampling loop in a helper thread.
    # For simplicity, we just sample at boundaries and let the
    # caller manually sample inside long loops.
    import threading

    stop_flag = threading.Event()

    def _bg_sampler() -> None:
        while not stop_flag.is_set():
            probe.sample()
            stop_flag.wait(interval)

    bg = threading.Thread(target=_bg_sampler, daemon=True)
    bg.start()

    try:
        yield probe
    finally:
        stop_flag.set()
        bg.join(timeout=2)
        probe.stop()
        elapsed = time.perf_counter() - t0

        # Build aggregate
        if probe._samples:
            cpus = [s.cpu_pct for s in probe._samples]
            rsss = [s.rss_mib for s in probe._samples]
            vmss = [s.vms_mib for s in probe._samples]
            stage = StageProfile(
                stage=stage_name,
                wall_clock_s=elapsed,
                cpu_mean_pct=float(np.mean(cpus)),
                cpu_peak_pct=float(np.max(cpus)),
                rss_peak_mib=float(np.max(rsss)),
                rss_mean_mib=float(np.mean(rsss)),
                vms_peak_mib=float(np.max(vmss)),
                samples=probe._samples,
            )
        else:
            stage = StageProfile(
                stage=stage_name,
                wall_clock_s=elapsed,
                cpu_mean_pct=0.0,
                cpu_peak_pct=0.0,
                rss_peak_mib=0.0,
                rss_mean_mib=0.0,
                vms_peak_mib=0.0,
            )
        _logger.info(
            f"[PROFILE] {stage_name} finished in {elapsed:.2f}s  "
            f"RSS_peak={stage.rss_peak_mib:.0f}MiB  CPU_avg={stage.cpu_mean_pct:.0f}%"
        )
        # Store on context manager attr for retrieval
        _last_profile = stage


def _system_info() -> dict[str, Any]:
    """Gather static system info for the report header."""
    mem = psutil.virtual_memory()
    return {
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_gib": round(mem.total / (1024**3), 1),
        "platform": sys.platform,
        "python_version": sys.version.split()[0],
    }


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def profile_preprocessing(
    cleaned_path: Path, serialized_path: Path, raw_dir: Path
) -> StageProfile:
    """Reserialize raw XML → serialized Parquet → cleaned Parquet.

    Uses the smallest update batch available (~6k rows).
    """
    _logger.info("Reserializing {} -> {}", raw_dir, serialized_path)
    with profiled_stage("preprocessing") as probe:
        serialized_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()

        # Step 1: Reserialize — parse XML directory to Parquet
        parse_oai_xml_directory(raw_dir, serialized_path)
        probe.sample()
        _logger.info(f"  reserialize done in {time.perf_counter() - t0:.2f}s")

        # Step 2: Clean — load, clean, write back
        df = pd.read_parquet(serialized_path)
        probe.sample()
        df_cleaned = clean_df(df)
        probe.sample()
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        df_cleaned.to_parquet(cleaned_path, index=False)
        probe.sample()

    assert _last_profile is not None
    profile = _last_profile  # type: ignore[attr-defined]
    profile.metadata = {
        "input_rows": df_cleaned.shape[0],
        "input_cols": df_cleaned.shape[1],
        "serialized_size_mb": round(serialized_path.stat().st_size / 1e6, 1),
        "cleaned_size_mb": round(cleaned_path.stat().st_size / 1e6, 1),
    }
    return profile


def profile_embedding(
    cleaned_path: Path, output_path: Path, batch_size: int = 2500
) -> StageProfile:
    """Embed cleaned text data → output Parquet with embedding column.

    Parameters
    ----------
    cleaned_path : Path
        Input cleaned Parquet with 'title' column.
    output_path : Path
        Where to write the embedded Parquet.
    batch_size : int
        Rows per chunk (small for profiling visibility).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    embedder = get_embedder("embed-anything-onnx:AllMiniLML6V2Q")
    _logger.info(f"Embedder loaded: {embedder}, batch_size={batch_size}")

    with profiled_stage("embedding") as probe:
        parquet_file = pq.ParquetFile(cleaned_path)
        total_rows = parquet_file.metadata.num_rows
        writer: pq.ParquetWriter | None = None

        t0 = time.perf_counter()
        chunks_done = 0

        try:
            for i, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
                df_chunk = batch.to_pandas()
                # Remove existing embedding column if any
                if "embedding" in df_chunk.columns:
                    df_chunk = df_chunk.drop(columns=["embedding"])

                titles = df_chunk["title"].fillna("").tolist()
                probe.sample()

                embeddings = embedder.encode(titles, show_progress=False)

                df_chunk["embedding"] = list(embeddings)
                table = pa.Table.from_pandas(df_chunk)

                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)

                writer.write_table(table)
                chunks_done = i + 1
                probe.sample()
        finally:
            embedder.stop_pool()

        if writer:
            writer.close()

        elapsed = time.perf_counter() - t0
        _logger.info(f"  Embedded {total_rows} rows in {chunks_done} chunks ({elapsed:.2f}s)")

    assert _last_profile is not None
    profile = _last_profile  # type: ignore[attr-defined]
    profile.metadata = {
        "total_rows": total_rows,
        "chunks": chunks_done,
        "batch_size": batch_size,
        "output_size_mb": round(output_path.stat().st_size / 1e6, 1),
    }
    return profile


def profile_training(
    embeddings_dir: Path,
    model_output: Path,
    n_trees: int = 50,
    max_depth: int = 15,
) -> StageProfile:
    """Train an LSHiForest on embedded Parquet files.

    Uses a subset of trees and shallow depth for fast profiling while
    still exercising the full code path.
    """
    model_output.parent.mkdir(parents=True, exist_ok=True)
    model_output.mkdir(parents=True, exist_ok=True)
    model_path = str(model_output / "model.lshif")
    baseline_path = str(model_output / "baseline_depth.npy")

    embeddings_paths = sorted(embeddings_dir.glob("*.parquet"))
    if not embeddings_paths:
        raise FileNotFoundError(f"No parquet files in {embeddings_dir}")

    _logger.info(
        f"Training on {len(embeddings_paths)} file(s): {[p.name for p in embeddings_paths]}"
    )

    with profiled_stage("training") as probe:
        tmpdir = project_tempdir()
        try:
            mmap_path = tmpdir / "train_embeddings.mmap"

            # Convert Parquet → memmap
            t0 = time.perf_counter()
            embedding_dim, total_rows = convert_parquet_to_memmap(
                [str(p) for p in embeddings_paths], str(mmap_path)
            )
            probe.sample()
            _logger.info(
                f"  Parquet→memmap: {total_rows:,} rows × {embedding_dim}d "
                f"in {time.perf_counter() - t0:.2f}s"
            )

            embeddings_mmap = np.memmap(
                str(mmap_path), dtype=np.float32, mode="r", shape=(total_rows, embedding_dim)
            )
            probe.sample()

            # Fit
            t1 = time.perf_counter()
            model = LSHiForest(n_trees=n_trees, max_depth=max_depth, family="l2", eta=0.0)
            with mute_logging():
                model.fit(embeddings_mmap)
            probe.sample()
            _logger.info(f"  Fit {n_trees} trees in {time.perf_counter() - t1:.2f}s")

            model.save(model_path)
            probe.sample()

            # Baseline scoring
            t2 = time.perf_counter()
            with mute_logging():
                baseline_scores = model.score_chunked(embeddings_mmap, total_rows, chunk_size=5000)
            np.save(baseline_path, baseline_scores)
            probe.sample()
            _logger.info(f"  Baseline scoring in {time.perf_counter() - t2:.2f}s")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    assert _last_profile is not None
    profile = _last_profile  # type: ignore[attr-defined]
    profile.metadata = {
        "total_rows": total_rows,
        "embedding_dim": embedding_dim,
        "n_trees": n_trees,
        "max_depth": max_depth,
        "n_files": len(embeddings_paths),
        "model_size_mb": round(Path(model_path).stat().st_size / 1e6, 1),
    }
    return profile


def profile_evaluation(
    model_path: str,
    embeddings_dir: Path,
    n_workers: int | None = None,
) -> StageProfile:
    """Evaluate a trained LSHiForest model.

    Covers score distribution analysis, centroid correlation, and
    seed-stability evaluation.
    """
    embeddings_paths = sorted(embeddings_dir.glob("*.parquet"))
    if not embeddings_paths:
        raise FileNotFoundError(f"No parquet files in {embeddings_dir}")

    model = LSHiForest.load(model_path)
    _logger.info(
        f"Loaded model: {model.n_trees} trees, max_depth={model.max_depth}, "
        f"family={model.family_name}"
    )

    with profiled_stage("evaluation") as probe:
        tmpdir = project_tempdir()
        try:
            # Convert Parquet → memmap
            t0 = time.perf_counter()
            mmap_path = tmpdir / "eval_embeddings.mmap"
            embedding_dim, total_rows = convert_parquet_to_memmap(
                [str(p) for p in embeddings_paths], str(mmap_path)
            )
            probe.sample()
            _logger.info(
                f"  Parquet→memmap: {total_rows:,} rows in {time.perf_counter() - t0:.2f}s"
            )

            embeddings = np.memmap(
                str(mmap_path), dtype=np.float32, mode="r", shape=(total_rows, embedding_dim)
            )

            # Seed stability
            t1 = time.perf_counter()
            seed_stability = evaluate_params(
                [Path(p) for p in embeddings_paths],
                num_trees=model.n_trees,
                max_depth=model.max_depth,
                n_workers=n_workers,
                shared_mmap=(str(mmap_path), total_rows, embedding_dim),
            )
            probe.sample()
            _logger.info(f"  Seed stability in {time.perf_counter() - t1:.2f}s")

            # Score chunked
            t2 = time.perf_counter()
            with mute_logging():
                scores = model.score_chunked(embeddings, total_rows, chunk_size=5000)
            probe.sample()
            _logger.info(f"  Chunked scoring in {time.perf_counter() - t2:.2f}s")

            # Score distribution
            t3 = time.perf_counter()
            dist = analyze_score_distribution(scores)
            probe.sample()
            _logger.info(f"  Score distribution in {time.perf_counter() - t3:.2f}s")

            # Load metadata for anomaly export
            metadata = load_parquet_metadata([str(p) for p in embeddings_paths])
            probe.sample()

            # Top anomalies
            export_top_anomalies(scores, metadata, tmpdir / "top_anomalies.json", top_k=50)
            probe.sample()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    assert _last_profile is not None
    profile = _last_profile  # type: ignore[attr-defined]
    profile.metadata = {
        "total_rows": total_rows,
        "embedding_dim": embedding_dim,
        "spearman_aggregated": seed_stability["summary"].get("spearman_aggregated", None),
        "jaccard_aggregated": seed_stability["summary"].get("jaccard_aggregated", None),
        "score_mean": dist.get("mean", None),
        "score_median": dist.get("median", None),
        "score_skewness": dist.get("skewness", None),
    }
    return profile


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _format_profile_table(profiles: list[StageProfile]) -> str:
    """Build a markdown table of resource usage per stage."""
    header = (
        "| Stage | Duration | CPU avg | CPU peak | RSS peak | RSS avg | VMS peak |\n"
        "|-------|----------|---------|----------|----------|---------|----------|"
    )
    rows = []
    for p in profiles:
        rows.append(
            f"| {p.stage} | {p.wall_clock_s:.1f}s | {p.cpu_mean_pct:.0f}% | "
            f"{p.cpu_peak_pct:.0f}% | {p.rss_peak_mib:.0f} MiB | "
            f"{p.rss_mean_mib:.0f} MiB | {p.vms_peak_mib:.0f} MiB |"
        )
    return header + "\n" + "\n".join(rows)


def _format_metadata(meta: dict[str, Any]) -> str:
    """Render metadata dict as markdown bullets."""
    lines = []
    for k, v in meta.items():
        if isinstance(v, float):
            lines.append(f"- **{k}**: {v:.4f}")
        else:
            lines.append(f"- **{k}**: {v}")
    return "\n".join(lines)


def _format_timeline(profile: StageProfile) -> str:
    """Build an ASCII timeline of RSS memory over the stage duration."""
    samples = profile.samples
    if len(samples) < 2:
        return "(too few samples)"

    times = [s.elapsed_s for s in samples]
    rss = [s.rss_mib for s in samples]
    t_min, t_max = times[0], times[-1]
    r_min, r_max = min(rss), max(rss)
    r_range = r_max - r_min if r_max > r_min else 1

    width = 60
    chars = []
    for t, r in zip(times, rss):
        x = int((t - t_min) / (t_max - t_min) * (width - 1)) if t_max > t_min else 0
        y = int((r - r_min) / r_range * 9)
        line = [" "] * width
        line[x] = "▁▂▃▄▅▆▇█"[min(y, 7)]
        chars.append("".join(line))

    # Deduplicate adjacent identical rows
    deduped = []
    for c in chars:
        if not deduped or c != deduped[-1]:
            deduped.append(c)

    header = f"RSS: {r_min:.0f} MiB"
    footer = f"    {r_max:.0f} MiB"
    return header + "\n" + "\n".join(deduped[-20:]) + "\n" + footer  # last 20 lines


def generate_report(
    profiles: list[StageProfile],
    system: dict[str, Any],
    output_path: Path,
) -> None:
    """Write a comprehensive markdown profiling report."""
    total_wall = sum(p.wall_clock_s for p in profiles)
    total_rss_peak = max(p.rss_peak_mib for p in profiles) if profiles else 0

    lines = [
        "# MLOps-Patent Pipeline Profiling Report",
        "",
        f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## System Information",
        "",
        "| Property | Value |",
        "|----------|-------|",
        f"| Platform | {system['platform']} |",
        f"| Python | {system['python_version']} |",
        f"| CPU (logical) | {system['cpu_count_logical']} |",
        f"| CPU (physical) | {system['cpu_count_physical']} |",
        f"| Total RAM | {system['total_ram_gib']} GiB |",
        "",
        "## Aggregate Resource Usage",
        "",
        _format_profile_table(profiles),
        "",
        f"**Total pipeline wall-clock**: {total_wall:.1f}s",
        f"  **Peak RSS across all stages**: {total_rss_peak:.0f} MiB",
        "",
        "---",
        "",
    ]

    for p in profiles:
        lines.append(f"## Stage: {p.stage}")
        lines.append("")
        lines.append(f"**Wall-clock**: {p.wall_clock_s:.2f}s")
        lines.append("")
        lines.append("### Resource Timeline (RSS memory)")
        lines.append("")
        lines.append("```")
        lines.append(_format_timeline(p))
        lines.append("```")
        lines.append("")
        lines.append("### Metadata")
        lines.append("")
        lines.append(_format_metadata(p.metadata))
        lines.append("")
        lines.append("---")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    _logger.success(f"Report written to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run profiling on all pipeline stages and write the report."""

    # -- report path ----------------------------------------------------------
    report_path = PROJ_ROOT / "reports" / "figures" / "profile_report.md"
    if len(sys.argv) > 1:
        report_path = Path(sys.argv[1])

    _logger.info("=" * 60)
    _logger.info("  MLOps-Patent Pipeline Profiler")
    _logger.info("=" * 60)

    system = _system_info()
    _logger.info(f"System: {system}")

    # -- Pick input data -------------------------------------------------------
    # Use the smallest update batch for speed
    raw_dir = RAW_DATA_DIR / "updates" / "arxiv_updates_2026-04-15_to_2026-04-17_215004"
    if not raw_dir.exists():
        # Fallback: find first update directory
        updates = sorted((RAW_DATA_DIR / "updates").iterdir())
        if updates:
            raw_dir = updates[0]
        else:
            _logger.error("No raw updates directory found. Run 'data update' first.")
            sys.exit(1)

    _logger.info(f"Using raw data: {raw_dir}")

    # Scratch space for profiling intermediates
    profile_dir = project_tempdir()
    _logger.info(f"Scratch directory: {profile_dir}")

    profiles: list[StageProfile] = []

    # ---- Stage 1: Preprocessing -------------------------------------------
    serialized = profile_dir / "serialized.parquet"
    cleaned = profile_dir / "cleaned.parquet"
    _logger.info("\n" + "=" * 40)
    _logger.info("STAGE 1: Preprocessing (reserialize + clean)")
    _logger.info("=" * 40)
    p_preproc = profile_preprocessing(cleaned, serialized, raw_dir)
    profiles.append(p_preproc)

    # ---- Stage 2: Embedding -----------------------------------------------
    embedded = profile_dir / "embedded.parquet"
    _logger.info("\n" + "=" * 40)
    _logger.info("STAGE 2: Embedding")
    _logger.info("=" * 40)
    p_embed = profile_embedding(cleaned, embedded, batch_size=2500)
    profiles.append(p_embed)

    # ---- Stage 3: Training ------------------------------------------------
    # Put embedded file in a temp dir that we treat as the "processed" dir
    train_input_dir = profile_dir / "train_input"
    train_input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(embedded, train_input_dir / "embedded.parquet")

    model_dir = profile_dir / "models"
    _logger.info("\n" + "=" * 40)
    _logger.info("STAGE 3: Training")
    _logger.info("=" * 40)
    p_train = profile_training(train_input_dir, model_dir, n_trees=50, max_depth=15)
    profiles.append(p_train)

    # ---- Stage 4: Evaluation ----------------------------------------------
    model_file = str(model_dir / "model.lshif")
    _logger.info("\n" + "=" * 40)
    _logger.info("STAGE 4: Evaluation")
    _logger.info("=" * 40)
    p_eval = profile_evaluation(model_file, train_input_dir, n_workers=None)
    profiles.append(p_eval)

    # ---- Generate JSON results --------------------------------------------
    profiles_json = []
    for p in profiles:
        d = {
            "stage": p.stage,
            "wall_clock_s": p.wall_clock_s,
            "cpu_mean_pct": p.cpu_mean_pct,
            "cpu_peak_pct": p.cpu_peak_pct,
            "rss_peak_mib": p.rss_peak_mib,
            "rss_mean_mib": p.rss_mean_mib,
            "vms_peak_mib": p.vms_peak_mib,
            "n_samples": len(p.samples),
            "metadata": p.metadata,
        }
        profiles_json.append(d)

    json_path = report_path.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({"system": system, "profiles": profiles_json}, indent=2))
    _logger.info(f"Raw profile data written to {json_path}")

    # ---- Generate report --------------------------------------------------
    generate_report(profiles, system, report_path)

    # Print summary
    _logger.info("\n" + "=" * 60)
    _logger.info("  PROFILING SUMMARY")
    _logger.info("=" * 60)
    for p in profiles:
        _logger.info(
            f"  {p.stage:<20s}  {p.wall_clock_s:>6.1f}s  "
            f"RSS_peak={p.rss_peak_mib:>6.0f}MiB  CPU_avg={p.cpu_mean_pct:>5.0f}%"
        )
    total = sum(p.wall_clock_s for p in profiles)
    _logger.info(f"  {'TOTAL':<20s}  {total:>6.1f}s")


if __name__ == "__main__":
    main()
