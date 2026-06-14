# Drift Detection & Monitoring

## Architecture

```
Client → :8000 → nginx (inference-lb)
                    ├─ replica-1  FastAPI  /health /predict /metrics
                    ├─ replica-2  FastAPI  /health /predict /metrics
                    └─ replica-3  FastAPI  /health /predict /metrics
                           │
                           ▼  scrape every 5s
                    Prometheus ──► Grafana ──► /dispatch relay ──► GitHub CT workflow
                                                     │
                                                     ▼
                                          Model CT workflow
```

The inference server loads the Production model from the MLflow Model Registry
at startup and buffers anomaly scores from every `/predict` call.  When
Prometheus scrapes `/metrics`, a drift check is run against the buffered
scores using the saved training baseline.

---

## Score Buffer

- Size: 10,000 samples (configurable via `DRIFT_SCORE_BUFFER_SIZE`)
- Every `/predict` call appends raw scores to a `collections.deque`
- When the buffer has ≥100 samples, a drift check runs on `/metrics` scrape
- **After each drift check, the buffer is cleared** so the next check only
  sees fresh scores — this prevents old drifted scores from contaminating
  subsequent checks after normal data resumes

Drift checks run **opportunistically** — they execute inside the `/metrics`
endpoint handler.  No separate background task is needed.

---

## Drift Detection Signals

Four complementary tests run on the anomaly score distribution.  Drift is
flagged when **≥2 of the 4** tests agree (consensus gate).  All four use
`scipy.stats` — no additional dependencies.

```
┌─────────────────────────────────────────────────────────┐
│                   detect_drift()                         │
│                                                         │
│  baseline_scores  ──┬──► KS test ──► p-value < 0.05?   │
│                     ├──► Wasserstein > 0.1?              │
│                     ├──► Mean shift > 0.05σ?             │
│  new_scores     ────┴──► Energy dist. > 0.15?            │
│                                                         │
│  flags = [tests that triggered]                          │
│  drift_detected = len(flags) >= 2                       │
└─────────────────────────────────────────────────────────┘
```

---

### 1. Two-Sample Kolmogorov–Smirnov Test

**What it measures:** The maximum vertical distance between the two empirical
cumulative distribution functions (eCDFs):

```
D = max_x |F_baseline(x) - F_new(x)|
```

where `F(x)` = proportion of scores ≤ x.

**What it catches:** Any difference in the overall shape or location of the
score distribution — a shift in central tendency, a change in spread, or an
asymmetric tail.  It is **non-parametric** (no assumption about the
underlying distribution), which makes it suitable for anomaly scores that
are often skewed and bounded [0, 1].

**What it misses:** Because the test is most sensitive near the median (where
eCDFs naturally have the steepest slope), it has less power to detect changes
in the extreme tails — exactly where anomaly scores are most interesting.
This is why we pair it with Wasserstein distance and mean shift.

**Threshold interpretation (p-value < 0.05):**
- p-value is **not** a measure of effect size — with enough samples (~10k),
  even trivially small differences trigger p < 0.05
- The companion `ks_statistic` (0 = identical, 1 = completely disjoint) tells
  you **how much** the distributions differ
- A significant p-value + low statistic = subtle shift detected at scale
- A non-significant p-value = distributions are indistinguishable given the
  sample size

**Implementation:** `scipy.stats.ks_2samp()`, O(n log n).

---

### 2. Wasserstein Distance (Earth Mover's Distance)

**What it measures:** The minimum "work" required to transform the new score
distribution into the baseline distribution.  Intuitively — if each score is
a pile of dirt, how far must you carry it to match the reference shape?

```
W = ∫ |F^{-1}_baseline(t) - F^{-1}_new(t)| dt
```

(integral of the absolute difference between inverse CDFs).

**What it catches:** The **magnitude** of the shift — not just *that* the
distributions differ, but *by how much* on the score scale.  Two distributions
may have the same KS D-statistic but very different Wasserstein distances
depending on how far apart their masses are.

**What it misses:** Because Wasserstein integrates over the entire
distribution, a large shift in a small probability region is diluted — it may
flag late or not at all if the bulk of scores remain stable.

**Threshold interpretation (> 0.1):**
- 0.0 = identical distributions
- 0.1 = the average score had to move 0.1 on the [0, 1] scale
- > 0.3 = substantial shift (e.g. mean moved by ~1/3 of the full range)
- Scale is in the same units as the anomaly scores, so 0.1 is interpretable
  directly

---

### 3. Mean Shift (Relative)

**What it measures:** The difference in the first moment (average) of the
score distribution, normalised by the baseline standard deviation:

```
relative_shift = |mean_new - mean_baseline| / σ_baseline
```

This is Cohen's d — a standardised effect size.

**What it catches:** A change in the **central tendency** of anomaly scores.
If the model is suddenly scoring everything higher (or lower) on average,
this test triggers.  It is the simplest and cheapest of the four signals.

**What it misses:** Any drift that preserves the mean — for example, a
bimodal distribution where one cluster moves up and another moves down by the
same amount.  It also misses pure variance changes (scores become noisier
without shifting).

**Threshold interpretation (> 0.05σ):**
- 0.05 = the mean shifted by 5% of a baseline standard deviation
- Due to the normalisation, this is comparable across different model versions
  even if their baseline score ranges differ
- The raw (un-normalised) `mean_shift` is also returned for direct inspection

---

### 4. Energy Distance

**What it measures:** A distributional divergence derived from the pairwise
distances between all samples in both sets:

```
E² = 2·E||X - Y|| - E||X - X'|| - E||Y - Y'||
```

where X, X' are independent draws from the baseline, Y, Y' from the new
buffer, and ||·|| is absolute difference.  E² = 0 means the two distributions
are identical; higher values mean they differ.

**What it catches:** **Variance and spread shifts** that the other three
signals can miss.  Energy distance is sensitive to changes in dispersion
(scale) even when the mean and overall CDF shape stay similar.  Consider:

```
Baseline scores:  [0.1, 0.2, 0.3, 0.4, 0.5]    mean=0.30, σ=0.16
New scores:       [0.0, 0.1, 0.3, 0.5, 0.6]    mean=0.30, σ=0.26
```

Here KS and mean shift may not trigger (same mean, similar CDF path), but
energy distance catches the increased spread.

**What it misses:** Energy distance can be computationally heavier than the
others (O(n²) in the naive form), but for the 10k samples in the buffer
the `scipy` implementation is acceptable — it uses a fast algorithm.

**Threshold interpretation (> 0.15):**
- 0.0 = identical distributions
- 0.15 = moderate divergence in the pairwise distance structure
- Because energy distance squares the differences, it amplifies large
  deviations — useful for catching outlier-heavy drift
- Scale is dataset-dependent; 0.15 is a calibrated starting point

---

### Consensus Gate

The four flags are tallied; drift is declared when **≥2** fire:

```python
drift_detected = len(flags) >= 2
```

This guard prevents single-method false positives — each signal has blind
spots, but when multiple independent tests agree, the evidence is stronger.

Example scenarios:

| KS | Wasserstein | Mean shift | Energy | Consensus | Interpretation |
|----|-------------|------------|--------|-----------|----------------|
| ✓ | ✓ | ✓ | ✓ | **Drift** | All four agree — strong signal |
| ✓ | ✓ | ✗ | ✗ | **Drift** | Shape changed, central mass moved |
| ✗ | ✗ | ✗ | ✓ | **No drift** | Only energy triggered — likely noise |
| ✗ | ✗ | ✗ | ✗ | **No drift** | Distributions indistinguishable |
| ✓ | ✗ | ✗ | ✓ | **Drift** | Shape + spread changed, mean stable |

### Early exit

If either the baseline or the new buffer has fewer than 2 finite values,
drift defaults to `False` and all signal values return 0.

---

## Prometheus Metrics

Exposed at `GET /metrics` on every inference-server replica (and via nginx at
`inference-lb:80/metrics`).

| Metric | Type | Description |
|--------|------|-------------|
| `patent_drift_score_ks_statistic` | Gauge | KS statistic (0 = identical, 1 = disjoint) |
| `patent_drift_score_ks_pvalue` | Gauge | KS p-value |
| `patent_drift_score_mean_shift` | Gauge | Mean difference (new – baseline) |
| `patent_drift_score_energy_distance` | Gauge | Energy distance (0 = identical) |
| `patent_drift_detected` | Gauge | 1 if drift flagged, 0 otherwise |
| `patent_drift_score_distribution` | Histogram | Score distribution (11 buckets 0.0–1.0) |
| `patent_drift_new_samples_total` | Gauge | Samples in the last check |
| `patent_drift_last_checked_timestamp_seconds` | Gauge | Unix timestamp of last check |
| `patent_model_info` | Gauge | 1 = active, labels: model_version, model_name |
| `patent_data_total_rows` | Gauge | Total training rows |
| `patent_data_embedding_dim` | Gauge | Embedding dimensionality (384) |

Implementation: `patent/monitoring/metrics.py` — lazy-loads `prometheus_client`
at import time; metrics are no-ops when the library is absent.

---

## Drift Baseline

Created during training and stored on disk at `data/interim/drift_baseline/`:

```
data/interim/drift_baseline/
├── embedding_baseline.npz   # 10k × 384 float32 embeddings (subsampled)
├── score_baseline.npz       # 10k float64 anomaly scores
└── baseline_meta.json       # model_version, n_samples, timestamp
```

- **Creation**: `train_model()` in `patent/modeling/train.py` calls
  `_save_baseline_sample()` after scoring — a deterministic random 10k-row
  subset (seed 42) is saved via `save_drift_baseline()`.
- **Loading**: Inference server calls `load_drift_baseline()`.
  If the baseline files do not exist (e.g. first deploy without a volume),
  returns `None` and drift checks report no drift gracefully.
- **Persistence**: In CI/CD, the baseline is pushed to DVC after training.
  For production containers, it must be baked into the image or mounted as a
  volume.

---

## Grafana Alert → GitHub Dispatch

1. **Grafana alert rule** (`docker/grafana/provisioning/alerting/drift_rules.yml`):
   - Queries `patent_drift_detected{service="patent-inference"}` via Prometheus
   - Uses a reduce(`last`) + threshold(`>0`) expression pipeline (Grafana 11 unified alerting)
   - Evaluates every 1 minute, fires when drift is sustained for 5 minutes
   - Labels: `severity: warning`, `drift_type: score_distribution`

2. **Contact point** (`docker/grafana/provisioning/alerting/contact-points.yml`):
   - Webhook `POST http://inference-lb/dispatch` — the internal relay endpoint

3. **Relay** (`patent/api.py` `POST /dispatch`):
   - Receives the Grafana webhook, sends `repository_dispatch` to GitHub API
   - Requires `GITHUB_DISPATCH_URL` and `GITHUB_DISPATCH_TOKEN` env vars

4. **Workflow trigger** (`.github/workflows/model-ct.yml`):
   - `repository_dispatch: types: [drift-detected]`
   - Also triggered by cron (weekly) and `workflow_dispatch`

---

## Resolved Issues

| Issue | Fix | When |
|-------|-----|------|
| boto3 version conflict (pip installed timed out) | Changed `boto3>=1.38.1` → `boto3` (unconstrained) | Pipeline setup |
| Missing `patent/monitoring/` package in Docker | Package existed as directory; shadowed by old `monitoring.py` file | Pipeline setup |
| Grafana DS_PROMETHEUS template unresolvable | Replaced with hardcoded `prometheus` UID | Pipeline setup |
| Drift baseline never saved during training | `_save_baseline_sample()` added to `train_model()` | Code audit |
| Dummy zero embeddings in drift check | `api.py` switched to score-only detection (embedded shift deprecated) | Code audit |
| `drift_detected` not exposed to Prometheus | Added `patent_drift_detected` Gauge | Code audit |
| Embedding shift gauge always 0, dead code | Removed `DRIFT_GAUGE_EMB_SHIFT`, `_compute_embedding_mean_shift()`, `emb_shift=0.0` | Cleanup |
| `compute_drift_metrics()`, `DriftReport`, `compare_score_distributions()` dead | Removed from monitoring package | Cleanup |
| Energy Distance added as 4th signal | Added `energy_threshold=0.15` to `detect_drift()`, `patent_drift_score_energy_distance` gauge | Cleanup |
| Drift alert never reset after normal data resumed | Added `_score_buffer.clear()` after each drift check so old drifted scores don't contaminate subsequent checks | Score buffer audit |
| Grafana alert rule in Error state | Replaced raw `condition: A` with `reduce(last)` + `threshold(>0)` expression pipeline (Grafana 11 unified alerting) | Alert rule audit |
| "Drift Detected" dashboard panel TypeError | Changed panel from `gauge` to `stat` with numeric value mappings (`0→OK`, `1→DRIFT`) | Dashboard audit |
| CT pipeline `dvc add` conflict with `dvc.yaml` | Switched `data-process` and `train` jobs to `dvc repro preprocess` / `dvc repro training` | Pipeline audit |
| Missing DVC metadata git commits in CT pipeline | Added `git commit` + `git push` for `data/raw.dvc` and `dvc.lock` after each DVC push | Pipeline audit |

---

## Relevant Files

| Path | Purpose |
|------|---------|
| `patent/monitoring/drift_detector.py` | `detect_drift()` — 4-signal consensus |
| `patent/monitoring/drift.py` | Baseline save/load (`save_drift_baseline`, `load_drift_baseline`) |
| `patent/monitoring/metrics.py` | Prometheus gauges + `update_drift_metrics()` |
| `patent/api.py` | FastAPI server: `/health`, `/predict`, `/metrics` |
| `scripts/simulate_drift.py` | Drift simulation (sends garbage text to skew scores) |
| `docker/prometheus/prometheus.yml` | Scrape config (5s interval) |
| `docker/grafana/dashboards/drift.json` | Provisioned dashboard |
| `docker/grafana/provisioning/alerting/drift_rules.yml` | Alert rule on `patent_drift_detected` |
| `docker/grafana/provisioning/alerting/contact-points.yml` | Webhook → `/dispatch` relay |
| `docker/grafana/provisioning/alerting/policies.yml` | Routes `drift_type=score_distribution` alerts |
| `docker/grafana/dashboards/drift.json` | Provisioned dashboard (stat panel for drift status) |
| `.github/workflows/model-ct.yml` | Triggered by drift alert + cron; uses `dvc repro` |
| `.github/workflows/_train-evaluate-register.yml` | Reusable training workflow with `dvc repro training` |
| `pipelines/train.py` | DVC training stage entry point |
| `pipelines/preprocess.py` | DVC preprocess stage entry point |
| `patent/modeling/train.py` | `_save_baseline_sample()` — creates drift baseline during training |
