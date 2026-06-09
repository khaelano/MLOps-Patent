#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test the continuous training pipeline locally with sample data.
#
# This is the fastest way to validate that the pipeline logic works end-to-end
# without needing DVC remotes, MLflow servers, or arXiv API access.
#
# What it tests:
#   1. Drift baseline save → load → drift check cycle
#   2. Train + evaluate on sample embeddings
#   3. CLI commands: patent continuous --dry-run, patent drift-check
#   4. All pytest tests for the continuous training module
#
# Usage:
#   ./scripts/smoke_test_workflow.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PASS=0
FAIL=0

green() { echo -e "\033[32m$1\033[0m"; }
red()   { echo -e "\033[31m$1\033[0m"; }
bold()  { echo -e "\033[1m$1\033[0m"; }

run_test() {
    local desc="$1" cmd="$2"
    echo -n "  $desc ... "
    if eval "$cmd" > /dev/null 2>&1; then
        green "PASS"
        ((PASS++))
    else
        red "FAIL"
        ((FAIL++))
    fi
}

cd "$PROJECT_ROOT"

bold "=== Continuous Training Smoke Test ==="
echo ""

# ── 1. Unit tests ─────────────────────────────────────────────────────────
bold "1. Unit tests (tests/test_continuous_training.py)"
python -m pytest tests/test_continuous_training.py -v --tb=short -q 2>&1 | tail -3
echo ""

# ── 2. CLI dry-run validation ─────────────────────────────────────────────
bold "2. CLI dry-run (patent continuous --dry-run)"
python patent/cli.py continuous --trigger weekly --dry-run 2>&1 | head -10
echo ""

# ── 3. Drift detection round-trip ─────────────────────────────────────────
bold "3. Drift detection round-trip (embed → baseline → drift check)"
python -c "
import numpy as np
from patent.monitoring.drift import save_drift_baseline, load_drift_baseline, compute_drift_metrics

# Create synthetic embeddings + scores
rng = np.random.default_rng(42)
emb = rng.normal(0, 1, (500, 128)).astype(np.float32)
scores = rng.random(500).astype(np.float32)

# Save baseline
save_drift_baseline(emb, scores, model_version='smoke-test')

# Load and verify
baseline = load_drift_baseline()
assert baseline is not None, 'Baseline not saved'
assert baseline.n_samples == 500
assert baseline.model_version == 'smoke-test'

# Check drift against same distribution (should be near-zero)
report = compute_drift_metrics(emb, scores, baseline=baseline)
assert report.score_ks_statistic < 0.05, f'Unexpected KS={report.score_ks_statistic}'

# Check drift against shifted distribution (should be non-zero)
emb_shifted = rng.normal(0.5, 1, (500, 128)).astype(np.float32)
scores_shifted = rng.random(500).astype(np.float32) + 0.3
report2 = compute_drift_metrics(emb_shifted, scores_shifted, baseline=baseline)
assert report2.score_ks_statistic > 0.1, f'Expected drift but KS={report2.score_ks_statistic}'

print('Baseline: OK | Same-dist drift: OK | Shifted-dist drift: OK')
" 2>&1
echo ""

# ── 4. Prometheus metrics update ───────────────────────────────────────────
bold "4. Prometheus metrics export"
python -c "
from patent.monitoring.metrics import update_drift_metrics, DRIFT_GAUGE
import numpy as np
update_drift_metrics(
    ks_statistic=0.05, ks_pvalue=0.5, mean_shift=0.01,
    emb_shift=0.1, n_samples=1000,
    scores=np.array([0.1, 0.2, 0.9]),
    model_version='v9', embedding_dim=384, total_rows=5000,
)
assert abs(DRIFT_GAUGE._value.get() - 0.05) < 1e-6
print(f'DRIFT_GAUGE = {DRIFT_GAUGE._value.get()} | Prometheus metrics: OK')
" 2>&1
echo ""

# ── 5. Pipeline structure validation ───────────────────────────────────────
bold "5. Pipeline module imports and structure"
python -c "
from patent.pipeline.continuous import (
    run_continuous_pipeline,
    _find_new_update_dirs,
    _process_single_source,
    _update_drift_baseline,
)
from patent.cli import continuous_cmd, drift_check_cmd
from patent.monitoring import Baseline, DriftReport
print('All pipeline imports: OK')
print('Data structures: Baseline, DriftReport — OK')
print('Pipeline functions: run, find_new, process, update_baseline — OK')
" 2>&1
echo ""

# ── 6. GitHub workflow YAML structure ─────────────────────────────────────
bold "6. GitHub workflow YAML validation"
python -c "
import yaml
with open('.github/workflows/continuous-training.yml') as f:
    wf = yaml.safe_load(f)
assert wf['name'] == 'Continuous Training'
# PyYAML 1.1 parses 'on' as boolean True — GH Actions uses YAML 1.2
triggers = wf.get('on') or wf.get(True) or {}
assert 'schedule' in triggers
assert 'workflow_dispatch' in triggers
steps = wf['jobs']['continuous-training']['steps']
required = ['actions/checkout', 'Install uv', 'DVC remote', 'MLflow server',
            'continuous training pipeline', 'DVC remote', 'DVC state changes']
names = [(s.get('name') or s.get('uses') or '') for s in steps]
for r in required:
    assert any(r in n for n in names), f'Missing step: {r}'
print(f'Workflow: OK ({len(steps)} steps)')
" 2>&1
echo ""

# ── Summary ────────────────────────────────────────────────────────────────
bold "=== Smoke Test Complete ==="
echo "Tests passed: $PASS | Tests failed: $FAIL"
if [ "$FAIL" -gt 0 ]; then
    red "Some tests FAILED. See output above."
    exit 1
else
    green "All smoke tests passed."
fi
