#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Drift Check & Retrain — standalone script (no Prometheus required).
#
# Runs `patent drift-check`, parses the KS p-value from its output, and
# triggers `patent continuous --trigger drift` if drift exceeds the threshold.
#
# Schedule via cron (e.g., every 6 hours):
#   0 */6 * * *  /path/to/scripts/drift_check_and_retrain.sh >> /var/log/patent-drift.log 2>&1
#
# Environment variables (optional):
#   PATENT_DRIFT_PVALUE_THRESHOLD  KS p-value below which retraining fires (default 0.01)
#   PATENT_DRIFT_KS_THRESHOLD      KS statistic above which retraining fires (default 0.3)
#   GITHUB_TOKEN                   If set, triggers GitHub workflow_dispatch instead of local run
#   GITHUB_REPOSITORY              Required with GITHUB_TOKEN (e.g., "khaelano/MLOps-Patent")
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PVALUE_THRESHOLD="${PATENT_DRIFT_PVALUE_THRESHOLD:-0.01}"
KS_THRESHOLD="${PATENT_DRIFT_KS_THRESHOLD:-0.3}"
DRIFT_DETECTED=0

log() { echo "[drift-check][$(date '+%H:%M:%S')] $*"; }

cd "$PROJECT_ROOT"

# ── Load environment ─────────────────────────────────────────────────────
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    . "$PROJECT_ROOT/.env"
    set +a
fi

# ── Step 1: Run drift check ───────────────────────────────────────────────
log "Running drift check..."
output=$(uv run python patent/cli.py drift-check 2>&1) || {
    log "ERROR: drift-check failed"
    exit 1
}

# ── Step 2: Parse drift metrics from output ───────────────────────────────
ks_stat=$(echo "$output" | grep -oP 'KS=\K[\d.]+' | head -1)
ks_pval=$(echo "$output" | grep -oP 'p=\K[\d.]+'  | head -1)

log "Drift metrics: KS=$ks_stat  p=$ks_pval"

if [ -z "$ks_pval" ] || [ -z "$ks_stat" ]; then
    log "ERROR: Could not parse drift metrics from output"
    exit 1
fi

# ── Step 3: Compare against thresholds ────────────────────────────────────
if awk "BEGIN { exit !($ks_stat > $KS_THRESHOLD) }"; then
    log "DRIFT DETECTED: KS statistic $ks_stat > threshold $KS_THRESHOLD"
    DRIFT_DETECTED=1
fi

if [ "$DRIFT_DETECTED" -eq 0 ]; then
    log "No significant drift. Skipping retraining."
    exit 0
fi

# ── Step 4: Trigger retraining ────────────────────────────────────────────
log "Triggering drift retraining..."

if [ -n "${GITHUB_TOKEN:-}" ] && [ -n "${GITHUB_REPOSITORY:-}" ]; then
    # ── Trigger via GitHub Actions API ──
    log "Triggering repository_dispatch (drift-detected) on $GITHUB_REPOSITORY..."
    curl -sS -X POST \
        -H "Authorization: Bearer $GITHUB_TOKEN" \
        -H "Accept: application/vnd.github+json" \
        "https://api.github.com/repos/$GITHUB_REPOSITORY/dispatches" \
        -d '{"event_type":"drift-detected"}' \
        && log "GitHub workflow triggered (repository_dispatch)." \
        || log "WARNING: GitHub API call failed"
else
    # ── Trigger locally ──
    uv run python patent/cli.py continuous --trigger drift
fi

log "Done."
