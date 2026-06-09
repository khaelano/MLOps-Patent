#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Continuous Training Pipeline — cron entrypoint
# ──────────────────────────────────────────────────────────────────────────────
# Invoked weekly by cron.  Sources the project environment, activates the
# uv virtualenv, and runs the continuous training pipeline with the default
# embedder and MLflow experiment name.
#
# Crontab example (runs every Monday at 03:00 UTC):
#
#   0 3 * * 1  /home/khaelano/Projects/MLOps-Patent/scripts/continuous_training.sh >> /var/log/patent-ct.log 2>&1
#
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_PREFIX="[continuous-training][$(date '+%Y-%m-%dT%H:%M:%S%z')]"

echo "$LOG_PREFIX Starting weekly continuous training pipeline..."

# ── Load environment ─────────────────────────────────────────────────────
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    . "$PROJECT_ROOT/.env"
    set +a
fi

cd "$PROJECT_ROOT"

# ── Ensure MLflow tracking server is reachable ────────────────────────────
if [ -n "${MLFLOW_TRACKING_URI:-}" ]; then
    echo "$LOG_PREFIX MLflow tracking URI: $MLFLOW_TRACKING_URI"
fi

# ── Run the pipeline ──────────────────────────────────────────────────────
# Uses uv run to ensure the correct virtual environment and dependencies.
# The --trigger weekly flag identifies this as a scheduled run in logs/metrics.
uv run python patent/cli.py continuous \
    --trigger weekly \
    --mlflow-experiment "continuous-training" \
    --n-workers 4

echo "$LOG_PREFIX Pipeline completed."
