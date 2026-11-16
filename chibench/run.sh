#!/usr/bin/env bash
# Run one of our CHI-Bench configs through CHI-Bench's own runner.
# Usage: bash chibench/run.sh <config.yaml> [--domain pa_provider|pa_um|cm] [--dry-run]
# Set CHI_BENCH_DIR if CHI-Bench is not in ../chi-bench.
set -euo pipefail

CONFIG="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
shift
DIR="${CHI_BENCH_DIR:-$(cd "$(dirname "$0")/../.." && pwd)/chi-bench}"
DRY=0
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift;;
    *) ARGS+=("$1"); shift;;
  esac
done

cd "$DIR"
mapfile -t CMDS < <(uv run python scripts/_emit_run_table_commands.py --config "$CONFIG" "${ARGS[@]}" 2>/dev/null)
if [ ${#CMDS[@]} -eq 0 ]; then
  echo "No runs selected. Check the config and --domain."; exit 1
fi
for cmd in "${CMDS[@]}"; do
  echo ">>> uv run $cmd"
  if [ "$DRY" -eq 0 ]; then
    eval "uv run $cmd"
  fi
done
echo ">>> Trials are under $DIR/logs/experiments/. Audit them with:"
echo "    python -m chibench.audit $DIR/logs/experiments --out data/chibench/audit_ours.csv"
