#!/usr/bin/env bash
# Download the public CHI-Bench leaderboard (Apache 2.0): about 1,375 trials with trajectories and
# scorecards (about 350 MB), then audit them.
# Usage: bash chibench/fetch_public_traces.sh
set -euo pipefail
DEST="data/raw/chibench-leaderboard"
if [ ! -d "$DEST/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/actava-ai/leaderboard "$DEST"
fi
mkdir -p data/chibench reports
python -m chibench.audit "$DEST/benchmarks/chi-bench/submissions" --out data/chibench/audit_public.csv
python -m chibench.analyze data/chibench/audit_public.csv --out reports/chibench_public.md
