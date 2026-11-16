#!/usr/bin/env bash
# One-time CHI-Bench setup on your own machine (needs Docker running, Python 3.12+, and uv).
# Usage: bash chibench/setup.sh [path-for-chi-bench, default ../chi-bench]
#
# Before running:
#   1. Request access to the gated handbook (approval can take a while):
#      https://huggingface.co/datasets/actava/managed-care-operations-handbook
#   2. Have API keys ready: ANTHROPIC_API_KEY (always, for the grader and simulated patients)
#      and OPENAI_API_KEY (for Codex).
set -euo pipefail

DIR="${1:-../chi-bench}"
REV="chi-bench-v1.0.0"

command -v docker >/dev/null || { echo "Docker is not installed"; exit 1; }
docker info >/dev/null 2>&1 || { echo "Docker is installed but not running. Start Docker Desktop and retry."; exit 1; }
command -v uv >/dev/null || { echo "Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

if [ ! -d "$DIR/.git" ]; then
  git clone https://github.com/actava-ai/chi-bench "$DIR"
fi
cd "$DIR"
uv sync --extra dev

if [ ! -f .env ]; then
  cp .env.example .env
  echo ">>> Created $DIR/.env. Add ANTHROPIC_API_KEY and OPENAI_API_KEY, then rerun this script."
  exit 0
fi

echo ">>> Logging in to Hugging Face (needed for the task data and the gated handbook)"
uv run huggingface-cli whoami >/dev/null 2>&1 || uv run huggingface-cli login

if [ ! -f data/.chi-bench-version ]; then
  uv run huggingface-cli download actava/chi-bench --repo-type dataset --revision "$REV" --local-dir data/
  echo "$REV" > data/.chi-bench-version
fi
if [ ! -d data/skills/managed-care-operations-handbook ]; then
  uv run huggingface-cli download actava/managed-care-operations-handbook --repo-type dataset --local-dir data/skills/ \
    || { echo ">>> Handbook download failed. Has your access request been approved?"; exit 1; }
fi

uv run playwright install chromium
uv run python scripts/repair_pa_t035.py
uv run python scripts/regenerate_payer_request_forms.py
uv run cb docker build
uv run cb data verify
echo ">>> CHI-Bench is ready in $DIR. Next: bash chibench/run.sh chibench/configs/smoke_codex.yaml"
