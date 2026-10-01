#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/utils/shell_env.sh"
secat_load_dotenv_preserve "$ROOT/.env"

: "${OPENAI_API_KEY:?OPENAI_API_KEY is required for the four generic paired cases}"
: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY is required for smolagents #1374}"

python -m realworld_oca.paired_replay --dry-run
python -m realworld_oca.paired_replay_1374_claude --dry-run

python -m realworld_oca.paired_replay \
  --case all --mode both --model gpt-5.4-mini \
  --output-dir results/realworld_rq4_paired

python -m realworld_oca.paired_replay_1374_claude \
  --mode both --model anthropic/claude-sonnet-4-6 \
  --output-dir results/realworld_rq4_paired_1374
