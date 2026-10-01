#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/utils/shell_env.sh"
secat_load_dotenv_preserve "$ROOT/.env"

MODEL="${1:-gpt-5.4-mini}"
case "$MODEL" in
  gpt-5.4-mini)
    : "${OPENAI_API_KEY:?OPENAI_API_KEY is required}"
    unset DEEPSEEK_THINKING DEEPSEEK_REASONING_EFFORT DEEPSEEK_TOP_P DEEPSEEK_REASONING_MIN_TOKENS 2>/dev/null || true
    ;;
  deepseek-v4-flash)
    : "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY is required}"
    export DEEPSEEK_THINKING=enabled
    export DEEPSEEK_REASONING_EFFORT=high
    ;;
  *)
    echo "usage: $0 {gpt-5.4-mini|deepseek-v4-flash}" >&2
    exit 2
    ;;
esac

: "${TMDB_API_KEY:?TMDB_API_KEY is required}"
export MODEL
export OCA_TOKEN_PROFILE="${OCA_TOKEN_PROFILE:-adaptive}"
export COMPARABLE_SERIES_TAG="${COMPARABLE_SERIES_TAG:-paper_tmdb}"

exec bash "$ROOT/run_comparable_batch.sh" tmdb 1-100
