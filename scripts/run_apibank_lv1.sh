#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/utils/shell_env.sh"
secat_load_dotenv_preserve "$ROOT/.env"

MODEL="${1:-gpt-5.4-mini}"
TASKS="1-106,112-384,389"
export APIBANK_ROOT="${APIBANK_ROOT:-$ROOT/benchmarks/api_bank_vendor}"
export APIBANK_OCA_PROFILE="${APIBANK_OCA_PROFILE:-efficient}"
export APIBANK_OCA_EFFICIENT_MAX_REPAIRS="${APIBANK_OCA_EFFICIENT_MAX_REPAIRS:-1}"
export APIBANK_OCA_LLM_TIMEOUT_SEC="${APIBANK_OCA_LLM_TIMEOUT_SEC:-60}"
export APIBANK_OCA_TASK_TIMEOUT_SEC="${APIBANK_OCA_TASK_TIMEOUT_SEC:-240}"

case "$MODEL" in
  gpt-5.4-mini)
    : "${OPENAI_API_KEY:?OPENAI_API_KEY is required}"
    unset DEEPSEEK_THINKING DEEPSEEK_REASONING_EFFORT DEEPSEEK_TOP_P DEEPSEEK_REASONING_MIN_TOKENS DEEPSEEK_STRUCTURED_MIN_TOKENS 2>/dev/null || true
    SLUG="gpt54mini"
    ;;
  deepseek-v4-flash)
    : "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY is required}"
    export DEEPSEEK_THINKING=enabled
    export DEEPSEEK_REASONING_EFFORT=high
    export DEEPSEEK_STRUCTURED_MIN_TOKENS="${DEEPSEEK_STRUCTURED_MIN_TOKENS:-16384}"
    export APIBANK_OCA_DEEPSEEK_MAX_COMPLETION_TOKENS="${APIBANK_OCA_DEEPSEEK_MAX_COMPLETION_TOKENS:-16384}"
    SLUG="deepseek"
    ;;
  *)
    echo "usage: $0 {gpt-5.4-mini|deepseek-v4-flash}" >&2
    exit 2
    ;;
esac

for AGENT in codeact toolcoder oca; do
  TAG="paper_lv1_${AGENT}_${SLUG}"
  python -u run_apibank.py \
    --level 1 \
    --agent "$AGENT" \
    --model "$MODEL" \
    --tasks "$TASKS" \
    --expect-task-ids "$TASKS" \
    --state-mode published \
    --continue-infrastructure \
    --result-tag "$TAG" \
    --fresh
done
