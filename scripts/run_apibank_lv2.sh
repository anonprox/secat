#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/utils/shell_env.sh"
secat_load_dotenv_preserve "$ROOT/.env"

MODEL="${1:-gpt-5.4-mini}"
export APIBANK_ROOT="${APIBANK_ROOT:-$ROOT/benchmarks/api_bank_vendor}"
export APIBANK_OCA_PROFILE=lv2
export APIBANK_OCA_EFFICIENT_MAX_REPAIRS="${APIBANK_OCA_EFFICIENT_MAX_REPAIRS:-1}"
export APIBANK_OCA_LLM_TIMEOUT_SEC="${APIBANK_OCA_LLM_TIMEOUT_SEC:-60}"
export APIBANK_OCA_TASK_TIMEOUT_SEC="${APIBANK_OCA_TASK_TIMEOUT_SEC:-240}"
export APIBANK_OCA_LV2_SEARCH_MODE="${APIBANK_OCA_LV2_SEARCH_MODE:-lexical}"
export APIBANK_OCA_LV2_LATENT_RECOVERY="${APIBANK_OCA_LV2_LATENT_RECOVERY:-1}"
export APIBANK_OCA_LV2_NAME_RECOVERY="${APIBANK_OCA_LV2_NAME_RECOVERY:-0}"

case "$MODEL" in
  gpt-5.4-mini)
    : "${OPENAI_API_KEY:?OPENAI_API_KEY is required}"
    export OPENAI_REASONING_EFFORT="${OPENAI_REASONING_EFFORT:-high}"
    export OPENAI_REASONING_MIN_TOKENS="${OPENAI_REASONING_MIN_TOKENS:-8192}"
    unset DEEPSEEK_THINKING DEEPSEEK_REASONING_EFFORT DEEPSEEK_TOP_P DEEPSEEK_REASONING_MIN_TOKENS DEEPSEEK_STRUCTURED_MIN_TOKENS 2>/dev/null || true
    SLUG="gpt54mini"
    ;;
  deepseek-v4-flash)
    : "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY is required}"
    unset OPENAI_REASONING_EFFORT OPENAI_REASONING_MIN_TOKENS 2>/dev/null || true
    export DEEPSEEK_THINKING=enabled
    export DEEPSEEK_REASONING_EFFORT=high
    export DEEPSEEK_TOP_P="${DEEPSEEK_TOP_P:-0.95}"
    export DEEPSEEK_REASONING_MIN_TOKENS="${DEEPSEEK_REASONING_MIN_TOKENS:-32768}"
    export DEEPSEEK_STRUCTURED_MIN_TOKENS="${DEEPSEEK_STRUCTURED_MIN_TOKENS:-32768}"
    export APIBANK_OCA_DEEPSEEK_MAX_COMPLETION_TOKENS="${APIBANK_OCA_DEEPSEEK_MAX_COMPLETION_TOKENS:-32768}"
    SLUG="deepseek"
    ;;
  *)
    echo "usage: $0 {gpt-5.4-mini|deepseek-v4-flash}" >&2
    exit 2
    ;;
esac

# Warm the retrieval stack before scored calls, then keep it offline for the run.
python -u run_apibank.py \
  --level 2 --agent toolcoder --model "$MODEL" --tasks all \
  --preflight-only --apibank-root "$APIBANK_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

for AGENT in codeact toolcoder oca; do
  TAG="paper_lv2_${AGENT}_${SLUG}"
  python -u run_apibank.py \
    --level 2 \
    --agent "$AGENT" \
    --model "$MODEL" \
    --tasks all \
    --expect-task-ids all \
    --state-mode published \
    --continue-infrastructure \
    --result-tag "$TAG" \
    --fresh
done
