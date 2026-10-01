#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load project-local credentials/settings without overwriting values explicitly
# supplied by the caller (for example MODEL=deepseek-v4-flash).
# shellcheck disable=SC1091
source "$SCRIPT_DIR/utils/shell_env.sh"
secat_load_dotenv_preserve "$SCRIPT_DIR/.env"

BENCH_INPUT="${1:-}"
TASKS="${2:-}"
if [[ -z "$BENCH_INPUT" || -z "$TASKS" ]]; then
  echo "usage: $0 <spotify|tmdb> <task-list>" >&2
  echo "example: $0 spotify 1,2,3" >&2
  exit 2
fi

case "${BENCH_INPUT,,}" in
  spotify|spotify_verified) BENCHMARK="spotify_verified"; RUNTIME="spotify" ;;
  tmdb|tmdb_verified)       BENCHMARK="tmdb_verified";    RUNTIME="tmdb" ;;
  *) echo "benchmark must be spotify or tmdb" >&2; exit 2 ;;
esac

# Normalize ranges/whitespace once so result tags and resume state are deterministic.
TASKS="$(python - "$TASKS" <<'PY_TASKS'
import sys
from analysis.comparable_results import compact, parse_spec
print(compact(parse_spec(sys.argv[1])))
PY_TASKS
)"

# Derive one shared filesystem-safe batch identifier from the canonical task list.
# This is intentionally separate from TASKS: TASKS remains the complete comma-
# separated selection passed to execution/resume logic, while BATCH_SLUG is only
# a compact path/tag label (for example, tasks 1..100 -> "1-100").
BATCH_SLUG="$(python - "$TASKS" <<'PY_BATCH_SLUG'
import sys
from analysis.comparable_results import parse_spec, slug
print(slug(parse_spec(sys.argv[1])))
PY_BATCH_SLUG
)"

MODEL="${MODEL:-gpt-5.4-mini}"
TOKEN_PROFILE="${OCA_TOKEN_PROFILE:-adaptive}"
SERIES_BASE="${COMPARABLE_SERIES_TAG:-final_v4138}"
MODEL_SLUG="$(python - "$MODEL" <<'PY_MODEL_SLUG'
import sys
from utils.model_provider import model_run_slug
print(model_run_slug(sys.argv[1]))
PY_MODEL_SLUG
)"
SERIES="$SERIES_BASE"
# Never let two model families share one comparable manifest/result namespace.
if [[ "$MODEL" != "gpt-5.4-mini" ]]; then SERIES+="__${MODEL_SLUG}"; fi
PROFILE="isolated"

case "$TOKEN_PROFILE" in baseline|receipt|adaptive|lean|adaptive_lean) ;;
  *) echo "OCA_TOKEN_PROFILE must be baseline, receipt, adaptive, lean, or adaptive_lean" >&2; exit 2 ;;
esac
if ! [[ "$SERIES" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$ ]]; then
  echo "COMPARABLE_SERIES_TAG must be filesystem-safe" >&2; exit 2
fi

export PYTHONUNBUFFERED=1
export SECAT_PROJECT_ROOT="$SCRIPT_DIR"
export EVALUATION_PROFILE="$PROFILE"

if [[ "$RUNTIME" == "spotify" ]]; then
  export SPOTIFY_API_PROFILE="dev2026"
  export SECAT_ALLOW_SPOTIFY_WRITES="YES"
  export ALLOW_SPOTIFY_RESET="YES"
  export SPOTIFY_MIN_REQUEST_INTERVAL_SEC="${SPOTIFY_MIN_REQUEST_INTERVAL_SEC:-3}"
  export SPOTIFY_RATE_LIMIT_RETRIES="${SPOTIFY_RATE_LIMIT_RETRIES:-4}"
  export SPOTIFY_RATE_LIMIT_BACKOFF_SEC="${SPOTIFY_RATE_LIMIT_BACKOFF_SEC:-1}"
  export SPOTIFY_MAX_RETRY_AFTER_SEC="${SPOTIFY_MAX_RETRY_AFTER_SEC:-60}"
  export SECAT_SPOTIFY_PERSIST_OAUTH="${SECAT_SPOTIFY_PERSIST_OAUTH:-YES}"
  export SECAT_SPOTIFY_FIXTURE_CACHE="${SECAT_SPOTIFY_FIXTURE_CACHE:-$SCRIPT_DIR/results/spotify_fixture_seed_cache_v1.json}"
fi

# The manifest is created before any network call. It enforces exact canonical
# progression and makes rerunning an interrupted command resume that same batch.
BEGIN_STATE="$(python -m analysis.comparable_results begin \
  --benchmark "$BENCHMARK" --series "$SERIES" --tasks "$TASKS" \
  --model "$MODEL" --token-profile "$TOKEN_PROFILE")"
echo "$BEGIN_STATE"
if python - "$BEGIN_STATE" <<'PY_BEGIN'
import json, sys
raise SystemExit(0 if json.loads(sys.argv[1]).get("already_completed") else 1)
PY_BEGIN
then
  echo "[BATCH ALREADY COMPLETE] $BENCHMARK $TASKS | durable OCA + ToolCoder + CodeAct results verified."
  exit 0
fi

STATE_DIR="$SCRIPT_DIR/results/$BENCHMARK/_comparable_$SERIES"
mkdir -p "$STATE_DIR"
if [[ "$RUNTIME" == "spotify" ]]; then
  export SECAT_TOOLCODER_SPOTIFY_STATE="$STATE_DIR/toolcoder_spotify_state.json"
fi

# One preflight per batch invocation, before any agent/task LLM work.
# Keep the historical path unchanged for OpenAI; DeepSeek uses the shared router.
if [[ "$MODEL" == deepseek-* || "$MODEL" == "deepseek-chat" || "$MODEL" == "deepseek-reasoner" ]]; then
  : "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY is required for $MODEL}"
  OPENAI_API_KEY="$DEEPSEEK_API_KEY" python preflight.py --skip-network --benchmark "$BENCHMARK" --model "$MODEL"
  if [[ "$RUNTIME" == "spotify" ]]; then
    python -m utils.provider_preflight --benchmark "$BENCHMARK" --model "$MODEL" --spotify-profile dev2026
  else
    python -m utils.provider_preflight --benchmark "$BENCHMARK" --model "$MODEL"
  fi
else
  : "${OPENAI_API_KEY:?OPENAI_API_KEY is required for $MODEL}"
  python preflight.py --benchmark "$BENCHMARK" --model "$MODEL"
fi

root_for() {
  python - "$BENCHMARK" "$SERIES" "$TASKS" "$1" "$TOKEN_PROFILE" <<'PY'
import sys
from analysis.comparable_results import batch_root, parse_spec
print(batch_root(sys.argv[1], sys.argv[2], parse_spec(sys.argv[3]), sys.argv[4], sys.argv[5]))
PY
}

missing_for() {
  local agent="$1" root="$2"
  python -m analysis.comparable_results status \
    --root "$root" --expected "$TASKS" --agent "$agent" \
    --benchmark "$BENCHMARK" --model "$MODEL" --missing-only
}

run_agent() {
  local agent="$1"
  local root missing tag
  if [[ "$RUNTIME" == "spotify" && "$agent" == "toolcoder" ]]; then
    python -m analysis.comparable_results sync-toolcoder \
      --benchmark "$BENCHMARK" --series "$SERIES" \
      --state-path "$SECAT_TOOLCODER_SPOTIFY_STATE" \
      --model "$MODEL" --token-profile "$TOKEN_PROFILE"
  fi
  root="$(root_for "$agent")"
  missing="$(missing_for "$agent" "$root")"
  if [[ -z "$missing" ]]; then
    echo "[RESUME] $agent already complete for batch $TASKS; skipping."
    return 0
  fi

  tag="cmp_${SERIES}_b_${BATCH_SLUG}_${agent}"
  local fresh=()
  if [[ ! -d "$root" ]]; then fresh=(--fresh); fi
  local args=(
    --agent "$agent" --benchmark "$BENCHMARK"
    --tasks "$missing" --expect-task-ids "$missing"
    --model "$MODEL" --exec-mode kernel --evaluation-profile isolated
    --fail-fast-crash
  )
  if [[ "$agent" == "oca" ]]; then
    args+=(--oca-version v2 --contract-mode enforced --repair-mode typed
           --repair-endpoint-hint --oca-token-profile "$TOKEN_PROFILE"
           --oca-result-tag "$tag")
  else
    args+=(--result-tag "$tag")
  fi
  if [[ "$RUNTIME" == "spotify" ]]; then
    args+=(--spotify-profile dev2026 --allow-spotify-writes --spotify-reset-fixture)
  fi

  echo
  echo "======================================================================"
  echo "[COMPARABLE] benchmark=$BENCHMARK batch=$TASKS agent=$agent missing=$missing"
  echo "[COMPARABLE] result_root=$root"
  echo "======================================================================"

  set +e
  python run_experiment.py "${args[@]}" "${fresh[@]}"
  local rc=$?
  set -e

  # Rebuild from durable task files even after an interruption. This makes the
  # next invocation know exactly what is safely complete and what must resume.
  python -m analysis.comparable_results rebuild \
    --root "$root" --expected "$TASKS" --agent "$agent" \
    --benchmark "$BENCHMARK" --model "$MODEL" || true

  if [[ $rc -ne 0 ]]; then
    echo >&2
    echo "[STOP] $agent did not finish batch $TASKS (exit=$rc)." >&2
    echo "Completed task files were preserved. Re-run THIS SAME command later; completed work will be skipped." >&2
    exit "$rc"
  fi

  local still_missing
  still_missing="$(missing_for "$agent" "$root")"
  if [[ -n "$still_missing" ]]; then
    echo "[STOP] $agent exited but batch is still missing: $still_missing" >&2
    exit 2
  fi
}

# User-requested order. Every agent receives exactly the same task IDs and each
# task is fixture-reset independently on Spotify.
run_agent oca
run_agent toolcoder
run_agent codeact

python -m analysis.comparable_results complete \
  --benchmark "$BENCHMARK" --series "$SERIES" --tasks "$TASKS" \
  --model "$MODEL" --token-profile "$TOKEN_PROFILE"

echo
printf '[BATCH COMPLETE] %s %s | OCA + ToolCoder + CodeAct\n' "$BENCHMARK" "$TASKS"
echo "Next batch may now be started."
