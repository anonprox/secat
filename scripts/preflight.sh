#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -m compileall -q agents analysis benchmarks oca utils realworld_oca run_experiment.py run_apibank.py config.py preflight.py
python preflight.py --benchmark tmdb_verified --skip-network
python preflight.py --benchmark spotify_verified --skip-network
python -u run_apibank.py --level 1 --agent codeact --tasks 1 --preflight-only
python -u run_apibank.py --level 2 --agent toolcoder --tasks 1 --preflight-only
python -m realworld_oca.paired_replay --dry-run
python -m realworld_oca.paired_replay_1374_claude --dry-run
