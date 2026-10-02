# Systematic Evaluation of Code-as-Tool Agents

## Included

- `agents/`: CodeAct, ToolCoder, and OCA implementations.
- `benchmarks/`: verified TMDB and Spotify data plus the vendored API-Bank runtime/data.
- `oca/`: OCA capability definitions used by the typed compiler.
- `utils/`: runtime utilities required by the three agents.
- `analysis/comparable_results.py`: batch/resume bookkeeping used by the TMDB and Spotify runner.
- `realworld_oca/`: the five controlled RQ4 paired replays.
- `scripts/`: one-command reproduction scripts.
- `config.py`, `run_experiment.py`, `run_apibank.py`, `preflight.py`: core runners.
- `.env.example`: credential template.
- `requirements.txt`: Python dependencies.

Development tests, patch scripts, old agent variants, oracle-construction utilities, generated results, logs, caches, and historical release files are intentionally excluded.

## Setup

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader punkt_tab
cp .env.example .env
```

Fill only the credentials needed for the benchmarks you plan to run. Never commit `.env`.

## Preflight

```bash
bash scripts/preflight.sh
```

## TMDB

The verified TMDB set contains 100 tasks. Each command runs OCA, ToolCoder, and CodeAct on the same tasks.

```bash
bash scripts/run_tmdb.sh gpt-5.4-mini
bash scripts/run_tmdb.sh deepseek-v4-flash
```

Required: `TMDB_API_KEY` and the corresponding model-provider key.

## Spotify

The paper uses the 47 Development Mode-compatible tasks. The runner resets and mutates the connected Spotify account, so use a disposable benchmark account.

```bash
export ALLOW_SPOTIFY_BENCHMARK_RESET=YES
bash scripts/run_spotify.sh gpt-5.4-mini
bash scripts/run_spotify.sh deepseek-v4-flash
```

Required: Spotify credentials/token and the corresponding model-provider key.

## API-Bank Level 1

The paper task set is `1-106,112-384,389` (380 tasks).

```bash
bash scripts/run_apibank_lv1.sh gpt-5.4-mini
bash scripts/run_apibank_lv1.sh deepseek-v4-flash
```

## API-Bank Level 2

All 119 Level-2 tasks are used.

```bash
bash scripts/run_apibank_lv2.sh gpt-5.4-mini
bash scripts/run_apibank_lv2.sh deepseek-v4-flash
```

## RQ4 real-world replays

```bash
bash scripts/run_rq4.sh
```

The first four paired cases use `gpt-5.4-mini`; the dedicated `smolagents#1374` replay uses `anthropic/claude-sonnet-4-6`.

## Output

The runners create `results/` automatically. Generated outputs are ignored by Git.

## Anonymous artifact note

The package contains no project `.env`, generated live-service traces, local logs, or known author-identifying paths/account values. Model/provider names are retained because they are part of the experimental configuration.
