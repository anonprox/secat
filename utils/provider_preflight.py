"""Network preflight for non-OpenAI providers used by SECAT runners.

The historical preflight.py remains unchanged for OpenAI runs.  For DeepSeek,
runners execute the historical offline checks and then this provider-aware
network check so the benchmark credential and selected DeepSeek model are both
validated before an experiment starts.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Be robust both as ``python -m utils.provider_preflight`` and as a direct
# script invocation from any working directory.  Direct execution normally
# places only ``.../utils`` on sys.path, which makes sibling packages such as
# ``agents`` unavailable.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _check_llm(model: str) -> None:
    from agents.common import make_client
    from utils.model_provider import provider_settings, require_provider_key

    settings = provider_settings(model)
    require_provider_key(model)
    print(f"  ✓ {settings['key_env']} is set")
    client = make_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with OK only."}],
        max_completion_tokens=4096,
        temperature=0.0,
    )
    text = str(resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError(f"model {model} returned empty content")
    print(f"  ✓ {settings['provider']} model {model} responded")


def _check_tmdb() -> None:
    import requests
    key = os.getenv("TMDB_API_KEY", "").strip()
    if not key:
        raise RuntimeError("TMDB_API_KEY is missing")
    url = "https://api.themoviedb.org/3/configuration"
    if key.startswith("eyJ"):
        r = requests.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=20)
    else:
        r = requests.get(url, params={"api_key": key}, timeout=20)
    if not (200 <= r.status_code < 300):
        raise RuntimeError(f"TMDB authentication failed: HTTP {r.status_code}")
    print("  ✓ TMDB API authentication succeeded")


def _check_spotify() -> None:
    from utils.spotify_runtime import SpotifyRequestsWrapper, BASE_URL
    api = SpotifyRequestsWrapper(raise_on_rate_limit=True)
    r = api.get(BASE_URL + "/me")
    if not (200 <= r.status_code < 300):
        raise RuntimeError(f"Spotify authentication failed: HTTP {r.status_code}")
    print("  ✓ spotify_verified API authentication succeeded")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--spotify-profile", default=None)
    args = p.parse_args()

    if args.spotify_profile:
        os.environ["SPOTIFY_API_PROFILE"] = str(args.spotify_profile)

    print("=== SECAT provider preflight ===")
    _check_llm(args.model)
    name = args.benchmark.strip().lower()
    if name in {"tmdb", "tmdb_verified"}:
        _check_tmdb()
    elif name in {"spotify", "spotify_verified"}:
        _check_spotify()
    else:
        raise RuntimeError(f"unsupported benchmark in provider preflight: {args.benchmark}")
    print("  ✓ Provider preflight OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAILED] provider preflight: {exc}", file=sys.stderr)
        raise SystemExit(2)
