#!/usr/bin/env python3
"""Low-cost validation before an experiment run."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OK = "✓"
BAD = "✗"
WARN = "!"


def line(sym, msg):
    print(f"  {sym} {msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-network", action="store_true")
    ap.add_argument("--benchmark", default="tmdb")
    ap.add_argument("--model", default=None)
    ap.add_argument("--check-dataset-routes", action="store_true",
                    help="optional dataset-integrity check; never used by the clean runner")
    args = ap.parse_args()

    print("\n=== SECAT preflight ===\n")
    failed = False

    # Load project-local .env before checking credentials. This keeps preflight
    # consistent with run_experiment.py and avoids requiring `export` / `source`
    # for normal local runs.
    import config

    had_openai_key = bool(os.environ.get("OPENAI_API_KEY") or config.OPENAI_API_KEY)
    if config.OPENAI_API_KEY:
        os.environ.setdefault("OPENAI_API_KEY", config.OPENAI_API_KEY)
    # Several agent modules construct an OpenAI client at import time. Offline
    # validation should test imports without requiring or contacting a live key.
    if args.skip_network and not had_openai_key:
        os.environ["OPENAI_API_KEY"] = "offline-preflight-placeholder"

    for mod in ("openai", "requests", "dotenv"):
        try:
            importlib.import_module(mod)
            line(OK, f"import {mod}")
        except Exception as exc:
            line(BAD, f"import {mod} FAILED: {exc}")
            failed = True

    import benchmarks as B

    try:
        bench = B.get_benchmark(args.benchmark)
    except Exception as exc:
        line(BAD, str(exc))
        return 1

    model = args.model or config.DEFAULT_MODEL
    try:
        tasks = B.convert_toolcoder_dataset(bench["dataset_file"], bench["id_prefix"])
        line(OK, f"benchmark {args.benchmark}: {len(tasks)} tasks")
    except Exception as exc:
        line(BAD, f"benchmark dataset unreadable: {exc}")
        failed = True
        tasks = []

    try:
        catalog = B.load_oas(bench["oas_file"])
        line(OK, f"OpenAPI catalog: {len(catalog)} paths")
        if args.check_dataset_routes:
            missing = []
            for task in tasks:
                for endpoint in task.get("api_list") or []:
                    if endpoint not in catalog:
                        missing.append((task.get("id"), endpoint))
            if missing:
                line(BAD, f"{len(missing)} dataset route(s) absent from catalog; first={missing[0]}")
                failed = True
            else:
                line(OK, "optional dataset-route/catalog integrity check passed")
        else:
            line(OK, "dataset gold-route comparison disabled for clean preflight")
    except Exception as exc:
        line(BAD, f"OpenAPI catalog unreadable: {exc}")
        failed = True

    env_key = bench["env_key"]
    runtime_api = str(bench.get("runtime_api") or bench.get("name") or "").casefold()
    spotify_refresh_ready = (runtime_api == "spotify" and all(
        os.environ.get(k) for k in ("SPOTIFY_REFRESH_TOKEN", "SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET")
    ))
    if had_openai_key:
        line(OK, "OPENAI_API_KEY is set")
    else:
        line(WARN if args.skip_network else BAD, "OPENAI_API_KEY missing")
        failed = failed or not args.skip_network
    if os.environ.get(env_key):
        line(OK, f"{env_key} is set")
    elif spotify_refresh_ready:
        line(OK, "Spotify refresh credentials are set (access token can be renewed automatically)")
    else:
        line(WARN if args.skip_network else BAD, f"{env_key} missing")
        failed = failed or not args.skip_network

    for name, path in config.AGENTS.items():
        try:
            importlib.import_module(path)
        except Exception as exc:
            line(BAD, f"agent {name} import failed: {exc}")
            failed = True
    if not failed:
        line(OK, "all registered agents import")

    if args.skip_network:
        line(WARN, "network calls skipped")
    else:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "reply only with ok"}],
                temperature=0.0,
                max_completion_tokens=4,
            )
            line(OK, f"OpenAI model {model} responded")
        except Exception as exc:
            line(BAD, f"OpenAI/model check failed: {str(exc)[:160]}")
            failed = True

        try:
            credential = os.environ.get(env_key, "")
            auth_strategy = str(bench.get("auth_strategy") or "none")
            kwargs = {"timeout": 15}
            path = str(bench.get("preflight_path") or "")
            url = f"{bench['base_url'].rstrip('/')}/{path.lstrip('/')}"

            # Spotify access tokens are short-lived.  Use the same trusted
            # refresh-capable transport as the benchmark runtime so a 401 can be
            # refreshed and atomically persisted to the existing .env before any
            # agent tokens or task calls are spent.  Other providers retain the
            # generic raw-requests preflight.
            if runtime_api == "spotify":
                from utils.spotify_runtime import SpotifyRequestsWrapper
                r = SpotifyRequestsWrapper().get(url, timeout=15)
            else:
                import requests
                if auth_strategy == "bearer_or_query":
                    prefix = str(bench.get("bearer_prefix") or "")
                    if prefix and credential.startswith(prefix):
                        kwargs["headers"] = {"Authorization": f"Bearer {credential}"}
                    else:
                        kwargs["params"] = {str(bench.get("auth_query_param") or "api_key"): credential}
                elif auth_strategy == "bearer":
                    kwargs["headers"] = {"Authorization": f"Bearer {credential}"}
                r = requests.get(url, **kwargs)
            if r.status_code == 200:
                line(OK, f"{args.benchmark} API authentication succeeded")
            else:
                line(BAD, f"{args.benchmark} API returned HTTP {r.status_code}")
                failed = True
        except Exception as exc:
            line(BAD, f"benchmark API check failed: {str(exc)[:160]}")
            failed = True

    print()
    if failed:
        print(f"  {BAD} PREFLIGHT FAILED\n")
        return 1
    print(f"  {OK} Preflight OK\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
