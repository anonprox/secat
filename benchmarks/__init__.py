"""
benchmarks.py
A small registry that describes each RestBench-style benchmark in one place, so the
rest of the codebase is not hardcoded to TMDB. Adding a new dataset = add one entry.

Each benchmark provides:
  - name           short id
  - base_url       live API base
  - auth_strategy  transport authentication strategy ('bearer_or_query' | 'bearer' | 'none')
  - env_key        environment variable holding the credential
  - dataset_file   ToolCoder-format file: list of {query, solution}
  - oas_file       endpoint catalog (list or OpenAPI dict)
  - id_prefix      task id prefix (tmdb_, spotify_)
  - read_only      True if all gold paths are GET (safe to execute for the oracle)

The ToolCoder datasets use {query, solution} where solution is a list like
["GET /3/search/person", "GET /3/person/{person_id}/movie_credits"].
convert_toolcoder_dataset() turns that into our task format ({id, instruction, api_list}).
"""
import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
def _D(name):
    """Absolute path to a data file inside benchmarks/data/."""
    return os.path.join(_HERE, "data", name)

BENCHMARKS = {
    "tmdb_verified": {
        "name": "tmdb_verified",
        "runtime_api": "tmdb",
        "base_url": "https://api.themoviedb.org/3",
        "auth_strategy": "bearer_or_query",
        "auth_query_param": "api_key",
        "bearer_prefix": "eyJ",
        "preflight_path": "/configuration",
        "env_key": "TMDB_API_KEY",
        "dataset_file": _D("tmdb_verified.json"),
        "oas_file": _D("tmdb_oas.json"),
        "id_prefix": "tmdb_",
        "read_only": True,
        "api_label": "TMDB (The Movie Database)",
        "auth_note": 'Authenticate with TMDB_API_KEY.',
    },
    "spotify_verified": {
        "name": "spotify_verified",
        "runtime_api": "spotify",
        "requests_compat": "utils.spotify_runtime:install_requests_compat",
        "action_certificate": "utils.spotify_runtime:action_certificate",
        "base_url": "https://api.spotify.com/v1",
        "auth_strategy": "bearer",
        "preflight_path": "/me",
        "env_key": "SPOTIFY_ACCESS_TOKEN",
        "dataset_file": _D("spotify_verified.json"),
        "oas_file": _D("spotify_oas.json"),
        "toolcoder_oas_file": _D("spotify_oas_toolcoder.json"),
        "id_prefix": "spotify_",
        "read_only": False,
        "api_label": "Spotify Web API",
        "auth_note": 'Authenticate with SPOTIFY_ACCESS_TOKEN.',
    },
}


def get_benchmark(name):
    if name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark '{name}'. known: {list(BENCHMARKS)}")
    return BENCHMARKS[name]


def strip_method(step):
    """'GET /3/search/person' -> '/3/search/person' (drops the HTTP verb)."""
    return re.sub(r"^(GET|POST|PUT|DELETE|PATCH)\s+", "", step.strip(), flags=re.I)


def convert_toolcoder_dataset(path, id_prefix):
    """Load a ToolCoder {query, solution} file and return our task list.
    Handles both the original flat list and the verified {"_meta":..,"tasks":[..]}
    format. api_list holds the gold endpoint paths (verb stripped) for Path%."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    # verified format: {"_meta":{...}, "tasks":[{query, solution, status, ...}]}
    if isinstance(raw, dict) and "tasks" in raw:
        raw = raw["tasks"]
    tasks = []
    for i, item in enumerate(raw):
        sol = item.get("solution", [])
        api_list = [strip_method(s) for s in sol]
        solution_steps = []
        for step in sol:
            match = re.match(r"^(GET|POST|PUT|DELETE|PATCH)\s+(.+)$", step.strip(), re.I)
            if match:
                solution_steps.append({"method": match.group(1).upper(),
                                       "path": match.group(2).strip()})
        methods = [step["method"] for step in solution_steps]
        t = {
            "id": f"{id_prefix}{i+1:03d}",
            "instruction": item["query"].strip(),
            "query": item["query"].strip(),
            "api_list": api_list,         # gold path (endpoints), what we score against
            "solution": sol,              # (corrected) path, verb included
            "solution_steps": solution_steps,
            "http_methods": methods,
            "has_writes": any(m in {"POST", "PUT", "DELETE", "PATCH"}
                              for m in methods),
            "read_only_task": bool(methods) and all(m == "GET" for m in methods),
        }
        # carry verified-dataset metadata if present
        if "status" in item:
            t["verify_status"] = item["status"]
        if item.get("status") == "excluded_accuracy":
            t["exclude_accuracy"] = True
        tasks.append(t)
    return tasks


def load_oas(path):
    """Return a dict: endpoint_path -> {parameters, schema, ...}.
    Handles both the TMDB list format and the Spotify OpenAPI dict format."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    out = {}
    if isinstance(data, list):
        # TMDB style: list of {path, parameters, schema, ...}
        for item in data:
            out[item["path"]] = item
    elif isinstance(data, dict) and "paths" in data:
        # OpenAPI style (Spotify): {paths: {"/search": {get: {...}}, ...}}
        for p, methods in data["paths"].items():
            out[p] = {"path": p, "methods": list(methods.keys()), "spec": methods}
    return out


if __name__ == "__main__":
    # quick self-check
    for name, b in BENCHMARKS.items():
        df = b["dataset_file"]
        exists = os.path.exists(df)
        n = len(json.load(open(df))) if exists else 0
        print(f"{name:10} base={b['base_url']:35} read_only={b['read_only']} "
              f"dataset={'OK' if exists else 'MISSING'} ({n} tasks)")
