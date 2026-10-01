#!/usr/bin/env python3
"""State, resume, and assembly utilities for final comparable SECAT batches.

The live runner deliberately writes one isolated result root per {benchmark, batch,
agent}. This module makes that split safe and reproducible: it enforces canonical
batch order, treats semantic failures as completed experiments but infrastructure
crashes as resumable, rebuilds batch summaries after a resume, and assembles one
clean final root per agent without mixing older runs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

# This file is used both as an importable module and as a user-facing script
# (``python analysis/comparable_results.py ...``).  When Python executes a file
# by path, sys.path[0] is the file's directory (analysis/), not the project root.
# Add the root explicitly before importing project modules so the CLI behaves the
# same way as ``from analysis import comparable_results`` from every cwd.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
BUILD_ID = "oca-v4.1.63-semantic-preservation-certificate-20260912"
AGENTS = ("oca", "toolcoder", "codeact")


def benchmark_name(name: str) -> str:
    n = str(name).strip().lower()
    aliases = {"spotify": "spotify_verified", "tmdb": "tmdb_verified"}
    n = aliases.get(n, n)
    if n not in {"spotify_verified", "tmdb_verified"}:
        raise ValueError("benchmark must be spotify/spotify_verified or tmdb/tmdb_verified")
    return n


def canonical_tasks(benchmark: str) -> list[int]:
    benchmark = benchmark_name(benchmark)
    if benchmark == "spotify_verified":
        from utils.spotify_task_policy import DEV2026_STRICT_COMPATIBLE_TASKS
        return list(DEV2026_STRICT_COMPATIBLE_TASKS)
    import benchmarks as B
    spec = B.get_benchmark(benchmark)
    return list(range(1, len(B.convert_toolcoder_dataset(spec["dataset_file"], spec["id_prefix"])) + 1))


def parse_spec(spec: str) -> list[int]:
    from utils.task_select import parse_task_spec
    out = parse_task_spec(spec)
    if out is None or not out:
        raise ValueError("an explicit non-empty finite task list is required")
    return list(out)


def compact(nums: list[int]) -> str:
    return ",".join(str(x) for x in nums)


def slug(nums: list[int]) -> str:
    """Return a deterministic filesystem-safe label for a task batch.

    Short selections retain the historical underscore-separated form so
    existing small-batch result roots and resume behavior stay unchanged.
    Long selections are range-compacted (1..100 -> ``1-100``). If a highly
    fragmented selection is still long, a deterministic digest keeps the
    filesystem component bounded without changing the actual task list.
    """
    values = [int(x) for x in nums]
    if not values:
        return "none"

    raw = "_".join(str(x) for x in values)
    if len(raw) <= 80:
        return raw

    ordered = sorted(set(values))
    ranges: list[str] = []
    start = prev = ordered[0]
    for value in ordered[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = value
    ranges.append(str(start) if start == prev else f"{start}-{prev}")

    compacted = "_".join(ranges)
    if len(compacted) <= 80:
        return compacted

    import hashlib

    canonical = ",".join(str(x) for x in ordered)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"{ordered[0]}-{ordered[-1]}_n{len(ordered)}_{digest}"


def state_dir(benchmark: str, series: str) -> Path:
    return RESULTS / benchmark_name(benchmark) / f"_comparable_{series}"


def manifest_path(benchmark: str, series: str) -> Path:
    return state_dir(benchmark, series) / "series.json"


def batch_base(series: str, tasks: list[int], agent: str) -> str:
    return f"cmp_{series}_b_{slug(tasks)}_{agent}"


def batch_root(benchmark: str, series: str, tasks: list[int], agent: str,
               token_profile: str = "adaptive") -> Path:
    base = batch_base(series, tasks, agent)
    if agent == "oca":
        tag = f"{base}_enforced_repair_typed_ep"
        if token_profile != "baseline":
            tag += f"_tok_{token_profile}"
        tag += "_isolated"
    else:
        tag = f"{base}_isolated"
    return RESULTS / benchmark_name(benchmark) / tag


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _task_num(value: Any) -> int | None:
    m = re.search(r"(\d+)\s*$", str(value or ""))
    return int(m.group(1)) if m else None


def _external_task_num(value: Any, benchmark: str) -> int | None:
    """Return a task number only for a canonical external dataset id.

    Isolated agent runs intentionally use opaque runtime ids.  Those ids must
    never become durable resume evidence if post-run relabelling was interrupted.
    """
    b = benchmark_name(benchmark)
    prefix = "spotify" if b == "spotify_verified" else "tmdb"
    m = re.fullmatch(rf"{prefix}_(\d+)", str(value or "").strip(), re.I)
    return int(m.group(1)) if m else None


def _oca_graph_task_nums(root: Path, benchmark: str) -> set[int]:
    graph = root / "evidence_graph.jsonl"
    if not graph.exists():
        return set()
    nums: set[int] = set()
    for line_no, raw in enumerate(graph.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except Exception as exc:
            raise RuntimeError(f"invalid evidence graph line {line_no} in {graph}: {exc}") from exc
        n = _external_task_num(obj.get("task_id"), benchmark)
        if n is not None:
            nums.add(n)
    return nums


def _run_records(root: Path, *, agent: str, benchmark: str, model: str) -> dict[int, tuple[Path, dict]]:
    runs = root / "runs"
    selected: dict[int, tuple[float, Path, dict]] = {}
    if not runs.is_dir():
        return {}
    for path in runs.glob("*.json"):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = obj.get("meta") or {}
        summary = obj.get("summary") or {}
        n = _external_task_num(meta.get("task_id") or summary.get("task_id"), benchmark)
        if n is None:
            # Opaque isolated runtime ids are intentionally ignored. If relabelling
            # was interrupted, the task remains resumable rather than being guessed
            # from trailing hash digits.
            continue
        # Infrastructure crash files and post-run-incomplete trajectories are
        # resumable and never count as completed experimental outcomes. A semantic
        # agent failure with a durable canonical trajectory is a valid completion.
        if (summary.get("crash_type") or summary.get("post_run_incomplete")
                or path.name.endswith("_CRASH.json")):
            continue
        if meta.get("agent") and str(meta.get("agent")) != agent:
            raise RuntimeError(f"{path}: agent={meta.get('agent')!r}, expected {agent!r}")
        if meta.get("benchmark") and str(meta.get("benchmark")) != benchmark:
            raise RuntimeError(f"{path}: benchmark={meta.get('benchmark')!r}, expected {benchmark!r}")
        if meta.get("model") and str(meta.get("model")) != model:
            raise RuntimeError(f"{path}: model={meta.get('model')!r}, expected {model!r}")
        if agent == "oca" and summary.get("oca_build_id") and summary.get("oca_build_id") != BUILD_ID:
            raise RuntimeError(
                f"{path}: OCA build={summary.get('oca_build_id')!r}, expected current {BUILD_ID!r}")
        stamp = str(meta.get("timestamp") or "")
        score = path.stat().st_mtime
        # mtime is authoritative for retry ordering; timestamp is retained only as
        # a deterministic tie-breaker in the tuple encoded into a tiny float epsilon.
        score += (sum(ord(c) for c in stamp) % 1000) * 1e-9
        if n not in selected or score > selected[n][0]:
            selected[n] = (score, path, obj)
    return {n: (rec[1], rec[2]) for n, rec in selected.items()}


def status(root: Path, expected: list[int], *, agent: str, benchmark: str, model: str) -> dict[str, Any]:
    benchmark = benchmark_name(benchmark)
    records = _run_records(root, agent=agent, benchmark=benchmark, model=model)
    unexpected = sorted(set(records) - set(expected))
    if unexpected:
        raise RuntimeError(f"unexpected completed task(s) in {root}: {unexpected}")
    durable = set(records)
    if agent == "oca":
        graph_nums = _oca_graph_task_nums(root, benchmark)
        unexpected_graph = sorted(graph_nums - set(expected))
        if unexpected_graph:
            raise RuntimeError(f"unexpected OCA evidence task(s) in {root}: {unexpected_graph}")
        # OCA completion requires both trajectory and evidence graph. This makes
        # an interruption during post-run relabelling safely resumable.
        durable &= graph_nums
    complete = [n for n in expected if n in durable]
    missing = [n for n in expected if n not in durable]
    return {"complete": complete, "missing": missing, "root": str(root)}


def _dedupe_oca_graph(root: Path, expected: list[int], complete: list[int]) -> None:
    graph = root / "evidence_graph.jsonl"
    if not graph.exists():
        if complete:
            raise RuntimeError(f"OCA root has completed runs but no evidence graph: {root}")
        return
    latest: dict[int, dict] = {}
    for raw in graph.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except Exception as exc:
            raise RuntimeError(f"invalid evidence graph line in {graph}: {exc}") from exc
        n = _task_num(obj.get("task_id"))
        if n in expected:
            latest[n] = obj
    missing_graph = [n for n in complete if n not in latest]
    if missing_graph:
        raise RuntimeError(f"OCA completed run(s) lack evidence graph entries: {missing_graph}")
    lines = [json.dumps(latest[n], ensure_ascii=False) for n in expected if n in latest]
    graph.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def rebuild(root: Path, expected: list[int], *, agent: str, benchmark: str, model: str) -> dict[str, Any]:
    st = status(root, expected, agent=agent, benchmark=benchmark, model=model)
    records = _run_records(root, agent=agent, benchmark=benchmark, model=model)
    summaries = []
    durable_complete = set(st["complete"])
    for n in expected:
        if n not in records or n not in durable_complete:
            continue
        obj = records[n][1]
        summary = dict(obj.get("summary") or {})
        summary.setdefault("task_id", (obj.get("meta") or {}).get("task_id"))
        summaries.append(summary)
    if agent == "oca":
        _dedupe_oca_graph(root, expected, st["complete"])
    import config
    from utils.logger import save_experiment_summary, _write_json_durable
    config.set_output_base(str(root))
    save_experiment_summary(agent, summaries)
    payload = {
        "requested_tasks": len(expected),
        "completed_results": len(st["complete"]),
        "missing_task_ids": st["missing"],
        "complete": not st["missing"],
        "evaluation_profile": "isolated",
        "model": model,
        "benchmark": benchmark,
        "oca_build_id": BUILD_ID if agent == "oca" else None,
        "resumable": bool(st["missing"]),
    }
    _write_json_durable(str(root / "summary" / "run_status.json"), payload)
    return payload


def _new_manifest(benchmark: str, series: str, model: str, token_profile: str) -> dict[str, Any]:
    return {
        "version": 1,
        "series": series,
        "benchmark": benchmark,
        "model": model,
        "evaluation_profile": "isolated",
        "oca_token_profile": token_profile,
        "oca_build_id": BUILD_ID,
        "canonical_tasks": canonical_tasks(benchmark),
        "completed_tasks": [],
        "current_batch": None,
        "batches": [],
        "agents": list(AGENTS),
    }


def begin(benchmark: str, series: str, tasks: list[int], model: str, token_profile: str) -> dict[str, Any]:
    benchmark = benchmark_name(benchmark)
    canonical = canonical_tasks(benchmark)
    # Selected IDs must be a contiguous slice of the canonical comparison order.
    try:
        start = canonical.index(tasks[0])
    except ValueError as exc:
        raise RuntimeError(f"task {tasks[0]} is not in the canonical {benchmark} comparison set") from exc
    if canonical[start:start + len(tasks)] != tasks:
        raise RuntimeError(
            f"batch must be contiguous in canonical order; got {tasks}, expected slice begins {canonical[start:start+len(tasks)]}")
    path = manifest_path(benchmark, series)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        checks = {
            "benchmark": benchmark, "model": model,
            "oca_token_profile": token_profile, "oca_build_id": BUILD_ID,
            "evaluation_profile": "isolated",
        }
        for key, value in checks.items():
            if manifest.get(key) != value:
                raise RuntimeError(
                    f"comparable series configuration mismatch for {key}: "
                    f"existing={manifest.get(key)!r}, requested={value!r}. Use a new COMPARABLE_SERIES_TAG.")
    else:
        manifest = _new_manifest(benchmark, series, model, token_profile)
    current = manifest.get("current_batch")
    if current is not None:
        if list(current.get("tasks") or []) != tasks:
            raise RuntimeError(
                f"series has an unfinished batch {current.get('tasks')}; rerun that exact batch before starting {tasks}")
        return manifest
    completed = list(manifest.get("completed_tasks") or [])
    # Idempotent crash window: the process may be killed after complete_batch()
    # atomically commits the manifest but before the shell prints its final success
    # line. Re-running that exact *last* batch must be harmless, not out-of-order.
    batches = list(manifest.get("batches") or [])
    if batches and list(batches[-1].get("tasks") or []) == tasks:
        if completed[-len(tasks):] != tasks:
            raise RuntimeError("series manifest is internally inconsistent for its last completed batch")
        for agent in AGENTS:
            root = Path(batches[-1]["roots"][agent])
            st = status(root, tasks, agent=agent, benchmark=benchmark, model=model)
            if st["missing"]:
                raise RuntimeError(
                    f"last completed batch manifest exists but {agent} artifacts are missing {st['missing']}; "
                    "restore the durable batch artifacts before continuing")
        out = dict(manifest)
        out["_already_completed"] = True
        return out
    expected_next = canonical[len(completed):len(completed) + len(tasks)]
    if tasks != expected_next:
        raise RuntimeError(
            f"out-of-order batch. Next canonical task IDs are {expected_next}; got {tasks}. "
            "This guard preserves ToolCoder cross-task state and fair agent comparison.")
    manifest["current_batch"] = {"tasks": tasks, "slug": slug(tasks)}
    _atomic_json(path, manifest)
    return manifest


def complete_batch(benchmark: str, series: str, tasks: list[int], model: str, token_profile: str) -> dict[str, Any]:
    benchmark = benchmark_name(benchmark)
    path = manifest_path(benchmark, series)
    if not path.exists():
        raise RuntimeError("series manifest does not exist; run begin first")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    current = manifest.get("current_batch") or {}
    if list(current.get("tasks") or []) != tasks:
        raise RuntimeError(f"current batch is {current.get('tasks')}, not {tasks}")
    roots = {}
    for agent in AGENTS:
        root = batch_root(benchmark, series, tasks, agent, token_profile)
        payload = rebuild(root, tasks, agent=agent, benchmark=benchmark, model=model)
        if not payload["complete"]:
            raise RuntimeError(f"cannot complete batch; {agent} is missing {payload['missing_task_ids']}")
        roots[agent] = str(root)
    manifest.setdefault("batches", []).append({"tasks": tasks, "slug": slug(tasks), "roots": roots})
    manifest["completed_tasks"] = list(manifest.get("completed_tasks") or []) + tasks
    manifest["current_batch"] = None
    _atomic_json(path, manifest)
    return manifest


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists() and src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def assemble(benchmark: str, series: str) -> Path:
    benchmark = benchmark_name(benchmark)
    path = manifest_path(benchmark, series)
    if not path.exists():
        raise RuntimeError(f"missing series manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("current_batch") is not None:
        raise RuntimeError(f"cannot assemble while batch {manifest['current_batch'].get('tasks')} is unfinished")
    expected = list(manifest.get("canonical_tasks") or [])
    completed = list(manifest.get("completed_tasks") or [])
    if completed != expected:
        raise RuntimeError(f"series incomplete: completed {len(completed)}/{len(expected)} canonical tasks")
    model = str(manifest["model"])
    token_profile = str(manifest["oca_token_profile"])
    final_base = RESULTS / benchmark / f"comparable_{series}"
    if final_base.exists():
        shutil.rmtree(final_base)
    final_base.mkdir(parents=True, exist_ok=True)

    final_manifest = dict(manifest)
    final_manifest["final_roots"] = {}
    for agent in AGENTS:
        dest = final_base / agent
        (dest / "runs").mkdir(parents=True, exist_ok=True)
        if agent == "oca":
            graph_objects: dict[int, dict] = {}
        summaries = []
        seen: set[int] = set()
        for batch in manifest.get("batches") or []:
            tasks = list(batch["tasks"])
            root = Path(batch["roots"][agent])
            recs = _run_records(root, agent=agent, benchmark=benchmark, model=model)
            for n in tasks:
                if n not in recs:
                    raise RuntimeError(f"{agent} batch {tasks} lost completed task {n}")
                if n in seen:
                    raise RuntimeError(f"duplicate task {n} while assembling {agent}")
                seen.add(n)
                src, obj = recs[n]
                shutil.copy2(src, dest / "runs" / src.name)
                sm = dict(obj.get("summary") or {})
                sm.setdefault("task_id", (obj.get("meta") or {}).get("task_id"))
                summaries.append(sm)
                # Copy trusted Spotify artifacts without mixing setup into scored traces.
                tid = str((obj.get("meta") or {}).get("task_id") or "")
                _copy_if_exists(root / "traces" / f"{tid}.jsonl", dest / "traces" / f"{tid}.jsonl")
                _copy_if_exists(root / "fixture" / f"{tid}.json", dest / "fixture" / f"{tid}.json")
            if agent == "oca":
                gp = root / "evidence_graph.jsonl"
                if not gp.exists():
                    raise RuntimeError(f"missing OCA evidence graph in {root}")
                for raw in gp.read_text(encoding="utf-8").splitlines():
                    if not raw.strip():
                        continue
                    obj = json.loads(raw)
                    n = _task_num(obj.get("task_id"))
                    if n in tasks:
                        graph_objects[n] = obj
        if sorted(seen) != sorted(expected):
            raise RuntimeError(f"{agent} assembled task set mismatch: {len(seen)}/{len(expected)}")
        if agent == "oca":
            missing_graph = [n for n in expected if n not in graph_objects]
            if missing_graph:
                raise RuntimeError(f"OCA final graph missing tasks: {missing_graph}")
            (dest / "evidence_graph.jsonl").write_text(
                "\n".join(json.dumps(graph_objects[n], ensure_ascii=False) for n in expected) + "\n",
                encoding="utf-8")
        import config
        from utils.logger import save_experiment_summary, _write_json_durable
        config.set_output_base(str(dest))
        save_experiment_summary(agent, summaries)
        _write_json_durable(str(dest / "summary" / "run_status.json"), {
            "requested_tasks": len(expected), "completed_results": len(expected),
            "missing_task_ids": [], "complete": True, "evaluation_profile": "isolated",
            "model": model, "benchmark": benchmark,
            "oca_build_id": BUILD_ID if agent == "oca" else None,
            "series": series,
        })
        final_manifest["final_roots"][agent] = str(dest)
    _atomic_json(final_base / "manifest.json", final_manifest)
    return final_base



def _toolcoder_examples_from_code(code: str, valid: dict[str, set[str]], examples: dict[str, dict[str, str]]) -> None:
    """Mirror Spotify ToolCoder's successful-example learner from durable code."""
    import ast
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return
    lines = (code or "").splitlines()
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            for stmt in node.body:
                start = stmt.lineno - 1
                end = getattr(stmt, "end_lineno", stmt.lineno)
                body.append("\n".join(lines[start:end]).strip())
            break
    pat = re.compile(r'requests_wrapper\.(get|post|put|delete)\(\s*[f]?["\'](https?://[^\s"\']+)')
    base = "https://api.spotify.com/v1"
    for statement in body:
        for method, url in pat.findall(statement):
            path = url.replace(base, "")
            method = method.lower()
            if method in valid.get(path, set()):
                examples.setdefault(path, {})[method] = statement.strip()


def sync_toolcoder_state(benchmark: str, series: str, state_path: Path, model: str, token_profile: str) -> dict[str, Any]:
    """Rebuild Spotify ToolCoder learned examples from durable completed runs only.

    This is called before every resumable ToolCoder invocation.  It deliberately
    overwrites any process-interruption residue in the state file, making a resumed
    batched run equivalent to carrying the official mutable toolbox forward only
    after tasks whose logs are durable.
    """
    benchmark = benchmark_name(benchmark)
    if benchmark != "spotify_verified":
        _atomic_json(state_path, {"version": 1, "profile": "n/a", "implementation_examples": {}})
        return {"completed": 0, "examples": 0}
    mp = manifest_path(benchmark, series)
    if not mp.exists():
        raise RuntimeError("series manifest does not exist; run begin first")
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    for key, value in {"model": model, "oca_token_profile": token_profile, "oca_build_id": BUILD_ID}.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"series configuration mismatch for {key}")

    import benchmarks as B
    toolfile = B.get_benchmark("spotify_verified").get("toolcoder_oas_file")
    raw = json.loads(Path(toolfile).read_text(encoding="utf-8"))
    valid: dict[str, set[str]] = {}
    for item in raw:
        path = str(item.get("path") or "")
        for method, spec in item.items():
            if method.lower() in {"get", "post", "put", "delete"} and isinstance(spec, dict):
                valid.setdefault(path, set()).add(method.lower())

    ordered_records: dict[int, dict] = {}
    # Completed batches are authoritative.
    for batch in manifest.get("batches") or []:
        tasks = list(batch.get("tasks") or [])
        root = Path(batch["roots"]["toolcoder"])
        for n, (_path, obj) in _run_records(root, agent="toolcoder", benchmark=benchmark, model=model).items():
            if n in tasks:
                ordered_records[n] = obj
    # The current batch may be partially complete. Include only its durable runs.
    current = manifest.get("current_batch")
    if current:
        tasks = list(current.get("tasks") or [])
        root = batch_root(benchmark, series, tasks, "toolcoder", token_profile)
        recs = _run_records(root, agent="toolcoder", benchmark=benchmark, model=model)
        # Fail-fast orchestration means completed tasks in a partial ToolCoder batch
        # must form a prefix. Reject any leapfrog state rather than silently mixing it.
        seen_gap = False
        for n in tasks:
            if n not in recs:
                seen_gap = True
            elif seen_gap:
                raise RuntimeError(
                    f"ToolCoder partial batch contains completed task {n} after an unfinished earlier task; "
                    "use a new comparable series tag rather than mixing state")
            else:
                ordered_records[n] = recs[n][1]

    examples: dict[str, dict[str, str]] = {}
    canonical = canonical_tasks(benchmark)
    completed_count = 0
    for n in canonical:
        obj = ordered_records.get(n)
        if obj is None:
            continue
        completed_count += 1
        execute = None
        for turn in obj.get("turns") or []:
            inp = str(turn.get("llm_input") or "")
            if "Spotify ToolCoder: execute" in inp:
                execute = turn
        if not execute:
            continue
        execution = execute.get("execution") or {}
        success = bool(execution.get("success"))
        stdout = str(execution.get("stdout") or "")
        if success and "Failed" not in stdout:
            _toolcoder_examples_from_code(str(execute.get("code") or ""), valid, examples)
    payload = {
        "version": 1, "profile": "dev2026",
        "implementation_examples": examples,
        "source": "durable_comparable_runs",
        "completed_task_count": completed_count,
    }
    _atomic_json(state_path, payload)
    return {"completed": completed_count, "examples": sum(len(v) for v in examples.values())}

def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("begin")
    p.add_argument("--benchmark", required=True); p.add_argument("--series", required=True)
    p.add_argument("--tasks", required=True); p.add_argument("--model", required=True)
    p.add_argument("--token-profile", default="adaptive")

    p = sub.add_parser("status")
    p.add_argument("--root", required=True, type=Path); p.add_argument("--expected", required=True)
    p.add_argument("--agent", required=True, choices=AGENTS); p.add_argument("--benchmark", required=True)
    p.add_argument("--model", required=True); p.add_argument("--missing-only", action="store_true")

    p = sub.add_parser("rebuild")
    p.add_argument("--root", required=True, type=Path); p.add_argument("--expected", required=True)
    p.add_argument("--agent", required=True, choices=AGENTS); p.add_argument("--benchmark", required=True)
    p.add_argument("--model", required=True)

    p = sub.add_parser("complete")
    p.add_argument("--benchmark", required=True); p.add_argument("--series", required=True)
    p.add_argument("--tasks", required=True); p.add_argument("--model", required=True)
    p.add_argument("--token-profile", default="adaptive")

    p = sub.add_parser("assemble")
    p.add_argument("--benchmark", required=True); p.add_argument("--series", required=True)

    p = sub.add_parser("sync-toolcoder")
    p.add_argument("--benchmark", required=True); p.add_argument("--series", required=True)
    p.add_argument("--state-path", required=True, type=Path); p.add_argument("--model", required=True)
    p.add_argument("--token-profile", default="adaptive")

    args = ap.parse_args()
    if args.cmd == "begin":
        m = begin(args.benchmark, args.series, parse_spec(args.tasks), args.model, args.token_profile)
        print(json.dumps({
            "current_batch": m.get("current_batch"),
            "completed": len(m.get("completed_tasks") or []),
            "already_completed": bool(m.get("_already_completed")),
        }))
    elif args.cmd == "status":
        b = benchmark_name(args.benchmark); ex = parse_spec(args.expected)
        st = status(args.root, ex, agent=args.agent, benchmark=b, model=args.model)
        print(compact(st["missing"]) if args.missing_only else json.dumps(st, sort_keys=True))
    elif args.cmd == "rebuild":
        b = benchmark_name(args.benchmark); ex = parse_spec(args.expected)
        print(json.dumps(rebuild(args.root, ex, agent=args.agent, benchmark=b, model=args.model), sort_keys=True))
    elif args.cmd == "complete":
        m = complete_batch(args.benchmark, args.series, parse_spec(args.tasks), args.model, args.token_profile)
        print(json.dumps({"completed": len(m.get("completed_tasks") or []), "total": len(m.get("canonical_tasks") or [])}))
    elif args.cmd == "assemble":
        print(assemble(args.benchmark, args.series))
    elif args.cmd == "sync-toolcoder":
        print(json.dumps(sync_toolcoder_state(
            args.benchmark, args.series, args.state_path, args.model, args.token_profile), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
