"""
run_experiment.py — main entry point.

Usage:
  python run_experiment.py --agent codeact --tasks 30 --model gpt-5.4-mini
  python run_experiment.py --agent codeact --tasks 5  --model gpt-5.4-mini  # quick test
  python run_experiment.py --agent codeact --benchmark tmdb --sample tmdb_037  # 1-task smoke test
  python run_experiment.py --agent codeact --benchmark tmdb --sample 37        # same, by number
"""
import argparse, json, os, sys, importlib
from pathlib import Path


def _is_spotify_fixture_rate_limit(error: BaseException) -> bool:
    try:
        from utils.spotify_runtime import SpotifyRateLimitError
        return isinstance(error, SpotifyRateLimitError)
    except Exception:
        return False

def parse_args():
    p = argparse.ArgumentParser(description="SECAT Experiment Runner")
    p.add_argument("--agent",  default="codeact",     help="Agent to run")
    p.add_argument("--tasks",  default="5",
                   help="How many tasks (a count like 5) OR a selection: "
                        "single '10', range '20-25', list '1,5,9', or 'all'.")
    import config as _cfg
    p.add_argument("--model",  default=_cfg.DEFAULT_MODEL, help="LLM model name")
    p.add_argument("--start",  type=int, default=0,
                   help="Start index (only used when --tasks is a plain count)")
    p.add_argument("--sample", default=None,
                   help="Run EXACTLY ONE task for a cheap smoke test. Accepts a task id "
                        "(e.g. tmdb_037) or a 1-based number (e.g. 37). Overrides --tasks.")
    p.add_argument("--expect-task-ids", default=None,
                   help="Safety assertion for orchestration scripts. Comma/range spec of the "
                        "exact 1-based task IDs that must be selected before any agent/API call.")
    p.add_argument("--tasks-file", default="data/tmdb_tasks.json",
                   help="Task file (use data/tmdb_tasks_gold.json for gold api_list)")
    p.add_argument("--benchmark", default=None,
                   help="Load tasks from a registered benchmark (tmdb, spotify, ...) using its "
                        "ToolCoder-format dataset. Overrides --tasks-file when set.")
    p.add_argument("--apibank-root", default=None,
                   help="API-Bank root containing ToolManager/apis/Level-1/Level-2 data. Used with --benchmark apibank_lv1 or apibank_lv2; alternatively set APIBANK_ROOT.")
    p.add_argument("--apibank-bootstrap", dest="bootstrap", action="store_true",
                   help="With --benchmark apibank_lv1/apibank_lv2, download/install the ToolCoder API-Bank snapshot if missing.")
    p.add_argument("--state-mode", choices=["published", "replay"], default="published",
                   help="API-Bank only: published=fresh ToolManager per target (paper-compatible default); replay=explicit stateful sensitivity mode.")
    p.add_argument("--evaluation-profile", default="isolated",
                   choices=["isolated", "reported95"],
                   help="Agent/evaluator boundary. isolated (default) removes benchmark "
                        "solutions, route hints, gold-derived flags, and benchmark task IDs "
                        "before agent execution. reported95 preserves the old benchmark-assistance "
                        "boundary for compatibility, but uses current validators and is not "
                        "a clean or bit-exact historical rerun.")
    p.add_argument("--exec-mode", default=None, choices=["script", "kernel"],
                   help="CodeAct execution backend: 'script' (exec, exposes S4 perception "
                        "failures) or 'kernel' (IPython, auto-displays last expression like "
                        "original CodeAct). Overrides config.EXECUTION_MODE.")
    p.add_argument("--fresh", action="store_true",
                   help="Clear this benchmark/agent's runs dir before running, so old "
                        "runs don't accumulate and get mixed into scoring.")
    p.add_argument("--fail-fast-crash", action="store_true",
                   help="Stop the selected run immediately after any task-level exception. "
                        "Used by resumable comparable batches so later tasks never leapfrog "
                        "an infrastructure crash or ToolCoder state transition.")
    p.add_argument("--continue-infrastructure", action="store_true",
                   help="API-Bank only: continue running later selected tasks after an "
                        "infrastructure_error. The run remains non-reportable until those "
                        "infrastructure tasks are rerun successfully.")
    p.add_argument("--inject-mode", default=None,
                   choices=["wrapper", "ast", "regex", "llm"],
                   help="Log capture mode for this run (overrides config.LOG_INJECTION_MODE). "
                        "wrapper=library-boundary logging, agent code unmodified (default); "
                        "ast=static print injection; llm=LLM-based; regex=legacy.")
    p.add_argument("--oca-version", default=None, choices=["v2"],
                   help="OCA implementation included in this artifact.")
    p.add_argument("--oca-no-planner", action="store_true",
                   help="Ablation for OCA v2: disable the one-call evidence planner.")
    p.add_argument("--oca-no-deterministic-answer", action="store_true",
                   help="Ablation for OCA v2: disable host-side deterministic "
                        "answer materialization and always invoke Phase B.")
    p.add_argument("--oca-no-lineage-completion", action="store_true",
                   help="Ablation for OCA v2: disable exact downstream retries "
                        "for compiler-selected entity ids.")
    p.add_argument("--result-tag", default=None,
                   help="Optional result-directory base tag for non-OCA agents. Used by the comparable batch runner so finite batches never overwrite each other.")
    p.add_argument("--resume-from", default=None,
                   help="API-Bank only: import a compatible incomplete audit5-v1 result directory into a new corrected run.")
    p.add_argument("--oca-result-tag", default=None,
                   help="Optional OCA result-directory base tag. Useful for finite selected batches so --fresh never overwrites another completed batch.")
    p.add_argument("--oca-token-profile", default=None,
                   choices=["baseline", "receipt", "adaptive", "lean", "adaptive_lean"],
                   help="OCA token-efficiency profile. baseline preserves published prompts; "
                        "receipt uses the legacy heuristic receipt; adaptive uses LLM-planned, "
                        "schema-grounded observation projection; lean applies the legacy receipt "
                        "plus context compaction; adaptive_lean combines adaptive projection with "
                        "the safe context-compaction switches.")
    p.add_argument("--oca-observation-mode", default=None, choices=["raw", "receipt", "adaptive"],
                   help="Override the Phase-A API observation representation.")
    p.add_argument("--oca-phase-a-history", default=None, choices=["full", "compact"],
                   help="Keep the full Phase-A transcript or compact older code/receipts.")
    p.add_argument("--oca-planner-catalog", default=None, choices=["full", "compact"],
                   help="Planner catalog representation. compact retains every valid route but "
                        "omits verbose OAS descriptions.")
    p.add_argument("--oca-phase-b-focus-fallback", action="store_true",
                   help="When compiler focus is empty, focus Phase B on answer plan steps and "
                        "their dependencies instead of immediately serializing the full ledger.")
    p.add_argument("--oca-ledger-max-chars", type=int, default=None,
                   help="Override the Phase-B serialized ledger character budget.")
    p.add_argument("--contract-mode", default=None,
                   choices=["off", "advisory", "enforced"],
                   help="Step 0 evidence-contract mode for the OCA agent. "
                        "'advisory' shows the contract to Phase B (verifier logs only); "
                        "'enforced' uses the identical prompt but the verifier gates "
                        "emission. Outputs go to results/<bench>/oca_<mode>/ so the "
                        "three variants never mix.")
    p.add_argument("--repair-mode", default=None,
                   choices=["off", "generic", "typed", "shuffled"],
                   help="RQ3 certificate-guided re-fetch (needs --contract-mode enforced). "
                        "off=abstain on rejection; generic=control B (fetch more, no gap); "
                        "typed=method C (typed evidence gap, delta-gated); "
                        "shuffled=control D (wrong gap). Output dir suffixed with the mode.")
    hint_group = p.add_mutually_exclusive_group()
    hint_group.add_argument("--repair-endpoint-hint", dest="repair_endpoint_hint",
                            action="store_true", default=None,
                            help="Include an endpoint-family hint in typed repair.")
    hint_group.add_argument("--no-repair-endpoint-hint", dest="repair_endpoint_hint",
                            action="store_false",
                            help="Disable endpoint-family hints for the schema-only ablation.")
    p.add_argument("--spotify-profile", default="legacy", choices=["legacy", "dev2026"],
                   help="Spotify API compatibility profile. legacy preserves the original "
                        "RestBench endpoints; dev2026 applies only faithful endpoint migrations.")
    p.add_argument("--allow-spotify-writes", action="store_true",
                   help="Allow POST/PUT/DELETE/PATCH Spotify tasks. Use only with a disposable "
                        "benchmark account; writes are blocked by default.")
    p.add_argument("--spotify-reset-fixture", action="store_true",
                   help="Destructively recreate the benchmark account fixture before every task. "
                        "Requires ALLOW_SPOTIFY_RESET=YES. Setup writes are enabled only while "
                        "resetting; agent writes still require --allow-spotify-writes.")
    return p.parse_args()

def _select_one(all_tasks, sample):
    """Return a 1-element list with the task matching `sample` (a task id like
    'tmdb_037' or a 1-based number like '37'). Raises if not found."""
    from utils.task_select import task_id_to_num
    s = str(sample).strip()
    # exact id match first
    for t in all_tasks:
        if str(t.get("id", "")).strip().lower() == s.lower():
            return [t]
    # else treat as a 1-based number -> all_tasks[num-1]
    num = task_id_to_num(s) if not s.isdigit() else int(s)
    if num is not None and 1 <= num <= len(all_tasks):
        return [all_tasks[num - 1]]
    raise SystemExit(
        f"--sample {sample!r} matched no task. Use an id like 'tmdb_037' or a "
        f"1-based number 1..{len(all_tasks)}.")


def load_tasks(tasks_arg, start: int = 0, tasks_file: str = "data/tmdb_tasks.json",
               benchmark: str = None, sample: str = None, *,
               exact_selection: bool = False) -> list:
    # If a benchmark is named, load its ToolCoder-format dataset via the registry.
    if benchmark:
        import benchmarks as B
        bench = B.get_benchmark(benchmark)
        all_tasks = B.convert_toolcoder_dataset(bench["dataset_file"], bench["id_prefix"])
        for task in all_tasks:
            task["benchmark"] = benchmark
        from utils.task_select import parse_task_spec, task_id_to_num
        # --sample wins: run exactly one task (cheap smoke test)
        if sample is not None:
            sel = _select_one(all_tasks, sample)
            print(f"[INFO] benchmark={benchmark}: SAMPLE run of 1 task "
                  f"(id={sel[0].get('id')}) — smoke test")
            return sel
        arg = str(tasks_arg).strip().lower()
        if arg in ("all", ""):
            sel = all_tasks
        elif (not exact_selection) and arg.isdigit() and "," not in arg and "-" not in arg:
            # Legacy direct CLI behavior: a bare integer means a COUNT.
            # Clean selected runs pass --expect-task-ids and therefore set
            # exact_selection=True, where a bare integer is an exact task ID.
            sel = all_tasks[start:start + int(arg)]
        else:
            # parse_task_spec returns 1-based task numbers -> index with n-1
            nums = parse_task_spec(arg)
            sel = [all_tasks[n - 1] for n in nums if 1 <= n <= len(all_tasks)]
        print(f"[INFO] benchmark={benchmark}: loaded {len(sel)} tasks "
              f"(base_url={bench['base_url']}, read_only={bench['read_only']})")
        return sel

    if not os.path.exists(tasks_file):
        print("[INFO] Tasks not found. Downloading...")
        from utils.download_data import download
        download()

    with open(tasks_file) as f:
        all_tasks = json.load(f)

    # Ensure every task has an id
    for i, t in enumerate(all_tasks):
        if "id" not in t:
            t["id"] = f"tmdb_{i+1:03d}"

    # --sample wins here too
    if sample is not None:
        sel = _select_one(all_tasks, sample)
        print(f"[INFO] SAMPLE run of 1 task (id={sel[0].get('id')}) — smoke test")
        return sel

    # --tasks may be a plain count ("5") or a selection spec ("10", "20-25", "1,5,9", "all")
    from utils.task_select import parse_task_spec, task_id_to_num
    arg = str(tasks_arg).strip().lower()
    is_plain_count = arg.isdigit() and ("," not in arg) and ("-" not in arg)

    # A plain integer means "first N from --start" (backward compatible) UNLESS
    # the user clearly wants a selection. We treat a bare integer as a COUNT.
    if is_plain_count and not exact_selection:
        n = int(arg)
        selected = all_tasks[start: start + n]
        end_idx = start + len(selected) - 1 if selected else start
        print(f"[INFO] Loaded {len(selected)} tasks (index {start}–{end_idx} of {len(all_tasks)} total)")
        return selected

    # Otherwise it's a selection spec by task NUMBER (tmdb_014 -> 14).
    nums = parse_task_spec(arg)  # None means 'all'
    if nums is None:
        print(f"[INFO] Loaded all {len(all_tasks)} tasks")
        return all_tasks
    sel = [t for t in all_tasks if task_id_to_num(t["id"]) in nums]
    found = {task_id_to_num(t["id"]) for t in sel}
    missing = [n for n in nums if n not in found]
    print(f"[INFO] Selected {len(sel)} task(s) by spec '{tasks_arg}': "
          f"{[t['id'] for t in sel]}")
    if missing:
        print(f"[WARN] No task file entry for numbers: {missing}")
    return sel

def load_agent(agent_name: str):
    from config import AGENTS
    if agent_name not in AGENTS:
        print(f"[ERROR] Unknown agent '{agent_name}'.")
        print(f"  Available: {list(AGENTS.keys())}")
        sys.exit(1)
    module_path = AGENTS[agent_name]
    module = importlib.import_module(module_path)
    return module


def _snapshot_run_files(run_dir: str) -> dict[Path, tuple[int, int]]:
    """Capture enough state to identify the one trajectory written by a task."""
    root = Path(run_dir)
    if not root.exists():
        return {}
    return {path: (path.stat().st_mtime_ns, path.stat().st_size)
            for path in root.glob("*.json") if path.is_file()}


def _graph_line_count(result_base: str) -> int:
    path = Path(result_base) / "evidence_graph.jsonl"
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8").splitlines())


def _restore_post_run_identity(*, before_files: dict[Path, tuple[int, int]],
                               graph_lines_before: int, external_task: dict,
                               boundary: dict, summary: dict, agent_name: str,
                               benchmark: str | None, evaluation_profile: str,
                               expect_evidence_graph: bool = False) -> Path:
    """Restore dataset identity only after the isolated agent has returned.

    The model sees an opaque task id and no gold metadata.  Post-execution, the
    saved trajectory and evidence graph are relabelled with the authoritative
    dataset id so verification, resume, and packaging remain functional.
    """
    import config

    run_dir = Path(config.RUN_DIR)
    after = _snapshot_run_files(str(run_dir))
    changed = [path for path, state in after.items()
               if path not in before_files or before_files[path] != state]
    runtime_id = str(boundary.get("runtime_task_id") or "")
    if len(changed) != 1:
        matching = [p for p in changed if runtime_id and runtime_id in p.name]
        if len(matching) == 1:
            changed = matching
        else:
            raise RuntimeError(
                "could not identify exactly one newly written trajectory: "
                f"changed={[p.name for p in changed]}, runtime_id={runtime_id!r}")

    source = changed[0]
    obj = json.loads(source.read_text(encoding="utf-8"))
    external_id = str(external_task.get("id") or runtime_id or "unknown")
    external_benchmark = str(benchmark or external_task.get("benchmark") or "")
    instruction = str(external_task.get("instruction") or
                      external_task.get("query") or "")

    meta = obj.setdefault("meta", {})
    meta.update({
        "task_id": external_id,
        "runtime_task_id": runtime_id,
        "benchmark": external_benchmark,
        "instruction": instruction,
        "api_list": list(external_task.get("api_list") or []),
        "ground_truth": external_task.get("ground_truth"),
        "evaluation_profile": evaluation_profile,
        "agent_boundary": boundary,
    })
    summary.update({
        "task_id": external_id,
        "runtime_task_id": runtime_id,
        "evaluation_profile": evaluation_profile,
        "agent_boundary": boundary,
    })
    obj["summary"] = summary

    timestamp = str(meta.get("timestamp") or "run")
    target = run_dir / f"{agent_name}_{external_id}_{timestamp}.json"
    if target != source and target.exists():
        raise RuntimeError(f"refusing to overwrite existing trajectory: {target}")
    temp = source.with_suffix(source.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    if target != source:
        os.replace(temp, target)
        source.unlink()
    else:
        os.replace(temp, source)
        target = source

    graph_path = Path(config.RESULTS_BASE) / "evidence_graph.jsonl"
    if graph_path.exists():
        lines = graph_path.read_text(encoding="utf-8").splitlines()
        patched = 0
        for index in range(graph_lines_before, len(lines)):
            if not lines[index].strip():
                continue
            record = json.loads(lines[index])
            if str(record.get("task_id") or "") != runtime_id:
                continue
            record.update({
                "task_id": external_id,
                "runtime_task_id": runtime_id,
                "benchmark": external_benchmark,
                "query": instruction,
                "evaluation_profile": evaluation_profile,
                "agent_boundary": boundary,
            })
            lines[index] = json.dumps(record, ensure_ascii=False)
            patched += 1
        if expect_evidence_graph and patched != 1:
            raise RuntimeError(
                f"expected one new OCA evidence-graph entry, patched {patched}")
        graph_path.write_text("\n".join(lines) + ("\n" if lines else ""),
                              encoding="utf-8")
    elif expect_evidence_graph:
        raise RuntimeError("OCA v2 completed without writing evidence_graph.jsonl")

    return target


def _rewrite_saved_summary(path: Path, summary: dict) -> None:
    """Durably replace only the summary of an already rebound trajectory."""
    obj = json.loads(path.read_text(encoding="utf-8"))
    obj["summary"] = summary
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def main():
    args   = parse_args()

    # API-Bank is an executable local-tool benchmark, not a RestBench HTTP
    # dataset. Route it before the generic RestBench loader/agent imports so
    # its official ToolManager state and checker remain isolated.
    _benchmark_name = str(getattr(args, "benchmark", "") or "").strip().lower().replace("-", "_")
    if _benchmark_name in {"apibank", "api_bank", "apibank_lv1", "api_bank_lv1", "apibank_lv2", "api_bank_lv2"}:
        from run_apibank import run_from_namespace
        return run_from_namespace(args)

    import config
    config.EVALUATION_PROFILE = str(args.evaluation_profile)
    if args.evaluation_profile == "reported95":
        config.OCA_LEGACY_BENCHMARK_HINTS = True
        config.OCA_SEMANTIC_ADAPTERS = True
        print("[WARNING] evaluation-profile=reported95 preserves the historical "
              "non-isolated assistance boundary but uses current validators. "
              "Use the bundled v3.3.1 snapshot for bit-exact source reproduction; "
              "do not report this profile as a clean rerun.")
    else:
        config.OCA_LEGACY_BENCHMARK_HINTS = False
        config.OCA_SEMANTIC_ADAPTERS = False
        print("[SAFETY] evaluation-profile=isolated: gold-only task metadata will "
              "not cross the agent boundary.")

    # Per-run override of the log injection mode (in-memory; config file untouched).
    if args.inject_mode:
        import config
        config.LOG_INJECTION_MODE = args.inject_mode
        print(f"[INFO] Log injection mode for this run: {args.inject_mode}")

    tasks  = load_tasks(
        args.tasks, args.start, args.tasks_file,
        getattr(args, "benchmark", None), getattr(args, "sample", None),
        exact_selection=bool(getattr(args, "expect_task_ids", None)),
    )

    from utils.evaluation_isolation import runtime_benchmark_name
    _runtime_api = runtime_benchmark_name(args.benchmark) if getattr(args, "benchmark", None) else ""

    # All execution backends and subprocesses inherit the same benchmark runtime.
    _project_root = os.path.dirname(os.path.abspath(__file__))
    os.environ["SECAT_PROJECT_ROOT"] = _project_root
    os.environ["SECAT_EVALUATION_PROFILE"] = args.evaluation_profile
    if getattr(args, "benchmark", None):
        os.environ["SECAT_BENCHMARK"] = (
            args.benchmark if args.evaluation_profile == "reported95"
            else runtime_benchmark_name(args.benchmark))
    if _runtime_api == "spotify":
        os.environ["SPOTIFY_API_PROFILE"] = args.spotify_profile
        if args.allow_spotify_writes:
            os.environ["SECAT_ALLOW_SPOTIFY_WRITES"] = "YES"
        else:
            os.environ.pop("SECAT_ALLOW_SPOTIFY_WRITES", None)
        from utils.spotify_runtime import validate_task_selection
        _spotify_selection = validate_task_selection(
            tasks, selected_profile=args.spotify_profile,
            allow_writes=bool(args.allow_spotify_writes))
        for _task in tasks:
            _task["spotify_profile"] = args.spotify_profile
        print("[SPOTIFY] " + json.dumps(_spotify_selection, sort_keys=True))
        if args.agent == "toolcoder" and args.spotify_reset_fixture:
            # The supplied ToolCoder Spotify runner prepends init_spotify() to every
            # execution/revision. Its adapter mirrors that behavior inside each child.
            os.environ["SECAT_TOOLCODER_RESET_EACH_EXEC"] = "YES"
        else:
            os.environ.pop("SECAT_TOOLCODER_RESET_EACH_EXEC", None)

    # Fail closed before importing/running an agent if orchestration selected a
    # different set than intended. This prevents a bare integer such as "97"
    # (historically interpreted as a count) from silently becoming 97 API tasks.
    if args.expect_task_ids:
        from utils.task_select import parse_task_spec, task_id_to_num
        expected = parse_task_spec(str(args.expect_task_ids).strip().lower())
        if expected is None:
            expected = set(range(1, len(tasks) + 1))
        else:
            expected = set(expected)
        actual = {task_id_to_num(t.get("id")) for t in tasks}
        if None in actual or actual != expected:
            raise SystemExit(
                "TASK SELECTION SAFETY CHECK FAILED: "
                f"expected={sorted(expected)}, actual={sorted(x for x in actual if x is not None)}. "
                "No agent/API call was made.")
        print(f"[SAFETY] Exact task selection confirmed: {sorted(actual)}")

    # Record the selected model/provider in trusted runtime state before agent
    # import.  Agents still receive ``model`` explicitly; these values are for
    # diagnostics/result isolation only and never cross the benchmark boundary.
    os.environ["SECAT_MODEL"] = str(args.model)
    from utils.model_provider import provider_for_model, model_run_slug, public_model_settings
    _llm_provider = provider_for_model(args.model)
    _model_runtime = public_model_settings(args.model)
    os.environ["SECAT_LLM_PROVIDER"] = _llm_provider

    agent  = load_agent(args.agent)

    # organize outputs as results/<benchmark>/<agent>/...
    if getattr(args, "exec_mode", None):
        config.EXECUTION_MODE = args.exec_mode
        print(f"[INFO] execution mode for this run: {args.exec_mode}")
    _bench = getattr(args, "benchmark", None) or "tmdb"
    # OCA variants always receive distinct output directories so v1/v2 and
    # repair ablations cannot be mixed accidentally.
    _agent_dir = args.agent
    _generic_result_tag = str(getattr(args, "result_tag", "") or "").strip()
    if _generic_result_tag:
        import re as _re
        if not _re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,239}", _generic_result_tag):
            raise SystemExit("--result-tag must be a simple filesystem-safe name (letters, digits, '.', '_' or '-')")
        if args.agent == "oca" and getattr(args, "oca_result_tag", None):
            raise SystemExit("use only one of --result-tag or --oca-result-tag for OCA")
        _agent_dir = _generic_result_tag
    if args.agent == "oca":
        if getattr(args, "oca_version", None):
            config.OCA_VERSION = args.oca_version
        if getattr(args, "oca_no_planner", False):
            config.OCA_PLANNER_ENABLED = False
        if getattr(args, "oca_no_deterministic_answer", False):
            config.OCA_DETERMINISTIC_ANSWER_FIRST = False
        if getattr(args, "oca_no_lineage_completion", False):
            config.OCA_DETERMINISTIC_LINEAGE_COMPLETION = False

        token_profile = (getattr(args, "oca_token_profile", None) or
                         getattr(config, "OCA_TOKEN_PROFILE", "baseline"))
        config.OCA_TOKEN_PROFILE = token_profile
        if getattr(args, "oca_token_profile", None) is not None:
            # Named profiles are reproducible presets. Individual switches below
            # can still override any one setting for ablation experiments.
            config.OCA_OBSERVATION_MODE = "raw"
            config.OCA_PHASE_A_HISTORY = "full"
            config.OCA_PLANNER_CATALOG_MODE = "full"
            config.OCA_PHASE_B_FOCUS_FALLBACK = False
            config.OCA_LEDGER_MAX_CHARS = 24000
        if token_profile == "receipt":
            config.OCA_OBSERVATION_MODE = "receipt"
        elif token_profile == "adaptive":
            config.OCA_OBSERVATION_MODE = "adaptive"
        elif token_profile == "lean":
            config.OCA_OBSERVATION_MODE = "receipt"
            config.OCA_PHASE_A_HISTORY = "compact"
            config.OCA_PLANNER_CATALOG_MODE = "compact"
            config.OCA_PHASE_B_FOCUS_FALLBACK = True
            config.OCA_LEDGER_MAX_CHARS = min(
                int(getattr(config, "OCA_LEDGER_MAX_CHARS", 24000)), 12000)
        elif token_profile == "adaptive_lean":
            config.OCA_OBSERVATION_MODE = "adaptive"
            config.OCA_PHASE_A_HISTORY = "compact"
            config.OCA_PLANNER_CATALOG_MODE = "compact"
            config.OCA_PHASE_B_FOCUS_FALLBACK = True
            config.OCA_LEDGER_MAX_CHARS = min(
                int(getattr(config, "OCA_LEDGER_MAX_CHARS", 24000)), 12000)
        # Explicit switches override the selected profile and make ablations easy.
        if getattr(args, "oca_observation_mode", None):
            config.OCA_OBSERVATION_MODE = args.oca_observation_mode
        if getattr(args, "oca_phase_a_history", None):
            config.OCA_PHASE_A_HISTORY = args.oca_phase_a_history
        if getattr(args, "oca_planner_catalog", None):
            config.OCA_PLANNER_CATALOG_MODE = args.oca_planner_catalog
        if getattr(args, "oca_phase_b_focus_fallback", False):
            config.OCA_PHASE_B_FOCUS_FALLBACK = True
        if getattr(args, "oca_ledger_max_chars", None) is not None:
            config.OCA_LEDGER_MAX_CHARS = max(1000, int(args.oca_ledger_max_chars))

        version = str(getattr(config, "OCA_VERSION", "v2")).lower()
        _agent_dir = (str(getattr(args, "oca_result_tag", "") or "").strip()
                      or _generic_result_tag
                      or getattr(config, "OCA_RESULT_TAG", f"oca_{version}"))

        contract_mode = (getattr(args, "contract_mode", None) or
                         getattr(config, "OCA_CONTRACT_MODE", "enforced"))
        config.OCA_CONTRACT_MODE = contract_mode
        if contract_mode in ("advisory", "enforced"):
            _agent_dir = f"{_agent_dir}_{contract_mode}"
        print(f"[INFO] OCA version={version}, contract={contract_mode}, "
              f"planner={'on' if config.OCA_PLANNER_ENABLED else 'off'}, "
              f"det-answer={'on' if config.OCA_DETERMINISTIC_ANSWER_FIRST else 'off'}, "
              f"lineage-completion={'on' if config.OCA_DETERMINISTIC_LINEAGE_COMPLETION else 'off'}")
        if not config.OCA_DETERMINISTIC_ANSWER_FIRST:
            _agent_dir += "_no_det_answer"
        # Deterministic lineage-completion is disabled in the clean generic core;
        # do not encode this default safety setting into result-directory names.

        repair_mode = (getattr(args, "repair_mode", None) or
                       getattr(config, "OCA_REPAIR_MODE", "off"))
        config.OCA_REPAIR_MODE = repair_mode
        if getattr(args, "repair_endpoint_hint", None) is not None:
            config.OCA_REPAIR_ENDPOINT_HINT = bool(args.repair_endpoint_hint)
        ep = "_ep" if getattr(config, "OCA_REPAIR_ENDPOINT_HINT", False) else ""
        if repair_mode != "off":
            _agent_dir = f"{_agent_dir}_repair_{repair_mode}{ep}"
        print(f"[INFO] OCA repair={repair_mode}"
              f"{' (+endpoint hint)' if ep else ''}")
        custom_token_settings = (
            str(getattr(config, "OCA_OBSERVATION_MODE", "raw")) != "raw" or
            str(getattr(config, "OCA_PHASE_A_HISTORY", "full")) != "full" or
            str(getattr(config, "OCA_PLANNER_CATALOG_MODE", "full")) != "full" or
            bool(getattr(config, "OCA_PHASE_B_FOCUS_FALLBACK", False)) or
            int(getattr(config, "OCA_LEDGER_MAX_CHARS", 24000)) != 24000)
        if token_profile != "baseline":
            _agent_dir += f"_tok_{token_profile}"
        elif custom_token_settings:
            _agent_dir += "_tok_custom"
        print("[INFO] OCA token profile=" + token_profile +
              f" obs={config.OCA_OBSERVATION_MODE}" +
              f" history={config.OCA_PHASE_A_HISTORY}" +
              f" planner_catalog={config.OCA_PLANNER_CATALOG_MODE}" +
              f" phase_b_focus={'on' if config.OCA_PHASE_B_FOCUS_FALLBACK else 'off'}" +
              f" ledger_chars={config.OCA_LEDGER_MAX_CHARS}")
    else:
        # Ignore OCA-only flags for other agents.
        if getattr(args, "contract_mode", None):
            print("[WARN] --contract-mode applies only to --agent oca")

    # DeepSeek direct runs receive a model namespace automatically. Existing
    # OpenAI/custom direct-run paths retain their historical directory names;
    # supported SECAT runners pass explicit model-aware result tags for every
    # non-default model. This avoids changing legacy path semantics.
    _explicit_result_tag = bool(_generic_result_tag or (args.agent == "oca" and getattr(args, "oca_result_tag", None)))
    if _llm_provider == "deepseek" and not _explicit_result_tag:
        _agent_dir += "_model_" + model_run_slug(args.model)
    _agent_dir += "_isolated" if args.evaluation_profile == "isolated" else "_reported95"
    _base = os.path.join(config.RESULT_DIR, _bench, _agent_dir)
    config.set_output_base(_base)
    if getattr(args, "fresh", False):
        import shutil as _shutil
        # A fresh experiment must remove the complete configuration root,
        # including JSONL evidence graphs and aggregate summaries. Guard the path
        # before deletion so a malformed configuration cannot remove a broad tree.
        _normalized = os.path.normpath(_base)
        _results_root = os.path.normpath(config.RESULT_DIR)
        _relative = os.path.relpath(_normalized, _results_root)
        _parts = [x for x in _relative.split(os.sep) if x not in ("", ".", "..")]
        if not (_normalized.startswith(_results_root + os.sep) and
                len(_parts) == 2 and _parts[-1] == _agent_dir):
            raise RuntimeError(f"refusing unsafe --fresh deletion: {_base}")
        if os.path.isdir(_base):
            _shutil.rmtree(_base)
        config.set_output_base(_base)
        print(f"[INFO] --fresh: recreated clean result root {_base}/")
    print(f"[INFO] outputs -> {_base}/")

    print(f"\n{'='*60}")
    print(f"  SECAT Experiment")
    print(f"  Agent : {args.agent}")
    print(f"  Model : {args.model}")
    print(f"  Provider: {_llm_provider}")
    print(f"  Tasks : {len(tasks)}")
    print(f"{'='*60}\n")

    all_results = []
    interrupted = False
    crash_count = 0

    from utils.evaluation_isolation import sanitize_task_for_agent, assert_isolated_task

    def _trusted_spotify_fixture_reset(label: str):
        """Run destructive benchmark setup outside the agent boundary and verify it.

        Setup writes are trusted infrastructure, not agent behavior.  The fixture
        implementation performs a live read-back verification before returning.
        """
        from utils.spotify_runtime import reset_fixture
        _prior_write_flag = os.environ.get("SECAT_ALLOW_SPOTIFY_WRITES")
        os.environ["SECAT_ALLOW_SPOTIFY_WRITES"] = "YES"
        try:
            report = reset_fixture()
        finally:
            if _prior_write_flag is None:
                os.environ.pop("SECAT_ALLOW_SPOTIFY_WRITES", None)
            else:
                os.environ["SECAT_ALLOW_SPOTIFY_WRITES"] = _prior_write_flag
        _fixture_dir = os.path.join(_base, "fixture")
        os.makedirs(_fixture_dir, exist_ok=True)
        with open(os.path.join(_fixture_dir, f"{label}.json"), "w") as _fh:
            json.dump(report, _fh, indent=2, sort_keys=True)
        verification = dict(report.get("verification") or {})
        seed_cache = dict(report.get("seed_cache") or {})
        print(
            f"[SPOTIFY] fixture reset verified for {label}: "
            f"tracks={len(verification.get('saved_tracks') or [])} "
            f"albums={len(verification.get('saved_albums') or [])} "
            f"playlists={len(verification.get('playlists') or [])} "
            f"playback_track={verification.get('current_playback_track') or '-'} "
            f"playback_album={verification.get('current_playback_album') or '-'} "
            f"seed_cache_hits={seed_cache.get('hits', 0)} "
            f"seed_cache_misses={seed_cache.get('misses', 0)}"
        )
        return report

    # ToolCoder mirrors the supplied runner by resetting again inside each child
    # execution/revision.  Perform one trusted preflight reset here as well so an
    # unavailable playback device or failed library mutation aborts before any LLM
    # tokens are spent; the in-child reset remains for upstream behavioral parity.
    if _runtime_api == "spotify" and args.spotify_reset_fixture and args.agent == "toolcoder":
        _trusted_spotify_fixture_reset("toolcoder_preflight")

    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] Running task: {task.get('id')}")
        try:
            # Fixture calls are setup, not agent behavior. Clear the prior trace
            # before reset so setup traffic cannot contaminate either the previous
            # or current task's method/path score.
            if _runtime_api == "spotify":
                os.environ.pop("SECAT_SPOTIFY_TRACE_FILE", None)
            if (_runtime_api == "spotify" and args.spotify_reset_fixture
                    and args.agent != "toolcoder"):
                _trusted_spotify_fixture_reset(str(task.get("id") or f"task_{i+1}"))
            if _runtime_api == "spotify":
                _trace_dir = os.path.join(_base, "traces")
                os.makedirs(_trace_dir, exist_ok=True)
                _trace_path = os.path.join(_trace_dir, f"{task.get('id')}.jsonl")
                if os.path.exists(_trace_path):
                    os.remove(_trace_path)
                os.environ["SECAT_SPOTIFY_TRACE_FILE"] = _trace_path
            agent_task, boundary = sanitize_task_for_agent(
                task, profile=args.evaluation_profile, ordinal=i + 1,
                runtime_benchmark=args.benchmark or os.environ.get("SECAT_BENCHMARK"))
            if args.evaluation_profile == "isolated":
                assert_isolated_task(agent_task)
            _run_files_before = _snapshot_run_files(config.RUN_DIR)
            _graph_lines_before = _graph_line_count(config.RESULTS_BASE)
            summary = agent.run(task=agent_task, model=args.model)
            # Persist non-secret provider configuration so model-family comparisons
            # remain auditable even if environment defaults change later.
            summary["llm_provider"] = _llm_provider
            summary["model"] = str(args.model)
            summary["model_runtime"] = dict(_model_runtime)
            # First make the trajectory/evidence graph durably identifiable by the
            # authoritative external task id.  If anything after this point fails,
            # resume bookkeeping never has to infer a task number from an opaque
            # isolated runtime id.
            _saved_run = _restore_post_run_identity(
                before_files=_run_files_before,
                graph_lines_before=_graph_lines_before,
                external_task=task, boundary=boundary, summary=summary,
                agent_name=args.agent, benchmark=args.benchmark or task.get("benchmark"),
                evaluation_profile=args.evaluation_profile,
                expect_evidence_graph=(args.agent == "oca" and
                                       str(getattr(config, "OCA_VERSION", "v2")).lower() == "v2"))
            if _runtime_api == "spotify":
                # Gold-path scoring is strictly post-execution and labelled as an
                # oracle method/path/status metric, not semantic task accuracy.
                # A scoring exception marks this trajectory resumable; it must not
                # masquerade as a fully completed comparable task.
                from utils.spotify_runtime import evaluate_trace
                try:
                    summary["spotify_oracle_path_evaluation"] = evaluate_trace(task)
                    summary["spotify_oracle_path_score"] = summary[
                        "spotify_oracle_path_evaluation"]["score"]
                except Exception as _post_exc:
                    summary["post_run_incomplete"] = True
                    summary["post_run_error_type"] = type(_post_exc).__name__
                    summary["post_run_error"] = str(_post_exc)
                    _rewrite_saved_summary(_saved_run, summary)
                    raise
                _rewrite_saved_summary(_saved_run, summary)
            all_results.append(summary)
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Saving partial results...")
            interrupted = True
            break
        except Exception as e:
            _fixture_rate_limit_abort = bool(
                _runtime_api == "spotify" and args.spotify_reset_fixture
                and _is_spotify_fixture_rate_limit(e))
            crash_count += 1
            import traceback as _tb
            _crash_trace = _tb.format_exc()
            print(f"[ERROR] Task {task.get('id')} crashed: {e}")
            print(_crash_trace, end="" if _crash_trace.endswith("\n") else "\n")
            crash_summary = {
                "task_id":                    task.get("id"),
                "total_turns":                0,
                "num_errors":                 1,
                "error_types":                ["crash"],
                "scopes_hit":                 ["Other_Unknown"],
                "log_anomaly_types":          [],
                "first_error_turn":           None,
                "turns_after_first_error":    0,
                "final_answer":               None,
                "success":                    False,
                "silent_failure":             False,
                "code_runs":                  0,
                "answered_without_tool_use":  False,
                "crash_message":               str(e),
                "crash_type":                  type(e).__name__,
                "crash_trace":                 _crash_trace[-12000:],
                "llm_provider":                _llm_provider,
                "model":                       str(args.model),
                "model_runtime":               dict(_model_runtime),
            }
            # Save a minimal run file so no task is silently lost from disk
            try:
                import config as _cfg, json as _json
                os.makedirs(_cfg.RUN_DIR, exist_ok=True)
                from datetime import datetime as _dt
                _ts  = _dt.now().strftime("%Y%m%d_%H%M%S")
                _tid = task.get("id", "unknown")
                _fname = f"{args.agent}_{_tid}_{_ts}_CRASH.json"
                _path  = os.path.join(_cfg.RUN_DIR, _fname)
                from utils.logger import _write_json_durable
                _path = _write_json_durable(_path, {
                    "meta":    {"agent": args.agent, "task_id": _tid,
                                "instruction": task.get("instruction",""),
                                "timestamp": _ts},
                    "turns":   [],
                    "summary": crash_summary,
                    "crash":   _crash_trace,
                })
                print(f"  [LOG] Crash saved -> {_path}")
            except Exception as _se:
                print(f"  [WARN] Could not save crash file: {_se}")
            all_results.append(crash_summary)
            if _fixture_rate_limit_abort:
                print("[SPOTIFY] Aborting remaining tasks after fixture rate-limit failure; "
                      "no additional reset requests will be sent.")
                break
            if getattr(args, "fail_fast_crash", False):
                print("[RUNNER] fail-fast enabled; preserving completed task files and stopping after crash.")
                break

    # Save summary and a machine-readable completion status. Partial/crashed
    # experiments must not look successful to orchestration scripts.
    from utils.logger import save_experiment_summary
    save_experiment_summary(args.agent, all_results)
    status = {
        "requested_tasks": len(tasks),
        "completed_results": len(all_results),
        "crash_count": crash_count,
        "interrupted": interrupted,
        "complete": (not interrupted and crash_count == 0 and len(all_results) == len(tasks)),
        "evaluation_profile": args.evaluation_profile,
    }
    os.makedirs(config.SUMMARY_DIR, exist_ok=True)
    from utils.logger import _write_json_durable
    _write_json_durable(os.path.join(config.SUMMARY_DIR, "run_status.json"), status)
    if not status["complete"]:
        print("\n[FAILED] Experiment is incomplete or contains crashed tasks: " +
              json.dumps(status, sort_keys=True))
        raise SystemExit(2)
    print("\n[DONE] Experiment complete and status-checked.")

if __name__ == "__main__":
    main()
