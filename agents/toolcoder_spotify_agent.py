"""ToolCoder RestBench-Spotify adapter.

This follows the authors' Spotify pipeline (scaffold -> commented plan -> grounded
replan -> assembly -> execution repair) while routing execution through SECAT's
shared logger, model client, credential handling, write guard, and compatibility
profile.  The original prompt templates and reduced Spotify OAS are bundled
unchanged from the supplied ToolCoder project.
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from typing import Any

import config
from agents.common import make_client
from agents.toolcoder_spotify_template import (
    CODE_FUNCTION_PROMPT, PLANNER_TEMPLATE_STEP, REPLAN_TEMPLATE,
    REUSABLE_FUNCTION_TEMPLATE, EXECUTION_FAILURE_TEMPLATE,
)
from utils.error_classifier import classify_error
from utils.logger import TaskLogger

client = make_client()


def _single_run(prompt: str, model: str, retry: int = 3) -> str:
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}]
    last = None
    for _ in range(retry):
        try:
            r = client.chat.completions.create(
                model=model, messages=messages, n=1, temperature=0.0)
            return (r.choices[0].message.content or "").strip()
        except Exception as exc:
            last = exc
    raise RuntimeError(f"ToolCoder Spotify LLM stage failed: {last}")


def _python(text: str) -> str:
    """Parse Python exactly as the supplied ToolCoder Spotify runner does."""
    match = re.search(r"```python(.*?)```", text or "", re.DOTALL)
    if not match:
        raise ValueError("ToolCoder Spotify expected a ```python ... ``` block")
    return match.group(1).strip()


def _request_pairs(text: str) -> list[tuple[str, str]]:
    # Match the official RestBench-Spotify runner.
    return [(m.upper(), p) for m, p in
            re.findall(r"\b(GET|POST|PUT|DELETE)\s+(/[\w/{}/-]+)",
                       text or "", re.I)]


def _tool_desc(oas: list[dict]) -> str:
    lines = []
    for tool in oas:
        for method in ("get", "post", "delete", "put"):
            docs = tool.get(method)
            if docs:
                lines.append(f"- {method.upper()} https://api.spotify.com/v1{tool['path']}: "
                             f"{docs.get('detailed_description') or docs.get('description') or ''}")
    return "\n".join(lines)


def _remove_docstrings(code: str) -> str:
    return re.sub(r"\'\'\'[\s\S]*?\'\'\'", "", code or "")


def _planner(query: str, oas: list[dict], scaffold: str, model: str) -> str:
    """Faithful port of ToolCoder ``run_planner`` for RestBench-Spotify."""
    prompt = PLANNER_TEMPLATE_STEP.format(
        question=query,
        toolbox=_tool_desc(oas),
        pseudo_code_task=scaffold,
    )
    return _python(_single_run(prompt, model))


def _replan(query: str, plan: str, oas: list[dict], model: str) -> str:
    """Mirror the official run_spotify.py run_replan control flow.

    The upstream repository performs one reflection whenever METHOD /path
    comments are present, using the prompt text shipped by the authors.  We keep
    that behavior here rather than silently improving the baseline.
    """
    spotify_oas_paths = [item["path"] for item in oas]
    desc = _tool_desc(oas)
    pattern = r"(?i)Call\s+(\w+)\s+(/[\w/{}/-]+)"
    match = re.findall(pattern, _remove_docstrings(plan))

    if len(match):
        feedback = ("No planning content described by comments was detected in the code. "
                    "Please ensure that the planning results are provided in the form of "
                    "comments instead of implementing specific functions.")
        return _python(_single_run(REPLAN_TEMPLATE.format(
            toolbox=desc, question=query, pseudo_code_solution=plan,
            feedback=feedback), model))

    invalid_urls = []
    for method, url in match:
        if url not in spotify_oas_paths:
            invalid_urls.append(url)
    if invalid_urls:
        feedback = "\n".join(
            f"The `{url}` API is invalid as it is not existed in the provided toolbox. "
            "Please check it and replace it with an appropriate one. "
            for url in invalid_urls)
        return _python(_single_run(REPLAN_TEMPLATE.format(
            toolbox=desc, question=query, pseudo_code_solution=plan,
            feedback=feedback), model))
    return plan


def _api_docs(plan: str, oas: list[dict]) -> list[dict[str, Any]]:
    by_path = {x["path"]: x for x in oas}
    docs = []
    for method, path in _request_pairs(plan):
        item = by_path.get(path, {}).get(method.lower())
        if not item:
            continue
        docs.append({
            "path": path, "method": method.lower(),
            "description": item.get("detailed_description") or item.get("description", ""),
            "parameters": item.get("parameters", []),
            "schema": item.get("schema", {}),
            "requestBody": item.get("requestBody", {}),
            "implementation_example": item.get("implementation_example", ""),
        })
    return docs


def _assemble(query: str, plan: str, oas: list[dict], model: str) -> str:
    prompt = REUSABLE_FUNCTION_TEMPLATE.format(
        question=query, toolbox=json.dumps(_api_docs(plan, oas), ensure_ascii=False),
        pseudo_code_task=plan)
    return _python(_single_run(prompt, model))


def _runtime_code(main_code: str) -> str:
    # The supplied ToolCoder runner prepends init_spotify() to *every* execution,
    # including repair attempts.  When canonical reset mode is requested we mirror
    # that behavior in trusted infrastructure, then restore the agent write policy
    # and clear setup traffic from the scored API trace.
    return (
        "import os, json\n"
        "from utils.spotify_runtime import SpotifyRequestsWrapper, reset_fixture\n"
        "if os.environ.get('SECAT_TOOLCODER_RESET_EACH_EXEC') == 'YES':\n"
        "    _prior_write = os.environ.get('SECAT_ALLOW_SPOTIFY_WRITES')\n"
        "    _trace_file = os.environ.pop('SECAT_SPOTIFY_TRACE_FILE', None)\n"
        "    os.environ['SECAT_ALLOW_SPOTIFY_WRITES'] = 'YES'\n"
        "    try:\n"
        "        reset_fixture()\n"
        "    finally:\n"
        "        if _prior_write is None: os.environ.pop('SECAT_ALLOW_SPOTIFY_WRITES', None)\n"
        "        else: os.environ['SECAT_ALLOW_SPOTIFY_WRITES'] = _prior_write\n"
        "        if _trace_file: os.environ['SECAT_SPOTIFY_TRACE_FILE'] = _trace_file\n"
        "requests_wrapper = SpotifyRequestsWrapper()\n"
        + main_code
    )



def _spotify_env_file_values(project_root: str) -> dict[str, str]:
    """Read only Spotify credential keys from the current project .env.

    ToolCoder executes assembled programs in child Python processes.  A previous
    child may have refreshed/rotated OAuth material and persisted it atomically;
    rereading these keys before each execution prevents later children from
    inheriting stale parent-process credentials during long benchmark runs.
    """
    path = os.path.join(project_root, ".env")
    out: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if key in {"SPOTIFY_ACCESS_TOKEN", "SPOTIFY_REFRESH_TOKEN",
                           "SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET",
                           "SPOTIFY_REDIRECT_URI"}:
                    out[key] = value.strip()
    except OSError:
        pass
    return out

def _execute(main_code: str, timeout: int = 120) -> tuple[str, str, int]:
    env = dict(os.environ)
    env["SECAT_PROJECT_ROOT"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Prefer credentials persisted by trusted refresh infrastructure over stale
    # copies captured in the long-lived parent process.
    env.update(_spotify_env_file_values(env["SECAT_PROJECT_ROOT"]))
    env["SECAT_BENCHMARK"] = "spotify"
    try:
        p = subprocess.run([sys.executable, "-c", _runtime_code(main_code)],
                           capture_output=True, text=True, timeout=timeout,
                           cwd=env["SECAT_PROJECT_ROOT"], env=env)
        return p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired:
        return "", f"TimeoutError: execution exceeded {timeout}s", -1


def _is_spotify_rate_limit_failure(error: str | None) -> bool:
    text = str(error or "")
    return ("SpotifyRateLimitError" in text
            or "QUOTA_EXCEEDED" in text
            or "Spotify remained rate-limited" in text)


def _repair(query: str, code: str, error: str, model: str) -> tuple[str, str, str, int, bool, int]:
    attempts = 0
    output, rc = "", 1
    for _ in range(3):
        attempts += 1
        code = _python(_single_run(EXECUTION_FAILURE_TEMPLATE.format(
            question=query, python_code=code, execution_result=error), model))
        output, error, rc = _execute(code)
        if not error and rc == 0:
            return code, output, error, rc, True, attempts
    return code, output, error, rc, False, attempts


def _answer(output: str) -> str | None:
    if not output or not output.strip(): return None
    m = re.search(r"(?:FINAL ANSWER:|Answer:)\s*(.*)", output, re.S | re.I)
    if m and m.group(1).strip(): return m.group(1).strip()
    lines = [x.strip() for x in output.splitlines() if x.strip()]
    return lines[-1] if lines else None


_REUSABLE_TOOLBOX: list[dict] | None = None


def _toolbox_state_path() -> str:
    return str(os.environ.get("SECAT_TOOLCODER_SPOTIFY_STATE") or "").strip()


def _apply_persisted_toolbox_state(data: list[dict]) -> list[dict]:
    """Restore only learned implementation examples from a prior batch.

    The official Spotify ToolCoder runner carries one mutable toolbox across the
    whole benchmark. Comparable SECAT runs are intentionally split into small
    quota-safe batches, so this tiny state file preserves that same cross-task
    behavior without persisting prompts, credentials, oracle routes, or answers.
    """
    path = _toolbox_state_path()
    if not path or not os.path.isfile(path):
        return data
    try:
        payload = json.load(open(path, encoding="utf-8"))
        if isinstance(payload, dict):
            stored_profile = str(payload.get("profile") or "legacy")
            active_profile = os.environ.get("SPOTIFY_API_PROFILE", "legacy")
            if stored_profile != active_profile:
                raise RuntimeError(
                    f"ToolCoder Spotify toolbox state profile mismatch: stored={stored_profile!r}, active={active_profile!r}")
        examples = payload.get("implementation_examples") if isinstance(payload, dict) else None
        if not isinstance(examples, dict):
            return data
        for item in data:
            pth = str(item.get("path") or "")
            per_method = examples.get(pth)
            if not isinstance(per_method, dict):
                continue
            for method, spec in item.items():
                if not isinstance(spec, dict):
                    continue
                example = per_method.get(str(method).lower())
                if isinstance(example, str) and example.strip():
                    spec["implementation_example"] = example
    except Exception as exc:
        raise RuntimeError(f"invalid persisted ToolCoder Spotify toolbox state {path}: {exc}") from exc
    return data


def _persist_toolbox_state(data: list[dict]) -> None:
    path = _toolbox_state_path()
    if not path:
        return
    examples: dict[str, dict[str, str]] = {}
    for item in data:
        pth = str(item.get("path") or "")
        if not pth:
            continue
        for method, spec in item.items():
            if not isinstance(spec, dict):
                continue
            example = spec.get("implementation_example")
            if isinstance(example, str) and example.strip():
                examples.setdefault(pth, {})[str(method).lower()] = example
    payload = {
        "version": 1,
        "profile": os.environ.get("SPOTIFY_API_PROFILE", "legacy"),
        "implementation_examples": examples,
    }
    target = os.path.abspath(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp = target + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, target)


def _load_oas() -> list[dict]:
    # The official ToolCoder runner keeps one mutable reusable_toolbox across the
    # entire benchmark and injects successful implementation examples into later
    # tasks.  run_experiment executes all tasks in one Python process, so a module
    # global reproduces that run-scoped behavior without leaking across runs.
    global _REUSABLE_TOOLBOX
    if _REUSABLE_TOOLBOX is None:
        import benchmarks as B
        path = B.get_benchmark("spotify")["toolcoder_oas_file"]
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("spotify_oas_toolcoder.json must be a list")
        _REUSABLE_TOOLBOX = _apply_persisted_toolbox_state(data)
    return _REUSABLE_TOOLBOX


def _extract_first_function_body_as_strings(code: str) -> list[str]:
    tree = ast.parse(code)
    statements = []
    lines = code.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            for stmt in node.body:
                start_lineno = stmt.lineno - 1
                end_lineno = getattr(stmt, "end_lineno", stmt.lineno)
                statements.append("\n".join(lines[start_lineno:end_lineno]).strip())
            break
    return statements


def _extract_http_methods_and_urls(code: str) -> list[tuple[str, str]]:
    pattern = re.compile(
        r'requests_wrapper\.(get|post|put|delete)\(\s*[f]?["\'](https?://[^\s"\']+)')
    return pattern.findall(code or "")


def _collect_successful_api_calls(main_code: str, reusable_toolbox: list[dict]) -> list[dict]:
    by_path = {item["path"]: item for item in reusable_toolbox}
    for line in _extract_first_function_body_as_strings(main_code):
        for method, url in _extract_http_methods_and_urls(line):
            path = url.replace("https://api.spotify.com/v1", "")
            if path in by_path and method in by_path[path]:
                by_path[path][method]["implementation_example"] = line.strip()
    return reusable_toolbox


def run(task: dict, model: str = config.DEFAULT_MODEL) -> dict:
    query = task.get("query") or task.get("instruction") or ""
    task = {**task, "instruction": query, "benchmark": "spotify"}
    logger = TaskLogger("toolcoder", task, model)
    from utils.token_meter import reset as token_reset, read as token_read
    token_reset()
    oas = _load_oas()
    turn = 0

    def log(stage, text, code="", execution=None, recovered=False,
            initial_scope=None, initial_type=None):
        nonlocal turn
        turn += 1
        if execution is None:
            err = {"error_type": "none", "scope": "none", "is_error": False,
                   "is_silent": False, "raw_error": None}
            execution = {}
        else:
            err = classify_error(code, execution.get("stderr") or execution.get("stdout") or "")
            if recovered:
                err = {**err, "recovered_from_error": True,
                       "initial_scope": initial_scope,
                       "initial_error_type": initial_type}
        logger.log_turn(turn_num=turn,
                        llm_input=[{"role": "user", "content": f"[Spotify ToolCoder: {stage}]"}],
                        llm_output=text or "", code=code or "",
                        exec_result=execution, error_info=err,
                        agent_output=execution.get("stdout", ""))

    print("\n" + "=" * 60)
    print(f"[ToolCoder/Spotify] {task.get('id')} | {query}")
    print("=" * 60)

    scaffold = _python(_single_run(CODE_FUNCTION_PROMPT.format(question=query), model))
    log("scaffold", scaffold); print("[1/5] scaffold")
    plan = _planner(query, oas, scaffold, model)
    log("plan", plan); print(f"[2/5] plan: {_request_pairs(plan)}")
    plan = _replan(query, plan, oas, model)
    log("grounded_plan", plan); print(f"[3/5] grounded: {_request_pairs(plan)}")
    main_code = _assemble(query, plan, oas, model)
    log("assemble", main_code); print("[4/5] assembled")

    output, error, rc = _execute(main_code)
    initial_error = error if error or rc else None
    if _is_spotify_rate_limit_failure(initial_error):
        from utils.spotify_runtime import SpotifyRateLimitError
        raise SpotifyRateLimitError(
            "Spotify development-mode quota is exhausted/rate-limited during ToolCoder execution; "
            "aborting the batch before additional tasks",
            reason="QUOTA_EXCEEDED" if "QUOTA_EXCEEDED" in str(initial_error) else "RATE_LIMIT",
            quota_exceeded="QUOTA_EXCEEDED" in str(initial_error))
    recovered = False; initial_scope = initial_type = None
    repair_runs = 0
    if initial_error and not _is_spotify_rate_limit_failure(initial_error):
        info = classify_error(main_code, initial_error)
        initial_scope, initial_type = info.get("scope"), info.get("error_type")
        main_code, output, error, rc, recovered, repair_runs = _repair(
            query, main_code, initial_error, model)
    execution = {"stdout": output or "", "stderr": error or "",
                 "combined": (output or "") + (error or ""),
                 "exit_code": rc, "timed_out": rc == -1,
                 "success": rc == 0 and not error}
    log("execute", main_code, code=main_code, execution=execution,
        recovered=recovered, initial_scope=initial_scope, initial_type=initial_type)
    if execution["success"] and (output or "").strip():
        _exec_label = "observed"
    elif execution["success"]:
        _exec_label = "process-ok / no observation"
    else:
        _exec_label = "failed"
    print(f"[5/5] executed: {_exec_label}")

    # Stage learned examples now, but do not commit cross-task ToolCoder state
    # until this task's run file has been durably saved.  This prevents a
    # KeyboardInterrupt/crash between execution and logging from teaching later
    # tasks from an incomplete experiment.
    learned_toolbox = None
    if execution["success"] and "Failed" not in (output or ""):
        learned_toolbox = _collect_successful_api_calls(main_code, oas)

    final_answer = _answer(output)
    # RestBench-Spotify has dynamic account/API state and no stable semantic gold
    # answer in this harness.  Do not inflate the generic semantic-success field;
    # the shared post-run method/path/status oracle is the comparable metric.
    success = None if final_answer and execution["success"] else False
    logger.log["summary"].update({
        "recovered_from_error": recovered,
        "initial_error_scope": initial_scope if recovered else None,
        "initial_error_type": initial_type if recovered else None,
        "code_runs": 1 + repair_runs,
        "spotify_profile": task.get("spotify_profile") or os.environ.get("SPOTIFY_API_PROFILE", "legacy"),
        "planned_routes": [f"{m} {p}" for m, p in _request_pairs(plan)],
        "tokens": token_read(),
    })
    logger.finalize(final_answer, success=success, silent_failure=False)
    # Gold route metadata is intentionally unavailable inside the isolated agent
    # boundary. Shared Spotify route scoring is attached by run_experiment only
    # after the agent returns.
    logger.save()
    if learned_toolbox is not None:
        global _REUSABLE_TOOLBOX
        _REUSABLE_TOOLBOX = learned_toolbox
        _persist_toolbox_state(_REUSABLE_TOOLBOX)
    return logger.log["summary"]
