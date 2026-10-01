"""
common.py — shared plumbing for all CAT agents (CodeAct, ATC, ToolCoder, CodeTool).

The whole point of RQ1 is a *controlled* comparison: every agent must differ ONLY
in its strategy (how it plans / structures / selects code), while the measurement
machinery is identical. This module centralizes that machinery so the four agents
produce byte-compatible trajectory JSON and flow through the same analysis,
annotation, and viewer tools.

What is shared (identical across agents):
  - domain/tool context block (TMDB now; swappable per benchmark later)
  - log injection + output splitting (agent sees clean output; [Sx] logs are
    captured for analysis/display only)
  - log-anomaly analysis + error classification
  - per-turn logging via TaskLogger.log_turn (the 4 artifacts)
  - final-answer extraction
  - success finalization (verified / unverified / failed) + save

What is NOT shared (the independent variable):
  - the control flow / prompting strategy of each agent lives in its own file.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.executor         import PersistentInterpreter, extract_code_blocks
from utils.error_classifier import classify_error, detect_silent_failure
from utils.logger           import TaskLogger
from utils.log_injector     import inject_logs, strip_log_lines, keep_log_lines
from utils.log_analyzer     import analyze_logs, summarize_findings


def make_client():
    """Return the shared metered, provider-routed LLM client.

    Provider choice is made from the *model name on each request*, so CodeAct,
    ToolCoder, and OCA use exactly the same routing boundary.  Existing OpenAI
    calls are unchanged; ``deepseek-*`` models use DeepSeek's OpenAI-compatible
    endpoint and DEEPSEEK_API_KEY.  Keys are resolved lazily at call time so
    importing an agent never requires credentials for an unused provider.
    """
    from utils.token_meter import MeteredClient
    from utils.model_provider import make_routed_client, ProviderResilientClient
    # Resilience is outside the meter so every provider retry is still counted.
    return ProviderResilientClient(MeteredClient(make_routed_client()))


def chat_with_retry(client, *, model, messages, temperature=0.0, attempts=4):
    """Call chat.completions.create, retrying transient errors with backoff so a
    single API blip (rate limit, timeout) doesn't fail an otherwise-good task.
    Returns the message content string, or None if all attempts fail."""
    import time as _t
    for i in range(attempts):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature)
            return resp.choices[0].message.content
        except Exception as e:
            wait = 2 ** i  # 1,2,4,8s
            print(f"[WARN] LLM call failed (attempt {i+1}/{attempts}): {e}; retrying in {wait}s")
            _t.sleep(wait)
    return None


# ── Domain / tool context ─────────────────────────────────────────────────────
# Currently TMDB. To add a benchmark later, branch on task.get("benchmark")
# or task["api_list"] and return that domain's tool description here. Keeping
# it in ONE place means all four agents stay on the same footing per benchmark.
TMDB_TOOL_CONTEXT = """You have access to the following tools:
[1] requests: Python requests library for HTTP calls.
[2] os: Python os module. Use os.environ["TMDB_API_KEY"] to get the TMDB API key.

The TMDB (The Movie Database) API base URL is: https://api.themoviedb.org/3
Always pass api_key as a query parameter. Always print results so you can see them."""

def get_domain_context(task: dict) -> str:
    """Return the tool/API context block for the task's benchmark (TMDB for now).
    Auth instruction matches the actual key type present (v3 query param vs v4 Bearer)."""
    import os as _os
    cred = _os.environ.get("TMDB_API_KEY", "")
    if cred.startswith("eyJ"):
        auth = ('Authenticate by sending the token as a Bearer header: '
                'headers={"Authorization": f"Bearer {os.environ[\'TMDB_API_KEY\']}"}.')
    else:
        auth = 'Always pass api_key as a query parameter (params={"api_key": ...}).'
    return ('You have access to the following tools:\n'
            '[1] requests: Python requests library for HTTP calls.\n'
            '[2] os: Python os module. Use os.environ["TMDB_API_KEY"] to get the TMDB API key.\n\n'
            'The TMDB (The Movie Database) API base URL is: https://api.themoviedb.org/3\n'
            f'{auth} Always print results so you can see them.')


# ── Answer extraction (identical rule for every agent) ────────────────────────
def extract_answer(text: str):
    """Return the final answer string if present, else None.
    Paper format: 'Answer: ...'  (fallback also accepts 'FINAL ANSWER:').
    Captures EVERYTHING after the marker (multi-line), not just the first line —
    list/multi-item answers span several lines and were being truncated."""
    if not text:
        return None
    import re as _re
    m = _re.search(r"(?:FINAL ANSWER:|Answer:)\s*(.*)", text, _re.DOTALL)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return None


# ── Instrument → execute → split → analyze → classify → log (the 4 artifacts) ─
def run_code_action(*, interpreter, code, client, model, logger,
                    turn_num, messages, llm_output):
    """
    Execute ONE code action exactly the way CodeAct does, and log the turn.

    Returns a dict:
      agent_output  : clean output the agent should see (logs stripped)
      logged_output : injected [Sx] lines only (analysis/display)
      exec_result   : raw executor result
      error_info    : classifier output
      log_findings  : log-analyzer output
      instrumented_code : the code that actually ran
      success       : bool (execution succeeded with no classified error)
    """
    instrumented_code = inject_logs(code, client=client, model=model)
    exec_result = interpreter.execute(instrumented_code, timeout=config.CODE_TIMEOUT_SEC)
    full_output = exec_result["combined"]

    agent_output  = strip_log_lines(full_output)
    logged_output = keep_log_lines(full_output)

    log_findings = analyze_logs(full_output)
    error_info   = classify_error(code, agent_output)

    logger.log_turn(
        turn_num          = turn_num,
        llm_input         = messages.copy(),
        llm_output        = llm_output,
        code              = code,
        instrumented_code = instrumented_code,
        exec_result       = exec_result,
        error_info        = error_info,
        log_findings      = log_findings,
        agent_output      = agent_output,
        logged_output     = logged_output,
    )

    return {
        "agent_output":      agent_output,
        "logged_output":     logged_output,
        "exec_result":       exec_result,
        "error_info":        error_info,
        "log_findings":      log_findings,
        "instrumented_code": instrumented_code,
        "success":           exec_result.get("success", False) and not error_info.get("is_error"),
    }


def log_no_code_turn(*, logger, turn_num, messages, llm_output,
                     error_type, scope="S1_Intention", is_error=True):
    """Log a turn where the LLM produced no executable code."""
    logger.log_turn(
        turn_num    = turn_num,
        llm_input   = messages.copy(),
        llm_output  = llm_output,
        code        = "",
        exec_result = {"stdout":"","stderr":"","combined":"","exit_code":0,
                       "timed_out":False,"success":True},
        error_info  = {"error_type": error_type, "scope": scope,
                       "is_error": is_error, "is_silent": False, "raw_error": None},
    )


# ── Success finalization (identical semantics for every agent) ────────────────
def finalize_task(*, logger, interpreter, final_answer, ground_truth, code_runs,
                  extra_summary=None, explicit_abstention=False):
    """
    Determine success in ONE place and save. Returns the summary dict.

    success is True/False only when verifiable against ground truth; otherwise
    None = UNVERIFIED (never auto-marked correct). This matches codeact_agent.
    """
    answered_without_tool_use = (code_runs == 0 and not explicit_abstention)
    silent_failure = False
    task_success   = None

    if final_answer is None or str(final_answer).strip() == "":
        print("[FAILED] No final answer produced.")
        task_success = False
        final_answer = None
    elif ground_truth:
        silent_failure = detect_silent_failure(final_answer, ground_truth)
        task_success   = not silent_failure
        if silent_failure:
            print(f"[SILENT FAILURE] Expected: {ground_truth}")
    else:
        task_success = None
        print("[UNVERIFIED] Answer produced; no ground truth to check against.")

    if answered_without_tool_use and final_answer:
        print("[NOTE] Agent answered WITHOUT executing any code (no API query).")

    logger.finalize(final_answer=final_answer, success=task_success,
                    silent_failure=silent_failure)
    logger.log["summary"]["code_runs"] = code_runs
    logger.log["summary"]["answered_without_tool_use"] = bool(
        answered_without_tool_use and final_answer)
    if extra_summary:
        logger.log["summary"].update(extra_summary)
    logger.save()

    try:
        interpreter.reset()
    except Exception as e:
        print(f"[WARN] interpreter.reset() failed (non-fatal): {e}")

    return logger.log["summary"]
