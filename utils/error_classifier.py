"""
error_classifier.py — classifies a CodeAct turn's failure.

SCOPE MODEL (conceptual: a code-as-action step is a perception-action loop).
Every turn does: Interpret task -> Express as code -> Execute -> Perceive result
-> Integrate into answer. A failure is a break at one of these five boundaries.
These scopes are defined at the level of the PARADIGM (not API-specific), so
they generalize across agents (ATC/ToolCoder/CodeTool) and to real-world agents.

  S1_INTENTION    — wrong/absent goal BEFORE code. The agent misreads the task,
                    or never forms the intent to use a tool (answers from memory,
                    writes prose). Upstream of any code.
  S2_EXPRESSION   — intent is right, but the code does not faithfully encode it:
                    wrong endpoint, wrong args/params, wrong tool, wrong logic.
                    (The gap between what the agent meant and what it wrote.)
  S3_EXECUTION    — the code is what the agent meant, but fails AT RUNTIME:
                    exceptions, missing libs, timeouts, crashes.
                    (Gap between code-as-written and code-as-run.)
  S4_PERCEPTION   — code ran and produced a result, but the agent fails to
                    apprehend it: never printed it (bare expression), empty view.
                    *** CodeAct-EXCLUSIVE: JSON agents get results structurally;
                    CodeAct agents must construct their own perception. ***
  S5_INTEGRATION  — the agent had the result available but folds it into the
                    answer wrongly: fabrication, wrong field, stale memory
                    overriding fresh data. Where SILENT FAILURES live.

Orthogonal to scope is the TRAJECTORY AXIS (a multi-turn dynamic, not a single
-step break): progress / blind_retry / oscillation / premature_stop / give_up.
A failure has a coordinate on BOTH axes (see classify_trajectory below).
"""
import re

# Canonical scope constants (use these everywhere).
S1_INTENTION   = "S1_Intention"
S2_EXPRESSION  = "S2_Expression"
S3_EXECUTION   = "S3_Execution"
S4_PERCEPTION  = "S4_Perception"
S5_INTEGRATION = "S5_Integration"
S_NONE         = "none"
S_OTHER        = "Other"

def classify_error(code: str, output: str) -> dict:
    """
    Given the code that ran and the output it produced,
    return a classification dict (scope = one of the 5 perception-action scopes).
    """
    result = {
        "error_type": "none",
        "scope":      S_NONE,
        "is_error":   False,
        "is_silent":  False,
        "error_line": None,
        "raw_error":  None,
    }

    # Guard against None
    code   = code   or ""
    output = output or ""
    output_lower = output.lower()

    # ── 1. No code generated → INTENTION (never formed intent to act) ────
    if len(code.strip()) == 0:
        result.update({
            "error_type": "no_code_generated",
            "scope":      S1_INTENTION,
            "is_error":   True,
            "raw_error":  "LLM produced no executable code"
        })
        return result

    # ── 2. Code ran but agent saw nothing → PERCEPTION failure ───────────
    # The result may exist (wrapper captures it) but the agent never surfaced it.
    if len(output.strip()) == 0:
        result.update({
            "error_type": "unobserved_result",
            "scope":      S4_PERCEPTION,
            "is_error":   True,
            "is_silent":  True,
            "raw_error":  "Code executed but agent printed nothing — result not perceived"
        })
        return result

    # Trusted OCA plan-execution lock: this is an expression/planning violation,
    # not an HTTP transport failure. The request is blocked before network I/O.
    if "OCA_UNPLANNED_API_CALL:" in output:
        result.update({
            "error_type": "unplanned_api_call",
            "scope": S2_EXPRESSION,
            "is_error": True,
            "raw_error": "Generated code attempted an API call outside the validated plan",
        })
        return result

    if "OCA_ISOLATION_VIOLATION:" in output:
        result.update({
            "error_type": "isolation_violation",
            "scope": S2_EXPRESSION,
            "is_error": True,
            "raw_error": output.strip()[-1200:],
        })
        return result

    if "OCA_DEPENDENCY_NOT_READY:" in output:
        return {"error_type": "dependency_not_ready", "scope": "S2_Expression",
                "is_error": True, "is_silent": False,
                "raw_error": output.strip()[-1200:]}
    if "OCA_IDEMPOTENT_RETRY_MISMATCH:" in output:
        return {"error_type": "idempotent_retry_mismatch", "scope": "S2_Expression",
                "is_error": True, "is_silent": False,
                "raw_error": output.strip()[-1200:]}
    if "OCA_REQUEST_CONTRACT:" in output:
        return {"error_type": "request_contract", "scope": "S2_Expression",
                "is_error": True, "is_silent": False,
                "raw_error": output.strip()[-1200:]}
    if "OCA_UNAUTHORIZED_REQUEST_INSTANCE:" in output:
        result.update({
            "error_type": "unauthorized_request_instance",
            "scope": S2_EXPRESSION,
            "is_error": True,
            "raw_error": "Generated code used request path/query/body values not authorized by the validated plan",
        })
        return result

    if "OCA_UNAUTHORIZED_BINDING:" in output:
        result.update({
            "error_type": "unauthorized_binding",
            "scope": S2_EXPRESSION,
            "is_error": True,
            "raw_error": "Generated code used a concrete path binding not authorized by its declared producer",
        })
        return result

    if "OCA_PLAN_CALL_BUDGET_EXCEEDED:" in output:
        result.update({
            "error_type": "plan_call_budget",
            "scope": S2_EXPRESSION,
            "is_error": True,
            "raw_error": "Generated code exceeded the validated plan-step call budget",
        })
        return result

    if "OCA_TASK_CALL_BUDGET_EXCEEDED:" in output:
        result.update({
            "error_type": "task_call_budget",
            "scope": S2_EXPRESSION,
            "is_error": True,
            "raw_error": "Generated code exceeded the task API-call budget",
        })
        return result

    if "OCA_UNSUPPORTED_TRANSPORT:" in output:
        result.update({
            "error_type": "unsupported_transport",
            "scope": S2_EXPRESSION,
            "is_error": True,
            "raw_error": "Generated code attempted to bypass the trusted HTTP transport",
        })
        return result

    # ── 3. Python exceptions ─────────────────────────────────────────
    # Conceptual split:
    #  - Runtime crashes on data the code mishandled  → EXECUTION (ran, crashed)
    #  - Referencing things that don't exist / wrong env → EXPRESSION (mis-encoded)
    exception_patterns = [
        (r"KeyError:\s*(.+)",            "KeyError",        S3_EXECUTION),
        (r"IndexError:\s*(.+)",          "IndexError",      S3_EXECUTION),
        (r"AttributeError:\s*(.+)",      "AttributeError",  S3_EXECUTION),
        (r"JSONDecodeError:\s*(.+)",     "JSONDecodeError", S3_EXECUTION),
        (r"TypeError:\s*(.+)",           "TypeError",       S3_EXECUTION),
        (r"ValueError:\s*(.+)",          "ValueError",      S3_EXECUTION),
        (r"NameError:\s*(.+)",           "NameError",       S2_EXPRESSION),
        (r"ImportError:\s*(.+)",         "ImportError",     S2_EXPRESSION),
        (r"ModuleNotFoundError:\s*(.+)", "ImportError",     S2_EXPRESSION),
        (r"TimeoutError",                "Timeout",         S3_EXECUTION),
        (r"ConnectionError:\s*(.+)",     "ConnectionError", S3_EXECUTION),
        (r"RecursionError",              "RecursionError",  S3_EXECUTION),
    ]

    for pattern, err_type, scope in exception_patterns:
        match = re.search(pattern, output)
        if match:
            result["error_type"] = err_type
            result["scope"]      = scope
            result["is_error"]   = True
            result["raw_error"]  = match.group(0).strip()
            return result

    # ── 4. HTTP errors ────────────────────────────────────────────────
    # 404 / bad request = wrong endpoint or params  → EXPRESSION (mis-encoded call)
    # auth / connection  = environment/runtime issue → EXECUTION
    is_real_http_error = (
        "oca_http_transport_failure" in output_lower or
        "httperror"            in output_lower or
        "requests.exceptions"  in output_lower or
        "raise_for_status"     in output_lower or
        "response [4"          in output_lower or
        "404 not found"        in output_lower or
        "401 unauthorized"     in output_lower or
        "403 forbidden"        in output_lower or
        "400 bad request"      in output_lower
    )

    if is_real_http_error:
        if "404" in output or "not found" in output_lower:
            result.update({"error_type": "HTTP_404", "scope": S2_EXPRESSION,
                            "is_error": True, "raw_error": "HTTP 404 — wrong endpoint/resource"})
        elif "401" in output or "403" in output or \
             "unauthorized" in output_lower or "forbidden" in output_lower:
            result.update({"error_type": "HTTP_Auth", "scope": S3_EXECUTION,
                            "is_error": True, "raw_error": "HTTP Auth Error"})
        elif "400" in output or "bad request" in output_lower:
            result.update({"error_type": "HTTP_400", "scope": S2_EXPRESSION,
                            "is_error": True, "raw_error": "HTTP 400 — malformed request"})
        else:
            result.update({"error_type": "HTTP_Error", "scope": S3_EXECUTION,
                            "is_error": True, "raw_error": "HTTP Error"})
        return result

    # ── 5. API authentication error envelopes ───────────────────────
    # Some APIs report auth failures as JSON bodies even when the surrounding
    # execution output does not contain an HTTP exception string. Detect common
    # semantic messages without depending on a provider-specific envelope.
    if ("invalid api key" in output_lower or "authentication failed" in output_lower or
            "invalid access token" in output_lower or "token expired" in output_lower):
        result.update({"error_type": "HTTP_Auth", "scope": S3_EXECUTION,
                       "is_error": True, "raw_error": "API authentication error"})
        return result

    # ── 6. Syntax error / prose instead of code → INTENTION ──────────
    # The agent didn't form a valid coded action (wrote prose / malformed).
    if ("syntaxerror" in output_lower or "invalid syntax" in output_lower or
        "unterminated string literal" in output_lower) and \
        not "traceback (most recent call last)" in output_lower:
        result.update({
            "error_type": "prose_instead_of_code",
            "scope":      S1_INTENTION,
            "is_error":   True,
            "raw_error":  "LLM wrote natural language instead of Python code"
        })
        return result

    if "traceback (most recent call last)" in output_lower:
        lines = output.strip().splitlines()
        last_line = ""
        for line in reversed(lines):
            if line.strip():
                last_line = line.strip()
                break

        if ("syntaxerror" in output_lower or "invalid syntax" in output_lower or
            "unterminated string literal" in output_lower or
            "unterminated string" in output_lower):
            result.update({
                "error_type": "prose_instead_of_code",
                "scope":      S1_INTENTION,
                "is_error":   True,
                "raw_error":  "LLM wrote natural language instead of Python code"
            })
            return result

        # Unrecognized runtime crash → EXECUTION (it ran and threw)
        result.update({"error_type": "unknown", "scope": S3_EXECUTION,
                        "is_error": True, "raw_error": last_line})
        return result

    # ── 7. No error detected ──────────────────────────────────────────
    result["error_type"] = "none"
    result["scope"]      = S_NONE
    result["is_error"]   = False
    return result


def detect_silent_failure(final_answer: str, ground_truth: str) -> bool:
    """
    Returns True if agent reported an answer but it does not match ground truth.
    This is the most dangerous failure — no exception, wrong answer (S5_Integration).
    """
    if not ground_truth or not final_answer:
        return False
    return ground_truth.lower().strip() not in final_answer.lower().strip()


def detect_grounding_failure(final_answer: str, logged_outputs: list) -> dict:
    """
    S5_INTEGRATION detector that needs NO ground truth.
    Checks whether the concrete tokens in the final answer can be traced to any
    captured tool response ([S3] logged output). Catches fabrication structurally.

    Returns {is_grounded, unanchored_tokens, scope}. 'is_grounded' False = the
    answer asserts values that appear in NO tool response (fabrication candidate).
    """
    out = {"is_grounded": True, "unanchored_tokens": [], "scope": S_NONE}
    if not final_answer:
        return out
    corpus = " ".join(logged_outputs or [])
    # Extract 'concrete' tokens worth checking: numbers (incl. single digit) and
    # capitalized names. Numbers are the strongest fabrication signal (counts,
    # years, ids), so check all of them.
    nums  = re.findall(r"\b\d+\b", final_answer)
    names = re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", final_answer)
    # Names that are generic filler shouldn't count as checkable claims.
    _STOP = {"answer", "based", "http", "none", "the", "this", "that",
             "results", "result", "data", "json", "here", "response", "value"}
    checkable_names = [t for t in names if t.lower() not in _STOP]
    checkable = nums + checkable_names
    unanchored = [t for t in checkable if t not in corpus]
    # Flag if there are checkable tokens and any NUMBER is unanchored (numbers are
    # the high-signal case), or a majority of all tokens are unanchored.
    num_unanchored = [t for t in nums if t not in corpus]
    if checkable and (num_unanchored or len(unanchored) >= max(1, len(checkable) // 2)):
        out["is_grounded"] = False
        out["unanchored_tokens"] = unanchored
        out["scope"] = S5_INTEGRATION
    return out


def classify_trajectory(turns: list) -> dict:
    """
    TRAJECTORY AXIS (orthogonal to per-step scope). Looks at the SEQUENCE of a
    task's turns and labels the multi-turn dynamic:
      progress       — calls advance (different endpoints over turns)
      blind_retry    — same failing action repeated back-to-back
      oscillation    — repeats an earlier action after moving on
      premature_stop — answered while the tool chain was clearly incomplete
      give_up        — ran out of turns / ended with no answer
    Returns {dynamic, distinct_endpoints, repeated_calls}.
    """
    endpoints = []
    for t in turns:
        la = t.get("log_analysis", {}) or {}
        for c in (la.get("s2_calls") or []):
            ep = c.get("endpoint", "")
            if ep:
                endpoints.append(ep)
    distinct = list(dict.fromkeys(endpoints))
    repeated = len(endpoints) - len(distinct)

    summary = turns[-1] if turns else {}
    dynamic = "progress"
    if endpoints and repeated >= len(distinct):
        # as many repeats as distinct calls → stuck repeating
        # distinguish adjacent (blind_retry) vs non-adjacent (oscillation)
        adjacent = any(endpoints[i] == endpoints[i+1] for i in range(len(endpoints)-1))
        dynamic = "blind_retry" if adjacent else "oscillation"
    return {
        "dynamic": dynamic,
        "distinct_endpoints": len(distinct),
        "repeated_calls": repeated,
    }


def extract_error_line(traceback_text: str) -> str:
    """Pull the most relevant line from a Python traceback."""
    lines = traceback_text.strip().splitlines()
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith("Traceback") and \
           not line.startswith("File") and not line.startswith("During"):
            return line
    return lines[-1] if lines else ""
