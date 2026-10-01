"""
codeact_agent.py
CodeAct-style agent — faithful to the original paper:

1. LLM generates Python code as action
2. Code executes in a PERSISTENT interpreter (variables survive across turns)
3. Full output fed back to LLM as observation
4. LLM decides: emit next code action OR give final answer
5. Repeat up to MAX_TURNS (paper uses 10)

Key fidelity points vs original CodeAct:
- Persistent namespace: turn N variables available in turn N+1
- LLM sees FULL output, not truncated (CodeAct uses full context)
- No forced structure — LLM decides when to stop
- Temperature 0 for reproducibility
"""

import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
import config
from utils.executor         import PersistentInterpreter, extract_code_blocks
from utils.error_classifier import classify_error, detect_silent_failure
from utils.logger           import TaskLogger
from utils.log_injector     import inject_logs, strip_log_lines, keep_log_lines
from utils.log_analyzer     import analyze_logs, summarize_findings

from agents.common import make_client
client = make_client()

# ── System prompt — faithful to CodeAct paper style ──────────────────────────
# CodeAct paper: "a Python interpreter is integrated so that code actions
# can be executed and the output is returned as observations"
# ── System prompt — faithful to CodeAct paper Appendix E ─────────────────────
# Source: paper Appendix E (zero-shot system prompt, chatML format)
# Tool definitions injected per Appendix F
SYSTEM_PROMPT = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

The assistant can interact with an interactive Python (Jupyter Notebook) environment and receive the corresponding output when needed. The code should be enclosed using "<execute>" tag, for example: <execute> print("Hello World!") </execute>.

The assistant should attempt fewer things at a time instead of putting too much code in one <execute> block. The assistant can install packages through PIP by <execute> !pip install [package needed] </execute> and should always import packages and define variables before starting to use them.

The assistant should stop <execute> and provide an answer when they have already obtained the answer from the execution result. Whenever possible, execute the code for the user using <execute> instead of providing it.

The assistant's response should be concise, but do express their thoughts.

You have access to the following tools:
[1] requests: Python requests library for HTTP calls.
[2] os: Python os module. Use os.environ["TMDB_API_KEY"] to get the TMDB API key.

The TMDB (The Movie Database) API base URL is: https://api.themoviedb.org/3

Use the TMDB API (via <execute> Python code) to obtain the answer; do not rely on prior knowledge. Base your final answer on the printed execution result.

When you are done, output the result using 'Answer: your answer'.
"""

def _runtime_domain(name: str) -> str:
    try:
        import benchmarks as _B
        spec = _B.get_benchmark(str(name))
        return str(spec.get("runtime_api") or spec.get("name") or name)
    except Exception:
        return str(name or "")


def build_system_prompt(benchmark: str) -> str:
    """Return the CodeAct system prompt for one explicitly configured API.

    Provider selection is trusted runtime state.  Never silently default to a
    different API: that failure mode previously sent isolated Spotify tasks to
    TMDB and invalidated the run.
    """
    import os as _os
    try:
        import benchmarks as B
        b = B.get_benchmark(benchmark)
        label = b.get("api_label", "the target REST API")
        base = b["base_url"]
        env = b["env_key"]
        auth_note = b.get("auth_note", f'Use os.environ["{env}"] for the credential.')
    except Exception as exc:
        raise ValueError(f"unknown or unconfigured benchmark for CodeAct: {benchmark!r}") from exc
    # Make the auth instruction match the ACTUAL credential present. TMDB has two
    # key types: a v4 read-access token (long, starts 'eyJ') sent as a Bearer header,
    # and a v3 key (32 hex chars) sent as the ?api_key= query param. Telling the model
    # the wrong one causes 401s on every call.
    if env in ("TMDB_API_KEY",):
        cred = _os.environ.get(env, "")
        if cred.startswith("eyJ"):
            auth_note = ('Authenticate by sending the token as a Bearer header: '
                         'headers={"Authorization": f"Bearer {os.environ[\'%s\']}"}.' % env)
        else:
            auth_note = ('Authenticate by passing the key as a query parameter on every '
                         'request: params={"api_key": os.environ["%s"], ...}. '
                         'Do NOT use a Bearer header for this key.' % env)
    spotify_note = ""
    tool_lines = (
        "[1] requests: Python requests library for HTTP calls.\n"
        f'[2] os: Python os module. Use os.environ["{env}"] to get the API credential.\n\n'
    )
    if str(b.get("runtime_api") or benchmark) == "spotify":
        profile = _os.environ.get("SPOTIFY_API_PROFILE", "legacy")
        # In isolated evaluation the trusted kernel bootstrap captures the OAuth
        # token in the requests proxy and then removes it from os.environ before
        # generated code executes.  The prompt must describe that real execution
        # contract; instructing CodeAct to read SPOTIFY_ACCESS_TOKEN causes a
        # deterministic KeyError even though authentication is configured.
        tool_lines = (
            "[1] requests: Python requests-compatible object. Spotify OAuth "
            "authentication is injected by the runtime.\n\n"
        )
        auth_note = (
            "Authentication is handled by the runtime. Do not read environment "
            "variables, add Authorization headers, or print credentials."
        )
        spotify_note = (
            f"\nSpotify benchmark profile: {profile}. Use the endpoint paths from the "
            "original RestBench task; the runtime applies any permitted compatibility "
            "rewrite. For POST/PUT/DELETE, print the HTTP status and response body (if any) "
            "before claiming completion. Writes are blocked unless explicitly enabled.\n"
        )
    head = SYSTEM_PROMPT.split("You have access to the following tools:")[0]
    return (head +
        "You have access to the following tools:\n"
        + tool_lines +
        f"The {label} base URL is: {base}\n"
        f"{auth_note}\n"
        f"{spotify_note}\n"
        f"Use the {label} (via <execute> Python code) to obtain the answer; do not rely on "
        "prior knowledge. Base your final answer on the printed execution result.\n\n"
        "When you are done, output the result using 'Answer: your answer'.\n")


def run(task: dict, model: str = config.DEFAULT_MODEL) -> dict:
    """
    Run CodeAct agent on a single task.
    Returns summary dict.
    """
    instruction  = task.get("instruction", "")
    ground_truth = task.get("ground_truth", None)

    logger = TaskLogger(agent_name="codeact", task=task, model=model)
    from utils.token_meter import reset as _tok_reset, read as _tok_read
    _tok_reset()

    print(f"\n{'='*60}")
    print(f"[TASK] {task.get('id','?')} | {instruction}")
    print(f"{'='*60}")

    # ── One persistent interpreter per task ───────────────────────────
    # This is the key fix: variables survive across turns
    # pick execution backend: script (exec) or kernel (IPython, like original CodeAct)
    _exec_mode = getattr(config, "EXECUTION_MODE", "script")
    if _exec_mode == "kernel":
        from utils.kernel_executor import KernelInterpreter
        interpreter = KernelInterpreter()
        print(f"[INFO] execution mode: kernel (IPython, auto-displays last expression)")
    else:
        interpreter = PersistentInterpreter()

    # Pick the API context from trusted runtime configuration first.  In the
    # isolated evaluation profile the task object intentionally has an opaque
    # eval_* id and no benchmark field, so inferring the provider from the task
    # id would incorrectly fall back to TMDB for Spotify.
    _bench = task.get("benchmark") or os.environ.get("SECAT_BENCHMARK")
    if not _bench:
        raise RuntimeError(
            "CodeAct requires an explicit trusted benchmark (task['benchmark'] or "
            "SECAT_BENCHMARK); refusing to guess an API from an opaque task id")
    _sys_prompt = build_system_prompt(str(_bench))

    # ── Initial messages ──────────────────────────────────────────────
    messages = [
        {"role": "system", "content": _sys_prompt},
        {"role": "user",   "content": instruction},
    ]

    final_answer   = None
    task_success   = None
    silent_failure = False
    code_runs      = 0      # how many code actions actually executed (RQ1 metric)
    premature_answer_attempts = 0   # times the model answered before running any tool code

    for turn in range(config.MAX_TURNS):
        print(f"\n--- Turn {turn + 1} / {config.MAX_TURNS} ---")

        # ── Call LLM (retry transient errors so a blip doesn't fail the task) ──
        llm_output = None
        for _attempt in range(4):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                )
                llm_output = response.choices[0].message.content
                break
            except Exception as e:
                wait = 2 ** _attempt  # 1, 2, 4, 8s
                print(f"[WARN] LLM call failed (attempt {_attempt+1}/4): {e}; retrying in {wait}s")
                import time as _t; _t.sleep(wait)
        if llm_output is None:
            print(f"[ERROR] LLM call failed after retries; ending this task.")
            break

        print(f"[LLM OUTPUT]\n{llm_output[:400]}")

        # ── Extract code ──────────────────────────────────────────────
        code = extract_code_blocks(llm_output)

        if not code or len(code.strip()) < 5:
            # LLM produced no <execute> code this turn.
            print(f"[NO CODE] LLM responded with text only")

            text_answer = None
            # Capture from "Answer:"/"FINAL ANSWER:" to the END of the message, not
            # just the rest of that ONE line. Models put multi-line answers (lists,
            # several dates) after the marker; splitting per-line truncated them and
            # made correct multi-item answers grade as wrong.
            m = re.search(r"(?:FINAL ANSWER:|Answer:)\s*(.*)", llm_output, re.DOTALL)
            if m:
                text_answer = m.group(1).strip()

            # If the model gave a text-only response after running code but used no
            # 'Answer:' prefix, the whole response IS its grounded final answer.
            # (General models answer in prose; CodeAct's literal 'Answer:' convention
            # was a fine-tuning artifact. The answer is still grounded in the
            # observation it just saw, which is what CodeAct requires.)
            if text_answer is None and code_runs > 0:
                stripped = llm_output.strip()
                if stripped:
                    text_answer = stripped

            # RQ1 setup: we are cataloguing errors that occur IN CodeAct's tool-use
            # code, so the agent must actually use tools. If it answers with NO code
            # run yet, push it once to query the API (records the bypass attempt as a
            # Scope-1 finding). If it answers AFTER running code, accept it.
            if text_answer is not None and code_runs > 0:
                final_answer = text_answer
                logger.log_turn(
                    turn_num    = turn + 1,
                    llm_input   = messages.copy(),
                    llm_output  = llm_output,
                    code        = "",
                    exec_result = {"stdout":"","stderr":"","combined":"","exit_code":0,"timed_out":False,"success":True},
                    error_info  = {"error_type":"none","scope":"none","is_error":False,"is_silent":False,"raw_error":None},
                )
                break

            # Premature answer (no code yet) OR prose-only: log a Scope-1 finding and
            # push the agent to use tools. answered_without_tool_use is still recorded
            # at the end if the agent never runs code across all turns.
            premature = text_answer is not None and code_runs == 0
            if premature:
                premature_answer_attempts += 1   # tried to answer before any tool use
            logger.log_turn(
                turn_num    = turn + 1,
                llm_input   = messages.copy(),
                llm_output  = llm_output,
                code        = "",
                exec_result = {"stdout":"","stderr":"","combined":"","exit_code":0,"timed_out":False,"success":True},
                error_info  = {
                    "error_type": "answered_without_tool_use" if premature else "prose_instead_of_code",
                    "scope": "S1_Intention",
                    "is_error": True, "is_silent": False, "raw_error": None},
            )
            messages.append({"role": "assistant", "content": llm_output})
            if premature:
                messages.append({"role": "user", "content":
                    f"Use the {_bench} API with Python code in <execute>...</execute> and base your "
                    "answer on the printed result; do not answer from prior knowledge."})
            else:
                messages.append({"role": "user", "content":
                    "Please write Python code in <execute>...</execute> blocks to solve this."})
            continue

        print(f"[CODE]\n{code[:400]}")

        # ── Inject scope logs before execution ────────────────────────
        instrumented_code = inject_logs(code, client=client, model=model)

        # ── Execute in persistent interpreter ─────────────────────────
        exec_result = interpreter.execute(instrumented_code,
                                          timeout=config.CODE_TIMEOUT_SEC)
        output      = exec_result["combined"]
        code_runs  += 1

        # ── Separate log lines from agent output ─────────────────────
        # full_output    → raw instrumented execution (logs + agent prints)
        # agent_output   → fed back to agent (its own print() only, no logs)
        # logged_output  → injected [Sx] lines only (display/analysis ONLY)
        full_output   = output
        agent_output  = strip_log_lines(output)
        logged_output = keep_log_lines(output)

        print(f"[OUTPUT (agent sees)]\n{agent_output[:500]}")

        # ── Analyze logs from full output ─────────────────────────────
        log_findings = analyze_logs(full_output)
        if log_findings["anomalies"]:
            print(f"[LOG ANOMALIES]\n{summarize_findings(log_findings)}")

        # ── Classify error using agent output ─────────────────────────
        error_info = classify_error(code, agent_output)
        if error_info["is_error"]:
            print(f"[ERROR] type={error_info['error_type']} scope={error_info['scope']}")

        # ── Log turn ──────────────────────────────────────────────────
        logger.log_turn(
            turn_num          = turn + 1,
            llm_input         = messages.copy(),
            llm_output        = llm_output,
            code              = code,                # 1. original python code
            instrumented_code = instrumented_code,   # 2. code with injected logs
            exec_result       = exec_result,
            error_info        = error_info,
            log_findings      = log_findings,
            agent_output      = agent_output,         # 3. output agent sees
            logged_output     = logged_output,        # 4. injected [Sx] lines
        )

        # ── Check for final answer ───────────────────────────────────
        # Faithful CodeAct (Appendix E): accept an answer only when it was
        # "obtained from the execution result." An 'Answer:' co-emitted in the
        # SAME turn as the <execute> block was written BEFORE the result existed
        # (a memorized guess), so it is NOT treated as final — we run the code,
        # feed the result back, and let the model answer on a subsequent turn.
        # Therefore we look for the answer only in the code's printed output.
        m2 = re.search(r"(?:FINAL ANSWER:|Answer:)\s*(.*)", agent_output, re.DOTALL)
        if m2:
            final_answer = m2.group(1).strip()

        if final_answer is not None:
            print(f"[FINAL ANSWER] {final_answer}")
            break

        # ── Feed agent output back as observation ────────────────────
        # CodeAct paper: execution output returned as observation.
        # We feed agent_output only — our [Sx] log lines are stripped.
        # The agent sees only what its own print() statements produced.
        messages.append({"role": "assistant", "content": llm_output})
        messages.append({"role": "user", "content": agent_output})

    # ── Finalize: determine success in ONE place ──────────────────────
    # Faithful to CodeAct: whatever the model emitted as the answer is taken.
    # success is True/False only when we can verify against ground truth;
    # otherwise it is None = "unverified" (NOT auto-marked correct).
    answered_without_tool_use = (code_runs == 0)

    # Safety net: if the loop ended (e.g. hit MAX_TURNS) without capturing a
    # final answer, but the agent DID run code and its last message was prose
    # reasoning over the results, recover that as the answer. This prevents
    # discarding a correct, grounded answer just because it lacked the literal
    # 'Answer:' token. Only applies when code actually ran.
    if (final_answer is None or not str(final_answer).strip()) and code_runs > 0:
        for m in reversed(messages):
            if m["role"] == "assistant" and m["content"] and m["content"].strip():
                # skip messages that are ONLY a code block (no prose answer)
                without_code = extract_code_blocks(m["content"])
                prose = m["content"].replace(f"<execute>{without_code}</execute>", "").strip() if without_code else m["content"].strip()
                if prose and len(prose) > 3:
                    final_answer = prose
                    print(f"[RECOVERED ANSWER from last reasoning] {prose[:120]}")
                    break

    if final_answer is None or final_answer.strip() == "":
        # Agent never produced an answer.
        print("[FAILED] No final answer produced.")
        task_success = False
        final_answer = None
    elif ground_truth:
        # We can verify — check for silent failure (wrong answer, no error).
        silent_failure = detect_silent_failure(final_answer, ground_truth)
        task_success   = not silent_failure
        if silent_failure:
            print(f"[SILENT FAILURE] Expected: {ground_truth}")
    else:
        # Answer exists but no ground truth — correctness is UNVERIFIED.
        task_success = None
        print("[UNVERIFIED] Answer produced; no ground truth to check against.")

    if answered_without_tool_use and final_answer:
        print("[NOTE] Agent answered WITHOUT executing any code (no API query).")

    logger.finalize(
        final_answer   = final_answer,
        success        = task_success,
        silent_failure = silent_failure,
    )
    logger.log["summary"]["code_runs"] = code_runs
    logger.log["summary"]["answered_without_tool_use"] = bool(answered_without_tool_use and final_answer)
    # S1 transient signal: the model tried to answer before using tools, but was pushed
    # and then DID run code and produce a grounded answer. This is the CodeAct analogue
    # of ToolCoder's error-recovery — report it as recovered, not as a terminal failure.
    logger.log["summary"]["premature_answer_attempts"] = premature_answer_attempts
    logger.log["summary"]["recovered_from_premature"] = bool(
        premature_answer_attempts > 0 and code_runs > 0 and final_answer)
    logger.log["summary"]["tokens"] = _tok_read()   # per-task token usage
    if _runtime_domain(_bench) == "spotify":
        logger.log["summary"]["spotify_profile"] = os.environ.get("SPOTIFY_API_PROFILE", "legacy")
        # Gold route metadata is unavailable inside the isolated agent boundary.
        # run_experiment attaches the shared oracle method/path/status evaluation
        # only after the agent returns.
    logger.save()

    # Reset or shutdown interpreter for next task (defensive, never let cleanup
    # crash a task whose result is already computed).
    try:
        if hasattr(interpreter, "shutdown"):
            interpreter.shutdown()      # kernel mode: free the kernel process
        elif hasattr(interpreter, "reset"):
            interpreter.reset()         # script mode: clear the namespace
    except Exception as e:
        print(f"[WARN] interpreter cleanup failed (non-fatal): {e}")

    return logger.log["summary"]
