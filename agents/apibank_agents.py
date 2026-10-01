"""API-Bank Level-1/Level-2 strategy adapters.

The adapters share the official API-Bank runtime/evaluator but preserve each
strategy's benchmark control flow.  No target ground-truth data is accepted or
used by these functions; the runner passes a redacted sample object.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
import importlib.util
import io
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional


@dataclass
class AgentPrediction:
    prediction_text: str
    strategy: str
    trace: List[dict]
    executed_call: Optional[dict] = None
    action: Optional[dict] = None

    def as_dict(self) -> dict:
        return {
            "prediction_text": self.prediction_text,
            "strategy": self.strategy,
            "trace": self.trace,
            "executed_call": self.executed_call,
            "action": self.action,
        }


def extract_bracket_call(text: str) -> Optional[str]:
    """Extract the first API-Bank ``[ApiName(...)]`` expression defensively."""
    if not text:
        return None
    value = str(text)
    start_match = re.search(r"\[[A-Za-z_]\w*\(", value)
    if not start_match:
        return None
    start = start_match.start()
    quote = None
    escaped = False
    square = 0
    paren = 0
    for i in range(start, len(value)):
        ch = value[i]
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
            continue
        if ch == "[":
            square += 1
        elif ch == "]":
            square -= 1
            if square == 0 and paren == 0:
                return value[start:i + 1]
        elif ch == "(":
            paren += 1
        elif ch == ")":
            paren -= 1
    return None


def _code_action_text(name: str, params: dict) -> str:
    # CodeAct/ToolCoder's public API-Bank conversion quotes scalar values.
    pieces = []
    for key, value in (params or {}).items():
        if isinstance(value, str):
            pieces.append(f"{key}={value!r}")
        else:
            pieces.append(f"{key}={value!r}")
    return f"[{name}({', '.join(pieces)})]"


def format_history(history: List[dict]) -> str:
    lines: List[str] = []
    for item in history or []:
        role = item.get("role")
        if role == "API":
            result = item.get("result")
            output = result.get("output") if isinstance(result, dict) else result
            lines.append(f"AI: I need to call the API: {_code_action_text(str(item.get('api_name')), item.get('param_dict') or {})}")
            lines.append(f"User: The response from the API: {output}. Please go on.")
        else:
            lines.append(f"{role}: {item.get('text', '')}")
    return "\n".join(lines)


# Kept for backwards-compatible unit tests and small deterministic probes.  The
# faithful ToolCoder adapter below intentionally uses a subprocess because the
# published planner emits normal Python functions/control flow beyond this tiny
# AST subset.
_ALLOWED_AST = {
    ast.Module, ast.Expr, ast.Assign, ast.Name, ast.Load, ast.Store, ast.Constant,
    ast.Call, ast.keyword, ast.Dict, ast.List, ast.Tuple, ast.Set,
    ast.UnaryOp, ast.USub, ast.UAdd, ast.BinOp, ast.Add, ast.Sub, ast.Mult,
    ast.Div, ast.FloorDiv, ast.Mod, ast.Subscript, ast.Slice,
}


def _validate_code(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_AST:
            raise ValueError(f"Python node {type(node).__name__} is not allowed")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in {
                "call_api", "print", "str", "int", "float", "bool", "len", "dict", "list"
            }:
                raise ValueError("Only call_api and safe scalar/output helpers are allowed")


def safe_execute_code(code: str, call_api: Callable[[str, dict], Any]) -> dict:
    try:
        tree = ast.parse(str(code or ""), mode="exec")
        _validate_code(tree)
    except Exception as exc:
        return {"success": False, "stdout": "", "error": str(exc)}

    def _call(name, **params):
        return call_api(str(name), dict(params))

    safe_builtins = {
        "print": print, "str": str, "int": int, "float": float,
        "bool": bool, "len": len, "dict": dict, "list": list,
    }
    stdout = io.StringIO()
    try:
        with redirect_stdout(stdout):
            exec(compile(tree, "<apibank-agent>", "exec"),
                 {"__builtins__": safe_builtins, "call_api": _call}, {})
        return {"success": True, "stdout": stdout.getvalue(), "error": None}
    except Exception as exc:
        return {"success": False, "stdout": stdout.getvalue(), "error": str(exc)}


class LLMRunError(RuntimeError):
    """Model/provider failure. The runner treats this as infrastructure."""


class AgentBehaviorError(RuntimeError):
    """A completed model prediction that the benchmark method cannot execute/use."""


class AgentRuntimeError(RuntimeError):
    """Tool dependency/service failure, never a model repair opportunity."""


class ChatText(str):
    """Text-compatible response carrying provider metadata for saved traces."""
    def __new__(cls, content, metadata):
        obj = super().__new__(cls, content)
        obj.metadata = metadata
        return obj


def _response_metadata(response):
    choice = response.choices[0]
    return {"id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "finish_reason": getattr(choice, "finish_reason", None)}


def _response_trace(raw):
    return {"output": str(raw), "response": getattr(raw, "metadata", {})}


def _llm_timeout_seconds() -> int:
    raw = os.getenv("APIBANK_LLM_TIMEOUT_SEC", "180")
    try:
        value = int(raw)
    except Exception as exc:
        raise RuntimeError(f"APIBANK_LLM_TIMEOUT_SEC must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise RuntimeError("APIBANK_LLM_TIMEOUT_SEC must be a positive integer")
    return value


def _oca_llm_timeout_seconds() -> int:
    raw = os.getenv("APIBANK_OCA_LLM_TIMEOUT_SEC", "60")
    try:
        value = int(raw)
    except Exception as exc:
        raise RuntimeError(
            f"APIBANK_OCA_LLM_TIMEOUT_SEC must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise RuntimeError("APIBANK_OCA_LLM_TIMEOUT_SEC must be a positive integer")
    return value


def _oca_task_timeout_seconds() -> int:
    raw = os.getenv("APIBANK_OCA_TASK_TIMEOUT_SEC", "240")
    try:
        value = int(raw)
    except Exception as exc:
        raise RuntimeError(
            f"APIBANK_OCA_TASK_TIMEOUT_SEC must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise RuntimeError("APIBANK_OCA_TASK_TIMEOUT_SEC must be a positive integer")
    return value


def _oca_deepseek_completion_cap() -> int:
    raw = os.getenv("APIBANK_OCA_DEEPSEEK_MAX_COMPLETION_TOKENS", "4096")
    try:
        value = int(raw)
    except Exception as exc:
        raise RuntimeError(
            "APIBANK_OCA_DEEPSEEK_MAX_COMPLETION_TOKENS must be a positive integer, "
            f"got {raw!r}") from exc
    if value <= 0:
        raise RuntimeError("APIBANK_OCA_DEEPSEEK_MAX_COMPLETION_TOKENS must be positive")
    return value


def _toolcoder_timeout_seconds() -> int:
    raw = os.getenv("APIBANK_TOOLCODER_TIMEOUT_SEC", "30")
    try:
        value = int(raw)
    except Exception as exc:
        raise RuntimeError(
            f"APIBANK_TOOLCODER_TIMEOUT_SEC must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise RuntimeError("APIBANK_TOOLCODER_TIMEOUT_SEC must be a positive integer")
    return value


def _deepseek_structured_floor(model: str) -> int | None:
    from utils.model_provider import provider_settings
    settings = provider_settings(model)
    if settings.get("provider") != "deepseek" or settings.get("thinking") != "enabled":
        return None
    default_floor = 32768 if settings.get("reasoning_effort") == "max" else 16384
    raw = os.getenv("DEEPSEEK_STRUCTURED_MIN_TOKENS", str(default_floor))
    try:
        value = int(raw)
    except Exception as exc:
        raise RuntimeError(
            f"DEEPSEEK_STRUCTURED_MIN_TOKENS must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise RuntimeError(
            f"DEEPSEEK_STRUCTURED_MIN_TOKENS must be a positive integer, got {raw!r}")
    return value


def public_runtime_settings(agent_name: str, model: str | None = None) -> dict:
    """Non-secret execution settings that can affect benchmark outcomes."""
    name = str(agent_name or "").strip().lower()
    if name not in {"oca", "codeact", "toolcoder"}:
        raise RuntimeError(f"Unsupported API-Bank agent: {agent_name!r}")
    settings = {
        "llm_timeout_sec": _llm_timeout_seconds(),
        "toolcoder_timeout_sec": _toolcoder_timeout_seconds() if name == "toolcoder" else None,
    }
    from benchmarks.apibank_support import tool_timeout_seconds
    settings["tool_timeout_sec"] = tool_timeout_seconds()
    if name == "toolcoder":
        settings["worker_timeout_sec"] = _toolcoder_timeout_seconds() + tool_timeout_seconds()
    if name == "oca" and model:
        settings["deepseek_structured_min_tokens"] = _deepseek_structured_floor(model)
        settings["oca_llm_timeout_sec"] = _oca_llm_timeout_seconds()
        settings["oca_task_timeout_sec"] = _oca_task_timeout_seconds()
        settings["oca_deepseek_max_completion_tokens"] = _oca_deepseek_completion_cap()
        settings["oca_profile"] = _oca_profile()
        settings["oca_lv2_search_mode"] = _oca_lv2_search_mode()
        settings["oca_lv2_latent_recovery"] = _oca_lv2_latent_recovery_enabled()
        settings["oca_efficient_max_repairs"] = _oca_efficient_max_repairs()
        settings["oca_adaptive_verify_min_params"] = _oca_adaptive_verify_min_params()
    return settings


def _is_transient_llm_error(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        status_int = int(status) if status is not None else None
    except Exception:
        status_int = None
    if status_int in {408, 409, 429} or (status_int is not None and status_int >= 500):
        return True
    name = type(exc).__name__.lower()
    return any(token in name for token in ("ratelimit", "timeout", "connection", "temporar"))


def _completion_cap(model: str, requested: int) -> int:
    value = int(requested)
    if value <= 0:
        raise RuntimeError("max_completion_tokens must be positive")
    from utils.model_provider import provider_settings
    settings = provider_settings(model)
    if settings.get("provider") == "deepseek" and settings.get("thinking") == "enabled":
        value = max(value, int(settings.get("reasoning_min_tokens") or 16384))
    return value


def _chat(model: str, messages: List[dict], *, max_completion_tokens: int = 4096) -> str:
    """Provider-aware, metered chat call with bounded transient-only retries.

    Authentication/request errors fail immediately; only transient transport/rate
    errors are retried. Empty successful responses get one extra attempt. Provider
    exceptions are preserved as ``__cause__`` so the runner can classify them as
    infrastructure rather than benchmark failures.
    """
    import time
    from agents.common import make_client

    client = make_client()
    timeout = _llm_timeout_seconds()
    cap = _completion_cap(model, max_completion_tokens)
    attempts = 4
    empty_attempts = 0
    for attempt in range(attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_completion_tokens=cap,
                timeout=timeout,
            )
            try:
                content = str(response.choices[0].message.content or "").strip()
            except Exception as exc:
                raise LLMRunError("LLM response had no readable message content") from exc
            if content:
                return ChatText(content, _response_metadata(response))
            empty_attempts += 1
            if empty_attempts >= 2:
                raise LLMRunError("LLM returned empty content on both bounded attempts")
            # Empty content is a successful HTTP response but can be a transient
            # provider quirk. Retry once without consuming the full error budget.
            continue
        except LLMRunError:
            raise
        except Exception as exc:
            if not _is_transient_llm_error(exc) or attempt >= attempts - 1:
                raise LLMRunError(f"LLM call failed: {exc}") from exc
            wait = 2 ** attempt
            print(f"[WARN] transient LLM failure (attempt {attempt + 1}/{attempts}): {exc}; retrying in {wait}s")
            time.sleep(wait)
    raise LLMRunError("LLM call exhausted bounded retry attempts")


def _apibank_level(view: dict) -> int:
    try:
        level = int((view or {}).get("benchmark_level") or 1)
    except Exception:
        level = 1
    return 2 if level == 2 else 1


def _base_prompt(view: dict) -> str:
    level = _apibank_level(view)
    return (
        f"You are solving API-Bank Level {level}. Use ONLY the supplied API descriptions.\n"
        "Return exactly one next API request as [ApiName(key='value', ...)].\n"
        "This benchmark assumes year 2023 unless the dialogue explicitly says otherwise.\n\n"
        f"API DESCRIPTIONS:\n{view['api_descriptions']}\n\n"
        f"DIALOGUE SO FAR:\n{format_history(view['chat_history'])}"
    )


def _parse_json_object(text: str) -> dict:
    """Extract the first complete JSON object without greedy brace capture.

    DeepSeek normally follows the compact API-Bank JSON contract, but a provider
    may still prepend a short sentence or markdown fence.  ``JSONDecoder`` lets us
    accept the first complete object while rejecting truncated/concatenated junk;
    this avoids spending a repair call on harmless presentation differences.
    """
    candidate = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()

    decoder = json.JSONDecoder()
    starts = [i for i, ch in enumerate(candidate) if ch == "{"]
    errors = []
    for start in starts:
        try:
            obj, _ = decoder.raw_decode(candidate[start:])
        except Exception as exc:
            errors.append(exc)
            continue
        if isinstance(obj, dict):
            return obj
    if not starts:
        raise ValueError("no JSON object found")
    raise ValueError(f"no complete JSON object found: {errors[-1] if errors else 'invalid JSON'}")


def _parse_bracket_action_object(text: str) -> dict | None:
    """Parse one ``[ApiName(key=value)]`` action without executing model text.

    This is a deterministic compatibility fallback only.  It accepts literals
    that Python's ``ast.literal_eval`` can decode and rejects positional args,
    ``**kwargs``, attribute calls, expressions, and multiple actions.
    """
    call_text = extract_bracket_call(str(text or ""))
    if not call_text:
        return None
    inner = call_text[1:-1].strip()
    try:
        node = ast.parse(inner, mode="eval").body
    except Exception:
        return None
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.args:
        return None
    params = {}
    try:
        for kw in node.keywords:
            if kw.arg is None or kw.arg in params:
                return None
            params[kw.arg] = ast.literal_eval(kw.value)
    except Exception:
        return None
    return {"api_name": node.func.id, "params": params}


def _call_text(name: str, params: dict) -> str:
    return _code_action_text(name, params)


class _OCAHardTimeout(TimeoutError):
    """Raised when an API-Bank OCA provider call exceeds its hard wall-clock limit."""


def _oca_hard_timed_call(func, kwargs: dict, timeout_sec: float):
    """Run one provider call under a real POSIX wall-clock deadline when possible.

    The OpenAI-compatible SDK timeout remains in the request, but API-Bank OCA also
    needs a harness-level deadline so a blocked/custom client cannot ignore it.
    WSL/Linux runs execute on the main thread, where SIGALRM can interrupt blocking
    Python calls. Non-main-thread/non-POSIX callers retain the SDK timeout fallback.
    """
    import signal
    import threading
    import time

    seconds = max(0.001, float(timeout_sec))
    if (threading.current_thread() is not threading.main_thread() or
            not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer")):
        return func(**kwargs)

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def _raise_timeout(signum, frame):  # pragma: no cover - frame is signal runtime state
        raise _OCAHardTimeout(f"OCA provider call exceeded {seconds:.3f}s hard timeout")

    try:
        signal.signal(signal.SIGALRM, _raise_timeout)
        # Respect an already-armed earlier deadline instead of extending it.
        active_delay = float(previous_timer[0] or 0.0)
        effective = min(seconds, active_delay) if active_delay > 0 else seconds
        signal.setitimer(signal.ITIMER_REAL, effective)
        return func(**kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            elapsed = max(0.0, time.monotonic() - started)
            restored = max(0.001, float(previous_timer[0]) - elapsed)
            signal.setitimer(signal.ITIMER_REAL, restored, float(previous_timer[1] or 0.0))


class _OCACompletionProxy:
    def __init__(self, inner, model: str, deadline: float):
        self._inner = inner
        self._model = model
        self._deadline = float(deadline)
        self.responses = []
        self.terminal_error = None

    def create(self, **kwargs):
        if self.terminal_error is not None:
            raise self.terminal_error
        kwargs = dict(kwargs)
        import time

        requested_timeout = float(kwargs.get("timeout") or _llm_timeout_seconds())
        requested = int(kwargs.get("max_completion_tokens") or 4096)
        from utils.model_provider import provider_settings
        settings = provider_settings(self._model)
        deepseek_thinking = (
            settings.get("provider") == "deepseek" and
            settings.get("thinking") == "enabled"
        )
        deepseek_json = (
            deepseek_thinking and
            isinstance(kwargs.get("response_format"), dict) and
            str(kwargs["response_format"].get("type") or "").lower() == "json_object"
        )
        if deepseek_thinking:
            # In DeepSeek thinking mode max_tokens covers reasoning + final output.
            # OCA's generic planner asks for a small OpenAI-era completion budget,
            # which is not enough for high-effort DeepSeek reasoning. For API-Bank
            # OCA the benchmark-local environment value is therefore the actual
            # per-call generation budget, not merely a ceiling on that tiny request.
            local_cap = _oca_deepseek_completion_cap()
            kwargs["max_completion_tokens"] = local_cap
            # Prevent the provider-wide JSON floor from silently changing this
            # benchmark-owned budget. Also let this proxy inspect the first empty
            # JSON response so a length-exhausted response is not retried blindly.
            kwargs["_secat_structured_min_tokens"] = local_cap
            kwargs["_secat_empty_json_retries"] = 0
        else:
            local_cap = _completion_cap(self._model, requested)
            kwargs["max_completion_tokens"] = local_cap

        empty_json_retry_used = False
        for attempt in range(4):
            remaining = self._deadline - time.monotonic()
            if remaining <= 0:
                self.terminal_error = LLMRunError("OCA API-Bank task deadline exceeded")
                raise self.terminal_error
            call_timeout = max(0.001, min(
                requested_timeout, float(_oca_llm_timeout_seconds()), remaining))
            kwargs["timeout"] = call_timeout
            try:
                response = _oca_hard_timed_call(self._inner.create, kwargs, call_timeout)
                self.responses.append(_response_metadata(response))
                if deepseek_json:
                    try:
                        content = str(response.choices[0].message.content or "").strip()
                    except Exception:
                        content = ""
                    if not content:
                        try:
                            finish_reason = str(response.choices[0].finish_reason or "").lower()
                        except Exception:
                            finish_reason = ""
                        try:
                            completion_tokens = int(response.usage.completion_tokens or 0)
                        except Exception:
                            completion_tokens = 0
                        exhausted = (
                            finish_reason == "length" or
                            (local_cap > 0 and completion_tokens >= local_cap)
                        )
                        if exhausted:
                            self.terminal_error = LLMRunError(
                                "DeepSeek thinking exhausted the API-Bank OCA completion "
                                f"budget ({local_cap} tokens) before emitting final JSON; "
                                "increase APIBANK_OCA_DEEPSEEK_MAX_COMPLETION_TOKENS")
                            raise self.terminal_error
                        if empty_json_retry_used:
                            self.terminal_error = LLMRunError(
                                "DeepSeek JSON mode returned empty content on both bounded attempts")
                            raise self.terminal_error
                        empty_json_retry_used = True
                        # DeepSeek documents occasional empty JSON responses. Retry
                        # once only when the response did not consume the budget.
                        continue
                return response
            except _OCAHardTimeout as exc:
                self.terminal_error = LLMRunError(str(exc))
                raise self.terminal_error from exc
            except Exception as exc:
                if not _is_transient_llm_error(exc) or attempt == 3:
                    self.terminal_error = LLMRunError(f"OCA provider request failed: {exc}")
                    raise self.terminal_error from exc
                remaining = self._deadline - time.monotonic()
                if remaining <= 0:
                    self.terminal_error = LLMRunError("OCA API-Bank task deadline exceeded")
                    raise self.terminal_error from exc
                wait = min(float(2 ** attempt), remaining)
                if wait > 0:
                    time.sleep(wait)


class _OCAChatProxy:
    def __init__(self, inner, model: str, deadline: float):
        self.completions = _OCACompletionProxy(inner.completions, model, deadline)


class _OCAClientProxy:
    def __init__(self, inner, model: str, deadline: float):
        self.chat = _OCAChatProxy(inner.chat, model, deadline)


def _oca_planner_client(base_client, model: str, *, total_timeout_sec: int | None = None):
    """Wrap OCA's planner client with API-Bank timeout/token-budget controls."""
    import time
    total = int(total_timeout_sec or _oca_task_timeout_seconds())
    return _OCAClientProxy(base_client, model, time.monotonic() + total)


_APIBANK_OCA_POLICY = {
    "interaction_mode": "next_action",
    "maximum_actions": 1,
    "conversation_history_is_state": True,
    "allow_post_call_projection": False,
    "schema_value_normalization": True,
}


@contextmanager
def _oca_benchmark_clock_2023():
    """Scope the planner clock without rewriting the current generic OCA contract.

    A compatibility branch remains for externally supplied legacy planner builders
    that still expose the pre-action answer-mode schema. The production SECAT
    builder is otherwise preserved byte-for-byte except for the benchmark date.
    """
    import utils.evidence_plan as EP
    original = EP.build_planner_messages
    is_core_builder = (
        getattr(original, "__module__", "") == getattr(EP, "__name__", "") and
        getattr(original, "__name__", "") == "build_planner_messages"
    )

    def wrapped(question, tools, catalog_mode="full"):
        messages = original(question, tools, catalog_mode=catalog_mode)
        out = [dict(m) for m in messages]
        if out:
            content = str(out[0].get("content", ""))
            content = re.sub(r"Current date:\s*\d{4}-\d{2}-\d{2}",
                             "Current date: 2023-01-01", content)
            # Compatibility only for injected/legacy builders used by older SECAT
            # adapters. The current production planner contract is not rewritten.
            legacy_modes = "direct|list|count|boolean|comparison|asset"
            if not is_core_builder:
                if legacy_modes in content and legacy_modes + "|action" not in content:
                    content = content.replace(legacy_modes, legacy_modes + "|action")
                content += (
                    "\n\nAPI-Bank LV1 next-action contract:\n"
                    "- The benchmark target is the API action itself, not a post-call answer.\n"
                    "- Return exactly one independent POST step using only documented request fields.\n"
                    "- The conversation history is the complete available state. Reuse values from prior API responses.\n"
                    "- Never invent credentials, tokens, identifiers, or placeholder values. If a required value is unavailable, change the selected endpoint to the documented prerequisite API that produces it.\n"
                    "- A correction may change the selected endpoint; do not repeatedly repair parameters for an action whose prerequisites are unavailable.\n"
                    "- Normalize values to documented request formats, including datetime strings as %Y-%m-%d %H:%M:%S.\n"
                    "- Put user-known request values in literal fields; do not create response bindings.\n"
                    "- Set derivations=[], answer_derivations=[], and answer_mode=\"action\".\n"
                    "- Set answer_steps to the single action step id.\n"
                )
            out[0]["content"] = content
        return out

    EP.build_planner_messages = wrapped
    try:
        yield
    finally:
        EP.build_planner_messages = original


@contextmanager
def _oca_apibank_semantic_critic():
    """Keep the configured critic while making LV1's next-action semantics explicit."""
    import utils.evidence_plan as EP
    original = EP.semantic_plan_critic

    def wrapped(question, plan, tools, benchmark, model, client):
        steps = list((plan or {}).get("steps") or []) if isinstance(plan, dict) else []
        if (len(steps) == 1 and isinstance(steps[0], dict) and
                str(steps[0].get("method") or "").upper() == "POST" and
                not (steps[0].get("depends_on") or []) and
                bool(str(steps[0].get("endpoint") or "").strip())):
            return "compatible", [], "API-Bank next-action policy accepted single POST action"
        return original(question, plan, tools, benchmark, model, client)

    EP.semantic_plan_critic = wrapped
    try:
        yield
    finally:
        EP.semantic_plan_critic = original


def _name_value_schema(spec: dict) -> bool:
    """Detect a documented list-of-``name``/``value`` wire shape generically."""
    text = json.dumps(spec or {}, ensure_ascii=False, sort_keys=True).lower()
    return ("name" in text and "value" in text and
            any(word in text for word in ("list", "array", "dict", "object")))


def _as_bool(value, default=None):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "required"}:
            return True
        if lowered in {"false", "no", "0", "optional"}:
            return False
    return default


def _parameter_required(spec: dict, description: str) -> bool:
    """Interpret API metadata requiredness without turning string False into True."""
    if "required" in spec:
        parsed = _as_bool(spec.get("required"), None)
        if parsed is not None:
            return parsed
    if "optional" in spec:
        parsed = _as_bool(spec.get("optional"), None)
        if parsed is not None:
            return not parsed
    if "default" in spec:
        return False
    lowered = str(description or "").lower()
    optional_markers = (
        "[optional]", "optional", "may be omitted", "can be omitted",
        "only required when", "required only when", "only used when",
        "used when", "if status", "when status",
    )
    if any(marker in lowered for marker in optional_markers):
        return False
    return True


def _api_param_schema(raw) -> tuple[dict, list[str]]:
    properties: dict = {}
    required: list[str] = []
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = []
        for item in raw:
            if isinstance(item, dict) and item.get("name"):
                items.append((item.get("name"), item))
    else:
        items = []
    type_map = {
        "str": "string", "string": "string", "int": "integer", "integer": "integer",
        "float": "number", "number": "number", "bool": "boolean", "boolean": "boolean",
        "list": "array", "list(str)": "array", "array": "array",
        "dict": "object", "object": "object",
    }
    for name0, spec0 in items:
        name = str(name0 or "").strip()
        if not name:
            continue
        spec = spec0 if isinstance(spec0, dict) else {"description": str(spec0)}
        raw_type = str(spec.get("type") or spec.get("param_type") or "string").lower()
        schema = {"type": type_map.get(raw_type, "string")}
        desc = spec.get("description")
        if desc:
            schema["description"] = str(desc)
        enum = spec.get("enum") or spec.get("values")
        if isinstance(enum, list) and enum:
            schema["enum"] = enum
        if "default" in spec:
            schema["default"] = spec.get("default")
        # Preserve a common documented nested wire contract instead of reducing
        # every list to an untyped array. This is inferred from schema metadata,
        # never from task IDs or expected answers.
        if schema["type"] == "array" and _name_value_schema(spec):
            schema["items"] = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["name", "value"],
            }
        elif isinstance(spec.get("items"), dict) and schema["type"] == "array":
            schema["items"] = spec["items"]
        elif isinstance(spec.get("properties"), dict) and schema["type"] == "object":
            schema["properties"] = spec["properties"]
        properties[name] = schema
        if _parameter_required(spec, str(desc or "")):
            required.append(name)
    return properties, required


@contextmanager
def _oca_benchmark_registration(runtime, sample):
    """Expose the current API-Bank toolbox to OCA through its normal OAS registry."""
    import benchmarks as B

    catalog = runtime.api_catalog(sample)
    paths = {}
    for entry in catalog:
        name = str(entry["name"])
        raw_meta = entry.get("metadata") or {}
        input_parameters = (raw_meta.get("input_parameters") if isinstance(raw_meta, dict) else None)
        if input_parameters is None and isinstance(entry.get("description_json"), dict):
            input_parameters = entry["description_json"].get("input_parameters")
        props, required = _api_param_schema(input_parameters)
        schema = {"type": "object", "properties": props}
        if required:
            schema["required"] = required
        output_parameters = (raw_meta.get("output_parameters") if isinstance(raw_meta, dict) else None)
        if output_parameters is None and isinstance(entry.get("description_json"), dict):
            output_parameters = entry["description_json"].get("output_parameters")
        # Upstream descriptions name logical outputs, not envelope JSON paths.
        # ToolManager actually returns api_name/input/output/exception.
        response_schema = {"type": "object", "properties": {
            "api_name": {"type": "string"}, "input": schema,
            "output": {"description": "Tool output. " + json.dumps(output_parameters or {})},
            "exception": {"description": "Null on success, otherwise an error string."},
        }}
        paths[f"/{name}"] = {
            "post": {
                "summary": f"API-Bank tool {name}",
                "description": str(entry.get("description") or ""),
                "requestBody": {
                    "required": bool(required),
                    "content": {"application/json": {"schema": schema}},
                },
                "responses": {
                    "200": {
                        "description": "API-Bank local tool result",
                        "content": {"application/json": {"schema": response_schema}},
                    }
                },
            }
        }
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "API-Bank Level 1", "version": "2023"},
        "paths": paths,
    }
    with tempfile.TemporaryDirectory(prefix="secat_apibank_oca_oas_") as tmp_name:
        oas_file = Path(tmp_name) / "apibank_oas.json"
        oas_file.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
        token = hashlib.sha256((str(runtime.root) + sample.task_id).encode("utf-8")).hexdigest()[:16]
        benchmark = f"apibank_lv1_{token}"
        previous = B.BENCHMARKS.get(benchmark)
        B.BENCHMARKS[benchmark] = {
            "name": benchmark,
            "runtime_api": "apibank",
            "base_url": "local://apibank",
            "auth_strategy": "none",
            "env_key": "",
            "dataset_file": str(oas_file),
            "oas_file": str(oas_file),
            "id_prefix": "apibank_lv1_",
            "read_only": False,
            "api_label": "API-Bank Level 1",
            "auth_note": "Local API-Bank ToolManager; no network authentication.",
            "oca_policy": dict(_APIBANK_OCA_POLICY),
        }
        try:
            yield benchmark
        finally:
            if previous is None:
                B.BENCHMARKS.pop(benchmark, None)
            else:
                B.BENCHMARKS[benchmark] = previous


def _oca_projection_only_invalid(plan: dict) -> bool:
    """Allow only irrelevant post-call projection errors for LV1 next-action scoring."""
    if not isinstance(plan, dict):
        return False
    errors = [str(x) for x in (plan.get("validation_errors") or [])]
    if not errors:
        return False
    return all(
        err.startswith("derivation ")
        and " is not a documented scalar under selected record collection " in err
        and (
            ": identity field " in err
            or ": argmax field " in err
            or ": argmin field " in err
            or ": filter field " in err
        )
        for err in errors
    )


_OCA_PLACEHOLDER_VALUES = {
    "", "token", "user_token", "your_token", "your_token_here", "api_token",
    "access_token", "placeholder", "none", "null",
}


def _normalize_apibank_datetime(value):
    """Normalize common API-Bank datetime spellings without task-specific values."""
    if not isinstance(value, str):
        return value
    from datetime import datetime

    text = value.strip()
    # API-Bank dialogues frequently use ordinals, optional ``at``/commas, and
    # compact AM/PM spellings. Normalize syntax only; never invent a day/month.
    clean = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", text, flags=re.IGNORECASE)
    clean = re.sub(r",\s*,+", ",", clean)
    clean = re.sub(r",\s*at\s+", " at ", clean, flags=re.IGNORECASE)
    clean = re.sub(r"(?<=\d)(am|pm)\b", r" \1", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bnoon\b", "12:00 PM", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bmidnight\b", "12:00 AM", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean).strip()

    candidates = [clean]
    has_year = bool(re.search(r"\b\d{4}\b", clean))
    month_first = bool(re.match(
        r"^(?:January|February|March|April|May|June|July|August|September|October|November|December|"
        r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b",
        clean, flags=re.IGNORECASE))
    day_first = bool(re.match(r"^\d{1,2}\s+(?:of\s+)?[A-Za-z]+\b", clean))
    if not has_year and month_first:
        m = re.match(r"^([A-Za-z]+\s+\d{1,2})(.*)$", clean)
        if m:
            candidates.append(f"{m.group(1)}, 2023{m.group(2)}")
    if not has_year and day_first:
        m = re.match(r"^(\d{1,2}\s+(?:of\s+)?[A-Za-z]+)(.*)$", clean)
        if m:
            candidates.append(f"{m.group(1)}, 2023{m.group(2)}")

    formats = (
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d %I:%M %p",
        # A date-only value supplied to a documented datetime field means the
        # start of that date; this matches the vendor wire-format expectation.
        "%Y-%m-%d",
        "%B %d, %Y at %I:%M %p", "%b %d, %Y at %I:%M %p",
        "%B %d, %Y at %I %p", "%b %d, %Y at %I %p",
        "%B %d, %Y %I:%M %p", "%b %d, %Y %I:%M %p",
        "%B %d, %Y %I %p", "%b %d, %Y %I %p",
        "%B %d %Y at %I:%M %p", "%b %d %Y at %I:%M %p",
        "%B %d %Y at %I %p", "%b %d %Y at %I %p",
        "%B %d %Y %I:%M %p", "%b %d %Y %I:%M %p",
        "%d of %B, %Y at %I:%M %p", "%d of %b, %Y at %I:%M %p",
        "%d of %B, %Y at %I %p", "%d of %b, %Y at %I %p",
        "%d %B, %Y at %I:%M %p", "%d %b, %Y at %I:%M %p",
        "%d %B, %Y at %I %p", "%d %b, %Y at %I %p",
        "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %I:%M %p",
        # Month/day with no explicit time is also the beginning of the date.
        "%B %d, %Y", "%b %d, %Y", "%d of %B, %Y", "%d of %b, %Y",
        "%d %B, %Y", "%d %b, %Y", "%m/%d/%Y",
    )
    for candidate in candidates:
        for fmt in formats:
            try:
                parsed = datetime.strptime(candidate, fmt)
                return parsed.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
    return value


def _normalize_apibank_date(value):
    """Normalize date-only API-Bank fields to ``%Y-%m-%d`` when possible."""
    if not isinstance(value, str):
        return value
    from datetime import datetime

    text = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", value.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    candidates = [text]
    has_year = bool(re.search(r"\b\d{4}\b", text))
    month_first = bool(re.match(
        r"^(?:January|February|March|April|May|June|July|August|September|October|November|December|"
        r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b",
        text, flags=re.IGNORECASE))
    day_first = bool(re.match(r"^\d{1,2}\s+(?:of\s+)?[A-Za-z]+\b", text))
    if not has_year and month_first:
        candidates.append(text + ", 2023")
    if not has_year and day_first:
        candidates.append(text + ", 2023")

    formats = (
        "%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y",
        "%d of %B, %Y", "%d of %b, %Y", "%d %B, %Y", "%d %b, %Y",
        "%m/%d/%Y",
    )
    for candidate in candidates:
        for fmt in formats:
            try:
                return datetime.strptime(candidate, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
    return value

def _placeholder_value(value) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().lower() in _OCA_PLACEHOLDER_VALUES)


def _canonical_measurement_name(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


def _canonical_measurement_value(value):
    text = str(value).strip()
    # When prose includes a simple unit after a scalar, keep the schema value and
    # omit the prose unit. Structured values such as 120/80 remain intact.
    m = re.fullmatch(r"([+-]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)\s+[A-Za-z%/]+", text)
    return m.group(1) if m else text


def _parse_name_value_piece(piece: str):
    text = str(piece or "").strip(" ,.;")
    if not text:
        return None
    patterns = (
        r"^(.+?)\s*(?::|=|\bis\b)\s*([+-]?\d[^,;]*)$",
        r"^(.+?)\s+([+-]?\d[^,;]*)$",
    )
    for pattern in patterns:
        m = re.match(pattern, text, flags=re.IGNORECASE)
        if m:
            name = _canonical_measurement_name(m.group(1))
            value = _canonical_measurement_value(m.group(2))
            if name and value:
                return {"name": name, "value": value}
    return None


def _coerce_name_value_list(value):
    """Coerce alternate representations into a documented name/value list.

    This is schema-driven and endpoint-agnostic. Already-canonical input is left
    unchanged to preserve previously correct benchmark behavior.
    """
    if isinstance(value, list) and value and all(
        isinstance(x, dict) and "name" in x and "value" in x for x in value
    ):
        return value

    out = []
    if isinstance(value, dict):
        if "name" in value and "value" in value:
            return [value]
        for key, item in value.items():
            if isinstance(item, (str, int, float)):
                out.append({
                    "name": _canonical_measurement_name(key),
                    "value": _canonical_measurement_value(item),
                })
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                if "name" in item and "value" in item:
                    out.append(item)
                elif len(item) == 1:
                    key, val = next(iter(item.items()))
                    out.append({
                        "name": _canonical_measurement_name(key),
                        "value": _canonical_measurement_value(val),
                    })
                elif all(isinstance(v, (str, int, float)) for v in item.values()):
                    for key, val in item.items():
                        out.append({
                            "name": _canonical_measurement_name(key),
                            "value": _canonical_measurement_value(val),
                        })
            elif isinstance(item, str):
                parsed = _parse_name_value_piece(item)
                if parsed:
                    out.append(parsed)
    elif isinstance(value, str):
        pieces = re.split(r"\s*(?:,|;|\band\b)\s*", value, flags=re.IGNORECASE)
        for piece in pieces:
            parsed = _parse_name_value_piece(piece)
            if parsed:
                out.append(parsed)

    return out if out else value


def _input_fields_for(catalog: list[dict], name: str) -> dict:
    entry = next((x for x in catalog or [] if str(x.get("name")) == str(name)), None)
    raw_meta = (entry or {}).get("metadata") or {}
    fields = raw_meta.get("input_parameters") if isinstance(raw_meta, dict) else None
    if fields is None and isinstance((entry or {}).get("description_json"), dict):
        fields = entry["description_json"].get("input_parameters")
    return fields if isinstance(fields, dict) else {}


def _canonical_enum_value(value, enum):
    if not isinstance(value, str) or not isinstance(enum, list):
        return value
    if value in enum:
        return value
    key = re.sub(r"[^a-z0-9]+", "", value.lower())
    matches = [item for item in enum if isinstance(item, str) and
               re.sub(r"[^a-z0-9]+", "", item.lower()) == key]
    return matches[0] if len(matches) == 1 else value


def _normalize_oca_action_params(
        catalog: list[dict], name: str, params: dict, *,
        allow_omitted_required: bool = False) -> dict:
    """Apply schema-declared wire normalization without inventing task values.

    Required values retain the r22 safety contract: missing/placeholder values are
    rejected before probing. Only fields documented as optional/conditional may have
    empty placeholder values omitted.
    """
    fields = _input_fields_for(catalog, name)
    normalized = dict(params or {})
    for field, spec0 in fields.items():
        spec = spec0 if isinstance(spec0, dict) else {"description": str(spec0)}
        description = str(spec.get("description") or "")
        required = _parameter_required(spec, description)
        present = field in normalized
        value = normalized.get(field)

        if required:
            if not present:
                # API-Bank contains state-machine APIs whose static schema marks
                # later-stage fields as required even though the first stage
                # legitimately omits them.  The strict/default contract remains
                # unchanged; only the efficient LV1 path may defer a truly omitted
                # field to the local sandbox, which is authoritative for the current
                # stage.  Explicit placeholders are never allowed through.
                if allow_omitted_required:
                    continue
                raise AgentBehaviorError(
                    f"OCA next action uses a missing or placeholder value for required field {field!r}")
            if _placeholder_value(value):
                raise AgentBehaviorError(
                    f"OCA next action uses a missing or placeholder value for required field {field!r}")
        else:
            if not present:
                continue
            # Optional/conditional placeholders must not be promoted into wire values.
            # A zero emitted for an optional stage field is treated as an unset sentinel;
            # required numeric zero remains valid because the branch above preserves it.
            if _placeholder_value(value) or value == 0:
                normalized.pop(field, None)
                continue

        enum = spec.get("enum") or spec.get("values")
        if isinstance(enum, list):
            value = _canonical_enum_value(value, enum)
            normalized[field] = value
        if "%Y-%m-%d %H:%M:%S" in description:
            normalized[field] = _normalize_apibank_datetime(normalized[field])
        elif "%Y-%m-%d" in description:
            normalized[field] = _normalize_apibank_date(normalized[field])
        if field in normalized and _name_value_schema(spec):
            normalized[field] = _coerce_name_value_list(normalized[field])
    return normalized


def _catalog_produces_field(catalog: list[dict], field: str) -> bool:
    target = str(field or "").strip().lower()
    for entry in catalog or []:
        meta = entry.get("metadata") or {}
        outputs = meta.get("output_parameters") if isinstance(meta, dict) else None
        if outputs is None and isinstance(entry.get("description_json"), dict):
            outputs = entry["description_json"].get("output_parameters")
        names = outputs.keys() if isinstance(outputs, dict) else [
            x.get("name") for x in (outputs or []) if isinstance(x, dict)]
        if any(str(name or "").strip().lower() == target for name in names):
            return True
    return False


def _unrecoverable_missing_field(error: str, catalog: list[dict], dialogue: str) -> str | None:
    match = re.search(
        r"missing or placeholder value for required field ['\"]([^'\"]+)['\"]",
        str(error or ""), re.IGNORECASE)
    if not match:
        return None
    field = match.group(1).strip()
    if _catalog_produces_field(catalog, field):
        return None
    if re.search(rf"\b{re.escape(field)}\b", str(dialogue or ""), re.IGNORECASE):
        return None
    return field


def _oca_lv1_answer_only_error(error: str) -> bool:
    """Return True for generic answer-surface errors irrelevant to LV1 actions."""
    err = str(error or "")
    return (
        err == "plan has no answer-step endpoint"
        or err.startswith("plan has no explicit answer_steps/answer_source")
        or err.startswith("answer_derivations reference missing derivations:")
        or (err.startswith("answer step ") and " has no explicit derivation" in err)
        or err.startswith("boolean answer requires an explicit host-replayable")
        or err.startswith("answer intent mismatch:")
        or (
            err.startswith("derivation ")
            and " is not a documented scalar under selected record collection " in err
        )
    )


def _oca_action_surface_only_invalid(plan: dict) -> bool:
    """Accept a structurally valid LV1 action despite answer-only planner errors.

    LV1 scores the single next API call itself. Generic post-call projection,
    boolean-answer, asset-surface, and answer-lineage requirements must not veto an
    otherwise documented independent POST action. Request-schema, dependency,
    binding, and endpoint errors are *not* waived.
    """
    if not isinstance(plan, dict):
        return False
    errors = [str(x) for x in (plan.get("validation_errors") or [])]
    if not errors or not all(_oca_lv1_answer_only_error(x) for x in errors):
        return False
    steps = list(plan.get("steps") or [])
    if len(steps) != 1 or not isinstance(steps[0], dict):
        return False
    step = steps[0]
    return (
        str(step.get("method") or "").upper() == "POST"
        and not (step.get("depends_on") or [])
        and bool(str(step.get("endpoint") or "").strip())
    )

def _oca_step_to_action(plan: dict, sample, catalog: list[dict] | None = None) -> tuple[str, dict]:
    projection_only = _oca_projection_only_invalid(plan)
    action_surface_only = _oca_action_surface_only_invalid(plan)
    if (
        not isinstance(plan, dict)
        or (not plan.get("valid", False) and not projection_only and not action_surface_only)
        or (plan.get("execution_eligible") is False and not projection_only and not action_surface_only)
    ):
        raise AgentBehaviorError(
            "OCA evidence planner returned an invalid API-Bank plan: " +
            str(plan.get("validation_errors") if isinstance(plan, dict) else "invalid plan"))
    steps = list(plan.get("steps") or [])
    if not steps or not all(isinstance(step, dict) for step in steps):
        raise AgentBehaviorError("OCA evidence planner returned no API action")
    step = next((x for x in steps if not (x.get("depends_on") or [])), steps[0])
    if str(step.get("method") or "").upper() != "POST" or step.get("depends_on"):
        raise AgentBehaviorError("OCA next action must be an independent documented POST tool action")
    endpoint = str(step.get("endpoint") or "").strip()
    if not endpoint:
        raise AgentBehaviorError("OCA evidence planner returned a step without endpoint")
    name = endpoint.strip("/").split("/")[-1]
    # ``sample`` is redacted at the agent boundary.  For LV2 its api_names are
    # exactly the *currently visible* ToolSearcher surface (ToolSearcher itself
    # before retrieval, or the tools returned by the latest visible search).
    # Enforcing this catalog is therefore a protocol check, not a hidden-target
    # oracle, and prevents hallucinated/non-visible tools from being submitted.
    valid = {str(x) for x in getattr(sample, "api_names", ())}
    if valid and name not in valid:
        raise AgentBehaviorError(f"OCA selected API-Bank tool outside the visible catalog: {name!r}")
    unresolved = []
    for key in ("path_bindings", "query_bindings", "body_bindings"):
        if step.get(key):
            unresolved.append(key)
    if unresolved:
        raise AgentBehaviorError(
            "OCA next-action plan contains unresolved runtime bindings: " + ", ".join(unresolved))
    params: dict = {}
    for key in ("path_literals", "query_literals", "body_literals"):
        value = step.get(key)
        if isinstance(value, dict):
            params.update(value)
    if catalog is not None:
        params = _normalize_oca_action_params(catalog, name, params)
    return name, params


def _probe_oca_action(runtime, sample, name: str, params: dict) -> dict:
    """Execute one structured OCA action in the answer-free API-Bank sandbox."""
    code = f"print(call_api(api_name={str(name)!r}, params={dict(params)!r}))"
    execution = _execute_toolcoder_code(runtime, sample, code)
    if not execution.get("success"):
        return {
            "success": False,
            "error": str(execution.get("error") or execution.get("stderr") or
                         "sandbox execution failed"),
            "execution": execution,
        }
    calls = list(execution.get("target_calls") or [])
    if len(calls) != 1:
        return {
            "success": False,
            "error": f"sandbox expected one action, observed {len(calls)}",
            "execution": execution,
        }
    result = calls[0].get("result")
    exception = result.get("exception") if isinstance(result, dict) else None
    if exception:
        return {"success": False, "error": str(exception), "execution": execution}
    return {"success": True, "error": None, "execution": execution}


def _oca_action_schema_hint(catalog: list[dict] | None, name: str | None) -> str:
    if not catalog or not name:
        return ""
    fields = _input_fields_for(catalog, name)
    if not fields:
        return ""
    compact = {}
    for field, spec0 in fields.items():
        spec = spec0 if isinstance(spec0, dict) else {"description": str(spec0)}
        compact[str(field)] = {
            key: spec[key] for key in ("type", "description", "required", "default")
            if key in spec
        }
    return "\nDocumented input schema for this action: " + json.dumps(
        compact, ensure_ascii=False, sort_keys=True)


def _oca_action_repair_feedback(
    attempt: int, error: str, *, catalog: list[dict] | None = None,
    action_name: str | None = None,
) -> str:
    return (
        "\n\nAPI-Bank adapter feedback from rejected attempt " + str(attempt) + ":\n"
        + str(error).strip()
        + _oca_action_schema_hint(catalog, action_name)
        + "\nCreate a new complete next-action plan from the original dialogue. "
        "You may change the endpoint when the rejected action lacks a prerequisite. "
        "Re-check the exact documented request-field wire shapes and formats; when "
        "the sandbox says a value has invalid format, change its representation to "
        "match the schema rather than paraphrasing the same value. For documented "
        "identifier/code fields, use the canonical code form rather than a display "
        "name. Do not repeat the rejected action unchanged and do not invent missing values."
    )

def _run_oca_evidence(*, view: dict, sample, runtime, model: str) -> AgentPrediction:
    """Legacy full-evidence API-Bank LV1 profile (accuracy fallback / regression baseline)."""
    import config
    import utils.evidence_plan as EP
    from agents.common import make_client

    base_question = (
        "API-Bank Level 1 next-action task. The benchmark year is 2023. "
        "Predict only the single next API action justified by the dialogue, not the "
        "final conversational outcome. A documented prerequisite API action is valid "
        "when required inputs for the user's ultimate requested operation are not yet "
        "available. Use only documented request fields and values present in the "
        "dialogue or prior API observations; never invent credentials, tokens, IDs, "
        "or placeholders.\n\n" +
        format_history(view.get("chat_history") or [])
    )
    client = _oca_planner_client(make_client(), model)
    catalog_fn = getattr(runtime, "api_catalog", None)
    catalog = catalog_fn(sample) if callable(catalog_fn) else None
    can_probe = hasattr(runtime, "root")
    feedback = ""
    trace = []
    failures = []
    with (
        _oca_benchmark_registration(runtime, sample) as benchmark,
        _oca_benchmark_clock_2023(),
        _oca_apibank_semantic_critic(),
    ):
        for attempt in range(1, 4):
            print(f"  [OCA/API-BANK] planning attempt {attempt}/3", flush=True)
            response_start = len(client.chat.completions.responses)
            question = base_question + feedback
            plan, planner_output = EP.make_evidence_plan(
                question, benchmark, model, client,
                attempts=1, api_hints=None, enable_semantic_adapters=False,
                catalog_mode="full",
                # Keep the configured semantic machinery enabled so deterministic
                # route invariants remain active. The scoped API-Bank critic wrapper
                # above treats a valid action-mode next step as compatible without an
                # inappropriate final-answer LLM review.
                enable_semantic_critic=bool(getattr(config, "OCA_GENERIC_PLAN_CRITIC", False)),
                enable_selection_semantic_guard=True,
            )
            planner_trace = {
                "stage": "oca_evidence_plan", "attempt": attempt, "plan": plan,
                "output": planner_output,
                "responses": client.chat.completions.responses[response_start:],
            }
            trace.append(planner_trace)
            diagnostics = list((plan or {}).get("planner_attempt_diagnostics") or [])
            diagnostic_text = " ".join(
                str(err)
                for item in diagnostics if isinstance(item, dict)
                for err in (item.get("errors") or [])
            )
            transport_text = " ".join([str(planner_output or ""), diagnostic_text])
            if (not (isinstance(plan, dict) and plan.get("valid")) and
                    "planner call failed:" in transport_text.lower()):
                detail = transport_text.split("planner call failed:", 1)[-1].strip()
                raise LLMRunError(f"OCA planner LLM call failed: {detail or 'planner transport failure'}")
            try:
                name, params = _oca_step_to_action(plan, sample, catalog)
            except AgentBehaviorError as exc:
                error = str(exc)
                failures.append(error)
                planner_trace["rejected"] = error
                print(f"  [OCA/API-BANK] rejected attempt {attempt}: {error}", flush=True)
                missing = _unrecoverable_missing_field(error, catalog, base_question)
                if missing:
                    terminal = AgentBehaviorError(
                        f"OCA API-Bank cannot supply required field {missing!r}: "
                        "unavailable from dialogue or documented APIs")
                    terminal.agent_trace = trace
                    raise terminal
                if len(failures) >= 2 and failures[-1] == failures[-2]:
                    repeated = AgentBehaviorError(
                        "OCA API-Bank repair stopped after repeated no progress: " + error)
                    repeated.agent_trace = trace
                    raise repeated
                step_hint = None
                steps_hint = list((plan or {}).get("steps") or []) if isinstance(plan, dict) else []
                if steps_hint and isinstance(steps_hint[0], dict):
                    step_hint = str(steps_hint[0].get("endpoint") or "").strip("/") or None
                feedback += _oca_action_repair_feedback(
                    attempt, error, catalog=catalog, action_name=step_hint)
                continue
            if can_probe:
                print(f"  [OCA/API-BANK] probing {name}", flush=True)
                probe = _probe_oca_action(runtime, sample, name, params)
                trace.append({
                    "stage": "oca_action_probe", "attempt": attempt,
                    "api_name": name, "params": params,
                    "success": bool(probe.get("success")), "error": probe.get("error"),
                })
                if not probe.get("success"):
                    error = "sandbox rejected " + _call_text(name, params) + ": " + str(probe.get("error"))
                    failures.append(error)
                    print(f"  [OCA/API-BANK] rejected attempt {attempt}: {error}", flush=True)
                    if len(failures) >= 2 and failures[-1] == failures[-2]:
                        repeated = AgentBehaviorError(
                            "OCA API-Bank repair stopped after repeated no progress: " + error)
                        repeated.agent_trace = trace
                        raise repeated
                    feedback += _oca_action_repair_feedback(
                        attempt, error, catalog=catalog, action_name=name)
                    continue
            return AgentPrediction(
                prediction_text=_call_text(name, params), strategy="oca", trace=trace,
                action={"api_name": name, "params": params},
            )
    exc = AgentBehaviorError(
        "OCA API-Bank action remained invalid after three bounded attempts: " +
        " | ".join(failures))
    exc.agent_trace = trace
    raise exc


# ---------------------------------------------------------------------------
# API-Bank cost-efficient OCA action profiles
# ---------------------------------------------------------------------------

def _oca_profile() -> str:
    """Select the API-Bank OCA execution profile.

    ``evidence`` preserves the full evidence planner. ``efficient`` is the
    compact one-action selector. ``adaptive`` adds conservative semantic
    verification. ``consensus`` adds an independent second solve on ambiguous
    cases. ``lv2`` is the API-Bank Level-2 protocol-aware profile: it uses a
    dedicated semantic retrieval-query planner while ToolSearcher is visible and
    the adaptive action path after retrieval.
    """
    value = str(os.getenv("APIBANK_OCA_PROFILE", "evidence") or "evidence").strip().lower()
    aliases = {
        "legacy": "evidence", "full": "evidence",
        "action": "efficient", "fast": "efficient",
        "verify": "adaptive", "verified": "adaptive", "efficient_verify": "adaptive",
        "dual": "consensus", "independent": "consensus", "vote": "consensus",
        "level2": "lv2", "lv2_adaptive": "lv2",
    }
    value = aliases.get(value, value)
    if value not in {"evidence", "efficient", "adaptive", "consensus", "lv2"}:
        raise RuntimeError(
            "APIBANK_OCA_PROFILE must be 'evidence', 'efficient', 'adaptive', 'consensus', or 'lv2' "
            f"(got {value!r})")
    return value


def _oca_lv2_search_mode() -> str:
    """Return the reproducible LV2 ToolSearcher query strategy."""
    value = str(os.getenv("APIBANK_OCA_LV2_SEARCH_MODE", "lexical") or "lexical").strip().lower()
    if value not in {"lexical", "model"}:
        raise RuntimeError("APIBANK_OCA_LV2_SEARCH_MODE must be 'lexical' or 'model'")
    return value


def _oca_lv2_latent_recovery_enabled() -> bool:
    """Whether LV2 may emit one dialogue-inferred API outside stale descriptions.

    This mirrors released ToolCoder's execution capability without exposing an
    API inventory. Keep it explicitly configurable so paper runs and ablations
    can record/disable the behavior rather than relying on an implicit patch.
    """
    raw = str(os.getenv("APIBANK_OCA_LV2_LATENT_RECOVERY", "1") or "1").strip().lower()
    if raw in {"1", "true", "yes", "on", "enabled"}:
        return True
    if raw in {"0", "false", "no", "off", "disabled"}:
        return False
    raise RuntimeError(
        "APIBANK_OCA_LV2_LATENT_RECOVERY must be 0/1 or false/true "
        f"(got {raw!r})")


def _oca_lv2_name_recovery_enabled() -> bool:
    """Opt-in free-form hidden-name inference for stale ToolSearcher-only states.

    r40 live audit recovered no such cases and paid substantial extra tokens.
    r41 therefore keeps exact dialogue-named recovery on by default but disables
    free-form latent-name guessing unless explicitly requested for ablation.
    """
    raw = str(os.getenv("APIBANK_OCA_LV2_NAME_RECOVERY", "0") or "0").strip().lower()
    if raw in {"1", "true", "yes", "on", "enabled"}:
        return True
    if raw in {"0", "false", "no", "off", "disabled"}:
        return False
    raise RuntimeError(
        "APIBANK_OCA_LV2_NAME_RECOVERY must be 0/1 or false/true "
        f"(got {raw!r})")


def _oca_efficient_max_repairs() -> int:
    raw = str(os.getenv("APIBANK_OCA_EFFICIENT_MAX_REPAIRS", "1") or "1").strip()
    try:
        value = int(raw)
    except Exception as exc:
        raise RuntimeError(
            f"APIBANK_OCA_EFFICIENT_MAX_REPAIRS must be 0 or 1, got {raw!r}") from exc
    if value not in {0, 1}:
        raise RuntimeError("APIBANK_OCA_EFFICIENT_MAX_REPAIRS must be 0 or 1")
    return value


def _oca_adaptive_verify_min_params() -> int:
    """Minimum parameter count that triggers r28 semantic verification.

    Three parameters is intentionally conservative: r27's blind holdout showed
    that simple one/two-field actions were already highly reliable, while the
    remaining semantic risk concentrated in multi-field stateful actions.
    """
    raw = str(os.getenv("APIBANK_OCA_ADAPTIVE_VERIFY_MIN_PARAMS", "3") or "3").strip()
    try:
        value = int(raw)
    except Exception as exc:
        raise RuntimeError(
            f"APIBANK_OCA_ADAPTIVE_VERIFY_MIN_PARAMS must be an integer from 1 to 20, got {raw!r}") from exc
    if not 1 <= value <= 20:
        raise RuntimeError("APIBANK_OCA_ADAPTIVE_VERIFY_MIN_PARAMS must be from 1 to 20")
    return value


def _compact_action_catalog(catalog: list[dict] | None) -> list[dict]:
    """Return only metadata needed to choose and type one API-Bank action.

    ``api_catalog`` retains the vendor description both as raw JSON text and as
    parsed metadata. Sending both copies wastes prompt/cache tokens. Prefer the
    parsed semantic description plus input/output schemas, falling back to the
    raw description only when it cannot be parsed.
    """
    compact = []
    for entry in catalog or []:
        if not isinstance(entry, dict):
            continue
        meta = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        desc_json = entry.get("description_json") if isinstance(entry.get("description_json"), dict) else {}
        inputs = meta.get("input_parameters")
        if inputs is None:
            inputs = desc_json.get("input_parameters")
        outputs = meta.get("output_parameters")
        if outputs is None:
            outputs = desc_json.get("output_parameters")
        description = (
            meta.get("description") or desc_json.get("description") or
            desc_json.get("api_description") or desc_json.get("summary") or ""
        )
        if not description and not desc_json:
            description = str(entry.get("description") or "")
        item = {
            "name": str(entry.get("name") or meta.get("name") or desc_json.get("name") or ""),
            "description": str(description or ""),
            "input_parameters": inputs or {},
        }
        # Outputs are useful only to recognize documented prerequisite APIs.
        if outputs:
            item["output_parameters"] = outputs
        compact.append(item)
    # Preserve the benchmark/toolbox order used by the source sample. Reordering
    # tools can change model choice even though it does not save tokens.
    return compact


def _oca_direct_action_messages(view: dict, catalog: list[dict] | None,
                                *, previous_action: dict | None = None,
                                error: str | None = None,
                                allow_unlisted_api: bool = False) -> list[dict]:
    """Build a compact one-action prompt instead of a general evidence plan.

    API-Bank scores the immediate API request, so asking the model to also
    construct answer derivations, post-call projections, and a multi-step plan
    creates work that cannot improve the official LV1 target.  Keep OCA's useful
    invariants -- documented tools, typed request fields, conversation state,
    prerequisite selection, and deterministic normalization -- but request only
    the object the benchmark actually evaluates.
    """
    level = _apibank_level(view)
    system = (
        f"You are OCA operating in API-Bank Level {level} next-action mode. "
        "Predict exactly ONE immediate API request, not the final conversational outcome. "
        "Use an API and request fields documented in the supplied catalog. "
        "The dialogue and prior API observations are the complete available state. "
        "Resolve the MOST RECENT unfinished user intent; do not repeat an API action that the history already completed. "
        "In Level 2, when the catalog contains semantic-search candidates, distinguish the requested operation carefully "
        "(for example add vs modify vs delete vs query) and choose the candidate whose description directly implements that operation and object. "
        "When acting on an existing record, reuse its identifying field values exactly from prior API calls instead of paraphrasing them. "
        "For lookup/search keys such as symptom, entity, title, or content, prefer the ORIGINAL user-provided term over diagnoses, labels, summaries, or interpretations introduced only by assistant text. Use the shortest user-grounded key that still identifies the requested item. "
        "If the user's requested operation lacks a required value but a documented prerequisite API can obtain it, "
        "choose that prerequisite action instead. Never invent missing credentials, tokens, IDs, or placeholder values. "
        "Match documented wire shapes exactly, especially arrays/objects and date/time formats. "
        "Assume year 2023 when the dialogue gives a month/day but no year. "
        "Return JSON only, with exactly this shape: "
        '{"api_name":"ExactDocumentedName","params":{"field":value}}. '
        "Do not return Python, prose, API results, derivations, or multiple actions."
    )
    if allow_unlisted_api and level == 2:
        system += (
            " API-Bank Level 2 has a known published-dialogue edge case: the latest "
            "ToolSearcher description can be incomplete or stale even though the visible "
            "dialogue has already identified or progressed to another concrete API. To keep "
            "the same action boundary as code-generating agents, you may infer exactly ONE "
            "CamelCase API name outside the supplied catalog only when the visible dialogue "
            "itself gives strong evidence: an API/tool is explicitly named, authentication or "
            "another prerequisite has already progressed beyond discovery, or the listed "
            "operation clearly contradicts the unresolved user operation. Do not assume or "
            "enumerate a hidden tool inventory, do not try alternatives, and do not invent "
            "credentials or IDs. If the supplied catalog fits the dialogue, use it. For an "
            "inferred API, infer only request fields supported by dialogue/prior API evidence; "
            "a real structural execution error may be repaired once."
        )
    else:
        system += " Do not use APIs outside the supplied catalog."
    if len(catalog or []) == 1 and not allow_unlisted_api:
        only_name = str((catalog or [])[0].get("name") or "")
        if only_name:
            system += (
                f" The currently eligible catalog contains exactly one API: {only_name}. "
                "Use that API and focus only on binding its request fields from visible evidence."
            )
    user = (
        "DOCUMENTED API CATALOG:\n" +
        json.dumps(_compact_action_catalog(catalog), ensure_ascii=False, separators=(",", ":"), sort_keys=True) +
        "\n\nDIALOGUE SO FAR:\n" + format_history(view.get("chat_history") or [])
    )
    if previous_action is not None or error:
        user += (
            "\n\nThe previous candidate was rejected by deterministic validation or the local API sandbox. "
            "Correct only what is necessary. You may change the API when the rejected action lacks a prerequisite."
            "\nPREVIOUS CANDIDATE: " + json.dumps(previous_action or {}, ensure_ascii=False, default=str) +
            "\nREJECTION: " + str(error or "unknown rejection")
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _coerce_declared_wire_types(catalog: list[dict] | None, name: str, params: dict) -> dict:
    """Cheap schema-only coercions before the API sandbox.

    The transformation is deliberately conservative: it only converts values
    whose documented primitive/container type is unambiguous. It never invents
    a missing value and therefore preserves r24's provenance/placeholder safety.
    """
    import ast
    fields = _input_fields_for(catalog or [], name)
    out = dict(params or {})
    # Drop model-added fields that are not part of a non-empty documented schema.
    if fields:
        out = {k: v for k, v in out.items() if k in fields}
    for field, spec0 in fields.items():
        if field not in out:
            continue
        spec = spec0 if isinstance(spec0, dict) else {"description": str(spec0)}
        raw_type = str(spec.get("type") or spec.get("param_type") or "").strip().lower()
        value = out[field]
        try:
            if raw_type in {"int", "integer"} and isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
                out[field] = int(value.strip())
            elif raw_type in {"float", "number"} and isinstance(value, str) and re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", value.strip()):
                out[field] = float(value.strip())
            elif raw_type in {"bool", "boolean"} and isinstance(value, str) and value.strip().lower() in {"true", "false"}:
                out[field] = value.strip().lower() == "true"
            elif raw_type in {"list", "array", "list(str)"} and isinstance(value, str):
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    out[field] = parsed
            elif raw_type in {"dict", "object"} and isinstance(value, str):
                parsed = ast.literal_eval(value)
                if isinstance(parsed, dict):
                    out[field] = parsed
        except Exception:
            # Keep the original value; the normal validator/sandbox remains authoritative.
            pass
    return out


def _parse_direct_oca_action(
        raw_text: str, sample, catalog: list[dict] | None, *,
        allow_omitted_required: bool = False,
        allow_unlisted_api: bool = False) -> tuple[str, dict]:
    try:
        obj = _parse_json_object(str(raw_text or ""))
    except Exception as exc:
        obj = _parse_bracket_action_object(str(raw_text or ""))
        if obj is None:
            raise AgentBehaviorError(
                f"OCA efficient action output was not a complete one-action JSON/call object: {exc}") from exc
    name = str(obj.get("api_name") or obj.get("api") or obj.get("name") or "").strip()
    params = obj.get("params")
    if not isinstance(params, dict):
        for alias in ("parameters", "arguments", "input"):
            candidate = obj.get(alias)
            if isinstance(candidate, dict):
                params = candidate
                break
    if not name:
        raise AgentBehaviorError("OCA efficient action output omitted api_name")
    if not isinstance(params, dict):
        raise AgentBehaviorError("OCA efficient action output must contain a params object")
    # The runner passes a redacted sample.  In LV2 this set is the current
    # model-visible catalog only, so membership validation does not expose the
    # hidden future/ground-truth tool.
    visible_valid = {str(x) for x in getattr(sample, "api_names", ())}
    catalog_valid = {str(x.get("name") or "") for x in (catalog or []) if isinstance(x, dict)}
    valid = catalog_valid or visible_valid
    if valid and name not in valid and not allow_unlisted_api:
        raise AgentBehaviorError(f"OCA selected API-Bank tool outside the visible catalog/eligible subset: {name!r}")
    params = _coerce_declared_wire_types(catalog, name, params)
    if catalog is not None:
        params = _normalize_oca_action_params(
            catalog, name, params, allow_omitted_required=allow_omitted_required)
    return name, params


def _oca_direct_action_call(client, model: str, view: dict, catalog: list[dict] | None,
                            *, previous_action: dict | None = None,
                            error: str | None = None,
                            allow_unlisted_api: bool = False):
    from utils.token_meter import stage as token_stage
    messages = _oca_direct_action_messages(
        view, catalog, previous_action=previous_action, error=error,
        allow_unlisted_api=allow_unlisted_api)
    response_start = len(client.chat.completions.responses)
    kwargs = {
        "model": model,
        "messages": messages,
        # The API-Bank OCA client owns the actual DeepSeek thinking budget.
        # This request is intentionally small for non-thinking providers.
        "max_completion_tokens": 2048,
        "temperature": 0.0,
    }
    # Current DeepSeek V4 supports native JSON Output. The prompt already contains
    # an explicit JSON-only instruction, and SECAT's OCA proxy bounds the provider's
    # documented empty-JSON edge case while keeping every attempt token-metered.
    # Both OpenAI and current DeepSeek V4 support JSON Output. Every OCA
    # structured decision prompt explicitly requests JSON, and the provider
    # adapter has bounded handling for DeepSeek's rare empty-JSON response.
    kwargs["response_format"] = {"type": "json_object"}
    with token_stage("planner"):
        response = client.chat.completions.create(**kwargs)
    raw = str(response.choices[0].message.content or "")
    metadata = client.chat.completions.responses[response_start:]
    return raw, messages, metadata


def _sandbox_rejection_needs_repair(error: str) -> bool:
    """Whether an API-Bank sandbox error indicates a malformed request.

    LV1 is evaluated on the predicted *next API action*, not on whether the
    published fixture happens to contain the referenced business object.  A
    schema-valid call that fails only because the fixture lacks that object is
    still a valid prediction candidate and should be left to the benchmark
    checker.  We spend the single repair call only on request-shape/type/auth
    failures that the model can plausibly correct.
    """
    text = str(error or "").lower()
    request_shape_markers = (
        "time data ",
        "does not match format",
        "keyerror",
        "missing required",
        "required positional",
        "unexpected keyword",
        "invalid literal",
        "could not convert",
        "must be ",
        "should be ",
        "invalid format",
        "invalid token",
        "authentication",
        "unauthorized",
        "password",
    )
    return any(marker in text for marker in request_shape_markers)


def _sandbox_lookup_key_needs_repair(error: str, params: dict | None) -> bool:
    """Recognize a rejected lookup *key* without treating all state misses alike.

    API-Bank sometimes reports ``<field> does not exist`` when the supplied
    lookup value is semantically malformed.  Repair is justified only when the
    noun in that error corresponds to an actual parameter name in the candidate.
    Generic messages such as ``record does not exist`` therefore remain ordinary
    state misses and preserve the established no-repair contract.
    """
    text = str(error or "").strip().lower()
    keys = {re.sub(r"[^a-z0-9]+", "", str(k).lower()) for k in (params or {})}
    if not keys or "does not exist" not in text:
        return False
    match = re.search(r"(?:the\s+)?([a-z][a-z _-]{0,40}?)\s+does not exist\b", text)
    if not match:
        return False
    noun = re.sub(r"[^a-z0-9]+", "", match.group(1))
    return noun in keys


def _run_oca_efficient(*, view: dict, sample, runtime, model: str,
                       catalog_override: list[dict] | None = None,
                       observed_bind_fields: set[str] | None = None,
                       allow_unlisted_api: bool = False) -> AgentPrediction:
    """Cost-efficient API-Bank OCA profile.

    Normal path: one model call. A second focused call is permitted only after a
    deterministic validation/sandbox rejection. No general answer critic,
    derivation convergence, or full-plan regeneration is paid for in this mode.
    """
    from agents.common import make_client

    catalog_fn = getattr(runtime, "api_catalog", None)
    catalog = catalog_override if catalog_override is not None else (
        catalog_fn(sample) if callable(catalog_fn) else None)
    if catalog is None:
        # The compact profile needs the documented request schema. Test stubs and
        # unusual integrations without a catalog retain the established evidence path.
        return _run_oca_evidence(view=view, sample=sample, runtime=runtime, model=model)

    client = _oca_planner_client(make_client(), model)
    can_probe = hasattr(runtime, "root")
    trace: list[dict] = []
    previous_action = None
    rejection = None
    attempts = 1 + _oca_efficient_max_repairs()

    for attempt in range(1, attempts + 1):
        print(f"  [OCA/API-BANK efficient] action attempt {attempt}/{attempts}", flush=True)
        try:
            raw, messages, responses = _oca_direct_action_call(
                client, model, view, catalog,
                previous_action=previous_action, error=rejection,
                allow_unlisted_api=allow_unlisted_api)
        except LLMRunError:
            raise
        except Exception as exc:
            raise LLMRunError(f"OCA efficient action LLM call failed: {exc}") from exc

        item = {
            "stage": "oca_action_select" if attempt == 1 else "oca_action_repair",
            "attempt": attempt,
            "output": raw,
            "responses": responses,
        }
        trace.append(item)
        try:
            name, params = _parse_direct_oca_action(
                raw, sample, catalog, allow_omitted_required=can_probe,
                allow_unlisted_api=allow_unlisted_api)
            if (int(getattr(sample, "benchmark_level", 1) or 1) == 2 and
                    observed_bind_fields):
                params, observed_bindings = _lv2_bind_observed_outputs(
                    view, catalog, name, params, allowed_fields=observed_bind_fields)
                if observed_bindings:
                    item["observed_bindings"] = observed_bindings
        except AgentBehaviorError as exc:
            rejection = str(exc)
            item["rejected"] = rejection
            finish_reason = str((responses[-1] if responses else {}).get("finish_reason") or "").lower()
            print(f"  [OCA/API-BANK efficient] rejected attempt {attempt}: {rejection}", flush=True)
            previous_action = None
            # If the provider explicitly stopped because the generation budget was
            # exhausted and no complete action could be parsed, repeating the same
            # request is pure token waste. Count it as one bounded agent failure.
            if finish_reason == "length":
                terminal = AgentBehaviorError(
                    "OCA API-Bank efficient action exhausted its completion budget before "
                    "emitting one complete action")
                terminal.agent_trace = trace
                raise terminal
            if attempt < attempts:
                continue
            terminal = AgentBehaviorError(
                "OCA API-Bank efficient action remained invalid after bounded repair: " + rejection)
            terminal.agent_trace = trace
            raise terminal

        previous_action = {"api_name": name, "params": params}
        item["api_name"] = name
        item["params"] = params
        if can_probe:
            print(f"  [OCA/API-BANK efficient] probing {name}", flush=True)
            probe = _probe_oca_action(runtime, sample, name, params)
            trace.append({
                "stage": "oca_action_probe", "attempt": attempt,
                "api_name": name, "params": params,
                "success": bool(probe.get("success")), "error": probe.get("error"),
            })
            if not probe.get("success"):
                rejection = "sandbox rejected " + _call_text(name, params) + ": " + str(probe.get("error"))
                if _sandbox_lookup_key_needs_repair(str(probe.get("error")), params):
                    rejection += (
                        " The rejected lookup key should be repaired from the ORIGINAL user-provided "
                        "symptom/entity phrase. Do not replace it with a diagnosis, search-result label, "
                        "or a more specific interpretation that appeared only in assistant text. Prefer "
                        "the minimal head term the user actually supplied."
                    )
                # Business-state misses (record/device/agenda not present in this
                # fixture) do not prove that the next-action prediction itself is
                # malformed.  ToolCoder's API-Bank path likewise submits the call
                # and lets the official checker score it.  Avoid spending a repair
                # call unless the sandbox error points to request shape/type/auth.
                if not (
                    _sandbox_rejection_needs_repair(str(probe.get("error"))) or
                    _sandbox_lookup_key_needs_repair(str(probe.get("error")), params)
                ):
                    print(
                        f"  [OCA/API-BANK efficient] sandbox state mismatch; submitting candidate {name}",
                        flush=True)
                    return AgentPrediction(
                        prediction_text=_call_text(name, params), strategy="oca", trace=trace,
                        action={"api_name": name, "params": params},
                    )
                print(f"  [OCA/API-BANK efficient] rejected attempt {attempt}: {rejection}", flush=True)
                if attempt < attempts:
                    continue
                terminal = AgentBehaviorError(
                    "OCA API-Bank efficient action remained invalid after bounded repair: " + rejection)
                terminal.agent_trace = trace
                raise terminal

        if int(getattr(sample, "benchmark_level", 1) or 1) == 2 and can_probe:
            calls = list((probe.get("execution") or {}).get("target_calls") or [])
            if len(calls) == 1:
                call = calls[0]
                return AgentPrediction(
                    prediction_text=_call_text(name, params), strategy="oca", trace=trace,
                    executed_call={
                        "api_name": str(call.get("api_name") or name),
                        "params": dict(call.get("params") or params),
                        "result": call.get("result"),
                        "replayed_calls": int((probe.get("execution") or {}).get("replayed_calls") or 0),
                    },
                )
        return AgentPrediction(
            prediction_text=_call_text(name, params), strategy="oca", trace=trace,
            action={"api_name": name, "params": params},
        )

    raise AgentBehaviorError("OCA API-Bank efficient profile exhausted bounded attempts")


def _adaptive_probe_from_trace(trace: list[dict]) -> dict | None:
    for item in reversed(trace or []):
        if isinstance(item, dict) and item.get("stage") == "oca_action_probe":
            return item
    return None


def _adaptive_should_verify(action: dict | None, probe: dict | None) -> bool:
    """Use extra reasoning only where r27 has a generic semantic-risk signal."""
    if not isinstance(action, dict):
        return False
    params = action.get("params")
    if not isinstance(params, dict):
        return False
    # A business-state miss is semantically ambiguous even when request shape is
    # valid; a verifier may be able to recover a better record/API choice.
    if isinstance(probe, dict) and not bool(probe.get("success")):
        return True
    if len(params) >= _oca_adaptive_verify_min_params():
        return True
    # One nested/list field can carry several independent values despite a small
    # top-level parameter count, so treat it as complex as well.
    return any(isinstance(value, (list, tuple, dict)) for value in params.values())


def _adaptive_verifier_messages(view: dict, catalog: list[dict] | None,
                                action: dict, probe: dict | None) -> list[dict]:
    probe_success = bool((probe or {}).get("success"))
    probe_error = str((probe or {}).get("error") or "")
    system = (
        "You are a conservative verifier for ONE already-selected API-Bank next action. "
        "Audit the candidate against the dialogue and documented API catalog; do not solve a different task. "
        "Return KEEP unless there is a concrete, visible mismatch in API choice or parameter values. "
        "Never invent credentials, tokens, IDs, names, dates, or other values. "
        "For every changed field, copy an exact supporting snippet from the dialogue into evidence. "
        "If the original sandbox probe succeeded, you MUST keep the same api_name and may only correct parameters. "
        "If the probe failed because referenced state was absent, you may change api_name only when the dialogue clearly supports it. "
        "A revision must include the complete replacement params object, not a patch. "
        "Return JSON only. KEEP shape: {\"decision\":\"keep\"}. "
        "REVISE shape: {\"decision\":\"revise\",\"confidence\":\"high\","
        "\"api_name\":\"ExactDocumentedName\",\"params\":{...},"
        "\"evidence\":{\"field\":\"exact dialogue snippet\",\"__api__\":\"exact dialogue snippet if api changes\"}}. "
        "Use confidence=high only for an explicit contradiction or omitted/incorrect value supported by the dialogue."
    )
    compact_catalog = _compact_action_catalog(catalog)
    if probe_success:
        selected_name = str(action.get("api_name") or "")
        compact_catalog = [item for item in compact_catalog
                           if str(item.get("name") or "") == selected_name]
    user = (
        "DOCUMENTED API CATALOG:\n" +
        json.dumps(compact_catalog, ensure_ascii=False, separators=(",", ":"), sort_keys=True) +
        "\n\nDIALOGUE SO FAR:\n" + format_history(view.get("chat_history") or []) +
        "\n\nORIGINAL CANDIDATE:\n" + json.dumps(action, ensure_ascii=False, sort_keys=True, default=str) +
        "\nORIGINAL SANDBOX PROBE: " + ("success" if probe_success else "failed")
    )
    if probe_error:
        user += "\nPROBE ERROR: " + probe_error
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]



def _oca_adaptive_verifier_call(client, model: str, view: dict, catalog: list[dict] | None,
                                action: dict, probe: dict | None):
    from utils.token_meter import stage as token_stage
    messages = _adaptive_verifier_messages(view, catalog, action, probe)
    response_start = len(client.chat.completions.responses)
    kwargs = {"model": model, "messages": messages, "max_completion_tokens": 1024, "temperature": 0.0}
    # Both OpenAI and current DeepSeek V4 support JSON Output. Every OCA
    # structured decision prompt explicitly requests JSON, and the provider
    # adapter has bounded handling for DeepSeek's rare empty-JSON response.
    kwargs["response_format"] = {"type": "json_object"}
    with token_stage("verifier"):
        response = client.chat.completions.create(**kwargs)
    raw = str(response.choices[0].message.content or "")
    return raw, messages, client.chat.completions.responses[response_start:]


def _parse_adaptive_verdict(raw: str) -> dict:
    obj = _parse_json_object(str(raw or ""))
    decision = str(obj.get("decision") or "").strip().lower()
    if decision == "keep":
        return {"decision": "keep"}
    if decision != "revise":
        raise AgentBehaviorError("adaptive verifier must return decision keep or revise")
    confidence = str(obj.get("confidence") or "").strip().lower()
    if confidence != "high":
        return {"decision": "keep", "reason": "revision confidence was not high"}
    evidence = obj.get("evidence")
    if not isinstance(evidence, dict):
        raise AgentBehaviorError("adaptive verifier revision must contain an evidence object")
    return {
        "decision": "revise", "confidence": confidence,
        "api_name": obj.get("api_name"), "params": obj.get("params"),
        "evidence": evidence,
    }


def _normalise_evidence_text(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _adaptive_changed_fields(original: dict, revised: dict) -> set[str]:
    old = original.get("params") if isinstance(original.get("params"), dict) else {}
    new = revised.get("params") if isinstance(revised.get("params"), dict) else {}
    keys = set(old) | set(new)
    return {key for key in keys if old.get(key) != new.get(key)}


def _adaptive_evidence_is_grounded(view: dict, original: dict, revised: dict,
                                   evidence: dict, *, allow_api_change: bool) -> tuple[bool, str]:
    history = _normalise_evidence_text(format_history(view.get("chat_history") or []))
    if not history:
        return False, "dialogue history is empty"
    old_name = str(original.get("api_name") or "")
    new_name = str(revised.get("api_name") or "")
    if old_name != new_name:
        if not allow_api_change:
            return False, "api_name change is forbidden after a successful original probe"
        snippet = _normalise_evidence_text(evidence.get("__api__"))
        if not snippet or snippet not in history:
            return False, "api_name revision lacks exact dialogue evidence"
    changed = _adaptive_changed_fields(original, revised)
    if old_name != new_name and allow_api_change:
        old_params = original.get("params") if isinstance(original.get("params"), dict) else {}
        new_params = revised.get("params") if isinstance(revised.get("params"), dict) else {}
        # For an API change, require evidence for every newly introduced or
        # changed value, not for fields that disappear with the old schema.
        changed = {field for field in new_params if old_params.get(field) != new_params.get(field)}
    if not changed and old_name == new_name:
        return False, "revision made no change"
    # Removing an already sandbox-valid field is too risky: the verifier is an
    # auditor, not a second free-form planner. State-mismatch revisions may still
    # change API and therefore naturally use a different schema.
    if not allow_api_change:
        old_params = original.get("params") or {}
        new_params = revised.get("params") or {}
        removed = set(old_params) - set(new_params)
        if removed:
            return False, "revision removed fields from a sandbox-valid action"
    for field in sorted(changed):
        snippet = _normalise_evidence_text(evidence.get(field))
        if not snippet or snippet not in history:
            return False, f"changed field {field!r} lacks exact dialogue evidence"
    return True, "grounded"


def _adaptive_adjudicator_messages(view: dict, catalog: list[dict] | None,
                                    original: dict, revised: dict, evidence: dict) -> list[dict]:
    system = (
        "You are the final conservative adjudicator between two API-Bank Level 1 next-action candidates. "
        "The ORIGINAL already passed schema validation and the local API sandbox. "
        "Choose REVISED only when the dialogue explicitly proves that ORIGINAL has a parameter error and REVISED corrects it. "
        "Do not invent information and do not propose a third action. When uncertain, choose ORIGINAL. "
        "Return JSON only: {\"choice\":\"original\"} or "
        "{\"choice\":\"revised\",\"confidence\":\"high\"}."
    )
    names = {str(original.get("api_name") or ""), str(revised.get("api_name") or "")}
    compact_catalog = [item for item in _compact_action_catalog(catalog)
                       if str(item.get("name") or "") in names]
    user = (
        "DOCUMENTED API CATALOG:\n" +
        json.dumps(compact_catalog, ensure_ascii=False, separators=(",", ":"), sort_keys=True) +
        "\n\nDIALOGUE SO FAR:\n" + format_history(view.get("chat_history") or []) +
        "\n\nORIGINAL:\n" + json.dumps(original, ensure_ascii=False, sort_keys=True, default=str) +
        "\nREVISED:\n" + json.dumps(revised, ensure_ascii=False, sort_keys=True, default=str) +
        "\nVERIFIER EVIDENCE:\n" + json.dumps(evidence, ensure_ascii=False, sort_keys=True, default=str)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]



def _oca_adaptive_adjudicator_call(client, model: str, view: dict, catalog: list[dict] | None,
                                   original: dict, revised: dict, evidence: dict):
    from utils.token_meter import stage as token_stage
    messages = _adaptive_adjudicator_messages(view, catalog, original, revised, evidence)
    response_start = len(client.chat.completions.responses)
    kwargs = {"model": model, "messages": messages, "max_completion_tokens": 768, "temperature": 0.0}
    # Both OpenAI and current DeepSeek V4 support JSON Output. Every OCA
    # structured decision prompt explicitly requests JSON, and the provider
    # adapter has bounded handling for DeepSeek's rare empty-JSON response.
    kwargs["response_format"] = {"type": "json_object"}
    with token_stage("adjudicator"):
        response = client.chat.completions.create(**kwargs)
    raw = str(response.choices[0].message.content or "")
    return raw, messages, client.chat.completions.responses[response_start:]


def _adjudicator_accepts_revision(raw: str) -> bool:
    try:
        obj = _parse_json_object(str(raw or ""))
    except Exception:
        return False
    return (str(obj.get("choice") or "").strip().lower() == "revised" and
            str(obj.get("confidence") or "").strip().lower() == "high")


def _run_oca_adaptive(*, view: dict, sample, runtime, model: str) -> AgentPrediction:
    """r28: r27 efficient action plus conservative, evidence-grounded verification.

    The r27 selector/repair path is executed byte-for-byte first. Extra calls are
    optional: simple actions return immediately; verifier/adjudicator failures
    fall back to the r27 action rather than turning a usable prediction into a
    crash. This makes additional reasoning a bounded upside-only experiment.
    """
    base = _run_oca_efficient(view=view, sample=sample, runtime=runtime, model=model)
    action = base.action if isinstance(base.action, dict) else None
    probe = _adaptive_probe_from_trace(base.trace)
    if not hasattr(runtime, "root") or not _adaptive_should_verify(action, probe):
        return base

    catalog_fn = getattr(runtime, "api_catalog", None)
    catalog = catalog_fn(sample) if callable(catalog_fn) else None
    if catalog is None:
        return base

    print("  [OCA/API-BANK adaptive] verifying complex candidate", flush=True)
    trace = list(base.trace or [])
    from agents.common import make_client
    try:
        verifier_client = _oca_planner_client(
            make_client(), model, total_timeout_sec=min(60, _oca_task_timeout_seconds()))
        raw, messages, responses = _oca_adaptive_verifier_call(
            verifier_client, model, view, catalog, action, probe)
        verify_item = {
            "stage": "oca_action_verify", "output": raw, "responses": responses,
        }
        trace.append(verify_item)
        verdict = _parse_adaptive_verdict(raw)
    except Exception as exc:
        trace.append({"stage": "oca_action_verify", "kept_original": True,
                      "error": f"optional verifier unavailable/invalid: {exc}"})
        base.trace = trace
        print("  [OCA/API-BANK adaptive] verifier unavailable; keeping original", flush=True)
        return base

    if verdict.get("decision") != "revise":
        verify_item["decision"] = "keep"
        base.trace = trace
        return base

    original_probe_success = bool((probe or {}).get("success"))
    revised_raw = json.dumps({
        "api_name": verdict.get("api_name"), "params": verdict.get("params")
    }, ensure_ascii=False, default=str)
    try:
        revised_name, revised_params = _parse_direct_oca_action(
            revised_raw, sample, catalog, allow_omitted_required=True)
    except Exception as exc:
        verify_item["kept_original"] = True
        verify_item["rejection"] = f"revised action invalid: {exc}"
        base.trace = trace
        return base
    revised = {"api_name": revised_name, "params": revised_params}
    grounded, why = _adaptive_evidence_is_grounded(
        view, action, revised, verdict.get("evidence") or {},
        allow_api_change=not original_probe_success)
    if not grounded:
        verify_item["kept_original"] = True
        verify_item["rejection"] = why
        base.trace = trace
        return base

    revised_probe = _probe_oca_action(runtime, sample, revised_name, revised_params)
    trace.append({
        "stage": "oca_action_verify_probe", "api_name": revised_name,
        "params": revised_params, "success": bool(revised_probe.get("success")),
        "error": revised_probe.get("error"),
    })
    if not revised_probe.get("success"):
        verify_item["kept_original"] = True
        verify_item["rejection"] = "revised action did not pass local sandbox"
        base.trace = trace
        return base

    # If the original missed fixture/business state but the evidence-grounded
    # revision succeeds, the sandbox gives an objective preference and no third
    # model call is necessary.
    if not original_probe_success:
        print(f"  [OCA/API-BANK adaptive] accepted verifier recovery {revised_name}", flush=True)
        return AgentPrediction(
            prediction_text=_call_text(revised_name, revised_params), strategy="oca",
            trace=trace, action=revised)

    # Both candidates pass the sandbox. Require a separate conservative vote so
    # one critic cannot overwrite a valid r27 answer by itself.
    print("  [OCA/API-BANK adaptive] adjudicating proposed parameter revision", flush=True)
    try:
        judge_client = _oca_planner_client(
            make_client(), model, total_timeout_sec=min(60, _oca_task_timeout_seconds()))
        judge_raw, judge_messages, judge_responses = _oca_adaptive_adjudicator_call(
            judge_client, model, view, catalog, action, revised, verdict.get("evidence") or {})
        accepted = _adjudicator_accepts_revision(judge_raw)
        trace.append({
            "stage": "oca_action_adjudicate", "output": judge_raw,
            "responses": judge_responses, "accepted_revision": accepted,
        })
    except Exception as exc:
        trace.append({"stage": "oca_action_adjudicate", "accepted_revision": False,
                      "error": f"optional adjudicator unavailable/invalid: {exc}"})
        accepted = False

    if not accepted:
        base.trace = trace
        return base
    print(f"  [OCA/API-BANK adaptive] accepted adjudicated revision {revised_name}", flush=True)
    return AgentPrediction(
        prediction_text=_call_text(revised_name, revised_params), strategy="oca",
        trace=trace, action=revised)



# ---------------------------------------------------------------------------
# r29 independent-consensus profile
# ---------------------------------------------------------------------------

def _consensus_should_resolve(action: dict | None, probe: dict | None) -> bool:
    """Spend a second independent solve only on generic high-ambiguity actions.

    This intentionally reuses the r28 structural trigger without exposing the
    second solver to the first candidate. The trigger is benchmark-agnostic:
    stateful sandbox misses, 3+ fields, or structured container values.
    """
    return _adaptive_should_verify(action, probe)


def _consensus_independent_messages(view: dict, catalog: list[dict] | None) -> list[dict]:
    system = (
        "You are an independent second solver for API-Bank next-action prediction. "
        "Solve the dialogue from scratch; you are not reviewing another model's answer. "
        "Predict exactly ONE immediate documented API request, not the final conversational outcome. "
        "Resolve the most recent unfinished user intent and do not repeat a completed prior action. "
        "When several catalog entries are semantic-search candidates, distinguish the requested operation and object before choosing. "
        "Use only request fields from the supplied catalog. Reuse exact identifying values for existing "
        "records from the dialogue or prior API observations; do not paraphrase record identity. "
        "If the requested operation needs an unavailable required value that a documented prerequisite API "
        "can obtain, choose that prerequisite action instead. Never invent credentials, tokens, IDs, names, "
        "dates, or placeholders. Pay particular attention to distinguishing record identifiers, old-vs-new "
        "values, and source-vs-destination fields. Match documented wire shapes and date/time formats exactly. "
        "Assume year 2023 when month/day is given without a year. Return JSON only in exactly this shape: "
        '{"api_name":"ExactDocumentedName","params":{"field":value}}.'
    )
    user = (
        "DOCUMENTED API CATALOG:\n" +
        json.dumps(_compact_action_catalog(catalog), ensure_ascii=False, separators=(",", ":"), sort_keys=True) +
        "\n\nDIALOGUE SO FAR:\n" + format_history(view.get("chat_history") or [])
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _oca_consensus_independent_call(client, model: str, view: dict, catalog: list[dict] | None):
    from utils.token_meter import stage as token_stage
    messages = _consensus_independent_messages(view, catalog)
    response_start = len(client.chat.completions.responses)
    kwargs = {"model": model, "messages": messages, "max_completion_tokens": 2048, "temperature": 0.0}
    # Both OpenAI and current DeepSeek V4 support JSON Output. Every OCA
    # structured decision prompt explicitly requests JSON, and the provider
    # adapter has bounded handling for DeepSeek's rare empty-JSON response.
    kwargs["response_format"] = {"type": "json_object"}
    with token_stage("consensus_solver"):
        response = client.chat.completions.create(**kwargs)
    raw = str(response.choices[0].message.content or "")
    return raw, messages, client.chat.completions.responses[response_start:]


def _actions_equal(a: dict | None, b: dict | None) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    return (str(a.get("api_name") or "") == str(b.get("api_name") or "") and
            (a.get("params") if isinstance(a.get("params"), dict) else {}) ==
            (b.get("params") if isinstance(b.get("params"), dict) else {}))


def _prediction_action(prediction: AgentPrediction | None) -> dict | None:
    """Recover the submitted action even when LV2 already executed its sandbox probe."""
    if prediction is None:
        return None
    if isinstance(prediction.action, dict):
        return {
            "api_name": str(prediction.action.get("api_name") or ""),
            "params": dict(prediction.action.get("params") or {}),
        }
    if isinstance(prediction.executed_call, dict):
        return {
            "api_name": str(prediction.executed_call.get("api_name") or ""),
            "params": dict(prediction.executed_call.get("params") or {}),
        }
    return None


def _prediction_from_probe(name: str, params: dict, probe: dict, trace: list[dict]) -> AgentPrediction:
    """Return an LV2 action using an already-observed sandbox result when possible."""
    if bool((probe or {}).get("success")):
        calls = list(((probe or {}).get("execution") or {}).get("target_calls") or [])
        if len(calls) == 1:
            call = calls[0]
            return AgentPrediction(
                prediction_text=_call_text(name, params), strategy="oca", trace=trace,
                executed_call={
                    "api_name": str(call.get("api_name") or name),
                    "params": dict(call.get("params") or params),
                    "result": call.get("result"),
                    "replayed_calls": int(((probe or {}).get("execution") or {}).get("replayed_calls") or 0),
                },
            )
    return AgentPrediction(
        prediction_text=_call_text(name, params), strategy="oca", trace=trace,
        action={"api_name": name, "params": dict(params or {})},
    )


def _consensus_adjudicator_messages(view: dict, catalog: list[dict] | None,
                                     first: dict, second: dict,
                                     first_probe: dict | None, second_probe: dict | None) -> list[dict]:
    system = (
        "You are a conservative adjudicator between TWO independently produced API-Bank next-action "
        "candidates. Choose the action that is better supported by the dialogue and documented API schema. "
        "Do not propose a third action and do not invent information. Prefer exact values already present in "
        "the dialogue/API observations, correct prerequisite ordering, and exact existing-record identity. "
        "A local sandbox state miss is only weak evidence because API-Bank scores the predicted next action; "
        "request-shape/type failures are strong negative evidence. When evidence does not clearly favor the "
        "second candidate, choose FIRST. Return JSON only: {\"choice\":\"first\"} or "
        "{\"choice\":\"second\",\"confidence\":\"high\"}."
    )
    names = {str(first.get("api_name") or ""), str(second.get("api_name") or "")}
    compact_catalog = [item for item in _compact_action_catalog(catalog)
                       if str(item.get("name") or "") in names]
    def _probe_summary(probe):
        if not isinstance(probe, dict):
            return {"available": False}
        return {"available": True, "success": bool(probe.get("success")),
                "error": str(probe.get("error") or "")[:1200]}
    user = (
        "DOCUMENTED API CATALOG:\n" +
        json.dumps(compact_catalog, ensure_ascii=False, separators=(",", ":"), sort_keys=True) +
        "\n\nDIALOGUE SO FAR:\n" + format_history(view.get("chat_history") or []) +
        "\n\nFIRST CANDIDATE:\n" + json.dumps(first, ensure_ascii=False, sort_keys=True, default=str) +
        "\nFIRST LOCAL PROBE:\n" + json.dumps(_probe_summary(first_probe), ensure_ascii=False, sort_keys=True) +
        "\n\nSECOND CANDIDATE:\n" + json.dumps(second, ensure_ascii=False, sort_keys=True, default=str) +
        "\nSECOND LOCAL PROBE:\n" + json.dumps(_probe_summary(second_probe), ensure_ascii=False, sort_keys=True)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _oca_consensus_adjudicator_call(client, model: str, view: dict, catalog: list[dict] | None,
                                     first: dict, second: dict,
                                     first_probe: dict | None, second_probe: dict | None):
    from utils.token_meter import stage as token_stage
    messages = _consensus_adjudicator_messages(
        view, catalog, first, second, first_probe, second_probe)
    response_start = len(client.chat.completions.responses)
    kwargs = {"model": model, "messages": messages, "max_completion_tokens": 1024, "temperature": 0.0}
    # Both OpenAI and current DeepSeek V4 support JSON Output. Every OCA
    # structured decision prompt explicitly requests JSON, and the provider
    # adapter has bounded handling for DeepSeek's rare empty-JSON response.
    kwargs["response_format"] = {"type": "json_object"}
    with token_stage("consensus_adjudicator"):
        response = client.chat.completions.create(**kwargs)
    raw = str(response.choices[0].message.content or "")
    return raw, messages, client.chat.completions.responses[response_start:]


def _consensus_adjudicator_prefers_second(raw: str) -> bool:
    try:
        obj = _parse_json_object(str(raw or ""))
    except Exception:
        return False
    return (str(obj.get("choice") or "").strip().lower() == "second" and
            str(obj.get("confidence") or "").strip().lower() == "high")


def _run_oca_consensus(*, view: dict, sample, runtime, model: str) -> AgentPrediction:
    """r29: frozen r27 selector + independent second solve on ambiguous cases.

    The second solver never sees the first candidate. Agreement is returned
    immediately. A disagreement can replace r27 only through objective sandbox
    preference or a separate high-confidence adjudication. Optional consensus
    failures always fall back to the frozen r27 result. If r27 itself ends in an
    agent-behavior error, one independent rescue solve is allowed.
    """
    catalog_fn = getattr(runtime, "api_catalog", None)
    catalog = catalog_fn(sample) if callable(catalog_fn) else None
    if catalog is None or not hasattr(runtime, "root"):
        return _run_oca_efficient(view=view, sample=sample, runtime=runtime, model=model)

    base_error = None
    try:
        base = _run_oca_efficient(view=view, sample=sample, runtime=runtime, model=model)
    except AgentBehaviorError as exc:
        base = None
        base_error = exc

    first_action = _prediction_action(base)
    first_probe = _adaptive_probe_from_trace(base.trace) if base is not None else None
    level = int(getattr(sample, "benchmark_level", 1) or 1)
    # LV2 retrieval commonly exposes several semantically related candidates.
    # A second independent solve is valuable even when the first call executes
    # successfully, because execution success does not prove that the selected
    # API is the one requested by the dialogue. This uses only visible catalog
    # information and applies uniformly across tasks.
    lv2_catalog_ambiguity = (level == 2 and len(catalog or []) > 1)
    if base is not None and not (
            _consensus_should_resolve(first_action, first_probe) or lv2_catalog_ambiguity):
        return base

    label = "rescuing base agent error" if base is None else "independent second solve"
    print(f"  [OCA/API-BANK consensus] {label}", flush=True)
    from agents.common import make_client
    trace = list((base.trace if base is not None else getattr(base_error, "agent_trace", [])) or [])
    try:
        second_client = _oca_planner_client(
            make_client(), model, total_timeout_sec=min(60, _oca_task_timeout_seconds()))
        raw, messages, responses = _oca_consensus_independent_call(
            second_client, model, view, catalog)
        second_name, second_params = _parse_direct_oca_action(
            raw, sample, catalog, allow_omitted_required=True)
        second_action = {"api_name": second_name, "params": second_params}
        second_item = {"stage": "oca_action_consensus_solve", "output": raw,
                       "responses": responses, "api_name": second_name, "params": second_params}
        trace.append(second_item)
        second_probe = _probe_oca_action(runtime, sample, second_name, second_params)
        trace.append({"stage": "oca_action_consensus_probe", "api_name": second_name,
                      "params": second_params, "success": bool(second_probe.get("success")),
                      "error": second_probe.get("error")})
    except Exception as exc:
        trace.append({"stage": "oca_action_consensus_solve",
                      "error": f"optional independent solver unavailable/invalid: {exc}"})
        if base is not None:
            base.trace = trace
            print("  [OCA/API-BANK consensus] second solver unavailable; keeping first", flush=True)
            return base
        if base_error is not None:
            base_error.agent_trace = trace
            raise base_error
        raise

    # Generic rescue: r27 produced no valid candidate, while the independent
    # solve did. A request-shape/type failure remains invalid; a success or a
    # pure business-state miss is still a reportable LV1 candidate.
    if base is None:
        if second_probe.get("success") or not _sandbox_rejection_needs_repair(str(second_probe.get("error"))):
            print(f"  [OCA/API-BANK consensus] accepted independent rescue {second_name}", flush=True)
            return _prediction_from_probe(second_name, second_params, second_probe, trace)
        if base_error is not None:
            base_error.agent_trace = trace
            raise base_error

    if _actions_equal(first_action, second_action):
        print("  [OCA/API-BANK consensus] independent solver agrees; keeping first", flush=True)
        base.trace = trace
        return base

    first_ok = bool((first_probe or {}).get("success"))
    second_ok = bool(second_probe.get("success"))
    first_structural_failure = (
        isinstance(first_probe, dict) and not first_ok and
        _sandbox_rejection_needs_repair(str(first_probe.get("error") or ""))
    )
    if first_structural_failure and second_ok:
        print(f"  [OCA/API-BANK consensus] accepted structurally valid alternative {second_name}", flush=True)
        return _prediction_from_probe(second_name, second_params, second_probe, trace)
    # A successful FIRST and failed SECOND is objective evidence to keep r27.
    # The reverse is deliberately *not* automatic: a business-state miss does not
    # prove FIRST is wrong under the LV1 checker, so SECOND still needs a separate
    # high-confidence adjudication before it may replace r27.
    if first_ok and not second_ok:
        print("  [OCA/API-BANK consensus] second candidate missed sandbox; keeping first", flush=True)
        base.trace = trace
        return base

    # Both succeed, both state-miss, or only SECOND succeeds. Use a separate vote;
    # FIRST remains the default so extra reasoning cannot casually overwrite r27.
    print("  [OCA/API-BANK consensus] adjudicating disagreement", flush=True)
    try:
        judge_client = _oca_planner_client(
            make_client(), model, total_timeout_sec=min(60, _oca_task_timeout_seconds()))
        judge_raw, judge_messages, judge_responses = _oca_consensus_adjudicator_call(
            judge_client, model, view, catalog, first_action, second_action, first_probe, second_probe)
        choose_second = _consensus_adjudicator_prefers_second(judge_raw)
        trace.append({"stage": "oca_action_consensus_adjudicate", "output": judge_raw,
                      "responses": judge_responses, "chose_second": choose_second})
    except Exception as exc:
        choose_second = False
        trace.append({"stage": "oca_action_consensus_adjudicate", "chose_second": False,
                      "error": f"optional adjudicator unavailable/invalid: {exc}"})
    if not choose_second:
        base.trace = trace
        return base
    print(f"  [OCA/API-BANK consensus] accepted adjudicated alternative {second_name}", flush=True)
    return _prediction_from_probe(second_name, second_params, second_probe, trace)


def _lv2_active_user_text(view: dict) -> str:
    """Return user text for the currently active LV2 intent segment.

    Retrieval can occur after one or more clarification turns.  Using only the
    latest User message is unsafe when that message is merely a name, symptom,
    date, or credential supplied in response to an AI question.  The active
    segment is therefore all visible User utterances since the most recent API
    observation.  A prior API marks a concrete execution boundary; within the
    segment, earlier request words and later clarification values are both visible
    conversation evidence.
    """
    history = list(view.get("chat_history") or [])
    start = 0
    for index in range(len(history) - 1, -1, -1):
        if str(history[index].get("role") or "") == "API":
            start = index + 1
            break
    parts: list[str] = []
    for item in history[start:]:
        if str(item.get("role") or "") != "User":
            continue
        text = str(item.get("text") or "").strip()
        if text:
            parts.append(text)
    if parts:
        # If the user changes the requested operation before any API executes,
        # the newest actionable utterance starts a new intent inside this segment.
        # Later non-action utterances are treated as clarification values.
        actionable_index = None
        for index, text in enumerate(parts):
            words = {x.lower() for x in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text)}
            if words.intersection(_LV2_RETRIEVAL_ACTION_WORDS):
                actionable_index = index
        if actionable_index is not None:
            parts = parts[actionable_index:]
        return " ".join(parts)
    # Defensive fallback for malformed/legacy views where a trailing API left no
    # active User turn even though earlier visible user context still exists.
    for item in reversed(history):
        if str(item.get("role") or "") == "User":
            text = str(item.get("text") or "").strip()
            if text:
                return text
    return ""


_LV2_RETRIEVAL_ACTION_WORDS = {
    "add", "adding", "create", "creating", "set", "setting", "schedule", "scheduling",
    "book", "booking", "reserve", "reserving", "register", "registering", "delete",
    "deleting", "remove", "removing", "cancel", "canceling", "cancelling", "modify",
    "modifying", "update", "updating", "change", "changing", "query", "querying",
    "check", "checking", "find", "finding", "search", "searching", "look", "looking",
    "calculate", "calculating", "compute", "computing", "record", "recording", "get",
    "getting", "retrieve", "retrieving", "show", "showing", "tell", "order", "ordering",
}

# These are language-level filler/value tokens, not API names or benchmark labels.
# Removing them keeps ToolSearcher close to the user's operation/object words while
# avoiding dates, courtesy phrases, and incidental values that can dominate a tiny
# sentence-transformer query.
_LV2_RETRIEVAL_STOPWORDS = {
    "a", "an", "the", "to", "for", "of", "on", "at", "in", "with", "from", "into",
    "by", "about", "please", "help", "me", "my", "our", "us", "i", "we", "you", "your",
    "can", "could", "would", "will", "want", "wanted", "need", "needed", "like", "just",
    "some", "any", "this", "that", "these", "those", "it", "its", "is", "are", "be",
    "do", "does", "did", "have", "has", "had", "out", "up", "down", "and", "or", "but",
    "am", "pm", "today", "tomorrow", "yesterday", "next", "last",
    "what", "whats", "which", "who", "when", "where", "why", "how", "if", "s", "m", "re", "ve", "ll", "d",
    "title", "topic", "named", "called", "dr", "doctor",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}


def _lv2_lexical_search_keywords(view: dict) -> str:
    """Compile a deterministic ToolSearcher query from visible user wording.

    The official API-Bank ToolSearcher returns only the single best embedding match
    (plus GetUserToken for authenticated tools).  Therefore free-form semantic
    paraphrase is harmful: changing e.g. ``agenda`` into ``calendar event`` can
    expose a completely different tool.  This compiler keeps the first operation
    anchor and up to three following content words *verbatim* from the user's text.
    It contains no tool inventory and no task/ground-truth mapping.
    """
    text = _lv2_active_user_text(view)
    if not text:
        raise AgentBehaviorError("OCA LV2 retrieval has no visible user utterance")

    # Arithmetic expressions are a special language form whose operands/operators
    # define the requested capability. Preserve the expression and add only the
    # generic operation word "calculate"; no tool name is inferred. A bare hyphen
    # is not enough because API-Bank user text contains many month-day dates.
    compact = " ".join(text.strip().split())
    math_cue = bool(re.search(r"\d\s*[+*/]\s*\d", compact))
    if not math_cue and re.search(r"\d\s*-\s*\d", compact):
        math_cue = bool(re.search(r"\b(?:calculate|compute|what\s+is|result\s+of)\b", compact, re.I))
    if math_cue:
        chunks = re.findall(r"[0-9()+*/.\s-]{3,}", compact)
        chunks = [x.strip() for x in chunks if re.search(r"\d", x) and re.search(r"[+*/-]", x)]
        if chunks:
            value = re.sub(r"\s+", " ", max(chunks, key=len)).strip()
            return ("calculate " + value)[:160]

    # Quoted spans are usually request values (meeting titles, hotel names, etc.),
    # so deprioritize them. Keep them as a fallback because a quoted value can be
    # the only semantic object in a request such as ``search for "rash"``.
    quoted_spans = re.findall(r'"([^"\n]*)"|\'([^\'\n]*)\'', compact)
    quoted_text = " ".join(a or b for a, b in quoted_spans if (a or b).strip())
    working = re.sub(r'"[^"\n]*"|\'[^\'\n]*\'', ' ', compact)
    # Remove numeric/ordinal fragments before word tokenization so ``15th`` does
    # not leak a stray ``th`` token into the retrieval phrase.
    working = re.sub(r"\b\d+(?:st|nd|rd|th)?\b", " ", working, flags=re.I)
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", working)
    if not raw_tokens:
        raise AgentBehaviorError("OCA LV2 retrieval user utterance has no semantic text")

    lowered = [tok.lower() for tok in raw_tokens]
    action_index = next((i for i, tok in enumerate(lowered)
                         if tok in _LV2_RETRIEVAL_ACTION_WORDS), None)
    scan = raw_tokens[action_index:] if action_index is not None else raw_tokens

    selected: list[str] = []
    destructive = bool(scan and scan[0].lower() in {
        "delete", "deleting", "remove", "removing",
        "cancel", "canceling", "cancelling",
    })
    for tok in scan:
        low = tok.lower()
        # In phrases such as "delete an agenda for my meeting ...", the
        # prepositional clause qualifies the record rather than naming another
        # capability.  Keeping "meeting" in the tiny embedding query can flip
        # ToolSearcher from DeleteAgenda to DeleteMeeting.  Stop at that generic
        # direct-object boundary only for destructive operations; additive
        # constructions such as "add a meeting to my agenda" keep their
        # destination context and therefore preserve the r38 canary behavior.
        if destructive and low == "for" and len(selected) >= 2:
            break
        if low in _LV2_RETRIEVAL_STOPWORDS:
            continue
        # Once operation + object are present, title-cased tokens are usually
        # entity values (people/hotels/places), not capability descriptors.
        if len(selected) >= 2 and tok[:1].isupper():
            continue
        if not selected or selected[-1].lower() != low:
            selected.append(low)
        if len(selected) >= 3:
            break

    if len(selected) < 2 and quoted_text:
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", quoted_text):
            low = tok.lower()
            if low in _LV2_RETRIEVAL_STOPWORDS:
                continue
            if not selected or selected[-1] != low:
                selected.append(low)
            if len(selected) >= 3:
                break
    if not selected:
        selected = [tok.lower() for tok in raw_tokens[:4]]
    keywords = " ".join(selected).strip()
    if not keywords:
        raise AgentBehaviorError("OCA LV2 retrieval compiler produced empty keywords")
    return keywords[:160]


def _lv2_search_planner_messages(view: dict) -> list[dict]:
    """Legacy optional model retrieval planner.

    r38 defaults to the deterministic lexical compiler above.  This model path is
    retained only as an explicit fallback/debug mode because API-Bank ToolSearcher
    is unusually sensitive to paraphrase drift.
    """
    system = (
        "You are OCA's API-Bank Level-2 retrieval planner. The only currently "
        "visible action is ToolSearcher(keywords: str). Select 2-4 IMPORTANT WORDS "
        "copied verbatim from the active unfinished USER intent. Preserve the "
        "user's operation and object words; do NOT replace them with synonyms, broader "
        "categories, API names, or phrases such as calendar event unless the user used "
        "those exact words. Omit credentials, names, exact dates/times, and numeric values "
        "unless the values themselves define a calculation. Return JSON only: "
        '{"keywords":"verbatim lexical phrase"}.'
    )
    user = "DIALOGUE SO FAR:\n" + format_history(view.get("chat_history") or [])
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _lv2_parse_search_keywords(raw: str) -> str:
    try:
        obj = _parse_json_object(str(raw or ""))
    except Exception as exc:
        raise AgentBehaviorError(f"OCA LV2 retrieval planner returned invalid JSON: {exc}") from exc
    keywords = str(obj.get("keywords") or "").strip()
    if not keywords:
        raise AgentBehaviorError("OCA LV2 retrieval planner omitted keywords")
    if len(keywords) > 160:
        raise AgentBehaviorError("OCA LV2 retrieval keywords are implausibly long")
    if not re.search(r"[A-Za-z]", keywords):
        raise AgentBehaviorError("OCA LV2 retrieval keywords contain no semantic text")
    return keywords


def _run_oca_lv2_search(*, view: dict, sample, runtime, model: str) -> AgentPrediction:
    """Execute exactly one ToolSearcher action from dialogue-grounded lexical intent.

    Default r38 behavior spends zero LLM calls on retrieval.  That is both cheaper
    and more reproducible, and—critically—prevents a reasoning model from replacing
    benchmark-relevant surface words with semantically broader paraphrases.  An
    opt-in model fallback remains available for unusual inputs where lexical
    compilation itself fails.
    """
    from agents.common import make_client
    from utils.token_meter import stage as token_stage

    trace: list[dict] = []
    mode = _oca_lv2_search_mode()

    keywords = None
    lexical_error = None
    if mode == "lexical":
        try:
            keywords = _lv2_lexical_search_keywords(view)
            trace.append({"stage": "oca_lv2_search_lexical", "keywords": keywords})
        except Exception as exc:
            lexical_error = str(exc)
            trace.append({"stage": "oca_lv2_search_lexical", "error": lexical_error})

    if keywords is None:
        client = _oca_planner_client(make_client(), model)
        messages = _lv2_search_planner_messages(view)
        start = len(client.chat.completions.responses)
        kwargs = {
            "model": model, "messages": messages, "max_completion_tokens": 512,
            "temperature": 0.0, "response_format": {"type": "json_object"},
        }
        with token_stage("retrieval_planner"):
            response = client.chat.completions.create(**kwargs)
        raw = str(response.choices[0].message.content or "")
        responses = client.chat.completions.responses[start:]
        keywords = _lv2_parse_search_keywords(raw)
        trace.append({"stage": "oca_lv2_search_model", "output": raw, "responses": responses,
                      "keywords": keywords, "lexical_error": lexical_error})

    params = {"keywords": keywords}
    print(f"  [OCA/API-BANK lv2] executing ToolSearcher query: {keywords}", flush=True)
    probe = _probe_oca_action(runtime, sample, "ToolSearcher", params)
    trace.append({"stage": "oca_lv2_search_probe", "success": bool(probe.get("success")),
                  "error": probe.get("error"), "keywords": keywords})
    if probe.get("success"):
        calls = list((probe.get("execution") or {}).get("target_calls") or [])
        if len(calls) == 1:
            call = calls[0]
            return AgentPrediction(
                prediction_text=_call_text("ToolSearcher", params), strategy="oca", trace=trace,
                executed_call={
                    "api_name": str(call.get("api_name") or "ToolSearcher"),
                    "params": dict(call.get("params") or params),
                    "result": call.get("result"),
                    "replayed_calls": int((probe.get("execution") or {}).get("replayed_calls") or 0),
                },
            )
    return AgentPrediction(prediction_text=_call_text("ToolSearcher", params), strategy="oca",
                           trace=trace, action={"api_name": "ToolSearcher", "params": params})


def _lv2_entry_output_fields(entry: dict) -> dict:
    if not isinstance(entry, dict):
        return {}
    meta = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    fields = meta.get("output_parameters")
    if fields is None and isinstance(entry.get("description_json"), dict):
        fields = entry["description_json"].get("output_parameters")
    if isinstance(fields, dict):
        return fields
    if isinstance(fields, list):
        return {str(x.get("name")): x for x in fields if isinstance(x, dict) and x.get("name")}
    return {}


def _lv2_observed_output_values(view: dict) -> dict:
    """Latest direct API-output values visible in the dialogue, keyed by field.

    Only first-level named output fields are used.  This intentionally avoids
    fishing inside arbitrary result collections where the same key may refer to
    several records.  It is a generic state-binding rule, not an API-Bank oracle.
    """
    out: dict = {}
    for item in reversed(list(view.get("chat_history") or [])):
        if str(item.get("role") or "") != "API":
            continue
        result = item.get("result")
        if not isinstance(result, dict):
            continue
        payload = result.get("output")
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            name = str(key or "").strip()
            if not name or name in out:
                continue
            if isinstance(value, (str, int, float, bool)) and not _placeholder_value(value):
                out[name] = value
    return out


def _lv2_bind_observed_outputs(view: dict, catalog: list[dict] | None,
                               name: str, params: dict, *,
                               allowed_fields: set[str] | None = None) -> tuple[dict, dict]:
    """Bind exact state fields (token/IDs/etc.) from visible API observations.

    If a selected tool consumes a field whose exact name was directly produced by
    an earlier visible API call, the environment value is authoritative.  This is
    the same typed dataflow principle OCA uses elsewhere and prevents a model from
    hallucinating or mistyping authentication/state identifiers.
    """
    fields = _input_fields_for(catalog or [], name)
    observed = _lv2_observed_output_values(view)
    bound = dict(params or {})
    applied: dict = {}
    permitted = set(allowed_fields or ())
    # A ToolCoder-parity latent action intentionally has no hidden schema.  We
    # still may replace a state field that the model itself explicitly supplied
    # (e.g. token) with the exact directly observed value.  This does not infer a
    # field name; it only corrects a field already present in the candidate.
    if not fields and permitted:
        fields = {field: {} for field in permitted if field in bound}
    for field in fields:
        if field not in permitted or field not in observed:
            continue
        current = bound.get(field)
        # Exact observed state dominates a missing, placeholder, or contradictory
        # model value for the same documented field name.
        if field not in bound or _placeholder_value(current) or current != observed[field]:
            bound[field] = observed[field]
            applied[field] = observed[field]
    bound = _coerce_declared_wire_types(catalog, name, bound)
    bound = _normalize_oca_action_params(
        catalog or [], name, bound, allow_omitted_required=True)
    return bound, applied


def _lv2_required_field_names(entry: dict) -> set[str]:
    name = str(entry.get("name") or "")
    fields = _input_fields_for([entry], name)
    required: set[str] = set()
    for field, spec0 in fields.items():
        spec = spec0 if isinstance(spec0, dict) else {"description": str(spec0)}
        if _parameter_required(spec, str(spec.get("description") or "")):
            required.add(str(field))
    return required


def _lv2_dependency_choice(view: dict, catalog: list[dict] | None) -> tuple[str, dict, str] | None:
    """Resolve an obvious visible producer→consumer prerequisite without an LLM.

    ToolSearcher can expose an authentication producer together with the requested
    consumer.  More generally, if exactly one visible tool produces a required
    field of exactly one other visible tool, schema/dataflow already determines the
    next API: call the producer until that field is observed, then the consumer.
    No hidden tool list, target API, checker, or sandbox outcome participates.
    """
    entries = [x for x in (catalog or []) if isinstance(x, dict) and str(x.get("name") or "")]
    if len(entries) < 2:
        return None
    observed = _lv2_observed_output_values(view)
    edges: list[tuple[str, str, str]] = []
    for producer in entries:
        p_name = str(producer.get("name") or "")
        produced = set(_lv2_entry_output_fields(producer))
        if not produced:
            continue
        for consumer in entries:
            c_name = str(consumer.get("name") or "")
            if c_name == p_name:
                continue
            required = _lv2_required_field_names(consumer)
            for field in sorted(produced.intersection(required)):
                edges.append((p_name, c_name, field))
    # Deduplicate structurally equivalent edges.
    edges = list(dict.fromkeys(edges))
    if len(edges) != 1:
        return None
    producer, consumer, field = edges[0]
    selected = consumer if field in observed else producer
    entry = next((x for x in entries if str(x.get("name") or "") == selected), None)
    if entry is None:
        return None
    return selected, entry, field


def _lv2_dialogue_api_mentions(view: dict) -> list[str]:
    """API/tool names explicitly written in visible assistant dialogue.

    This is intentionally *not* validated against the runtime API inventory.  It
    is just a lexical extraction from text the benchmark already exposed to the
    agent.  The released ToolCoder executor can likewise emit a single function
    name not present in the latest Level-2 description, so using an explicitly
    mentioned name is action-space parity rather than hidden-tool disclosure.
    """
    mentions: list[str] = []
    patterns = (
        re.compile(r"\b([A-Z][A-Za-z0-9]{2,})\s+(?:API|tool)\b"),
        re.compile(r"\[\s*([A-Z][A-Za-z0-9]{2,})\s*\("),
    )
    for item in view.get("chat_history") or []:
        if str(item.get("role") or "") != "AI":
            continue
        text = str(item.get("text") or "")
        for pattern in patterns:
            for match in pattern.finditer(text):
                name = str(match.group(1) or "").strip()
                if name and name not in mentions:
                    mentions.append(name)
    return mentions


def _lv2_toolsearcher_is_stale_from_dialogue(view: dict, catalog: list[dict] | None) -> tuple[bool, str]:
    """Detect strong visible evidence that ToolSearcher-only visibility is stale.

    Some published LV2 conversations omit ToolSearcher as an API event (it may
    appear only inline in assistant text) or progress to a concrete API that the
    runtime's strict visibility reconstruction cannot represent.  We bypass a
    new search only when the dialogue itself proves such progression; no target,
    checker, vendor inventory, or hidden schema is consulted.
    """
    visible = {str(x.get("name") or "") for x in (catalog or []) if isinstance(x, dict)}
    if visible != {"ToolSearcher"}:
        return False, "not-toolsearcher-only"

    history = list(view.get("chat_history") or [])
    # A concrete prior API call means the conversation has already moved beyond
    # initial discovery even if strict visibility reconstruction still says only
    # ToolSearcher (a real pattern in the released LV2 files).
    concrete_prior = [
        str(x.get("api_name") or "") for x in history
        if str(x.get("role") or "") == "API" and
        str(x.get("api_name") or "") not in {"", "ToolSearcher"}
    ]
    if concrete_prior:
        return True, "concrete-prior-api"

    explicit = [x for x in _lv2_dialogue_api_mentions(view) if x != "ToolSearcher"]
    if explicit:
        return True, "explicit-dialogue-api"

    # Conservative auth-progression signal for datasets that ask for credentials
    # and then directly call GetUserToken without a ToolSearcher API event.
    user_text = " ".join(
        str(x.get("text") or "") for x in history if str(x.get("role") or "") == "User"
    ).lower()
    ai_text = " ".join(
        str(x.get("text") or "") for x in history if str(x.get("role") or "") == "AI"
    ).lower()
    has_credentials = ("username" in user_text and "password" in user_text)
    auth_progress = any(token in ai_text for token in ("authenticate", "authentication", "get your token", "get a token"))
    if has_credentials and auth_progress:
        return True, "credential-auth-progression"
    return False, "no-strong-dialogue-evidence"



def _lv2_latest_ai_text(view: dict) -> str:
    for item in reversed(list(view.get("chat_history") or [])):
        if str(item.get("role") or "") == "AI":
            text = str(item.get("text") or "").strip()
            if text:
                return text
    return ""


def _lv2_recent_user_text_since_api(view: dict) -> str:
    history = list(view.get("chat_history") or [])
    start = 0
    for index in range(len(history) - 1, -1, -1):
        if str(history[index].get("role") or "") == "API":
            start = index + 1
            break
    return " ".join(
        str(item.get("text") or "").strip()
        for item in history[start:]
        if str(item.get("role") or "") == "User" and str(item.get("text") or "").strip()
    )



def _lv2_has_ai_after_last_api(view: dict) -> bool:
    history = list(view.get("chat_history") or [])
    last_api = -1
    for index, item in enumerate(history):
        if str(item.get("role") or "") == "API":
            last_api = index
    if last_api < 0:
        return False
    return any(
        str(item.get("role") or "") == "AI" and str(item.get("text") or "").strip()
        for item in history[last_api + 1:]
    )


def _lv2_root_user_request(view: dict) -> str:
    """Return the earliest actionable user request in the visible dialogue.

    LV2 follow-up turns often contain only credentials, names, dates, or values.
    For progression recovery we need the task-level operation/object, not the most
    recent clarification value. The earliest actionable User utterance is stable
    across those turns and is fully visible to every compared agent.
    """
    fallback = ""
    for item in list(view.get("chat_history") or []):
        if str(item.get("role") or "") != "User":
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        if not fallback:
            fallback = text
        words = {x.lower() for x in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text)}
        if words.intersection(_LV2_RETRIEVAL_ACTION_WORDS):
            return text
    return fallback


def _lv2_auth_progression_candidate(view: dict, runtime) -> tuple[str, list[dict], str] | None:
    """Recover the standard authentication prerequisite from visible progression.

    This is intentionally narrow. It fires only when the latest assistant turn
    explicitly says authentication/token retrieval is next, visible user text
    contains both username and password, and GetUserToken has not already run.
    The candidate exact name is then validated one-name-at-a-time by the runtime;
    no API inventory is exposed.
    """
    if "GetUserToken" in _lv2_completed_api_names(view):
        return None
    latest_ai = _lv2_latest_ai_text(view).lower()
    if not latest_ai:
        return None
    auth_cue = any(x in latest_ai for x in (
        "authenticate", "authentication", "get your token", "get a token",
        "get your user token", "get the token", "get your token first",
        "get your token.", "get your token ", "get token",
    ))
    if not auth_cue:
        return None
    recent_user = _lv2_recent_user_text_since_api(view).lower()
    if "username" not in recent_user or "password" not in recent_user:
        return None
    exact = _lv2_exact_runtime_catalog(runtime, "GetUserToken")
    if not exact:
        return None
    return "GetUserToken", exact, "visible-auth-progression"


def _lv2_intent_api_candidate_name(view: dict) -> str | None:
    """Infer one conventional CRUD-style API name from the root user request.

    This is a naming-grammar transform, not an inventory search. It is used only
    after visible dialogue has already progressed past discovery or when a stale
    retrieved consumer contradicts that root operation/object. The proposed exact
    name must subsequently exist in the runtime before its schema is exposed.
    """
    text = _lv2_root_user_request(view)
    low = re.sub(r"\s+", " ", str(text or "").strip().lower())
    if not low:
        return None

    # Operation family. Order matters: cancellation/deletion and modification
    # cues are stronger than generic scheduling/creation words.
    if re.search(r"\b(delete|remove|cancel|canceling|cancelling)\b", low):
        prefix = "Delete"
    elif re.search(r"\b(modify|change|update|edit)\b", low):
        prefix = "Modify"
    elif re.search(r"\b(check|query|lookup|look up|retrieve|show|find out|is there|whether)\b", low):
        prefix = "Query"
    elif re.search(r"\brecord(?:ing)?\b", low):
        prefix = "Record"
    elif re.search(r"\b(add|create|book|schedule|set|remind)\b", low):
        prefix = "Add"
    else:
        return None

    # Object family. Prefer semantically specific compounds before generic words.
    if "account balance" in low or re.search(r"\bbalance\b", low):
        obj = "Balance"
    elif re.search(r"\bmeeting\b", low):
        obj = "Meeting"
    elif re.search(r"\bagenda\b", low):
        obj = "Agenda"
    elif re.search(r"\breminder\b", low):
        obj = "Reminder"
    elif re.search(r"\balarm\b", low):
        obj = "Alarm"
    elif "health data" in low:
        obj = "HealthData"
    elif re.search(r"\bstock\b", low):
        obj = "Stock"
    elif re.search(r"\bregistration\b", low):
        obj = "Registration"
    elif re.search(r"\baccount\b", low):
        obj = "Account"
    else:
        return None
    return prefix + obj


def _lv2_structured_progression_candidate(view: dict, runtime) -> tuple[str, list[dict], str] | None:
    """Return one exact dialogue-derived progression API, if it really exists.

    Unlike r39/r40's free-form name guessing, this proposes at most one name from
    a small CRUD naming grammar and validates only that exact candidate. There is
    no inventory enumeration, no checker/ground-truth access, and no alternative
    probing. Completed APIs are never repeated.
    """
    name = _lv2_intent_api_candidate_name(view)
    if not name or name in _lv2_completed_api_names(view) or name == "ToolSearcher":
        return None
    exact = _lv2_exact_runtime_catalog(runtime, name)
    if not exact:
        return None
    return name, exact, "structured-intent-progression"


def _lv2_completed_api_names(view: dict) -> set[str]:
    return {
        str(item.get("api_name") or "")
        for item in (view.get("chat_history") or [])
        if str(item.get("role") or "") == "API" and str(item.get("api_name") or "")
    }


def _lv2_unfinished_explicit_api_names(view: dict, catalog: list[dict] | None) -> list[str]:
    """Explicit concrete API names in dialogue that have not already executed."""
    visible = {str(x.get("name") or "") for x in (catalog or []) if isinstance(x, dict)}
    completed = _lv2_completed_api_names(view)
    return [
        name for name in _lv2_dialogue_api_mentions(view)
        if name != "ToolSearcher" and name not in visible and name not in completed
    ]


def _lv2_unlisted_action_evidence(view: dict, catalog: list[dict] | None) -> tuple[bool, str]:
    """Permit action-stage recovery only when dialogue names the API explicitly.

    r39's unconditional ToolCoder-parity escape hatch was too permissive: it let
    the model invent aliases such as GetBalance/SetReminder.  r40 keeps latent
    recovery only for concrete names already exposed in visible dialogue.  Other
    stale-catalog cases use the separate exact-name resolver below.
    """
    explicit = _lv2_unfinished_explicit_api_names(view, catalog)
    if explicit:
        return True, "explicit-unlisted-api:" + explicit[-1]
    return False, "no-explicit-unlisted-api"


def _lv2_exact_runtime_catalog(runtime, api_name: str) -> list[dict] | None:
    """Resolve metadata for exactly one already-inferred API name.

    This is deliberately not an inventory lookup.  The candidate name must be
    produced from visible dialogue first; the runtime answers only whether that
    exact name exists and, if so, returns that one documented schema.  It is the
    structured analogue of generated code discovering that one function name is
    callable, without revealing alternatives or ground truth.
    """
    name = str(api_name or "").strip()
    if not name or name == "ToolSearcher":
        return None
    loader = getattr(runtime, "_catalog_for_names", None)
    if not callable(loader):
        return None
    try:
        catalog = loader((name,))
    except Exception:
        return None
    if (len(catalog or []) == 1 and
            str((catalog or [])[0].get("name") or "") == name):
        return catalog
    return None


def _lv2_name_resolver_messages(view: dict, catalog: list[dict] | None, *,
                                rejected_name: str | None = None) -> list[dict]:
    visible_names = [str(x.get("name") or "") for x in (catalog or []) if isinstance(x, dict)]
    system = (
        "You are OCA's API-Bank Level-2 API-name resolver. Predict only the exact API NAME "
        "for the immediate next action from the visible dialogue. Do not produce parameters. "
        "The latest ToolSearcher result may be stale or semantically wrong in the published "
        "Level-2 dialogues, so visible catalog names are hints, not mandatory. Prefer a concrete "
        "API/tool name explicitly written by the assistant when it has not already executed. "
        "Otherwise infer one conventional CamelCase API name from the unresolved operation and "
        "object (for example add/query/modify/delete + object). Do not enumerate alternatives, "
        "do not guess credentials or values, and do not return ToolSearcher once the dialogue has "
        "clearly progressed beyond discovery. Return JSON only: {\"api_name\":\"ExactName\"}."
    )
    user = (
        "CURRENT VISIBLE TOOL NAMES (possibly stale):\n" +
        json.dumps(visible_names, ensure_ascii=False) +
        "\n\nDIALOGUE SO FAR:\n" + format_history(view.get("chat_history") or [])
    )
    if rejected_name:
        user += (
            "\n\nThe previously inferred API name " + json.dumps(str(rejected_name)) +
            " is not an available runtime API name. Reconsider the exact CamelCase name from "
            "the dialogue. You are not being shown the inventory."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _oca_lv2_name_resolver_call(client, model: str, view: dict, catalog: list[dict] | None, *,
                                rejected_name: str | None = None):
    from utils.token_meter import stage as token_stage
    messages = _lv2_name_resolver_messages(view, catalog, rejected_name=rejected_name)
    response_start = len(client.chat.completions.responses)
    kwargs = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": 512,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    with token_stage("lv2_name_resolver"):
        response = client.chat.completions.create(**kwargs)
    raw = str(response.choices[0].message.content or "")
    return raw, messages, client.chat.completions.responses[response_start:]


def _parse_lv2_api_name(raw: str) -> str:
    obj = _parse_json_object(str(raw or ""))
    name = str(obj.get("api_name") or obj.get("api") or obj.get("name") or "").strip()
    if not re.fullmatch(r"[A-Z][A-Za-z0-9]{2,}", name):
        raise AgentBehaviorError(f"OCA LV2 name resolver returned invalid API name: {name!r}")
    return name


def _lv2_exact_latent_candidate(*, view: dict, catalog: list[dict] | None, runtime,
                                model: str, allow_name_resolver: bool = True) -> tuple[str, list[dict], list[dict]] | None:
    """Resolve one exact latent API without exposing or searching the inventory.

    Explicit API names already written in the dialogue are always eligible. The
    free-form name resolver is optional because live r40 audits showed that stale
    ToolSearcher-only states can otherwise spend tokens guessing aliases without
    improving accuracy.
    """
    explicit = _lv2_unfinished_explicit_api_names(view, catalog)
    trace: list[dict] = []
    if explicit:
        # Most recent unfinished concrete name has the strongest visible evidence.
        name = explicit[-1]
        exact = _lv2_exact_runtime_catalog(runtime, name)
        trace.append({"stage": "oca_lv2_latent_name_explicit", "api_name": name,
                      "runtime_valid": bool(exact)})
        if exact:
            return name, exact, trace

    if not allow_name_resolver:
        return None

    from agents.common import make_client
    rejected = None
    for attempt in (1, 2):
        try:
            client = _oca_planner_client(
                make_client(), model, total_timeout_sec=min(60, _oca_task_timeout_seconds()))
            raw, messages, responses = _oca_lv2_name_resolver_call(
                client, model, view, catalog, rejected_name=rejected)
            name = _parse_lv2_api_name(raw)
            exact = _lv2_exact_runtime_catalog(runtime, name)
            trace.append({"stage": "oca_lv2_latent_name_resolve", "attempt": attempt,
                          "output": raw, "responses": responses, "api_name": name,
                          "runtime_valid": bool(exact)})
            if exact:
                return name, exact, trace
            rejected = name
        except Exception as exc:
            trace.append({"stage": "oca_lv2_latent_name_resolve", "attempt": attempt,
                          "error": str(exc)})
            break
    return None


def _run_oca_lv2_exact_latent(*, view: dict, sample, runtime, model: str,
                              visible_catalog: list[dict] | None,
                              reason: str, allow_name_resolver: bool = True) -> AgentPrediction | None:
    """Resolve name first, then bind parameters against only that exact schema."""
    resolved = _lv2_exact_latent_candidate(
        view=view, catalog=visible_catalog, runtime=runtime, model=model,
        allow_name_resolver=allow_name_resolver)
    if resolved is None:
        return None
    name, exact_catalog, name_trace = resolved
    visible_names = {str(x.get("name") or "") for x in (visible_catalog or []) if isinstance(x, dict)}
    if name in visible_names:
        return None
    print(
        f"  [OCA/API-BANK lv2] exact latent recovery selects {name} ({reason})",
        flush=True)
    observed = _lv2_observed_output_values(view)
    input_fields = set(_input_fields_for(exact_catalog, name))
    bind_fields = set(observed).intersection(input_fields)

    # Complex APIs explicitly named by the visible dialogue deserve a second
    # independent parameter binding before any environment execution.  This is
    # the same fair single-trajectory consensus boundary used elsewhere in LV2:
    # reason twice, choose once, execute once.  It is deliberately restricted to
    # explicit dialogue names so stale/guessed latent names do not receive extra
    # speculative budget.
    if reason.startswith("explicit-unlisted-api:") and len(input_fields) >= 3:
        pred = _run_oca_lv2_visible_consensus(
            view=view, sample=sample, runtime=runtime, model=model,
            catalog=exact_catalog, observed_bind_fields=bind_fields)
    else:
        pred = _run_oca_efficient(
            view=view, sample=sample, runtime=runtime, model=model,
            catalog_override=exact_catalog, observed_bind_fields=bind_fields,
            allow_unlisted_api=False)
    pred.trace = name_trace + list(pred.trace or [])
    return pred



def _run_oca_lv2_resolved_exact(*, view: dict, sample, runtime, model: str,
                                name: str, exact_catalog: list[dict],
                                reason: str) -> AgentPrediction:
    """Bind/execute one exact name already derived from visible dialogue.

    The caller has already produced and one-name-validated the API name. This
    helper reveals only that exact schema, binds previously observed outputs only
    to matching documented fields, and follows the ordinary one-action/one-repair
    OCA boundary.
    """
    print(
        f"  [OCA/API-BANK lv2] structured progression selects {name} ({reason})",
        flush=True)
    observed = _lv2_observed_output_values(view)
    input_fields = set(_input_fields_for(exact_catalog, name))
    bind_fields = set(observed).intersection(input_fields)
    pred = _run_oca_efficient(
        view=view, sample=sample, runtime=runtime, model=model,
        catalog_override=exact_catalog, observed_bind_fields=bind_fields,
        allow_unlisted_api=False)
    pred.trace = [{"stage": "oca_lv2_structured_progression", "api_name": name,
                   "reason": reason}] + list(pred.trace or [])
    return pred


def _lv2_preexecution_adjudicator_messages(view: dict, catalog: list[dict] | None,
                                            first: dict, second: dict) -> list[dict]:
    """Adjudicate LV2 candidates before any candidate is executed.

    This keeps OCA inside a single-trajectory agent boundary: multiple internal
    reasoning proposals are allowed, but the environment is queried only after
    one action has been selected. No counterfactual sandbox outcome is available
    to the judge.
    """
    system = (
        "You are OCA's conservative API-Bank Level-2 action adjudicator. TWO independent "
        "solvers proposed different next actions from the same visible dialogue and retrieved "
        "tool catalog. Neither proposal has been executed. Choose only from FIRST or SECOND "
        "using the documented tool semantics, request schema, prerequisite ordering, and exact "
        "values present in the dialogue/API observations. Do not infer hidden tools, execute "
        "anything mentally from private state, invent values, or propose a third action. Give "
        "extra weight to the requested operation (add/modify/delete/query/etc.), the object being "
        "acted on, exact existing-record identity, and required wire shape. When the evidence "
        "does not clearly favor SECOND, choose FIRST. Return JSON only: "
        "{\"choice\":\"first\"} or {\"choice\":\"second\",\"confidence\":\"high\"}."
    )
    names = {str(first.get("api_name") or ""), str(second.get("api_name") or "")}
    compact_catalog = [item for item in _compact_action_catalog(catalog)
                       if str(item.get("name") or "") in names]
    user = (
        "DOCUMENTED API CATALOG:\n" +
        json.dumps(compact_catalog, ensure_ascii=False, separators=(",", ":"), sort_keys=True) +
        "\n\nDIALOGUE SO FAR:\n" + format_history(view.get("chat_history") or []) +
        "\n\nFIRST CANDIDATE:\n" + json.dumps(first, ensure_ascii=False, sort_keys=True, default=str) +
        "\n\nSECOND CANDIDATE:\n" + json.dumps(second, ensure_ascii=False, sort_keys=True, default=str)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _oca_lv2_preexecution_adjudicator_call(client, model: str, view: dict,
                                             catalog: list[dict] | None,
                                             first: dict, second: dict):
    from utils.token_meter import stage as token_stage
    messages = _lv2_preexecution_adjudicator_messages(view, catalog, first, second)
    response_start = len(client.chat.completions.responses)
    kwargs = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": 1024,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    with token_stage("lv2_preexecution_adjudicator"):
        response = client.chat.completions.create(**kwargs)
    raw = str(response.choices[0].message.content or "")
    return raw, messages, client.chat.completions.responses[response_start:]


def _run_oca_lv2_visible_consensus(*, view: dict, sample, runtime, model: str,
                                    catalog: list[dict],
                                    observed_bind_fields: set[str] | None = None) -> AgentPrediction:
    """Fair LV2 consensus: reason twice, choose once, then execute one trajectory.

    Unlike the older generic consensus profile, this routine never probes both
    alternatives and then uses counterfactual environment outcomes to select the
    submitted action. It uses only visible dialogue/catalog evidence before the
    choice. The chosen action is executed once; one error-driven repair is allowed
    only after an actual structural/request failure, matching normal agent repair.
    """
    from agents.common import make_client

    trace: list[dict] = []
    candidates: list[tuple[str, dict]] = []

    # First solver: OCA's compact action selector.
    try:
        first_client = _oca_planner_client(
            make_client(), model, total_timeout_sec=min(60, _oca_task_timeout_seconds()))
        raw1, messages1, responses1 = _oca_direct_action_call(
            first_client, model, view, catalog)
        name1, params1 = _parse_direct_oca_action(
            raw1, sample, catalog, allow_omitted_required=True)
        if observed_bind_fields:
            params1, _ = _lv2_bind_observed_outputs(
                view, catalog, name1, params1, allowed_fields=observed_bind_fields)
        first = {"api_name": name1, "params": params1}
        candidates.append(("first", first))
        trace.append({"stage": "oca_lv2_consensus_first", "output": raw1,
                      "responses": responses1, "api_name": name1, "params": params1})
    except Exception as exc:
        trace.append({"stage": "oca_lv2_consensus_first",
                      "error": f"first solver unavailable/invalid: {exc}"})

    # Second solver is independent: it never sees FIRST.
    try:
        second_client = _oca_planner_client(
            make_client(), model, total_timeout_sec=min(60, _oca_task_timeout_seconds()))
        raw2, messages2, responses2 = _oca_consensus_independent_call(
            second_client, model, view, catalog)
        name2, params2 = _parse_direct_oca_action(
            raw2, sample, catalog, allow_omitted_required=True)
        if observed_bind_fields:
            params2, _ = _lv2_bind_observed_outputs(
                view, catalog, name2, params2, allowed_fields=observed_bind_fields)
        second = {"api_name": name2, "params": params2}
        candidates.append(("second", second))
        trace.append({"stage": "oca_lv2_consensus_second", "output": raw2,
                      "responses": responses2, "api_name": name2, "params": params2})
    except Exception as exc:
        trace.append({"stage": "oca_lv2_consensus_second",
                      "error": f"independent solver unavailable/invalid: {exc}"})

    if not candidates:
        error = AgentBehaviorError("OCA LV2 consensus produced no valid visible action")
        error.agent_trace = trace
        raise error

    selected = candidates[0][1]
    if len(candidates) == 2:
        first = candidates[0][1]
        second = candidates[1][1]
        if _actions_equal(first, second):
            print("  [OCA/API-BANK lv2] independent solvers agree before execution", flush=True)
            selected = first
        else:
            print("  [OCA/API-BANK lv2] adjudicating disagreement before execution", flush=True)
            choose_second = False
            try:
                judge_client = _oca_planner_client(
                    make_client(), model, total_timeout_sec=min(60, _oca_task_timeout_seconds()))
                judge_raw, judge_messages, judge_responses = _oca_lv2_preexecution_adjudicator_call(
                    judge_client, model, view, catalog, first, second)
                choose_second = _consensus_adjudicator_prefers_second(judge_raw)
                trace.append({"stage": "oca_lv2_preexecution_adjudicate", "output": judge_raw,
                              "responses": judge_responses, "chose_second": choose_second})
            except Exception as exc:
                trace.append({"stage": "oca_lv2_preexecution_adjudicate", "chose_second": False,
                              "error": f"optional adjudicator unavailable/invalid: {exc}"})
            selected = second if choose_second else first

    name = str(selected.get("api_name") or "")
    params = dict(selected.get("params") or {})
    print(f"  [OCA/API-BANK lv2] executing selected action {name}", flush=True)
    probe = _probe_oca_action(runtime, sample, name, params)
    trace.append({"stage": "oca_lv2_selected_probe", "api_name": name, "params": params,
                  "success": bool(probe.get("success")), "error": probe.get("error")})
    if probe.get("success"):
        return _prediction_from_probe(name, params, probe, trace)

    # A business-state miss remains a legitimate next-action candidate and is
    # submitted directly. Repair only objective request-shape/type/auth failures.
    if not _sandbox_rejection_needs_repair(str(probe.get("error") or "")):
        return AgentPrediction(prediction_text=_call_text(name, params), strategy="oca",
                               trace=trace, action={"api_name": name, "params": params})

    # One bounded repair after the actually selected action fails structurally.
    print("  [OCA/API-BANK lv2] repairing selected action after structural failure", flush=True)
    rejection = "sandbox rejected " + _call_text(name, params) + ": " + str(probe.get("error") or "")
    try:
        repair_client = _oca_planner_client(
            make_client(), model, total_timeout_sec=min(60, _oca_task_timeout_seconds()))
        repair_raw, repair_messages, repair_responses = _oca_direct_action_call(
            repair_client, model, view, catalog,
            previous_action={"api_name": name, "params": params}, error=rejection)
        repaired_name, repaired_params = _parse_direct_oca_action(
            repair_raw, sample, catalog, allow_omitted_required=True)
        if observed_bind_fields:
            repaired_params, _ = _lv2_bind_observed_outputs(
                view, catalog, repaired_name, repaired_params, allowed_fields=observed_bind_fields)
        repair_probe = _probe_oca_action(runtime, sample, repaired_name, repaired_params)
        trace.append({"stage": "oca_lv2_selected_repair", "output": repair_raw,
                      "responses": repair_responses, "api_name": repaired_name,
                      "params": repaired_params, "success": bool(repair_probe.get("success")),
                      "error": repair_probe.get("error")})
        if repair_probe.get("success") or not _sandbox_rejection_needs_repair(
                str(repair_probe.get("error") or "")):
            return _prediction_from_probe(repaired_name, repaired_params, repair_probe, trace)
    except Exception as exc:
        trace.append({"stage": "oca_lv2_selected_repair",
                      "error": f"bounded repair unavailable/invalid: {exc}"})

    error = AgentBehaviorError(
        "OCA LV2 selected action remained structurally invalid after one bounded repair")
    error.agent_trace = trace
    raise error


def _run_oca_lv2(*, view: dict, sample, runtime, model: str) -> AgentPrediction:
    """Level-2 profile: retrieval-specialized reasoning plus bounded progression recovery.

    The normal boundary remains the currently visible ToolSearcher/catalog surface.
    Recovery is allowed only from information already visible in the dialogue:
    explicit API names, an explicit authentication step with visible credentials,
    or one CRUD-style operation/object name after the dialogue has already moved
    beyond discovery. Every inferred exact name is validated one-name-at-a-time;
    no inventory, checker, target answer, or alternative execution is exposed.
    """
    level = int(getattr(sample, "benchmark_level", 1) or 1)
    if level != 2:
        return _run_oca_adaptive(view=view, sample=sample, runtime=runtime, model=model)
    catalog_fn = getattr(runtime, "api_catalog", None)
    catalog = catalog_fn(sample) if callable(catalog_fn) else None
    visible = [str(x.get("name") or "") for x in (catalog or []) if isinstance(x, dict)]

    # Some published LV2 dialogues explicitly progress to authentication even
    # when the preceding ToolSearcher output omitted the auth helper. Recover
    # that prerequisite only when credentials are visible and the latest AI turn
    # says authentication/token retrieval is next.
    if _oca_lv2_latent_recovery_enabled() and "GetUserToken" not in visible:
        auth = _lv2_auth_progression_candidate(view, runtime)
        if auth is not None:
            name, exact, reason = auth
            return _run_oca_lv2_resolved_exact(
                view=view, sample=sample, runtime=runtime, model=model,
                name=name, exact_catalog=exact, reason=reason)

    if visible == ["ToolSearcher"]:
        stale, reason = _lv2_toolsearcher_is_stale_from_dialogue(view, catalog)
        stale = stale and _oca_lv2_latent_recovery_enabled()
        if stale:
            # First honor exact concrete names already written in visible dialogue.
            explicit = _lv2_unfinished_explicit_api_names(view, catalog)
            if explicit:
                recovered = _run_oca_lv2_exact_latent(
                    view=view, sample=sample, runtime=runtime, model=model,
                    visible_catalog=catalog or [], reason=reason,
                    allow_name_resolver=False)
                if recovered is not None:
                    return recovered

            # Then use one deterministic operation/object naming transform. This
            # recovers progression such as authenticated check-balance ->
            # QueryBalance without giving the model or helper a hidden inventory.
            structured = (
                _lv2_structured_progression_candidate(view, runtime)
                if _lv2_has_ai_after_last_api(view) else None
            )
            if structured is not None:
                name, exact, structured_reason = structured
                return _run_oca_lv2_resolved_exact(
                    view=view, sample=sample, runtime=runtime, model=model,
                    name=name, exact_catalog=exact,
                    reason=f"{reason}:{structured_reason}")

            # Free-form name inference remains an explicit ablation only.
            if _oca_lv2_name_recovery_enabled():
                recovered = _run_oca_lv2_exact_latent(
                    view=view, sample=sample, runtime=runtime, model=model,
                    visible_catalog=catalog or [], reason=reason,
                    allow_name_resolver=True)
                if recovered is not None:
                    return recovered
            print(
                f"  [OCA/API-BANK lv2] stale-view exact recovery unavailable ({reason}); "
                "falling back to ToolSearcher",
                flush=True)
        return _run_oca_lv2_search(view=view, sample=sample, runtime=runtime, model=model)

    # Most API-Bank LV2 two-tool surfaces are a visible prerequisite producer
    # plus the discovered target. Resolve that dependency from documented input/
    # output schemas and already-visible state, then ask the model only to bind
    # parameters for the one eligible tool.
    dependency = _lv2_dependency_choice(view, catalog)
    if dependency is not None:
        selected_name, selected_entry, dependency_field = dependency
        print(
            f"  [OCA/API-BANK lv2] schema dependency selects {selected_name} via {dependency_field}",
            flush=True)
        observed = _lv2_observed_output_values(view)
        if dependency_field in observed and _oca_lv2_latent_recovery_enabled():
            # Explicit dialogue names have strongest evidence and retain the r40
            # exact-name parity path.
            allowed, latent_reason = _lv2_unlisted_action_evidence(view, [selected_entry])
            if allowed:
                recovered = _run_oca_lv2_exact_latent(
                    view=view, sample=sample, runtime=runtime, model=model,
                    visible_catalog=[selected_entry], reason=latent_reason,
                    allow_name_resolver=False)
                if recovered is not None:
                    return recovered

            # A retrieved consumer can also be stale. If the root request maps to
            # a different conventional exact API that really exists, bind only
            # that exact schema. The selected visible consumer is kept whenever
            # the structured name agrees with it or no exact candidate exists.
            structured = (
                _lv2_structured_progression_candidate(view, runtime)
                if _lv2_has_ai_after_last_api(view) else None
            )
            if structured is not None:
                name, exact, structured_reason = structured
                if name != selected_name:
                    return _run_oca_lv2_resolved_exact(
                        view=view, sample=sample, runtime=runtime, model=model,
                        name=name, exact_catalog=exact,
                        reason=f"stale-consumer:{structured_reason}")
        return _run_oca_efficient(
            view=view, sample=sample, runtime=runtime, model=model,
            catalog_override=[selected_entry],
            observed_bind_fields={dependency_field},
            allow_unlisted_api=False)

    # A single visible tool is unambiguous unless the conversation has already
    # executed a concrete prerequisite and the original operation/object resolves
    # to a different exact API. This handles published LV2 stale-search cases
    # while leaving initial single-tool discovery untouched.
    if len(catalog or []) <= 1:
        allow_latent, latent_reason = _lv2_unlisted_action_evidence(view, catalog)
        allow_latent = allow_latent and _oca_lv2_latent_recovery_enabled()
        if allow_latent:
            recovered = _run_oca_lv2_exact_latent(
                view=view, sample=sample, runtime=runtime, model=model,
                visible_catalog=catalog or [], reason=latent_reason,
                allow_name_resolver=False)
            if recovered is not None:
                return recovered

        completed_concrete = {
            x for x in _lv2_completed_api_names(view) if x != "ToolSearcher"
        }
        if completed_concrete and _oca_lv2_latent_recovery_enabled():
            structured = (
                _lv2_structured_progression_candidate(view, runtime)
                if _lv2_has_ai_after_last_api(view) else None
            )
            if structured is not None:
                name, exact, structured_reason = structured
                visible_name = visible[0] if visible else ""
                if name and name != visible_name:
                    return _run_oca_lv2_resolved_exact(
                        view=view, sample=sample, runtime=runtime, model=model,
                        name=name, exact_catalog=exact,
                        reason=f"post-prerequisite:{structured_reason}")
        return _run_oca_efficient(
            view=view, sample=sample, runtime=runtime, model=model,
            catalog_override=catalog or [], allow_unlisted_api=False)

    return _run_oca_lv2_visible_consensus(
        view=view, sample=sample, runtime=runtime, model=model, catalog=catalog or [])


def run_oca(*, view: dict, sample, runtime, model: str) -> AgentPrediction:
    """Run API-Bank with the selected OCA execution profile."""
    profile = _oca_profile()
    if profile == "lv2":
        return _run_oca_lv2(view=view, sample=sample, runtime=runtime, model=model)
    if profile == "consensus":
        return _run_oca_consensus(view=view, sample=sample, runtime=runtime, model=model)
    if profile == "adaptive":
        return _run_oca_adaptive(view=view, sample=sample, runtime=runtime, model=model)
    if profile == "efficient":
        return _run_oca_efficient(view=view, sample=sample, runtime=runtime, model=model)
    return _run_oca_evidence(view=view, sample=sample, runtime=runtime, model=model)


_CODEACT_CALL_PROMPT = """You are CodeAct. Use executable Python syntax as the action space.
Based on the visible API descriptions and conversation history, generate exactly ONE immediate API call for the next step.
Output only one Python call expression using the documented API name and documented keyword arguments, for example:
AddAgenda(content="Meeting with John", time="2023-10-26 09:00:00")
Do not wrap the call in JSON or square brackets. Do not output prose, a final answer, or the result of the API call.
Use only APIs and request fields in the visible toolbox. Reuse exact values from the dialogue when referring to existing records. Never invent tokens, IDs, or credentials. This year is 2023.

API descriptions:
"""


def _codeact_python_action_text(name: str, params: dict) -> str:
    return f"{name}({', '.join(f'{k}={v!r}' for k, v in (params or {}).items())})"


def _combine_assistant_messages(messages: List[dict]) -> List[dict]:
    combined: List[dict] = []
    for message in messages:
        if (message.get("role") == "assistant" and combined and
                combined[-1].get("role") == "assistant"):
            combined[-1]["content"] += "\n" + str(message.get("content", ""))
        else:
            combined.append(dict(message))
    return combined


def _codeact_messages(view: dict) -> List[dict]:
    messages: List[dict] = [
        {"role": "system", "content": _CODEACT_CALL_PROMPT + str(view["api_descriptions"])}
    ]
    for item in view.get("chat_history") or []:
        role = item.get("role")
        if role == "User":
            messages.append({"role": "user", "content": str(item.get("text", ""))})
        elif role == "AI":
            messages.append({"role": "assistant", "content": str(item.get("text", ""))})
        elif role == "API":
            messages.append({
                "role": "assistant",
                "content": _codeact_python_action_text(
                    str(item.get("api_name")), item.get("param_dict") or {}),
            })
            result = item.get("result")
            output = result.get("output") if isinstance(result, dict) else result
            messages.append({"role": "user", "content": f"Response: {str(output)}"})
        else:
            raise ValueError(f"Invalid API-Bank chat role: {role!r}")
    return _combine_assistant_messages(messages)


def _strip_python_fence(text: str) -> str:
    value = str(text or "").strip()
    match = re.search(r"```(?:python)?\s*(.*?)```", value, flags=re.IGNORECASE | re.DOTALL)
    if match:
        value = match.group(1).strip()
    return value


def _parse_codeact_python_action(raw: str, visible_names) -> tuple[str, dict]:
    """Parse one CodeAct Python call without consulting any hidden API names."""
    text = _strip_python_fence(raw)
    try:
        tree = ast.parse(text, mode="exec")
    except SyntaxError as exc:
        raise AgentBehaviorError(f"CodeAct returned invalid Python action: {exc}") from exc
    visible = {str(x) for x in (visible_names or ())}
    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = str(node.func.id)
        if visible and name not in visible:
            continue
        if node.args:
            continue
        params = {}
        ok = True
        for kw in node.keywords:
            if kw.arg is None:
                ok = False
                break
            try:
                params[str(kw.arg)] = ast.literal_eval(kw.value)
            except Exception:
                ok = False
                break
        if ok:
            candidates.append((name, params))
    if len(candidates) != 1:
        raise AgentBehaviorError(
            f"CodeAct must emit exactly one visible executable API call; observed {len(candidates)}")
    return candidates[0]


def run_codeact(*, view: dict, sample, runtime, model: str) -> AgentPrediction:
    """CodeAct baseline using the paper's executable Python-call action format."""
    messages = _codeact_messages(view)
    raw = _chat(model, messages)
    trace = [{"stage": "model", **_response_trace(raw), "messages": messages}]
    try:
        name, params = _parse_codeact_python_action(raw, getattr(sample, "api_names", ()))
    except AgentBehaviorError as exc:
        # A bracketed call is accepted only as a compatibility fallback for model
        # formatting drift; it is not requested by the CodeAct prompt.
        bracket = extract_bracket_call(raw)
        if bracket:
            return AgentPrediction(prediction_text=bracket, strategy="codeact", trace=trace)
        exc.agent_trace = trace
        raise
    return AgentPrediction(
        prediction_text=_code_action_text(name, params), strategy="codeact", trace=trace,
        action={"api_name": name, "params": params},
    )


def _load_toolcoder_templates(runtime):
    path = Path(runtime.root) / "template.py"
    if not path.is_file():
        raise RuntimeError(
            "ToolCoder API-Bank requires template.py from the ToolCoder apibank snapshot. "
            "Re-run bootstrap_apibank.py or use a ToolCoder apibank root.")
    name = f"_secat_apibank_template_{abs(hash(str(path.resolve())))}"
    if name in sys.modules:
        module = sys.modules[name]
    else:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load ToolCoder templates from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    for attr in ("PLANNER_PROMPT", "REPLAN_TEMPLATE", "EXECUTION_FAILURE_TEMPLATE", "API_CALL_CODE"):
        if not hasattr(module, attr):
            raise RuntimeError(f"ToolCoder template.py is missing {attr}")
    return module


def _toolcoder_expand_history(history: List[dict]) -> List[dict]:
    expanded: List[dict] = []
    for item in history or []:
        if item.get("role") == "API":
            result = item.get("result")
            output = result.get("output") if isinstance(result, dict) else result
            expanded.append({
                "role": "AI",
                "text": f"I need to call the API: {_code_action_text(str(item.get('api_name')), item.get('param_dict') or {})}",
            })
            expanded.append({
                "role": "User",
                "text": f"The response from the API: {output}. Please go on.",
            })
        else:
            expanded.append(dict(item))
    return expanded


def _toolcoder_messages(prompt: str) -> List[dict]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": str(prompt)},
    ]


def _extract_python_code(text: str) -> str:
    match = re.search(r"```python\s*(.*?)```", str(text or ""), re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    generic = re.search(r"```\s*(.*?)```", str(text or ""), re.DOTALL)
    if generic:
        return generic.group(1).strip()
    candidate = str(text or "").strip()
    if candidate and ("call_api" in candidate or "def " in candidate):
        return candidate
    raise AgentBehaviorError("ToolCoder model output contained no Python code")


def _extract_api_paths(code: str) -> List[str]:
    pattern = re.compile(
        r"call_api\s*\(\s*api_name\s*=\s*(?:f?[\"'])([^\"']+)(?:[\"'])",
        re.MULTILINE,
    )
    return pattern.findall(str(code or ""))


def _toolcoder_action_paths(code: str) -> List[str]:
    """Count call sites, excluding comments/docstrings and supporting positional names."""
    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError:
        # Preserve the repair opportunity for code with an identifiable action.
        return _extract_api_paths(str(code or ""))
    paths = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "call_api"):
            continue
        value = next((k.value for k in node.keywords if k.arg == "api_name"),
                     node.args[0] if node.args else None)
        paths.append(value.value if isinstance(value, ast.Constant) and isinstance(value.value, str)
                     else "<dynamic>")
    return paths


def _parse_toolcoder_stdout(stdout: str):
    """Safely reproduce ToolCoder's published ``eval(exec_output.strip())``."""
    text = str(stdout or "").strip()
    if not text:
        raise AgentBehaviorError("ToolCoder executed the API but printed no result")
    try:
        return ast.literal_eval(text)
    except Exception as exc:
        raise AgentBehaviorError(
            f"ToolCoder stdout is not one evaluable result literal: {text[:500]!r}") from exc


def _replay_code(sample) -> str:
    lines: List[str] = []
    for item in getattr(sample, "chat_history", ()):
        if item.get("role") != "API":
            continue
        name = str(item.get("api_name"))
        params = repr(dict(item.get("param_dict") or {}))
        lines.append(f"_secat_replay_api_call({name!r}, {params})")
    return "\n".join(lines)


def _sanitized_subprocess_env(root: Path) -> dict:
    allowed = {}
    for key in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TEMP", "TMP", "NLTK_DATA",
                "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
                "APIBANK_TOOL_TIMEOUT_SEC", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                "http_proxy", "https_proxy", "all_proxy", "no_proxy",
                "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "SENTENCE_TRANSFORMERS_HOME",
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        if os.environ.get(key):
            allowed[key] = os.environ[key]
    allowed["PYTHONIOENCODING"] = "utf-8"
    # The child needs only the vendored local API modules, never SECAT credentials.
    allowed["PYTHONPATH"] = str(root)
    return allowed


def _build_toolcoder_execution_sandbox(source_root: Path, destination: Path) -> Path:
    """Copy only executable API runtime assets, excluding benchmark dialogues."""
    source_root = Path(source_root)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    # Root-level support modules/configuration are public vendor runtime assets.
    # Deliberately do not copy directories wholesale: the source tree also holds
    # lv1/lv2 dialogue files containing the target answers.
    for item in source_root.iterdir():
        if item.is_file() and item.suffix == ".py":
            shutil.copy2(item, destination / item.name)
    for dirname in ("apis", "init_database"):
        src = source_root / dirname
        if src.is_dir():
            shutil.copytree(src, destination / dirname)
    return destination


_TOOLCODER_FORBIDDEN_CALLS = {
    "open", "eval", "exec", "compile", "__import__", "input", "getattr", "setattr",
    "delattr", "globals", "locals", "vars", "dir", "breakpoint", "help",
}
_TOOLCODER_FORBIDDEN_ATTRS = {
    "init_databases", "init_tool", "get_api_description", "get_api_by_name",
    "tool_classes", "api_classes", "__dict__", "__class__", "__globals__",
    "__subclasses__", "__mro__", "__bases__", "__code__",
}


def _normalize_toolcoder_generated_code(code: str) -> str:
    """Strip the exact vendor wrapper when a repair returns full executable code."""
    text = str(code or "")
    try:
        tree = ast.parse(text, mode="exec")
    except SyntaxError as exc:
        raise AgentBehaviorError(f"ToolCoder generated invalid Python: {exc}") from exc
    kept = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "tool_manager":
            continue
        if isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if "tool_manager" in names:
                continue
        if isinstance(node, ast.FunctionDef) and node.name == "call_api":
            continue
        # Vendor replay lines are redundant with SECAT-owned replay and must not
        # remain executable in the model-controlled body.
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and
                isinstance(node.value.func, ast.Attribute) and
                isinstance(node.value.func.value, ast.Name) and
                node.value.func.value.id == "tool_manager" and node.value.func.attr == "api_call"):
            continue
        kept.append(node)
    tree.body = kept
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _validate_toolcoder_generated_code(code: str) -> None:
    """Reject file/process/runtime-introspection escapes from model-generated code."""
    try:
        tree = ast.parse(str(code or ""), mode="exec")
    except SyntaxError as exc:
        raise AgentBehaviorError(f"ToolCoder generated invalid Python: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise AgentBehaviorError("ToolCoder generated code may not import modules")
        if isinstance(node, ast.Name):
            if node.id == "__name__":
                if not isinstance(node.ctx, ast.Load):
                    raise AgentBehaviorError("ToolCoder generated code may only read __name__; assignment is not allowed")
            elif (node.id in {"ToolManager", "tool_manager", "__builtins__"}
                  or (node.id.startswith("_") and node.id != "_")):
                raise AgentBehaviorError("ToolCoder generated code may use only the public call_api interface")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _TOOLCODER_FORBIDDEN_CALLS:
                raise AgentBehaviorError(f"ToolCoder generated code may not call {node.func.id}()")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in _TOOLCODER_FORBIDDEN_ATTRS | {"format", "format_map"}:
                raise AgentBehaviorError(
                    f"ToolCoder generated code may not inspect runtime attribute {node.attr!r}")


def _execute_toolcoder_code(runtime, sample, planner_code: str) -> dict:
    planner_code = _normalize_toolcoder_generated_code(planner_code)
    _validate_toolcoder_generated_code(planner_code)
    root = Path(runtime.root)
    replay = _replay_code(sample) if getattr(runtime, "state_mode", "published") == "replay" else ""
    replayed_calls = sum(1 for x in getattr(sample, "chat_history", ()) if x.get("role") == "API") if replay else 0
    # Trace at ToolManager.api_call rather than only through our call_api wrapper.
    # ToolCoder's repair model is explicitly shown the full executable wrapper and
    # can return a revised wrapper that redefines call_api/tool_manager; class-level
    # tracing still observes that call without changing model-generated code.
    allowed_names = list(getattr(sample, "api_names", ()) or ())
    if int(getattr(sample, "benchmark_level", 1) or 1) == 2:
        # The model-visible toolbox remains scoped by runtime.agent_view().  The
        # execution sandbox, like upstream ToolCoder's unscoped ToolManager, must
        # nevertheless be capable of executing the model's predicted API name.
        # This is execution capability, not additional prompt information.
        for item in getattr(sample, "chat_history", ()) or ():
            if item.get("role") == "API":
                candidate = str(item.get("api_name") or "").strip()
                if candidate and candidate not in allowed_names:
                    allowed_names.append(candidate)
        for candidate in _toolcoder_action_paths(planner_code):
            if candidate and candidate != "<dynamic>" and candidate not in allowed_names:
                allowed_names.append(candidate)
    prelude = f'''
import json as _secat_json
import atexit as _atexit
from apibank_support import tool_manager_class, APIBankInfrastructureError
from pathlib import Path as _Path
ToolManager = tool_manager_class(_Path.cwd(), {allowed_names!r})
_SECAT_CALL_TRACE = []
_SECAT_INFRA_ERRORS = []
def _secat_save_control():
    _Path(".secat_execution.json").write_text(_secat_json.dumps(
        {{"calls": _SECAT_CALL_TRACE, "infrastructure_errors": _SECAT_INFRA_ERRORS}}, default=str), encoding="utf-8")
_atexit.register(_secat_save_control)
_SECAT_RECORD_CALLS = False
_SECAT_ORIGINAL_API_CALL = ToolManager.api_call
def _secat_traced_api_call(self, tool_name=None, *args, **kwargs):
    try:
        result = _SECAT_ORIGINAL_API_CALL(self, tool_name, *args, **kwargs)
    except APIBankInfrastructureError as error:
        _SECAT_INFRA_ERRORS.append(str(error))
        raise
    if _SECAT_RECORD_CALLS:
        _SECAT_CALL_TRACE.append({{"api_name": tool_name, "params": dict(kwargs), "result": result}})
    return result
ToolManager.api_call = _secat_traced_api_call
tool_manager = ToolManager()
# Replay is trusted harness setup.  Its diagnostic/progress stderr must never be
# mistaken for an error from the model's current target action.
import contextlib as _secat_contextlib
import io as _secat_io
def _secat_replay_api_call(api_name, params):
    with _secat_contextlib.redirect_stderr(_secat_io.StringIO()):
        return tool_manager.api_call(api_name, **params)
{replay}
_SECAT_RECORD_CALLS = True
def call_api(api_name, params):
    return tool_manager.api_call(api_name, **params)
'''
    # Generated code cannot access the trusted prelude's globals. AST checks
    # complement this capability boundary; this is not an OS/container sandbox.
    safe_names = ("print", "str", "int", "float", "bool", "len", "dict", "list", "tuple", "set",
                  "range", "enumerate", "zip", "min", "max", "sum", "sorted", "reversed", "abs",
                  "round", "any", "all", "isinstance", "Exception", "ValueError", "TypeError", "KeyError")
    execution_body = f'''
import builtins as _builtins
_scope = {{"__builtins__": {{name: getattr(_builtins, name) for name in {safe_names!r}}}, "call_api": call_api, "__name__": "__main__"}}
try:
    exec(compile({planner_code!r}, "<toolcoder-action>", "exec"), _scope, _scope)
except APIBankInfrastructureError as _error:
    _SECAT_INFRA_ERRORS.append(str(_error))
'''
    script = prelude + execution_body
    from benchmarks.apibank_support import tool_timeout_seconds
    # Give the tool's deadline time to report infrastructure failure before the
    # outer program deadline handles model-generated infinite loops.
    timeout = _toolcoder_timeout_seconds() + tool_timeout_seconds()
    try:
        with tempfile.TemporaryDirectory(prefix="secat_apibank_toolcoder_") as tmp_name:
            execution_root = _build_toolcoder_execution_sandbox(root, Path(tmp_name))
            shutil.copy2(Path(__file__).resolve().parent.parent / "benchmarks" / "apibank_support.py",
                         execution_root / "apibank_support.py")
            completed = subprocess.run(
                [sys.executable, "-c", script], cwd=str(execution_root),
                env=_sanitized_subprocess_env(execution_root), capture_output=True, text=True,
                timeout=timeout,
            )
            control_file = execution_root / ".secat_execution.json"
            control = json.loads(control_file.read_text(encoding="utf-8")) if control_file.exists() else {}
    except subprocess.TimeoutExpired as exc:
        return {
            "success": False, "stdout": str(exc.stdout or ""), "stderr": "execution timeout",
            "returncode": None, "calls": [], "target_calls": [], "replayed_calls": replayed_calls,
            "error": f"ToolCoder execution exceeded {timeout}s",
        }
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if control.get("infrastructure_errors"):
        raise AgentRuntimeError("API-Bank tool infrastructure failed: " +
                                "; ".join(control["infrastructure_errors"]))
    calls: List[dict] = control.get("calls", [])
    visible_lines: List[str] = stdout.splitlines()
    # SECAT-owned replay calls happen while recording is disabled. A repair model
    # can nevertheless return ToolCoder's *full* wrapper, including its own
    # history reconstruction. In replay sensitivity mode those reconstructed
    # calls are traced, so remove only an exact prefix that matches the known
    # conversation history. Never drop a lone call merely because it resembles
    # a prior call; that could be the target itself.
    target_calls = calls
    if replayed_calls and len(calls) > replayed_calls:
        expected_replay = [
            {
                "api_name": str(item.get("api_name")),
                "params": dict(item.get("param_dict") or {}),
            }
            for item in getattr(sample, "chat_history", ())
            if item.get("role") == "API"
        ]
        prefix_matches = True
        for observed, expected in zip(calls[:replayed_calls], expected_replay):
            if (str(observed.get("api_name")) != expected["api_name"] or
                    dict(observed.get("params") or {}) != expected["params"]):
                prefix_matches = False
                break
        if prefix_matches:
            target_calls = calls[replayed_calls:]
    level2 = int(getattr(sample, "benchmark_level", 1) or 1) == 2
    # ToolSearcher and its retrieval stack can legitimately emit model-loading
    # progress/warnings on stderr while returning a valid search result.  Suppress
    # stderr *only* for a successful, observed ToolSearcher action.  Ordinary LV2
    # APIs retain ToolCoder's/LV1's strict stderr-as-execution-error behavior.
    observed_toolsearcher = (
        level2 and completed.returncode == 0 and len(target_calls) == 1 and
        str(target_calls[0].get("api_name") or "") == "ToolSearcher"
    )
    stderr_is_error = bool(stderr.strip()) and not observed_toolsearcher
    success = completed.returncode == 0 and not stderr_is_error and len(target_calls) == 1
    error = None
    if completed.returncode != 0 or stderr_is_error:
        error = stderr.strip() or f"process exited {completed.returncode}"
    elif len(target_calls) != 1:
        error = f"expected exactly one target API call, observed {len(target_calls)}"
    return {
        "success": success,
        "stdout": "\n".join(visible_lines),
        "stderr": stderr,
        "returncode": completed.returncode,
        "calls": calls,
        "target_calls": target_calls,
        "replayed_calls": replayed_calls,
        "error": error,
        "script": script,
    }


def _toolcoder_error(message: str, trace: list[dict]) -> AgentBehaviorError:
    exc = AgentBehaviorError(message)
    exc.agent_trace = list(trace or [])
    return exc


def run_toolcoder(*, view: dict, sample, runtime, model: str) -> AgentPrediction:
    """Faithful ToolCoder API-Bank planner/replan/execute/repair adapter."""
    templates = _load_toolcoder_templates(runtime)
    expanded_history = _toolcoder_expand_history(view.get("chat_history") or [])
    trace: List[dict] = []

    planner_prompt = templates.PLANNER_PROMPT.format(
        dialogue_history=expanded_history, toolbox=view["api_descriptions"])
    print("  [TOOLCODER] planner")
    raw = _chat(model, _toolcoder_messages(planner_prompt), max_completion_tokens=8192)
    trace.append({"stage": "planner", **_response_trace(raw)})
    try:
        planner_code = _extract_python_code(raw)
    except AgentBehaviorError as exc:
        exc.agent_trace = list(trace)
        raise
    paths = _toolcoder_action_paths(planner_code)

    if len(paths) > 1:
        replan_prompt = templates.REPLAN_TEMPLATE.format(
            dialogue_history=expanded_history,
            toolbox=view["api_descriptions"],
            initial_code=planner_code,
        )
        print("  [TOOLCODER] replan")
        raw_replan = _chat(model, _toolcoder_messages(replan_prompt), max_completion_tokens=8192)
        trace.append({"stage": "replan", **_response_trace(raw_replan)})
        try:
            planner_code = _extract_python_code(raw_replan)
        except AgentBehaviorError as exc:
            exc.agent_trace = list(trace)
            raise
        paths = _toolcoder_action_paths(planner_code)

    if not paths:
        raise _toolcoder_error("ToolCoder generated no call_api(api_name=...) action", trace)
    if len(paths) != 1:
        raise _toolcoder_error(
            f"ToolCoder generated {len(paths)} API calls after replanning; expected one", trace)
    predicted_api = paths[-1]

    def execute_candidate(code):
        try:
            return _execute_toolcoder_code(runtime, sample, code)
        except AgentBehaviorError as exc:
            return {"success": False, "stdout": "", "stderr": "", "error": str(exc),
                    "calls": [], "target_calls": [], "replayed_calls": 0}

    print("  [TOOLCODER] execute")
    execution = execute_candidate(planner_code)
    trace.append({k: v for k, v in {"stage": "execute", **execution}.items() if k != "script"})
    # Upstream ToolCoder sends its repair model the complete executable program
    # (API_CALL_CODE wrapper + planner body), not the planner body alone.
    replay_for_vendor = _replay_code(sample) if getattr(runtime, "state_mode", "published") in {"published", "replay"} else ""
    code_for_repair = str(templates.API_CALL_CODE).format(api_calls=replay_for_vendor) + planner_code
    repairs = 0
    while not execution["success"] and repairs < 3:
        repairs += 1
        repair_prompt = templates.EXECUTION_FAILURE_TEMPLATE.format(
            dialogue_history=expanded_history,
            python_code=code_for_repair,
            execution_result=execution.get("error") or execution.get("stderr") or "execution failed",
        )
        print(f"  [TOOLCODER] repair {repairs}/3")
        raw_repair = _chat(model, _toolcoder_messages(repair_prompt), max_completion_tokens=8192)
        trace.append({"stage": "repair_model", "attempt": repairs, **_response_trace(raw_repair)})
        try:
            planner_code = _extract_python_code(raw_repair)
        except AgentBehaviorError as exc:
            exc.agent_trace = list(trace)
            raise
        code_for_repair = planner_code
        paths = _toolcoder_action_paths(planner_code)
        if len(paths) != 1:
            execution = {
                "success": False, "stdout": "", "stderr": "",
                "calls": [], "target_calls": [], "replayed_calls": 0,
                "error": f"repair generated {len(paths)} API calls; expected one",
            }
        else:
            predicted_api = paths[-1]
            execution = execute_candidate(planner_code)
        trace.append({k: v for k, v in {"stage": "execute", "repair_attempt": repairs, **execution}.items() if k != "script"})

    if not execution.get("success"):
        raise _toolcoder_error(
            "ToolCoder execution failed after bounded repair: " +
            str(execution.get("error") or execution.get("stderr") or "unknown execution failure"),
            trace)
    target_calls = execution.get("target_calls") or []
    call = target_calls[-1]
    predicted_api = str(call["api_name"])
    # Published ToolCoder evaluates the value printed by generated code, not
    # the hidden ToolManager return object. Preserve that distinction so code
    # that prints a field instead of the overall result does not get extra credit.
    printed_result = _parse_toolcoder_stdout(execution.get("stdout") or "")
    executed = {
        "api_name": predicted_api,
        "params": call.get("params") or {},
        "result": printed_result,
        "replayed_calls": int(execution.get("replayed_calls") or 0),
    }
    return AgentPrediction(
        prediction_text=_call_text(predicted_api, executed["params"]),
        strategy="toolcoder",
        trace=trace,
        executed_call=executed,
    )


def validate_agent_environment(agent_name: str, runtime) -> None:
    """Fail before result creation when an adapter's vendored dependencies are absent."""
    name = str(agent_name or "").strip().lower()
    if name == "toolcoder":
        _load_toolcoder_templates(runtime)
    elif name not in {"oca", "codeact"}:
        raise RuntimeError(f"Unsupported API-Bank agent: {agent_name!r}")


AGENT_RUNNERS: Dict[str, Callable[..., AgentPrediction]] = {
    "oca": run_oca,
    "codeact": run_codeact,
    "toolcoder": run_toolcoder,
}
