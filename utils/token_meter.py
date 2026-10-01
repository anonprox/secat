"""
utils/token_meter.py — uniform token accounting across all agents.

The public top-level counters are backwards compatible with the original SECAT
meter.  OCA can additionally mark an LLM call with ``stage(...)`` so a run can
separate planner, acquisition, answer, and repair costs without changing the
request sent to the model.
"""
from __future__ import annotations

from contextlib import contextmanager
import threading

_lock = threading.Lock()
_tls = threading.local()


def _blank():
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
        # Provider-neutral detailed accounting. These remain zero for providers
        # that do not expose the fields, preserving all historical totals.
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
        "reasoning_tokens": 0,
    }


_state = _blank()
_state["by_stage"] = {}


def reset():
    with _lock:
        _state.clear()
        _state.update(_blank())
        _state["by_stage"] = {}
    _tls.stage = None


def read():
    with _lock:
        out = {k: v for k, v in _state.items() if k != "by_stage"}
        out["by_stage"] = {
            name: dict(values) for name, values in _state.get("by_stage", {}).items()
        }
        return out


def current_stage() -> str:
    return str(getattr(_tls, "stage", None) or "unattributed")


@contextmanager
def stage(name: str):
    """Attribute LLM calls in this context to ``name``.

    This is deliberately thread-local and metadata-only: it does not alter model
    prompts, request parameters, or responses. Nested stages restore the previous
    label on exit.
    """
    previous = getattr(_tls, "stage", None)
    _tls.stage = str(name or "unattributed")
    try:
        yield
    finally:
        _tls.stage = previous


def _add(prompt, completion, total, calls=1, *, cache_hit=0, cache_miss=0, reasoning=0):
    values = {
        "prompt_tokens": int(prompt or 0),
        "completion_tokens": int(completion or 0),
        "total_tokens": int(total or 0),
        "calls": int(calls or 0),
        "prompt_cache_hit_tokens": int(cache_hit or 0),
        "prompt_cache_miss_tokens": int(cache_miss or 0),
        "reasoning_tokens": int(reasoning or 0),
    }
    label = current_stage()
    with _lock:
        for key, value in values.items():
            _state[key] += value
        bucket = _state.setdefault("by_stage", {}).setdefault(label, _blank())
        for key, value in values.items():
            bucket[key] += value


def _field(obj, name, default=0):
    """Read a usage field from SDK objects, dicts, or Pydantic model extras."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    value = getattr(obj, name, None)
    if value is not None:
        return value
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict) and name in extra:
        return extra.get(name, default)
    try:
        dumped = obj.model_dump()
    except Exception:
        dumped = None
    if isinstance(dumped, dict):
        return dumped.get(name, default)
    return default


def _usage_details(usage):
    prompt = int(_field(usage, "prompt_tokens", 0) or 0)
    cache_hit = int(_field(usage, "prompt_cache_hit_tokens", 0) or 0)
    cache_miss = int(_field(usage, "prompt_cache_miss_tokens", 0) or 0)

    # OpenAI-style usage can expose cached input under prompt_tokens_details;
    # DeepSeek currently exposes hit/miss at the usage top level. Supporting
    # both keeps this meter provider-neutral.
    prompt_details = _field(usage, "prompt_tokens_details", None)
    if not cache_hit:
        cache_hit = int(_field(prompt_details, "cached_tokens", 0) or 0)
    if not cache_miss and cache_hit and prompt >= cache_hit:
        cache_miss = prompt - cache_hit

    completion_details = _field(usage, "completion_tokens_details", None)
    reasoning = int(_field(completion_details, "reasoning_tokens", 0) or 0)
    return cache_hit, cache_miss, reasoning


# ── optional fallback estimator (only used if the API omits usage) ────────────
_ENC = None


def _estimate(messages, output_text):
    global _ENC
    try:
        import tiktoken
        if _ENC is None:
            _ENC = tiktoken.get_encoding("cl100k_base")
        pt = sum(len(_ENC.encode(str(m.get("content", "")))) for m in (messages or []))
        ct = len(_ENC.encode(str(output_text or "")))
        return pt, ct, pt + ct
    except Exception:
        # last-resort: ~4 chars/token
        pt = sum(len(str(m.get("content", ""))) for m in (messages or [])) // 4
        ct = len(str(output_text or "")) // 4
        return pt, ct, pt + ct


class MeteredClient:
    """Wrap an OpenAI client so every chat.completions.create() is counted."""

    def __init__(self, inner):
        self._inner = inner
        self.chat = _Chat(inner)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _Chat:
    def __init__(self, inner):
        self._inner = inner
        self.completions = _Completions(inner)


class _Completions:
    def __init__(self, inner):
        self._inner = inner

    def create(self, **kwargs):
        resp = self._inner.chat.completions.create(**kwargs)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            cache_hit, cache_miss, reasoning = _usage_details(usage)
            _add(_field(usage, "prompt_tokens", 0),
                 _field(usage, "completion_tokens", 0),
                 _field(usage, "total_tokens", 0),
                 cache_hit=cache_hit, cache_miss=cache_miss, reasoning=reasoning)
        else:
            try:
                out = resp.choices[0].message.content
            except Exception:
                out = ""
            pt, ct, tt = _estimate(kwargs.get("messages"), out)
            _add(pt, ct, tt)
        return resp
