"""Provider-safe LLM routing for SECAT.

This module keeps agent implementations provider-agnostic.  Agents continue to
use the OpenAI-compatible ``client.chat.completions.create(...)`` interface,
while this router chooses the correct backend from the model name.

Supported providers:
  - OpenAI: any model not recognized as DeepSeek (existing behavior)
  - DeepSeek: deepseek-v4-flash and deepseek-v4-pro

DeepSeek exposes an OpenAI-compatible API, but a few request fields differ.
The router translates only those provider-boundary differences; prompts,
agent control flow, benchmark state, and evaluation logic are untouched.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Tuple


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
OFFICIAL_DEEPSEEK_MODELS = {
    # DeepSeek's current release/pricing pages use deepseek-v4-flash, while
    # the current Chat Completions schema also lists deepseek-flash. Accept
    # both official spellings without silently rewriting either identifier.
    "deepseek-v4-flash",
    "deepseek-flash",
    "deepseek-v4-pro",
}


def provider_for_model(model: str) -> str:
    """Return the provider for a model identifier.

    Fail-open to the historical OpenAI path for unknown names so existing custom
    OpenAI/Azure-compatible aliases are not silently reclassified.  DeepSeek is
    recognized only by its explicit model-name namespace.
    """
    name = str(model or "").strip().lower()
    if name.startswith("deepseek-") or name in {"deepseek-chat", "deepseek-reasoner"}:
        return "deepseek"
    return "openai"


def model_slug(model: str) -> str:
    """Filesystem-safe, deterministic model slug for result isolation."""
    import re
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model or "model").strip())
    value = value.strip("._-") or "model"
    return value[:100]


def _normalize_bool_word(value: str, *, default: str = "enabled") -> str:
    word = str(value or default).strip().lower()
    aliases = {
        "1": "enabled", "true": "enabled", "yes": "enabled", "on": "enabled",
        "0": "disabled", "false": "disabled", "no": "disabled", "off": "disabled",
    }
    word = aliases.get(word, word)
    if word not in {"enabled", "disabled"}:
        raise RuntimeError(
            "DEEPSEEK_THINKING must be enabled or disabled "
            f"(got {value!r})")
    return word


def _normalize_deepseek_effort(value: str) -> str:
    word = str(value or "high").strip().lower()
    # DeepSeek maps medium/xhigh to high. Normalize here so the persisted runtime
    # configuration is explicit and identical across agents.
    if word in {"medium", "xhigh"}:
        word = "high"
    if word not in {"low", "high", "max"}:
        raise RuntimeError(
            "DEEPSEEK_REASONING_EFFORT must be low, high, or max "
            f"(got {value!r})")
    return word


def _normalize_deepseek_top_p(value: str | None) -> float:
    """Normalize DeepSeek thinking-mode nucleus sampling.

    DeepSeek V4 documents top_p as active in thinking mode with an effective
    lower bound of 0.95. SECAT makes the benchmark value explicit instead of
    relying on a provider default that may change between model releases.
    """
    raw = "0.95" if value is None or not str(value).strip() else str(value).strip()
    try:
        top_p = float(raw)
    except Exception as exc:
        raise RuntimeError(f"DEEPSEEK_TOP_P must be a number in [0.95, 1.0] (got {value!r})") from exc
    if not (0.95 <= top_p <= 1.0):
        raise RuntimeError(f"DEEPSEEK_TOP_P must be in [0.95, 1.0] (got {value!r})")
    return top_p


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except Exception as exc:
        raise RuntimeError(f"{name} must be a positive integer (got {raw!r})") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer (got {raw!r})")
    return value


def _normalize_openai_effort(value: str | None) -> str | None:
    """Normalize an explicitly requested GPT-5.4 reasoning regime.

    GPT-5.4 Mini defaults to ``none`` at the provider. SECAT keeps historical
    behavior when OPENAI_REASONING_EFFORT is unset, but benchmark runners may
    opt into a fixed effort so all compared agents receive the same reasoning
    budget.
    """
    if value is None or not str(value).strip():
        return None
    word = str(value).strip().lower()
    if word not in {"none", "low", "medium", "high", "xhigh"}:
        raise RuntimeError(
            "OPENAI_REASONING_EFFORT must be none, low, medium, high, or xhigh "
            f"(got {value!r})")
    return word


def provider_settings(model: str) -> dict:
    provider = provider_for_model(model)
    if provider == "deepseek":
        normalized_model = str(model or "").strip().lower()
        if normalized_model not in OFFICIAL_DEEPSEEK_MODELS:
            legacy = {"deepseek-chat", "deepseek-reasoner"}
            if normalized_model in legacy:
                raise RuntimeError(
                    f"DeepSeek legacy model {model!r} is no longer supported by the current API. "
                    "Use deepseek-v4-flash or deepseek-v4-pro.")
            raise RuntimeError(
                f"Unsupported DeepSeek model {model!r}. Supported text models are "
                "deepseek-v4-flash (also documented as deepseek-flash) and deepseek-v4-pro.")
        thinking = _normalize_bool_word(os.getenv("DEEPSEEK_THINKING", "enabled"))
        effort = (_normalize_deepseek_effort(os.getenv("DEEPSEEK_REASONING_EFFORT", "high"))
                  if thinking == "enabled" else None)
        return {
            "provider": "deepseek",
            "model": str(model),
            "base_url": os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL).rstrip("/"),
            "key_env": "DEEPSEEK_API_KEY",
            "thinking": thinking,
            "reasoning_effort": effort,
            "top_p": (_normalize_deepseek_top_p(os.getenv("DEEPSEEK_TOP_P"))
                      if thinking == "enabled" else None),
            "reasoning_min_tokens": (_positive_int_env(
                "DEEPSEEK_REASONING_MIN_TOKENS",
                32768 if effort == "max" else 16384)
                if thinking == "enabled" else None),
        }
    effort = _normalize_openai_effort(os.getenv("OPENAI_REASONING_EFFORT"))
    return {
        "provider": "openai",
        "model": str(model),
        # Preserve the historical SECAT OpenAI client construction exactly.
        # v4.1.52 did not consult OPENAI_BASE_URL.
        "base_url": None,
        "key_env": "OPENAI_API_KEY",
        "thinking": ("enabled" if effort and effort != "none" else "disabled" if effort == "none" else None),
        "reasoning_effort": effort,
        "top_p": None,
        "reasoning_min_tokens": (_positive_int_env("OPENAI_REASONING_MIN_TOKENS", 8192)
                                 if effort not in (None, "none") else None),
    }




def model_run_slug(model: str) -> str:
    """Filesystem-safe slug for a model *and its provider reasoning regime*.

    DeepSeek includes thinking mode/effort and OpenAI includes an explicitly
    configured reasoning effort, so rerunning the same model under a different
    reasoning regime cannot overwrite or mix experimental artifacts.
    """
    base = model_slug(model)
    settings = provider_settings(model)
    if settings["provider"] == "deepseek":
        if settings["thinking"] == "enabled":
            return f"{base}_think_{settings['reasoning_effort']}"
        return f"{base}_think_disabled"
    if settings.get("reasoning_effort"):
        return f"{base}_reason_{settings['reasoning_effort']}"
    return base


def public_model_settings(model: str) -> dict:
    """Non-secret model configuration safe to persist in experiment artifacts."""
    settings = dict(provider_settings(model))
    return {
        "provider": settings["provider"],
        "model": settings["model"],
        "base_url": settings["base_url"],
        "thinking": settings["thinking"],
        "reasoning_effort": settings["reasoning_effort"],
        "top_p": settings.get("top_p"),
        "reasoning_min_tokens": settings.get("reasoning_min_tokens"),
    }

def require_provider_key(model: str) -> str:
    settings = provider_settings(model)
    key = os.getenv(settings["key_env"], "").strip()
    if not key:
        raise RuntimeError(
            f"{settings['key_env']} is missing for model {model!r}. "
            "Add it to the project .env file; do not hard-code credentials in SECAT.")
    return key


def prepare_chat_kwargs(model: str, kwargs: dict) -> dict:
    """Translate only provider-level request differences.

    OpenAI requests may receive an explicit ``reasoning_effort`` and compatible
    completion ceiling. DeepSeek Chat Completions currently uses ``max_tokens``
    rather than OpenAI's ``max_completion_tokens`` and uses
    ``extra_body.thinking`` + ``reasoning_effort`` for thinking control.
    """
    out = dict(kwargs)
    # Internal SECAT-only override used by narrow benchmark adapters that need a
    # lower structured-output ceiling than the provider-wide DeepSeek default.
    # Pop it before forwarding so it can never leak into the provider request.
    structured_floor_override = out.pop("_secat_structured_min_tokens", None)
    # Internal retry-policy marker is consumed by ProviderResilientClient when
    # that wrapper is present. Pop defensively here so it can never leak to a
    # provider if wrapper ordering changes.
    out.pop("_secat_empty_json_retries", None)
    provider = provider_for_model(model)
    if provider == "openai":
        settings = provider_settings(model)
        effort = settings.get("reasoning_effort")
        if effort is not None:
            # GPT-5.4 Chat Completions accepts reasoning_effort directly.
            out["reasoning_effort"] = effort
            # GPT-5.4 sampling parameters are only accepted with effort=none.
            if effort != "none":
                out.pop("temperature", None)
                out.pop("top_p", None)
                out.pop("logprobs", None)
                # Reasoning tokens count against max_completion_tokens. A tiny
                # structured-output cap can otherwise end before final JSON is
                # emitted. This is a ceiling, not a pre-spend.
                if "max_completion_tokens" in out:
                    floor = int(settings.get("reasoning_min_tokens") or 8192)
                    out["max_completion_tokens"] = max(
                        int(out.get("max_completion_tokens") or 0), floor)
        return out

    if "max_completion_tokens" in out:
        # DeepSeek Chat Completions uses max_tokens. If a caller supplied both,
        # preserve the explicit DeepSeek-native max_tokens and remove the OpenAI-only alias.
        if "max_tokens" not in out:
            out["max_tokens"] = out["max_completion_tokens"]
        out.pop("max_completion_tokens", None)

    # DeepSeek's current Chat Completions request schema does not expose OpenAI's
    # multi-choice `n` parameter. SECAT only ever requests n=1, so dropping that
    # redundant field is semantics-preserving. Fail closed for any future n>1 use.
    if "n" in out:
        n_value = out.pop("n")
        if n_value not in (None, 1):
            raise RuntimeError(
                f"DeepSeek Chat Completions does not support SECAT n={n_value!r}; only n=1 is supported")

    messages = out.get("messages") or []
    if any(isinstance(m, dict) and str(m.get("role") or "").lower() == "developer" for m in messages):
        raise RuntimeError("DeepSeek Chat Completions does not support the developer role; use system role")

    settings = provider_settings(model)
    thinking = settings["thinking"]

    # In DeepSeek thinking mode, max_tokens is a cap over generated output that
    # includes reasoning. OCA's compact OpenAI-era structured-output caps (often
    # 220-1800) can therefore be exhausted before any final JSON is emitted.
    # Raising the *cap* does not pre-spend tokens; it only prevents provider-
    # specific truncation. Apply this only to JSON-structured requests and keep
    # any larger caller-provided cap unchanged.
    response_format = out.get("response_format")
    is_json_request = (
        isinstance(response_format, dict) and
        str(response_format.get("type") or "").lower() == "json_object"
    )
    if thinking == "enabled" and is_json_request and "max_tokens" in out:
        default_floor = 32768 if settings.get("reasoning_effort") == "max" else 16384
        raw_floor = (structured_floor_override if structured_floor_override is not None
                     else os.getenv("DEEPSEEK_STRUCTURED_MIN_TOKENS", str(default_floor)))
        try:
            floor = int(raw_floor)
        except Exception as exc:
            name = ("_secat_structured_min_tokens" if structured_floor_override is not None
                    else "DEEPSEEK_STRUCTURED_MIN_TOKENS")
            raise RuntimeError(
                f"{name} must be a positive integer (got {raw_floor!r})") from exc
        if floor <= 0:
            name = ("_secat_structured_min_tokens" if structured_floor_override is not None
                    else "DEEPSEEK_STRUCTURED_MIN_TOKENS")
            raise RuntimeError(f"{name} must be a positive integer (got {raw_floor!r})")
        out["max_tokens"] = max(int(out.get("max_tokens") or 0), floor)

    extra_body = dict(out.get("extra_body") or {})
    # Preserve caller-provided extra_body fields, but make SECAT's run-level
    # thinking setting authoritative so all three agents use the same regime.
    extra_body["thinking"] = {"type": thinking}
    out["extra_body"] = extra_body

    if thinking == "enabled":
        # SECAT currently uses prompt-based code/action interaction rather than
        # OpenAI function tools. DeepSeek thinking-mode tool calls require replaying
        # reasoning_content on every subsequent tool turn; fail closed if a future
        # caller adds tools before that replay protocol is implemented.
        if out.get("tools"):
            raise RuntimeError(
                "DeepSeek thinking-mode tool calls require reasoning_content replay; "
                "SECAT's provider adapter intentionally refuses this unsupported path")
        out["reasoning_effort"] = settings["reasoning_effort"]
        # V4 thinking mode ignores temperature/presence/frequency penalties but
        # *does* honor top_p with an effective lower bound of 0.95. Make the
        # benchmark value explicit and identical across all compared agents.
        out.pop("temperature", None)
        out["top_p"] = settings.get("top_p", 0.95)
        out.pop("presence_penalty", None)
        out.pop("frequency_penalty", None)
    else:
        out.pop("reasoning_effort", None)

    return out


class RoutedClient:
    """OpenAI-compatible client facade that dispatches each call by model name."""

    def __init__(self, client_factory: Optional[Callable[..., Any]] = None):
        self._client_factory = client_factory
        self._clients: Dict[Tuple[str, str, Optional[str]], Any] = {}
        self.chat = _RoutedChat(self)

    def _factory(self):
        if self._client_factory is not None:
            return self._client_factory
        # Lazy import keeps offline/unit-test import paths independent of the SDK.
        from openai import OpenAI
        return OpenAI

    def _client_for_model(self, model: str):
        settings = provider_settings(model)
        provider = settings["provider"]
        key = require_provider_key(model)
        base_url = settings["base_url"]
        cache_key = (provider, key, base_url)
        if cache_key in self._clients:
            return self._clients[cache_key]

        factory = self._factory()
        kwargs = {"api_key": key}
        if base_url:
            kwargs["base_url"] = base_url
        client = factory(**kwargs)
        self._clients[cache_key] = client
        return client


class _RoutedChat:
    def __init__(self, owner: RoutedClient):
        self.completions = _RoutedCompletions(owner)


class _RoutedCompletions:
    def __init__(self, owner: RoutedClient):
        self._owner = owner

    def create(self, **kwargs):
        model = str(kwargs.get("model") or "").strip()
        if not model:
            raise RuntimeError("LLM request is missing the model name")
        client = self._owner._client_for_model(model)
        prepared = prepare_chat_kwargs(model, kwargs)
        return client.chat.completions.create(**prepared)


class DeepSeekEmptyJSONError(RuntimeError):
    """Raised after bounded retries when DeepSeek JSON mode returns no content."""


def _response_content(resp: Any) -> str:
    try:
        return str(resp.choices[0].message.content or "")
    except Exception:
        return ""


def _is_deepseek_json_request(kwargs: dict) -> bool:
    model = str(kwargs.get("model") or "").strip()
    if provider_for_model(model) != "deepseek":
        return False
    fmt = kwargs.get("response_format")
    return isinstance(fmt, dict) and str(fmt.get("type") or "").lower() == "json_object"


class ProviderResilientClient:
    """Thin provider-quirk guard wrapped *outside* the token meter.

    DeepSeek documents that JSON mode may occasionally return empty content.
    SECAT gives such a request one bounded retry. Because this wrapper sits
    outside ``MeteredClient``, both attempts remain fully accounted for.
    """

    def __init__(self, inner):
        self._inner = inner
        self.chat = _ProviderResilientChat(inner)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _ProviderResilientChat:
    def __init__(self, inner):
        self.completions = _ProviderResilientCompletions(inner)


class _ProviderResilientCompletions:
    def __init__(self, inner):
        self._inner = inner

    def create(self, **kwargs):
        kwargs = dict(kwargs)
        # Default behavior remains one bounded empty-JSON retry. Narrow benchmark
        # adapters may set this internal marker to 0 when they need to inspect the
        # first response (for example, to distinguish token exhaustion from an
        # occasional provider-side empty JSON response).
        raw_retries = kwargs.pop("_secat_empty_json_retries", 1)
        try:
            empty_retries = int(raw_retries)
        except Exception as exc:
            raise RuntimeError("_secat_empty_json_retries must be an integer") from exc
        if empty_retries < 0 or empty_retries > 1:
            raise RuntimeError("_secat_empty_json_retries must be 0 or 1")

        resp = self._inner.chat.completions.create(**kwargs)
        if not _is_deepseek_json_request(kwargs) or _response_content(resp).strip():
            return resp
        if empty_retries == 0:
            return resp

        # One retry only: enough to cover DeepSeek's documented occasional
        # empty-JSON response without turning a provider fault into an unbounded
        # cost/retry loop. Higher-level SECAT structured-call fallbacks remain
        # responsible for the terminal failure path.
        retry = self._inner.chat.completions.create(**kwargs)
        if _response_content(retry).strip():
            return retry
        raise DeepSeekEmptyJSONError(
            "DeepSeek JSON mode returned empty content on both bounded attempts")


def make_routed_client() -> RoutedClient:
    return RoutedClient()
