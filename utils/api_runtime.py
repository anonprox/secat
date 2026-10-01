"""Configuration-driven API runtime adapters used by OCA.

The OCA reasoning/validation core never branches on a concrete API name. API
transport details live in the benchmark registry. For ordinary REST APIs this
module injects the configured credential automatically, so generated code never
needs to read, print, or format secrets. APIs with custom transport requirements
may still provide ``requests_compat`` in the registry.
"""
from __future__ import annotations

import importlib
import os
import urllib.parse
from typing import Any, Callable


def _resolve(spec: str | None) -> Callable | None:
    if not spec:
        return None
    module_name, sep, attr = str(spec).partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"invalid adapter callable {spec!r}; expected module:function")
    module = importlib.import_module(module_name)
    fn = getattr(module, attr)
    if not callable(fn):
        raise TypeError(f"adapter {spec!r} is not callable")
    return fn


def api_spec(name: str) -> dict[str, Any]:
    import benchmarks as B
    return B.get_benchmark(str(name))


def wire_endpoint_template(base_url: str, endpoint: str) -> str:
    """Map an OAS endpoint path to the configured API's on-wire URL path.

    OpenAPI sources differ in how they encode a server base path. Some include
    it directly in every path (``base=/3``, endpoint ``/3/search``); others use
    standard server-relative paths (``base=/v1``, endpoint ``/search``). The
    execution guard must support both without provider-specific branches.
    """
    endpoint = str(endpoint or "/").split("?", 1)[0]
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    endpoint = endpoint.rstrip("/") or "/"
    try:
        base_path = urllib.parse.urlsplit(str(base_url or "")).path.rstrip("/")
    except Exception:
        base_path = ""
    if not base_path or base_path == "/":
        return endpoint
    if endpoint == base_path or endpoint.startswith(base_path + "/"):
        return endpoint
    return (base_path + "/" + endpoint.lstrip("/")).rstrip("/") or "/"


def _same_origin(url: str, base_url: str) -> bool:
    try:
        u = urllib.parse.urlsplit(str(url))
        b = urllib.parse.urlsplit(str(base_url))
        return bool(u.scheme and u.netloc and
                    (u.scheme.lower(), u.netloc.lower()) ==
                    (b.scheme.lower(), b.netloc.lower()))
    except Exception:
        return False


class _RegistryAuthenticatedRequests:
    """Small requests-compatible proxy with registry-owned authentication.

    Authentication is injected only for the configured API origin. This keeps
    credentials out of model-generated code and makes bearer/query-key handling
    deterministic across APIs. A custom ``requests_compat`` adapter remains the
    preferred mechanism for APIs with refresh/stateful auth.
    """
    def __init__(self, name: str, transport):
        self._name = str(name)
        self._transport = transport
        self._spec = api_spec(self._name)
        env_key = str(self._spec.get("env_key") or "")
        self._credential = os.environ.get(env_key, "") if env_key else ""
        self.exceptions = transport.exceptions
        self.RequestException = transport.RequestException

    def _auth_kwargs(self, url: str, kw: dict[str, Any]) -> dict[str, Any]:
        if not _same_origin(url, str(self._spec.get("base_url") or "")):
            return kw
        strategy = str(self._spec.get("auth_strategy") or "none").lower()
        if strategy == "none":
            return kw
        credential = self._credential
        if not credential:
            return kw

        out = dict(kw)
        headers = dict(out.get("headers") or {})
        params = dict(out.get("params") or {})
        auth_header_key = next((k for k in headers if str(k).lower() == "authorization"), None)

        if strategy == "bearer":
            # Registry transport owns the credential form; override stale/model-
            # formatted credentials for this API origin.
            if auth_header_key and auth_header_key != "Authorization":
                headers.pop(auth_header_key, None)
            headers["Authorization"] = f"Bearer {credential}"
            out["headers"] = headers
            return out

        if strategy == "bearer_or_query":
            prefix = str(self._spec.get("bearer_prefix") or "")
            query_name = str(self._spec.get("auth_query_param") or "api_key")
            if prefix and credential.startswith(prefix):
                if auth_header_key and auth_header_key != "Authorization":
                    headers.pop(auth_header_key, None)
                headers["Authorization"] = f"Bearer {credential}"
                params.pop(query_name, None)
                out["headers"] = headers
                if params or "params" in out:
                    out["params"] = params
            else:
                # Query-style credentials must not be accompanied by an
                # Authorization header invented by generated code; the registry
                # transport is authoritative for this API origin.
                if auth_header_key:
                    headers.pop(auth_header_key, None)
                params[query_name] = credential
                out["params"] = params
                if headers or "headers" in out:
                    out["headers"] = headers
            return out
        return out

    def _call(self, method: str, url: str, **kw):
        fn = getattr(self._transport, method.lower())
        return fn(url, **self._auth_kwargs(url, kw))

    def get(self, url, **kw): return self._call("GET", url, **kw)
    def post(self, url, **kw): return self._call("POST", url, **kw)
    def put(self, url, **kw): return self._call("PUT", url, **kw)
    def delete(self, url, **kw): return self._call("DELETE", url, **kw)
    def patch(self, url, **kw): return self._call("PATCH", url, **kw)
    def __getattr__(self, name): return getattr(self._transport, name)


def install_requests_compat(name: str):
    import requests
    spec = api_spec(name)
    fn = _resolve(spec.get("requests_compat"))
    if fn:
        # Custom adapters own transport/auth semantics (for example token refresh).
        return fn()
    return _RegistryAuthenticatedRequests(name, requests)



def credential_env_key(name: str) -> str:
    """Return the registry-owned credential variable name, if any.

    Execution backends call this only in trusted startup code, after the
    transport has captured the credential and before generated code runs.
    """
    return str(api_spec(name).get("env_key") or "")

def action_certificate(name: str, ledger, plan=None):
    fn = _resolve(api_spec(name).get("action_certificate"))
    if fn is None:
        raise RuntimeError(
            f"API {name!r} executed a state-changing request but has no "
            "configured action-certificate adapter")
    return fn(ledger, plan=plan)
