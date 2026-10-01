"""Provider-semantic registry for OCA's typed capability compiler.

The compiler consumes a small declarative interface exported by each provider
module.  Provider modules contain API ontology only: resources, semantic fields,
relations, lookup routes and population routes.  They never contain task ids,
benchmark gold paths, entity examples or expected answers.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class ProviderSemantics:
    provider: str
    catalog: Any
    resource_types: frozenset[str]
    semantic_fields: dict[str, Any]
    find_routes: dict[str, Any]
    population_routes: dict[tuple[str, str], Any]
    population_aliases: dict[str, str]
    resource_aliases: dict[str, str]
    placeholder_resource: dict[str, str]
    action_routes: dict[str, Any]
    detail_fields: dict[str, tuple[str, ...]]


def _runtime_provider(benchmark: str) -> str:
    """Resolve benchmark aliases (e.g. tmdb_verified) to a provider module."""
    name = str(benchmark or "").strip()
    if not name:
        return ""
    try:
        import benchmarks as B
        spec = B.get_benchmark(name)
        return str(spec.get("runtime_api") or spec.get("name") or name).casefold()
    except Exception:
        return name.casefold().split("_", 1)[0]


def get_provider_semantics(benchmark: str) -> ProviderSemantics | None:
    provider = _runtime_provider(benchmark)
    if not provider:
        return None
    try:
        mod = import_module(f"oca.capability.{provider}")
    except Exception:
        return None
    view = {}
    profile_view = getattr(mod, "profile_view", None)
    if callable(profile_view):
        try:
            view = dict(profile_view() or {})
        except Exception:
            view = {}
    catalog = view.get("catalog") or getattr(mod, "CATALOG", None)
    if catalog is None:
        return None
    return ProviderSemantics(
        provider=provider,
        catalog=catalog,
        resource_types=frozenset(getattr(mod, "RESOURCE_TYPES", ())),
        semantic_fields=dict(view.get("semantic_fields") or getattr(mod, "SEMANTIC_FIELDS", {})),
        find_routes=dict(getattr(mod, "FIND_ROUTES", {})),
        population_routes=dict(view.get("population_routes") or getattr(mod, "POPULATION_ROUTES", {})),
        population_aliases={str(k).casefold(): str(v) for k, v in dict(getattr(mod, "POPULATION_ALIASES", {})).items()},
        resource_aliases={str(k).casefold(): str(v) for k, v in dict(getattr(mod, "RESOURCE_ALIASES", {})).items()},
        placeholder_resource=dict(getattr(mod, "PLACEHOLDER_RESOURCE", {})),
        action_routes=dict(view.get("action_routes") or getattr(mod, "ACTION_ROUTES", {})),
        detail_fields={str(k): tuple(str(x) for x in v) for k, v in dict(getattr(mod, "DETAIL_FIELDS", {})).items()},
    )


def route_endpoint(spec: Any) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        return str(spec.get("endpoint") or spec.get("path") or "")
    return ""


def route_metadata(spec: Any) -> dict[str, Any]:
    if isinstance(spec, str):
        return {"endpoint": spec}
    return dict(spec) if isinstance(spec, dict) else {}
