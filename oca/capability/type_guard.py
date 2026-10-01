"""Semantic type checks for evidence comparisons and certification.

Ordering and difference operations are meaningful only when their operands
resolve to compatible canonical semantic fields. These checks operate on typed
lineage metadata rather than question wording or benchmark answers.
"""

from __future__ import annotations

import re
from typing import Any

from .types import SEMANTIC_FIELDS, canonical_field

STATUS = "semantic_type_violation"

#: Endpoint shape -> resource type of the records it returns.
_ENDPOINT_RESOURCE: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"/person/\{[^}]+\}/movie_credits$"), "movie"),
    (re.compile(r"/person/\{[^}]+\}/tv_credits$"), "tv"),
    (re.compile(r"/person/\{[^}]+\}/images$"), "image"),
    (re.compile(r"/person/\{[^}]+\}$"), "person"),
    (re.compile(r"/search/person$"), "person"),
    (re.compile(r"/person/popular$"), "person"),
    (re.compile(r"/(movie|tv)/\{[^}]+\}/credits$"), "person"),
    (re.compile(r"/season/\{[^}]+\}/credits$"), "person"),
    (re.compile(r"/episode/\{[^}]+\}/credits$"), "person"),
    (re.compile(r"/movie/\{[^}]+\}/(similar|recommendations)$"), "movie"),
    (re.compile(r"/movie/\{[^}]+\}/keywords$"), "keyword"),
    (re.compile(r"/movie/\{[^}]+\}/reviews$"), "review"),
    (re.compile(r"/movie/\{[^}]+\}/images$"), "image"),
    (re.compile(r"/movie/\{[^}]+\}/release_dates$"), "movie"),
    (re.compile(r"/movie/\{[^}]+\}$"), "movie"),
    (re.compile(r"/movie/(popular|top_rated|now_playing|upcoming|latest)$"), "movie"),
    (re.compile(r"/search/movie$"), "movie"),
    (re.compile(r"/discover/movie$"), "movie"),
    (re.compile(r"/tv/\{[^}]+\}/(similar|recommendations)$"), "tv"),
    (re.compile(r"/tv/\{[^}]+\}/keywords$"), "keyword"),
    (re.compile(r"/tv/\{[^}]+\}/reviews$"), "review"),
    (re.compile(r"/tv/\{[^}]+\}/images$"), "image"),
    (re.compile(r"/season/\{[^}]+\}/images$"), "image"),
    (re.compile(r"/episode/\{[^}]+\}/images$"), "image"),
    (re.compile(r"/episode/\{[^}]+\}$"), "episode"),
    (re.compile(r"/season/\{[^}]+\}$"), "season"),
    (re.compile(r"/tv/\{[^}]+\}$"), "tv"),
    (re.compile(r"/tv/(popular|top_rated|on_the_air|airing_today|latest)$"), "tv"),
    (re.compile(r"/search/tv$"), "tv"),
    (re.compile(r"/discover/tv$"), "tv"),
    (re.compile(r"/collection/\{[^}]+\}/images$"), "image"),
    (re.compile(r"/collection/\{[^}]+\}$"), "collection"),
    (re.compile(r"/search/collection$"), "collection"),
    (re.compile(r"/company/\{[^}]+\}/images$"), "image"),
    (re.compile(r"/company/\{[^}]+\}$"), "company"),
    (re.compile(r"/search/company$"), "company"),
    (re.compile(r"/network/\{[^}]+\}/images$"), "image"),
    (re.compile(r"/network/\{[^}]+\}$"), "network"),
    (re.compile(r"/trending/"), "any"),
]

_ORDERING = {"max", "min", "gt", "gte", "lt", "lte", "argmax", "argmin"}


def resource_of_endpoint(endpoint: str) -> str:
    """Resource type of the records an endpoint returns."""
    ep = re.sub(r"^\s*(?:GET|POST|PUT|DELETE)\s+", "", str(endpoint or ""),
                flags=re.I).rstrip("/")
    for pattern, resource in _ENDPOINT_RESOURCE:
        if pattern.search(ep):
            return resource
    return ""


def _step_endpoint(plan: dict[str, Any], step_id: str) -> str:
    for step in (plan.get("steps") or []):
        if str(step.get("id") or "") == str(step_id):
            return str(step.get("endpoint") or "")
    return ""


def _leaf_field(field: Any) -> str:
    """Strip a record-root prefix: ``cast[*].birthday`` -> ``birthday``."""
    text = str(field or "").replace("[*]", "")
    return text.split(".")[-1].strip()


def operand_pairs(derivation: dict[str, Any], der_index: dict[str, dict],
                  plan: dict[str, Any], *, _depth: int = 0
                  ) -> list[tuple[str, str, str]]:
    """Resolve a derivation's operands to ``(canonical_field, resource, id)``.

    Recurses through ``source_derivations`` until it reaches derivations that
    project a field from a step, which is where semantic type actually lives.
    """
    if _depth > 8:
        return []
    out: list[tuple[str, str, str]] = []
    did = str(derivation.get("id") or "")

    parents = [str(x) for x in (derivation.get("source_derivations") or [])]
    if parents:
        for parent_id in parents:
            parent = der_index.get(parent_id)
            if isinstance(parent, dict):
                out.extend(operand_pairs(parent, der_index, plan,
                                         _depth=_depth + 1))
        if out:
            return out

    field = _leaf_field(derivation.get("field"))
    canon = canonical_field(field) if field else None
    for step_id in (derivation.get("source_steps") or []):
        resource = resource_of_endpoint(_step_endpoint(plan, str(step_id)))
        if canon and resource:
            out.append((canon, resource, did))
    return out


def check_comparison_types(plan: dict[str, Any], der_index: dict[str, dict],
                           cited_derivations: list[dict[str, Any]]
                           ) -> list[str]:
    """Return a violation message for each ill-typed ordering comparison."""
    violations: list[str] = []
    for derivation in cited_derivations:
        operation = str(derivation.get("operation")
                        or derivation.get("operator") or "").lower()
        if operation not in {"compare", "difference"}:
            continue
        mode = str(derivation.get("comparison_mode") or "").lower()
        if not mode and isinstance(derivation.get("value"), dict):
            mode = str((derivation.get("value") or {}).get("comparison")
                       or "").lower()
        if operation == "compare" and mode and mode not in _ORDERING:
            continue  # equality of entities is not an ordering comparison

        pairs = operand_pairs(derivation, der_index, plan)
        if len(pairs) < 2:
            continue

        fields = {c for c, _r, _d in pairs}
        if len(fields) > 1:
            detail = ", ".join(f"{c} on {r}" for c, r, _d in pairs)
            violations.append(
                f"derivation {derivation.get('id')} compares different "
                f"semantic fields ({detail}); an ordering comparison requires "
                "one semantic field across all operands")
            continue

        canon = next(iter(fields))
        spec = SEMANTIC_FIELDS.get(canon)
        if spec is None:
            continue
        bad = [(c, r) for c, r, _d in pairs if spec.field_for(r) is None]
        if bad:
            detail = ", ".join(f"{c} on {r}" for c, r in bad)
            violations.append(
                f"derivation {derivation.get('id')} projects {canon} from a "
                f"resource that does not carry it ({detail}); valid owners: "
                f"{sorted(spec.owners)}")
    return violations


