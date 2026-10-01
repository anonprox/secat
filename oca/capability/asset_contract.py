"""Type-driven validation of ``answer.mode = asset``.

This module contains no benchmark wording or route dispatch.  Asset semantics are
resolved from the typed semantic-field registry and the provider capability
catalog.  The compiler and normalizer call these helpers directly; there is no
runtime monkey-patching seam.
"""
from __future__ import annotations

from typing import Any

from .catalog import CapabilityCatalog
from .types import ASSET_PATH, SEMANTIC_FIELDS, canonical_field, resolve_field, FieldTypeError

ASSET_RESOURCES = frozenset({"image", "asset"})
# Owner-specific asset concepts.  These are canonical semantic field IDs from
# the typed registry, not surface words.  Generic poster/backdrop/file-path
# requests may be satisfied through an owner's documented images capability.
STRICT_ASSET_OWNER_FIELDS = frozenset({"logo", "profile_image", "still"})


def field_is_asset(field: Any) -> bool:
    canon = canonical_field(str(field or ""))
    return bool(canon and SEMANTIC_FIELDS[canon].value_type == ASSET_PATH)


def asset_field_canonical(field: Any) -> str | None:
    canon = canonical_field(str(field or ""))
    if canon and SEMANTIC_FIELDS[canon].value_type == ASSET_PATH:
        return canon
    return None


def asset_vocabulary() -> frozenset[str]:
    """Human/provider asset labels derived only from the semantic registry."""
    words: set[str] = set()
    generic = {"of", "path", "file", "art", "background", "brand", "episode", "the", "a", "to", "image_path", "in"}
    def add(term: str) -> None:
        term = " ".join(str(term or "").replace("_", " ").casefold().split())
        if not term or term in generic:
            return
        words.add(term)
        if " " not in term and not term.endswith("s"):
            words.add(term + ("es" if term.endswith(("s","x","z","ch","sh")) else "s"))
    for name, spec in SEMANTIC_FIELDS.items():
        if spec.value_type != ASSET_PATH:
            continue
        add(name); add(name.replace("_", " "))
        for alias in spec.aliases: add(alias)
        for provider in spec.provider_field.values():
            add(provider)
            add(provider.replace("_path", ""))
    return frozenset(words)


def _node_resource(node_id: str, by_id: dict[str, dict], *, depth: int = 0) -> str:
    if depth > 20:
        return ""
    node = by_id.get(str(node_id or "")) or {}
    explicit = str(node.get("resource") or "").strip().casefold()
    op = str(node.get("op") or "")
    # project/filter/select preserve the source resource even when a model left a
    # stale resource annotation on the value node.
    if op in {"project", "filter", "select", "count", "compare", "difference", "membership", "logical_and", "logical_or"}:
        src = str(node.get("source") or node.get("left") or "")
        if src:
            inherited = _node_resource(src, by_id, depth=depth + 1)
            if inherited:
                return inherited
    return explicit


def _asset_owner(node_id: str, by_id: dict[str, dict], catalog: CapabilityCatalog | None, *, depth: int = 0) -> str:
    """Return the semantic owner of an asset lineage (movie/company/person/etc.)."""
    if depth > 20:
        return ""
    node = by_id.get(str(node_id or "")) or {}
    op = str(node.get("op") or "")
    src = str(node.get("source") or "")
    resource = str(node.get("resource") or "").strip().casefold()
    if resource in ASSET_RESOURCES:
        return _node_resource(src, by_id, depth=depth + 1) if src else ""
    if op == "relation" and catalog is not None and src:
        src_resource = _node_resource(src, by_id, depth=depth + 1)
        cap = catalog.try_lookup(src_resource, str(node.get("relation") or ""), resource or None)
        if cap is not None and cap.target_resource in ASSET_RESOURCES:
            return src_resource
    if op in {"project", "filter", "select"} and src:
        owner = _asset_owner(src, by_id, catalog, depth=depth + 1)
        if owner:
            return owner
    return _node_resource(node_id, by_id, depth=depth + 1)


def _project_is_asset(node: dict[str, Any], by_id: dict[str, dict]) -> tuple[bool, str]:
    field = asset_field_canonical(node.get("field"))
    if not field:
        return False, ""
    src = str(node.get("source") or "")
    src_resource = _node_resource(src, by_id) if src else str(node.get("resource") or "").strip().casefold()
    if field == "file_path":
        if src_resource in ASSET_RESOURCES:
            return True, field
        # file_path on an entity is not directly present, but it is a valid
        # generic image value after the owner's image capability is acquired.
        return False, "asset field requires image acquisition"
    try:
        resolve_field(field, src_resource)
        return True, field
    except FieldTypeError as exc:
        if field in STRICT_ASSET_OWNER_FIELDS:
            return False, str(exc)
        # Generic visual concepts (poster/backdrop/cover aliases) may be mapped
        # to the owner's documented image collection by the capability layer.
        return False, "asset field requires image acquisition"


def terminal_produces_asset(node_id: str, by_id: dict[str, dict],
                            catalog: CapabilityCatalog | None = None,
                            *, depth: int = 0) -> tuple[bool, str]:
    """Whether the terminal value is asset-typed, following typed lineage."""
    if depth > 20:
        return False, "lineage too deep"
    node = by_id.get(str(node_id or ""))
    if node is None:
        return False, "no such node"
    resource = str(node.get("resource") or "").strip().casefold()
    op = str(node.get("op") or "")
    if resource in ASSET_RESOURCES:
        return True, f"node {node_id} produces resource {resource!r}"
    if op == "project":
        ok, detail = _project_is_asset(node, by_id)
        if ok:
            return True, f"node {node_id} projects asset field {detail!r}"
        # A concrete non-asset projection is an explicit value choice.  Do not
        # walk past it and claim an earlier owner merely *could* have images.
        return False, detail or "terminal projects a non-asset field"
    if op == "relation" and catalog is not None:
        src = str(node.get("source") or "")
        src_resource = _node_resource(src, by_id) if src else ""
        cap = catalog.try_lookup(src_resource, str(node.get("relation") or ""), resource or None)
        if cap is not None and cap.target_resource in ASSET_RESOURCES:
            return True, f"relation resolves to {cap.semantic_name}"
    if op in {"select", "filter"}:
        src = str(node.get("source") or "")
        if src:
            return terminal_produces_asset(src, by_id, catalog, depth=depth + 1)
    return False, "no asset-typed value in this lineage"


def _answer_asset_fields(answer: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for raw in answer.get("fields") or []:
        canon = asset_field_canonical(raw)
        if canon and canon not in out:
            out.append(canon)
    return out


def _owner_field_error(fields: list[str], owner: str) -> str | None:
    if not owner:
        return None
    for field in fields:
        if field not in STRICT_ASSET_OWNER_FIELDS:
            continue
        try:
            resolve_field(field, owner)
        except FieldTypeError as exc:
            return str(exc)
    return None


def asset_contract_satisfied(terminal_ids: list[str], by_id: dict[str, dict],
                             answer: dict[str, Any],
                             catalog: CapabilityCatalog | None = None
                             ) -> tuple[bool, str, str | None]:
    """Validate an asset answer by value type and provider ownership.

    Returns ``(ok, reason, ownership_error)``.  A missing final image edge may be
    considered closeable only when the terminal is still an entity/selection
    value (not an explicit non-asset projection) and the provider catalog
    documents that owner's canonical ``images`` capability.
    """
    asset_fields = _answer_asset_fields(answer)
    reasons: list[str] = []
    for tid in terminal_ids:
        node = by_id.get(str(tid or "")) or {}
        owner = _asset_owner(tid, by_id, catalog)
        owner_error = _owner_field_error(asset_fields, owner)
        if owner_error:
            return False, "", owner_error
        ok, why = terminal_produces_asset(tid, by_id, catalog)
        if ok:
            return True, why, None
        reasons.append(f"{tid}: {why}")

        # A concrete non-asset projection cannot be repaired merely because the
        # upstream entity owns some image capability.  A projection whose field
        # is itself asset-typed may be promoted through that capability.
        if str(node.get("op") or "") == "project" and not asset_field_canonical(node.get("field")):
            continue
        owner = owner or _node_resource(tid, by_id)
        if catalog is not None and owner:
            cap = catalog.try_lookup(owner, "images", "image")
            if cap is not None:
                return True, f"typed asset contract is closeable through {cap.semantic_name}", None
    return False, "; ".join(reasons) or "no terminals", None
