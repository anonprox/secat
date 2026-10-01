"""Provider-aware semantic field ownership compatibility helpers.

The authoritative semantic field definitions live in ``oca.capability`` provider
modules.  This module remains as a small compatibility facade for older compiler
call sites; it no longer maintains a second TMDB-only ontology.
"""
from __future__ import annotations
import re
from typing import Any


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text or "").casefold()).split())


def _default_fields():
    # No provider is a safe implicit default.  Callers that know the runtime
    # provider must pass its semantic field registry explicitly; otherwise this
    # compatibility facade performs no ownership inference.  This prevents a
    # Spotify/other-provider compile path from silently inheriting TMDB ontology.
    return {}


def semantic_field(text: str, semantic_fields: dict[str, Any] | None = None):
    fields = semantic_fields if semantic_fields is not None else _default_fields()
    raw = _norm(text)
    if not raw:
        return None
    tokens = set(raw.split())
    best = None
    for sf in fields.values():
        forms = {sf.name, sf.name.replace("_", " "), *getattr(sf, "aliases", ())}
        for alias in forms:
            norm = _norm(alias)
            if not norm:
                continue
            atoks = set(norm.split())
            if tokens == atoks:
                return sf
            if len(atoks) >= 2 and atoks <= tokens:
                score = len(atoks)
                if best is None or score > best[0]:
                    best = (score, sf)
    return best[1] if best else None


def owner_error(field_text: str, resource: str, *, semantic_fields: dict[str, Any] | None = None) -> str | None:
    fields = semantic_fields if semantic_fields is not None else _default_fields()
    if not resource:
        return None
    r = _norm(resource).replace("television", "tv")
    raw = _norm(field_text).replace(" ", "_")
    # A provider can expose the same concrete property under a broader semantic
    # label (TMDB TV `name` is semantic `title`).  Treat the documented concrete
    # field as owned by that resource before applying alias-level rejection.
    for candidate in fields.values():
        concrete = str(getattr(candidate, "provider_field", {}).get(r, "") or "")
        if concrete and _norm(concrete).replace(" ", "_") == raw:
            return None
    sf = semantic_field(field_text, fields)
    if sf is None:
        return None
    owners = set(getattr(sf, "owners", ()) or ())
    if r in owners:
        return None
    # Visual terminology is intentionally surface-flexible: a request for a
    # poster/photo/cover on a resource may compile through that resource's
    # documented asset capability (e.g. episode stills, person profiles).
    # Keep the type guard strict by permitting this only when both sides are
    # asset-path semantic fields.
    if getattr(sf, "value_type", None) == "asset_path":
        if any(getattr(other, "value_type", None) == "asset_path" and
               r in set(getattr(other, "owners", ()) or ()) for other in fields.values()):
            return None
    return (f"requested semantic field {field_text!r} ({sf.name}) belongs to {sorted(owners)}, "
            f"not resource {resource!r}")
