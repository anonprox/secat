"""Provider capability graph.

A capability is a *typed, documented* statement that a provider exposes a given
semantic relation, together with everything needed to compile it: which endpoint
serves it, how the parent id is bound, where the records live in the response,
which filter isolates the requested role, and what the resulting entity is.

The point is to replace lexical endpoint scoring for relations. In v3.8.46,
``relation="director"`` on a movie produced a ten-way score tie and was resolved
alphabetically to ``/3/movie/{movie_id}`` -- the detail endpoint -- because
"director" appears nowhere in the ``/credits`` path text. A capability lookup
cannot tie, and cannot silently substitute a detail endpoint for a relation.

Capabilities encode *API* knowledge (where TMDB puts crew), never *benchmark*
knowledge (which task expects which answer).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .types import PLACEHOLDER_RESOURCE, RESOURCE_TYPES


AUTHORITATIVE = "authoritative"
DERIVED = "derived"


def _norm_semantic_phrase(value: str) -> str:
    """Normalize a short semantic label without provider/question knowledge.

    This is deliberately conservative: case/underscore/hyphen normalization plus
    singularization of the *last* token.  It exists so a model spelling
    ``season`` can resolve the catalog's canonical ``seasons`` relation without
    adding question-shaped regexes.
    """
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()
    return " ".join(text.split())


def _semantic_forms(value: str) -> frozenset[str]:
    raw = _norm_semantic_phrase(value)
    if not raw:
        return frozenset()
    forms = {raw}
    parts = raw.split()
    last = parts[-1]
    variants = {last}
    if last.endswith("ies") and len(last) > 4:
        variants.add(last[:-3] + "y")
    elif last.endswith("ses") and len(last) > 4:
        # e.g. ``images`` is handled by the generic trailing-s rule below;
        # keep this branch intentionally narrow to avoid aggressive stemming.
        variants.add(last[:-1])
    elif last.endswith("s") and not last.endswith("ss") and len(last) > 3:
        variants.add(last[:-1])
    else:
        # Add a plural spelling only as a lookup form; canonical catalog labels
        # remain unchanged.
        if last.endswith("y") and len(last) > 2 and last[-2] not in "aeiou":
            variants.add(last[:-1] + "ies")
        else:
            variants.add(last + "s")
    for variant in variants:
        forms.add(" ".join(parts[:-1] + [variant]))
    return frozenset(forms)


@dataclass(frozen=True)
class Capability:
    """One typed relation edge in the provider graph."""

    semantic_name: str          # e.g. "movie.director"
    source_resource: str        # "movie"
    relation: str               # "director"
    target_resource: str        # "person"
    endpoint: str               # "/3/movie/{movie_id}/credits"
    record_path: str            # "crew[*]"
    method: str = "GET"
    parent_binding: dict[str, str] = field(default_factory=dict)
    # Provider-required request literals that are intrinsic to the semantic
    # capability (for example Spotify /me/following?type=artist). They are API
    # contract facts, not task-specific values.
    path_literals: dict[str, Any] = field(default_factory=dict)
    query_literals: dict[str, Any] = field(default_factory=dict)
    # Dynamic request bindings sourced from the selected semantic owner. Values
    # use provider-neutral specs such as ``source.name`` or ``source.value``.
    query_bindings: dict[str, str] = field(default_factory=dict)
    record_filter: dict[str, Any] = field(default_factory=dict)
    selection: str = "all"      # all | first | endpoint_order_first
    id_field: str = "id"
    label_field: str = "name"
    #: True when the relation is already present in a response the plan will
    #: fetch anyway (e.g. tv detail carries ``networks[*]``). Compiling such a
    #: relation must not issue a second identical request.
    embedded_in_source: bool = False
    confidence: str = AUTHORITATIVE
    aliases: tuple[str, ...] = ()
    notes: str = ""
    #: Set when the bundled provider specification does not document this
    #: endpoint's response body, and the record path comes from the provider's
    #: public documentation instead. Reported separately by validation so a
    #: reviewer can see exactly where the specification was incomplete.
    schema_supplement: bool = False

    @property
    def placeholders(self) -> list[str]:
        return re.findall(r"\{([^{}]+)\}", self.endpoint)

    def alias_set(self) -> frozenset[str]:
        base = {self.relation.casefold()}
        base.update(a.casefold() for a in self.aliases)
        base.add(self.relation.replace("_", " ").casefold())
        return frozenset(base)


class CapabilityError(LookupError):
    """No documented capability serves the requested relation."""


class CapabilityCatalog:
    """Indexed, validated set of capabilities for one provider."""

    def __init__(self, provider: str, capabilities: Sequence[Capability]):
        self.provider = provider
        self._all = list(capabilities)
        self._by_source: dict[str, list[Capability]] = {}
        for cap in self._all:
            self._by_source.setdefault(cap.source_resource, []).append(cap)
        self._check_internal()

    # -- construction checks ------------------------------------------------

    def _check_internal(self) -> None:
        seen: set[str] = set()
        for cap in self._all:
            if cap.semantic_name in seen:
                raise ValueError(f"duplicate capability {cap.semantic_name}")
            seen.add(cap.semantic_name)
            if cap.source_resource not in RESOURCE_TYPES:
                raise ValueError(
                    f"{cap.semantic_name}: unknown source resource "
                    f"{cap.source_resource!r}")
            if cap.target_resource not in RESOURCE_TYPES:
                raise ValueError(
                    f"{cap.semantic_name}: unknown target resource "
                    f"{cap.target_resource!r}")
        # A source/relation pair must not resolve ambiguously.
        index: dict[tuple[str, str], list[str]] = {}
        for cap in self._all:
            for alias in cap.alias_set():
                index.setdefault((cap.source_resource, alias), []).append(
                    cap.semantic_name)
        for (src, alias), names in index.items():
            if len(names) > 1:
                raise ValueError(
                    f"ambiguous relation {alias!r} on {src!r}: {sorted(names)}")

    def schema_supplements(self) -> list[Capability]:
        """Capabilities whose response shape is not in the bundled spec."""
        return [c for c in self._all if c.schema_supplement]

    def validate_against_oas(self, cards: Iterable[dict[str, Any]]) -> list[str]:
        """Prove every capability names a real endpoint with real fields.

        Returns a list of problems; empty means the catalog is consistent with
        the provider specification. Run this at import time and in CI so a
        catalog edit cannot silently reference an endpoint that does not exist.

        Capabilities marked ``schema_supplement`` skip response-field checks
        (the specification documents no response body for them) but still have
        their endpoint and path parameters verified.
        """
        by_endpoint: dict[tuple[str, str], dict[str, Any]] = {}
        for card in cards:
            key = (str(card.get("method") or "GET").upper(),
                   str(card.get("endpoint") or card.get("path") or ""))
            by_endpoint[key] = card

        problems: list[str] = []
        for cap in self._all:
            card = by_endpoint.get((cap.method.upper(), cap.endpoint))
            if card is None:
                problems.append(
                    f"{cap.semantic_name}: endpoint not in provider spec: "
                    f"{cap.method} {cap.endpoint}")
                continue

            leaves = {str(lp.get("path") or "") if isinstance(lp, dict) else str(lp)
                      for lp in (card.get("leaf_paths") or [])}
            if cap.schema_supplement:
                if leaves:
                    problems.append(
                        f"{cap.semantic_name}: marked schema_supplement but "
                        f"{cap.endpoint} does document a response body")
            root = cap.record_path
            if root not in {"", "$"} and not cap.schema_supplement:
                prefix = root if root.endswith("[*]") else root
                if not (prefix in leaves or any(leaf.startswith(prefix + ".") for leaf in leaves)):
                    problems.append(
                        f"{cap.semantic_name}: record_path {root!r} has no "
                        f"documented fields on {cap.endpoint}")
                else:
                    for fname, label in ((cap.id_field, "id_field"),
                                         (cap.label_field, "label_field")):
                        if fname and f"{prefix}.{fname}" not in leaves:
                            problems.append(
                                f"{cap.semantic_name}: {label} {fname!r} absent "
                                f"from {root} on {cap.endpoint}")
                    for fkey in cap.record_filter:
                        if f"{prefix}.{fkey}" not in leaves:
                            problems.append(
                                f"{cap.semantic_name}: filter field {fkey!r} "
                                f"absent from {root} on {cap.endpoint}")

            declared = {str(p.get("name") or "")
                        for p in (card.get("parameters") or [])
                        if str(p.get("in") or "") == "path"}
            for ph in cap.placeholders:
                if ph not in declared:
                    problems.append(
                        f"{cap.semantic_name}: path placeholder {ph!r} not "
                        f"declared by {cap.endpoint}")
                expected = PLACEHOLDER_RESOURCE.get(ph)
                bound = cap.parent_binding.get(ph)
                if bound == "source.id" and expected not in {
                        cap.source_resource, None}:
                    problems.append(
                        f"{cap.semantic_name}: {ph!r} identifies {expected!r} "
                        f"but is bound from source {cap.source_resource!r}")
            for ph in cap.parent_binding:
                if ph not in cap.placeholders:
                    problems.append(
                        f"{cap.semantic_name}: binding for {ph!r} which is not "
                        f"a placeholder of {cap.endpoint}")
            query_declared = {str(p.get("name") or "")
                              for p in (card.get("parameters") or [])
                              if str(p.get("in") or "query") == "query"}
            for name in cap.query_literals:
                if name not in query_declared:
                    problems.append(
                        f"{cap.semantic_name}: query literal {name!r} not "
                        f"declared by {cap.endpoint}")
            for name in cap.query_bindings:
                if name not in query_declared:
                    problems.append(
                        f"{cap.semantic_name}: query binding {name!r} not "
                        f"declared by {cap.endpoint}")
            for name in cap.path_literals:
                if name not in cap.placeholders:
                    problems.append(
                        f"{cap.semantic_name}: path literal {name!r} is not a "
                        f"placeholder of {cap.endpoint}")
        return problems

    # -- lookup -------------------------------------------------------------

    def lookup(self, source_resource: str, relation: str,
               target_resource: str | None = None) -> Capability:
        """Return the single capability serving ``relation`` on the source type.

        Raises :class:`CapabilityError` when the provider documents no such
        relation. Callers must treat that as a compile failure, *not* as
        permission to fall back to a lexically similar endpoint.
        """
        want = _norm_semantic_phrase(relation)
        if not want:
            raise CapabilityError("empty relation")
        candidates = [
            cap for cap in self._by_source.get(source_resource, [])
            if want in {_norm_semantic_phrase(x) for x in cap.alias_set()}
        ]
        # If exact semantic wording is absent, allow only conservative
        # Accept a model spelling the provider-qualified semantic capability
        # name itself (for example ``tv.images``) as a typed ontology label.
        # This is catalog knowledge, not question-language dispatch.
        if not candidates:
            qualified = _norm_semantic_phrase(want).replace(" ", ".")
            candidates = [
                cap for cap in self._by_source.get(source_resource, [])
                if _norm_semantic_phrase(cap.semantic_name).replace(" ", ".") == qualified
            ]

        # singular/plural normalization against catalog aliases.  This is
        # provider-ontology canonicalization, never question dispatch.
        if not candidates:
            want_forms = _semantic_forms(want)
            candidates = [
                cap for cap in self._by_source.get(source_resource, [])
                if want_forms & set().union(*(_semantic_forms(x) for x in cap.alias_set()))
            ]
        if target_resource:
            # Target type is part of the semantic contract, not a soft hint.
            # Keeping an exact-word candidate of the wrong type silently turns
            # e.g. person->TV into person->movie when relation wording is stale.
            candidates = [c for c in candidates if c.target_resource == target_resource]
        if not candidates:
            raise CapabilityError(
                f"no documented capability for relation {relation!r} on "
                f"{source_resource!r}")
        if len(candidates) > 1:  # pragma: no cover - blocked by _check_internal
            raise CapabilityError(
                f"ambiguous relation {relation!r} on {source_resource!r}: "
                f"{[c.semantic_name for c in candidates]}")
        return candidates[0]

    def canonical_relation(self, source_resource: str, relation: str,
                           target_resource: str | None = None) -> str | None:
        """Return the catalog's canonical relation label for a model phrase."""
        cap = self.try_lookup(source_resource, relation, target_resource)
        return cap.relation if cap is not None else None

    def try_lookup(self, source_resource: str, relation: str,
                   target_resource: str | None = None) -> Capability | None:
        try:
            return self.lookup(source_resource, relation, target_resource)
        except CapabilityError:
            return None

    def relations_for(self, source_resource: str) -> list[Capability]:
        return list(self._by_source.get(source_resource, []))

    def __len__(self) -> int:
        return len(self._all)

    def __iter__(self):
        return iter(self._all)
