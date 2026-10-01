"""Normalize model-produced semantic intent against a typed provider capability catalog.

This pure pass sits between language understanding and low-level compilation. It
anchors relations to provider semantics, inserts required detail hops, preserves
explicit scope, and reports unresolved capabilities without inventing answers.
It is intentionally phrasing-independent: it consumes typed IR rather than the
original natural-language question.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from .catalog import Capability, CapabilityCatalog
from .route_control import sentinel_for
from .asset_contract import (
    asset_contract_satisfied, asset_field_canonical, terminal_produces_asset,
)
from .types import (
    FieldTypeError, PLACEHOLDER_RESOURCE, WORK_TYPES, canonical_field,
    resolve_field,
)


@dataclass
class NormalizationReport:
    """What normalisation did, for the run artifact and the audit."""

    anchored: dict[str, str] = field(default_factory=dict)      # node id -> capability
    inserted_filters: list[str] = field(default_factory=list)
    inserted_selects: list[str] = field(default_factory=list)
    inserted_detail_hops: list[str] = field(default_factory=list)
    unresolved_relations: list[str] = field(default_factory=list)
    type_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "anchored": dict(self.anchored),
            "inserted_filters": list(self.inserted_filters),
            "inserted_selects": list(self.inserted_selects),
            "inserted_detail_hops": list(self.inserted_detail_hops),
            "unresolved_relations": list(self.unresolved_relations),
            "type_errors": list(self.type_errors),
            "warnings": list(self.warnings),
        }

    @property
    def clean(self) -> bool:
        return not self.type_errors and not self.unresolved_relations


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _endpoint_relation_token(cap: Capability) -> str:
    """The word the downstream route scorer needs in order to land on the
    capability's endpoint deterministically.

    For a dedicated child route this is the trailing path segment
    (``/movie/{movie_id}/credits`` -> ``credits``). For a relation embedded in a
    detail response it is the record root (``networks[*]`` -> ``networks``),
    which keeps the compiler on the detail route it already has.
    """
    if cap.embedded_in_source:
        root = cap.record_path.split("[")[0].split(".")[0]
        return root if root not in {"", "$"} else "detail"
    tail = cap.endpoint.rstrip("/").split("/")[-1]
    if tail.startswith("{"):
        return "detail"
    return tail.replace("_", " ")


def _record_root_name(cap: Capability) -> str:
    root = cap.record_path.split("[")[0].split(".")[0]
    return "" if root in {"", "$"} else root


def _season_scoped(node: dict[str, Any], nodes_by_id: dict[str, dict[str, Any]]) -> bool:
    """True when this relation is scoped to a season rather than the series.

    Scope is semantic IR, not wording.  An explicit ``season_number`` literal
    is therefore sufficient even when the relation itself is simply
    ``director`` or ``credits``.
    """
    if node.get("_oca_season_scope") is not None:
        return True
    literals = node.get("literals") or {}
    if literals.get("season_number") is not None or literals.get("season") is not None:
        return True
    text = " ".join(str(node.get(k) or "") for k in ("relation", "collection", "name"))
    if re.search(r"\bseason\b", text, re.I):
        return True
    src = nodes_by_id.get(str(node.get("source") or ""))
    if src is not None:
        if str(src.get("resource") or "") == "season":
            return True
        stext = " ".join(str(src.get(k) or "") for k in ("relation", "name", "collection"))
        if re.search(r"\bseason\b", stext, re.I):
            return True
    return False


def _episode_scoped(node: dict[str, Any], nodes_by_id: dict[str, dict[str, Any]]) -> bool:
    """True when typed IR explicitly scopes a relation to an episode."""
    literals = node.get("literals") or {}
    # A literal episode number on a relation that *produces* episodes is a
    # selection qualifier for that child collection, not proof that the source
    # handle is already an episode.  This distinction matters for compact IR
    # such as tv -> seasons(season=1) -> episodes(episode=2): after season scope
    # compression, the episodes edge must still resolve as season.episodes.
    target_resource = str(node.get("resource") or "")
    if target_resource != "episode" and (
            literals.get("episode_number") is not None or literals.get("episode") is not None):
        return True
    # ``node.resource`` is the *target* type of a relation.  Source lineage, not
    # the target label, establishes an already-selected episode handle.
    src = nodes_by_id.get(str(node.get("source") or ""))
    return bool(src and str(src.get("resource") or "") == "episode")


def _episode_number(node: dict[str, Any],
                    nodes_by_id: dict[str, dict[str, Any]]) -> int | None:
    """Find an explicit episode number on the node or its source chain."""
    current: dict[str, Any] | None = node
    depth = 0
    while current is not None and depth < 8:
        lits = current.get("literals") or {}
        for key in ("episode_number", "episode"):
            if key in lits:
                try:
                    return int(lits[key])
                except (TypeError, ValueError):
                    pass
        current = nodes_by_id.get(str(current.get("source") or ""))
        depth += 1
    return None


def _season_number(node: dict[str, Any],
                   nodes_by_id: dict[str, dict[str, Any]]) -> int | None:
    """Find an explicit season number on the node or its source chain."""
    _WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
              "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
    current: dict[str, Any] | None = node
    depth = 0
    while current is not None and depth < 8:
        scoped = current.get("_oca_season_scope")
        if scoped is not None:
            try:
                return int(scoped)
            except (TypeError, ValueError):
                pass
        lits = current.get("literals") or {}
        for key in ("season_number", "season"):
            if key in lits:
                try:
                    return int(lits[key])
                except (TypeError, ValueError):
                    pass
        text = " ".join(str(current.get(k) or "")
                        for k in ("name", "relation", "collection"))
        m = re.search(r"season\s+(\d+)", text, re.I)
        if m:
            return int(m.group(1))
        m = re.search(r"(" + "|".join(_WORDS) + r")\s+season", text, re.I)
        if m:
            return _WORDS[m.group(1).lower()]
        current = nodes_by_id.get(str(current.get("source") or ""))
        depth += 1
    return None


def _fresh_id(prefix: str, used: set[str]) -> str:
    n = 1
    while f"{prefix}{n}" in used:
        n += 1
    ident = f"{prefix}{n}"
    used.add(ident)
    return ident


def _card_index(cards: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for card in cards:
        out[str(card.get("endpoint") or card.get("path") or "")] = card
    return out


def _leaf_set(card: dict[str, Any]) -> set[str]:
    return {str(lp.get("path") or "") if isinstance(lp, dict) else str(lp)
            for lp in (card.get("leaf_paths") or [])}


def _unique_relation_bridge(catalog: CapabilityCatalog, source_resource: str,
                            relation: str, target_resource: str | None) -> list[Capability]:
    """Return one unique short provider-ontology path to a requested relation.

    Models sometimes compress a hierarchy edge (``tv -> episodes``) although the
    provider ontology requires ``tv -> seasons -> episodes``.  The final relation
    phrase and target type are still explicit in typed IR, so the host can expand
    a *unique* provider path without reading question wording.  Ambiguous paths
    are never guessed.
    """
    src = str(source_resource or "")
    rel = str(relation or "")
    target = str(target_resource or "") or None
    if not src or not rel or not target:
        return []
    # Bridge expansion is only for compressed hierarchy navigation where the
    # relation itself names the requested child resource (episode(s), season(s),
    # etc.). It must never reinterpret a semantic role such as credits/director
    # merely because some longer graph path happens to end at the same type.
    def forms(value: str) -> set[str]:
        text = re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()
        if not text:
            return set()
        parts = text.split(); last = parts[-1]
        out = {text}
        if last.endswith("ies") and len(last) > 4:
            out.add(" ".join(parts[:-1] + [last[:-3] + "y"]))
        elif last.endswith("s") and not last.endswith("ss") and len(last) > 3:
            out.add(" ".join(parts[:-1] + [last[:-1]]))
        else:
            out.add(" ".join(parts[:-1] + [last + "s"]))
        return out
    if not (forms(rel) & forms(target)):
        return []
    candidates: list[list[Capability]] = []
    # At most two bridge edges before the requested terminal relation. This is
    # enough for ordinary provider hierarchies while avoiding broad graph search.
    for first in catalog.relations_for(src):
        if first.target_resource in {"image", "review"}:
            continue
        last = catalog.try_lookup(first.target_resource, rel, target)
        if last is not None:
            candidates.append([first, last])
        for second in catalog.relations_for(first.target_resource):
            if second.target_resource in {"image", "review"}:
                continue
            last = catalog.try_lookup(second.target_resource, rel, target)
            if last is not None:
                candidates.append([first, second, last])
    # Deduplicate semantic-name sequences; accept only a unique shortest path.
    uniq: dict[tuple[str, ...], list[Capability]] = {}
    for path in candidates:
        uniq[tuple(c.semantic_name for c in path)] = path
    if not uniq:
        return []
    shortest = min(len(x) for x in uniq)
    paths = [p for p in uniq.values() if len(p) == shortest]
    return paths[0] if len(paths) == 1 else []


# --------------------------------------------------------------------------
# resource inference
# --------------------------------------------------------------------------

def infer_resource(node_id: str, nodes_by_id: dict[str, dict[str, Any]],
                   catalog: CapabilityCatalog, *, _depth: int = 0) -> str:
    """Best-effort resource type of the records a node produces."""
    if _depth > 12:
        return ""
    node = nodes_by_id.get(node_id)
    if node is None:
        return ""
    explicit = str(node.get("resource") or "").strip()
    op = str(node.get("op") or "")
    if explicit and op in {"find", "population", "relation"}:
        return explicit
    if op in {"find", "population"}:
        return explicit
    if op == "relation":
        src = infer_resource(str(node.get("source") or ""), nodes_by_id,
                             catalog, _depth=_depth + 1)
        cap = catalog.try_lookup(src, str(node.get("relation") or ""), explicit or None)
        if cap is not None:
            return cap.target_resource
        return explicit or src
    for key in ("source", "left"):
        ref = node.get(key)
        if ref:
            got = infer_resource(str(ref), nodes_by_id, catalog, _depth=_depth + 1)
            if got:
                return got
    return explicit


# --------------------------------------------------------------------------
# main pass
# --------------------------------------------------------------------------

def normalize_intent(intent: dict[str, Any], catalog: CapabilityCatalog,
                     cards: list[dict[str, Any]] | None = None
                     ) -> tuple[dict[str, Any], NormalizationReport]:
    """Anchor every relation node to a documented capability.

    Returns a new intent dict plus a report. The input is never mutated.
    """
    report = NormalizationReport()
    obj = copy.deepcopy(intent or {})
    nodes: list[dict[str, Any]] = list(obj.get("nodes") or [])
    if not nodes:
        return obj, report

    cards = cards or []
    card_by_ep = _card_index(cards)
    used_ids = {str(n.get("id") or "") for n in nodes}
    nodes_by_id = {str(n.get("id") or ""): n for n in nodes}

    # Consumers of each node, so an inserted filter/select can be spliced in.
    def rewire(old_id: str, new_id: str, skip: set[int]) -> None:
        for idx, other in enumerate(nodes):
            if idx in skip:
                continue
            for key in ("source", "left", "right"):
                if str(other.get(key) or "") == old_id:
                    other[key] = new_id
        answer = obj.get("answer")
        if isinstance(answer, dict):
            if str(answer.get("source") or "") == old_id:
                answer["source"] = new_id
            if isinstance(answer.get("sources"), list):
                answer["sources"] = [
                    new_id if str(x or "") == old_id else x
                    for x in answer.get("sources") or []
                ]

    # ---- pre-pass: repair typed graph compression and scalar-count shape --
    # These rewrites consume only typed IR. They do not inspect the user question.

    # A model can compress "select one X, then follow relation R" into a select
    # node that carries relation/resource metadata. Selection itself preserves the
    # source resource; materialize the relation as an explicit child node so the
    # compiler cannot return the selected owner when the requested answer is the
    # related target (e.g. TV -> lead actor).
    index = 0
    while index < len(nodes):
        node = nodes[index]
        if str(node.get("op") or "") == "select" and str(node.get("relation") or "").strip():
            nid = str(node.get("id") or "")
            source_id = str(node.get("source") or "")
            source_resource = infer_resource(source_id, nodes_by_id, catalog)
            target_resource = str(node.get("resource") or "").strip().casefold()
            relation = str(node.get("relation") or "").strip()
            if source_resource and target_resource and target_resource != source_resource:
                child_id = _fresh_id("cap_after_select", used_ids)
                # Rewire existing consumers/answer first; the newly inserted child
                # deliberately continues to consume the original selection.
                rewire(nid, child_id, skip={index})
                node.pop("relation", None)
                node["resource"] = source_resource
                child = {
                    "id": child_id, "op": "relation", "source": nid,
                    "relation": relation, "resource": target_resource,
                    "literals": {}, "_oca_contract_closure": "post_selection_relation",
                }
                nodes.insert(index + 1, child)
                nodes_by_id[child_id] = child
                report.warnings.append(
                    f"{child_id}: expanded relation {relation!r} after typed selection {nid}")
                index += 1
        index += 1

    # Canonicalize entity-in-collection membership. Models often emit
    # membership(left=<entity>, right=<relation collection>) while the compiler
    # expects the collection as ``source`` plus one scalar target derivation.
    # Project provider identity from the entity so replay compares ids rather
    # than names or opaque records.
    index = 0
    while index < len(nodes):
        node = nodes[index]
        if str(node.get("op") or "") != "membership":
            index += 1
            continue
        left_id = str(node.get("left") or "")
        right_id = str(node.get("right") or "")
        right_node = nodes_by_id.get(right_id) or {}
        if left_id and right_id and str(right_node.get("op") or "") in {"relation", "filter", "select"}:
            left_node = nodes_by_id.get(left_id) or {}
            target_id = left_id
            if str(left_node.get("op") or "") != "project":
                pid = _fresh_id("cap_member_id", used_ids)
                project = {"id": pid, "op": "project", "source": left_id,
                           "field": "id", "_oca_semantic_field": "id",
                           "_oca_contract_closure": "membership_identity"}
                nodes.insert(index, project)
                nodes_by_id[pid] = project
                target_id = pid
                index += 1
            node["source"] = right_id
            node["left"] = target_id
            node["field"] = "id"
            report.warnings.append(
                f"{node.get('id')}: canonicalized entity membership to identity-in-collection")
        index += 1

    # Canonicalize ranking that a model encoded as compare(mode=argmax/argmin).
    # Ranking a collection is selection, not binary comparison.  Recover the
    # collection owner and scalar field from typed references only.
    for node in nodes:
        if str(node.get("op") or "") != "compare":
            continue
        rank_mode = str(node.get("mode") or "").casefold()
        if rank_mode not in {"argmax", "argmin"}:
            continue
        collection_source = str(node.get("source") or "")
        rank_field = str(node.get("field") or (node.get("literals") or {}).get("field") or "")
        refs = [str(node.get("left") or ""), str(node.get("right") or "")]
        for ref in refs:
            ref_node = nodes_by_id.get(ref) or {}
            if str(ref_node.get("op") or "") == "project":
                collection_source = collection_source or str(ref_node.get("source") or "")
                rank_field = rank_field or str(ref_node.get("_oca_semantic_field") or ref_node.get("field") or "")
            elif not collection_source and ref:
                collection_source = ref
        if collection_source and rank_field:
            node["op"] = "select"
            node["source"] = collection_source
            node["mode"] = rank_mode
            node["field"] = rank_field
            node.pop("left", None); node.pop("right", None); node.pop("comparison", None)
            report.warnings.append(
                f"{node.get('id')}: canonicalized ranking comparison to select({rank_mode}) on {rank_field}")

    # Counting an already-scalar count property is a type error (count([137]) ->
    # 1). Likewise, when the provider exposes a stored scalar total for a child
    # collection (TV.seasons -> number_of_seasons), prefer that scalar rather than
    # counting a collection that may include provider-specific special records.
    for node in nodes:
        if str(node.get("op") or "") != "count":
            continue
        src_id = str(node.get("source") or "")
        src = nodes_by_id.get(src_id) or {}
        if str(src.get("op") or "") == "project":
            canon = canonical_field(str(src.get("_oca_semantic_field") or src.get("field") or ""))
            if canon and canon.endswith("_count"):
                node["op"] = "project"
                node["source"] = str(src.get("source") or "")
                node["field"] = str(src.get("_oca_semantic_field") or src.get("field") or canon)
                node["_oca_semantic_field"] = canon
                report.warnings.append(
                    f"{node.get('id')}: replaced count-of-scalar with direct {canon} projection")
                continue
        if str(src.get("op") or "") == "relation":
            parent_id = str(src.get("source") or "")
            parent_resource = infer_resource(parent_id, nodes_by_id, catalog)
            target_resource = str(src.get("resource") or "").strip().casefold()
            candidate = f"{target_resource}_count" if target_resource else ""
            if candidate and parent_resource:
                try:
                    resolve_field(candidate, parent_resource)
                except FieldTypeError:
                    pass
                else:
                    node["op"] = "project"
                    node["source"] = parent_id
                    node["field"] = candidate
                    node["_oca_semantic_field"] = candidate
                    report.warnings.append(
                        f"{node.get('id')}: used provider scalar {candidate} instead of counting {target_resource} records")

    # ---- pass 0: close typed asset terminals ------------------------------
    # ``answer.mode=asset`` is a typed semantic obligation.  Asset recognition
    # is resolved from field/resource types and provider capabilities; no visual
    # word list or terminal-op whitelist is involved.
    answer0 = obj.get("answer") if isinstance(obj.get("answer"), dict) else {}
    if str(answer0.get("mode") or "").casefold() == "asset":
        terminal_ids = [str(x) for x in (answer0.get("sources") or []) if str(x)]
        if not terminal_ids and answer0.get("source"):
            terminal_ids = [str(answer0.get("source"))]
        for terminal_id in list(terminal_ids):
            terminal = nodes_by_id.get(terminal_id) or {}

            # Already asset-typed, including select/filter over an image lineage.
            already_asset, _asset_reason = terminal_produces_asset(
                terminal_id, nodes_by_id, catalog)
            if already_asset:
                continue

            # An explicit non-asset projection is a real semantic mismatch; do
            # not mask it by appending images to some upstream entity.
            if str(terminal.get("op") or "") == "project":
                canon = asset_field_canonical(terminal.get("field"))
                if canon is None:
                    continue

            # Entity-only terminals with no asset-typed requested field are not
            # eligible for automatic asset closure.  A mistaken language-layer
            # mode=asset must be repaired there instead of silently turning a
            # title/entity answer into a poster.
            explicit_asset_field = any(asset_field_canonical(x) for x in (answer0.get("fields") or []))
            if not explicit_asset_field:
                continue

            closeable, _why, owner_error = asset_contract_satisfied(
                [terminal_id], nodes_by_id, answer0, catalog)
            if owner_error:
                report.type_errors.append(
                    f"{terminal_id}: asset field owner mismatch: {owner_error}")
                continue
            if not closeable:
                continue

            owner_resource = infer_resource(terminal_id, nodes_by_id, catalog)
            if not owner_resource:
                continue
            asset_fields = [asset_field_canonical(x) for x in (answer0.get("fields") or [])]
            relation_name = "backdrops" if "backdrop" in asset_fields else "images"
            cap = catalog.try_lookup(owner_resource, relation_name, "image")
            if cap is None and relation_name != "images":
                cap = catalog.try_lookup(owner_resource, "images", "image")
            if cap is None:
                continue
            new_id = _fresh_id("cap_asset", used_ids)
            new_node = {"id": new_id, "op": "relation", "source": terminal_id,
                        "relation": cap.relation, "resource": cap.target_resource,
                        "_oca_contract_closure": "asset"}
            nodes.append(new_node)
            nodes_by_id[new_id] = new_node
            if str(answer0.get("source") or "") == terminal_id:
                answer0["source"] = new_id
            if isinstance(answer0.get("sources"), list):
                answer0["sources"] = [new_id if str(x or "") == terminal_id else x
                                      for x in answer0.get("sources") or []]
            report.warnings.append(
                f"{new_id}: appended documented {owner_resource}.{cap.relation} relation to satisfy typed asset contract")

    # ---- pass 1: anchor relations -----------------------------------------
    index = 0
    while index < len(nodes):
        node = nodes[index]
        if str(node.get("op") or "") != "relation":
            index += 1
            continue

        nid = str(node.get("id") or "")
        relation = str(node.get("relation") or "").strip()
        source_resource = infer_resource(str(node.get("source") or ""),
                                         nodes_by_id, catalog)
        # Explicit scope in typed IR selects the corresponding provider resource.
        # Episode scope is stronger than season scope. This is provider semantics,
        # not question wording: no natural-language string is inspected here.
        lookup_resource = source_resource
        if source_resource in {"tv", "season"} and _episode_scoped(node, nodes_by_id):
            lookup_resource = "episode"
        elif source_resource == "tv" and _season_scoped(node, nodes_by_id):
            lookup_resource = "season"

        # A season qualifier is scope on the series handle, not a resource to
        # fetch. Bare season navigation is absorbed into the scoped child relation.
        if (relation.strip().casefold() in {"season", "seasons"}
                and source_resource == "tv"
                and not _consumed_as_collection(nid, nodes)):
            season = _season_number(node, nodes_by_id)
            for other in nodes:
                if str(other.get("source") or "") == nid:
                    other["_oca_season_scope"] = season
            node["op"] = "_scope"
            rewire(nid, str(node.get("source") or ""), skip=set())
            index += 1
            continue

        target_hint = str(node.get("resource") or "") or None
        cap = catalog.try_lookup(lookup_resource, relation, target_hint)
        if cap is None and lookup_resource != source_resource:
            cap = catalog.try_lookup(source_resource, relation, target_hint)

        # If the model's relation noun is stale but its typed target resource is
        # explicit, use that type to disambiguate among provider capabilities.
        # This repairs shapes such as a relation phrase mentioning "movies" while
        # the typed target is TV, without reading the natural-language question.
        if cap is None and source_resource and target_hint and relation:
            wanted = set(re.findall(r"[a-z0-9]+", relation.casefold()))
            scored_caps = []
            for candidate in catalog.relations_for(source_resource):
                if str(candidate.target_resource or "").casefold() != str(target_hint).casefold():
                    continue
                text = " ".join([candidate.relation, *candidate.aliases]).casefold()
                tokens = set(re.findall(r"[a-z0-9]+", text))
                score = len(wanted & tokens)
                if score:
                    scored_caps.append((score, candidate.semantic_name, candidate))
            scored_caps.sort(key=lambda x: (-x[0], x[1]))
            if scored_caps and (len(scored_caps) == 1 or scored_caps[0][0] > scored_caps[1][0]):
                cap = scored_caps[0][2]
                report.warnings.append(
                    f"{nid}: resolved relation {relation!r} by typed target {target_hint!r} to {cap.semantic_name}")

        # Expand a compressed typed hierarchy only when the provider graph has a
        # unique short path whose *final* edge is exactly the requested relation.
        # Example shape: tv --seasons--> season --episodes--> episode.
        if cap is None and lookup_resource == source_resource:
            bridge = _unique_relation_bridge(catalog, source_resource, relation, target_hint)
            if bridge:
                parent_id = str(node.get("source") or "")
                # All but the final capability become explicit relation nodes.
                for bridge_cap in bridge[:-1]:
                    bridge_id = _fresh_id("cap_bridge", used_ids)
                    bridge_node = {
                        "id": bridge_id, "op": "relation", "source": parent_id,
                        "relation": sentinel_for(bridge_cap.semantic_name),
                        "resource": bridge_cap.target_resource,
                        "_oca_relation_word": _endpoint_relation_token(bridge_cap),
                        "_oca_capability": bridge_cap.semantic_name,
                        "_oca_bridge": True,
                    }
                    root_name = _record_root_name(bridge_cap)
                    if root_name:
                        bridge_node["collection"] = root_name
                    nodes.insert(index, bridge_node)
                    nodes_by_id[bridge_id] = bridge_node
                    report.anchored[bridge_id] = bridge_cap.semantic_name
                    parent_id = bridge_id
                    index += 1
                node["source"] = parent_id
                source_resource = bridge[-2].target_resource
                lookup_resource = source_resource
                cap = bridge[-1]
                report.warnings.append(
                    f"{nid}: expanded unique provider hierarchy through "
                    + " -> ".join(c.semantic_name for c in bridge))

        if cap is None:
            if relation:
                report.unresolved_relations.append(
                    f"{nid}: {source_resource or '?'}.{relation}")
            index += 1
            continue

        report.anchored[nid] = cap.semantic_name
        node["resource"] = cap.target_resource
        # A sentinel takes the relation out of lexical scoring entirely; see
        # route_control.install_route_control.
        node["relation"] = sentinel_for(cap.semantic_name)
        node["_oca_relation_word"] = _endpoint_relation_token(cap)
        root_name = _record_root_name(cap)
        if root_name:
            node["collection"] = root_name
        node["_oca_capability"] = cap.semantic_name
        # Scoped capabilities need their explicit numbers as path literals.
        if "season_number" in cap.placeholders:
            season = _season_number(node, nodes_by_id)
            if season is not None:
                lits = dict(node.get("literals") or {})
                lits.setdefault("season_number", season)
                node["literals"] = lits
            else:
                # A season-producing source can bind its own season_number at
                # execution time; lack of a literal is not an ambiguity/error.
                src_resource = infer_resource(str(node.get("source") or ""), nodes_by_id, catalog)
                if src_resource not in {"season", "episode"}:
                    report.warnings.append(
                        f"{nid}: {cap.semantic_name} needs a season number and none "
                        "was stated or available from typed source lineage")
        if "episode_number" in cap.placeholders:
            episode = _episode_number(node, nodes_by_id)
            if episode is not None:
                lits = dict(node.get("literals") or {})
                lits.setdefault("episode_number", episode)
                node["literals"] = lits
            else:
                src_resource = infer_resource(str(node.get("source") or ""), nodes_by_id, catalog)
                if src_resource != "episode":
                    report.warnings.append(
                        f"{nid}: {cap.semantic_name} needs an episode number and none "
                        "was stated or available from typed source lineage")

        tail_id = nid
        insert_at = index + 1

        # A capability that isolates a role does so with an explicit filter.
        # "Director" is a value of crew[*].job, never a route.
        if cap.record_filter:
            for fkey, fval in cap.record_filter.items():
                filt_id = _fresh_id("cap_f", used_ids)
                filt = {"id": filt_id, "op": "filter", "source": tail_id,
                        "field": fkey, "comparison": "eq", "value": fval,
                        "resource": cap.target_resource,
                        "_oca_capability": cap.semantic_name}
                rewire(tail_id, filt_id, skip={index})
                nodes.insert(insert_at, filt)
                nodes_by_id[filt_id] = filt
                report.inserted_filters.append(f"{filt_id}: {fkey}=={fval}")
                tail_id = filt_id
                insert_at += 1

        # Endpoint-ordered "lead" semantics: take the first record, do not
        # re-rank by popularity or any other field.
        if cap.selection == "endpoint_order_first":
            sel_id = _fresh_id("cap_s", used_ids)
            sel = {"id": sel_id, "op": "select", "source": tail_id,
                   "mode": "first", "resource": cap.target_resource,
                   "_oca_capability": cap.semantic_name}
            rewire(tail_id, sel_id, skip={index})
            nodes.insert(insert_at, sel)
            nodes_by_id[sel_id] = sel
            report.inserted_selects.append(f"{sel_id}: first of {cap.semantic_name}")
            insert_at += 1

        index = insert_at

    # ---- pass 1.5: expand fielded entity comparisons ---------------------
    # ``compare(left=movieA, right=movieB, field=rating)`` is compact typed IR.
    # The evidence compiler compares scalar derivations, so make the field
    # projections explicit on both operands before the ordinary projection pass.
    index = 0
    while index < len(nodes):
        node = nodes[index]
        if str(node.get("op") or "") != "compare" or not str(node.get("field") or "").strip():
            index += 1
            continue
        raw_field = str(node.get("field") or "").strip()
        canon = canonical_field(raw_field) or raw_field
        projects = []
        for key in ("left", "right"):
            ref = str(node.get(key) or "")
            if not ref:
                continue
            src = nodes_by_id.get(ref) or {}
            if str(src.get("op") or "") in {"project", "count", "difference"}:
                continue
            pid = _fresh_id("cap_cmp_field", used_ids)
            project = {"id": pid, "op": "project", "source": ref, "field": canon}
            projects.append(project)
            nodes_by_id[pid] = project
            node[key] = pid
        node.pop("field", None)
        if projects:
            # Both scalar projections must precede the comparison in topological
            # order so compilation has both handles available.
            nodes[index:index] = projects
            report.warnings.append(
                f"{node.get('id')}: expanded entity comparison field {canon!r} into scalar projections")
            index += len(projects)
        index += 1

    # ---- pass 2: type-check projections, insert detail hops ---------------
    index = 0
    while index < len(nodes):
        node = nodes[index]
        op = str(node.get("op") or "")
        if op not in {"project", "filter", "compare", "difference"}:
            index += 1
            continue
        raw_field = str(node.get("field") or "").strip()
        if not raw_field or node.get("_oca_capability"):
            index += 1
            continue

        source_id = str(node.get("source") or node.get("left") or "")
        resource = infer_resource(source_id, nodes_by_id, catalog)
        if not resource:
            index += 1
            continue

        canon = canonical_field(raw_field)
        if canon is None:
            report.warnings.append(
                f"{node.get('id')}: unrecognised field {raw_field!r}; left to "
                "the legacy resolver")
            index += 1
            continue

        try:
            _sf, provider_field = resolve_field(canon, resource)
        except FieldTypeError as exc:
            report.type_errors.append(f"{node.get('id')}: {exc}")
            index += 1
            continue

        # From this boundary onward the compiler works with the provider field
        # selected by the typed semantic registry.  Keep the canonical concept as
        # metadata for auditability, but do not make downstream OAS matching guess
        # again from a free-form model phrase.
        node["_oca_semantic_field"] = canon
        node["field"] = provider_field

        # The field belongs to this resource type, but the record we currently
        # hold may not carry it. A person record inside movie credits has a
        # name and an id; it has no birthday. That needs the detail request.
        src_node = nodes_by_id.get(source_id) or {}
        cap_name = str(src_node.get("_oca_capability") or "")
        producing = next((c for c in catalog if c.semantic_name == cap_name), None)
        needs_detail = False
        if producing is not None and not producing.record_path in {"", "$"}:
            card = card_by_ep.get(producing.endpoint)
            if card is not None:
                prefix = producing.record_path
                if f"{prefix}.{provider_field}" not in _leaf_set(card):
                    needs_detail = True

        if needs_detail:
            detail_cap = catalog.try_lookup(resource, "detail", resource)
            if detail_cap is None:
                report.warnings.append(
                    f"{node.get('id')}: {resource}.{canon} needs a detail "
                    f"request but no detail capability is documented")
            else:
                hop_id = _fresh_id("cap_d", used_ids)
                hop = {"id": hop_id, "op": "relation", "source": source_id,
                       "relation": _endpoint_relation_token(detail_cap),
                       "resource": resource,
                       "_oca_capability": detail_cap.semantic_name}
                nodes.insert(index, hop)
                nodes_by_id[hop_id] = hop
                node["source"] = hop_id
                report.inserted_detail_hops.append(
                    f"{hop_id}: {detail_cap.semantic_name} for {canon}")
                index += 1
        index += 1

    # ---- pass 3: make entity equality replayable -------------------------
    # A model may naturally connect entity-producing relation nodes directly to
    # compare(eq/neq). The evidence layer must compare explicit values, not
    # opaque record sets. Insert provider-neutral identity projections (id) on
    # those branches. Selected entities become scalars; unresolved candidate
    # sets remain explicit identity sets and are replayed as entity overlap.
    index = 0
    while index < len(nodes):
        node = nodes[index]
        if (str(node.get("op") or "") != "compare" or
                str(node.get("comparison") or "eq").casefold() not in {"eq", "neq", "equal", "same", "different"}):
            index += 1
            continue
        for key in ("left", "right"):
            ref = str(node.get(key) or "")
            src = nodes_by_id.get(ref) or {}
            if not ref or str(src.get("op") or "") in {"project", "count", "difference", "compare", "membership", "logical_and", "logical_or"}:
                continue
            resource = infer_resource(ref, nodes_by_id, catalog)
            if not resource:
                continue
            try:
                _sf, provider_id = resolve_field("id", resource)
            except FieldTypeError:
                continue
            pid = _fresh_id("cap_identity", used_ids)
            project = {"id": pid, "op": "project", "source": ref,
                       "field": provider_id, "resource": resource,
                       "_oca_semantic_field": "id",
                       "_oca_contract_closure": "entity_identity_compare"}
            nodes.insert(index, project)
            nodes_by_id[pid] = project
            node[key] = pid
            report.warnings.append(
                f"{pid}: inserted explicit {resource} identity for entity equality")
            index += 1
        index += 1

    nodes = [n for n in nodes if str(n.get("op") or "") != "_scope"]
    obj["nodes"] = nodes
    obj["_oca_normalization"] = report.as_dict()
    return obj, report


def _consumed_as_collection(node_id: str, nodes: list[dict[str, Any]]) -> bool:
    """True when a node's records are consumed as a collection downstream.

    Filters are transparent collection transforms.  A relation feeding
    ``filter -> select/count/project`` is still collection-valued and must not
    be collapsed into a scope-only hop merely because the immediate consumer is
    a filter.  This is typed graph structure, independent of question wording.
    """
    children: dict[str, list[dict[str, Any]]] = {}
    for other in nodes:
        children.setdefault(str(other.get("source") or ""), []).append(other)

    seen: set[str] = set()
    stack = [str(node_id)]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        for other in children.get(current, []):
            op = str(other.get("op") or "")
            if op in {"count", "select", "project"}:
                return True
            if op == "relation":
                # A direct child relation does consume a collection in general.
                # Numbered season navigation is the exception: ``tv -> season(2)
                # -> episodes/director/images`` is scope syntax, and the season
                # collection need not be materialized when either the season node
                # itself or its child already carries the explicit provider number.
                # By contrast ``tv -> seasons -> filter(season_number=2) -> ...``
                # must remain a real collection so the filter has the correct
                # record owner.
                current = next((x for x in nodes if str(x.get("id") or "") == str(node_id)), {})
                current_rel = str(current.get("relation") or "").strip().casefold()
                current_lits = current.get("literals") or {}
                child_lits = other.get("literals") or {}
                if (current_rel in {"season", "seasons"}
                        and (current_lits.get("season_number") is not None
                             or current_lits.get("season") is not None
                             or child_lits.get("season_number") is not None
                             or child_lits.get("season") is not None
                             or other.get("_oca_season_scope") is not None)):
                    continue
                return True
            if op == "filter":
                # A typed filter is an explicit record-level operation.  Its
                # source collection must remain materialized; collapsing the
                # relation into path scope makes fields such as season_number
                # disappear before compilation.
                return True
    return False


# --------------------------------------------------------------------------
# post-compile plan checks
# --------------------------------------------------------------------------

def check_plan(plan: dict[str, Any], intent: dict[str, Any] | None = None
               ) -> list[str]:
    """Check generic binding and obligation invariants before execution.

    Every required placeholder must have an upstream producer, and every
    obligation-bearing branch must survive plan normalization/pruning.
    """
    problems: list[str] = []
    steps = list(plan.get("steps") or [])
    produced: set[str] = set()
    for step in steps:
        for name in (step.get("binds") or []):
            produced.add(str(name))
        for name in (step.get("path_bindings") or {}).values():
            produced.add(str(name))

    literal_names: set[str] = set()
    for step in steps:
        literal_names.update(str(k) for k in (step.get("path_literals") or {}))

    for step in steps:
        endpoint = str(step.get("endpoint") or "")
        bindings = step.get("path_bindings") or {}
        for ph in re.findall(r"\{([^{}]+)\}", endpoint):
            if ph in bindings or ph in literal_names:
                continue
            if ph in (step.get("path_literals") or {}):
                continue
            problems.append(
                f"step {step.get('id')}: path parameter {{{ph}}} on {endpoint} "
                f"has no producing step or literal "
                f"(identifies {PLACEHOLDER_RESOURCE.get(ph, 'unknown')})")

    step_ids = {str(s.get("id")) for s in steps}
    for step in steps:
        for dep in (step.get("depends_on") or []):
            if str(dep) not in step_ids:
                problems.append(
                    f"step {step.get('id')}: depends on missing step {dep!r}")

    # Every capability the normaliser anchored must survive into the plan.
    if intent:
        anchored = {
            str(n.get("_oca_capability")): str(n.get("id"))
            for n in (intent.get("nodes") or [])
            if n.get("_oca_capability")
        }
        endpoints = {str(s.get("endpoint") or "") for s in steps}
        for cap_name, node_id in anchored.items():
            module = _capability_endpoint(cap_name, catalog)
            if module and module not in endpoints:
                problems.append(
                    f"intent node {node_id} was anchored to {cap_name} "
                    f"({module}) but no plan step acquires it")
    return problems


def _capability_endpoint(cap_name: str, catalog: CapabilityCatalog) -> str:
    for cap in catalog:
        if cap.semantic_name == cap_name:
            return "" if cap.embedded_in_source else cap.endpoint
    return ""
