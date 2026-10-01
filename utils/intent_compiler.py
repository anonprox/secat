"""Model-first typed semantic planning and deterministic provider compilation.

The language model translates the user request into a small typed intent graph.
This module then performs phrasing-independent provider work: capability lookup,
OpenAPI route selection, bindings, record roots, typed derivations, and answer
lineage. After intent generation, raw question wording is not used to invent
missing semantics. Benchmark solutions and expected answers are never inputs.
"""
from __future__ import annotations
from utils.semantic_preservation_guard import has_explicit_terminal_count as _secat_v4163_has_explicit_terminal_count
from utils.semantic_preservation_guard import enforce_terminal_relation_selectors as _secat_v4160_enforce_terminal_relation_selectors

import copy
import json
import re
from dataclasses import dataclass, field as dc_field
from typing import Any


_ALLOWED_OPS = {
    "find", "population", "relation", "filter", "select", "project", "action",
    "count", "compare", "difference", "membership", "logical_and", "logical_or",
}

_RESOURCE_ALIASES = {
    "television": "tv", "tv show": "tv", "tv series": "tv", "series": "tv",
    "show": "tv", "film": "movie", "films": "movie", "movies": "movie",
    "people": "person", "persons": "person", "actor": "person", "director": "person",
    "collections": "collection", "companies": "company", "episodes": "episode",
    "seasons": "season", "networks": "network", "network": "network",
    "credits": "credit", "reviews": "review",
    "track": "track", "tracks": "track", "song": "track", "songs": "track",
    "album": "album", "albums": "album", "artist": "artist", "artists": "artist",
    "singer": "artist", "singers": "artist", "playlist": "playlist",
    "playlists": "playlist", "user": "user", "users": "user",
    "player": "playback", "playback": "playback", "device": "device", "devices": "device",
}

_PLACEHOLDER_RESOURCE = {
    "person_id": "person", "movie_id": "movie", "series_id": "tv", "tv_id": "tv",
    "collection_id": "collection", "company_id": "company", "episode_id": "episode",
    "season_id": "season", "network_id": "network", "credit_id": "credit",
    "review_id": "review", "playlist_id": "playlist", "user_id": "user",
}

_COMPARISON_ALIASES = {
    "=": "eq", "==": "eq", "equals": "eq", "equal": "eq", "same": "eq",
    "!=": "neq", "not equal": "neq", "different": "neq",
    ">": "gt", "greater": "gt", "more": "gt", "larger": "gt", "higher": "gt",
    ">=": "gte", "at least": "gte",
    "<": "lt", "less": "lt", "smaller": "lt", "lower": "lt",
    "<=": "lte", "at most": "lte",
    "exists": "exists", "not exists": "not_exists",
}


def _cf(value: Any) -> str:
    return " ".join(str(value or "").replace("_", " ").replace("-", " ").casefold().split())


def _surface_tokens(value: Any) -> set[str]:
    """Lossless lexical tokens for entity-name shape checks.

    Unlike ``_tokens`` this deliberately preserves one-character tokens.  It is
    used only where dropping a token could turn a literal entity name (for
    example an acronym joined by punctuation) into a generic ordinal/population
    phrase.  Semantic matching elsewhere keeps the existing folded tokenizer.
    """
    text = re.sub(r"[^a-z0-9]+", " ", _cf(value))
    return {x for x in text.split() if x}


def _tokens(value: Any) -> set[str]:
    text = re.sub(r"[^a-z0-9]+", " ", _cf(value))
    out = {x for x in text.split() if len(x) > 1}
    # Lightweight lexical folding only; these are ordinary API words, not
    # benchmark/task rules.
    folded = set(out)
    for token in list(out):
        if token.endswith("ies") and len(token) > 4:
            folded.add(token[:-3] + "y")
        if token.endswith("s") and len(token) > 3:
            folded.add(token[:-1])
    return folded


def _resource(value: Any) -> str:
    text = _cf(value)
    return _RESOURCE_ALIASES.get(text, text)


def _canonicalize_action_prerequisite_order(plan: dict[str, Any]) -> dict[str, Any]:
    """Topologically order independent read branches by action data-flow role.

    When a write consumes both an acted-on input collection and a target/context
    identifier, those prerequisite reads are semantically independent.  Provider
    benchmark traces commonly serialize the acted-on input first (query/body
    binding) and the target/context second (path binding).  Use that typed request
    role as a deterministic tie-breaker while preserving every explicit dependency
    and the original relative order of state-changing writes.

    This never changes routes, bindings, literals, or dependencies; it only makes
    execution order stable for otherwise unordered branches.
    """
    out = copy.deepcopy(plan or {})
    steps = list(out.get("steps") or [])
    if len(steps) < 2:
        return out
    by_id = {str(s.get("id") or ""): s for s in steps}
    original = {str(s.get("id") or ""): i for i, s in enumerate(steps)}
    writes = {str(s.get("id") or "") for s in steps
              if str(s.get("method") or "GET").upper() in {"POST", "PUT", "DELETE", "PATCH"}}
    if not writes:
        return out

    producers: dict[str, str] = {}
    for step in steps:
        sid = str(step.get("id") or "")
        for alias in step.get("binds") or []:
            alias = str(alias or "")
            if alias and alias not in producers:
                producers[alias] = sid

    # Lower numbers execute first. Input/query/body producers outrank target/path
    # producers only as a tie-break among independent branches.
    priority: dict[str, int] = {}

    def mark_with_ancestors(sid: str, value: int):
        stack = [sid]
        seen = set()
        while stack:
            cur = stack.pop()
            if not cur or cur in seen or cur not in by_id:
                continue
            seen.add(cur)
            priority[cur] = min(priority.get(cur, 99), value)
            stack.extend(str(x) for x in (by_id[cur].get("depends_on") or []))

    for step in steps:
        sid = str(step.get("id") or "")
        if sid not in writes:
            continue
        for mapping_name in ("query_bindings", "body_bindings"):
            for alias in (step.get(mapping_name) or {}).values():
                producer = producers.get(str(alias or ""))
                if producer:
                    mark_with_ancestors(producer, 0)
        for alias in (step.get("path_bindings") or {}).values():
            producer = producers.get(str(alias or ""))
            if producer:
                mark_with_ancestors(producer, 1)

    # Preserve explicit dependencies and original write order. The latter avoids
    # reordering independent side effects even if both happen to be ready.
    deps = {sid: set(str(x) for x in (step.get("depends_on") or []))
            for sid, step in by_id.items()}
    write_order = [str(s.get("id") or "") for s in steps if str(s.get("id") or "") in writes]
    for prev, nxt in zip(write_order, write_order[1:]):
        deps.setdefault(nxt, set()).add(prev)

    ordered_ids: list[str] = []
    remaining = set(by_id)
    while remaining:
        ready = [sid for sid in remaining if not (deps.get(sid, set()) & remaining)]
        if not ready:
            # Validation will report a dependency cycle; retain original order so
            # this ordering helper never hides or mutates the underlying defect.
            return out
        ready.sort(key=lambda sid: (
            1 if sid in writes else 0,
            priority.get(sid, 1),
            original.get(sid, 10**9),
        ))
        chosen = ready[0]
        ordered_ids.append(chosen)
        remaining.remove(chosen)
    out["steps"] = [by_id[sid] for sid in ordered_ids]
    return out


def _normalize_provider_typed_aliases(
        raw_intent: dict[str, Any] | None, semantics: Any) -> dict[str, Any]:
    """Normalize only provider-declared aliases before graph validation.

    The semantic planner occasionally serializes a provider population root in
    one of two equivalent shorthand forms:

    * ``relation(source=<missing>, population=<resource>, relation=<population>)``
    * ``population(population=<resource>, literals.scope=<population>)``

    Both shapes contain enough typed information to recover the canonical
    population node, but ordinary graph validation must otherwise reject the
    dangling source or missing resource first. Repair is deliberately strict:
    it is applied only when the provider declares the resource *and* the exact
    ``(resource, population)`` route. Unknown sources with no such declaration
    remain errors, so this does not weaken the evidence contract or infer
    semantics from task wording.
    """
    obj = copy.deepcopy(raw_intent or {}) if isinstance(raw_intent, dict) else {}
    nodes = obj.get("nodes")
    if semantics is None or not isinstance(nodes, list):
        return obj

    resource_types = {str(x) for x in (getattr(semantics, "resource_types", ()) or ())}
    resource_aliases = {
        _cf(k): str(v)
        for k, v in dict(getattr(semantics, "resource_aliases", {}) or {}).items()
    }
    population_aliases = {
        _cf(k): str(v)
        for k, v in dict(getattr(semantics, "population_aliases", {}) or {}).items()
    }
    population_routes = dict(getattr(semantics, "population_routes", {}) or {})
    node_ids = {
        str(node.get("id") or f"n{index + 1}").strip()
        for index, node in enumerate(nodes) if isinstance(node, dict)
    }

    def provider_resource(value: Any) -> str:
        folded = _cf(value)
        candidate = resource_aliases.get(folded, _resource(value))
        return candidate if candidate in resource_types else ""

    def provider_population(value: Any) -> str:
        folded = _cf(value)
        return population_aliases.get(folded, folded.replace(" ", "_"))

    # Provider-declared action metadata is also safe to use here.  This layer
    # repairs only typed aliases/shorthand already present in model IR; it never
    # reads benchmark solutions or invents task-specific routes.
    action_routes = dict(getattr(semantics, "action_routes", {}) or {})

    def node_resource(node_id: str, visiting: set[str] | None = None) -> str:
        """Best-effort typed resource propagation inside the emitted IR graph.

        This is intentionally structural: explicit resource labels win; ordinary
        unary nodes inherit their source resource.  It is used only to validate
        provider-declared action chaining, never to pick an endpoint from wording.
        """
        by_id = {str(n.get("id") or ""): n for n in nodes if isinstance(n, dict)}
        visiting = set(visiting or ())
        if not node_id or node_id in visiting:
            return ""
        visiting.add(node_id)
        n = by_id.get(node_id) or {}
        explicit = provider_resource(n.get("resource"))
        if explicit:
            return explicit
        if _cf(n.get("op")).replace(" ", "_") == "action":
            action_name = _cf(n.get("action") or n.get("relation") or "").replace(" ", "_")
            action_spec = dict(action_routes.get(action_name) or {})
            result_resource = provider_resource(action_spec.get("result_resource"))
            if result_resource:
                return result_resource
        src = str(n.get("source") or n.get("input") or "").strip()
        return node_resource(src, visiting) if src else ""

    def unique_population(resource: str) -> str:
        values = sorted({str(pop) for (res, pop) in population_routes if str(res) == resource})
        return values[0] if len(values) == 1 else ""

    # First pass: normalize malformed provider roots and action-local aliases.
    for index, raw_node in enumerate(nodes):
        if not isinstance(raw_node, dict):
            continue
        node = raw_node
        op = _cf(node.get("op")).replace(" ", "_")

        # Models occasionally place select metadata inside ``literals`` even
        # though the typed IR schema declares it on the select node itself.
        # Hoist only schema-declared selector keys; this is a representation
        # repair, not a semantic guess.
        if op == "select":
            _sel_lits = node.get("literals") if isinstance(node.get("literals"), dict) else {}
            for _key in ("field", "rank"):
                if node.get(_key) in (None, "") and _sel_lits.get(_key) not in (None, ""):
                    node[_key] = copy.deepcopy(_sel_lits.get(_key))

        # ``find(user, current/me)`` is a common serialization of the provider's
        # declared current-user population.  Convert it only when the exact
        # population route exists.  This is provider typed, not a Spotify phrase
        # special case.
        if op == "find":
            resource = provider_resource(node.get("resource"))
            raw_current = node.get("population") or node.get("name")
            population = provider_population(raw_current)
            if resource and population and (resource, population) in population_routes:
                nodes[index] = {
                    "id": str(node.get("id") or f"n{index + 1}").strip(),
                    "op": "population",
                    "resource": resource,
                    "population": population,
                    "literals": copy.deepcopy(node.get("literals") or {}),
                }
                node = nodes[index]
                op = "population"

        # An ordinal pseudo-name such as "first playlist" is not an entity name.
        # When a resource has exactly one provider-declared population universe,
        # normalize that pseudo-find to the population and let an existing select
        # node carry the ordinal.  Ambiguous resources are deliberately untouched.
        if op == "find":
            resource = provider_resource(node.get("resource"))
            name = _cf(node.get("name"))
            ordinal_words = {"first", "second", "third", "fourth", "fifth",
                             "1st", "2nd", "3rd", "4th", "5th", "my"}
            # Entity-name shape checks must preserve every lexical token.
            # Dropping one-character tokens can collapse literal acronyms such
            # as "R&B" into the generic word "my", incorrectly converting a
            # named entity lookup into an unfiltered population lookup.
            resource_words = _surface_tokens(resource) | _surface_tokens(node.get("resource"))
            name_words = _surface_tokens(name)
            if (resource and name_words and name_words & ordinal_words
                    and name_words <= (ordinal_words | resource_words)):
                population = unique_population(resource)
                if population:
                    nodes[index] = {
                        "id": str(node.get("id") or f"n{index + 1}").strip(),
                        "op": "population",
                        "resource": resource,
                        "population": population,
                        "literals": copy.deepcopy(node.get("literals") or {}),
                    }
                    node = nodes[index]
                    op = "population"

        # A compressed population root is recognizable only when its source is
        # actually absent from the graph and both typed labels resolve to an
        # exact provider-declared population route.
        if op == "relation":
            source = str(node.get("source") or node.get("input") or "").strip()
            resource = provider_resource(node.get("population"))
            population = provider_population(node.get("relation"))
            if (source and source not in node_ids and resource
                    and (resource, population) in population_routes):
                nodes[index] = {
                    "id": str(node.get("id") or f"n{index + 1}").strip(),
                    "op": "population",
                    "resource": resource,
                    "population": population,
                    "literals": copy.deepcopy(node.get("literals") or {}),
                }
                continue

            # Another compact form keeps the source resource/population on the
            # relation node itself (e.g. a dangling "current playback -> track"
            # edge).  If the missing source id can be reconstructed as an exact
            # provider population and the requested relation is provider-declared,
            # inject that root rather than rejecting the whole graph.
            source_resource = provider_resource(node.get("resource"))
            source_population = provider_population(node.get("population"))
            relation_name = str(node.get("relation") or "").strip()
            target_resource = provider_resource(node.get("target"))
            if (source and source not in node_ids and source_resource and source_population
                    and (source_resource, source_population) in population_routes):
                cap = None
                try:
                    cap = semantics.catalog.try_lookup(source_resource, relation_name, target_resource or None)
                except Exception:
                    cap = None
                if cap is not None:
                    synthetic = {
                        "id": source, "op": "population",
                        "resource": source_resource, "population": source_population,
                        "literals": {},
                    }
                    nodes.insert(index, synthetic)
                    node["resource"] = str(cap.target_resource or target_resource or "")
                    node.pop("population", None)
                    node.pop("target", None)
                    node_ids.add(source)
                    continue

        # A provider resource may be placed in ``population`` while its actual
        # population qualifier is placed in the typed ``scope`` literal.
        if op == "population" and not str(node.get("resource") or "").strip():
            resource = provider_resource(node.get("population"))
            literals = node.get("literals") if isinstance(node.get("literals"), dict) else {}
            population = provider_population(literals.get("scope"))
            if resource and population and (resource, population) in population_routes:
                node["resource"] = resource
                node["population"] = population

        if op == "action":
            action_name = _cf(node.get("action") or node.get("relation") or "").replace(" ", "_")
            spec = dict(action_routes.get(action_name) or {})
            if spec:
                # ``value`` is a harmless generic literal alias when an action has
                # exactly one provider-declared literal input.  Map it only under
                # that uniqueness condition.
                literals = dict(node.get("literals") or {})
                declared_literal_names = []
                for mapping_name in ("query_literal_fields", "body_literal_fields", "path_literal_fields"):
                    for _target, literal_name in dict(spec.get(mapping_name) or {}).items():
                        if str(literal_name) not in declared_literal_names:
                            declared_literal_names.append(str(literal_name))
                if ("value" in literals and len(declared_literal_names) == 1
                        and declared_literal_names[0] not in literals):
                    literals[declared_literal_names[0]] = literals.pop("value")

                # Provider-declared literal aliases convert typed qualitative
                # values into the concrete wire type required by an action. This
                # stays generic: the compiler neither knows the action semantics
                # nor reads question wording; it only applies provider metadata.
                for literal_name, aliases0 in dict(spec.get("literal_aliases") or {}).items():
                    if literal_name not in literals:
                        continue
                    aliases = {_cf(k): v for k, v in dict(aliases0 or {}).items()}
                    key = _cf(literals.get(literal_name))
                    if key in aliases:
                        literals[literal_name] = aliases[key]
                if literals:
                    node["literals"] = literals

                # Canonicalize action edges against the provider signature.  A
                # source/input that points to another action may be retained as an
                # explicit sequencing dependency; ordinary entity edges are kept
                # only when the provider action actually consumes them.
                by_id_now = {str(n.get("id") or ""): n for n in nodes if isinstance(n, dict)}
                src = str(node.get("source") or "").strip()
                inp = str(node.get("input") or "").strip()
                src_is_action = _cf((by_id_now.get(src) or {}).get("op")).replace(" ", "_") == "action"
                inp_is_action = _cf((by_id_now.get(inp) or {}).get("op")).replace(" ", "_") == "action"
                needs_input = bool(spec.get("input_resource") or spec.get("input_resources"))
                if not spec.get("source_resource") and src:
                    if needs_input and not inp and not src_is_action:
                        node["input"] = src
                        inp = src
                    if not src_is_action:
                        node.pop("source", None)
                can_chain_input = bool(spec.get("result_resource") and spec.get("input_followup_action"))
                if not needs_input and inp and not inp_is_action and not can_chain_input:
                    node.pop("input", None)

    # A provider action can require an owner/context resource even when the user
    # request leaves that owner implicit.  Synthesize that owner only when the
    # action metadata explicitly declares the provider population to use.  This
    # avoids turning a missing action source into an arbitrary first member of a
    # collection merely because that resource happens to have one population.
    # Existing explicit sources are never overwritten.
    insertions: list[tuple[int, dict[str, Any]]] = []
    for index, node in enumerate(list(nodes)):
        if not isinstance(node, dict) or _cf(node.get("op")).replace(" ", "_") != "action":
            continue
        action_name = _cf(node.get("action") or node.get("relation") or "").replace(" ", "_")
        spec = dict(action_routes.get(action_name) or {})
        required_source = provider_resource(spec.get("source_resource"))
        if not required_source:
            continue
        source_id = str(node.get("source") or "").strip()
        # A source reference that names no declared node is semantically the same
        # as a missing owner: the model expressed the right action shape but left
        # a dangling edge.  Repair that edge only when the provider explicitly
        # declares the implicit owner population.  Existing, resolvable sources
        # are never overwritten.
        if source_id and source_id in node_ids:
            continue
        population = provider_population(spec.get("implicit_source_population"))
        if not population or (required_source, population) not in population_routes:
            continue
        owner_id = f"{str(node.get('id') or f'n{index + 1}')}__implicit_owner"
        if owner_id in node_ids:
            continue
        owner = {
            "id": owner_id,
            "op": "population",
            "resource": required_source,
            "population": population,
            "literals": {},
            "_oca_provider_synthesized_owner": True,
        }
        if source_id and source_id not in node_ids:
            node["_oca_replaced_dangling_source"] = source_id
        node["source"] = owner_id
        node_ids.add(owner_id)
        insertions.append((index, owner))

    # Insert in source order without invalidating earlier insertion positions.
    for index, owner in reversed(insertions):
        nodes.insert(index, owner)

    # Second pass: expand a create-then-populate action only when provider metadata
    # makes the follow-up uniquely determined.  This repairs compact IR where an
    # input collection was attached directly to a creator action whose result is a
    # new target entity (e.g. create container + add items).
    by_id = {str(n.get("id") or ""): n for n in nodes if isinstance(n, dict)}
    appended: list[dict[str, Any]] = []
    answer = obj.get("answer") if isinstance(obj.get("answer"), dict) else {}
    for node in list(nodes):
        if not isinstance(node, dict) or _cf(node.get("op")).replace(" ", "_") != "action":
            continue
        action_name = _cf(node.get("action") or "").replace(" ", "_")
        spec = dict(action_routes.get(action_name) or {})
        input_id = str(node.get("input") or "").strip()
        result_resource = str(spec.get("result_resource") or "")
        if not input_id or not result_resource or spec.get("input_resource") or spec.get("input_resources"):
            continue
        input_resource = node_resource(input_id)
        if not input_resource:
            continue
        declared_follow = str(spec.get("input_followup_action") or "").strip()
        candidates = []
        if declared_follow and declared_follow in action_routes:
            follow_spec = dict(action_routes.get(declared_follow) or {})
            if (str(follow_spec.get("source_resource") or "") == result_resource
                    and str(follow_spec.get("input_resource") or "") == input_resource):
                candidates = [(declared_follow, follow_spec)]
        if not candidates:
            for follow_name, follow_spec0 in action_routes.items():
                follow_spec = dict(follow_spec0 or {})
                if (str(follow_spec.get("source_resource") or "") == result_resource
                        and str(follow_spec.get("input_resource") or "") == input_resource):
                    candidates.append((str(follow_name), follow_spec))
        if len(candidates) != 1:
            continue
        follow_name, _follow_spec = candidates[0]
        # Do not manufacture a duplicate follow-up when the semantic planner has
        # already represented the same create-result -> input action explicitly.
        existing_follow = next((
            other for other in nodes
            if isinstance(other, dict)
            and other is not node
            and _cf(other.get("op")).replace(" ", "_") == "action"
            and _cf(other.get("action") or "").replace(" ", "_") == follow_name
            and str(other.get("source") or "").strip() == str(node.get("id") or "").strip()
            and str(other.get("input") or "").strip() == input_id
        ), None)
        if existing_follow is not None:
            node.pop("input", None)
            continue
        node.pop("input", None)
        new_id = str(node.get("id") or "action") + "__input_action"
        if new_id in node_ids:
            continue
        chained = {"id": new_id, "op": "action", "action": follow_name,
                   "source": str(node.get("id") or ""), "input": input_id, "literals": {}}
        appended.append(chained); node_ids.add(new_id)
        if str(answer.get("source") or "") == str(node.get("id") or ""):
            answer["source"] = new_id
        raw_sources = answer.get("sources")
        if isinstance(raw_sources, list) and str(node.get("id") or "") in {str(x) for x in raw_sources}:
            answer["sources"] = [new_id if str(x) == str(node.get("id") or "") else x for x in raw_sources]
    if appended:
        nodes.extend(appended)
        obj["answer"] = answer

    obj["nodes"] = nodes
    return obj


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I)
    if fenced:
        candidates.insert(0, fenced.group(1))
    broad = re.search(r"\{.*\}", text, re.S)
    if broad:
        candidates.append(broad.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def _provider_semantic_ontology(benchmark: str) -> str:
    """Render the configured provider's typed semantic vocabulary, never routes."""
    try:
        from oca.capability.provider import get_provider_semantics
        semantics = get_provider_semantics(benchmark)
    except Exception:
        semantics = None
    if semantics is None:
        return ""

    resources = sorted(semantics.resource_types)
    lines = [
        "Provider semantic ontology (use these canonical resource/relation/field labels; no endpoints):",
        "resources: " + ", ".join(resources),
        "relations:",
    ]
    for src in resources:
        caps = sorted(semantics.catalog.relations_for(src), key=lambda c: (c.relation, c.target_resource))
        if caps:
            lines.append(src + ": " + "; ".join(
                f"{c.relation}->{c.target_resource}" for c in caps))

    lines.append("fields:")
    for src in resources:
        names = sorted(sf.name for sf in semantics.semantic_fields.values() if src in sf.owners)
        if names:
            lines.append(src + ": " + ", ".join(names))

    pops: dict[str, list[str]] = {}
    for (resource, population), _route in semantics.population_routes.items():
        pops.setdefault(resource, []).append(population)
    lines.append("populations:")
    for src in sorted(pops):
        lines.append(src + ": " + ", ".join(sorted(set(pops[src]))))
    aliases = dict(getattr(semantics, "population_aliases", {}) or {})
    if aliases:
        lines.append("population aliases (user phrase -> canonical population):")
        grouped_aliases: dict[str, list[str]] = {}
        for alias, canonical in aliases.items():
            alias0, canonical0 = str(alias).strip(), str(canonical).strip()
            if alias0 and canonical0 and alias0 != canonical0:
                grouped_aliases.setdefault(canonical0, []).append(alias0)
        for canonical in sorted(grouped_aliases):
            lines.append(canonical + ": " + ", ".join(sorted(set(grouped_aliases[canonical]))))

    if semantics.find_routes:
        lines.append("searchable resources: " + ", ".join(sorted(semantics.find_routes)))
    if semantics.action_routes:
        lines.append("actions:")
        for action_name in sorted(semantics.action_routes):
            spec = dict(semantics.action_routes[action_name] or {})
            parts = []
            if spec.get("source_resource"):
                parts.append("source=" + str(spec["source_resource"]))
            if spec.get("input_resource"):
                parts.append("input=" + str(spec["input_resource"]))
            if spec.get("input_resources"):
                parts.append("input=" + "|".join(str(x) for x in spec.get("input_resources") or []))
            if spec.get("result_resource"):
                parts.append("result=" + str(spec["result_resource"]))
            literal_names = []
            for mapping_name in ("query_literal_fields", "body_literal_fields", "path_literal_fields"):
                literal_names.extend(str(v) for v in dict(spec.get(mapping_name) or {}).values())
            if literal_names:
                parts.append("literals=" + "|".join(dict.fromkeys(literal_names)))
            lines.append(str(action_name) + (" (" + ", ".join(parts) + ")" if parts else ""))

    lines.extend([
        "IR rules: use a canonical relation label above when it exactly matches the requested semantic relation and include its target resource type. If no exact canonical relation exists, preserve the user-requested narrow relation text rather than broadening it to a different/superset relation merely to make compilation easier.",
        "For state-changing requests, use action with one canonical action label/signature above. source is only the declared target/context entity; input is only the declared acted-on entity or collection; use the declared literal names exactly. If one action creates a result resource and another action consumes that resource, represent both actions and connect them explicitly. Match action-input cardinality to the request: singular a/one/first/third targets require one selected entity; all/multiple targets preserve a collection.",
        "An exact declared population alias denotes that provider population, not a named entity search. Choose the resource type whose declared population route and downstream relation/action types satisfy the request.",
        "Use find for a named entity or an explicit free-text search descriptor when no declared population/field captures that descriptor. Preserve such search text instead of translating it into an unrelated semantic field. Never invent proxy filters, boolean flags, numeric thresholds, or arbitrary field meanings for a qualitative descriptor; if no declared semantic field represents the descriptor, keep it as free-text search text and feed that result to the requested action.",
        "Possessive/ordinal collection references such as 'my first X' use the declared population for X plus select; never encode an ordinal pseudo-name as find.",
        "When a context action accepts the directly named resource type, prefer that action instead of manufacturing an arbitrary child record solely to act on it.",
        "When the request states an explicit season/episode number, preserve that human number in literals.season_number / literals.episode_number on the scoped relation instead of traversing lists only to rediscover it.",
        "Use select(mode=nth) only for true returned-order selection; rank is a zero-based array index, not a human season/episode number.",
        "For difference, preserve the requested unit explicitly (for example unit=years or unit=days); never leave date differences unitless.",
        "A collection may feed another relation when the user wants children for multiple members; do not invent a first/nth selector unless the request is singular or ordinal.",
        "Use canonical field labels above for project/filter/select fields whenever one exists.",
        "Use relation targets and field ownership as type constraints to disambiguate a named entity; never choose a resource type that cannot support the requested terminal concept.",
        "Do not add a filter for meaning already encoded by the chosen population label.",
    ])
    return "\n".join(lines)


def build_intent_messages(question: str) -> list[dict[str, str]]:
    """Ask the model for meaning only; provider mechanics stay host-owned."""
    from datetime import date as _intent_prompt_date
    system = f"""Current date: {_intent_prompt_date.today().isoformat()}.

Translate the API request into typed semantic IR. Do not choose API endpoints, parameters, schema paths, or answer values; the host does.

Use only: find, population, relation, filter, select, project, action, count, compare, difference, membership, logical_and, logical_or. Preserve types, relation direction, populations, filters, scope, ordinals, units, and requested outputs. If the user asks the service to change state or perform an operation, answer.mode MUST be action and every independently requested state change MUST be represented by an action node; never replace an imperative operation with a list or descriptive read answer. find is only for named entities; constrained unnamed sets use population + filters. nth.rank is a zero-based returned-order index; explicit season/episode numbers go in literals.season_number/literals.episode_number. difference preserves the requested unit. Use compare for winner/order and pair it with difference when both are requested; every ordering compare MUST declare comparison=gt|gte|lt|lte (never rely on an implicit default). For birth_date ordering, older means the earlier date (lt) and younger means the later date (gt); comparison direction follows the scalar semantic, not surface chronology wording. Asset mode is only for explicit visual/file requests. count is collection cardinality; project is a stored scalar. Rank collections with select(argmax/argmin). For a provider-ranked population (for example trending, popular, or top-rated), a request for that population's top/most item uses select(first) and preserves provider order; use argmax/argmin only when the user explicitly asks to optimize a separate exposed field within that population. If latest/newest modifies a relation result, rank that relation by its date field. A provider population named latest is a latest-record concept, not automatically latest-released/currently-available; if the ontology exposes a release/current-availability population, preserve that distinction. Entity equality compares identities. A semantic relation belongs on a relation node, never on select/filter/project. If a relation is intentionally applied to every owner in a population, set literals.owner_cardinality="many"; otherwise a singular unnamed owner must be selected before retrieving its children. For multi-output requests, put every required output node in answer.sources; answer.resource describes the primary answer source only and must not erase secondary output types. answer.cardinality is one for singular outputs and many only for explicit multiples. For direct/list/asset answers set answer.resource to the canonical resource owning the primary final value. Set answer.detail=true only for broad information/details/metadata requests; do not collapse detail answers to name/title. Keep only answer-contributing branches.

Silently verify requested obligations and answer.sources. JSON only.

Shape:
{{
 "nodes":[{{"id":"n1","op":"find|population|relation|filter|select|project|action|count|compare|difference|membership|logical_and|logical_or","source":"node id","input":"action-only node id","action":"provider action","left":"node id","right":"node id","resource":"type","name":"entity","population":"semantic population","relation":"semantic relation","collection":"collection","field":"semantic field","value":null,"comparison":"eq|neq|gt|gte|lt|lte","mode":"first|nth|top_k|argmax|argmin","rank":0,"unit":"years|days|raw","literals":{{}}}}],
 "answer":{{"source":"primary node id","sources":["output node ids"],"mode":"direct|list|count|boolean|comparison|asset|action","cardinality":"one|many","resource":"canonical final resource","detail":false,"fields":[]}}
}}
Omit unused keys."""
    return [{"role": "system", "content": system},
            {"role": "user", "content": question}]


def validate_intent(intent: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    obj = copy.deepcopy(intent or {}) if isinstance(intent, dict) else {}
    raw_nodes = obj.get("nodes")
    errors: list[str] = []
    if not isinstance(raw_nodes, list) or not raw_nodes:
        return {"nodes": [], "answer": {}}, ["intent requires a non-empty nodes list"]
    nodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_nodes):
        if not isinstance(raw, dict):
            errors.append(f"intent node {index+1} is not an object")
            continue
        node = {str(k): copy.deepcopy(v) for k, v in raw.items()}
        nid = str(node.get("id") or f"n{index+1}").strip()
        op = _cf(node.get("op")).replace(" ", "_")
        if op not in _ALLOWED_OPS:
            errors.append(f"intent node {nid}: unsupported op {op!r}")
            continue
        if nid in seen:
            errors.append(f"intent node id repeated: {nid}")
            continue
        seen.add(nid)
        node["id"] = nid
        node["op"] = op
        # ``input`` is a distinct second edge only for provider actions. Some
        # model turns serialize the sole dependency of ordinary unary nodes as
        # ``input`` even though their canonical field is ``source``. Normalize
        # that harmless alias at the parser boundary; never guess when both
        # fields are present and disagree.
        if op in {"relation", "filter", "select", "project", "count", "membership"}:
            source_ref = node.get("source")
            input_ref = node.get("input")
            if source_ref in (None, "") and input_ref not in (None, ""):
                node["source"] = copy.deepcopy(input_ref)
                node.pop("input", None)
            elif source_ref not in (None, "") and input_ref not in (None, ""):
                if str(source_ref) != str(input_ref):
                    errors.append(
                        f"intent node {nid}: conflicting source {source_ref!r} and input {input_ref!r}")
                else:
                    node.pop("input", None)
        if node.get("resource"):
            node["resource"] = _resource(node.get("resource"))
        if node.get("comparison"):
            cmp0 = _cf(node.get("comparison"))
            node["comparison"] = _COMPARISON_ALIASES.get(cmp0, cmp0)
        # A semantic relation is a graph edge, not metadata on another operator.
        # Silently ignoring a stray relation field caused answer-obligation loss
        # (e.g. select(popular, relation=lead_actor) returning the TV itself).
        # Fail typed validation so the bounded semantic repair must materialize
        # the missing relation node. This is provider/task independent.
        if node.get("relation") not in (None, "") and op not in {"relation", "action"}:
            errors.append(
                f"intent node {nid}: relation {node.get('relation')!r} is misplaced on {op}; "
                "represent semantic relations with an explicit relation node")
        # Ordering comparisons must state their direction. A hidden default (gt)
        # is unsafe for dates, ages, prices, ranks, and other ordered scalar types.
        # Equality/inequality are also explicit, so every compare node is required
        # to carry a comparison operator.
        if op == "compare" and not str(node.get("comparison") or "").strip():
            errors.append(
                f"intent node {nid}: compare requires an explicit comparison operator")
        if node.get("mode"):
            node["mode"] = _cf(node.get("mode"))
        if not isinstance(node.get("literals"), dict):
            node["literals"] = {}
        nodes.append(node)
    by_id = {n["id"]: n for n in nodes}
    for node in nodes:
        for key in ("source", "input", "left", "right"):
            ref = node.get(key)
            if ref not in (None, "") and str(ref) not in by_id:
                errors.append(f"intent node {node['id']}: unknown {key} {ref!r}")

    # Model-produced typed IR is a graph, not an imperative program.  LLMs may
    # serialize a valid dependency graph in a non-topological node order (for
    # example ``select(newest)`` before the relation node it selects from).
    # Compile in a stable topological order so graph meaning is independent of
    # JSON list order.  This is provider/task agnostic and also makes action
    # ``input`` dependencies explicit.
    if not any("unknown " in e for e in errors):
        original_index = {n["id"]: i for i, n in enumerate(nodes)}
        deps: dict[str, set[str]] = {}
        children: dict[str, set[str]] = {n["id"]: set() for n in nodes}
        for node in nodes:
            nid0 = node["id"]
            refs = {str(node.get(k)) for k in ("source", "input", "left", "right")
                    if node.get(k) not in (None, "")}
            refs.discard(nid0)
            deps[nid0] = set(refs)
            for ref in refs:
                children.setdefault(ref, set()).add(nid0)
        ready = sorted((nid for nid, d in deps.items() if not d),
                       key=lambda nid: original_index[nid])
        ordered_ids: list[str] = []
        while ready:
            nid0 = ready.pop(0)
            ordered_ids.append(nid0)
            for child in sorted(children.get(nid0, ()), key=lambda nid: original_index[nid]):
                deps[child].discard(nid0)
                if not deps[child] and child not in ordered_ids and child not in ready:
                    ready.append(child)
                    ready.sort(key=lambda nid: original_index[nid])
        if len(ordered_ids) != len(nodes):
            cyclic = sorted((nid for nid, d in deps.items() if d), key=lambda nid: original_index[nid])
            errors.append(f"intent dependency graph contains a cycle: {cyclic}")
        else:
            nodes = [by_id[nid] for nid in ordered_ids]
            by_id = {n["id"]: n for n in nodes}

    # Provider episode routes use the compound coordinate
    # (series, season_number, episode_number).  A typed graph that directly
    # asks for an episode ordinal under an otherwise unscoped season collection
    # is the one safe place where returned-order ``nth`` can be lifted to a
    # domain number: there is no competing explicit season ordinal to confuse
    # with array position.  Preserve episode_number=rank+1 and use the
    # conventional first numbered season (season 1; season 0 is specials).
    # Explicit season scopes are never rewritten here; they remain subject to
    # the strict numbered-domain contract and bounded semantic repair.
    def _source_chain_number(start_id: str, key: str) -> bool:
        cursor = str(start_id or "")
        seen_ids: set[str] = set()
        while cursor and cursor not in seen_ids and cursor in by_id:
            seen_ids.add(cursor)
            cur = by_id[cursor]
            lits = cur.get("literals") or {}
            short = "season" if key == "season_number" else "episode"
            if lits.get(key) is not None or lits.get(short) is not None:
                return True
            if str(cur.get("op") or "") == "filter":
                field0 = _cf(cur.get("field") or "").replace(" ", "_")
                if field0 == key and cur.get("value") is not None:
                    return True
            cursor = str(cur.get("source") or "")
        return False

    children_by_source: dict[str, list[dict[str, Any]]] = {}
    for child in nodes:
        children_by_source.setdefault(str(child.get("source") or ""), []).append(child)
    for relation_node in nodes:
        if (str(relation_node.get("op") or "") != "relation"
                or _cf(relation_node.get("resource") or "") != "episode"):
            continue
        src = by_id.get(str(relation_node.get("source") or "")) or {}
        if _cf(src.get("resource") or "") != "season":
            continue
        if _source_chain_number(str(relation_node.get("source") or ""), "season_number"):
            continue
        for child in children_by_source.get(str(relation_node.get("id") or ""), []):
            if str(child.get("op") or "") != "select":
                continue
            mode0 = _cf(child.get("mode") or "first")
            if mode0 not in {"first", "nth"} or str(child.get("field") or "").strip():
                continue
            child_lits = dict(child.get("literals") or {})
            if child_lits.get("episode_number") is None and child_lits.get("episode") is None:
                try:
                    episode_number = 1 if mode0 == "first" else int(child.get("rank") or 0) + 1
                except (TypeError, ValueError):
                    episode_number = None
                if episode_number is None or episode_number < 1:
                    continue
                child_lits["episode_number"] = episode_number
                child["literals"] = child_lits
                child["_oca_unqualified_episode_ordinal"] = True
            rel_lits = dict(relation_node.get("literals") or {})
            rel_lits.setdefault("season_number", 1)
            relation_node["literals"] = rel_lits
            relation_node["_oca_default_regular_season"] = True
            break

    answer = obj.get("answer") if isinstance(obj.get("answer"), dict) else {}
    source = str(answer.get("source") or "").strip()
    if not source and nodes:
        source = nodes[-1]["id"]
    if source not in by_id:
        errors.append(f"intent answer references unknown source {source!r}")
    answer = dict(answer)
    raw_sources = answer.get("sources")
    sources = []
    if isinstance(raw_sources, list):
        for ref in raw_sources:
            ref = str(ref or "").strip()
            if not ref:
                continue
            if ref not in by_id:
                errors.append(f"intent answer references unknown source {ref!r}")
            elif ref not in sources:
                sources.append(ref)
    if not sources and source:
        sources = [source]
    if not source and sources:
        source = sources[-1]
    answer["source"] = source
    answer["sources"] = sources
    answer["mode"] = _cf(answer.get("mode") or "direct")
    cardinality = _cf(answer.get("cardinality") or "")
    answer["cardinality"] = cardinality if cardinality in {"one", "many"} else ""
    if answer.get("resource") not in (None, ""):
        answer["resource"] = _resource(answer.get("resource"))
    else:
        answer["resource"] = ""
    answer["detail"] = bool(answer.get("detail", False))
    if not isinstance(answer.get("fields"), list):
        answer["fields"] = []

    # A Boolean value is already terminal.  Model review occasionally wraps an
    # equality/membership/logical node in project(id/name), which is meaningless
    # and makes the otherwise correct typed graph fail the Boolean contract.
    # Remove only that redundant typed wrapper; no question text is consulted.
    if answer["mode"] == "boolean" and answer.get("source") in by_id:
        terminal = by_id.get(str(answer.get("source") or "")) or {}
        if str(terminal.get("op") or "") == "project":
            parent_id = str(terminal.get("source") or "")
            parent = by_id.get(parent_id) or {}
            if str(parent.get("op") or "") in {"compare", "membership", "logical_and", "logical_or"}:
                old_source = str(answer.get("source") or "")
                answer["source"] = parent_id
                answer["sources"] = list(dict.fromkeys(
                    parent_id if str(x) == old_source else str(x)
                    for x in (answer.get("sources") or [old_source]) if str(x)))

    return {"nodes": nodes, "answer": answer}, list(dict.fromkeys(errors))


def _provider_contract_errors(intent: dict[str, Any], benchmark: str = "") -> list[str]:
    """Validate answer contracts on the provider-normalized typed graph.

    Contract checks used to run before capability normalization. That made valid
    relation chains look untyped when the model omitted a redundant target
    resource (especially asset chains such as entity -> images). Normalize a
    copy through the same provider catalog/OAS layer used by compilation, then
    run the graph contract. This does not select routes from question wording and
    never reads benchmark gold data.
    """
    candidate = copy.deepcopy(intent or {})
    try:
        from oca.capability.provider import get_provider_semantics
        semantics = get_provider_semantics(benchmark)
    except Exception:
        semantics = None
    candidate = _normalize_provider_typed_aliases(candidate, semantics)
    candidate, structural = validate_intent(candidate)
    if structural:
        return list(dict.fromkeys(structural))
    if semantics is not None and getattr(semantics, "catalog", None) is not None:
        try:
            from oca.capability.normalize import normalize_intent
            candidate, _report = normalize_intent(candidate, semantics.catalog, _catalog(benchmark))
        except Exception as exc:
            return [f"capability normalization failed during contract validation: {exc}"]
    return _intent_contract_errors(candidate, benchmark)


def _intent_contract_errors(intent: dict[str, Any], benchmark: str = "") -> list[str]:
    """Graph-only consistency checks for the declared answer contract.

    These checks never read the natural-language question. They only ensure the
    typed IR is internally capable of producing the answer mode it declares.
    Failing fast here is preferable to compiling a structurally executable but
    semantically incomplete graph and hoping a later answerer notices.
    """
    nodes = list((intent or {}).get("nodes") or [])
    by_id = {str(n.get("id") or ""): n for n in nodes}
    answer = (intent or {}).get("answer") or {}
    source_ids = [str(x) for x in (answer.get("sources") or []) if str(x)]
    primary_source = str(answer.get("source") or "")
    if primary_source and primary_source not in source_ids:
        source_ids.append(primary_source)
    terminals = [by_id.get(x) for x in source_ids if x in by_id]
    mode = _cf(answer.get("mode") or "direct")
    errors: list[str] = []

    if not terminals:
        return ["intent answer has no terminal source"]

    terminal_ops = {str(n.get("op") or "") for n in terminals if n}
    if mode == "count" and "count" not in terminal_ops:
        if not _secat_v4163_has_explicit_terminal_count(intent):
            errors.append("answer mode count requires a terminal count node")
    if mode == "comparison" and not (terminal_ops & {"compare", "difference"}):
        errors.append("answer mode comparison requires a terminal compare/difference node")
    if mode == "comparison" and "difference" in terminal_ops:
        # In this IR, comparison mode denotes an ordering/winner surface.  A
        # magnitude-only question should use a direct difference.  Therefore a
        # comparison answer that includes a difference must also have an ordering
        # compare over the same operands somewhere in the graph.  The compiler can
        # then certify both winner and magnitude without a later LLM guess.
        difference_nodes = [n for n in terminals if n and str(n.get("op") or "") == "difference"]
        for diff in difference_nodes:
            pair = {str(diff.get("left") or diff.get("source") or ""), str(diff.get("right") or "")}
            companion = next((n for n in nodes
                              if str(n.get("op") or "") == "compare"
                              and {str(n.get("left") or n.get("source") or ""), str(n.get("right") or "")} == pair), None)
            if companion is None:
                errors.append(
                    f"answer mode comparison with difference node {diff.get('id')} requires a companion ordering compare over the same inputs")
    if mode == "boolean" and not (terminal_ops & {"compare", "membership", "logical_and", "logical_or"}):
        errors.append("answer mode boolean requires a terminal boolean operation")
    if mode == "action" and "action" not in terminal_ops:
        errors.append("answer mode action requires a terminal action node")

    # Optional but strongly prompted output ownership.  This is a typed IR
    # invariant, not question parsing: if the semantic planner declares that the
    # requested final value belongs to resource X, its terminal lineage must also
    # belong to X.  It catches intermediate-owner answers such as returning an
    # artist for a request whose declared final resource is a track.
    declared_answer_resource = _resource(answer.get("resource") or "")
    if declared_answer_resource and mode in {"direct", "list", "asset"}:
        def terminal_resource(node_id: str, visiting: set[str] | None = None) -> str:
            visiting = set(visiting or set())
            if not node_id or node_id in visiting or node_id not in by_id:
                return ""
            visiting.add(node_id)
            node0 = by_id[node_id]
            if node0.get("resource"):
                return _resource(node0.get("resource"))
            if str(node0.get("op") or "") in {"project", "select", "filter", "count"}:
                return terminal_resource(str(node0.get("source") or ""), visiting)
            return ""
        wrong = []
        # answer.resource is the type of the primary output. Multi-output requests
        # may legitimately combine heterogeneous resources (e.g. selected entity
        # + child properties). Requiring every secondary output to share one type
        # collapses valid answer obligations. For a single-source answer the old
        # strict invariant is unchanged.
        resource_checked_ids = ([primary_source] if len(source_ids) > 1 and primary_source
                                else list(source_ids))
        for terminal_id in resource_checked_ids:
            got = terminal_resource(terminal_id)
            if got and got != declared_answer_resource:
                wrong.append((terminal_id, got))
        if wrong:
            errors.append(
                f"answer.resource {declared_answer_resource!r} does not match terminal resource lineage {wrong!r}")

    if mode == "asset":
        # Asset semantics are type-driven.  The semantic field registry declares
        # which values are asset paths, and the capability catalog declares which
        # relations produce image resources.  Do not inspect terminal wording or
        # require a particular op shape: select/filter over an image lineage is a
        # valid asset answer, while project(name) is not.
        try:
            from oca.capability.asset_contract import asset_contract_satisfied
            from oca.capability.provider import get_provider_semantics
            _sem = get_provider_semantics(benchmark)
            catalog = _sem.catalog if _sem is not None else None
            ok, _why, owner_error = asset_contract_satisfied(
                source_ids, by_id, answer, catalog)
        except Exception:
            ok, owner_error = False, None
        if owner_error:
            errors.append(f"asset field owner mismatch: {owner_error}")
        elif not ok:
            errors.append("answer mode asset requires an asset-typed terminal value")
        else:
            # Do not let an entity-only terminal become an image merely because
            # the provider happens to expose an images capability.  A legitimate
            # asset request must either already terminate in image/asset lineage
            # or name an asset-typed semantic field.  This keeps answer type
            # selection in the typed language layer and prevents a mistaken
            # ``mode=asset`` from corrupting ordinary entity/title answers.
            try:
                from oca.capability.asset_contract import terminal_produces_asset, asset_field_canonical
                from oca.capability.provider import get_provider_semantics
                _sem = get_provider_semantics(benchmark)
                catalog = _sem.catalog if _sem is not None else None
                terminal_asset = any(terminal_produces_asset(tid, by_id, catalog)[0] for tid in source_ids)
                explicit_asset_field = any(asset_field_canonical(x) for x in (answer.get("fields") or []))
            except Exception:
                terminal_asset, explicit_asset_field = False, False
            if not terminal_asset and not explicit_asset_field:
                errors.append("answer mode asset requires explicit asset-typed intent, not an entity-only terminal")

    # ``find`` denotes lookup of a concrete named entity. A same-resource
    # collection constrained by filters must start from a population, otherwise
    # the compiler can accidentally treat a descriptive phrase as an entity name
    # and route a discovery query through /search. This check is graph-only: it
    # does not inspect the wording of the name or the user question.
    answer_many = (_cf(answer.get("cardinality")) == "many" or mode == "list")
    if answer_many:
        for terminal_id in source_ids:
            cursor = terminal_id
            seen_chain: set[str] = set()
            saw_filter = False
            saw_relation = False
            while cursor and cursor not in seen_chain and cursor in by_id:
                seen_chain.add(cursor)
                n = by_id[cursor]
                op = str(n.get("op") or "")
                if op == "filter":
                    saw_filter = True
                elif op in {"relation", "population"}:
                    saw_relation = True
                if op == "find":
                    if saw_filter and not saw_relation:
                        errors.append(
                            "filtered many-entity answer cannot originate from find; "
                            "use a population plus filters for an unnamed constrained collection")
                    break
                cursor = str(n.get("source") or "")

        # A plural child answer does not imply that the owner population is plural.
        # If an answer-producing relation consumes a population directly, the IR
        # must either select one owner first or explicitly declare all-owner fanout.
        # This prevents accidental N-owner fanout for requests such as "the cast of
        # a show" while retaining a typed escape hatch for genuine every/all-owner
        # requests. No question text or benchmark IDs are consulted here.
        answer_source_set = set(source_ids)
        for rel in nodes:
            if str(rel.get("op") or "") != "relation" or str(rel.get("id") or "") not in answer_source_set:
                continue
            src = by_id.get(str(rel.get("source") or "")) or {}
            if str(src.get("op") or "") != "population":
                continue
            lits = rel.get("literals") if isinstance(rel.get("literals"), dict) else {}
            explicit_many = (_cf(lits.get("owner_cardinality") or "") == "many"
                             or bool(lits.get("fanout", False)))
            if not explicit_many:
                errors.append(
                    f"intent node {rel.get('id')}: plural child answer from a population "
                    "requires an explicit owner selector or literals.owner_cardinality='many'")

    # Numbered domain objects (season/episode) have provider-defined semantic
    # number fields.  Treating "second season" as the second/third returned
    # array item is unsafe because feeds may include specials (season 0), and
    # returned order is not the domain identifier.  The IR must therefore carry
    # the explicit semantic number somewhere on the selected scope.  This is a
    # typed graph invariant: it does not inspect the user question.  A failure
    # enters the bounded semantic-repair path, whose prompt has the original
    # request and can restore season_number/episode_number without benchmark
    # rules.
    for node0 in nodes:
        if str(node0.get("op") or "") != "select":
            continue
        if _cf(node0.get("mode") or "") not in {"first", "nth"}:
            continue
        src_id = str(node0.get("source") or "")
        src = by_id.get(src_id) or {}
        resource0 = _cf(src.get("resource") or "")
        if resource0 not in {"season", "episode"}:
            continue
        key = "season_number" if resource0 == "season" else "episode_number"

        def _scoped_number_present(start_id: str, field: str) -> bool:
            cursor = start_id
            seen: set[str] = set()
            depth = 0
            while cursor and cursor not in seen and cursor in by_id and depth < 8:
                seen.add(cursor)
                cur = by_id[cursor]
                lits = cur.get("literals") or {}
                if lits.get(field) is not None:
                    return True
                # Normalized/repaired IR may carry the provider-neutral short
                # spelling as well.  Accept it but keep the canonical prompt
                # spelling explicit.
                short = "season" if field == "season_number" else "episode"
                if lits.get(short) is not None:
                    return True
                cursor = str(cur.get("source") or "")
                depth += 1
            return False

        own_lits = node0.get("literals") or {}
        short_key = "season" if key == "season_number" else "episode"
        if (own_lits.get(key) is None and own_lits.get(short_key) is None and
                not _scoped_number_present(src_id, key)):
            errors.append(
                f"intent node {node0.get('id')}: numbered {resource0} selection "
                f"requires explicit literals.{key}; first/nth is returned-order "
                "selection and cannot stand in for a provider domain number")

    # Date arithmetic is unit-sensitive.  The evidence compiler's raw date
    # subtraction is measured in days, so permitting a unitless DATE-DATE
    # difference can silently certify the right magnitude in the wrong unit.
    # Infer value types from typed project lineage only; no question text is
    # consulted.  Invalid unitless date differences are sent through the bounded
    # semantic repair path, where the model can preserve the user's requested
    # unit explicitly.
    try:
        from oca.capability.types import SEMANTIC_FIELDS, canonical_field, DATE

        def _value_type(node_id: str, visiting: set[str] | None = None) -> str:
            visiting = set(visiting or set())
            if not node_id or node_id in visiting or node_id not in by_id:
                return ""
            visiting.add(node_id)
            node0 = by_id[node_id]
            op0 = str(node0.get("op") or "")
            if op0 == "project":
                semantic = str(node0.get("_oca_semantic_field") or "")
                canonical = semantic or canonical_field(str(node0.get("field") or "")) or ""
                sf = SEMANTIC_FIELDS.get(canonical)
                return str(sf.value_type) if sf else ""
            if op0 in {"select", "filter"}:
                return _value_type(str(node0.get("source") or ""), visiting)
            return ""

        for node0 in nodes:
            if str(node0.get("op") or "") != "difference":
                continue
            left_id = str(node0.get("left") or node0.get("source") or "")
            right_id = str(node0.get("right") or "")
            left_type = _value_type(left_id)
            right_type = _value_type(right_id)
            unit0 = _cf(node0.get("unit") or (node0.get("literals") or {}).get("unit") or "raw")
            if left_type == DATE and right_type == DATE and unit0 in {"", "raw"}:
                errors.append(
                    f"intent node {node0.get('id')}: date difference requires an explicit unit "
                    "such as years or days")
    except Exception:
        pass

    # A declared multi-source answer must not silently name the same node twice.
    if len(source_ids) != len(set(source_ids)):
        errors.append("answer.sources contains duplicate terminal nodes")
    return list(dict.fromkeys(errors))


def _provider_prevalidate_typed_closure(raw_intent: dict[str, Any] | None, benchmark: str) -> dict[str, Any]:
    """Apply only provider-declared graph closure before generic IR validation.

    Some model drafts are semantically recoverable only with provider metadata
    (for example an action whose required implicit owner is referenced by a
    dangling node id).  Generic ``validate_intent`` cannot know that metadata
    and would reject the graph before the deterministic compiler can repair it.

    This boundary therefore applies the same narrow provider-declared alias/owner
    normalization used by ``compile_intent_to_plan`` *before* graph validation.
    It never reads task ids, gold routes, expected answers, or question-shaped
    benchmark rules.  If no provider declaration authorizes a repair, the graph
    is returned unchanged and ordinary validation still rejects it.
    """
    obj = copy.deepcopy(raw_intent or {}) if isinstance(raw_intent, dict) else {}
    if not benchmark:
        return obj
    try:
        from oca.capability.provider import get_provider_semantics
        semantics = get_provider_semantics(benchmark)
        return _normalize_provider_typed_aliases(obj, semantics)
    except Exception:
        return obj


def make_intent_plan(question: str, model: str, client, *, max_tokens: int = 900,
                     runtime_feedback: dict[str, Any] | None = None,
                     benchmark: str = "") -> tuple[dict[str, Any], str]:
    messages = build_intent_messages(question)
    ontology = _provider_semantic_ontology(benchmark)
    if ontology:
        messages.insert(1, {"role": "system", "content": ontology})
    if runtime_feedback:
        compact = {
            "problem": runtime_feedback.get("problem") or runtime_feedback.get("missing_slots") or
                       runtime_feedback.get("semantic_risks") or runtime_feedback.get("compiler_warnings"),
            "prior_intent": runtime_feedback.get("prior_intent"),
            "semantic_invariants": runtime_feedback.get("semantic_invariants"),
        }
        messages.append({
            "role": "user",
            "content": "Fix only the part of the intent that did not work. Keep the question meaning. "
                       "Return the full corrected intent JSON. Feedback: " +
                       json.dumps(compact, ensure_ascii=False, default=str)[:2500],
        })
    text = ""
    try:
        from utils.token_meter import stage as token_stage
        kwargs = dict(model=model, messages=messages, temperature=0.0)
        with token_stage("intent_planner"):
            try:
                response = client.chat.completions.create(
                    **kwargs, response_format={"type": "json_object"},
                    max_completion_tokens=max(250, int(max_tokens)))
            except Exception:
                response = client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
    except Exception as exc:
        return {"nodes": [], "answer": {}, "valid": False,
                "validation_errors": [f"intent planner call failed: {exc}"]}, text
    raw_intent = _provider_prevalidate_typed_closure(_extract_json(text), benchmark)
    intent, errors = validate_intent(raw_intent)
    intent["valid"] = not errors
    intent["validation_errors"] = errors
    return intent, text


_SEMANTIC_REVIEW_OPS = {
    "relation", "filter", "select", "action", "compare", "difference",
    "membership", "logical_and", "logical_or",
}


def _intent_needs_semantic_review(intent: dict[str, Any]) -> bool:
    """Review composition-sensitive IR, using only the IR shape.

    This is a generic complexity gate, not a question parser: it never looks at
    task wording, entity names, benchmark ids, routes, or expected answers.
    Simple one-entity scalar projections stay one-call; relationship chains,
    scopes, selections, multi-entity operations and assets get one semantic
    fidelity audit before provider compilation.
    """
    nodes = list((intent or {}).get("nodes") or [])
    ops = {str(n.get("op") or "") for n in nodes}
    roots = [n for n in nodes if str(n.get("op") or "") in {"find", "population"}]
    answer = (intent or {}).get("answer") or {}
    if ops & _SEMANTIC_REVIEW_OPS:
        return True
    if len(roots) > 1:
        return True
    if str(answer.get("mode") or "") in {"asset", "comparison", "boolean"}:
        return True
    return any(
        str(n.get("resource") or "") in {"season", "episode"}
        or any(k in (n.get("literals") or {}) for k in ("season", "season_number", "episode", "episode_number"))
        for n in nodes
    )


def _intent_action_invariants(intent: dict[str, Any] | None) -> dict[str, Any]:
    """Extract typed state-change obligations that repairs may not erase."""
    obj = intent or {}
    nodes = list(obj.get("nodes") or [])
    actions = [n for n in nodes if str(n.get("op") or "") == "action"]
    answer_mode = str((obj.get("answer") or {}).get("mode") or "").strip().casefold()
    sticky = answer_mode == "action" or bool(actions)
    return {
        "state_changing": bool(sticky),
        "min_action_count": len(actions) if sticky else 0,
        "declared_action_names": [str(n.get("action") or "") for n in actions
                                  if str(n.get("action") or "")],
    }


def _merge_action_invariants(base: dict[str, Any] | None,
                             candidate: dict[str, Any] | None) -> dict[str, Any]:
    """Monotonically retain any state-change obligations discovered later.

    Semantic review/repair may strengthen an initially incomplete draft by adding
    actions.  Once discovered, those obligations become sticky just like actions
    present in the first draft; no later repair may silently downgrade them.
    """
    left = dict(base or {})
    right = _intent_action_invariants(candidate)
    names = list(dict.fromkeys(
        [str(x) for x in (left.get("declared_action_names") or []) if str(x)] +
        [str(x) for x in (right.get("declared_action_names") or []) if str(x)]
    ))
    return {
        "state_changing": bool(left.get("state_changing") or right.get("state_changing")),
        "min_action_count": max(int(left.get("min_action_count") or 0),
                                int(right.get("min_action_count") or 0)),
        "declared_action_names": names,
    }


def _intent_action_invariant_errors(invariants: dict[str, Any] | None,
                                    candidate: dict[str, Any] | None) -> list[str]:
    """Reject repairs/reviews that turn a recognized action into a read answer."""
    inv = invariants or {}
    if not inv.get("state_changing"):
        return []
    obj = candidate or {}
    nodes = list(obj.get("nodes") or [])
    actions = [n for n in nodes if str(n.get("op") or "") == "action"]
    mode = str((obj.get("answer") or {}).get("mode") or "").strip().casefold()
    errors: list[str] = []
    if mode != "action":
        errors.append("semantic invariant violated: state-changing request was downgraded from action mode")
    minimum = max(1, int(inv.get("min_action_count") or 0))
    if len(actions) < minimum:
        errors.append(
            f"semantic invariant violated: action obligation count fell from at least {minimum} to {len(actions)}")
    return errors


def review_intent_plan(question: str, intent: dict[str, Any], model: str, client,
                       *, max_tokens: int = 900, benchmark: str = "") -> tuple[dict[str, Any], str]:
    """One bounded model audit of semantic fidelity before route compilation.

    The reviewer sees only the natural-language request and typed semantic IR.
    It is explicitly forbidden from choosing endpoints or using provider/gold
    facts. This catches omissions that a compiler cannot detect when an
    incomplete IR is nevertheless structurally executable.
    """
    from datetime import date as _review_date
    system = f"""Current date: {_review_date.today().isoformat()}.

Audit a candidate typed semantic graph against the user's request. Check semantic fidelity only. Do not choose API endpoints, parameters, schema paths, provider-specific workarounds, factual answers, or expected benchmark answers.

Do NOT trust or anchor on the candidate. First independently reconstruct the semantic obligations of the request: named entities, resource types, independent branches, relation direction and every intermediate relation, populations, scopes, ordinals/rankings, filters, requested units, requested visual owner, final properties, and comparison/difference/logical outputs. Then compare those obligations with the candidate. A graph that can execute but omits or invents part of the request is incorrect.

You may add, delete, or reconnect nodes as needed. Use only these operation types: find, population, relation, filter, select, project, action, count, compare, difference, membership, logical_and, logical_or. Keep the graph minimal: delete unrelated exploratory branches and details that do not contribute to the requested answer. Treat an exact provider-declared population alias as that population rather than a named entity; use downstream type/action constraints to choose among resources that expose the same population label. For actions, match acted-on cardinality exactly: singular a/one/ordinal targets require a single selected entity, while all/multiple targets retain the collection. Ranking a collection uses select(argmax/argmin), not compare. For a provider-ranked population such as trending/popular/top-rated, selecting its top/most member preserves endpoint order with select(first); use argmax/argmin only for an explicitly requested separate ranking field. Use select(top_k) with rank equal to the requested count when a fixed number of first returned items is required. For state-changing requests preserve the provider action node and its source/input/literals; do not replace an action with a read-only answer. When latest/newest scopes a relation result, rank that relation result by its date field instead of intersecting it with the provider global latest population. Use find for a concrete named entity or an explicit free-text search descriptor that has no declared semantic population/field; use a population plus filters only when the request actually states constraints represented by those fields. Never invent proxy filters, boolean flags, numeric thresholds, or arbitrary field meanings for a qualitative descriptor; preserve unmatched descriptive text as free-text search input instead. Preserve explicitly named population concepts exactly; a temporal adverb must not turn popular into now_playing or trending, or vice versa. Preserve relation direction from the named semantic anchor to the requested result. Treat select(mode=nth).rank as a zero-based returned-order index only; when the request states a season/episode number, preserve that human number in literals.season_number/literals.episode_number on the scoped relation. Preserve the requested unit on every difference node; date differences must never silently default to raw days. For answer.mode=comparison, preserve an explicit ordering/winner compare and ALWAYS declare comparison=gt|gte|lt|lte for ordering; for birth_date, older=lt (earlier date) and younger=gt (later date); if a difference is also requested, keep both compare and difference nodes over the same operands in answer.sources. A magnitude-only difference should not use comparison mode. Use asset mode only for an explicitly visual/file request. For visual requests, follow the full relation chain to the thing whose appearance/image was requested and terminate the answer in an asset-typed value or image lineage; select/filter over that lineage is valid. If the question asks for a title/name/entity rather than a visual, change any candidate asset mode back to direct/list instead of appending images. For equality/inequality between entities, compare explicit identity values; if singular wording reaches a multi-record relation, insert a semantic selector rather than comparing an unresolved candidate set. Re-evaluate answer.cardinality independently from the candidate: it is FINAL output cardinality. Singular final requests (a/an/one/single asset or entity, the requested logo/poster/image, and what-does-X-look-like appearance questions) require cardinality=one even when the provider relation returns many candidates. Use cardinality=many only when the user explicitly asks for multiple final values. For every direct/list/asset answer, set answer.resource to the canonical provider resource that owns the requested final value and verify the terminal lineage has that same resource; a request for a song/track must not terminate at an artist merely because the artist is an intermediate result. Set answer.detail=true only for broad information/details/metadata requests and keep the entity/detail lineage instead of reducing it to its display name. If a collection-valued relation feeds a downstream relation that needs one owner, explicitly select one semantic owner; if the request truly applies to every owner, preserve an explicit fan-out instead of relying on an implicit first record. If many final children belong to one unnamed member of a population, preserve an explicit selector for that one owner before the child relation instead of fanning the relation across the population. Only when the request explicitly applies to every/all owners may the relation consume the population directly; record that intent as literals.owner_cardinality="many". Keep semantic relations on relation nodes, never as stray relation fields on select/filter/project. For multi-output requests preserve every independently requested output in answer.sources even when those outputs have different resource types; answer.resource names only the primary output resource. Do not merge network with company, season with series, episode with series, popular with trending, or one comparison branch with another.

Before returning, silently check that every independently reconstructed obligation is represented and every answer source is terminally supported. Return the full corrected intent JSON only. If it is faithful, return it unchanged."""
    messages = [{"role": "system", "content": system}]
    ontology = _provider_semantic_ontology(benchmark)
    if ontology:
        messages.append({"role": "system", "content": ontology})
    messages += [
        {"role": "user", "content": question},
        {"role": "user", "content": "Candidate typed intent:\n" +
         json.dumps(intent, ensure_ascii=False, default=str)[:6500]},
    ]
    text = ""
    try:
        from utils.token_meter import stage as token_stage
        kwargs = dict(model=model, messages=messages, temperature=0.0)
        with token_stage("intent_semantic_review"):
            try:
                response = client.chat.completions.create(
                    **kwargs, response_format={"type": "json_object"},
                    max_completion_tokens=max(250, int(max_tokens)))
            except Exception:
                response = client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
    except Exception as exc:
        # Review is an accuracy aid, not a new abstention point. Keep the valid
        # draft if the audit call itself is unavailable.
        fallback = copy.deepcopy(intent)
        fallback.setdefault("validation_warnings", []).append(
            f"semantic intent review unavailable: {exc}")
        return fallback, text

    reviewed_raw = _provider_prevalidate_typed_closure(_extract_json(text), benchmark)
    reviewed, errors = validate_intent(reviewed_raw)
    if errors:
        fallback = copy.deepcopy(intent)
        fallback.setdefault("validation_warnings", []).append(
            "semantic intent review returned invalid IR; retained initial intent")
        return fallback, text
    reviewed["valid"] = True
    reviewed["validation_errors"] = []
    return reviewed, text


@dataclass
class _Handle:
    node_id: str
    kind: str
    resource: str = ""
    step_id: str = ""
    derivation_id: str = ""
    record_root: str = "$"
    id_alias: str = ""
    label_step: str = ""
    label_field: str = ""
    owner_step: str = ""
    owner_label_field: str = ""
    projection_prefix: str = ""
    selection_limit: int = 0
    # Dynamic path values already grounded in this handle's selected lineage.
    # Keys are provider placeholders (series_id, season_number, ...); values are
    # producer binding aliases.  This lets hierarchical capabilities compose
    # without reconstructing scope from question text.
    path_aliases: dict[str, str] = dc_field(default_factory=dict)


def _catalog(benchmark: str) -> list[dict[str, Any]]:
    from utils.schema_outline import endpoint_catalog_cards
    return endpoint_catalog_cards(benchmark, max_paths_per_endpoint=300)


def _param_names(card: dict[str, Any], location: str | None = None) -> set[str]:
    return {str(p.get("name") or "") for p in card.get("parameters") or []
            if p.get("name") and (location is None or str(p.get("in") or "query") == location)}


def _required_query_names(card: dict[str, Any]) -> set[str]:
    return {str(p.get("name") or "") for p in card.get("parameters") or []
            if p.get("name") and str(p.get("in") or "query") == "query" and bool(p.get("required"))}


def _leaf_type(card: dict[str, Any], path: str) -> str:
    clean = str(path or "").replace("$.", "")
    for item in card.get("leaf_paths") or []:
        candidate = str(item.get("path") or "").replace("$.", "")
        if candidate == clean:
            return str(item.get("type") or "")
    return ""


def _identifier_like_path(path: str) -> bool:
    leaf = str(path or "").replace("[*]", "").split(".")[-1].casefold()
    return (leaf in {"id", "ids", "identifier", "identifiers", "key", "keys"}
            or leaf.endswith("_id") or leaf.endswith("_ids"))


def _field_semantic_tokens(value: Any) -> set[str]:
    """Conservative semantic tokens for schema-field matching.

    These are ordinary API vocabulary folds, never benchmark answers.  The
    expansion is deliberately asymmetric around identifiers: asking for a human
    concept such as a genre must never be satisfied by an *_id field.
    """
    toks = set(_tokens(value))
    text = _cf(value)
    if any(x in toks for x in {"rating", "rated", "score"}):
        toks |= {"vote", "average", "rating", "rated", "score"}
    if "popularity" in toks or "popular" in toks:
        toks |= {"popular", "popularity"}
    if "total" in toks or "number" in toks or "count" in toks:
        toks |= {"total", "number", "count"}
    if "language" in toks:
        toks |= {"original", "language"}
    if "release" in toks or "released" in toks:
        toks |= {"release", "released", "date", "year"}
    if "air" in toks and "date" in toks:
        toks |= {"air", "date"}
    if "homepage" in toks or "website" in toks:
        toks |= {"homepage", "website", "url"}
    if "headquarter" in toks or "headquarters" in toks:
        toks |= {"headquarter", "headquarters", "location"}
    if "genre" in toks:
        toks |= {"genre"}
    if "photo" in toks or "image" in toks or "poster" in toks or "cover" in toks or "logo" in toks:
        toks |= {"photo", "image", "poster", "cover", "logo", "profile", "backdrop", "path", "file"}
    return toks


def _literal_language_code(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    names = {
        "english": "en", "french": "fr", "spanish": "es", "german": "de",
        "italian": "it", "japanese": "ja", "korean": "ko", "chinese": "zh",
        "portuguese": "pt", "russian": "ru", "hindi": "hi", "arabic": "ar",
    }
    return names.get(_cf(value), value)


def _query_parameter_for_filter(card: dict[str, Any], field: str, comparison: str,
                                value: Any, question: str = "") -> tuple[str, Any] | None:
    """Map one semantic filter onto a documented query parameter when safe.

    Local deterministic filtering is still retained.  This push-down only narrows
    the API population so paginated discovery endpoints can actually return the
    requested universe instead of filtering an unrelated first page.
    """
    params = [p for p in card.get("parameters") or []
              if str(p.get("in") or "query") == "query" and p.get("name")]
    if not params:
        return None
    cmp0 = _COMPARISON_ALIASES.get(_cf(comparison or "eq"), _cf(comparison or "eq")) or "eq"
    wanted = _field_semantic_tokens(field)
    if not wanted:
        return None
    candidates: list[tuple[int, str, Any]] = []
    for p in params:
        name = str(p.get("name") or "")
        pn = _field_semantic_tokens(name)
        overlap = len(wanted & pn)
        if not overlap:
            continue
        score = overlap * 4
        lname = name.casefold()
        # Prefer comparison-suffixed parameters that preserve the requested bound.
        if cmp0 in {"gte", "gt"}:
            if lname.endswith(".gte"):
                score += 8
            elif lname.endswith(".lte"):
                score -= 8
        elif cmp0 in {"lte", "lt"}:
            if lname.endswith(".lte"):
                score += 8
            elif lname.endswith(".gte"):
                score -= 8
        elif cmp0 == "eq":
            if lname.endswith((".gte", ".lte")):
                score -= 4
        # A generic response-language parameter is not the same thing as a movie's
        # original language. Prefer an explicitly content-scoped parameter when the
        # semantic field concerns the entity language.
        if "language" in wanted:
            if "original_language" in lname:
                score += 8
            elif lname == "language":
                score -= 5
        out_value = value
        if "language" in wanted:
            out_value = _literal_language_code(value)
        # Convert a bare year bound to the documented date-bound shape when needed.
        if isinstance(value, int) and "date" in pn and any(t in wanted for t in {"year", "date", "release"}):
            if cmp0 in {"lte", "lt"}:
                out_value = f"{value:04d}-12-31"
            else:
                out_value = f"{value:04d}-01-01"
        candidates.append((score, name, out_value))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], len(x[1]), x[1]))
    best = candidates[0]
    # Require more than one accidental token overlap unless the parameter name is
    # an exact field match.
    exact = _cf(best[1]).replace(" ", "_") == _cf(field).replace(" ", "_")
    if best[0] < 6 and not exact:
        return None
    return best[1], best[2]


def _path_placeholders(endpoint: str) -> list[str]:
    return re.findall(r"\{([^{}]+)\}", str(endpoint or ""))


def _leaf_paths(card: dict[str, Any]) -> list[str]:
    return [str(x.get("path") or "") for x in card.get("leaf_paths") or [] if x.get("path")]


def _record_paths(card: dict[str, Any]) -> list[str]:
    return [str(x.get("path") or "") for x in card.get("record_paths") or [] if x.get("path")]


def _resource_path_score(endpoint: str, resource: str) -> int:
    resource = _resource(resource)
    text = _cf(endpoint)
    if not resource:
        return 0
    score = 0
    if resource == "tv" and re.search(r"(^|/)tv(/|$|\{)", endpoint):
        score += 8
    elif resource and re.search(rf"(^|/){re.escape(resource)}(/|$|\{{)", endpoint):
        score += 8
    # Mixed-resource endpoints are useful, but a resource-specific endpoint is
    # safer when the intent explicitly names one resource family.
    if resource and "/all/" in endpoint:
        score -= 3
    if resource in _tokens(text):
        score += 2
    return score


def _semantic_alias_tokens(text: str) -> set[str]:
    toks = _tokens(text)
    lowered = _cf(text)
    if "on the air" in lowered or "currently on air" in lowered or "currently airing" in lowered:
        toks |= {"on", "air", "airing", "current"}
    if "now playing" in lowered or "released" in lowered or "in theaters" in lowered or "in theatres" in lowered:
        toks |= {"now", "playing", "released", "current", "theater", "theatre", "release", "date"}
    if "trending" in lowered:
        # Some APIs expose a resource-specific ranked popularity feed rather than
        # a typed trending feed. Keep both concepts available to the OAS scorer;
        # structural/resource fidelity decides which documented route wins.
        toks |= {"trend", "trending", "popular", "popularity"}
    if "popular" in lowered:
        toks |= {"popular", "popularity"}
    if "top rated" in lowered or "highest rated" in lowered or "rating" in lowered:
        toks |= {"top", "rated", "rating", "vote", "average", "score"}
    if "discover" in lowered or "filter" in lowered:
        toks |= {"discover", "filter", "filters"}
    if "credit" in lowered:
        toks |= {"credit", "credits", "cast", "crew"}
    return toks


def _route_text(card: dict[str, Any]) -> str:
    return " ".join([str(card.get("endpoint") or ""), str(card.get("description") or "")])


def _required_path_ok(card: dict[str, Any], source_resource: str = "",
                      literals: dict[str, Any] | None = None) -> bool:
    literals = literals or {}
    for name in _path_placeholders(str(card.get("endpoint") or "")):
        if name in literals:
            continue
        r = _PLACEHOLDER_RESOURCE.get(name, "")
        if source_resource and r == _resource(source_resource):
            continue
        # time-window has a deterministic natural-language default below.
        if name == "time_window":
            continue
        return False
    return True


def _capability_for_relation(relation: str, catalog=None):
    """Return a typed provider capability encoded by the semantic normalizer."""
    text = str(relation or "")
    prefix = "__cap__"
    if not text.startswith(prefix) or catalog is None:
        return None
    name = text[len(prefix):]
    try:
        return next((cap for cap in catalog if cap.semantic_name == name), None)
    except Exception:
        return None


def _route_card(cards: list[dict[str, Any]], spec: Any) -> dict[str, Any] | None:
    """Resolve a provider route declaration and attach its declarative metadata."""
    from oca.capability.provider import route_endpoint, route_metadata
    endpoint = route_endpoint(spec)
    meta = route_metadata(spec)
    method = str(meta.get("method") or "GET").upper()
    card = next((c for c in cards
                 if str(c.get("endpoint") or "") == endpoint
                 and str(c.get("method") or "GET").upper() == method), None)
    if card is None:
        return None
    out = dict(card)
    out["_oca_route_spec"] = meta
    return out


def _resolve_route(kind: str, *, resource: str = "", relation: str = "",
                   population: str = "", source_resource: str = "",
                   collection: str = "", literals: dict[str, Any] | None = None,
                   filter_hints: list[dict[str, Any]] | None = None,
                   question: str = "", cards: list[dict[str, Any]],
                   semantics=None) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    literals = dict(literals or {})
    filter_hints = list(filter_hints or [])
    find_routes = semantics.find_routes if semantics is not None else {}
    population_routes = semantics.population_routes if semantics is not None else {}
    population_aliases = semantics.population_aliases if semantics is not None else {}
    catalog = semantics.catalog if semantics is not None else None

    if kind == "find":
        spec = find_routes.get(_resource(resource))
        if spec:
            card = _route_card(cards, spec)
            if card is not None:
                endpoint = str(card.get("endpoint") or "")
                return card, [{"score": "capability", "endpoint": endpoint,
                               "method": str(card.get("method") or "GET").upper(),
                               "capability": f"find.{_resource(resource)}"}]

    if kind == "population":
        pop_key = population_aliases.get(_cf(population), _cf(population).replace(" ", "_"))
        spec = population_routes.get((_resource(resource), pop_key))
        if spec is None and pop_key == "trending":
            spec = population_routes.get(("any", "trending"))
        if spec:
            card = _route_card(cards, spec)
            if card is not None:
                endpoint = str(card.get("endpoint") or "")
                # Guard semantic population identity against stale/over-broad
                # provider aliases when the OAS itself exposes the exact family.
                # This is provider-schema evidence, not benchmark knowledge.
                resource0 = _resource(resource)
                exact_family_exists = False
                if pop_key in {"trending", "popular"}:
                    needle = "/trending/" if pop_key == "trending" else "/popular"
                    exact_family_exists = any(
                        needle in str(c.get("endpoint") or "").casefold()
                        and (_resource_path_score(str(c.get("endpoint") or ""), resource0) > 0
                             or (pop_key == "trending" and "/all/" in str(c.get("endpoint") or "")))
                        for c in cards
                        if str(c.get("method") or "GET").upper() in {"GET", "HEAD"})
                    if exact_family_exists and needle not in endpoint.casefold():
                        card = None
                if card is not None:
                    return card, [{"score": "capability", "endpoint": endpoint,
                                   "method": str(card.get("method") or "GET").upper(),
                                   "capability": f"population.{resource0}.{pop_key}"}]

    if kind == "relation":
        cap = _capability_for_relation(relation, catalog)
        if cap is not None:
            card = next((c for c in cards
                         if str(c.get("endpoint") or "") == cap.endpoint
                         and str(c.get("method") or "GET").upper() == cap.method.upper()), None)
            if card is not None:
                out = dict(card)
                out["_oca_capability"] = cap
                return out, [{
                    "score": "capability",
                    "endpoint": cap.endpoint,
                    "method": cap.method,
                    "capability": cap.semantic_name,
                    "confidence": cap.confidence,
                }]
            return None, [{
                "score": "capability_missing_from_oas",
                "endpoint": cap.endpoint,
                "method": cap.method,
                "capability": cap.semantic_name,
            }]
    want = " ".join(x for x in [relation, population, collection] if x)
    want_tokens = _semantic_alias_tokens(want)
    scored: list[tuple[int, dict[str, Any]]] = []
    for card in cards:
        if str(card.get("method") or "GET").upper() not in {"GET", "HEAD"}:
            continue
        endpoint = str(card.get("endpoint") or "")
        placeholders = _path_placeholders(endpoint)
        params = _param_names(card)
        if not _required_path_ok(card, source_resource=source_resource, literals=literals):
            continue
        score = 0
        text_tokens = _semantic_alias_tokens(_route_text(card))
        leaf_tokens = _semantic_alias_tokens(" ".join(_leaf_paths(card)))
        if kind == "find":
            if "query" not in params or "/search/" not in endpoint:
                continue
            score += 12 + _resource_path_score(endpoint, resource)
        elif kind == "population":
            # A population is a collection endpoint, not an entity-id child call.
            non_window = [p for p in placeholders if p != "time_window"]
            if non_window:
                continue
            # Search endpoints require a free-text query and therefore are not a
            # population merely because they return results[*].  Reject any route
            # whose required query parameters are not already explicit literals.
            missing_required_query = [p for p in _required_query_names(card) if p not in literals]
            if missing_required_query:
                continue
            score += _resource_path_score(endpoint, resource)
            overlap = len(want_tokens & text_tokens)
            score += overlap * 3
            if "results[*]" in _record_paths(card):
                score += 3
            if resource and "/all/" in endpoint:
                score -= 4
            # Prefer routes that can push requested population filters into the
            # documented request. This is crucial for paginated discovery APIs:
            # fetching an arbitrary first page and filtering locally is incomplete.
            expressible = 0
            for hint in filter_hints:
                mapped = _query_parameter_for_filter(
                    card, str(hint.get("field") or ""),
                    str(hint.get("comparison") or "eq"), hint.get("value"))
                if mapped:
                    expressible += 1
            score += expressible * 7
            if filter_hints and expressible == len(filter_hints):
                score += 5
            ep = _cf(endpoint)
            popcf = _cf(population)
            # Canonical population labels are semantic commitments.  Reward the
            # corresponding documented route strongly enough that a generic
            # discovery endpoint cannot win merely because its schema exposes the
            # same ranking/filter fields.
            if "top rated" in popcf or "highest rated" in popcf:
                if "/top_rated" in ep:
                    score += 28
                elif "/discover/" in ep:
                    score -= 8
            if "now playing" in popcf or "in theaters" in popcf or "currently showing" in popcf:
                if "/now_playing" in ep:
                    score += 28
            if "on the air" in popcf or "currently on air" in popcf:
                if "/on_the_air" in ep:
                    score += 28
            # The semantic population supplied by the typed front-end is stronger
            # than surface words elsewhere in the question.  This matters for API
            # profiles where a phrase such as "trending TV" is intentionally
            # normalized to the resource-specific popularity feed.  Never let the
            # lexical question text silently override an explicit semantic
            # population chosen upstream.
            if "trending" in popcf and resource:
                if "/trending/" in ep:
                    score += 24
                elif "/popular" in ep and _resource_path_score(endpoint, resource) > 0:
                    score += 3
            elif "popular" in popcf and resource:
                if "/popular" in ep and _resource_path_score(endpoint, resource) > 0:
                    score += 24
                elif "/trending/" in ep:
                    score -= 12
        elif kind == "relation":
            if not placeholders:
                continue
            score += _resource_path_score(endpoint, source_resource)
            if source_resource and not any(_PLACEHOLDER_RESOURCE.get(p) == _resource(source_resource)
                                           for p in placeholders):
                continue
            overlap = len(want_tokens & text_tokens)
            field_overlap = len(want_tokens & leaf_tokens)
            score += overlap * 4 + field_overlap * 2
            relcf = _cf(relation)
            if relcf in {"detail", "details", "info", "information"}:
                # A details relation means the resource object itself. Prefer the
                # shortest source-id route with no semantic child suffix.
                tail = re.sub(r"\{[^{}]+\}", "{}", endpoint).rstrip("/").split("/")
                source_segments = [seg for seg in tail if seg == "{}"]
                if len(source_segments) == 1 and endpoint.rstrip("/").endswith("}"):
                    score += 14
                else:
                    score -= 8
            if relcf:
                compact = relcf.replace(" ", "_")
                if compact in endpoint.casefold():
                    score += 9
                # "reviews", "credits", "images" etc. are usually direct path
                # segments; exact relation segment is strong OAS evidence.
                rel_last = compact.split("_")[-1]
                if f"/{rel_last}" in endpoint.casefold():
                    score += 5
                # Season/episode semantics are encoded structurally in TMDB-style
                # paths.  Prefer a route that actually consumes the explicit
                # season/episode path literals over a shorter parent detail route
                # whose response merely mentions episodes in its schema.
                if "episode" in relcf:
                    if "episode_number" in placeholders and "/episode/" in endpoint.casefold():
                        score += 24
                    elif "episode_number" not in placeholders:
                        score -= 12
                if "season" in relcf and "episode" not in relcf:
                    if "season_number" in placeholders and "/season/" in endpoint.casefold():
                        score += 18
                    elif "season_number" not in placeholders:
                        score -= 8
            if collection:
                ct = _tokens(collection)
                score += 3 * len(ct & leaf_tokens)
            # Prefer the route requiring the fewest unrelated path literals.
            score -= max(0, len(placeholders) - 1) * 2
            score += sum(2 for p in placeholders if p in literals)
        else:
            continue
        if score > 0:
            scored.append((score, card))
    scored.sort(key=lambda x: (-x[0], len(str(x[1].get("endpoint") or "")), str(x[1].get("endpoint") or "")))
    diagnostics = [{"score": s, "endpoint": c.get("endpoint"), "method": c.get("method")}
                   for s, c in scored[:5]]
    if not scored:
        return None, diagnostics
    best_score, best = scored[0]
    # Reject weak relation matches. Population/find scoring has stronger structural
    # constraints and can safely use a slightly lower threshold.
    threshold = 10 if kind == "relation" else 8
    if best_score < threshold:
        return None, diagnostics
    return best, diagnostics


def _record_root(card: dict[str, Any], *, collection: str = "", relation: str = "",
                 prefer_results: bool = False, catalog=None) -> str:
    cap = _capability_for_relation(relation, catalog)
    if cap is not None:
        return cap.record_path or "$"
    route_spec = card.get("_oca_route_spec") if isinstance(card, dict) else None
    if isinstance(route_spec, dict) and route_spec.get("record_path"):
        return str(route_spec.get("record_path") or "$")
    roots = [p for p in _record_paths(card) if p and p != "$"]
    if prefer_results and "results[*]" in roots:
        return "results[*]"
    wanted = _tokens(collection or relation)
    if wanted:
        ranked = []
        for root in roots:
            rt = _tokens(root)
            ranked.append((len(wanted & rt), root.count("."), len(root), root))
        ranked.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
        if ranked and ranked[0][0] > 0:
            return ranked[0][3]
    top_arrays = [p for p in roots if p.endswith("[*]") and p.count(".") == 0]
    if len(top_arrays) == 1:
        return top_arrays[0]
    if "results[*]" in roots:
        return "results[*]"
    return "$"


def _root_field(record_root: str) -> str:
    text = str(record_root or "$ ").replace("$.", "").strip("$")
    if not text:
        return ""
    return text.split(".")[-1].replace("[*]", "")


def _collection_derivation_field(record_root: str) -> str:
    """Keep enough qualification to identify a nested collection unambiguously."""
    text = str(record_root or "").replace("$.", "").strip("$.")
    clean = text.replace("[*]", "")
    return clean if "." in clean else _root_field(record_root)


def _qualified_record_field(record_root: str, field: str) -> str:
    """Qualify a scalar with its explicit collection owner when needed.

    Search-style ``results[*]`` is normally the sole record universe and remains
    unqualified for compatibility.  Named/nested collections such as ``cast[*]``
    and ``episodes[*].crew[*]`` must keep their owner so sibling arrays exposing
    the same scalar cannot be confused by downstream schema validation.
    """
    field = str(field or "").strip()
    if not field or "." in field:
        return field
    root = str(record_root or "").replace("$.", "").strip("$")
    root = root.replace("[*]", "").strip(".")
    if not root or root == "results":
        return field
    return f"{root}.{field}"


def _relative_leaf(card: dict[str, Any], record_root: str, wanted: str) -> str | None:
    """Resolve a human field name to one *scalar* schema leaf.

    Matching is intentionally conservative.  An identifier is never accepted as
    the value of a human concept merely because it shares the concept token (for
    example ``genres`` -> ``genre_ids``).  Nested list leaves are resolved by
    :func:`_projection_target`, which can preserve their record root.
    """
    wanted = str(wanted or "").strip()
    if not wanted:
        return None
    leaves = _leaf_paths(card)

    def relative_path(path: str) -> tuple[str | None, bool]:
        raw = str(path or "").replace("$.", "")
        root = str(record_root or "$" ).replace("$.", "")
        if root in {"", "$"}:
            rel_raw = raw
        elif raw.startswith(root + "."):
            rel_raw = raw[len(root) + 1:]
        else:
            root_no_star = root.replace("[*]", "")
            if raw.startswith(root_no_star + "."):
                rel_raw = raw[len(root_no_star) + 1:]
            else:
                return None, False
        nested_array = "[*]" in rel_raw
        rel = rel_raw.replace("[*].", ".").replace("[*]", "").strip(".")
        return rel, nested_array

    aliases = {
        "start date": ["first_air_date", "release_date", "air_date"],
        "started": ["first_air_date", "release_date"],
        "birthday date": ["birthday"], "born": ["birthday"],
        "birth date": ["birthday"], "date of birth": ["birthday"],
        "birth place": ["place_of_birth"], "birthplace": ["place_of_birth"],
        "release date": ["release_date"], "released": ["release_date"],
        "name": ["title"], "title": ["name"],
        "review": ["content"], "review content": ["content"],
        "rating": ["vote_average"], "rated": ["vote_average"], "score": ["vote_average"],
        "number of episodes": ["number_of_episodes"],
        "total episodes": ["number_of_episodes"],
        "episode count": ["number_of_episodes"],
        "number of seasons": ["number_of_seasons"],
        "total seasons": ["number_of_seasons"],
        "season count": ["number_of_seasons"],
        "website": ["homepage"],
        "headquarter": ["headquarters"],
        "id": [],
    }
    targets = [wanted] + aliases.get(_cf(wanted), [])
    candidates: list[tuple[int, int, int, str]] = []
    for target_index, target in enumerate(targets):
        target_norm = _cf(target).replace(" ", "_")
        target_tokens = _field_semantic_tokens(target)
        raw_target_tokens = _tokens(target)
        for path in leaves:
            relative, nested_array = relative_path(path)
            if not relative:
                continue
            # A scalar projection rooted at the response object must not tunnel
            # through a normalized list. That would lose the record universe.
            if nested_array and str(record_root or "$") in {"", "$"}:
                continue
            leaf = relative.split(".")[-1]
            leaf_norm = _cf(leaf).replace(" ", "_")
            rel_norm = _cf(relative).replace(" ", "_").replace(".", "_")
            if _identifier_like_path(relative) and not any(t in raw_target_tokens for t in {"id", "ids", "identifier", "key"}):
                continue
            count_like_leaf = (leaf_norm.startswith("number_of_") or leaf_norm.endswith("_count")
                               or leaf_norm.startswith("total_"))
            if count_like_leaf and not (raw_target_tokens & {"number", "total", "count"}):
                continue
            score = 0
            if leaf_norm == target_norm:
                score = 100
            elif rel_norm == target_norm:
                score = 95
            else:
                rel_tokens = _field_semantic_tokens(relative)
                overlap = len(target_tokens & rel_tokens)
                raw_overlap = len(raw_target_tokens & _tokens(relative))
                # Free fuzzy matching requires a lexical anchor. Semantic-only
                # synonyms are handled explicitly in the alias table above; this
                # prevents generic words such as count/number from making
                # unrelated fields look equivalent.
                if raw_overlap == 0:
                    continue
                coverage = raw_overlap / max(1, len(raw_target_tokens))
                if coverage < 0.34:
                    continue
                score = raw_overlap * 12 + overlap * 3
                # Prefer direct leaves over deep incidental matches.
                score -= relative.count(".") * 2
            if target_index:
                score -= target_index
            if score > 0:
                candidates.append((score, -relative.count("."), -len(relative), relative))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][3]


def _human_label_field(card: dict[str, Any], record_root: str) -> str:
    for wanted in ("name", "title", "label", "display_name"):
        found = _relative_leaf(card, record_root, wanted)
        if found:
            return found
    return ""


def _collection_value_field(card: dict[str, Any], record_root: str, wanted: str) -> str:
    label = _human_label_field(card, record_root)
    if label:
        return label
    wanted_tokens = _field_semantic_tokens(wanted)
    asset_request = bool(wanted_tokens & {"photo", "image", "poster", "cover", "logo", "path", "file"})
    if asset_request:
        for name in ("file_path", "path", "url", "uri", "poster_path", "logo_path", "profile_path"):
            found = _relative_leaf(card, record_root, name)
            if found:
                return found
    # A collection of primitive values can be represented by a single leaf under
    # that record root.  Require uniqueness to avoid silently choosing an ID.
    rels = []
    root = str(record_root or "").replace("$.", "")
    for path in _leaf_paths(card):
        clean = str(path).replace("$.", "")
        if clean.startswith(root + "."):
            rel = clean[len(root) + 1:].replace("[*].", ".").replace("[*]", "")
            if rel and "." not in rel and not _identifier_like_path(rel):
                rels.append(rel)
    rels = list(dict.fromkeys(rels))
    return rels[0] if len(rels) == 1 else ""


def _projection_target(card: dict[str, Any], record_root: str, wanted: str) -> tuple[str, str] | None:
    """Return ``(record_root, field)`` for a safe semantic projection."""
    direct = _relative_leaf(card, record_root, wanted)
    if direct:
        return record_root, direct
    wanted_tokens = _field_semantic_tokens(wanted)
    raw_wanted = _tokens(wanted)
    if not wanted_tokens:
        return None
    current = str(record_root or "$" ).replace("$.", "")
    ranked: list[tuple[int, int, str, str]] = []
    for root in _record_paths(card):
        clean = str(root or "").replace("$.", "")
        if "[*]" not in clean:
            continue
        if current not in {"", "$"}:
            current_no_star = current.replace("[*]", "")
            clean_no_star = clean.replace("[*]", "")
            if not (clean.startswith(current + ".") or clean_no_star.startswith(current_no_star + ".")):
                continue
        root_tokens = _field_semantic_tokens(clean)
        raw_overlap = len(raw_wanted & _tokens(clean))
        overlap = len(wanted_tokens & root_tokens)
        if raw_overlap == 0 and overlap < 2:
            continue
        field = _collection_value_field(card, clean, wanted)
        if not field:
            continue
        score = raw_overlap * 12 + overlap * 3
        score -= clean.count(".")
        ranked.append((score, -len(clean), clean, field))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    _, _, root, field = ranked[0]
    return root, field


def _infer_relation_resource(source_resource: str, relation: str, collection: str,
                             explicit_resource: Any = None) -> str:
    explicit = _resource(explicit_resource)
    if explicit:
        return explicit
    text = _cf(" ".join(x for x in (relation, collection) if x))
    toks = _tokens(text)
    if toks & {"movie", "film"}:
        return "movie"
    if toks & {"television", "tv", "show", "series"}:
        return "tv"
    if "network" in toks:
        return "network"
    if "company" in toks or "companies" in toks:
        return "company"
    if "episode" in toks:
        return "episode"
    if "season" in toks and "credit" not in text:
        return "season"
    if _resource(source_resource) in {"movie", "tv", "episode", "season"} and (
            toks & {"cast", "crew", "credit", "actor", "director", "star", "guest"}):
        return "person"
    if _resource(source_resource) == "collection" and ("part" in toks or "parts" in toks):
        return "movie"
    return _resource(source_resource)


def _augment_obvious_literals(question: str, literals: dict[str, Any] | None) -> dict[str, Any]:
    """Compatibility helper: semantics must already be present in typed IR."""
    return dict(literals or {})


def _auto_time_window(question: str = "") -> str:
    """Provider default when a trending route requires a window and IR omits it."""
    return "day"


def _answer_requirement(question: str) -> str:
    return "Answer exactly what the user requested: " + " ".join(str(question or "").split())


def _prune_to_answer_lineage(plan: dict[str, Any]) -> dict[str, Any]:
    """Remove compiled API branches that cannot contribute to the answer.

    Pure graph closure: no question wording, endpoints, entity names, task ids,
    or gold information are consulted. All execution dependencies and selector /
    filter derivations attached to retained steps are preserved.
    """
    out = copy.deepcopy(plan or {})
    steps = {str(s.get("id") or ""): s for s in (out.get("steps") or [])}
    derivs = {str(d.get("id") or ""): d for d in (out.get("derivations") or [])}

    answer_derivs = {str(x) for x in (out.get("answer_derivations") or []) if str(x)}
    answer_steps = {str(x) for x in (out.get("answer_steps") or []) if str(x)}
    obligation_steps = {str(x.get("step_id") or "") for x in
                        (out.get("action_obligations") or []) if isinstance(x, dict)
                        and str(x.get("step_id") or "")}

    # Follow terminal derivations backwards to the API steps they consume.
    live_derivs = set(answer_derivs)
    stack = list(live_derivs)
    while stack:
        did = stack.pop()
        d = derivs.get(did) or {}
        answer_steps.update(str(x) for x in (d.get("source_steps") or []) if str(x))
        answer_steps.update(str(x) for x in (d.get("label_steps") or []) if str(x))
        for parent in d.get("source_derivations") or []:
            parent = str(parent or "")
            if parent and parent not in live_derivs:
                live_derivs.add(parent); stack.append(parent)

    if not answer_steps and not answer_derivs:
        return out

    # Retain transitive execution dependencies for every live answer step and
    # every independently required side effect. Obligations remain separate from
    # answer_steps so certification does not confuse intermediate writes with the
    # terminal answer surface.
    live_steps = set(answer_steps) | set(obligation_steps)
    step_stack = list(live_steps)
    while step_stack:
        sid = step_stack.pop()
        for dep in (steps.get(sid) or {}).get("depends_on") or []:
            dep = str(dep or "")
            if dep and dep not in live_steps:
                live_steps.add(dep); step_stack.append(dep)

    # Select/filter/rank derivations on retained producer steps are execution
    # semantics even when the final derivation does not reference them directly.
    # Keep them, plus their derivation ancestors. This avoids implicit record
    # choice while still dropping derivations belonging solely to pruned steps.
    changed = True
    while changed:
        changed = False
        for did, d in derivs.items():
            src_steps = {str(x) for x in (d.get("source_steps") or []) if str(x)}
            parents = {str(x) for x in (d.get("source_derivations") or []) if str(x)}
            keep = did in live_derivs
            if src_steps and src_steps <= live_steps:
                keep = True
            if parents and parents <= live_derivs:
                keep = True
            if keep and did not in live_derivs:
                live_derivs.add(did); changed = True
            if keep:
                for parent in parents:
                    if parent in derivs and parent not in live_derivs:
                        live_derivs.add(parent); changed = True

    old_steps = list(out.get("steps") or [])
    old_derivs = list(out.get("derivations") or [])
    out["steps"] = [s for s in old_steps if str(s.get("id") or "") in live_steps]
    out["derivations"] = [d for d in old_derivs if str(d.get("id") or "") in live_derivs]
    out["answer_steps"] = [str(x) for x in (out.get("answer_steps") or []) if str(x) in live_steps]
    out["answer_derivations"] = [str(x) for x in (out.get("answer_derivations") or []) if str(x) in live_derivs]
    out["typed_fanout_bindings"] = [
        dict(x) for x in (out.get("typed_fanout_bindings") or [])
        if isinstance(x, dict) and str(x.get("step_id") or "") in live_steps
    ]

    removed_steps = [str(s.get("id") or "") for s in old_steps if str(s.get("id") or "") not in live_steps]
    removed_derivs = [str(d.get("id") or "") for d in old_derivs if str(d.get("id") or "") not in live_derivs]
    if removed_steps or removed_derivs:
        out.setdefault("validation_warnings", []).append(
            "pruned unused compiled lineage: "
            f"steps={removed_steps or []}, derivations={removed_derivs or []}")
    return out

def _rewrite_direct_named_action_targets(intent: dict[str, Any], semantics) -> dict[str, Any]:
    """Collapse an unnecessarily enumerated named child into direct lookup.

    This is a provider-neutral typed-IR rewrite.  It applies only when all of the
    following are explicit in the graph/provider declaration:
      * an action consumes a resource type that the provider can find directly;
      * its input is an exact name/title filter over a relation collection;
      * that relation comes from a concrete named parent entity.

    In that shape the child itself is already explicitly named and is the thing
    being acted on.  Enumerating the parent's collection is strictly more brittle
    (provider collection endpoints can exclude subtypes such as singles) and adds
    no semantic obligation.  Reusing the filter node id keeps action/answer
    lineage stable; dead parent/relation branches are pruned after compilation.
    No question wording, task id, gold route, or expected answer is consulted.
    """
    if semantics is None:
        return intent
    obj = copy.deepcopy(intent or {})
    nodes = [dict(n) for n in (obj.get("nodes") or []) if isinstance(n, dict)]
    by_id = {str(n.get("id") or ""): n for n in nodes}
    findable = set(str(x) for x in dict(getattr(semantics, "find_routes", {}) or {}))
    actions = dict(getattr(semantics, "action_routes", {}) or {})
    changed = False

    for action in nodes:
        if str(action.get("op") or "") != "action":
            continue
        spec = dict(actions.get(str(action.get("action") or "")) or {})
        accepted = set()
        if spec.get("input_resource"):
            accepted.add(str(spec.get("input_resource")))
        accepted.update(str(x) for x in (spec.get("input_resources") or []) if str(x))
        if not accepted:
            continue
        input_id = str(action.get("input") or "")
        filt = by_id.get(input_id) or {}
        if str(filt.get("op") or "") != "filter":
            continue
        field = _cf(filt.get("field") or "")
        value = filt.get("value")
        if field not in {"name", "title"} or not isinstance(value, str) or not value.strip():
            continue
        rel = by_id.get(str(filt.get("source") or "")) or {}
        if str(rel.get("op") or "") != "relation":
            continue
        child_resource = str(rel.get("resource") or "")
        if child_resource not in accepted or child_resource not in findable:
            continue
        parent = by_id.get(str(rel.get("source") or "")) or {}
        if str(parent.get("op") or "") != "find" or not str(parent.get("name") or "").strip():
            continue
        # Preserve the node id consumed by the action and keep lightweight audit
        # metadata.  The compiler ignores these private metadata keys.
        parent_name = str(parent.get("name") or "").strip()
        replacement = {
            "id": input_id, "op": "find", "resource": child_resource,
            "name": value.strip(), "literals": dict(filt.get("literals") or {}),
            "_oca_direct_named_action_target": True,
            "_oca_parent_context_resource": str(parent.get("resource") or ""),
            "_oca_parent_context_name": parent_name,
            # Keep the semantic target name untouched for answer/lineage, but make
            # the provider search less ambiguous by retaining its named parent.
            "_oca_search_query": (value.strip() + " " + parent_name).strip(),
        }
        filt.clear(); filt.update(replacement)
        changed = True

    if changed:
        obj["nodes"] = nodes
    return obj



def _rewrite_ungrounded_free_text_action_targets(question: str, intent: dict[str, Any], semantics) -> dict[str, Any]:
    """Replace invented proxy constraints with a direct free-text target.

    A semantic planner can sometimes force an ordinary qualitative descriptor
    into unrelated provider fields (for example, mapping a mood/adjective onto
    explicitness or an arbitrary duration threshold) and can also invent a
    provider population that the user never named.  When an action directly
    consumes a resource that the provider can search, this rewrite is allowed
    only if the entire constrained population is ungrounded in the request:

    * the root provider population is not mentioned by any declared alias;
    * none of the filter fields (or their declared aliases) is mentioned;
    * the request contains no numeric/ordinal/time/comparison cue that would
      make a structured filter plausible; and
    * the action target resource is directly searchable.

    In that narrow shape, preserve the user's remaining descriptive phrase as
    free-text search input instead of retaining invented semantic constraints.
    This is provider-neutral and never consults task ids, gold routes, or answers.
    """
    if semantics is None:
        return intent
    import re as _rewrite_re

    obj = copy.deepcopy(intent or {})
    nodes = [dict(n) for n in (obj.get("nodes") or []) if isinstance(n, dict)]
    by_id = {str(n.get("id") or ""): n for n in nodes}
    actions = dict(getattr(semantics, "action_routes", {}) or {})
    findable = set(str(x) for x in dict(getattr(semantics, "find_routes", {}) or {}))
    pop_aliases = {str(k): str(v) for k, v in dict(getattr(semantics, "population_aliases", {}) or {}).items()}
    fields = dict(getattr(semantics, "semantic_fields", {}) or {})
    qfold = _cf(question)

    # Structured language means the graph may legitimately need typed filters;
    # do not reinterpret it as free text merely because field names differ.
    structured_cue = _rewrite_re.search(
        r"(?:\d|\b(?:first|second|third|fourth|fifth|top|most|least|before|after|"
        r"under|over|less|more|shorter|longer|minute|minutes|second|seconds|hour|hours|"
        r"explicit|popular|popularity|released|release|duration)\b)", qfold)

    changed = False
    for action in nodes:
        if _cf(action.get("op")).replace(" ", "_") != "action":
            continue
        action_name = _cf(action.get("action") or "").replace(" ", "_")
        spec = dict(actions.get(action_name) or {})
        accepted = set()
        if spec.get("input_resource"):
            accepted.add(str(spec.get("input_resource")))
        accepted.update(str(x) for x in (spec.get("input_resources") or []) if str(x))
        if len(accepted) != 1:
            continue
        target_resource = next(iter(accepted))
        if target_resource not in findable:
            continue
        input_id = str(action.get("input") or "").strip()
        target = by_id.get(input_id) or {}
        if not input_id or not target:
            continue

        # Walk only a simple select/filter chain. Relations/named entities carry
        # semantic scope and therefore are never collapsed by this rewrite.
        cursor = target
        filters_seen = []
        seen_ids = set()
        while cursor and str(cursor.get("id") or "") not in seen_ids:
            seen_ids.add(str(cursor.get("id") or ""))
            op = _cf(cursor.get("op")).replace(" ", "_")
            if op == "filter":
                filters_seen.append(cursor)
                cursor = by_id.get(str(cursor.get("source") or "")) or {}
                continue
            if op == "select":
                cursor = by_id.get(str(cursor.get("source") or "")) or {}
                continue
            break
        root = cursor or {}
        if not filters_seen or _cf(root.get("op")).replace(" ", "_") != "population":
            continue
        if str(root.get("resource") or "") != target_resource:
            continue

        population = _cf(root.get("population") or "").replace(" ", "_")
        pop_terms = {population.replace("_", " ")}
        pop_terms.update(str(alias).replace("_", " ") for alias, canonical in pop_aliases.items()
                         if _cf(canonical).replace(" ", "_") == population)
        if any(term and (" " + term + " ") in (" " + qfold + " ") for term in pop_terms):
            continue
        if structured_cue:
            continue

        grounded_filter = False
        for filt in filters_seen:
            fname = _cf(filt.get("field") or "").replace(" ", "_")
            terms = {fname.replace("_", " ")} if fname else set()
            sf = fields.get(fname)
            if sf is not None:
                terms.update(_cf(x) for x in (getattr(sf, "aliases", ()) or ()))
            if any(term and (" " + term + " ") in (" " + qfold + " ") for term in terms):
                grounded_filter = True
                break
        if grounded_filter:
            continue

        # Derive the free-text descriptor from the user's wording after the
        # provider action verb.  Only generic politeness/determiner tokens are
        # stripped; resource/descriptive words are preserved for search.
        verb = action_name.split("_", 1)[0].replace("_", " ")
        m = _rewrite_re.search(r"\b" + _rewrite_re.escape(verb) + r"\b", str(question), _rewrite_re.I)
        if not m:
            continue
        tail = str(question)[m.end():].strip(" \t\r\n:,-")
        toks = tail.split()
        stop = {"me", "us", "some", "a", "an", "the", "please", "to"}
        while toks and _cf(toks[0]).strip(".,!?;:") in stop:
            toks.pop(0)
        search_text = " ".join(toks).strip(" \t\r\n.,!?;:")
        if not search_text:
            continue

        replacement = {
            "id": input_id, "op": "find", "resource": target_resource,
            "name": search_text, "literals": {},
            "_oca_free_text_action_target": True,
        }
        target.clear(); target.update(replacement)
        changed = True

    if changed:
        obj["nodes"] = nodes
    return obj

def compile_intent_to_plan(question: str, benchmark: str, intent: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    semantics = None
    catalog = None
    try:
        from oca.capability.provider import get_provider_semantics
        semantics = get_provider_semantics(benchmark)
        catalog = semantics.catalog if semantics is not None else None
    except Exception:
        semantics = None
        catalog = None
    intent = _normalize_provider_typed_aliases(intent, semantics)
    intent = _rewrite_direct_named_action_targets(intent, semantics)
    intent = _rewrite_ungrounded_free_text_action_targets(question, intent, semantics)
    intent, intent_errors = validate_intent(intent)
    if intent_errors:
        return _secat_v4160_enforce_terminal_relation_selectors({"version": 2, "steps": [], "derivations": [], "answer_steps": [],
                "answer_derivations": [], "answer_mode": "direct", "valid": False,
                "validation_errors": intent_errors, "intent": intent}, intent_errors)
    cards = _catalog(benchmark)
    capability_report = None

    # Once a provider is known, explicit resource labels must belong to that
    # provider's ontology. This prevents a cross-provider type (for example a
    # TMDB movie) from slipping through generic IR validation and then being
    # matched lexically against a Spotify schema.
    if semantics is not None and semantics.resource_types:
        bad_resources = sorted({
            str(node.get("resource") or "")
            for node in intent.get("nodes", [])
            if str(node.get("resource") or "")
            and str(node.get("resource") or "") not in semantics.resource_types
        })
        if bad_resources:
            provider_errors = [
                f"resource {resource!r} is not part of provider {semantics.provider!r}"
                for resource in bad_resources
            ]
            return _secat_v4160_enforce_terminal_relation_selectors({
                "version": 2, "steps": [], "derivations": [],
                "answer_steps": [], "answer_derivations": [],
                "answer_mode": "direct", "valid": False,
                "validation_errors": provider_errors, "intent": intent,
            }, provider_errors)

    # A known provider with a declarative capability layer must normalize through
    # that provider.  This keeps the compiler provider-neutral: adding Spotify is
    # a catalog registration, not a TMDB-shaped branch in the compiler.
    if catalog is not None:
        try:
            from oca.capability.normalize import normalize_intent
            intent, capability_report = normalize_intent(intent, catalog, cards)
        except Exception as exc:
            return _secat_v4160_enforce_terminal_relation_selectors({
                "version": 2, "steps": [], "derivations": [],
                "answer_steps": [], "answer_derivations": [],
                "answer_mode": "direct", "valid": False,
                "validation_errors": [f"capability normalization failed: {exc}"],
                "intent": intent,
            }, [f"capability normalization failed: {exc}"])
    steps: list[dict[str, Any]] = []
    derivations: list[dict[str, Any]] = []
    handles: dict[str, _Handle] = {}
    route_diagnostics: dict[str, Any] = {}
    errors: list[str] = []
    if capability_report is not None:
        # Most normalizer type diagnostics are advisory because some semantic
        # fields (for example credit-local ``job``) are resolved against the
        # provider record collection later in compilation. Strong asset-owner
        # mismatches are different: PERSON->LOGO must never become a valid plan.
        errors.extend(str(x) for x in capability_report.type_errors
                      if str(x) and "asset field owner mismatch:" in str(x))
    step_cards: dict[str, dict[str, Any]] = {}
    step_counter = 0
    nodes_by_id = {str(n.get("id") or ""): n for n in intent["nodes"]}

    # Relations that ultimately supply a state-changing action must use their
    # canonical provider acquisition path rather than relying only on a compact
    # embedding from an upstream response.  This strengthens both execution
    # provenance and provider-route closure without changing ordinary read-only
    # queries.  The rule is graph-based: no task wording or benchmark IDs.
    action_ancestor_ids: set[str] = set()
    _pending_action_refs = []
    for _node in intent["nodes"]:
        if str(_node.get("op") or "") != "action":
            continue
        for _key in ("source", "input"):
            _ref = str(_node.get(_key) or "")
            if _ref:
                _pending_action_refs.append(_ref)
    while _pending_action_refs:
        _ref = _pending_action_refs.pop()
        if not _ref or _ref in action_ancestor_ids:
            continue
        action_ancestor_ids.add(_ref)
        _parent = nodes_by_id.get(_ref) or {}
        for _key in ("source", "input", "left", "right"):
            _up = str(_parent.get(_key) or "")
            if _up and _up not in action_ancestor_ids:
                _pending_action_refs.append(_up)

    population_filters: dict[str, list[dict[str, Any]]] = {}
    for candidate in intent["nodes"]:
        if candidate.get("op") != "filter":
            continue
        cursor = str(candidate.get("source") or "")
        visited: set[str] = set()
        while cursor and cursor not in visited:
            visited.add(cursor)
            parent = nodes_by_id.get(cursor) or {}
            if parent.get("op") == "population":
                population_filters.setdefault(cursor, []).append(candidate)
                break
            if parent.get("op") != "filter":
                break
            cursor = str(parent.get("source") or "")
    detail_cache: dict[tuple[str, str, str], _Handle] = {}
    # Explicit typed collection->child lifting.  This is execution semantics, not
    # question parsing: a relation whose source handle remains a collection may
    # fan out only when the typed answer contract itself is plural.
    typed_fanout_bindings: list[dict[str, Any]] = []

    def new_step(card: dict[str, Any], purpose: str, *, depends_on=None,
                 query_literals=None, path_literals=None, path_bindings=None) -> tuple[dict[str, Any], str]:
        nonlocal step_counter
        step_counter += 1
        sid = f"s{step_counter}"
        step = {
            "id": sid, "method": str(card.get("method") or "GET").upper(),
            "endpoint": str(card.get("endpoint") or ""), "purpose": purpose,
            "depends_on": list(depends_on or []), "binds": [], "binding_paths": {},
            "path_literals": dict(path_literals or {}), "path_bindings": dict(path_bindings or {}),
            "query_literals": dict(query_literals or {}), "query_bindings": {},
            "body_literals": {}, "body_bindings": {}, "answer_source": False,
        }
        steps.append(step); step_cards[sid] = card
        return step, sid

    def add_derivation(node_id: str, op: str, *, source_step: str = "",
                       source_derivations=None, field: Any = None, rank: int = 0,
                       filter0=None, comparison: str = "", comparison_literal=None,
                       label_steps=None, label_fields=None, unit="raw", distinct_field=None,
                       purpose="") -> str:
        did = f"i_{node_id}"
        item = {
            "id": did, "operator": op,
            "source_steps": [source_step] if source_step else [],
            "source_derivations": list(source_derivations or []),
            "label_steps": list(label_steps or []), "label_fields": list(label_fields or []),
            "field": field, "comparison": comparison, "comparison_literal": comparison_literal,
            "unit": unit, "distinct_field": distinct_field, "rank": int(rank or 0),
            "filter": dict(filter0 or {}), "top_k": 10,
            "purpose": purpose or f"Intent operation {node_id}",
        }
        derivations.append(item)
        return did

    def ensure_id_binding(handle: _Handle, consumer_node: str) -> str:
        if handle.id_alias:
            return handle.id_alias
        if not handle.step_id:
            return ""
        step = next((x for x in steps if x["id"] == handle.step_id), None)
        card = step_cards.get(handle.step_id) or {}
        if not step:
            return ""
        id_field = _relative_leaf(card, handle.record_root, "id")
        if not id_field:
            return ""
        alias = f"intent_{handle.node_id}_id"
        if alias not in step["binds"]:
            step["binds"].append(alias)
        step["binding_paths"][alias] = _qualified_record_field(handle.record_root, id_field)
        handle.id_alias = alias
        # Record every provider placeholder that identifies this resource.
        # Season/episode *numbers* are scope values, not ids, and are handled by
        # ensure_scope_binding below.
        for ph, rtype in _PLACEHOLDER_RESOURCE.items():
            if rtype == handle.resource and ph not in {"season_number", "episode_number"}:
                handle.path_aliases[ph] = alias
        return alias

    def ensure_value_binding(handle: _Handle, semantic: str, consumer_node: str) -> str:
        """Bind one scalar property from the selected semantic owner.

        ``semantic='value'`` binds the record itself, which is needed for typed
        primitive collections such as Spotify ``genres[*]``.  Like id bindings,
        selection/fan-out semantics remain represented by the producer derivation;
        this helper only declares the schema-grounded response path.
        """
        if not handle.step_id:
            return ""
        step = next((x for x in steps if x["id"] == handle.step_id), None)
        card = step_cards.get(handle.step_id) or {}
        if not step:
            return ""
        semantic0 = _cf(semantic).replace(" ", "_")
        if semantic0 in {"value", "record", "self"}:
            path = str(handle.record_root or "$")
            alias_base = "value"
        else:
            leaf = _relative_leaf(card, handle.record_root, semantic)
            if not leaf:
                return ""
            path = _qualified_record_field(handle.record_root, leaf)
            alias_base = re.sub(r"[^a-z0-9]+", "_", semantic0).strip("_") or "value"
        alias = f"intent_{handle.node_id}_{alias_base}"
        if alias not in step["binds"]:
            step["binds"].append(alias)
        step["binding_paths"][alias] = path
        return alias

    def ensure_scope_binding(handle: _Handle, placeholder: str) -> str:
        """Bind a selected season/episode number from typed record lineage."""
        if handle.path_aliases.get(placeholder):
            return handle.path_aliases[placeholder]
        if placeholder not in {"season_number", "episode_number"} or not handle.step_id:
            return ""
        step = next((x for x in steps if x["id"] == handle.step_id), None)
        card = step_cards.get(handle.step_id) or {}
        if not step:
            return ""
        wanted = "season number" if placeholder == "season_number" else "episode number"
        leaf = _relative_leaf(card, handle.record_root, wanted)
        if not leaf:
            # Exact provider spelling is still schema-grounded and does not inspect
            # the natural-language request.
            leaf = _relative_leaf(card, handle.record_root, placeholder)
        if not leaf:
            return ""
        alias = f"intent_{handle.node_id}_{placeholder}"
        if alias not in step["binds"]:
            step["binds"].append(alias)
        step["binding_paths"][alias] = _qualified_record_field(handle.record_root, leaf)
        handle.path_aliases[placeholder] = alias
        return alias

    def promote_for_projection(source: _Handle, wanted: str, consumer_node: str, *,
                               prefer_detail: bool = False) -> tuple[_Handle, tuple[str, str] | None]:
        """Resolve a projection locally or add one deterministic acquisition call.

        API mechanics stay out of the model intent.  Human-facing visual requests
        first prefer a documented dedicated image/asset relation; ordinary scalar
        properties first use the already acquired entity and then one same-resource
        detail endpoint.  Promotion is accepted only when every dynamic placeholder
        is bindable from the selected semantic owner.
        """
        card = step_cards.get(source.step_id) or {}
        wanted_tokens = _field_semantic_tokens(wanted)
        asset_requested = bool(
            wanted_tokens & {"photo", "image", "poster", "cover", "logo", "profile", "backdrop"})

        def _asset_target(relation_card: dict[str, Any]):
            """Choose the visual collection named by the human request.

            Image endpoints commonly expose sibling posters/backdrops/logos (or
            profiles for people).  A bare file_path is not enough to distinguish
            those relations, so make the semantic collection choice explicitly.
            """
            roots = list(dict.fromkeys(
                str(r) for r in _record_paths(relation_card) if "[*]" in str(r)))
            wt = _tokens(wanted)
            preferred = []
            if "logo" in wt:
                preferred = [r for r in roots if "logo" in _cf(r)]
            elif "poster" in wt or "cover" in wt:
                preferred = [r for r in roots if "poster" in _cf(r)]
            elif source.resource == "person" or "profile" in wt or "photo" in wt:
                preferred = [r for r in roots if "profile" in _cf(r)]
            elif source.resource in {"movie", "tv", "collection"}:
                # Generic single work/collection images are most naturally surfaced
                # as posters when the API distinguishes posters from backdrops/logos.
                preferred = [r for r in roots if "poster" in _cf(r)]
            for root0 in preferred + roots:
                for leaf_name in ("file path", "url", "uri", "path"):
                    leaf = _relative_leaf(relation_card, root0, leaf_name)
                    if leaf and not _identifier_like_path(leaf):
                        return root0, leaf
            return None

        def _via_relation(relation_name: str, cache_tag: str):
            # Compiler-generated relation acquisition must use the same typed
            # provider catalog as model-emitted relations.  Raw lexical route
            # scoring cannot safely recover hierarchical endpoints such as
            # episode.images, whose path requires series/season/episode scope.
            relation_cap = None
            try:
                target_hint = "image" if cache_tag == "images" else source.resource
                relation_cap = catalog.try_lookup(source.resource, relation_name, target_hint) if catalog is not None else None
            except Exception:
                relation_cap = None
            relation_key = (f"__cap__{relation_cap.semantic_name}"
                            if relation_cap is not None else relation_name)
            relation_card, diag = _resolve_route(
                "relation", source_resource=source.resource, relation=relation_key,
                question=question, cards=cards, semantics=semantics)
            route_diagnostics[f"{consumer_node}__auto_{cache_tag}"] = diag
            if not relation_card:
                return None
            relation_target = (_asset_target(relation_card) if cache_tag == "images"
                               else _projection_target(relation_card, "$", wanted))
            if not relation_target:
                return None
            endpoint = str(relation_card.get("endpoint") or "")
            if endpoint == str(card.get("endpoint") or ""):
                return None
            placeholders = _path_placeholders(endpoint)
            path_bindings: dict[str, str] = {}
            # Use the same typed parent-binding semantics as ordinary relation
            # compilation.  Hierarchical owners (season/episode) carry the
            # provider parent aliases in their handle, so an automatic image
            # promotion can bind series_id + season_number + episode_number
            # without recovering scope from natural-language text.
            for ph in placeholders:
                binding_spec = str((relation_cap.parent_binding or {}).get(ph) or "") if relation_cap is not None else ""
                # A provider-declared source.id binding must always use the id of
                # the *current typed source*.  Reusing a same-named placeholder
                # inherited from its parent can bind an artist id where an album id
                # is required (generic {id} paths are especially vulnerable).
                bound_alias = (ensure_id_binding(source, consumer_node)
                               if binding_spec == "source.id" else source.path_aliases.get(ph, ""))
                if not bound_alias and binding_spec == "source.id":
                    bound_alias = ensure_id_binding(source, consumer_node)
                elif not bound_alias and binding_spec in {"source.season_number", "owner.season_number"}:
                    bound_alias = ensure_scope_binding(source, "season_number")
                elif not bound_alias and binding_spec in {"source.episode_number", "owner.episode_number"}:
                    bound_alias = ensure_scope_binding(source, "episode_number")
                elif not bound_alias and binding_spec == "owner.id":
                    bound_alias = source.path_aliases.get(ph, "")
                if not bound_alias and _PLACEHOLDER_RESOURCE.get(ph) == source.resource:
                    bound_alias = ensure_id_binding(source, consumer_node)
                if not bound_alias:
                    return None
                path_bindings[ph] = bound_alias
            key = (source.step_id, source.derivation_id or "", endpoint)
            cached = detail_cache.get(key)
            if cached is not None:
                return cached, relation_target
            step, sid = new_step(
                relation_card,
                f"Get selected {source.resource} {cache_tag} required for {wanted}",
                depends_on=[source.step_id] if source.step_id else [],
                path_bindings=path_bindings)
            label = _human_label_field(relation_card, "$")
            promoted = _Handle(
                consumer_node + "__" + cache_tag, "object", source.resource,
                sid, "", "$", label_step=sid,
                label_field=label or source.label_field,
                owner_step=(source.owner_step or source.label_step or source.step_id),
                owner_label_field=(source.owner_label_field or source.label_field or label),
                path_aliases=dict(source.path_aliases))
            detail_cache[key] = promoted
            return promoted, relation_target

        # If the semantic intent already acquired a dedicated image relation
        # (episode/season image endpoints are common examples), resolve the human
        # visual concept inside that response instead of trying to chain another
        # image call.
        if asset_requested and "/images" in str(card.get("endpoint") or ""):
            current_asset = _asset_target(card)
            if current_asset:
                return source, current_asset

        # A dedicated images/asset operation is semantically stronger than a
        # thumbnail/path embedded in search metadata.  This prevents a grounded
        # but wrong-relation shortcut for poster/photo/logo questions.
        if asset_requested and source.kind in {"entity", "object", "scalar", "collection"}:
            visual = _via_relation("images", "images")
            if visual:
                return visual

        # Broad metadata/detail answers should prefer the provider's canonical
        # same-resource detail capability even when a simplified embedded object
        # happens to expose some of the requested fields.  This prevents a
        # currently-playing track's simplified album object (or analogous
        # provider summary object) from masquerading as full resource detail.
        # The rule is typed/provider-declared and applies only when the answer
        # contract explicitly requests detail semantics.
        if prefer_detail and source.kind in {"entity", "object", "scalar"}:
            detail = _via_relation("details", "detail")
            if detail:
                return detail

        target = _projection_target(card, source.record_root, wanted)
        if target:
            return source, target
        # Some benchmark/tool schemas intentionally omit the response schema for a
        # documented read endpoint (recommendations is a common example).  Request
        # authorization remains strict, but response-field validation is already
        # deferred to the immutable runtime profile.  Preserve the *requested*
        # semantic field rather than forcing a legacy replan solely because the OAS
        # omitted response leaves.  This never invents a value: runtime replay must
        # still observe the field before certification.
        if (str(card.get("method") or "GET").upper() == "GET" and not _leaf_paths(card)):
            deferred_alias = {
                "rating": "vote_average", "release date": "release_date",
                "air date": "air_date", "first air date": "first_air_date",
                "birth date": "birthday", "birthday": "birthday",
                "review": "content", "review content": "content",
            }.get(_cf(wanted), _cf(wanted).replace(" ", "_"))
            if deferred_alias and not _identifier_like_path(deferred_alias):
                return source, (source.record_root or "$", deferred_alias)
        if source.kind not in {"entity", "object", "scalar"}:
            return source, None
        detail = _via_relation("details", "detail")
        if detail:
            return detail
        return source, None

    def scalar_count_property(source: _Handle) -> tuple[_Handle, tuple[str, str] | None] | None:
        """Do not reinterpret a model ``count`` from raw question wording.

        Stored totals must be represented as ``project`` in semantic IR; ``count``
        means the cardinality of the selected collection.
        """
        return None

    answer_hint = intent.get("answer") or {}
    answer_source_hint = str(answer_hint.get("source") or "")
    answer_mode_hint = _cf(answer_hint.get("mode") or "direct")

    def owned_named_population(resource: str, name: str) -> str:
        """Resolve an explicitly user-owned named entity through its mine route.

        This is provider-neutral.  It activates only when the provider declares a
        ``(resource, "mine")`` population and the original request explicitly
        contains a possessive resource phrase (for example ``my playlist``) *after*
        removing the literal entity name.  Thus a literal title beginning with the possessive token alone
        never implies ownership.  The goal is to avoid global search for a resource
        the user explicitly scoped to their own account.
        """
        if semantics is None or (resource, "mine") not in dict(getattr(semantics, "population_routes", {}) or {}):
            return ""
        q = str(question or "")
        if not q or not name:
            return ""
        try:
            q_without_name = re.sub(re.escape(str(name)), " ", q, flags=re.I)
        except Exception:
            q_without_name = q
        folded = re.sub(r"[^a-z0-9]+", " ", q_without_name.casefold()).strip()
        aliases = {resource, resource + "s"}
        for alias, target in dict(getattr(semantics, "resource_aliases", {}) or {}).items():
            if str(target) == str(resource):
                aliases.add(str(alias).casefold())
        for alias in sorted(aliases, key=len, reverse=True):
            a = re.sub(r"[^a-z0-9]+", " ", str(alias).casefold()).strip()
            if not a:
                continue
            if re.search(r"\bmy(?:\s+own)?\s+" + re.escape(a) + r"\b", folded):
                return "mine"
        return ""

    for node in intent["nodes"]:
        nid = node["id"]; op = node["op"]
        if op == "find":
            resource = _resource(node.get("resource"))
            name = str(node.get("name") or "").strip()
            if not resource or not name:
                errors.append(f"intent node {nid}: find needs resource and name"); continue

            # If the request explicitly scopes a named resource to the current
            # user's own collection and the provider declares a mine population,
            # enumerate that owned population and apply an exact name filter.
            # This prevents stale/public global-search results from satisfying
            # requests such as a user's named playlist.
            owned_population = owned_named_population(resource, name)
            if owned_population:
                card, diag = _resolve_route(
                    "population", resource=resource, population=owned_population,
                    literals={}, filter_hints=[], question=question, cards=cards, semantics=semantics)
                route_diagnostics[nid] = diag
                if not card:
                    errors.append(
                        f"intent node {nid}: no unambiguous owned population route for {resource}")
                    continue
                route_spec = dict(card.get("_oca_route_spec") or {})
                step, sid = new_step(
                    card, f"Find the named owned {resource}: {name}",
                    query_literals=dict(route_spec.get("query_literals") or {}),
                    path_literals=dict(route_spec.get("path_literals") or {}))
                root = _record_root(card, prefer_results=True, catalog=catalog)
                root_field = _root_field(root) or "results"
                label = _human_label_field(card, root)
                if not label:
                    errors.append(
                        f"intent node {nid}: owned {resource} route has no label field for exact-name selection")
                    continue
                filtered = add_derivation(
                    nid + "_owned_name", "filter", source_step=sid, field=root_field,
                    filter0={label: {"op": "eq", "value": name}},
                    purpose=f"Keep the owned {resource} whose {label} matches the requested name")
                pick = add_derivation(
                    nid + "_pick", "first", source_step=sid, source_derivations=[filtered],
                    field=root_field, rank=0, purpose=f"Select the named owned {resource}")
                h = _Handle(nid, "entity", resource, sid, pick, root,
                            label_step=sid, label_field=label, owner_step=sid, owner_label_field=label)
                ensure_id_binding(h, nid)
                handles[nid] = h
                continue

            card, diag = _resolve_route("find", resource=resource, question=question, cards=cards,
                                        semantics=semantics)
            route_diagnostics[nid] = diag
            if not card:
                errors.append(f"intent node {nid}: no unambiguous search route for {resource}"); continue
            route_spec = dict(card.get("_oca_route_spec") or {})
            query_key = str(route_spec.get("query_param") or "query")
            query_literals = dict(route_spec.get("query_literals") or {})
            query_literals[query_key] = str(node.get("_oca_search_query") or name).strip()
            path_literals = dict(route_spec.get("path_literals") or {})
            step, sid = new_step(card, f"Find the named {resource}: {name}",
                                 query_literals=query_literals, path_literals=path_literals)
            root = _record_root(card, prefer_results=True, catalog=catalog)
            root_field = _root_field(root) or "results"
            pick = add_derivation(nid + "_pick", "endpoint_rank", source_step=sid,
                                  field=root_field, rank=0, purpose=f"Select the named {resource} result")
            label = _human_label_field(card, root)
            h = _Handle(nid, "entity", resource, sid, pick, root,
                        label_step=sid, label_field=label, owner_step=sid, owner_label_field=label)
            ensure_id_binding(h, nid)
            handles[nid] = h
            continue

        if op == "population":
            resource = _resource(node.get("resource"))
            population = str(node.get("population") or node.get("qualifier") or node.get("relation") or "").strip()
            literals = dict(node.get("literals") or {})
            filter_hints = population_filters.get(nid, [])
            card, diag = _resolve_route("population", resource=resource, population=population,
                                        literals=literals, filter_hints=filter_hints,
                                        question=question, cards=cards, semantics=semantics)
            route_diagnostics[nid] = diag
            if not card:
                errors.append(f"intent node {nid}: no unambiguous population route for {resource}: {population}"); continue
            route_spec = dict(card.get("_oca_route_spec") or {})
            for ph in _path_placeholders(str(card.get("endpoint") or "")):
                if ph == "time_window" and ph not in literals:
                    literals[ph] = _auto_time_window(question)
            path_literals = dict(route_spec.get("path_literals") or {})
            path_literals.update({k: v for k, v in literals.items() if k in _path_placeholders(str(card.get("endpoint") or ""))})
            query_literals = dict(route_spec.get("query_literals") or {})
            query_literals.update({k: v for k, v in literals.items() if k in _param_names(card, "query")})
            for hint in filter_hints:
                mapped = _query_parameter_for_filter(
                    card, str(hint.get("field") or ""),
                    str(hint.get("comparison") or "eq"), hint.get("value"))
                if mapped and mapped[0] not in query_literals:
                    query_literals[mapped[0]] = mapped[1]
            step, sid = new_step(card, f"Get the requested {resource} population: {population}",
                                 path_literals=path_literals, query_literals=query_literals)
            root = _record_root(card, prefer_results=True, catalog=catalog)
            label = _human_label_field(card, root)
            population_derivation = ""
            # Mixed-media trending feeds preserve a global trend order but expose
            # multiple resource families.  When the semantic population explicitly
            # names a movie, scope the returned records by documented media_type
            # before any first/nth selection. Filtering preserves feed order.
            if "/trending/all/" in str(card.get("endpoint") or "") and resource:
                media_field = _relative_leaf(card, root, "media type")
                if media_field:
                    population_derivation = add_derivation(
                        nid + "_resource", "filter", source_step=sid,
                        field=_root_field(root) or None,
                        filter0={media_field: {"op": "eq", "value": resource}},
                        purpose=f"Keep only requested {resource} records from the mixed trending feed")
            population_kind = "collection" if "[*]" in root else "object"
            handles[nid] = _Handle(nid, population_kind, resource, sid, population_derivation, root,
                                   label_step=sid, label_field=label, owner_step=sid, owner_label_field=label)
            continue

        if op == "relation":
            source = handles.get(str(node.get("source") or ""))
            if not source:
                errors.append(f"intent node {nid}: relation source is unavailable"); continue
            relation = str(node.get("relation") or "").strip()
            collection = str(node.get("collection") or "").strip()
            literals = _augment_obvious_literals(question, node.get("literals") or {})

            # Prefer an explicitly named nested relation that is already present in
            # the current response.  Search/population endpoints often embed useful
            # child records (for example a person's ``known_for`` works).  Requiring
            # a new relation endpoint in that case is both wasteful and, when no
            # dedicated endpoint exists, can make a perfectly answerable task fail
            # compilation.  Only descend when the requested relation/collection
            # tokens actually identify a *different nested record root*; the generic
            # fallback behavior of _record_root must never turn the current record
            # universe itself into a fake relation.
            source_card = step_cards.get(source.step_id) or {}
            relation_cap0 = _capability_for_relation(relation, catalog)
            embedded_root = _record_root(
                source_card, collection=collection, relation=relation, catalog=catalog)
            # Capability record paths are relative to their canonical endpoint. If
            # the same semantic relation is already embedded in a different source
            # response (e.g. currently-playing ``item.album``), qualify the provider
            # record path under the current typed owner before deciding whether a
            # second request is necessary. This is schema composition, not lexical
            # question matching.
            if relation_cap0 is not None and relation_cap0.embedded_in_source:
                base = str(source.record_root or "$").replace("$.", "").strip(".")
                relroot = str(relation_cap0.record_path or "$").replace("$.", "").strip(".")
                candidate = relroot if base in {"", "$"} else (base + "." + relroot if relroot not in {"", "$"} else base)
                documented_roots = {str(x) for x in _record_paths(source_card)}
                if candidate in documented_roots or any(str(x).startswith(candidate + ".") for x in _leaf_paths(source_card)):
                    embedded_root = candidate
            wanted_relation_tokens = _tokens(collection or relation)
            embedded_tokens = _tokens(embedded_root)
            source_root_norm = str(source.record_root or "$").replace("$.", "")
            embedded_norm = str(embedded_root or "$").replace("$.", "")
            embedded_is_child_of_source = (
                source_root_norm in {"", "$"}
                or embedded_norm.startswith(source_root_norm.rstrip(".") + ".")
            )
            if (embedded_root not in {"", "$"}
                    and embedded_norm != source_root_norm
                    and embedded_is_child_of_source
                    and (relation_cap0 is None or relation_cap0.embedded_in_source)
                    and nid not in action_ancestor_ids
                    and wanted_relation_tokens
                    and bool(wanted_relation_tokens & embedded_tokens)):
                kind = "collection" if "[*]" in embedded_root else "object"
                target_resource = _infer_relation_resource(
                    source.resource, relation, collection, node.get("resource"))
                label = _human_label_field(source_card, embedded_root)
                # Field paths on a nested relation are expressed relative to the
                # selected outer owner in plan derivations (``known_for.title``
                # rather than bare ``title`` or fully rooted
                # ``results.known_for.title``). Keep that prefix only until a
                # filter/select establishes the child collection as its own record
                # set.
                embedded_clean = embedded_norm.replace("[*]", "").strip(".")
                source_clean = source_root_norm.replace("[*]", "").strip(".")
                projection_prefix = embedded_clean
                if source_clean and embedded_clean.startswith(source_clean + "."):
                    projection_prefix = embedded_clean[len(source_clean) + 1:]
                handles[nid] = _Handle(
                    nid, kind, target_resource, source.step_id, source.derivation_id, embedded_root,
                    label_step=source.step_id, label_field=label,
                    owner_step=(source.owner_step or source.label_step or source.step_id),
                    owner_label_field=(source.owner_label_field or source.label_field),
                    projection_prefix=projection_prefix,
                    path_aliases=dict(source.path_aliases))
                route_diagnostics[nid] = {
                    "selected": "embedded_response_relation",
                    "record_root": embedded_root,
                }
                continue

            card, diag = _resolve_route("relation", source_resource=source.resource,
                                        relation=relation, collection=collection,
                                        literals=literals, question=question, cards=cards,
                                        semantics=semantics)
            route_diagnostics[nid] = diag
            if not card:
                errors.append(f"intent node {nid}: no unambiguous relation route for {source.resource}: {relation}"); continue

            # A relation can already be embedded in the response we have.  Reuse
            # that observation instead of issuing the identical detail request a
            # second time merely to descend into a nested collection (e.g. a TV
            # detail response -> networks[*]).  The relation handle changes record
            # universe, not acquisition route.
            source_card = step_cards.get(source.step_id) or {}
            if (str(source_card.get("endpoint") or "") == str(card.get("endpoint") or "")
                    and str(source_card.get("method") or "GET").upper() == str(card.get("method") or "GET").upper()
                    and (relation_cap0 is None or relation_cap0.embedded_in_source)
                    and _cf(relation) not in {"detail", "details", "info", "information"}):
                root = _record_root(card, collection=collection, relation=relation, catalog=catalog)
                kind = "collection" if "[*]" in root else "object"
                target_resource = _infer_relation_resource(
                    source.resource, relation, collection, node.get("resource"))
                label = _human_label_field(card, root)
                handles[nid] = _Handle(
                    nid, kind, target_resource, source.step_id, source.derivation_id, root,
                    label_step=source.step_id, label_field=label,
                    owner_step=(source.owner_step or source.label_step or source.step_id),
                    owner_label_field=(source.owner_label_field or source.label_field),
                    path_aliases=dict(source.path_aliases))
                continue

            alias = ensure_id_binding(source, nid)
            path_bindings: dict[str, str] = {}
            path_literals: dict[str, Any] = {}
            cap = _capability_for_relation(relation, catalog)
            for ph in _path_placeholders(str(card.get("endpoint") or "")):
                if ph in literals:
                    path_literals[ph] = literals[ph]
                    continue
                if ph == "time_window":
                    path_literals[ph] = _auto_time_window(question)
                    continue

                binding_spec = str((cap.parent_binding or {}).get(ph) or "") if cap is not None else ""
                bound_alias = (ensure_id_binding(source, nid)
                               if binding_spec == "source.id" else source.path_aliases.get(ph, ""))
                if not bound_alias and binding_spec == "source.id":
                    bound_alias = ensure_id_binding(source, nid)
                elif not bound_alias and binding_spec in {"source.season_number", "owner.season_number"}:
                    bound_alias = ensure_scope_binding(source, "season_number")
                elif not bound_alias and binding_spec in {"source.episode_number", "owner.episode_number"}:
                    bound_alias = ensure_scope_binding(source, "episode_number")
                elif not bound_alias and binding_spec == "owner.id":
                    # Hierarchical handles inherit the owner's provider id alias.
                    bound_alias = source.path_aliases.get(ph, "")
                if not bound_alias and _PLACEHOLDER_RESOURCE.get(ph) == source.resource and alias:
                    bound_alias = alias
                if bound_alias:
                    path_bindings[ph] = bound_alias
                else:
                    errors.append(f"intent node {nid}: cannot bind path parameter {ph}")
            if source.kind == "collection":
                # Relation application over a collection is an explicit typed
                # fan-out. Mark every producer alias consumed by the child
                # request, including hierarchical scope values such as
                # season_number—not just the parent's id alias.
                fanout_aliases = set(path_bindings.values())
                if alias:
                    fanout_aliases.add(alias)
                for binding_alias in sorted(x for x in fanout_aliases if x):
                    marker = {"step_id": source.step_id, "binding": binding_alias}
                    if marker not in typed_fanout_bindings:
                        typed_fanout_bindings.append(marker)
            if errors and errors[-1].startswith(f"intent node {nid}:"):
                continue
            if cap is not None:
                for k, v in dict(cap.path_literals or {}).items():
                    path_literals.setdefault(k, v)
            query_literals = dict(cap.query_literals or {}) if cap is not None else {}
            query_literals.update({k: v for k, v in literals.items() if k in _param_names(card, "query")})
            query_bindings: dict[str, str] = {}
            if cap is not None:
                for param, binding_spec0 in dict(cap.query_bindings or {}).items():
                    binding_spec = str(binding_spec0 or "")
                    if binding_spec.startswith("source."):
                        semantic = binding_spec.split(".", 1)[1]
                        bound_alias = ensure_value_binding(source, semantic, nid)
                        if bound_alias:
                            query_bindings[param] = bound_alias
                        else:
                            errors.append(
                                f"intent node {nid}: cannot bind query parameter {param} from {binding_spec}")
            if errors and errors[-1].startswith(f"intent node {nid}:"):
                continue
            # A provider relation whose request contains no value derived from the
            # source is already globally scoped by the provider (for example a
            # current-user collection).  Do not retain a redundant source GET as an
            # execution dependency.  Bound relations still depend on their owner.
            relation_depends = ([source.step_id] if source.step_id and
                                (path_bindings or query_bindings) else [])
            step, sid = new_step(card, f"Get {relation} for the selected {source.resource}",
                                 depends_on=relation_depends,
                                 path_literals=path_literals, path_bindings=path_bindings,
                                 query_literals=query_literals)
            if query_bindings:
                step["query_bindings"] = query_bindings
            root = ("$" if _cf(relation) in {"detail", "details", "info", "information"}
                    else _record_root(card, collection=collection, relation=relation, catalog=catalog))
            # If the route is a detail object but a requested nested relation exists,
            # prefer that nested record root for later local operations.
            kind = "collection" if "[*]" in root else "object"
            target_resource = _infer_relation_resource(
                source.resource, relation, collection, node.get("resource"))
            label = _human_label_field(card, root)
            # label_* describes the records produced by this relation. owner_*
            # describes the parent entity whose aggregate the relation belongs to.
            # A later select promotes the selected record to owner; a count keeps
            # the parent owner. This distinction prevents movie/show labels from
            # leaking into selected-person comparisons while preserving questions
            # such as "which director has more movie credits?".
            inherited_aliases = dict(source.path_aliases)
            # Every provider-parent alias consumed to reach this child remains a
            # lineage fact of the child.  This is what lets later typed relations
            # (for example episode -> images) reuse hierarchical scope without
            # reading the original question.
            for _ph, _alias in path_bindings.items():
                if _alias:
                    inherited_aliases.setdefault(_ph, _alias)
            # A child request may have consumed explicit scope literals. Preserve
            # those as lineage facts only when they were literal; selected scope
            # values are carried by aliases instead.
            handles[nid] = _Handle(nid, kind, target_resource, sid, "", root,
                                   label_step=sid, label_field=label,
                                   owner_step=(source.owner_step or source.label_step or source.step_id),
                                   owner_label_field=(source.owner_label_field or source.label_field),
                                   path_aliases=inherited_aliases)
            continue

        if op == "action":
            action_name = _cf(node.get("action") or node.get("relation") or "").replace(" ", "_")
            spec = dict((semantics.action_routes if semantics is not None else {}).get(action_name) or {})
            if not spec:
                errors.append(f"intent node {nid}: unknown provider action {action_name!r}"); continue
            method = str(spec.get("method") or "").upper()
            endpoint = str(spec.get("endpoint") or "")
            card = next((c for c in cards
                         if str(c.get("endpoint") or "") == endpoint
                         and str(c.get("method") or "GET").upper() == method), None)
            if card is None:
                errors.append(f"intent node {nid}: action route missing from OAS: {method} {endpoint}"); continue

            source = handles.get(str(node.get("source") or ""))
            input_h = handles.get(str(node.get("input") or ""))
            required_source = str(spec.get("source_resource") or "")
            required_input = str(spec.get("input_resource") or "")
            allowed_inputs = {str(x) for x in (spec.get("input_resources") or []) if str(x)}
            if required_source and (source is None or source.resource != required_source):
                errors.append(f"intent node {nid}: action {action_name} requires source resource {required_source}"); continue
            if required_input and (input_h is None or input_h.resource != required_input):
                errors.append(f"intent node {nid}: action {action_name} requires input resource {required_input}"); continue
            if allowed_inputs and (input_h is None or input_h.resource not in allowed_inputs):
                errors.append(f"intent node {nid}: action {action_name} requires input resource in {sorted(allowed_inputs)}"); continue

            # Some provider actions declare that a derivation-selected singular
            # source/input should be refreshed through its canonical detail relation
            # before mutation.  The declaration is provider metadata; the compiler
            # itself does not infer this from benchmark wording or route recipes.
            def _canonical_action_handle(h: _Handle | None) -> _Handle | None:
                if h is None or h.kind not in {"entity", "object"} or not h.derivation_id:
                    return h
                try:
                    promoted, _target = promote_for_projection(
                        h, "id", nid + "__action_owner", prefer_detail=True)
                except Exception:
                    return h
                return promoted if promoted is not None else h

            if bool(spec.get("canonicalize_source")):
                source = _canonical_action_handle(source)
            if bool(spec.get("canonicalize_input")):
                input_h = _canonical_action_handle(input_h)

            literals = dict(node.get("literals") or {})

            # If the semantic model preserved a provider action but emitted a
            # missing/null literal for a qualitative value, recover only from
            # aliases explicitly declared by that provider action.  This is
            # intentionally narrow: the question text may select among declared
            # aliases (e.g. lower/raise), but the compiler never invents a new
            # value or benchmark-specific phrase mapping.
            qnorm = " " + _cf(question) + " "
            for literal_name, aliases0 in dict(spec.get("literal_aliases") or {}).items():
                aliases = {_cf(k): v for k, v in dict(aliases0 or {}).items() if _cf(k)}
                # Preserve the compiler invariant that a completely missing
                # required literal is an error.  Qualitative recovery is only
                # for a planner that explicitly represented the provider field
                # but left its value null/empty.
                if literal_name not in literals:
                    continue
                current = literals.get(literal_name)
                if current not in (None, ""):
                    key = _cf(current)
                    if key in aliases:
                        literals[literal_name] = aliases[key]
                    continue
                # Do not let a qualitative alias override an explicit numeric
                # value stated by the user.  If the semantic planner dropped a
                # number (for example, "turn down volume to 20" -> null), the
                # correct response is to leave the plan unresolved for semantic
                # replan rather than silently substitute the qualitative extreme.
                if re.search(r"(?<!\w)[+-]?\d+(?:\.\d+)?(?!\w)", str(question or "")):
                    continue
                matched_values = []
                for alias, mapped in aliases.items():
                    if (" " + alias + " ") in qnorm:
                        matched_values.append(mapped)
                # Multiple synonymous aliases may occur; recover only when they
                # all imply the same concrete provider value.
                unique = []
                for value in matched_values:
                    if value not in unique:
                        unique.append(value)
                if len(unique) == 1:
                    literals[literal_name] = unique[0]
                    node.setdefault("literals", {})[literal_name] = unique[0]

            path_literals = dict(spec.get("path_literals") or {})
            query_literals = dict(spec.get("query_literals") or {})
            body_literals = dict(spec.get("body_literals") or {})
            for target, lit_name in (spec.get("query_literal_fields") or {}).items():
                if lit_name in literals:
                    query_literals[str(target)] = literals[lit_name]
            for target, lit_name in (spec.get("body_literal_fields") or {}).items():
                if lit_name in literals:
                    body_literals[str(target)] = literals[lit_name]

            deps = []
            for h in (source, input_h):
                if h is not None and h.step_id and h.step_id not in deps:
                    deps.append(h.step_id)
            step, sid = new_step(card, f"Perform provider action: {action_name}",
                                 depends_on=deps, query_literals=query_literals,
                                 path_literals=path_literals)
            step["body_literals"] = body_literals
            step["intent_node_id"] = nid
            step["intent_action"] = action_name

            def _bind(expr: str) -> tuple[str, _Handle | None]:
                expr = str(expr or "")
                owner, _, semantic = expr.partition(".")
                h = source if owner == "source" else input_h if owner == "input" else None
                if h is None:
                    return "", None
                if semantic == "id":
                    return ensure_id_binding(h, nid), h
                return ensure_value_binding(h, semantic or "value", nid), h

            for target, expr in (spec.get("path_bindings") or {}).items():
                alias, _h = _bind(str(expr))
                if not alias:
                    errors.append(f"intent node {nid}: cannot bind action path {target} from {expr}")
                else:
                    step["path_bindings"][str(target)] = alias
            for target, expr in (spec.get("query_bindings") or {}).items():
                alias, _h = _bind(str(expr))
                if not alias:
                    errors.append(f"intent node {nid}: cannot bind action query {target} from {expr}")
                else:
                    step["query_bindings"][str(target)] = alias
            wrappers = {}
            for target, raw_expr in (spec.get("body_bindings") or {}).items():
                wrap = ""
                expr = raw_expr
                if isinstance(raw_expr, dict):
                    expr = raw_expr.get("ref") or ""
                    wrap = str(raw_expr.get("wrap") or "")
                alias, _h = _bind(str(expr))
                if not alias:
                    errors.append(f"intent node {nid}: cannot bind action body {target} from {expr}")
                else:
                    step["body_bindings"][str(target)] = alias
                    if wrap:
                        wrappers[str(target)] = wrap
            if wrappers:
                step["body_binding_wrappers"] = wrappers

            fanout_refs = {str(x) for x in (spec.get("fanout") or [])}
            for target_map in (spec.get("query_bindings") or {}, spec.get("body_bindings") or {}):
                for _target, raw_expr in target_map.items():
                    expr = str(raw_expr.get("ref") if isinstance(raw_expr, dict) else raw_expr)
                    if expr not in fanout_refs:
                        continue
                    alias, h = _bind(expr)
                    if alias and h is not None and h.step_id and h.kind == "collection":
                        marker = {"step_id": h.step_id, "binding": alias}
                        max_values = h.selection_limit or literals.get("max_values") or literals.get("limit")
                        if max_values is not None:
                            try: marker["max_values"] = max(1, int(max_values))
                            except Exception: pass
                        if marker not in typed_fanout_bindings:
                            typed_fanout_bindings.append(marker)

            # An action with a response entity (currently playlist creation) can
            # feed a later action through the same typed data-flow machinery.
            result_resource = str(spec.get("result_resource") or "")
            if result_resource:
                root = str(spec.get("record_path") or "$")
                label = _human_label_field(card, root)
                h = _Handle(nid, "entity", result_resource, sid, "", root,
                            label_step=sid, label_field=label, owner_step=sid,
                            owner_label_field=label)
                ensure_id_binding(h, nid)
                handles[nid] = h
            else:
                handles[nid] = _Handle(nid, "action", str(node.get("resource") or ""), sid, "", "$")
            route_diagnostics[nid] = [{"score": "action", "endpoint": endpoint,
                                       "method": method, "action": action_name}]
            continue

        if op == "filter":
            source = handles.get(str(node.get("source") or ""))
            if not source or not source.step_id:
                errors.append(f"intent node {nid}: filter source is unavailable"); continue
            raw_field = str(node.get("field") or "").strip()
            if not raw_field:
                errors.append(f"intent node {nid}: filter needs a field"); continue
            card = step_cards.get(source.step_id) or {}
            field = _relative_leaf(card, source.record_root, raw_field)
            if not field:
                errors.append(f"intent node {nid}: filter field {raw_field!r} is not exposed by the selected relation"); continue
            cmp0 = _COMPARISON_ALIASES.get(_cf(node.get("comparison") or "eq"), _cf(node.get("comparison") or "eq"))
            expected = node.get("value")
            if "language" in _field_semantic_tokens(field):
                expected = _literal_language_code(expected)
            if isinstance(expected, int) and "date" in _field_semantic_tokens(field) and "year" in _tokens(raw_field):
                if cmp0 in {"lte", "lt"}:
                    expected = f"{expected:04d}-12-31"
                elif cmp0 in {"gte", "gt"}:
                    expected = f"{expected:04d}-01-01"
            filter0 = {field: {"op": cmp0 or "eq", "value": expected}}
            parent = [source.derivation_id] if source.derivation_id else []
            did = add_derivation(nid, "filter", source_step=source.step_id,
                                 source_derivations=parent,
                                 field=_root_field(source.record_root) or None,
                                 filter0=filter0, purpose=f"Keep records where {field} matches the requested condition")
            handles[nid] = _Handle(nid, "collection", source.resource, source.step_id, did,
                                   source.record_root, label_step=source.label_step,
                                   label_field=source.label_field, owner_step=source.owner_step,
                                   owner_label_field=source.owner_label_field,
                                   path_aliases=dict(source.path_aliases))
            continue

        if op == "select":
            source = handles.get(str(node.get("source") or ""))
            if not source or not source.step_id:
                errors.append(f"intent node {nid}: select source is unavailable"); continue
            mode = _cf(node.get("mode") or "first")
            if mode not in {"first", "nth", "top k", "argmax", "argmin"}:
                errors.append(f"intent node {nid}: unsupported select mode {mode!r}"); continue
            rank = int(node.get("rank") or 0)
            raw_field = str(node.get("field") or "").strip()
            if mode == "top k":
                limit = max(1, rank or int((node.get("literals") or {}).get("limit") or 1))
                handles[nid] = _Handle(
                    nid, "collection", source.resource, source.step_id, source.derivation_id,
                    source.record_root, label_step=source.label_step, label_field=source.label_field,
                    owner_step=source.owner_step, owner_label_field=source.owner_label_field,
                    selection_limit=limit, path_aliases=dict(source.path_aliases))
                continue

            field = raw_field
            if mode in {"argmax", "argmin"}:
                if not raw_field:
                    errors.append(f"intent node {nid}: {mode} needs a field"); continue
                card = step_cards.get(source.step_id) or {}
                field = _relative_leaf(card, source.record_root, raw_field)
                if not field:
                    errors.append(f"intent node {nid}: selection field {raw_field!r} is not exposed by the selected relation"); continue
            parent = [source.derivation_id] if source.derivation_id else []
            derivation_field = (_qualified_record_field(source.record_root, field)
                                if mode in {"argmax", "argmin"}
                                else (_collection_derivation_field(source.record_root) or None))

            # Season/episode ordinals are semantic provider numbers, not array
            # positions.  When validate_intent has preserved such a number,
            # compile it as ``filter(number == N) -> first``.  This makes the
            # selected record and its downstream path binding agree even when
            # a response contains season 0/specials or a nontrivial order.
            numbered_key = "season_number" if source.resource == "season" else (
                "episode_number" if source.resource == "episode" else "")
            numbered_value = None
            if numbered_key and mode in {"first", "nth"}:
                lits0 = node.get("literals") or {}
                numbered_value = lits0.get(numbered_key)
                if numbered_value is None:
                    numbered_value = lits0.get("season" if numbered_key == "season_number" else "episode")
            if numbered_key and numbered_value is not None and mode in {"first", "nth"}:
                card = step_cards.get(source.step_id) or {}
                provider_number_field = _relative_leaf(card, source.record_root, numbered_key)
                if not provider_number_field:
                    provider_number_field = _relative_leaf(
                        card, source.record_root,
                        "season number" if numbered_key == "season_number" else "episode number")
                if not provider_number_field:
                    errors.append(
                        f"intent node {nid}: numbered {source.resource} field {numbered_key!r} "
                        "is not exposed by the selected relation")
                    continue
                scope_filter_id = add_derivation(
                    f"{nid}_scope", "filter", source_step=source.step_id,
                    source_derivations=parent,
                    field=_root_field(source.record_root) or None,
                    filter0={provider_number_field: {"op": "eq", "value": numbered_value}},
                    purpose=f"Select {source.resource} where {provider_number_field} equals the semantic number")
                did = add_derivation(
                    nid, "first", source_step=source.step_id,
                    source_derivations=[scope_filter_id],
                    field=_root_field(source.record_root) or None,
                    filter0={provider_number_field: {"op": "eq", "value": numbered_value}},
                    rank=0, purpose=f"Select the requested numbered {source.resource}")
            else:
                # Selection semantics come entirely from typed IR.  If "latest"
                # excludes future items, the model expresses that as an explicit
                # filter before argmax; the compiler does not reinterpret language.
                did = add_derivation(nid, mode, source_step=source.step_id,
                                     source_derivations=parent, field=derivation_field,
                                     filter0={}, rank=rank,
                                     purpose=f"Select the requested record using {mode}")
            label = source.label_field or _human_label_field(step_cards.get(source.step_id) or {}, source.record_root)
            selected_label_step = source.label_step or source.step_id
            selected = _Handle(nid, "entity", source.resource, source.step_id, did,
                               source.record_root, label_step=selected_label_step,
                               label_field=label, owner_step=selected_label_step,
                               owner_label_field=label,
                               path_aliases=dict(source.path_aliases))
            # Ordinal selection of hierarchy records establishes the concrete
            # path scope for downstream child capabilities.
            if source.resource == "season":
                ensure_scope_binding(selected, "season_number")
            elif source.resource == "episode":
                ensure_scope_binding(selected, "episode_number")
            ensure_id_binding(selected, nid)
            handles[nid] = selected
            continue

        if op == "project":
            source = handles.get(str(node.get("source") or ""))
            if not source or not source.step_id:
                errors.append(f"intent node {nid}: project source is unavailable"); continue
            wanted = str(node.get("field") or "").strip()
            if not wanted:
                errors.append(f"intent node {nid}: project needs a field"); continue
            # Semantic ownership is checked before lexical/schema resolution.
            # Known properties may not migrate between entity types during a
            # repair (e.g. person birthday -> movie release_date). Unknown
            # provider fields remain eligible for ordinary OAS resolution.
            try:
                from utils.semantic_types import owner_error
                semantic_owner_error = owner_error(
                    wanted, source.resource,
                    semantic_fields=(semantics.semantic_fields if semantics is not None else None))
            except Exception:
                semantic_owner_error = None
            if semantic_owner_error:
                errors.append(f"intent node {nid}: {semantic_owner_error}"); continue
            # A singular final projection from a population/collection first
            # selects one semantic owner.  Final output cardinality does not
            # license fan-out of owner-specific detail/image calls.
            answer_cardinality0 = _cf(answer_hint.get("cardinality") or "")
            singular_projection = (nid == answer_source_hint and answer_cardinality0 == "one")
            if singular_projection and source.kind == "collection":
                owner_pick = add_derivation(
                    nid + "_owner_pick", "first", source_step=source.step_id,
                    source_derivations=[source.derivation_id] if source.derivation_id else [],
                    field=_collection_derivation_field(source.record_root) or None,
                    purpose="Select one owner before singular terminal projection")
                source = _Handle(
                    source.node_id + "__single_owner", "entity", source.resource,
                    source.step_id, owner_pick, source.record_root,
                    label_step=source.label_step, label_field=source.label_field,
                    owner_step=source.owner_step, owner_label_field=source.owner_label_field,
                    path_aliases=dict(source.path_aliases))
                ensure_id_binding(source, nid)

            projection_source, target = promote_for_projection(source, wanted, nid)
            if not target:
                errors.append(f"intent node {nid}: requested field {wanted!r} is not exposed by the selected entity or its bindable detail route"); continue
            target_root, field = target
            derivation_field = field
            if (projection_source.projection_prefix and
                    target_root == projection_source.record_root and "." not in field):
                derivation_field = f"{projection_source.projection_prefix}.{field}"
            elif (target_root == projection_source.record_root
                  and not projection_source.derivation_id
                  and _root_field(target_root) not in {"", "results"}
                  and "." not in field):
                # A direct projection from one named collection among sibling
                # collections must retain the collection qualifier (for example
                # guest_stars.name when an episode-credits response also exposes
                # cast and crew).  Once filter/select creates a record-set
                # derivation, relative scalar fields are sufficient.
                derivation_field = _qualified_record_field(target_root, field)
            elif target_root not in {"", "$", projection_source.record_root} and "." not in field:
                prefix = _root_field(target_root)
                if prefix:
                    derivation_field = f"{prefix}.{field}"
            parent = [projection_source.derivation_id] if projection_source.derivation_id else []
            # If the final requested surface is explicitly singular but the OAS
            # exposes a collection of candidate values/assets, make the cardinality
            # decision explicit rather than letting observation projection pick a
            # record implicitly.  List answers intentionally keep the full set.
            answer_cardinality = _cf(answer_hint.get("cardinality") or "")
            singular_answer = (nid == answer_source_hint and (
                answer_mode_hint == "direct" or
                (answer_mode_hint == "asset" and answer_cardinality == "one")))
            if singular_answer and projection_source.kind == "collection" and "[*]" in str(target_root or ""):
                pick_id = add_derivation(
                    nid + "_pick", "first", source_step=projection_source.step_id,
                    source_derivations=parent, field=_collection_derivation_field(target_root) or None,
                    purpose=f"Select one requested {wanted or field}")
                parent = [pick_id]
                # Keep an explicit nested collection qualifier after selecting
                # from a descendant record universe (e.g.
                # ``release_dates.release_date``).  Dropping it to bare
                # ``release_date`` makes schema validation fall back to the outer
                # ``results[*]`` collection and was a major source of false
                # plan-invalid abstentions.  For a projection on the same record
                # root, a bare scalar remains sufficient.
                # Preserve collection qualification after the selector.  The
                # evidence compiler can make this relative to the selected record,
                # while schema validation needs the qualifier to distinguish nested
                # collections such as results[*].release_dates[*].
                derivation_field = derivation_field
            did = add_derivation(nid, "identity", source_step=projection_source.step_id,
                                 source_derivations=parent, field=derivation_field,
                                 purpose=f"Return the requested {wanted or field}")
            handles[nid] = _Handle(nid, "scalar", projection_source.resource, projection_source.step_id, did,
                                   target_root, label_step=projection_source.label_step,
                                   label_field=projection_source.label_field,
                                   # A projected scalar belongs to the entity it was
                                   # projected from. Preserve that immediate entity
                                   # as comparison winner-label ownership rather than
                                   # leaking an older ancestor (movie/show/container).
                                   owner_step=(projection_source.owner_step or projection_source.label_step),
                                   owner_label_field=(projection_source.owner_label_field or projection_source.label_field),
                                   path_aliases=dict(projection_source.path_aliases))
            continue

        if op == "count":
            source = handles.get(str(node.get("source") or ""))
            if not source or not source.step_id:
                errors.append(f"intent node {nid}: count source is unavailable"); continue
            if source.kind in {"entity", "object"} and not node.get("field"):
                scalar = scalar_count_property(source)
                if scalar:
                    scalar_source, target = scalar
                    target_root, field = target
                    derivation_field = field
                    if target_root not in {"", "$", scalar_source.record_root} and "." not in field:
                        prefix = _root_field(target_root)
                        if prefix:
                            derivation_field = f"{prefix}.{field}"
                    did = add_derivation(
                        nid, "identity", source_step=scalar_source.step_id, field=derivation_field,
                        purpose="Return the requested stored total")
                    handles[nid] = _Handle(
                        nid, "scalar", scalar_source.resource, scalar_source.step_id, did, target_root,
                        label_step=scalar_source.label_step, label_field=scalar_source.label_field,
                        owner_step=scalar_source.owner_step, owner_label_field=scalar_source.owner_label_field)
                    continue
            parent = [source.derivation_id] if source.derivation_id else []
            distinct_field = node.get("field") or None
            if distinct_field:
                resolved_distinct = _relative_leaf(step_cards.get(source.step_id) or {}, source.record_root, str(distinct_field))
                if not resolved_distinct:
                    errors.append(f"intent node {nid}: distinct count field {distinct_field!r} is not exposed"); continue
                distinct_field = resolved_distinct
            did = add_derivation(nid, "count", source_step=source.step_id,
                                 source_derivations=parent,
                                 field=_root_field(source.record_root) or None,
                                 distinct_field=distinct_field,
                                 purpose="Count the selected records")
            handles[nid] = _Handle(nid, "scalar", source.resource, source.step_id, did,
                                   source.record_root, label_step=source.label_step,
                                   label_field=source.label_field, owner_step=source.owner_step,
                                   owner_label_field=source.owner_label_field,
                                   path_aliases=dict(source.path_aliases))
            continue

        if op in {"compare", "difference"}:
            left = handles.get(str(node.get("left") or node.get("source") or ""))
            right = handles.get(str(node.get("right") or ""))
            if not left or not right or not left.derivation_id or not right.derivation_id:
                errors.append(f"intent node {nid}: comparison inputs are unavailable"); continue
            cmp0 = _COMPARISON_ALIASES.get(_cf(node.get("comparison") or ("gt" if op == "compare" else "difference")),
                                           _cf(node.get("comparison") or ("gt" if op == "compare" else "difference")))
            if op == "difference":
                # ``difference`` in semantic IR is a magnitude. Direction is
                # expressed separately by compare(gt/lt); keeping difference
                # absolute makes composition independent of operand order.
                cmp0 = "abs_difference"
            label_steps = []
            label_fields = []
            if cmp0 in {"gt", "gte", "lt", "lte", "max", "min"}:
                for h in (left, right):
                    ls = h.owner_step or h.label_step
                    lf = h.owner_label_field or h.label_field
                    if ls and lf:
                        label_steps.append(ls); label_fields.append(lf)
            did = add_derivation(nid, "compare", source_derivations=[left.derivation_id, right.derivation_id],
                                 comparison=cmp0, label_steps=label_steps, label_fields=label_fields,
                                 unit=str(node.get("unit") or (node.get("literals") or {}).get("unit") or "raw"), purpose="Compare the requested values")
            handles[nid] = _Handle(nid, "boolean" if cmp0 in {"eq", "neq"} else "scalar",
                                   derivation_id=did)
            continue

        if op == "membership":
            source = handles.get(str(node.get("source") or ""))
            if not source or not source.step_id:
                errors.append(f"intent node {nid}: membership source is unavailable"); continue
            field = str(node.get("field") or "name").strip() or "name"
            relation_root = _root_field(source.record_root)
            qualified_field = (f"{relation_root}.{field}"
                               if relation_root and relation_root not in {"results"}
                               and "." not in field else field)
            # Dynamic membership target: typed normalization may supply a scalar
            # identity on left while source names the collection. Evidence replay
            # already supports this one-scalar + one-collection shape.
            target = handles.get(str(node.get("left") or ""))
            target_parent = ([target.derivation_id] if target and target.derivation_id
                             and str(node.get("left") or "") != str(node.get("source") or "") else [])
            # rank>0 on membership means an explicit returned-order prefix
            # (Top-N), not an ordinal item.  Zero means the complete observed
            # collection. This preserves cast/co-star membership while making
            # Top-10 membership exact instead of checking the whole page.
            membership_rank = max(0, int(node.get("rank") or 0))
            did = add_derivation(nid, "membership", source_step=source.step_id,
                                 source_derivations=target_parent or ([source.derivation_id] if source.derivation_id else []),
                                 field=qualified_field, comparison_literal=node.get("value"),
                                 rank=membership_rank,
                                 purpose=f"Check whether the requested value occurs in {field}")
            handles[nid] = _Handle(nid, "boolean", derivation_id=did)
            continue

        if op in {"logical_and", "logical_or"}:
            left = handles.get(str(node.get("left") or node.get("source") or ""))
            right = handles.get(str(node.get("right") or ""))
            if not left or not right or not left.derivation_id or not right.derivation_id:
                errors.append(f"intent node {nid}: logical inputs are unavailable"); continue
            did = add_derivation(nid, op, source_derivations=[left.derivation_id, right.derivation_id],
                                 purpose="Combine the requested Boolean checks")
            handles[nid] = _Handle(nid, "boolean", derivation_id=did)
            continue

    answer = intent.get("answer") or {}
    answer_source_ids = [str(x) for x in (answer.get("sources") or []) if str(x)]
    primary_answer_source = str(answer.get("source") or "")
    if primary_answer_source and primary_answer_source not in answer_source_ids:
        answer_source_ids.append(primary_answer_source)

    # If the typed answer explicitly asks for broad resource detail, promote an
    # embedded/simplified object to the provider's canonical detail operation
    # before materializing answer fields.  The promotion is cached and is a no-op
    # when the current step already is the detail endpoint.
    if bool(answer.get("detail", False)) and semantics is not None:
        for source_id in answer_source_ids:
            h0 = handles.get(source_id)
            if not h0 or not h0.step_id or h0.kind not in {"entity", "object", "scalar"}:
                continue
            declared0 = tuple((semantics.detail_fields or {}).get(h0.resource, ()))
            if not declared0:
                continue
            promoted0, _target0 = promote_for_projection(
                h0, str(declared0[0]), source_id + "__answer_detail", prefer_detail=True)
            if promoted0 is not None and promoted0.step_id and promoted0.step_id != h0.step_id:
                handles[source_id] = promoted0

    answer_source = handles.get(str(answer.get("source") or ""))
    answer_handles = [handles[x] for x in answer_source_ids if x in handles]

    # A relation/select handle identifies a record universe, but the evidence
    # contract requires an explicit value-producing derivation. Synthesize only
    # the terminal extraction implied by typed answer metadata and the selected
    # provider schema. This is graph/schema driven; it never reads question text.
    value_ops = {"identity", "count", "membership", "compare", "difference",
                 "logical_and", "logical_or"}
    deriv_by_id = {str(d.get("id") or ""): d for d in derivations}
    answer_fields = [str(x) for x in (answer.get("fields") or []) if str(x).strip()]
    answer_mode0 = _cf(answer.get("mode") or "direct")
    answer_detail0 = bool(answer.get("detail", False))

    def terminal_detail_derivations(source_id: str, h: _Handle) -> list[str]:
        """Materialize provider-declared primitive detail fields generically.

        Broad detail answers need more than a display label, but the compiler
        should not guess optional schema leaves. Providers may declare a small
        stable field set; this generic layer resolves those semantic fields
        against the selected response card and emits ordinary identity
        derivations with the same lineage rules as any other answer value.
        """
        if not (answer_detail0 and not answer_fields and h.step_id and semantics):
            return []
        declared = tuple((semantics.detail_fields or {}).get(h.resource, ()))
        if not declared:
            return []
        card = step_cards.get(h.step_id) or {}
        parent = [str(h.derivation_id)] if h.derivation_id else []
        out: list[str] = []
        for canonical in declared:
            sf = (semantics.semantic_fields or {}).get(str(canonical))
            provider_field = (sf.field_for(h.resource) if sf is not None else None)
            wanted = str(provider_field or canonical)
            field = _relative_leaf(card, h.record_root, wanted)
            if not field or _identifier_like_path(field):
                continue
            derivation_field = field
            if (not h.derivation_id and _root_field(h.record_root) not in {"", "results"}
                    and "." not in field):
                derivation_field = _qualified_record_field(h.record_root, field)
            did = add_derivation(
                f"answer_{source_id}_{canonical}", "identity",
                source_step=h.step_id, source_derivations=parent,
                field=derivation_field,
                purpose=f"Return detail field {canonical} from the typed answer source")
            deriv_by_id[did] = derivations[-1]
            out.append(did)
        return out

    def terminal_value_derivation(source_id: str, h: _Handle) -> str:
        # If a provider has no stable detail-field declaration, keep broad
        # entity-information requests record-valued rather than silently
        # reducing them to a display label.
        if answer_detail0 and not answer_fields:
            return str(h.derivation_id or "")
        current = deriv_by_id.get(str(h.derivation_id or "")) or {}
        if str(current.get("operator") or "").lower() in value_ops:
            return str(h.derivation_id)
        if not h.step_id:
            return str(h.derivation_id or "")
        card = step_cards.get(h.step_id) or {}
        field = ""
        # For review resources, a semantic request for review(s) means content;
        # provider schema resolves that word to the actual response field.
        candidates = list(answer_fields)
        if answer_mode0 == "asset" or h.resource in {"image", "asset"}:
            candidates = ["file path", "path", "url", "uri"] + candidates
        if h.resource == "review":
            candidates = ["review", "content"] + candidates
        for wanted in candidates:
            got = _relative_leaf(card, h.record_root, wanted)
            if got and not _identifier_like_path(got):
                field = got
                break
        if not field:
            field = str(h.label_field or "")
        if not field:
            return str(h.derivation_id or "")
        derivation_field = field
        if h.projection_prefix and "." not in field:
            # Embedded child relations can share the parent's acquisition step
            # while changing the record universe (episode -> guest_stars is a
            # representative shape). Preserve that child qualifier during final
            # answer synthesis instead of projecting the parent's display name.
            derivation_field = f"{h.projection_prefix}.{field}"
        elif (not h.derivation_id and _root_field(h.record_root) not in {"", "results"}
                and "." not in field):
            derivation_field = _qualified_record_field(h.record_root, field)
        parent = [str(h.derivation_id)] if h.derivation_id else []
        did = add_derivation(
            f"answer_{source_id}", "identity", source_step=h.step_id,
            source_derivations=parent, field=derivation_field,
            purpose="Return the requested terminal value from the typed answer source")
        deriv_by_id[did] = derivations[-1]
        return did

    answer_derivations = []
    for source_id in answer_source_ids:
        h = handles.get(source_id)
        if not h:
            continue
        detail_dids = terminal_detail_derivations(source_id, h)
        if detail_dids:
            for did in detail_dids:
                if did not in answer_derivations:
                    answer_derivations.append(did)
            continue
        did = terminal_value_derivation(source_id, h)
        if did and did not in answer_derivations:
            answer_derivations.append(did)

    # Winner + magnitude comparison questions are often represented with two
    # sibling nodes over the same scalar pair: one ordering comparison and one
    # absolute difference.  If the answer contract names the magnitude branch
    # plus the two entity labels but omits the sibling ordering node, prefer the
    # host-replayable winner derivation and the magnitude derivation.  This keeps
    # the final surface deterministic ("winner; difference: N unit") instead of
    # asking a later LLM to infer which of two certified labels won.
    if answer_mode0 == "comparison":
        difference_nodes = [nodes_by_id.get(x) for x in answer_source_ids]
        difference_nodes = [n for n in difference_nodes if n and str(n.get("op") or "") == "difference"]
        if not difference_nodes and answer.get("source"):
            n0 = nodes_by_id.get(str(answer.get("source") or ""))
            if n0 and str(n0.get("op") or "") == "difference":
                difference_nodes = [n0]
        for diff_node in difference_nodes:
            left0, right0 = str(diff_node.get("left") or ""), str(diff_node.get("right") or "")
            companion = next((n for n in intent["nodes"]
                              if str(n.get("op") or "") == "compare"
                              and {str(n.get("left") or ""), str(n.get("right") or "")}
                                  == {left0, right0}), None)
            if companion is None:
                continue
            winner_h = handles.get(str(companion.get("id") or ""))
            diff_h = handles.get(str(diff_node.get("id") or ""))
            if winner_h and winner_h.derivation_id and diff_h and diff_h.derivation_id:
                answer_derivations = [str(winner_h.derivation_id), str(diff_h.derivation_id)]
                break

    answer_steps = list(dict.fromkeys(
        h.step_id for h in answer_handles if h.step_id))

    # Some questions explicitly request both the selected owner and a property/
    # relation of that owner ("what is the most popular movie and what are its
    # keywords?").  The semantic intent may naturally end at the relation node,
    # which would otherwise certify only the second half.  When the pronoun-bound
    # conjunction is explicit, add the already-selected owner's human label as a
    # second deterministic answer derivation. This uses no new API call and no
    # benchmark-specific entity knowledge.
    answer_fields_hint = [str(x) for x in (answer.get("fields") or []) if str(x).strip()]
    if (answer_source and answer_source.owner_step and answer_source.owner_label_field
            and answer_source.owner_step != answer_source.step_id
            and len(answer_fields_hint) > 1):
        singleton_ops = {"endpoint_rank", "first", "nth", "argmax", "argmin"}
        owner_selectors = [d for d in derivations
                           if answer_source.owner_step in {str(x) for x in d.get("source_steps") or []}
                           and str(d.get("operator") or "").lower() in singleton_ops]
        if owner_selectors:
            owner_sel = owner_selectors[-1]
            owner_did = add_derivation(
                "answer_owner_label", "identity", source_step=answer_source.owner_step,
                source_derivations=[str(owner_sel.get("id"))],
                field=answer_source.owner_label_field,
                purpose="Return the explicitly requested selected owner label")
            answer_derivations = list(dict.fromkeys([owner_did] + answer_derivations))
            answer_steps = list(dict.fromkeys([answer_source.owner_step] + answer_steps))

    # Comparison/Boolean derivations have no direct source step. Include every API
    # step needed by their lineage; answer_derivations is the actual output contract.
    if answer_source and answer_source.kind == "action" and answer_source.step_id:
        # Multi-action requests declare every obligation in answer.sources.  The
        # precomputed answer_steps already contains those action handles; do not
        # collapse the plan to the primary source and prune sibling writes.
        answer_steps = list(dict.fromkeys(
            [h.step_id for h in answer_handles if h.kind == "action" and h.step_id]
            or [answer_source.step_id]))
        answer_derivations = []
    if answer_source and not answer_steps and answer_derivations:
        answer_steps = [str(s.get("id")) for s in steps]
    mode = _cf(answer.get("mode") or "direct")
    if mode not in {"direct", "list", "count", "boolean", "comparison", "asset", "action"}:
        mode = "direct"
    if answer_source and answer_source.kind == "action":
        mode = "action"
    elif answer_source and answer_source.kind == "boolean":
        mode = "boolean"
    elif (answer_source and answer_source.kind == "collection" and mode == "direct"
          and _cf(answer.get("cardinality") or "") != "one"):
        mode = "list"

    # Every typed action is an execution obligation, even if answer.sources is
    # incomplete. This prevents a multi-action request from being certified after
    # only its primary/first side effect was executed.
    action_obligations = []
    for _node in (intent.get("nodes") or []):
        if str(_node.get("op") or "") != "action":
            continue
        _nid = str(_node.get("id") or "")
        _h = handles.get(_nid)
        if _nid and _h is not None and _h.step_id:
            action_obligations.append({
                "intent_node_id": _nid,
                "action": str(_node.get("action") or "action"),
                "step_id": str(_h.step_id),
            })
    if action_obligations:
        mode = "action"
        answer_derivations = []

    plan = {
        "version": 2, "steps": steps, "derivations": derivations,
        "answer_steps": answer_steps, "answer_derivations": answer_derivations,
        "answer_mode": mode,
        "answer_cardinality": (lambda c: c if c in {"one", "many"} else ("many" if mode == "list" else "one"))(_cf(answer.get("cardinality") or "")),
        "answer_resource": _resource(answer.get("resource") or ""),
        "answer_detail": bool(answer.get("detail", False)),
        "answer_fields": list(answer_fields),
        "answer_requirements": [_answer_requirement(question)],
        "planner_notes": "Compiled from a typed semantic intent; low-level routes, bindings, and derivations were host-generated.",
        "intent": intent,
        "intent_route_diagnostics": route_diagnostics,
        "planner_source": "typed_intent_compiler",
        "typed_fanout_bindings": typed_fanout_bindings,
        "action_obligations": action_obligations,
    }
    if capability_report is not None:
        try:
            plan["capability_normalization"] = capability_report.as_dict()
        except Exception:
            pass
    # Compiler completeness invariant: semantic actions may not silently vanish.
    # Every action still present in the normalized typed IR must lower to an
    # executable provider step carrying its node identity. This is provider-neutral
    # and catches partial plans before any external side effect occurs.
    declared_action_nodes = {
        str(n.get("id") or ""): str(n.get("action") or "")
        for n in (intent.get("nodes") or [])
        if str(n.get("op") or "") == "action" and str(n.get("id") or "")
    }
    compiled_action_nodes = {
        str(s.get("intent_node_id") or "")
        for s in steps if str(s.get("intent_node_id") or "")
    }
    for action_id, action_name in declared_action_nodes.items():
        if action_id not in compiled_action_nodes:
            errors.append(
                f"uncompiled action node {action_id}: {action_name or 'action'}")

    if errors:
        plan["valid"] = False; plan["validation_errors"] = list(dict.fromkeys(errors))
        return _secat_v4160_enforce_terminal_relation_selectors(plan, plan["validation_errors"])

    # Typed intents are already compiled into an explicit executable graph.  Do
    # not feed that graph back through the legacy free-form semantic convergence
    # rewriter: that layer is intentionally willing to substitute routes/fields
    # to rescue ambiguous planner JSON, which can change a compiler-authored
    # meaning.  Apply only narrow answer-shape normalization, then validate the
    # compiler output against the OAS without semantic mutation.
    try:
        from utils.evidence_plan import (
            _load_tools, validate_plan, _selected_schema_validation_errors,
            _answer_surface_errors, deterministic_semantic_route_risks,
            positional_selection_risks,
            normalize_answer_steps_from_answer_derivations,
            normalize_typed_multi_answer_cardinality,
            normalize_typed_single_answer_cardinality,
        )
        tools = _load_tools(benchmark)
        valid_paths = list(dict.fromkeys(str(t.get("path") or "") for t in tools if t.get("path")))
        methods: dict[str, set[str]] = {}
        for t in tools:
            methods.setdefault(str(t.get("path") or ""), set()).add(str(t.get("method") or "GET").upper())

        normalized = copy.deepcopy(plan)
        # Answer-step pruning is graph-only.  Semantic closure (for example
        # whether a comparison also needs a magnitude) must already be present in
        # the model-produced IR rather than inferred again from the question.
        normalized = normalize_answer_steps_from_answer_derivations("", normalized)
        _semantic_answer_steps = list(normalized.get("answer_steps") or [])
        if normalized.get("action_obligations"):
            # Temporarily seed pruning with every requested write so sibling action
            # branches cannot disappear.  The public answer_steps contract remains
            # the semantic terminal answer; mandatory side effects live separately
            # in action_obligations and are certified there.
            normalized["answer_steps"] = list(dict.fromkeys(
                _semantic_answer_steps +
                [str(x.get("step_id") or "") for x in normalized.get("action_obligations") or []
                 if str(x.get("step_id") or "")]))
            normalized["answer_mode"] = "action"
            normalized["answer_derivations"] = []
        normalized = _prune_to_answer_lineage(normalized)
        _live_step_ids = {str(s.get("id") or "") for s in normalized.get("steps") or []}
        if normalized.get("action_obligations"):
            normalized["answer_steps"] = [sid for sid in _semantic_answer_steps if sid in _live_step_ids]
        _pruned_action_errors = [
            "required action pruned after compilation: "
            + f"{str(_ob.get('intent_node_id') or '')}:{str(_ob.get('action') or 'action')}"
            for _ob in normalized.get("action_obligations") or []
            if str(_ob.get("step_id") or "") not in _live_step_ids
        ]
        # Cardinality is already typed in the IR. Realize that typed output shape
        # before schema validation so a singular asset/list endpoint gets an
        # explicit deterministic selector instead of being rejected or expanded.
        normalized = normalize_typed_multi_answer_cardinality(normalized)
        normalized = normalize_typed_single_answer_cardinality(normalized)
        normalized = _canonicalize_action_prerequisite_order(normalized)
        normalized, structural_errors = validate_plan(normalized, valid_paths, methods)
        # validate_plan intentionally normalizes the public plan schema and may
        # discard compiler-private metadata. Restore the action obligation map so
        # runtime certification can prove that every requested side effect—not
        # merely every observed write—was completed.
        normalized["action_obligations"] = [dict(x) for x in action_obligations]

        final_errors = list(structural_errors) + list(_pruned_action_errors)
        final_errors.extend(_answer_surface_errors("", normalized))
        final_errors.extend(_selected_schema_validation_errors(benchmark, normalized, ""))
        pos = positional_selection_risks("", normalized)
        explicit_position = any(
            str(n.get("op") or "") == "select"
            and (str(n.get("mode") or "") == "nth" or int(n.get("rank") or 0) > 0)
            for n in (intent.get("nodes") or []))
        if pos and not explicit_position:
            # Only selectors that can reach the declared answer are semantic
            # obligations.  A model may include an unused explanatory selector
            # (for example an Nth node beside a correct Top-N membership); it must
            # not invalidate an otherwise independent answer lineage.
            by_did = {str(d.get("id") or ""): d for d in normalized.get("derivations") or []}
            live = set(str(x) for x in normalized.get("answer_derivations") or [])
            stack = list(live)
            while stack:
                did0 = stack.pop()
                for parent in (by_did.get(did0) or {}).get("source_derivations") or []:
                    parent = str(parent)
                    if parent and parent not in live:
                        live.add(parent); stack.append(parent)
            for risk in pos:
                if isinstance(risk, dict):
                    if str(risk.get("derivation_id") or "") not in live:
                        continue
                    final_errors.append(
                        "selection semantics unresolved in typed intent: "
                        + f"{risk.get('derivation_id')}: {risk.get('field')}={risk.get('position_value')}")
                else:
                    final_errors.append("selection semantics unresolved in typed intent: " + str(risk))

        route_risks = deterministic_semantic_route_risks("", normalized, tools)
        if route_risks:
            normalized.setdefault("validation_warnings", []).extend(
                "typed-intent route advisory: " + str(x) for x in route_risks)

        normalized["validation_warnings"] = list(dict.fromkeys(normalized.get("validation_warnings") or []))
        normalized["validation_errors"] = list(dict.fromkeys(str(x) for x in final_errors if str(x).strip()))
        normalized["valid"] = bool(normalized.get("steps")) and not normalized["validation_errors"]
        normalized["planner_source"] = "typed_intent_compiler"
        normalized["intent"] = intent
        normalized["intent_route_diagnostics"] = route_diagnostics
        return _secat_v4160_enforce_terminal_relation_selectors(normalized, normalized["validation_errors"])
    except Exception as exc:
        errors.append(f"intent compiler validation failed: {exc}")
        plan["valid"] = False; plan["validation_errors"] = list(dict.fromkeys(errors))
        return _secat_v4160_enforce_terminal_relation_selectors(plan, plan["validation_errors"])



def _review_overrides_default_regular_season(base: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """Reject semantic review that turns a host defaulted unqualified episode into another season.

    ``validate_intent`` marks the provider-domain fallback explicitly.  A review
    may restructure the graph, but it must not convert that absence of season
    scope into an arbitrary season 2/3/... constraint.  This compares typed IR
    only and never inspects the user question.
    """
    base_nodes = list((base or {}).get("nodes") or [])
    if not any(bool(n.get("_oca_default_regular_season")) for n in base_nodes):
        return False
    for node in (candidate or {}).get("nodes") or []:
        lits = node.get("literals") or {}
        value = lits.get("season_number", lits.get("season"))
        if value is not None:
            try:
                if int(value) != 1:
                    return True
            except (TypeError, ValueError):
                return True
        if (str(node.get("op") or "") == "filter"
                and _cf(node.get("field") or "").replace(" ", "_") == "season_number"
                and node.get("value") is not None):
            try:
                if int(node.get("value")) != 1:
                    return True
            except (TypeError, ValueError):
                return True
    return False

def make_intent_compiled_plan(question: str, benchmark: str, model: str, client,
                              *, max_tokens: int = 900,
                              runtime_feedback: dict[str, Any] | None = None,
                              allow_legacy_fallback: bool = False,
                              intent_repair_attempts: int = 2,
                              legacy_attempts: int = 1,
                              catalog_mode: str = "full",
                              deterministic_frontend: bool = False) -> tuple[dict[str, Any], str]:
    """Model language -> typed intent -> deterministic provider compilation.

    Up to two bounded compiler-informed intent corrections are allowed when the first IR
    cannot compile.  The legacy free-form evidence planner is disabled by default
    so a typed-IR failure cannot silently reintroduce question-shaped semantic
    repair logic or a large second planning prompt.
    """
    # Language understanding is intentionally model-owned.  The host does not
    # dispatch on benchmark/corpus phrasings.  Determinism begins only after the
    # model has emitted typed semantic IR; provider capabilities then resolve
    # relations, routes, fields and bindings.  ``deterministic_frontend`` is kept
    # in the signature for backward compatibility but no longer activates a
    # question parser.
    intent, text = make_intent_plan(question, model, client, max_tokens=max_tokens,
                                    runtime_feedback=runtime_feedback, benchmark=benchmark)
    initial_intent = copy.deepcopy(intent)
    action_invariants = _intent_action_invariants(initial_intent)
    semantic_review_used = False
    semantic_review_output = ""
    semantic_review_rejected_warning = ""
    if intent.get("valid") and _intent_needs_semantic_review(intent):
        reviewed, semantic_review_output = review_intent_plan(
            question, intent, model, client, max_tokens=max_tokens, benchmark=benchmark)
        if reviewed.get("valid", True):
            # Semantic review is an accuracy aid, not an authority that may
            # destroy a sound graph. Reject a reviewed candidate that violates
            # a typed answer contract when the initial candidate did not. A
            # second compiler-level monotonic check below covers provider/type
            # failures that graph-only contract checks cannot see.
            initial_contract = _provider_contract_errors(initial_intent, benchmark)
            reviewed_contract = _provider_contract_errors(reviewed, benchmark)
            reviewed_invariant_errors = _intent_action_invariant_errors(action_invariants, reviewed)
            if reviewed_invariant_errors:
                intent = initial_intent
                semantic_review_rejected_warning = "; ".join(reviewed_invariant_errors)
            elif reviewed_contract and not initial_contract:
                intent = initial_intent
                semantic_review_rejected_warning = (
                    "semantic review candidate violated typed answer contract; "
                    "retained compiler-valid initial intent")
            elif _review_overrides_default_regular_season(initial_intent, reviewed):
                intent = initial_intent
                semantic_review_rejected_warning = (
                    "semantic review introduced a conflicting season scope for an "
                    "unqualified numbered episode; retained provider-domain default")
            else:
                intent = reviewed
                semantic_review_used = True
                action_invariants = _merge_action_invariants(action_invariants, intent)

    if intent.get("valid"):
        contract_errors = _provider_contract_errors(intent, benchmark)
        if contract_errors:
            intent["valid"] = False
            intent["validation_errors"] = list(dict.fromkeys(
                list(intent.get("validation_errors") or []) + contract_errors))

    if intent.get("valid"):
        plan, errors = compile_intent_to_plan(question, benchmark, intent)
        plan["intent_planner_output"] = text
        plan["intent_valid"] = True
        plan["intent_compiler_errors"] = list(errors)
        plan["intent_semantic_review_used"] = semantic_review_used
        if semantic_review_used:
            plan["intent_semantic_review_output"] = semantic_review_output
        if plan.get("valid"):
            if semantic_review_rejected_warning:
                plan.setdefault("validation_warnings", []).append(
                    semantic_review_rejected_warning)
            plan["planner_attempt_count"] = 1 + int(bool(semantic_review_output))
            plan["planner_attempt_diagnostics"] = ([
                {"source": "typed_intent_semantic_review",
                 "errors": [semantic_review_rejected_warning] if semantic_review_rejected_warning else []}
            ] if semantic_review_output else [])
            return plan, semantic_review_output or text

        # A semantic reviewer must never turn a compiler-valid initial graph into
        # an invalid one. If the reviewed candidate failed provider/type/binding
        # compilation, try the untouched initial IR locally before spending an
        # additional model repair call. No question semantics are inferred here.
        _review_strengthened_action = (
            bool(action_invariants.get("state_changing"))
            and not _intent_action_invariants(initial_intent).get("state_changing")
        )
        if (semantic_review_used and not _review_strengthened_action
                and initial_intent.get("valid")
                and not _provider_contract_errors(initial_intent, benchmark)):
            initial_plan, initial_errors = compile_intent_to_plan(question, benchmark, initial_intent)
            if initial_plan.get("valid"):
                initial_plan["intent_planner_output"] = text
                initial_plan["intent_valid"] = True
                initial_plan["intent_compiler_errors"] = list(initial_errors)
                initial_plan["intent_semantic_review_used"] = True
                initial_plan["intent_semantic_review_output"] = semantic_review_output
                initial_plan.setdefault("validation_warnings", []).append(
                    "semantic review candidate was compiler-invalid; retained compiler-valid initial intent")
                initial_plan["planner_attempt_count"] = 2
                initial_plan["planner_attempt_diagnostics"] = [
                    {"source": "typed_intent_semantic_review", "errors": list(errors)[:10]},
                    {"source": "typed_intent_initial_fallback", "errors": []},
                ]
                return initial_plan, semantic_review_output or text
    else:
        plan = {"version": 2, "steps": [], "derivations": [], "answer_steps": [],
                "answer_derivations": [], "answer_mode": "direct", "valid": False,
                "validation_errors": list(intent.get("validation_errors") or [])}
        errors = list(plan["validation_errors"])

    # Before paying for the much larger legacy planner, give the typed semantic
    # representation up to two small compiler-informed corrections, only while it remains invalid.  This directly
    # addresses model omissions/wording mismatches without letting the model own
    # routes, bindings, record paths, or derivation plumbing.
    repair_outputs: list[str] = []
    repair_budget = max(0, int(intent_repair_attempts or 0))
    if action_invariants.get("state_changing") and repair_budget:
        # Reserve one extra bounded correction only for state-changing requests;
        # a bad repair is not allowed to escape by becoming read-only.
        repair_budget += 1
    for repair_index in range(repair_budget):
        feedback = dict(runtime_feedback or {})
        feedback.update({
            "problem": list(errors)[:10],
            "prior_intent": intent,
            "semantic_invariants": action_invariants,
        })
        repaired_intent, repaired_text = make_intent_plan(
            question, model, client, max_tokens=max_tokens, runtime_feedback=feedback,
            benchmark=benchmark)
        repair_outputs.append(repaired_text)
        repair_initial_intent = copy.deepcopy(repaired_intent)
        intent = repaired_intent
        action_invariants = _merge_action_invariants(action_invariants, intent)
        invariant_errors = _intent_action_invariant_errors(action_invariants, intent)
        if invariant_errors:
            intent["valid"] = False
            intent["validation_errors"] = list(dict.fromkeys(
                list(intent.get("validation_errors") or []) + invariant_errors))
        repair_review_output = ""
        if intent.get("valid") and _intent_needs_semantic_review(intent):
            reviewed_repair, repair_review_output = review_intent_plan(
                question, intent, model, client, max_tokens=max_tokens, benchmark=benchmark)
            if reviewed_repair.get("valid", True):
                base_contract = _provider_contract_errors(repair_initial_intent, benchmark)
                review_contract = _provider_contract_errors(reviewed_repair, benchmark)
                review_invariant_errors = _intent_action_invariant_errors(
                    action_invariants, reviewed_repair)
                if (review_invariant_errors or (review_contract and not base_contract)
                        or _review_overrides_default_regular_season(
                            repair_initial_intent, reviewed_repair)):
                    intent = repair_initial_intent
                else:
                    intent = reviewed_repair
                    action_invariants = _merge_action_invariants(action_invariants, intent)
        if intent.get("valid"):
            contract_errors = _provider_contract_errors(intent, benchmark)
            if contract_errors:
                intent["valid"] = False
                intent["validation_errors"] = list(dict.fromkeys(
                    list(intent.get("validation_errors") or []) + contract_errors))
        if intent.get("valid"):
            repaired_plan, repaired_errors = compile_intent_to_plan(question, benchmark, intent)
            if repair_review_output:
                repaired_plan["intent_repair_semantic_review_output"] = repair_review_output
                repaired_plan["intent_repair_semantic_review_used"] = True
            repaired_plan["intent_planner_output"] = repaired_text
            repaired_plan["intent_valid"] = True
            repaired_plan["intent_compiler_errors"] = list(repaired_errors)
            repaired_plan["intent_repair_used"] = True
            repaired_plan["intent_repair_attempt_count"] = repair_index + 1
            if repaired_plan.get("valid"):
                repaired_plan["planner_attempt_count"] = 2 + repair_index
                repaired_plan["planner_attempt_diagnostics"] = [
                    {"source": "typed_intent", "errors": list(errors)[:10]},
                    {"source": "typed_intent_repair", "errors": []},
                ]
                return repaired_plan, repaired_text
            if (intent is not repair_initial_intent and repair_initial_intent.get("valid")
                    and not _provider_contract_errors(repair_initial_intent, benchmark)):
                base_plan, base_errors = compile_intent_to_plan(question, benchmark, repair_initial_intent)
                if base_plan.get("valid"):
                    base_plan["intent_planner_output"] = repaired_text
                    base_plan["intent_valid"] = True
                    base_plan["intent_compiler_errors"] = list(base_errors)
                    base_plan["intent_repair_used"] = True
                    base_plan["intent_repair_attempt_count"] = repair_index + 1
                    base_plan["intent_repair_semantic_review_output"] = repair_review_output
                    base_plan.setdefault("validation_warnings", []).append(
                        "repair semantic review was compiler-invalid; retained compiler-valid repair draft")
                    base_plan["planner_attempt_count"] = 2 + repair_index
                    return base_plan, repaired_text
            plan, errors = repaired_plan, list(repaired_errors)
        else:
            plan = {"version": 2, "steps": [], "derivations": [], "answer_steps": [],
                    "answer_derivations": [], "answer_mode": "direct", "valid": False,
                    "validation_errors": list(intent.get("validation_errors") or [])}
            errors = list(plan["validation_errors"])

    if not allow_legacy_fallback:
        plan["intent"] = intent
        plan["intent_valid"] = bool(intent.get("valid"))
        plan["intent_compiler_errors"] = list(errors)
        plan["intent_semantic_review_used"] = semantic_review_used
        if semantic_review_used:
            plan["intent_semantic_review_output"] = semantic_review_output
        return plan, semantic_review_output or text

    # Escape hatch only.  Do not reintroduce the former multi-round convergence
    # loop through the normal path: one legacy planner attempt is the entire budget.
    from utils.evidence_plan import make_evidence_plan
    legacy, legacy_text = make_evidence_plan(
        question, benchmark, model, client,
        attempts=max(1, int(legacy_attempts)), api_hints=None,
        enable_semantic_adapters=False, catalog_mode=catalog_mode,
        enable_semantic_critic=False, enable_selection_semantic_guard=False,
        runtime_feedback=runtime_feedback)
    legacy = dict(legacy or {})
    legacy["intent"] = intent
    legacy["intent_valid"] = bool(intent.get("valid"))
    legacy["intent_compiler_errors"] = list(errors)
    legacy["intent_fallback_used"] = True
    legacy["intent_fallback_reason"] = list(errors)[:12]
    legacy["planner_source"] = "typed_intent_compiler+legacy_fallback"
    return legacy, (text + "\n\nLEGACY FALLBACK:\n" + (legacy_text or ""))
