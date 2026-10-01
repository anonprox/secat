"""Generic post-compile semantic-preservation guards for OCA.

v4.1.60 adds one narrowly-scoped invariant:
when a relation carries explicit literal constraints and a terminal scalar
projection is compiled as an unconditional ``first`` over that relation's
collection, any relation literals not already consumed by the acquisition
request must remain as filters on that ``first`` derivation.

The guard intentionally does *not* rewrite routes, prompts, planner intents,
answer modes, or downstream binding chains.  It only repairs the exact
terminal-projection shape described above.  Complex multi-hop/downstream cases
are left untouched for later, separately-tested fixes.
"""
from __future__ import annotations

from copy import deepcopy
import os
from typing import Any


_NO_ERRORS = object()


def _canon(value: Any) -> Any:
    """Canonical form used only to compare literal values conservatively."""
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("num", str(value))
    if isinstance(value, float):
        return ("num", str(int(value)) if value.is_integer() else repr(value))
    if value is None:
        return ("none", None)
    if isinstance(value, str):
        return ("str", value)
    return ("complex", repr(value))


def _scalar_literal(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _literal_map(step: dict[str, Any]) -> dict[str, Any]:
    """Collect only direct request literals from a compiled acquisition step."""
    out: dict[str, Any] = {}
    for key in ("path_literals", "query_literals", "body_literals"):
        value = step.get(key)
        if isinstance(value, dict):
            out.update(value)
    return out


def terminal_relation_selector_repairs(plan: Any) -> list[dict[str, Any]]:
    """Describe safe terminal-selector repairs without mutating *plan*.

    A repair is eligible only when all of these are true:
      * intent has relation -> project directly;
      * relation has one or more scalar literals;
      * compiler emitted exactly one empty-filter ``first`` over the relation
        collection;
      * the project's terminal derivation consumes that same ``first``;
      * both derivations read from exactly the same source step(s), so there is
        no downstream identity/path binding to rewrite;
      * at least one relation literal is not already consumed by that source
        request as a path/query/body literal.

    These conditions deliberately exclude multi-hop cases and asset/detail
    routes that depend on a selected owner later in the plan.
    """
    if not isinstance(plan, dict):
        return []
    intent = plan.get("intent")
    if not isinstance(intent, dict):
        return []
    nodes = intent.get("nodes")
    steps = plan.get("steps")
    derivations = plan.get("derivations")
    if not isinstance(nodes, list) or not isinstance(steps, list) or not isinstance(derivations, list):
        return []

    by_node = {str(n.get("id")): n for n in nodes if isinstance(n, dict) and n.get("id") is not None}
    by_step = {str(s.get("id")): s for s in steps if isinstance(s, dict) and s.get("id") is not None}
    repairs: list[dict[str, Any]] = []

    for project in nodes:
        if not isinstance(project, dict) or project.get("op") != "project":
            continue
        source_id = project.get("source")
        relation = by_node.get(str(source_id))
        if not isinstance(relation, dict) or relation.get("op") != "relation":
            continue
        relation_literals = relation.get("literals")
        collection = relation.get("collection")
        if not isinstance(relation_literals, dict) or not relation_literals or not isinstance(collection, str) or not collection:
            continue
        if not all(_scalar_literal(v) for v in relation_literals.values()):
            continue

        owner_candidates = [
            d for d in derivations
            if isinstance(d, dict)
            and d.get("operator") == "first"
            and d.get("field") == collection
            and (not isinstance(d.get("filter"), dict) or not d.get("filter"))
        ]
        if len(owner_candidates) != 1:
            continue
        owner = owner_candidates[0]
        owner_id = str(owner.get("id"))
        owner_steps = [str(x) for x in (owner.get("source_steps") or [])]
        if len(owner_steps) != 1 or owner_steps[0] not in by_step:
            continue

        # Require an identity-like terminal derivation for this project that
        # consumes this exact owner selector and stays on the same acquisition
        # step.  This is what excludes downstream owner-binding routes.
        terminals = []
        for d in derivations:
            if not isinstance(d, dict) or d is owner:
                continue
            sources = [str(x) for x in (d.get("source_derivations") or [])]
            source_steps = [str(x) for x in (d.get("source_steps") or [])]
            if owner_id not in sources or source_steps != owner_steps:
                continue
            if d.get("operator") not in {"identity", "project"}:
                continue
            # Compiler derivation ids consistently encode the project node id;
            # if they do not, field equality is an additional conservative
            # fallback rather than a reason to touch an unrelated derivation.
            project_id = str(project.get("id"))
            project_field = project.get("field")
            if project_id not in str(d.get("id")) and d.get("field") != project_field:
                continue
            terminals.append(d)
        if len(terminals) != 1:
            continue

        request_literals = _literal_map(by_step[owner_steps[0]])
        unconsumed: dict[str, Any] = {}
        for key, value in relation_literals.items():
            if key in request_literals and _canon(request_literals[key]) == _canon(value):
                continue
            unconsumed[str(key)] = value
        if not unconsumed:
            continue

        # Never overwrite or merge with an existing semantic filter.  Existing
        # filtered selections are already explicit compiler behavior and are
        # outside this release's scope.
        new_filter = {k: {"op": "eq", "value": v} for k, v in sorted(unconsumed.items())}
        repairs.append({
            "relation_node_id": str(relation.get("id")),
            "project_node_id": str(project.get("id")),
            "derivation_id": owner_id,
            "source_step_id": owner_steps[0],
            "collection": collection,
            "unconsumed_literals": deepcopy(unconsumed),
            "filter": new_filter,
        })

    return repairs



def has_explicit_terminal_count(intent: Any) -> bool:
    """Return True only when the typed answer directly names a count node.

    This predicate is intentionally syntax-level and benchmark-agnostic. It
    does not infer counts from cardinality, collections, or wording.
    """
    if not isinstance(intent, dict):
        return False
    answer = intent.get("answer")
    nodes = intent.get("nodes")
    if not isinstance(answer, dict) or answer.get("mode") != "count" or not isinstance(nodes, list):
        return False
    source = answer.get("source")
    if source is None:
        return False
    for node in nodes:
        if isinstance(node, dict) and str(node.get("id")) == str(source):
            return node.get("op") == "count"
    return False


def _shadow_literal_key(key: Any) -> bool:
    """Ignore compiler-control metadata rather than domain-specific fields."""
    text = str(key or "")
    return text.startswith("_") or text.endswith("_cardinality")


def semantic_preservation_audit(plan: Any) -> dict[str, Any]:
    """Read-only semantic obligation audit for a compiled plan.

    The audit is diagnostic. It never changes routes, derivations, validity,
    execution eligibility, or answers. Enabled repairs remain separately
    constrained by their own exact predicates.
    """
    report: dict[str, Any] = {
        "version": "v4.1.63",
        "explicit_terminal_count": False,
        "gaps": {
            "terminal_count_compilation": False,
            "multifield_projection": [],
            "unconsumed_relation_literals": [],
        },
    }
    if not isinstance(plan, dict):
        return report
    intent = plan.get("intent")
    if not isinstance(intent, dict):
        return report
    nodes = intent.get("nodes")
    if not isinstance(nodes, list):
        return report

    report["explicit_terminal_count"] = has_explicit_terminal_count(intent)
    derivations = plan.get("derivations")
    if not isinstance(derivations, list):
        derivations = []
    if report["explicit_terminal_count"] and not any(
        isinstance(d, dict) and d.get("operator") == "count" for d in derivations
    ):
        report["gaps"]["terminal_count_compilation"] = True

    multifield = []
    for node in nodes:
        if not isinstance(node, dict) or node.get("op") != "project":
            continue
        fields = node.get("fields")
        if not node.get("field") and isinstance(fields, list) and len(fields) > 1:
            multifield.append(str(node.get("id")))
    report["gaps"]["multifield_projection"] = multifield

    consumed_request: dict[str, list[Any]] = {}
    steps = plan.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            for bucket in ("path_literals", "query_literals", "body_literals"):
                values = step.get(bucket)
                if isinstance(values, dict):
                    for key, value in values.items():
                        consumed_request.setdefault(str(key), []).append(value)

    consumed_filters: dict[str, list[Any]] = {}
    for derivation in derivations:
        if not isinstance(derivation, dict):
            continue
        filt = derivation.get("filter")
        if not isinstance(filt, dict):
            continue
        for key, value in filt.items():
            if isinstance(value, dict) and "value" in value:
                value = value.get("value")
            consumed_filters.setdefault(str(key), []).append(value)

    unconsumed = []
    for node in nodes:
        if not isinstance(node, dict) or node.get("op") != "relation":
            continue
        literals = node.get("literals")
        if not isinstance(literals, dict) or not literals:
            continue
        missing = {}
        for key, value in literals.items():
            if _shadow_literal_key(key):
                continue
            key = str(key)
            req_values = consumed_request.get(key, [])
            filter_values = consumed_filters.get(key, [])
            if any(_canon(v) == _canon(value) for v in req_values):
                continue
            if any(_canon(v) == _canon(value) for v in filter_values):
                continue
            missing[key] = deepcopy(value)
        if missing:
            unconsumed.append({"node_id": str(node.get("id")), "literals": missing})
    report["gaps"]["unconsumed_relation_literals"] = unconsumed
    return report

def enforce_terminal_relation_selectors(plan: Any, validation_errors: Any = _NO_ERRORS) -> Any:
    """Apply only the safe terminal-selector repairs described above.

    No eligible repair -> return the same object with no mutation.  Eligible
    repairs only replace an empty ``filter`` on the matched ``first``
    derivation and add diagnostic metadata on the plan.
    """
    repairs = terminal_relation_selector_repairs(plan)
    derivations = plan.get("derivations") if isinstance(plan, dict) else None

    # The compiler returns both bare plans in helper paths and ``(plan,
    # validation_errors)`` from its public compile path.  The structural
    # wrapper preserves the original return expression syntactically, which
    # means a two-value return arrives here as two positional arguments.
    # Preserve that API shape exactly while applying the guard only to plan.
    applied = []
    if isinstance(derivations, list):
        by_id = {str(d.get("id")): d for d in derivations if isinstance(d, dict) and d.get("id") is not None}
        for repair in repairs:
            der = by_id.get(repair["derivation_id"])
            if not isinstance(der, dict):
                continue
            if isinstance(der.get("filter"), dict) and der.get("filter"):
                continue
            der["filter"] = deepcopy(repair["filter"])
            der["semantic_selector_preserved"] = True
            applied.append(repair)

    if applied and isinstance(plan, dict):
        plan["semantic_preservation_guard"] = {
            "version": "v4.1.60",
            "terminal_relation_selectors": applied,
        }
    if isinstance(plan, dict) and os.getenv("OCA_SEMANTIC_AUDIT", "").strip().upper() in {"1", "YES", "TRUE", "ON"}:
        plan["semantic_preservation_audit"] = semantic_preservation_audit(plan)
    return plan if validation_errors is _NO_ERRORS else (plan, validation_errors)
