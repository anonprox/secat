"""Generic evidence contracts and commit verification for OCA.

This module is API-agnostic.  It derives obligations from the validated plan,
its generic derivation operators, and provenance bindings.  It contains no
benchmark names, task IDs, entity vocabularies, endpoint-name rules, or gold
answers/routes.
"""
from __future__ import annotations

import copy
import re
from typing import Any


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip().casefold()


def classify_task(question: str) -> dict[str, Any]:
    """Question-only coarse output shape; plan structure overrides it later."""
    q = str(question or "").strip()
    low = q.casefold()
    if re.match(r"^(is|are|was|were|do|does|did|has|have|had|can|could|will|would|should)\b", q, re.I):
        kind = "boolean"
    elif re.search(r"\bhow many\b|\bnumber of\b|\bcount\b|\btotal\b", low):
        kind = "count"
    elif re.search(r"\bversus\b|\bcompare\b|\bwhich (?:one )?(?:is|has)\b", low):
        kind = "comparison"
    elif re.search(r"\bsome\b|\blist\b|\ba few\b|\bwhat are\b|\bwhich are\b", low):
        kind = "list"
    else:
        kind = "direct"
    return {
        "version": 3, "question": q, "task_kind": kind, "answer_kind": kind,
        "answer_relation": "plan_declared", "asset_subtype": None,
        "required_value_fields": [], "selected_entity_requested": False,
        "selection_policies": [], "required_slots": ["citations", "answer_values"],
        "certificate_type": "claim_lineage", "trigger": kind,
    }


def augment_contract_with_plan(contract: dict, plan: dict | None) -> dict:
    """Derive strict generic obligations from the validated plan."""
    out = copy.deepcopy(contract or {})
    plan = plan or {}
    mode = str(plan.get("answer_mode") or out.get("answer_kind") or "direct").lower()
    if mode in {"direct", "list", "count", "boolean", "comparison", "asset"}:
        out["answer_kind"] = mode
        out["task_kind"] = mode
    out["answer_steps"] = [str(x) for x in plan.get("answer_steps") or []]
    cardinality = str(plan.get("answer_cardinality") or "").lower()
    # The typed compiler always supplies cardinality.  Preserve legacy/generic
    # hand-built plans that omit it instead of inventing a question- or
    # mode-derived constraint downstream.
    if cardinality in {"one", "many"}:
        out["answer_cardinality"] = cardinality
    else:
        out.pop("answer_cardinality", None)
    out["answer_requirements"] = list(plan.get("answer_requirements") or [])
    out["answer_resource"] = str(plan.get("answer_resource") or "")
    out["answer_detail"] = bool(plan.get("answer_detail", False))
    out["answer_fields"] = list(plan.get("answer_fields") or [])
    required_ops = []
    for d in plan.get("derivations") or []:
        required_ops.append({
            "id": str(d.get("id") or ""),
            "operator": str(d.get("operator") or "").lower(),
            "source_steps": [str(x) for x in d.get("source_steps") or []],
            "source_derivations": [str(x) for x in d.get("source_derivations") or []],
            "label_steps": [str(x) for x in d.get("label_steps") or []],
            "label_fields": [str(x) for x in d.get("label_fields") or []],
            "field": d.get("field"),
            "comparison": d.get("comparison"),
            "comparison_literal": d.get("comparison_literal"),
            "unit": d.get("unit"),
            "distinct_field": d.get("distinct_field"),
            "filter": dict(d.get("filter") or {}),
        })
    out["required_plan_derivations"] = required_ops
    policies = [x["operator"] for x in required_ops if x.get("operator")]
    if plan.get("observation_plan_valid"):
        for spec in plan.get("observation_specs") or []:
            if spec.get("filters"): policies.append("projection_filter")
            if spec.get("sort"): policies.append("projection_sort")
            if spec.get("select"): policies.append("projection_select")
            for agg in spec.get("aggregates") or []:
                policies.append(f"aggregate:{agg.get('op')}")
    out["selection_policies"] = list(dict.fromkeys(policies))
    required = ["citations", "answer_values"]
    if required_ops or policies:
        required.append("derivations")
    # Do not impose one universal derivation operator merely from the output type.
    # A generic API can return a count/boolean/comparison directly. Operator-specific
    # obligations exist only when the validated plan explicitly declared them.
    declared_ops = {str(x.get("operator") or "") for x in required_ops}
    if mode == "count" and "count" in declared_ops:
        required.append("count_derivation")
    if mode == "boolean" and "membership" in declared_ops:
        required.append("boolean_derivation")
    if mode == "comparison" and "compare" in declared_ops:
        required.append("comparison_derivation")
    out["required_slots"] = required
    return out


def contract_prompt_section(contract: dict) -> str:
    return f"""

Return JSON only:
{{
  "final_answer": "minimal answer or empty",
  "cited_observation_ids": ["obs_...", "der_..."],
  "answer_values": [{{"value": "supported value", "observation_id": "obs_... or der_..."}}],
  "derivation_ids": ["der_..."],
  "selection_anchor_id": "obs_... or null",
  "answer_observation_ids": ["obs_... or der_..."]
}}
Use only supported values. Cite the evidence used. Follow the validated selection. If answer_detail is true in the contract, certify several useful primitive fields from the final entity/detail evidence instead of reducing the response to only its name/title. If the evidence is missing or unclear, return an empty answer.
"""


def _obs_fields(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("fields") if isinstance(record, dict) and isinstance(record.get("fields"), dict) else {}


def _primitive_values(value: Any):
    if isinstance(value, (str, int, float, bool)) or value is None:
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _primitive_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _primitive_values(item)


def _value_supported(value: Any, record: dict[str, Any]) -> bool:
    target = _norm(value)
    if not target:
        return False
    surface = record.get("value") if (record.get("type") == "derived" or "operation" in record) else _obs_fields(record)
    for primitive in _primitive_values(surface):
        source = _norm(primitive)
        if source == target:
            return True
        # Deterministic comparison derivations may carry a declared presentation
        # unit (for example numeric 0 with comparison_unit="years").  Accept only
        # the exact scalar+declared-unit surface; this preserves numeric evidence
        # while allowing natural user-facing formatting such as "0 years".
        if (record.get("operation") == "compare" and isinstance(primitive, (int, float))
                and not isinstance(primitive, bool)):
            unit = _norm(record.get("comparison_unit"))
            if unit and unit not in {"raw", "none"}:
                forms = {_norm(f"{primitive} {unit}")}
                if unit.endswith("s"):
                    forms.add(_norm(f"{primitive} {unit[:-1]}"))
                else:
                    forms.add(_norm(f"{primitive} {unit}s"))
                if target in forms:
                    return True
        # Long textual observations may be quoted/excerpted in the answer. Support
        # only a verbatim normalized substring, never a paraphrase. Numeric and
        # boolean values still require exact equality above.
        if isinstance(value, str) and isinstance(primitive, str) and len(target) >= 8 and target in source:
            return True
    return False


def _answer_contains(final_answer: str, value: Any) -> bool:
    answer = _norm(final_answer)
    if isinstance(value, bool):
        return bool(re.search(r"\b(?:yes|true)\b", answer)) if value else bool(re.search(r"\b(?:no|false)\b", answer))
    return _norm(value) in answer


def _normalize_answer_values(response: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for item in (response or {}).get("answer_values") or []:
        if isinstance(item, dict) and item.get("observation_id") and item.get("value") not in (None, ""):
            out.append({"value": item.get("value"), "observation_id": str(item.get("observation_id"))})
    return out


def _answer_observation_ids(response: dict[str, Any], values: list[dict[str, Any]]) -> list[str]:
    ids = [str(x) for x in (response or {}).get("answer_observation_ids") or []]
    ids.extend(str(x.get("observation_id")) for x in values)
    return list(dict.fromkeys(x for x in ids if x))


def _field_path(fields: dict[str, Any], path: str | None):
    value: Any = fields or {}
    for part in [x for x in str(path or "").replace("$.", "").split(".") if x and x != "$"]:
        if not isinstance(value, dict): return None
        value = value.get(part)
    return value


def _binding_path(plan: dict, step_id: str, name: str) -> str | None:
    # Observation-plan bindings are the most execution-specific source.  Typed
    # compiler plans also carry the same authoritative path on the step itself;
    # verification must not lose lineage merely because an observation-plan
    # overlay was not merged back into ``plan``.
    for spec in plan.get("observation_specs") or []:
        if str(spec.get("step_id")) != step_id: continue
        for binding in spec.get("bindings") or []:
            if str(binding.get("name")) == name:
                return str(binding.get("path") or "")
    step = next((x for x in (plan.get("steps") or [])
                 if str(x.get("id") or "") == str(step_id)), {})
    path = (step.get("binding_paths") or {}).get(str(name))
    return str(path) if path not in (None, "") else None


def _binding_value(obs: dict, name: str, plan: dict, step_id: str):
    path = _binding_path(plan, step_id, name)
    fields = _obs_fields(obs)
    value = _field_path(fields, path) if path else None
    return value


def _dependency_distances(plan: dict, step_id: str) -> dict[str, int]:
    steps = {str(x.get("id")): x for x in plan.get("steps") or []}
    out: dict[str, int] = {}; queue = [(step_id, 0)]
    while queue:
        current, dist = queue.pop(0)
        for dep in (steps.get(current) or {}).get("depends_on") or []:
            dep = str(dep); nd = dist + 1
            if dep not in out or nd < out[dep]:
                out[dep] = nd; queue.append((dep, nd))
    return out


def _authorized_binding_values(plan: dict, der_index: dict[str, dict], obs_index: dict[str, dict],
                               answer_step: str, placeholder: str) -> tuple[set[Any], str | None]:
    steps = {str(x.get("id")): x for x in plan.get("steps") or []}
    step = steps.get(str(answer_step)) or {}
    literals = step.get("path_literals") or {}
    if str(placeholder).strip("{}") in literals:
        value = literals.get(str(placeholder).strip("{}"))
        return ({value} if value not in (None, "") else set()), "__literal__"
    binding_name = str((step.get("path_bindings") or {}).get(str(placeholder).strip("{}"))
                       or str(placeholder).strip("{}")).strip("{}")
    dists = _dependency_distances(plan, answer_step)
    producers = []
    for sid, dist in dists.items():
        binds = {str(x).strip("{}") for x in (steps.get(sid) or {}).get("binds") or []}
        if _binding_path(plan, sid, binding_name) is not None:
            binds.add(binding_name)
        if binding_name in binds:
            producers.append((dist, sid))
    if not producers:
        return set(), None
    nearest_dist = min(x[0] for x in producers)
    nearest = [sid for dist, sid in producers if dist == nearest_dist]
    if len(nearest) != 1:
        return set(), None
    producer = nearest[0]
    ds = [d for d in der_index.values() if str(d.get("plan_step_id") or "") == producer]
    selections = [d for d in ds if d.get("selected_obs_id")]
    values: set[Any] = set()
    if selections:
        # Prefer projection replay, then later deterministic selection.
        chosen = max(selections, key=lambda d: (
            2 if str(d.get("policy") or "") == "projection_declared" else 1,
            len(d.get("candidate_obs_ids") or []), str(d.get("obs_id") or "")))
        obs = obs_index.get(str(chosen.get("selected_obs_id"))) or {}
        value = _binding_value(obs, binding_name, plan, producer)
        if value not in (None, ""): values.add(value)
        return values, producer
    # Set-valued producer (e.g., filtered candidates feeding multiple downstream calls).
    set_ds = [d for d in ds if d.get("candidate_obs_ids") or d.get("selected_obs_ids")]
    if set_ds:
        chosen = max(set_ds, key=lambda d: len(d.get("candidate_obs_ids") or d.get("selected_obs_ids") or []))
        ids = chosen.get("selected_obs_ids") or chosen.get("candidate_obs_ids") or []
        for oid in ids:
            value = _binding_value(obs_index.get(str(oid)) or {}, placeholder, plan, producer)
            if value not in (None, ""): values.add(value)
    return values, producer


def _operation_equivalent(required: str, actual: str) -> bool:
    required = str(required or "").lower(); actual = str(actual or "").lower()
    if required == actual: return True
    if required in {"first", "nth", "endpoint_rank"} and actual in {
            "first", "nth", "endpoint_rank", "projection_select", "projection_nth"}:
        return True
    if required == "argmax" and actual in {"argmax", "projection_sort_select"}: return True
    if required == "argmin" and actual in {"argmin", "projection_sort_select"}: return True
    if required == "count" and actual in {"count", "aggregate:count"}: return True
    return False


def verify_contract(contract, response, obs_index, der_index, final_answer, plan=None):
    plan = plan or {}; response = response or {}
    result = {"certificate_accepted": True, "contract_satisfied": True,
              "verification_status": "pass", "missing_slots": [], "checks": {}, "notes": []}
    def fail(slot, note, status="certificate_gap"):
        result["contract_satisfied"] = False
        if slot not in result["missing_slots"]: result["missing_slots"].append(slot)
        result["notes"].append(note)
        if result["verification_status"] != "contradiction" or status == "contradiction":
            result["verification_status"] = status

    cited = [str(x) for x in response.get("cited_observation_ids") or []]
    derivation_ids = [str(x) for x in response.get("derivation_ids") or []]
    values = _normalize_answer_values(response)
    answer_ids = _answer_observation_ids(response, values)
    all_index = {**obs_index, **der_index}
    result["checks"]["citations_exist"] = bool(cited) and all(x in all_index for x in cited)
    if not result["checks"]["citations_exist"]: fail("citations", "certificate cites missing or no evidence")
    result["checks"]["answer_values_present"] = bool(values)
    if not values: fail("answer_values", "no structured answer values")
    unsupported = [x for x in values if x["observation_id"] not in all_index or
                   not _value_supported(x["value"], all_index.get(x["observation_id"], {}))]
    result["checks"]["answer_values_cited"] = not unsupported and all(x["observation_id"] in cited for x in values)
    if unsupported: fail("answer_values", f"unsupported answer values: {unsupported}", "contradiction")
    if any(not _answer_contains(final_answer, x["value"]) for x in values):
        fail("answer_surface", "final answer does not contain every certified answer value", "contradiction")
    if any(x not in all_index for x in answer_ids):
        fail("answer_observation_ids", "answer_observation_ids contain missing evidence")

    answer_steps = {str(x) for x in plan.get("answer_steps") or []}
    wrong_surface = []
    for item in values:
        oid = item["observation_id"]
        if oid in obs_index and answer_steps and str(obs_index[oid].get("plan_step_id") or "") not in answer_steps:
            wrong_surface.append(oid)
        elif oid in der_index:
            d = der_index[oid]
            # Upstream deterministic decisions can support a selected entity only
            # when the final answer is itself that derived value; otherwise raw
            # answer claims remain constrained to answer steps.
            if d.get("plan_step_id") and answer_steps and str(d.get("plan_step_id")) not in answer_steps:
                if oid not in derivation_ids:
                    wrong_surface.append(oid)
    result["checks"]["answer_step_lineage"] = not wrong_surface
    if wrong_surface: fail("answer_lineage", f"answer values outside planned answer steps: {wrong_surface}", "contradiction")

    missing_required = []
    cited_ds = [der_index[x] for x in derivation_ids if x in der_index]
    empty_answer_steps = {
        str(der_index[x["observation_id"]].get("plan_step_id") or "")
        for x in values
        if x.get("observation_id") in der_index
        and str(der_index[x["observation_id"]].get("operation") or "") == "empty_result"
    }
    empty_vacuous_ops = {"filter", "identity", "endpoint_rank", "first", "nth", "argmax", "argmin"}
    for req in contract.get("required_plan_derivations") or []:
        req_id = str(req.get("id") or "")
        if req_id:
            # Host compiler records the originating plan-derivation id.  When the
            # planner supplied one, require that exact replay rather than accepting
            # a merely similar operation (e.g. first in place of nth, or a generic
            # projection selection in place of the declared argmax).
            satisfied = any(
                str(d.get("plan_derivation_id") or "") == req_id and
                str(d.get("operation") or "").lower() == str(req.get("operator") or "").lower()
                for d in cited_ds)
        else:
            satisfied = any(
                _operation_equivalent(req.get("operator"), d.get("operation")) and
                (not req.get("source_steps") or
                 str(d.get("plan_step_id") or "") in set(req.get("source_steps") or []))
                for d in cited_ds)
        if not satisfied:
            # A successfully fetched empty terminal collection makes record-level
            # selection/extraction derivations on that same answer step vacuously
            # inapplicable.  The empty_result derivation is itself authoritative
            # evidence; upstream entity-selection/request-lineage obligations are
            # still required and are never waived here.
            req_steps = {str(x) for x in (req.get("source_steps") or [])}
            if (empty_answer_steps and req_steps and req_steps.issubset(empty_answer_steps)
                    and str(req.get("operator") or "").lower() in empty_vacuous_ops):
                continue
            # A deterministic answer derivation may be named directly in answer_values.
            answer_ds = [der_index[x["observation_id"]] for x in values if x["observation_id"] in der_index]
            if req_id:
                answer_ok = any(
                    str(d.get("plan_derivation_id") or "") == req_id and
                    str(d.get("operation") or "").lower() == str(req.get("operator") or "").lower()
                    for d in answer_ds)
            else:
                answer_ok = any(_operation_equivalent(req.get("operator"), d.get("operation"))
                                for d in answer_ds)
            if not answer_ok:
                missing_required.append(req.get("id") or req.get("operator"))
    result["checks"]["required_derivations"] = not missing_required
    if missing_required: fail("derivations", f"required plan derivations not certified: {missing_required}")

    # Generic terminal binding ownership.
    ownership_errors = []
    steps = {str(x.get("id")): x for x in plan.get("steps") or []}
    for answer_step in answer_steps:
        template = str((steps.get(answer_step) or {}).get("endpoint") or "")
        placeholders = re.findall(r"\{([^{}]+)\}", template)
        if not placeholders: continue
        answer_obs = [obs_index[oid] for oid in answer_ids if oid in obs_index and
                      str(obs_index[oid].get("plan_step_id") or "") == answer_step]
        for name in placeholders:
            authorized, producer = _authorized_binding_values(plan, der_index, obs_index, answer_step, name)
            if not authorized:
                # If there is no deterministic producer evidence, do not guess.
                ownership_errors.append((answer_step, name, "unresolved_producer"))
                continue
            for obs in answer_obs:
                actual = (obs.get("source_bindings") or {}).get(name)
                if actual not in authorized:
                    ownership_errors.append((answer_step, name, actual, sorted(map(str, authorized)), producer))
    result["checks"]["terminal_entity_ownership"] = not ownership_errors
    if ownership_errors:
        fail("answer_lineage", f"downstream binding not authorized by parent selection: {ownership_errors}", "contradiction")

    mode = str(contract.get("answer_kind") or "direct")
    ops = [str(d.get("operation") or "") for d in cited_ds] + [
        str(der_index[x["observation_id"]].get("operation") or "")
        for x in values if x["observation_id"] in der_index]
    required_operators = {str(x.get("operator") or "") for x in contract.get("required_plan_derivations") or []}
    if mode == "count" and "count" in required_operators and not any(x in {"count", "aggregate:count"} for x in ops):
        fail("count_derivation", "declared count answer lacks replayable count")
    if mode == "boolean":
        # A yes/no task must certify the truth value itself.  Citing a related
        # entity name (or any other grounded scalar) is not a Boolean conclusion.
        bool_values = [x for x in values if isinstance(x.get("value"), bool)]
        bool_derivations = [
            d for d in cited_ds
            if isinstance(d.get("value"), bool) and
            str(d.get("operation") or "") in {"membership", "compare", "logical_and", "logical_or"}
        ]
        result["checks"]["boolean_conclusion"] = bool(bool_values or bool_derivations)
        if not result["checks"]["boolean_conclusion"]:
            fail("boolean_conclusion", "boolean answer lacks a supported true/false conclusion", "contradiction")
    if mode == "boolean" and "membership" in required_operators and "membership" not in ops:
        fail("boolean_derivation", "declared membership answer lacks replayable membership")
    if mode == "comparison" and "compare" in required_operators and "compare" not in ops:
        fail("comparison_derivation", "declared comparison answer lacks replayable comparison")
    if mode == "comparison":
        ordering_modes = {"max", "min", "gt", "gte", "lt", "lte"}
        ordering = [d for d in cited_ds
                    if str(d.get("operation") or "") == "compare"
                    and str(d.get("comparison_mode") or
                            ((d.get("value") or {}).get("comparison")
                             if isinstance(d.get("value"), dict) else "")) in ordering_modes]
        structured = [d for d in ordering
                      if isinstance(d.get("value"), dict)
                      and ((d.get("value") or {}).get("winner_label") not in (None, "")
                           or bool((d.get("value") or {}).get("tie")))]
        if ordering and not structured:
            fail("comparison_winner_lineage",
                 "ordering comparison lacks a host-replayable winner-label mapping",
                 "contradiction")
        for d in structured:
            payload = d.get("value") or {}
            winner_label = payload.get("winner_label")
            tie_label = payload.get("tie_label") or ("Tie" if payload.get("tie") else None)
            expected_label = tie_label if payload.get("tie") else winner_label
            # A user-facing source/entity label must be tied to the host-replayed
            # ordering winner (or to the explicit deterministic tie outcome).
            # Otherwise citing the comparison plus the losing label would still
            # look grounded while expressing the wrong winner.
            non_numeric_values = [x.get("value") for x in values
                                  if not isinstance(x.get("value"), (int, float, bool))]
            if non_numeric_values and not any(
                    _norm(v) == _norm(expected_label) for v in non_numeric_values):
                fail("comparison_winner_lineage",
                     "comparison answer label does not match the host-replayed winner/tie",
                     "contradiction")
    # A certificate with grounded values is not "accepted" when any enforced
    # contract check found a contradiction or evidence gap.  Earlier builds could
    # report certificate_accepted=True alongside contract_satisfied=False, which
    # inflated apparent coverage and obscured semantic failures.
    result["certificate_accepted"] = bool(
        result.get("contract_satisfied") and
        result["checks"].get("citations_exist", False) and bool(values))
    return result


def _find_exact_support(value: Any, obs_index, der_index):
    for oid, record in {**der_index, **obs_index}.items():
        if _value_supported(value, record): return oid
    return None


def repair_certificate(contract, response, obs_index, der_index, final_answer, plan=None):
    """Repair certificate structure from host-generated evidence only.

    This function never changes an answer value.  In addition to fixing stale
    observation ids, it may attach deterministic derivations that the validated
    plan requires when those derivations already exist in the host-side compiler.
    That avoids asking Phase B to rediscover provenance formatting while keeping
    factual content immutable.
    """
    del final_answer
    out = copy.deepcopy(response or {})
    values = _normalize_answer_values(out)
    repaired = []
    all_index = {**obs_index, **der_index}
    for item in values:
        oid = item["observation_id"]
        record = all_index.get(oid) or {}
        # A model can cite an existing but wrong derivation id for an otherwise
        # exact ledger value (common for list answers where several items share one
        # nearby selection derivation). Repair provenance whenever the cited record
        # does not actually support the immutable value, not only when the id is
        # missing. The host may remap only to exact existing evidence.
        if oid not in all_index or not _value_supported(item["value"], record):
            oid = _find_exact_support(item["value"], obs_index, der_index) or oid
        repaired.append({"value": item["value"], "observation_id": oid})
    out["answer_values"] = repaired

    dids = [str(x) for x in out.get("derivation_ids") or [] if str(x) in der_index]
    for item in repaired:
        if item["observation_id"] in der_index and item["observation_id"] not in dids:
            dids.append(item["observation_id"])

    # Deterministically attach already-computed derivations required by the plan.
    # Match on generic operator equivalence and declared source step.  If several
    # derivations qualify, prefer the one with a selected output/candidate set and
    # then the newest stable observation id; no question/domain semantics are used.
    for req in (contract or {}).get("required_plan_derivations") or []:
        source_steps = set(str(x) for x in req.get("source_steps") or [])
        candidates = []
        for did, derivation in der_index.items():
            exact_id = bool(str(req.get("id") or "")) and str(derivation.get("plan_derivation_id") or "") == str(req.get("id"))
            if not exact_id and not _operation_equivalent(req.get("operator"), derivation.get("operation")):
                continue
            if not exact_id and source_steps and str(derivation.get("plan_step_id") or "") not in source_steps:
                continue
            score = (
                2 if exact_id else 0,
                1 if derivation.get("selected_obs_id") else 0,
                len(derivation.get("selected_obs_ids") or derivation.get("candidate_obs_ids") or []),
                str(did),
            )
            candidates.append((score, str(did)))
        if candidates:
            chosen = max(candidates)[1]
            if chosen not in dids:
                dids.append(chosen)
    out["derivation_ids"] = dids

    cited = [str(x) for x in out.get("cited_observation_ids") or []]
    cited.extend(x["observation_id"] for x in repaired)
    cited.extend(dids)
    out["cited_observation_ids"] = list(dict.fromkeys(
        x for x in cited if x in obs_index or x in der_index))
    answer_ids = [str(x) for x in out.get("answer_observation_ids") or []]
    answer_ids.extend(x["observation_id"] for x in repaired)
    out["answer_observation_ids"] = list(dict.fromkeys(answer_ids))
    return out

def _clip_verbatim_text(value: str, limit: int = 360) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    # Return an exact prefix with no synthetic ellipsis so provenance remains a
    # verbatim substring of the observed value.
    return text[:limit].rstrip()


def _render_projection_list(value: Any, did: str):
    if not isinstance(value, list) or not value:
        return None, []
    rows = []
    answer_values = []
    # The validated observation plan already caps selection cardinality (<=20).
    # Do not silently apply a second answer-level truncation.
    for item in value:
        if isinstance(item, dict):
            parts = []
            for key, raw in item.items():
                if raw in (None, "", [], {}):
                    continue
                if isinstance(raw, str):
                    shown = _clip_verbatim_text(raw)
                elif isinstance(raw, (int, float, bool)):
                    shown = raw
                elif isinstance(raw, list) and all(isinstance(x, (str, int, float, bool)) for x in raw[:10]):
                    shown = raw[:10]
                else:
                    continue
                parts.append(f"{key}: {shown}")
                if isinstance(shown, list):
                    for v in shown:
                        answer_values.append({"value": v, "observation_id": did})
                else:
                    answer_values.append({"value": shown, "observation_id": did})
            if parts:
                rows.append(" | ".join(parts))
        elif isinstance(item, (str, int, float, bool)):
            shown = _clip_verbatim_text(item) if isinstance(item, str) else item
            rows.append(str(shown)); answer_values.append({"value": shown, "observation_id": did})
    if not rows:
        return None, []
    return "\n".join(f"- {row}" for row in rows), answer_values


def _selected_record_certificate_fields(plan: dict | None, derivation: dict[str, Any],
                                        contract: dict | None = None) -> list[tuple[str, Any]]:
    """Choose the user's requested fields from a deterministically selected record.

    Ranking metrics, filter discriminators, binding identifiers, and other projected
    fields are evidence *support* by default, not answer values.  They become answer
    fields only when the question explicitly asks for them.  When no field is named,
    generic display fields (name/title/label) are preferred.  This keeps deterministic
    fast-path answers minimal without any benchmark/domain vocabulary.
    """
    value = derivation.get("value")
    if not isinstance(value, dict):
        return []
    sid = str(derivation.get("plan_step_id") or "")
    spec = next((x for x in (plan or {}).get("observation_specs") or []
                 if str(x.get("step_id") or "") == sid), {})
    projected = [str(x) for x in spec.get("project_paths") or [] if str(x)]
    bindings = {str(x.get("path")) for x in spec.get("bindings") or [] if x.get("path")}
    filters = {str(x.get("path")) for x in spec.get("filters") or [] if x.get("path")}
    comparisons = {str(x) for x in derivation.get("comparison_fields") or [] if str(x)}
    question = _norm((contract or {}).get("question") or "")
    requirements = " ".join(_norm(x) for x in (contract or {}).get("answer_requirements") or [])
    request_text = (question + " " + requirements).strip()
    mode = str((contract or {}).get("answer_kind") or (plan or {}).get("answer_mode") or "direct").lower()

    def get(path):
        if path in value:
            return value.get(path)
        leaf = path.split(".")[-1]
        return value.get(leaf)

    def leaf_phrase(path):
        return re.sub(r"[_\-]+", " ", path.split(".")[-1]).strip().casefold()

    def explicitly_requested(path):
        phrase = leaf_phrase(path)
        if not phrase or not request_text:
            return False
        # Whole-word/phrase matching prevents short fields such as `id` from
        # matching unrelated words like `did`.
        return bool(re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", request_text))

    primitive = []
    for path in projected:
        v = get(path)
        if v in (None, "", [], {}) or not isinstance(v, (str, int, float, bool)):
            continue
        primitive.append((path, v))
    # A comparison metric may have survived only in the selected derived record.
    for path in comparisons:
        if any(p == path for p, _ in primitive):
            continue
        v = get(path)
        if v not in (None, "", [], {}) and isinstance(v, (str, int, float, bool)):
            primitive.append((path, v))
    if not primitive:
        return []

    requested = [(p, v) for p, v in primitive if explicitly_requested(p)]
    if requested:
        return requested

    def identifier_like(path):
        leaf = path.split(".")[-1].casefold()
        return leaf == "id" or leaf.endswith("_id") or leaf in {"identifier", "key"}

    support = bindings | filters | comparisons
    answerable = [(p, v) for p, v in primitive if p not in support and not identifier_like(p)]

    # Broad information/details requests intentionally retain multiple
    # descriptive scalars from the selected terminal entity. Identifiers and
    # fields used only for routing/ranking remain support, not answer content.
    if bool((contract or {}).get("answer_detail")) and answerable:
        return answerable

    # Generic human-readable identity fields are the best default answer surface
    # for a selected entity/record. `character`, `role`, metrics, IDs, etc. stay
    # supporting evidence unless the question explicitly names them.
    display = []
    for p, v in answerable:
        leaf = p.split(".")[-1].casefold()
        if leaf in {"name", "title", "label", "display_name"} or leaf.endswith("_name") or leaf.endswith("_title"):
            display.append((p, v))
    if display:
        return display

    if mode == "asset":
        asset = []
        for p, v in answerable:
            leaf = p.split(".")[-1].casefold()
            if any(token in leaf for token in ("url", "uri", "path", "file", "image", "poster", "cover", "logo")):
                asset.append((p, v))
        if asset:
            return asset

    # If no obvious display/asset field exists, return remaining non-support
    # projected scalars. If everything is support-only, decline deterministic
    # materialization and let the evidence-aware answer path decide.
    return answerable


def certificate_from_compiled(contract: dict, compiled: dict | None,
                              obs_index: dict[str, dict], der_index: dict[str, dict],
                              plan: dict | None = None):
    """Deterministically emit outputs whose values are fully materialized by the compiler."""
    del obs_index
    compiled = compiled or {}
    ids = [str(x) for x in compiled.get("answer_derivation_ids") or [] if str(x) in der_index]
    if not ids:
        return None, None
    mode = str(contract.get("answer_kind") or "direct")
    all_dids = [str(x) for x in compiled.get("derivation_ids") or [] if str(x) in der_index]
    cited = list(dict.fromkeys(all_dids + ids))

    if mode == "boolean":
        did = ids[-1]; value = bool(der_index[did].get("value")); answer = "Yes" if value else "No"
        values = [{"value": value, "observation_id": did}]
    elif mode == "count":
        did = ids[-1]; value = der_index[did].get("value"); answer = str(value)
        values = [{"value": value, "observation_id": did}]
    elif mode == "comparison":
        values = []
        texts = []
        for did in ids:
            value = der_index[did].get("value")
            if isinstance(value, dict):
                # max/min can be emitted deterministically only when the planner
                # supplied an explicit host-replayable winner label mapping.
                label = value.get("winner_label")
                if label in (None, "") and value.get("tie"):
                    label = value.get("tie_label") or "Tie"
                if label in (None, ""):
                    return None, None
                texts.append(str(label))
                values.append({"value": label, "observation_id": did})
            elif isinstance(value, list):
                texts.extend(str(x) for x in value)
                values.append({"value": value, "observation_id": did})
            elif isinstance(value, bool):
                # A comparison-mode answer that asks which source wins cannot be
                # safely materialized as a bare True/False. Boolean-only questions
                # use answer_kind=boolean; comparison mode must either carry a
                # structured winner label or fall back to Phase B/verification.
                return None, None
            else:
                comparison_mode = str(der_index[did].get("comparison_mode") or "").lower()
                comparison_unit = str(der_index[did].get("comparison_unit") or "raw").lower()
                if comparison_mode in {"difference", "abs_difference"}:
                    suffix = "" if comparison_unit in {"", "raw"} else " " + comparison_unit
                    texts.append(f"difference: {value}{suffix}")
                else:
                    texts.append(str(value))
                values.append({"value": value, "observation_id": did})
        answer = "; ".join(dict.fromkeys(x for x in texts if x))
        if not answer:
            return None, None
    elif mode in {"direct", "list", "asset"} and all(
            str(der_index[x].get("operation") or "") == "identity" for x in ids):
        values = []
        texts = []
        for did in ids:
            value = der_index[did].get("value")
            if value in (None, "", [], {}) or isinstance(value, dict):
                return None, None
            if isinstance(value, list):
                for item in value:
                    if item in (None, "", [], {}) or isinstance(item, (dict, list)):
                        continue
                    values.append({"value": item, "observation_id": did})
                    texts.append(str(item))
            else:
                values.append({"value": value, "observation_id": did})
                texts.append(str(value))
        if not values:
            return None, None
        answer = "; ".join(dict.fromkeys(texts))
        # When the compiler explicitly kept a selected parent label alongside a
        # requested child relation, render the relation rather than flattening
        # both value sets into an unlabeled list.  This is generic (not task
        # specific) and improves surfaces such as "Movie — keywords: ...".
        if mode == "list" and plan and len(ids) >= 2:
            by_id = {str(d.get("id") or ""): d for d in (plan.get("derivations") or [])}
            def _plan_derivation(did):
                return str((der_index.get(did) or {}).get("plan_derivation_id") or did)
            owner_id = next((did for did in ids if
                             "answer_owner_label" in _plan_derivation(did) or
                             "selected owner label" in str((by_id.get(_plan_derivation(did)) or {}).get("purpose") or "").casefold()), None)
            child_ids = [did for did in ids if did != owner_id]
            if owner_id and len(child_ids) == 1:
                owner_value = der_index[owner_id].get("value")
                child_value = der_index[child_ids[0]].get("value")
                if owner_value not in (None, "", [], {}) and child_value not in (None, "", [], {}):
                    child_plan_id = _plan_derivation(child_ids[0])
                    child_field = str((by_id.get(child_plan_id) or {}).get("field") or
                                      ((der_index.get(child_ids[0]) or {}).get("comparison_fields") or ["values"])[0])
                    child_label = child_field.split(".")[0].replace("_", " ") or "values"
                    child_texts = [str(x) for x in (child_value if isinstance(child_value, list) else [child_value])
                                   if x not in (None, "", [], {}) and not isinstance(x, (dict, list))]
                    if child_texts:
                        answer = f"{owner_value} — {child_label}: " + ", ".join(dict.fromkeys(child_texts))
    elif mode in {"direct", "asset"} and len(ids) == 1 and str(der_index[ids[0]].get("operation") or "") in {
            "argmax", "argmin", "endpoint_rank", "first", "nth",
            "projection_select", "projection_sort_select", "projection_nth"}:
        did = ids[0]
        fields = _selected_record_certificate_fields(plan, der_index[did], contract)
        if not fields:
            return None, None
        answer = "; ".join(f"{path}: {value}" for path, value in fields)
        values = [{"value": value, "observation_id": did} for _, value in fields]
    elif mode == "list" and len(ids) == 1 and str(der_index[ids[0]].get("operation") or "") in {
            "projection_list", "projection_select", "projection_nth", "projection_sort_select",
            "endpoint_rank", "first", "nth"}:
        did = ids[0]
        raw = der_index[did].get("value")
        rows = raw if isinstance(raw, list) else [raw]
        answer, values = _render_projection_list(rows, did)
        if not answer or not values:
            return None, None
    elif any(str(der_index[x].get("operation") or "") == "empty_result" for x in ids):
        # An empty collection can answer an absence/list question, but it cannot
        # satisfy a request for an actual image/logo/file asset. Keep the gap open
        # so the accuracy-first recovery loop can try another owner/route.
        if mode == "asset":
            return None, None
        did = next(x for x in ids if str(der_index[x].get("operation") or "") == "empty_result")
        value = der_index[did].get("value"); answer = str(value)
        values = [{"value": value, "observation_id": did}]
    else:
        return None, None

    # Final-output cardinality is a typed contract property.  Do not infer it
    # again from the raw question or from provider collection size.  For ordinary
    # direct/list/asset surfaces, a singular contract emits exactly one final
    # value even when the evidence endpoint exposes many candidates. Comparison
    # certificates intentionally retain both winner and magnitude values.
    cardinality = str((contract or {}).get("answer_cardinality") or "").lower()
    composite_single = bool(
        (contract or {}).get("answer_detail") or
        (mode == "list" and plan
         and len(list((plan or {}).get("answer_derivations") or [])) >= 2
         and len(list((((plan or {}).get("intent") or {}).get("answer") or {}).get("fields") or [])) >= 2)
    )
    if cardinality == "one" and mode in {"direct", "list", "asset"} and values and not composite_single:
        first = values[0]
        values = [first]
        answer = str(first.get("value"))

    cert = {
        "cited_observation_ids": cited,
        "answer_values": values,
        "derivation_ids": cited,
        "selection_anchor_id": next((der_index[x].get("selected_obs_id") for x in reversed(ids)
                                     if der_index[x].get("selected_obs_id")), None),
        "answer_observation_ids": list(dict.fromkeys(v["observation_id"] for v in values)),
    }
    return answer, cert


def materialize_certified_answer(contract: dict, response: dict, der_index: dict[str, dict],
                                fallback: str = "", obs_index: dict[str, dict] | None = None,
                                plan: dict | None = None) -> str:
    """Render only certificate-locked values; never preserve unsupported prose.

    Phase B is an evidence selector, not an unconstrained final writer. Once its
    certificate passes, the host reconstructs the user-facing surface from those
    exact values. This removes the need for a second semantic LLM judge and makes
    unsupported extra prose impossible to emit.
    """
    del obs_index, fallback
    normalized_values = _normalize_answer_values(response or {})
    values = [x.get("value") for x in normalized_values]
    if not values:
        return ""
    mode = str((contract or {}).get("answer_kind") or "direct").lower()
    cardinality = str((contract or {}).get("answer_cardinality") or "").lower()
    composite_single = bool(
        (contract or {}).get("answer_detail") or
        (mode == "list" and plan
         and len(list((plan or {}).get("answer_derivations") or [])) >= 2
         and len(list((((plan or {}).get("intent") or {}).get("answer") or {}).get("fields") or [])) >= 2)
    )
    if cardinality == "one" and mode in {"direct", "list", "asset"} and values and not composite_single:
        values = values[:1]
    if mode == "boolean" and len(values) == 1 and isinstance(values[0], bool):
        return "Yes" if values[0] else "No"

    if bool((contract or {}).get("answer_detail")) and normalized_values:
        plan_derivations = {str(x.get("id") or ""): x for x in (plan or {}).get("derivations") or []}
        labeled = []
        for item in normalized_values:
            value = item.get("value")
            if value in (None, "", [], {}) or isinstance(value, (dict, list)):
                continue
            evidence = der_index.get(str(item.get("observation_id") or "")) or {}
            pd = plan_derivations.get(str(evidence.get("plan_derivation_id") or "")) or {}
            label = str(pd.get("field") or "").split(".")[-1].replace("_", " ").strip()
            if not label:
                # Raw observations do not carry a planner derivation id; preserve
                # the value without inventing a field label.
                labeled.append(str(value))
            else:
                labeled.append(f"{label}: {value}")
        if labeled:
            return "; ".join(dict.fromkeys(labeled))

    def flatten(value: Any) -> list[str]:
        if value in (None, "", [], {}):
            return []
        if isinstance(value, list):
            out: list[str] = []
            for item in value:
                out.extend(flatten(item))
            return out
        if isinstance(value, dict):
            out = []
            for key, item in value.items():
                if isinstance(item, (str, int, float, bool)) and item not in (None, ""):
                    out.append(f"{key}: {item}")
            return out
        return [str(value)]

    texts: list[str] = []
    for value in values:
        texts.extend(flatten(value))
    texts = list(dict.fromkeys(x for x in texts if x))
    if not texts:
        return ""

    if mode == "comparison" and len(texts) >= 2:
        required = list((contract or {}).get("required_plan_derivations") or [])
        diff = next((x for x in required
                     if str(x.get("comparison") or "").lower() in {"difference", "abs_difference"}), None)
        if diff is not None:
            unit = str(diff.get("unit") or "raw")
            suffix = "" if unit.lower() in {"", "raw"} else " " + unit
            return f"{texts[0]}; difference: {texts[1]}{suffix}"
    if mode == "list":
        return "\n".join(f"- {x}" for x in texts)
    return "; ".join(texts)


def evidence_gap(contract, verify_result, question,
                 include_endpoint_family=True, plan_progress=None):
    del include_endpoint_family
    missing = list((verify_result or {}).get("missing_slots") or [])
    if not missing: return None
    # Once every validated plan step has executed, certificate/derivation defects
    # cannot be repaired by repeating the same API calls. Recompile/repair locally
    # or abstain instead of paying for duplicate network/model work.
    if (plan_progress or {}).get("complete") and not (plan_progress or {}).get("missing_steps"):
        return None
    requirement = "; ".join(contract.get("answer_requirements") or []) or "the planned answer evidence"
    return {"gap_type": "missing_evidence", "required_slot": missing[0],
            "needed_relation": requirement, "source_entity_hint": "the values selected by the validated plan",
            "reason": "; ".join((verify_result or {}).get("notes") or []) or "verification failed",
            "missing_plan_steps": [x.get("id") for x in (plan_progress or {}).get("missing_steps") or []]}


def repair_prompt_section(gap):
    return """

The answer is missing evidence. Get only what is missing.
Reuse valid evidence already available. Follow the validated plan and use only valid API routes and fields. Do not change the user's question or guess.
Missing evidence:
""" + str(gap) + "\n"


def delta_check(gap, obs_index_before, obs_index_after, der_index_after):
    del gap, der_index_after
    added = [x for x in obs_index_after if x not in obs_index_before]
    return (bool(added), f"{len(added)} new observations recorded" if added else "no new observations")
