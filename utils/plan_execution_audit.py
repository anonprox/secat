"""Generic execution-instance and binding-lineage audit for OCA.

The route plan says which API operations may execute; observation specs and
host-replayed derivations say which concrete binding values are authorized.
This module joins those two layers without endpoint/entity vocabulary, benchmark
answers, or parameter-name heuristics.
"""
from __future__ import annotations
from utils.plan_progress_guard import enforce_binding_consistency as _secat_v4159_enforce_binding_consistency

from typing import Any
from urllib.parse import urlsplit




def _oas_required(value: Any) -> bool:
    """Interpret OpenAPI required flags without Python truthiness traps.

    Some real-world OAS documents encode booleans as strings.  In particular,
    ``bool("false")`` is True in Python, which would turn optional request
    parameters into mandatory ones.  Normalize defensively at the execution
    audit boundary even though schema cards normally normalize this earlier.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def _steps(plan: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {str(s.get("id")): s for s in (plan or {}).get("steps") or []}


def _specs(plan: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {str(s.get("step_id")): s for s in (plan or {}).get("observation_specs") or []}


def _placeholders(step: dict[str, Any]) -> list[str]:
    import re
    return re.findall(r"\{([^{}]+)\}", str(step.get("endpoint") or ""))


def _dependency_distances(plan: dict[str, Any] | None, step_id: str) -> dict[str, int]:
    steps = _steps(plan)
    out: dict[str, int] = {}
    queue = [(str(step_id), 0)]
    while queue:
        current, dist = queue.pop(0)
        for dep in (steps.get(current) or {}).get("depends_on") or []:
            dep = str(dep)
            nd = dist + 1
            if dep in steps and (dep not in out or nd < out[dep]):
                out[dep] = nd
                queue.append((dep, nd))
    return out


def binding_spec(plan: dict[str, Any] | None, step_id: str, name: str) -> dict[str, Any] | None:
    for binding in (_specs(plan).get(str(step_id)) or {}).get("bindings") or []:
        if str(binding.get("name") or "").strip("{}") == str(name).strip("{}"):
            return binding
    return None




def path_binding_name(plan: dict[str, Any] | None, consumer_step: str, placeholder: str) -> str:
    """Return the exact producer alias declared for one URL placeholder."""
    step = _steps(plan).get(str(consumer_step)) or {}
    key = str(placeholder).strip("{}")
    return str((step.get("path_bindings") or {}).get(key) or key).strip("{}")

def nearest_binding_producer(plan: dict[str, Any] | None, consumer_step: str,
                             placeholder: str) -> str | None:
    """Return the unique nearest producer in the declared dependency graph."""
    steps = _steps(plan)
    distances = _dependency_distances(plan, consumer_step)
    candidates: list[tuple[int, str]] = []
    for sid, dist in distances.items():
        names = {str(x).strip("{}") for x in (steps.get(sid) or {}).get("binds") or []}
        if binding_spec(plan, sid, placeholder) is not None:
            names.add(str(placeholder).strip("{}"))
        if str(placeholder).strip("{}") in names:
            candidates.append((dist, sid))
    if not candidates:
        return None
    nearest_dist = min(dist for dist, _ in candidates)
    nearest = [sid for dist, sid in candidates if dist == nearest_dist]
    return nearest[0] if len(nearest) == 1 else None


def _field(obs: dict[str, Any], path: str) -> Any:
    value: Any = obs.get("fields") or {}
    parts = [x for x in str(path or "").replace("$.", "").split(".") if x and x != "$"]
    if parts and str(obs.get("relation") or "") == parts[0]:
        parts = parts[1:]
    for part in parts:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _planner_binding_records(plan: dict[str, Any], ledger, producer: str) -> list[dict[str, Any]] | None:
    """Use the terminal host-replayed selection/filter on a producer when present."""
    plan_derivs = [d for d in plan.get("derivations") or []
                   if [str(x) for x in d.get("source_steps") or []] == [str(producer)]]
    terminal = None
    for d in plan_derivs:
        if str(d.get("operator") or "").lower() in {
                "filter", "first", "nth", "endpoint_rank", "argmax", "argmin"}:
            terminal = d
    if terminal is None:
        return None
    did = str(terminal.get("id") or "")
    replay = next((d for d in reversed(getattr(ledger, "derived", []))
                   if str(d.get("plan_derivation_id") or "") == did), None)
    if not replay:
        return None
    op = str(replay.get("operation") or "")
    if op == "filter":
        ids = replay.get("candidate_obs_ids") or replay.get("input_obs_ids") or []
    else:
        ids = replay.get("selected_obs_ids") or ([replay.get("selected_obs_id")]
              if replay.get("selected_obs_id") else [])
    records = [ledger.get(str(oid)) for oid in ids]
    return [x for x in records if x]


def _projection_records_for_calls(plan: dict[str, Any], ledger, step_id: str,
                                  call_ids: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replay the validated observation projection over observations from calls."""
    try:
        from utils.evidence_compiler import _records_from_projection_spec
        records = [o for o in getattr(ledger, "observations", [])
                   if str(o.get("call_id") or "") in call_ids]
        return _records_from_projection_spec(records, _specs(plan).get(str(step_id)) or {})
    except Exception:
        return [], []


def authorized_binding_values(plan: dict[str, Any] | None, ledger, consumer_step: str,
                              placeholder: str,
                              valid_call_ids_by_step: dict[str, set[str]] | None = None
                              ) -> tuple[set[str], str | None, str | None]:
    """Return schema/provenance-authorized concrete values for one placeholder.

    Values are canonicalized to strings because URL path captures are strings.
    No parameter-name convention (for example ``*_id -> id``) is used.
    """
    plan = plan or {}
    step = _steps(plan).get(str(consumer_step)) or {}
    literals = step.get("path_literals") or {}
    key = str(placeholder).strip("{}")
    if key in literals:
        value = literals.get(key)
        return ({str(value)} if value not in (None, "") else set()), "__literal__", "literal"
    binding_name = path_binding_name(plan, consumer_step, placeholder)
    producer = nearest_binding_producer(plan, consumer_step, binding_name)
    if not producer:
        return set(), None, None
    binding = binding_spec(plan, producer, binding_name)
    if not binding or not binding.get("path"):
        return set(), producer, None
    source = str(binding.get("source") or "selected_first")
    valid_map = valid_call_ids_by_step or {}
    call_ids = set(valid_map.get(str(producer)) or [])
    if not call_ids:
        # Historical/imported ledgers may lack trusted runtime tags. Use only
        # calls already annotated to the declared producer, never sibling calls.
        call_ids = {str(c.get("call_id")) for c in getattr(ledger, "api_calls", [])
                    if str(c.get("plan_step_id") or "") == str(producer)}
    ordered, selected = _projection_records_for_calls(plan, ledger, producer, call_ids)
    path = str(binding.get("path") or "")
    records = ordered if source == "selected_all" else selected[:1]
    # Stateful actions can produce both a synthetic ``action_result`` record and
    # a real response object at the same root pointer.  For response-value
    # bindings (id/name/uri/etc.), the synthetic certificate row must never mask
    # the actual returned entity merely because it was inserted first.  Fall
    # through only across same-root rows and only when the selected row is the
    # synthetic action result; ordinary nth/head selection semantics are left
    # untouched.
    if source != "selected_all" and records:
        row = records[0]
        if (_field(row, path) in (None, "")
                and str(row.get("kind") or "") == "action_result"):
            pointer = str(row.get("json_pointer") or "")
            replacement = next((r for r in ordered
                                if str(r.get("kind") or "") != "action_result"
                                and str(r.get("json_pointer") or "") == pointer
                                and _field(r, path) not in (None, "")), None)
            if replacement is not None:
                records = [replacement]
    if source == "selected_all" and binding.get("max_values"):
        try:
            records = records[:max(1, int(binding.get("max_values")))]
        except Exception:
            pass
    values: set[str] = set()
    for obs in records or []:
        value = _field(obs, str(binding.get("path") or ""))
        if value not in (None, ""):
            values.add(str(value))
    return values, producer, source


def authorized_binding_sequence(plan: dict[str, Any] | None, ledger, consumer_step: str,
                                binding_name: str,
                                valid_call_ids_by_step: dict[str, set[str]] | None = None
                                ) -> tuple[list[str], str | None, str | None]:
    """Return authorized values in the projection's deterministic order.

    Request bodies/query arrays may be order-sensitive (for example a requested
    sequence of resources).  The runtime guard checks ordered values, so the host
    audit must replay the same ordering rather than weakening it to a set.
    """
    plan = plan or {}
    producer = nearest_binding_producer(plan, consumer_step, binding_name)
    if not producer:
        return [], None, None
    binding = binding_spec(plan, producer, binding_name)
    if not binding or not binding.get("path"):
        return [], producer, None
    source = str(binding.get("source") or "selected_first")
    valid_map = valid_call_ids_by_step or {}
    call_ids = set(valid_map.get(str(producer)) or [])
    if not call_ids:
        call_ids = {str(c.get("call_id")) for c in getattr(ledger, "api_calls", [])
                    if str(c.get("plan_step_id") or "") == str(producer)}
    ordered, selected = _projection_records_for_calls(plan, ledger, producer, call_ids)
    path = str(binding.get("path") or "")
    records = ordered if source == "selected_all" else selected[:1]
    if source != "selected_all" and records:
        row = records[0]
        if (_field(row, path) in (None, "")
                and str(row.get("kind") or "") == "action_result"):
            pointer = str(row.get("json_pointer") or "")
            replacement = next((r for r in ordered
                                if str(r.get("kind") or "") != "action_result"
                                and str(r.get("json_pointer") or "") == pointer
                                and _field(r, path) not in (None, "")), None)
            if replacement is not None:
                records = [replacement]
    if source == "selected_all" and binding.get("max_values"):
        try:
            records = records[:max(1, int(binding.get("max_values")))]
        except Exception:
            pass
    values: list[str] = []
    for obs in records or []:
        value = _field(obs, str(binding.get("path") or ""))
        if value not in (None, ""):
            values.append(str(value))
    return values, producer, source


def empty_required_binding_precondition(plan: dict[str, Any] | None, ledger,
                                        progress: dict[str, Any] | None
                                        ) -> dict[str, Any] | None:
    """Detect a completed GET that returned an empty required source collection.

    This is deliberately narrower than generic binding failure.  It fires only
    when a missing downstream step needs a declared binding, the unique producer
    is a successful GET already accepted by the execution audit, and replaying
    that producer's validated projection yields zero records.  A malformed field,
    action-result propagation bug, or unresolved producer therefore remains a
    solver/runtime defect rather than being mislabeled as an environment
    precondition.
    """
    plan = plan or {}
    progress = progress or {}
    steps = _steps(plan)
    statuses = progress.get("step_status") or {}
    valid_map = {str(sid): set(str(x) for x in (st.get("valid_call_ids") or []))
                 for sid, st in statuses.items()}

    for consumer in progress.get("missing_steps") or []:
        consumer_sid = str(consumer.get("id") or "")
        binding_names: list[str] = []
        for mapping_name in ("path_bindings", "query_bindings", "body_bindings"):
            for binding_name in (consumer.get(mapping_name) or {}).values():
                name = str(binding_name or "")
                if name and name not in binding_names:
                    binding_names.append(name)
        for binding_name in binding_names:
            seq, producer, _source = authorized_binding_sequence(
                plan, ledger, consumer_sid, binding_name,
                valid_call_ids_by_step=valid_map)
            if seq or not producer:
                continue
            producer = str(producer)
            producer_step = steps.get(producer) or {}
            if str(producer_step.get("method") or "GET").upper() != "GET":
                continue
            status = statuses.get(producer) or {}
            call_ids = set(str(x) for x in (status.get("valid_call_ids") or []) if str(x))
            if not call_ids:
                continue
            binding = binding_spec(plan, producer, binding_name)
            if not binding or not binding.get("path"):
                continue
            ordered, _selected = _projection_records_for_calls(
                plan, ledger, producer, call_ids)
            if ordered:
                # There were records; absence of this field/value is not an empty
                # source population and should remain a normal binding defect.
                continue
            return {
                "producer_step_id": producer,
                "consumer_step_id": consumer_sid,
                "binding": binding_name,
                "endpoint": str(producer_step.get("endpoint") or ""),
                "reason": "empty_source_population",
            }
    return None


def _request_value(body: Any, path: str) -> tuple[Any, bool]:
    if not isinstance(body, (dict, list)):
        return None, False
    current = body
    for part in [x for x in str(path or "").replace("$.", "").split(".") if x]:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None, False
    return current, True


def _request_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        if isinstance(actual, str):
            return actual.strip().lower() in ({"true", "1"} if expected else {"false", "0"})
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return float(actual) == float(expected)
        except Exception:
            return False
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(
            _request_equal(actual[k], expected[k]) for k in expected)
    if isinstance(expected, (list, tuple)):
        seq = _as_sequence(actual)
        if seq is None and isinstance(actual, (list, tuple)):
            seq = list(actual)
        return seq is not None and len(seq) == len(expected) and all(
            _request_equal(a, b) for a, b in zip(seq, expected))
    return str(actual) == str(expected)


def _as_sequence(value: Any) -> list[Any] | None:
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str) and "," in value:
        return [x for x in value.split(",") if x != ""]
    return None


def _request_binding_matches(actual: Any, present: bool, allowed: list[str], wrapper: str = "") -> bool:
    if not present or not allowed:
        return False
    seq = _as_sequence(actual)
    if wrapper == "uri_objects":
        if not isinstance(actual, (list, tuple)):
            return False
        try:
            seq = [item.get("uri") for item in actual
                   if isinstance(item, dict) and set(item) == {"uri"}]
        except Exception:
            return False
        if len(seq) != len(actual):
            return False
    if seq is not None:
        return len(seq) == len(allowed) and all(str(a) == str(b) for a, b in zip(seq, allowed))
    return str(actual) in set(allowed)


def _body_leaf_paths(value: Any, prefix: str = "") -> set[str]:
    if isinstance(value, dict):
        out: set[str] = set()
        if not value and prefix:
            out.add(prefix)
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(_body_leaf_paths(item, path))
        return out
    if isinstance(value, (list, tuple)):
        return {prefix} if prefix else {"$"}
    return {prefix} if prefix else ({"$"} if value is not None else set())


def _body_path_covered(actual_path: str, declared_paths: set[str]) -> bool:
    for declared in declared_paths:
        if declared == "$" or actual_path == declared or actual_path.startswith(declared + "."):
            return True
    return False


def _correlated_binding_errors(plan: dict[str, Any], ledger, sid: str,
                               call: dict[str, Any],
                               valid_call_ids_by_step: dict[str, set[str]],
                               concrete_path: dict[str, Any]) -> list[str]:
    """Require bindings from one producer to come from one producer record.

    Checking each value independently is insufficient for a producer collection:
    rows ``(a=1,b=X),(a=2,b=Y)`` must not authorize a downstream request using
    ``(a=1,b=Y)``.  This check is API-agnostic and covers path/query/body scalars.
    Ordered list-valued bindings are already replayed as complete sequences.
    """
    step = _steps(plan).get(str(sid)) or {}
    grouped: dict[str, list[tuple[dict[str, Any], Any]]] = {}

    def add(binding_name: str, actual: Any, present: bool = True):
        if not present or _as_sequence(actual) is not None:
            return
        producer = nearest_binding_producer(plan, sid, str(binding_name))
        binding = binding_spec(plan, producer or "", str(binding_name)) if producer else None
        if producer and binding and binding.get("path"):
            grouped.setdefault(str(producer), []).append((binding, actual))

    for name in _placeholders(step):
        if name in (step.get("path_literals") or {}):
            continue
        add(path_binding_name(plan, sid, name), concrete_path.get(name), name in concrete_path)
    query = call.get("params") if isinstance(call.get("params"), dict) else {}
    for target, binding_name in (step.get("query_bindings") or {}).items():
        add(str(binding_name), query.get(target), target in query)
    body = call.get("request_body")
    for target, binding_name in (step.get("body_bindings") or {}).items():
        actual, present = _request_value(body, str(target))
        add(str(binding_name), actual, present)

    errors: list[str] = []
    for producer, items in grouped.items():
        if len(items) < 2:
            continue
        call_ids = set(valid_call_ids_by_step.get(producer) or [])
        if not call_ids:
            call_ids = {str(c.get("call_id")) for c in getattr(ledger, "api_calls", [])
                        if str(c.get("plan_step_id") or c.get("runtime_step_id") or "") == producer}
        ordered, selected = _projection_records_for_calls(plan, ledger, producer, call_ids)
        records = ordered
        if any(str(binding.get("source") or "selected_first") != "selected_all"
               for binding, _ in items):
            records = selected[:1]
        if not records or not any(
                all(_request_equal(actual, _field(record, str(binding.get("path") or "")))
                    for binding, actual in items)
                for record in records):
            errors.append(f"correlated binding mismatch from producer {producer}")
    return errors


def _call_request_contract_errors(plan: dict[str, Any], ledger, sid: str,
                                  call: dict[str, Any],
                                  valid_call_ids_by_step: dict[str, set[str]]) -> list[str]:
    """Host-replay query/body semantics for defense-in-depth certification."""
    step = _steps(plan).get(str(sid)) or {}
    errors: list[str] = []
    query = call.get("params") if isinstance(call.get("params"), dict) else {}
    q_literals = step.get("query_literals") or {}
    q_bindings = step.get("query_bindings") or {}
    declared_query = set(q_literals) | set(q_bindings)
    extra_query = sorted(set(str(x) for x in query) - set(str(x) for x in declared_query))
    if extra_query:
        errors.append("undeclared query arguments: " + ",".join(extra_query))
    required_query = {
        str(p.get("name")) for p in (step.get("request_parameters") or [])
        if str(p.get("in") or "").lower() == "query" and _oas_required(p.get("required")) and p.get("name")
    }
    missing_query = sorted(required_query - set(str(x) for x in query))
    if missing_query:
        errors.append("missing required query arguments: " + ",".join(missing_query))
    for name, expected in q_literals.items():
        if name not in query or not _request_equal(query.get(name), expected):
            errors.append(f"query literal mismatch: {name}")
    for target, binding_name in q_bindings.items():
        allowed, _, _ = authorized_binding_sequence(
            plan, ledger, sid, str(binding_name), valid_call_ids_by_step)
        if not _request_binding_matches(query.get(target), target in query, allowed):
            errors.append(f"query binding mismatch: {target}<-{binding_name}")

    body = call.get("request_body")
    b_literals = step.get("body_literals") or {}
    b_bindings = step.get("body_bindings") or {}
    declared_body = set(str(x) for x in b_literals) | set(str(x) for x in b_bindings)
    actual_body_paths = _body_leaf_paths(body) if body is not None else set()
    extra_body = sorted(path for path in actual_body_paths
                        if not _body_path_covered(path, declared_body))
    if extra_body:
        errors.append("undeclared body paths: " + ",".join(extra_body))
    for name, expected in b_literals.items():
        actual, present = _request_value(body, str(name))
        if not present or not _request_equal(actual, expected):
            errors.append(f"body literal mismatch: {name}")
    body_wrappers = step.get("body_binding_wrappers") or {}
    for target, binding_name in b_bindings.items():
        actual, present = _request_value(body, str(target))
        allowed, _, _ = authorized_binding_sequence(
            plan, ledger, sid, str(binding_name), valid_call_ids_by_step)
        if not _request_binding_matches(actual, present, allowed, str(body_wrappers.get(target) or "")):
            errors.append(f"body binding mismatch: {target}<-{binding_name}")
    required_body = ([str(x) for x in (step.get("request_body_required_fields") or [])]
                     if (declared_body or bool(step.get("request_body_required"))) else [])
    for name in required_body:
        _, present = _request_value(body, name)
        if not present:
            errors.append(f"missing required body field: {name}")
    return errors


def _topological_step_ids(plan: dict[str, Any]) -> list[str]:
    steps = _steps(plan)
    remaining = set(steps)
    out: list[str] = []
    while remaining:
        ready = [sid for sid in remaining
                 if all(str(dep) in out or str(dep) not in steps
                        for dep in (steps[sid].get("depends_on") or []))]
        if not ready:  # malformed cycle: preserve declaration order, audit will fail closed.
            out.extend([str(s.get("id")) for s in plan.get("steps") or []
                        if str(s.get("id")) in remaining])
            break
        declaration = [str(s.get("id")) for s in plan.get("steps") or []]
        ready.sort(key=lambda x: declaration.index(x) if x in declaration else len(declaration))
        for sid in ready:
            out.append(sid); remaining.remove(sid)
    return out


def _step_requires_complete_collection(plan: dict[str, Any], sid: str) -> bool:
    """Infer completeness from deterministic data-flow, not LLM prose alone."""
    spec = _specs(plan).get(str(sid)) or {}
    if "all" in str(spec.get("completeness") or "").casefold():
        return True
    if spec.get("sort"):
        return True
    if any(str(b.get("source") or "") == "selected_all" and not b.get("max_values")
           for b in spec.get("bindings") or []):
        return True
    if any(str(a.get("op") or "").lower() in {"count", "sum", "mean", "min", "max"}
           for a in spec.get("aggregates") or []):
        return True
    complete_ops = {"count", "argmax", "argmin"}
    for derivation in plan.get("derivations") or []:
        if str(derivation.get("operator") or "").lower() in complete_ops and \
                str(sid) in {str(x) for x in derivation.get("source_steps") or []}:
            return True
    return False


def _record_path_container_pointer(record_path: str) -> str | None:
    """Return the exact JSON-pointer container for a ``...[*]`` record root.

    Relation names alone are not sufficient: ``left.items[*]`` and
    ``right.items[*]`` are different candidate universes even though both have
    relation name ``items``.
    """
    rp = str(record_path or "$").strip()
    if rp == "$":
        return "/"
    if not rp.endswith("[*]"):
        return None
    container = rp[:-3].strip(".")
    if not container:
        return "/"
    parts = [x for x in container.replace("$.", "").split(".") if x and x != "$"]
    return "/" + "/".join(str(x).replace("~", "~0").replace("/", "~1") for x in parts)


def _truncated_required_collection(plan: dict[str, Any], sid: str,
                                   valid_call_ids: list[Any], calls: list[dict[str, Any]]) -> bool:
    spec = _specs(plan).get(str(sid)) or {}
    if not _step_requires_complete_collection(plan, sid):
        return False
    rp = str(spec.get("record_path") or "$")
    expected_pointer = _record_path_container_pointer(rp)
    import re
    relation_match = re.findall(r"(?:^|\.)([A-Za-z_][A-Za-z0-9_]*)\[\*\]", rp)
    relation = relation_match[-1] if relation_match else "$" if rp == "$" else None
    valid_ids = {str(x) for x in valid_call_ids if x}
    for call in calls:
        if str(call.get("call_id") or "") not in valid_ids:
            continue
        markers = call.get("truncated_collections") or []
        for marker in markers:
            pointer = str(marker.get("json_pointer") or "")
            if expected_pointer and pointer:
                if pointer == expected_pointer:
                    return True
                continue
            # Backward-compatible fallback for imported/historical ledgers that
            # predate JSON-pointer truncation markers.
            if relation is None or str(marker.get("relation")) == relation:
                return True
    return False



def _step_requires_global_collection(plan: dict[str, Any], sid: str) -> bool:
    """Return True when correctness requires more than a page-local candidate set.

    This is intentionally structural: global count/extremum/aggregate, explicit
    sorting across candidates, or selected_all fan-out cannot be certified from a
    response that advertises another page.  Simple first/top-k endpoint-order
    operations remain page-local unless the plan separately declares otherwise.
    """
    spec = _specs(plan).get(str(sid)) or {}
    if spec.get("sort"):
        return True
    if any(str(b.get("source") or "") == "selected_all" and not b.get("max_values")
           for b in spec.get("bindings") or []):
        return True
    if any(str(a.get("op") or "").lower() in {"count", "sum", "mean", "min", "max"}
           for a in spec.get("aggregates") or []):
        return True
    for derivation in plan.get("derivations") or []:
        if str(sid) not in {str(x) for x in derivation.get("source_steps") or []}:
            continue
        if str(derivation.get("operator") or "").lower() in {"count", "argmax", "argmin"}:
            return True
    return False


def _pagination_context(payload: Any, record_path: str) -> dict[str, Any] | None:
    """Return the object that owns pagination metadata for ``record_path``.

    Examples: ``results[*]`` -> root object; ``tracks.items[*]`` -> ``tracks``.
    Only dictionary parents are considered, so ordinary record fields named
    ``next`` cannot accidentally trigger this check.
    """
    if not isinstance(payload, dict):
        return None
    rp = str(record_path or "$").replace("$.", "").strip()
    if not rp or rp == "$" or not rp.endswith("[*]"):
        return payload
    container = rp[:-3].strip(".")
    parts = [x for x in container.split(".") if x]
    # Parent of the terminal collection owns page/next metadata.
    current: Any = payload
    for part in parts[:-1]:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current if isinstance(current, dict) else None


def _incomplete_paginated_required_collection(plan: dict[str, Any], sid: str,
                                               valid_call_ids: list[Any], ledger) -> bool:
    """Fail closed when a global operation is based on an advertised partial page.

    No provider names are used.  We recognize common top-level pagination shapes
    only when they occur on the object that owns the planned record collection.
    Multiple numbered pages are accepted when every advertised page is present;
    cursor/URL pagination is accepted when the last observed page has no next token.
    """
    if not _step_requires_global_collection(plan, sid):
        return False
    valid = {str(x) for x in valid_call_ids if x}
    if not valid:
        return False
    spec = _specs(plan).get(str(sid)) or {}
    record_path = str(spec.get("record_path") or "$")
    raws = [r for r in (getattr(ledger, "raw_responses", []) or [])
            if str(r.get("call_id") or "") in valid]
    if not raws:
        return False

    page_numbers: set[int] = set()
    total_pages: set[int] = set()
    contexts: list[dict[str, Any]] = []
    for raw in raws:
        payload = raw.get("payload")
        ctx = _pagination_context(payload, record_path)
        if not isinstance(ctx, dict):
            continue
        contexts.append(ctx)
        page = ctx.get("page")
        total = ctx.get("total_pages")
        try:
            if page is not None: page_numbers.add(int(page))
            if total is not None: total_pages.add(int(total))
        except (TypeError, ValueError):
            pass

    if total_pages:
        advertised = max(total_pages)
        if advertised > 1 and not set(range(1, advertised + 1)).issubset(page_numbers):
            return True

    if contexts:
        last = contexts[-1]
        if last.get("has_more") is True or last.get("has_next") is True:
            return True
        for key in ("next", "next_page", "next_url", "next_cursor", "next_token",
                    "continuation_token", "continuation"):
            value = last.get(key)
            if value not in (None, "", False, [], {}):
                return True
    return False


def _normalized_origin(value: str | None) -> str:
    try:
        parts = urlsplit(str(value or ""))
        return (parts.scheme.lower() + "://" + parts.netloc.lower()) if parts.scheme and parts.netloc else ""
    except Exception:
        return ""

def _origin_authorized(plan: dict[str, Any], call: dict[str, Any]) -> bool:
    expected = _normalized_origin((plan or {}).get("_runtime_base_url"))
    if not expected:
        return True
    return str(call.get("request_origin") or "").lower() == expected


def _successful_http_call(call: dict[str, Any]) -> bool:
    """Only observed 2xx responses may satisfy an evidence-plan step."""
    status = call.get("status_code")
    return isinstance(status, int) and 200 <= status < 300


def _runtime_contract_certified(call: dict[str, Any], step_id: str) -> bool:
    """Return whether the trusted live guard certified this exact plan instance.

    The kernel request wrapper validates origin, concrete path placeholders, query
    literals/bindings, body literals/bindings, correlated bindings and call budget
    against the full trusted producer payload *before* issuing the request.  The
    post-run observation ledger is intentionally compact and may omit the selected
    producer record.  Reconstructing authorization from that lossy view can
    therefore disagree with the live guard and falsely mark a successful action as
    incomplete.  A versioned runtime certificate closes that split-brain path.

    Historical/imported calls without this marker retain the full reconstruction
    audit below.
    """
    return bool(call.get("runtime_contract_authorized")) and (
        int(call.get("runtime_contract_version") or 0) >= 1
        and str(call.get("runtime_contract_step_id") or call.get("runtime_step_id") or "")
        == str(step_id or "")
    )


def audit_execution(plan: dict[str, Any] | None, ledger) -> dict[str, Any]:
    """Audit every concrete call against declared dependency/binding lineage.

    A templated path is not enough: each placeholder must equal a value selected
    by the unique nearest declared producer. ``selected_all`` fan-out is complete
    only after every selected value has been covered. Ambiguity fails closed.
    """
    from utils.evidence_plan import endpoint_matches, extract_path_bindings, assign_calls_to_steps

    plan = plan or {}
    steps = _steps(plan)
    calls = list(getattr(ledger, "api_calls", []) or [])
    base_assignments = assign_calls_to_steps(plan, calls)
    statuses: dict[str, dict[str, Any]] = {}
    completed: set[str] = set()
    valid_by_step: dict[str, set[str]] = {}
    lineage_errors: list[dict[str, Any]] = []

    for sid in _topological_step_ids(plan):
        step = steps.get(sid) or {}
        method = str(step.get("method") or "GET").upper()
        template = str(step.get("endpoint") or "")
        from utils.api_runtime import wire_endpoint_template
        wire_template = wire_endpoint_template(str(plan.get("_runtime_base_url") or ""), template)
        names = _placeholders(step)
        matching = [c for c in calls
                    if str(c.get("method") or "GET").upper() == method
                    and endpoint_matches(str(c.get("endpoint") or ""), wire_template)]
        trusted = [c for c in matching if str(c.get("runtime_step_id") or "") == sid]
        status = {
            "step_id": sid,
            "matching_call_ids": [c.get("call_id") for c in matching],
            "valid_call_ids": [], "invalid_call_ids": [],
            "expected_bindings": {}, "covered_bindings": {},
            "unresolved_placeholders": [], "truncated_required_collection": False,
            "incomplete_pagination": False,
            "request_contract_errors": {}, "http_status_errors": {}, "origin_errors": {},
        }

        if not names:
            if trusted:
                candidates = trusted
            else:
                assigned_ids = {str(cid) for cid, assigned_sid in base_assignments.items()
                                if str(assigned_sid) == sid}
                candidates = [c for c in matching if str(c.get("call_id")) in assigned_ids]
            valid = []
            for call in candidates:
                cid = call.get("call_id")
                if not _origin_authorized(plan, call):
                    status["invalid_call_ids"].append(cid)
                    status["origin_errors"][str(cid)] = call.get("request_origin")
                    lineage_errors.append({
                        "step_id": sid, "call_id": cid, "endpoint": call.get("endpoint"),
                        "request_origin": call.get("request_origin"), "reason": "unauthorized_origin",
                    })
                    continue
                if not _successful_http_call(call):
                    status["invalid_call_ids"].append(cid)
                    status["http_status_errors"][str(cid)] = call.get("status_code")
                    lineage_errors.append({
                        "step_id": sid, "call_id": cid, "endpoint": call.get("endpoint"),
                        "http_status": call.get("status_code"), "reason": "non_2xx_response",
                    })
                    continue
                # A live runtime certificate is stronger than post-hoc
                # reconstruction from compact observations: the exact concrete
                # request already passed the trusted guard against the full
                # producer payload. Imported/historical calls still use the
                # reconstruction path.
                request_errors = ([] if _runtime_contract_certified(call, sid) else
                    _call_request_contract_errors(plan, ledger, sid, call, valid_by_step))
                if request_errors:
                    status["invalid_call_ids"].append(cid)
                    status["request_contract_errors"][str(cid)] = request_errors
                    lineage_errors.append({
                        "step_id": sid, "call_id": cid,
                        "endpoint": call.get("endpoint"),
                        "request_contract_errors": request_errors,
                    })
                else:
                    valid.append(call)
            status["valid_call_ids"] = [c.get("call_id") for c in valid]
            valid_by_step[sid] = {str(c.get("call_id")) for c in valid if c.get("call_id")}
            if valid and _truncated_required_collection(plan, sid, status["valid_call_ids"], calls):
                status["truncated_required_collection"] = True
            elif valid and _incomplete_paginated_required_collection(
                    plan, sid, status["valid_call_ids"], ledger):
                status["incomplete_pagination"] = True
            elif valid:
                completed.add(sid)
            statuses[sid] = status
            continue

        candidate_calls = trusted or matching
        runtime_certified = any(_runtime_contract_certified(c, sid) for c in candidate_calls)
        expected_by_name: dict[str, set[str]] = {}
        source_by_name: dict[str, str | None] = {}
        for name in names:
            values, producer, source = authorized_binding_values(
                plan, ledger, sid, name, valid_by_step)
            expected_by_name[name] = values
            source_by_name[name] = source
            status["expected_bindings"][name] = sorted(values)
            # Compact observation views may omit the selected producer row. A
            # runtime-certified request has already proven the placeholder value
            # against the full trusted producer payload, so absence here is a
            # diagnostic limitation rather than an unresolved dependency.
            if (producer is None or not values) and not runtime_certified:
                status["unresolved_placeholders"].append(name)

        for call in candidate_calls:
            concrete = {k: str(v) for k, v in
                        extract_path_bindings(str(call.get("endpoint") or ""), wire_template).items()}
            origin_ok = _origin_authorized(plan, call)
            certified = _runtime_contract_certified(call, sid)
            valid = (origin_ok and _successful_http_call(call) and
                     (certified or not status["unresolved_placeholders"]))
            if not origin_ok:
                status["origin_errors"][str(call.get("call_id"))] = call.get("request_origin")
            if not _successful_http_call(call):
                status["http_status_errors"][str(call.get("call_id"))] = call.get("status_code")
            if valid and not certified:
                for name in names:
                    if concrete.get(name) not in expected_by_name.get(name, set()):
                        valid = False
            request_errors = []
            if valid and not certified:
                request_errors = _call_request_contract_errors(
                    plan, ledger, sid, call, valid_by_step)
                request_errors.extend(_correlated_binding_errors(
                    plan, ledger, sid, call, valid_by_step, concrete))
                if request_errors:
                    valid = False
            cid = call.get("call_id")
            if valid:
                status["valid_call_ids"].append(cid)
            else:
                status["invalid_call_ids"].append(cid)
                if request_errors:
                    status["request_contract_errors"][str(cid)] = request_errors
                lineage_errors.append({
                    "step_id": sid, "call_id": cid, "endpoint": call.get("endpoint"),
                    "actual_bindings": concrete,
                    "expected_bindings": status["expected_bindings"],
                    "request_contract_errors": request_errors,
                    "http_status": call.get("status_code"),
                    "request_origin": call.get("request_origin"),
                    "reason": ("unauthorized_origin" if not origin_ok else
                               "non_2xx_response" if not _successful_http_call(call) else None),
                })

        valid_calls = [c for c in candidate_calls if c.get("call_id") in status["valid_call_ids"]]
        valid_by_step[sid] = {str(c.get("call_id")) for c in valid_calls if c.get("call_id")}
        complete = bool(valid_calls) and (runtime_certified or not status["unresolved_placeholders"])
        for name in names:
            covered = {str(extract_path_bindings(str(c.get("endpoint") or ""), wire_template).get(name))
                       for c in valid_calls
                       if extract_path_bindings(str(c.get("endpoint") or ""), wire_template).get(name) is not None}
            status["covered_bindings"][name] = sorted(covered)
            if (not runtime_certified and source_by_name.get(name) == "selected_all"
                    and not expected_by_name[name].issubset(covered)):
                complete = False
        # A step claiming complete-collection semantics cannot be certified from
        # a structurally truncated ledger. Fail closed rather than ranking/counting
        # an arbitrary retained prefix.
        if complete and _truncated_required_collection(
                plan, sid, status.get("valid_call_ids") or [], calls):
            status["truncated_required_collection"] = True
            complete = False
        if complete and _incomplete_paginated_required_collection(
                plan, sid, status.get("valid_call_ids") or [], ledger):
            status["incomplete_pagination"] = True
            complete = False
        if complete:
            completed.add(sid)
        statuses[sid] = status

    missing = [step for step in (plan.get("steps") or [])
               if str(step.get("id")) not in completed]
    return _secat_v4159_enforce_binding_consistency({
        "completed_step_ids": [str(s.get("id")) for s in plan.get("steps") or []
                               if str(s.get("id")) in completed],
        "missing_steps": missing,
        "complete": bool(plan.get("steps")) and not missing,
        "step_status": statuses,
        "lineage_errors": lineage_errors,
        "call_step_assignments": base_assignments,
    })


def apply_audited_annotations(plan: dict[str, Any] | None, ledger, audit: dict[str, Any]) -> None:
    """Retag evidence so the compiler sees only calls authorized for each step."""
    # Clear non-forced annotations first. The audit will restore only valid calls.
    valid_map: dict[str, str] = {}
    for sid, status in (audit or {}).get("step_status", {}).items():
        for cid in status.get("valid_call_ids") or []:
            if cid:
                valid_map[str(cid)] = str(sid)
    # Do not restore ordinary endpoint assignments that the audited request
    # contract rejected. ``valid_call_ids`` is authoritative for both templated
    # and non-templated operations.

    from utils.evidence_plan import extract_path_bindings
    steps = _steps(plan)
    for call in getattr(ledger, "api_calls", []) or []:
        cid = str(call.get("call_id") or "")
        call.pop("plan_step_id", None)
        call.pop("source_bindings", None)
        sid = valid_map.get(cid)
        if not sid:
            continue
        call["plan_step_id"] = sid
        from utils.api_runtime import wire_endpoint_template
        logical_template = str((steps.get(sid) or {}).get("endpoint") or "")
        wire_template = wire_endpoint_template(str((plan or {}).get("_runtime_base_url") or ""),
                                               logical_template)
        bindings = extract_path_bindings(str(call.get("endpoint") or ""), wire_template)
        if bindings:
            call["source_bindings"] = bindings
    for collection in (getattr(ledger, "observations", []), getattr(ledger, "raw_responses", [])):
        for item in collection:
            cid = str(item.get("call_id") or "")
            item.pop("plan_step_id", None)
            item.pop("source_bindings", None)
            sid = valid_map.get(cid)
            if not sid:
                continue
            item["plan_step_id"] = sid
            call = next((c for c in getattr(ledger, "api_calls", []) if str(c.get("call_id")) == cid), None)
            if call and call.get("source_bindings"):
                item["source_bindings"] = dict(call["source_bindings"])


def runtime_execution_rules(plan: dict[str, Any] | None, *, max_fanout: int = 20) -> list[dict[str, Any]]:
    """Serialize the safe plan/data-flow subset used by the trusted kernel guard.

    The rules contain no benchmark answers or reference routes. They are derived
    exclusively from the validated plan and observation projection. A dependent
    concrete call is authorized only when its placeholder value can be replayed
    from the declared producer's already-observed response.
    """
    plan = plan or {}
    specs = _specs(plan)
    rules: list[dict[str, Any]] = []
    prior_write_step_ids: list[str] = []
    write_methods = {"POST", "PUT", "DELETE", "PATCH"}
    for step in plan.get("steps") or []:
        sid = str(step.get("id") or "")
        endpoint = str(step.get("endpoint") or "")
        if not sid or not endpoint:
            continue
        placeholders: dict[str, Any] = {}
        budget = 1
        for name in _placeholders(step):
            literals = step.get("path_literals") or {}
            literal_value = literals.get(name) if name in literals else None
            binding_name = path_binding_name(plan, sid, name)
            producer = None if name in literals else nearest_binding_producer(plan, sid, binding_name)
            binding = binding_spec(plan, producer or "", binding_name) if producer else None
            if binding and str(binding.get("source") or "selected_first") == "selected_all":
                try:
                    bound = max(1, int(binding.get("max_values") or max_fanout))
                except Exception:
                    bound = max_fanout
                budget = max(budget, min(max(1, int(max_fanout)), bound))
            placeholders[name] = {
                "producer_step_id": producer,
                "binding_name": binding_name,
                "binding": dict(binding or {}),
                "literal_value": literal_value,
                "source": "literal" if name in literals else "binding",
            }
        query_contract: dict[str, Any] = {}
        body_contract: dict[str, Any] = {}
        for target, value in (step.get("query_literals") or {}).items():
            query_contract[str(target)] = {"source": "literal", "literal_value": value}
        for target, binding_name in (step.get("query_bindings") or {}).items():
            producer = nearest_binding_producer(plan, sid, str(binding_name))
            binding = binding_spec(plan, producer or "", str(binding_name)) if producer else None
            if binding and str(binding.get("source") or "selected_first") == "selected_all":
                try:
                    bound = max(1, int(binding.get("max_values") or max_fanout))
                except Exception:
                    bound = max_fanout
                budget = max(budget, min(max(1, int(max_fanout)), bound))
            query_contract[str(target)] = {
                "source": "binding", "binding_name": str(binding_name),
                "producer_step_id": producer, "binding": dict(binding or {})}
        for target, value in (step.get("body_literals") or {}).items():
            body_contract[str(target)] = {"source": "literal", "literal_value": value}
        for target, binding_name in (step.get("body_bindings") or {}).items():
            producer = nearest_binding_producer(plan, sid, str(binding_name))
            binding = binding_spec(plan, producer or "", str(binding_name)) if producer else None
            if binding and str(binding.get("source") or "selected_first") == "selected_all":
                try:
                    bound = max(1, int(binding.get("max_values") or max_fanout))
                except Exception:
                    bound = max_fanout
                budget = max(budget, min(max(1, int(max_fanout)), bound))
            body_contract[str(target)] = {
                "source": "binding", "binding_name": str(binding_name),
                "producer_step_id": producer, "binding": dict(binding or {}),
                "wrapper": str((step.get("body_binding_wrappers") or {}).get(target) or ""),
            }
        base_url = str(plan.get("_runtime_base_url") or "")
        allowed_origin = _normalized_origin(base_url)
        from utils.api_runtime import wire_endpoint_template
        wire_endpoint = wire_endpoint_template(base_url, endpoint)
        method = str(step.get("method") or "GET").upper()
        completion_bindings: list[dict[str, Any]] = []
        for name, info in placeholders.items():
            binding = info.get("binding") or {}
            if str(info.get("source") or "") == "binding" and str(binding.get("source") or "") == "selected_all":
                completion_bindings.append({
                    "location": "path", "name": str(name),
                    "producer_step_id": info.get("producer_step_id"),
                    "binding": dict(binding), "wrapper": "",
                })
        for name, info in query_contract.items():
            binding = info.get("binding") or {}
            if str(info.get("source") or "") == "binding" and str(binding.get("source") or "") == "selected_all":
                completion_bindings.append({
                    "location": "query", "name": str(name),
                    "producer_step_id": info.get("producer_step_id"),
                    "binding": dict(binding), "wrapper": "",
                })
        for name, info in body_contract.items():
            binding = info.get("binding") or {}
            if str(info.get("source") or "") == "binding" and str(binding.get("source") or "") == "selected_all":
                completion_bindings.append({
                    "location": "body", "name": str(name),
                    "producer_step_id": info.get("producer_step_id"),
                    "binding": dict(binding), "wrapper": str(info.get("wrapper") or ""),
                })
        rules.append({
            "step_id": sid,
            "allowed_origin": allowed_origin,
            "method": method,
            "endpoint": endpoint,
            "wire_endpoint": wire_endpoint,
            "max_calls": budget,
            "depends_on": [str(x) for x in step.get("depends_on") or []],
            # State-changing operations preserve the validated plan's write order
            # even when the semantic graph contains independent sibling actions.
            # This prevents a later write (e.g. skip) from changing provider state
            # before an earlier write (e.g. unfollow current artist) completes.
            "prior_write_step_ids": list(prior_write_step_ids) if method in write_methods else [],
            # selected_all write steps may be realized by one complete batched
            # request or by several authorized scalar fan-out requests. Runtime
            # dependency readiness must not treat the first successful target as
            # completion of the whole step.
            "completion_bindings": completion_bindings,
            "placeholders": placeholders,
            "query_contract": query_contract,
            "body_contract": body_contract,
            "observation_spec": dict(specs.get(sid) or {}),
            "required_query_params": [
                str(p.get("name")) for p in (step.get("request_parameters") or [])
                if str(p.get("in") or "").lower() == "query" and _oas_required(p.get("required")) and p.get("name")
            ],
            "required_body_fields": ([str(x) for x in (step.get("request_body_required_fields") or [])]
                                     if (body_contract or bool(step.get("request_body_required"))) else []),
        })
        if method in write_methods:
            prior_write_step_ids.append(sid)
    return rules
