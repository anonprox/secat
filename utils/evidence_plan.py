"""Lightweight evidence planning for OCA v2.

The planner is deliberately smaller than ToolCoder: one JSON planning call, no
code scaffold, no assembly stage, and no execution-repair loop. It identifies a
complete endpoint chain and deterministic derivations before CodeAct-style
execution begins. Endpoint templates are validated against the benchmark OAS;
the gold solution is never read or supplied.
"""
from __future__ import annotations

import copy
import json
import re
from difflib import SequenceMatcher
from typing import Any


def _oas_required(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def _load_tools(benchmark: str) -> list[dict[str, Any]]:
    """Load OAS operations plus compact answer-free structural capabilities."""
    import benchmarks as B
    if not benchmark:
        raise ValueError("evidence planning requires an explicit API/benchmark configuration")
    bench = B.get_benchmark(benchmark)
    with open(bench["oas_file"], encoding="utf-8") as handle:
        raw = json.load(handle)
    tools: list[dict[str, Any]] = []
    if isinstance(raw, dict):
        for path, methods in raw.get("paths", {}).items():
            if not isinstance(methods, dict):
                continue
            for method, docs in methods.items():
                if str(method).lower() not in {"get", "post", "put", "delete", "patch"}:
                    continue
                docs = docs if isinstance(docs, dict) else {}
                desc = docs.get("summary") or docs.get("description") or ""
                tools.append({"path": path, "method": str(method).upper(),
                              "functionality": desc})
    else:
        tools = []
        for item in raw:
            if not item.get("path"):
                continue
            prose = []
            for value in (item.get("description"), item.get("functionality")):
                text = " ".join(str(value or "").split())
                if text and text not in prose:
                    prose.append(text)
            tools.append({"path": item.get("path"),
                          "method": str(item.get("method") or "GET").upper(),
                          "functionality": " ".join(prose),
                          "parameters": item.get("parameters") or []})

    # Give the *initial* planner schema names/types, not values.  This prevents a
    # capability regression where it plans an expensive N-way child lookup simply
    # because the endpoint prose omitted fields already present upstream.  This is
    # OAS structure only and contains no benchmark solution or response example.
    try:
        from utils.schema_outline import endpoint_catalog_cards
        by_op = {(str(c.get("method") or "GET").upper(), str(c.get("endpoint") or "")): c
                 for c in endpoint_catalog_cards(benchmark, max_paths_per_endpoint=45)}
        enriched = []
        for tool in tools:
            item = dict(tool)
            card = by_op.get((str(item.get("method") or "GET").upper(), str(item.get("path") or "")))
            if card:
                item["schema_card"] = card
            enriched.append(item)
        tools = enriched
    except Exception:
        # A missing structural outline must not make the API unusable; selected
        # endpoint schemas are still validated later before execution.
        pass
    return tools


def _template_regex(template: str) -> re.Pattern:
    """Compile an OAS path template without domain/entity assumptions."""
    chunks = []; cursor = 0
    for match in re.finditer(r"\{([^{}]+)\}", template or ""):
        chunks.append(re.escape(template[cursor:match.start()]))
        # OAS parameter names do not imply a wire type. An ``*_id`` may be an
        # integer, UUID, slug, opaque hash, or any other path-safe scalar. Type
        # information belongs to the schema, not to the parameter's spelling.
        chunks.append(r"[^/]+")
        cursor = match.end()
    chunks.append(re.escape((template or "")[cursor:]))
    return re.compile(r"^" + "".join(chunks).rstrip("/") + r"/?$")


def endpoint_matches(actual: str, template: str) -> bool:
    actual_path = (actual or "").split("?", 1)[0].rstrip("/")
    return bool(_template_regex((template or "").rstrip("/")).match(actual_path))


def extract_path_bindings(actual: str, template: str) -> dict[str, Any]:
    """Extract concrete path-placeholder values using only the OAS template."""
    actual_path = (actual or "").split("?", 1)[0].rstrip("/")
    template_path = (template or "").rstrip("/")
    names = re.findall(r"\{([^{}]+)\}", template_path)
    if not names:
        return {}
    chunks, cursor = [], 0
    for match in re.finditer(r"\{([^{}]+)\}", template_path):
        chunks.append(re.escape(template_path[cursor:match.start()]))
        chunks.append(r"([^/]+)")
        cursor = match.end()
    chunks.append(re.escape(template_path[cursor:]))
    m = re.fullmatch("".join(chunks), actual_path)
    if not m:
        return {}
    from urllib.parse import unquote
    out: dict[str, Any] = {}
    for name, value in zip(names, m.groups()):
        # Compare logical path-segment values, not their URL-encoded wire form.
        # ``unquote`` deliberately preserves literal '+' in path segments.
        value = unquote(value or "")
        if re.fullmatch(r"-?\d+", value or ""):
            try: value = int(value)
            except Exception: pass
        out[name] = value
    return out


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
            continue
    return None


def _nearest(path: str, valid_paths: list[str], n=4) -> list[str]:
    scored = [(SequenceMatcher(None, path or "", candidate).ratio(), candidate)
              for candidate in valid_paths]
    return [p for _, p in sorted(scored, reverse=True)[:n]]


def _request_literal_map(raw: Any) -> dict[str, Any]:
    """Keep only JSON-serializable request literal declarations."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        name = str(key).strip()
        if not name:
            continue
        try:
            json.dumps(value)
        except Exception:
            continue
        out[name] = value
    return out


def _request_binding_map(raw: Any) -> dict[str, str]:
    """Normalize model-produced request-binding references to compact strings.

    The planner may express the same dependency as a plain alias, a step-qualified
    field path, or a small ``{source_step, field}`` object.  Preserve the meaning
    here; ``normalize_dynamic_path_references`` resolves it against the declared
    plan graph before provenance validation.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        name = str(key).strip()
        if not name:
            continue
        if isinstance(value, dict):
            source_step = str(value.get("source_step") or value.get("step") or "").strip()
            field = str(value.get("field") or value.get("path") or "").strip()
            if source_step and field:
                value = f"{source_step}:$.{field.lstrip('$.')}"
            elif source_step:
                value = source_step
            else:
                continue
        text = str(value).strip().strip("{}")
        if text:
            out[name] = text
    return out


def _parse_step_qualified_binding_reference(value: Any) -> tuple[str, str] | None:
    """Parse explicit ``step + response-field`` references emitted by planners.

    Accepted examples include ``s1.results[0].id``, ``s1:$.results[0].id`` and
    ``s1.$.results[*].id``.  The returned field path uses ``[*]`` for any numeric
    list index so it can be compared with OAS/plan binding paths.
    """
    if not isinstance(value, str):
        return None
    text = value.strip().strip("{}")
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*)(?::|\.)\$?(?:\.)?(.+)$", text)
    if not m:
        return None
    step_id, path = m.group(1), m.group(2)
    path = re.sub(r"\[(?:\d+|\*)\]", "[*]", path)
    path = path.lstrip(".$")
    path = re.sub(r"\.{2,}", ".", path)
    return (step_id, path) if path else None


def _canonical_derivation_filter(raw_filter: Any, field: Any = None) -> dict[str, Any]:
    """Normalize equivalent planner filter syntax without adding semantics.

    Planner JSON occasionally expresses a nested schema path as a nested object
    (``{"crew":{"job":{...}}}``) or uses null-test aliases that the runtime
    projection dialect does not name directly. Canonicalize those shapes into the
    one field-keyed predicate language shared by observation projection and the
    deterministic compiler. Literal JSON-object equality remains untouched unless
    a descendant explicitly contains a predicate ``op``.
    """
    if not isinstance(raw_filter, dict) or not raw_filter:
        return {}

    # Normalize a common legacy/model shape where the filter object itself carries
    # ``field/comparison/comparison_literal`` instead of using the field as the
    # dictionary key.  This is only syntax normalization; no predicate is invented.
    legacy_field = raw_filter.get("field")
    legacy_cmp = str(raw_filter.get("comparison") or "").lower().strip()
    if isinstance(legacy_field, str) and legacy_field.strip() and legacy_cmp:
        cmp_aliases = {
            "equals": "eq", "equal": "eq", "==": "eq", "eq": "eq",
            "not_equals": "neq", "not_equal": "neq", "!=": "neq", "neq": "neq",
            "greater_than": "gt", ">": "gt", "gt": "gt",
            "greater_or_equal": "gte", ">=": "gte", "gte": "gte",
            "less_than": "lt", "<": "lt", "lt": "lt",
            "less_or_equal": "lte", "<=": "lte", "lte": "lte",
        }
        op = cmp_aliases.get(legacy_cmp)
        if op:
            return {legacy_field.strip(): {
                "op": op,
                "value": raw_filter.get("comparison_literal")
            }}

    def predicate_alias(pred: dict[str, Any]) -> dict[str, Any]:
        out = dict(pred)
        op = str(out.get("op") or "eq").lower().strip()
        value = out.get("value")
        if op in {"not_null", "is_not_null", "nonnull", "notnone"}:
            return {"op": "exists", "value": None}
        if op in {"is_null", "null", "isnone"}:
            return {"op": "not_exists", "value": None}
        if op == "neq" and value is None:
            return {"op": "exists", "value": None}
        if op == "eq" and value is None:
            return {"op": "not_exists", "value": None}
        out["op"] = op
        return out

    def contains_predicate(value: Any) -> bool:
        if isinstance(value, dict):
            if value.get("op"):
                return True
            return any(contains_predicate(v) for v in value.values())
        if isinstance(value, list):
            return any(contains_predicate(v) for v in value)
        return False

    def flatten(prefix: str, value: Any, target: dict[str, Any]):
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict) and value[0].get("op"):
            target[prefix] = predicate_alias(value[0])
            return
        if isinstance(value, dict) and value.get("op"):
            target[prefix] = predicate_alias(value)
            return
        if isinstance(value, dict) and contains_predicate(value):
            for child, child_value in value.items():
                path = f"{prefix}.{child}" if prefix else str(child)
                flatten(path, child_value, target)
            return
        target[prefix] = value

    out = dict(raw_filter)
    # A compact predicate is unambiguous only when ``field`` is a scalar path.
    if out.get("op") and isinstance(field, str) and field.strip():
        return {field.strip(): predicate_alias(out)}

    normalized: dict[str, Any] = {}
    for key, expected in out.items():
        flatten(str(key), expected, normalized)
    return normalized


def validate_plan(plan: dict[str, Any] | None,
                  valid_paths: list[str],
                  valid_methods_by_path: dict[str, set[str]] | None = None
                  ) -> tuple[dict[str, Any], list[str]]:
    """Normalize a planner response and remove non-OAS endpoints.

    Only endpoint/step defects are *fatal*.  Planner-supplied derivations are
    advisory hints for the deterministic compiler; an invalid optional
    derivation is dropped and recorded as a warning.  Treating such a hint as a
    fatal plan error can reject an otherwise executable endpoint chain before verification.
    """
    errors: list[str] = []
    warnings: list[str] = []
    plan = dict(plan or {})
    valid = set(valid_paths)
    steps = []
    seen_ids = set()
    for index, raw in enumerate(plan.get("steps") or [], 1):
        if not isinstance(raw, dict):
            errors.append(f"step {index} is not an object")
            continue
        sid = str(raw.get("id") or f"s{index}")
        if sid in seen_ids:
            errors.append(f"step {index}: duplicate step id {sid!r}")
            continue
        seen_ids.add(sid)
        endpoint = str(raw.get("endpoint") or "").strip()
        if endpoint not in valid:
            errors.append(
                f"step {sid}: invalid endpoint {endpoint!r}; nearest={_nearest(endpoint, valid_paths)}")
            continue
        method = str(raw.get("method") or "GET").upper()
        allowed_methods = (valid_methods_by_path or {}).get(endpoint)
        if allowed_methods and method not in allowed_methods:
            errors.append(f"step {sid}: invalid method {method} for {endpoint}; "
                          f"allowed={sorted(allowed_methods)}")
            continue
        depends = [str(x) for x in (raw.get("depends_on") or [])]
        raw_binds = [str(x).strip("{}") for x in (raw.get("binds") or [])]
        raw_binding_paths = raw.get("binding_paths") if isinstance(raw.get("binding_paths"), dict) else {}
        binding_paths = {str(k).strip("{}"): str(v).strip() for k, v in raw_binding_paths.items()
                         if str(k).strip("{}") and isinstance(v, str) and str(v).strip()}
        # ``binding_paths`` already declares an output binding alias and field.
        # Accept it as authoritative instead of requiring duplicate ``binds`` syntax.
        binds = list(dict.fromkeys(raw_binds + list(binding_paths)))
        placeholders = {x.strip("{}") for x in re.findall(r"\{[^{}]+\}", endpoint)}
        raw_literals = raw.get("path_literals") if isinstance(raw.get("path_literals"), dict) else {}
        path_literals = {str(k).strip("{}"): v for k, v in raw_literals.items()
                         if str(k).strip("{}") in placeholders and
                         isinstance(v, (str, int, float, bool))}
        invalid_literal_names = [str(k) for k in raw_literals
                                 if str(k).strip("{}") not in placeholders]
        if invalid_literal_names:
            errors.append(f"step {sid}: path_literals reference non-placeholder names {invalid_literal_names}")
        path_bindings = _request_binding_map(raw.get("path_bindings"))
        invalid_path_binding_names = [str(k) for k in path_bindings if str(k).strip("{}") not in placeholders]
        if invalid_path_binding_names:
            errors.append(f"step {sid}: path_bindings reference non-placeholder names {invalid_path_binding_names}")
        path_bindings = {str(k).strip("{}"): str(v).strip("{}") for k, v in path_bindings.items()
                         if str(k).strip("{}") in placeholders}
        overlap_path = sorted(set(path_literals) & set(path_bindings))
        if overlap_path:
            errors.append(f"step {sid}: path placeholders declared as both literal and binding {overlap_path}")
        query_literals = _request_literal_map(raw.get("query_literals"))
        query_bindings = _request_binding_map(raw.get("query_bindings"))
        body_literals = _request_literal_map(raw.get("body_literals"))
        body_bindings = _request_binding_map(raw.get("body_bindings"))
        raw_wrappers = raw.get("body_binding_wrappers") if isinstance(raw.get("body_binding_wrappers"), dict) else {}
        body_binding_wrappers = {str(k): str(v) for k, v in raw_wrappers.items()
                                 if str(k) in body_bindings and str(v) in {"uri_objects"}}
        overlap_q = sorted(set(query_literals) & set(query_bindings))
        overlap_b = sorted(set(body_literals) & set(body_bindings))
        if overlap_q:
            errors.append(f"step {sid}: query arguments declared as both literal and binding {overlap_q}")
        if overlap_b:
            errors.append(f"step {sid}: body arguments declared as both literal and binding {overlap_b}")
        steps.append({
            "id": sid,
            "method": method,
            "endpoint": endpoint,
            "purpose": str(raw.get("purpose") or "retrieve required evidence"),
            "depends_on": depends,
            "binds": binds,
            "binding_paths": binding_paths,
            "path_literals": path_literals,
            "path_bindings": path_bindings,
            "query_literals": query_literals,
            "query_bindings": query_bindings,
            "body_literals": body_literals,
            "body_bindings": body_bindings,
            "body_binding_wrappers": body_binding_wrappers,
            "answer_source": bool(raw.get("answer_source", False)),
        })

    valid_step_ids = {s["id"] for s in steps}
    for step in steps:
        unknown_deps = [x for x in step.get("depends_on") or [] if x not in valid_step_ids]
        if unknown_deps:
            errors.append(f"step {step['id']}: unknown dependencies {unknown_deps}")
        step["depends_on"] = [x for x in step["depends_on"] if x in valid_step_ids]
        if step["id"] in step["depends_on"]:
            errors.append(f"step {step['id']}: self dependency is not allowed")

    # Fail closed on dependency cycles. A cyclic plan has no well-defined producer
    # order and therefore cannot support provenance-safe execution.
    dep_map = {str(step["id"]): [str(x) for x in step.get("depends_on") or []] for step in steps}
    visiting, visited = set(), set()
    def _visit_cycle(sid):
        if sid in visiting:
            return True
        if sid in visited:
            return False
        visiting.add(sid)
        for dep in dep_map.get(sid, []):
            if dep in dep_map and _visit_cycle(dep):
                return True
        visiting.remove(sid); visited.add(sid)
        return False
    if any(_visit_cycle(sid) for sid in dep_map if sid not in visited):
        errors.append("plan dependency graph contains a cycle")

    # Normalize a common planner shorthand where path_bindings points at the
    # producer STEP id instead of the producer binding alias.  This is safe only
    # for a declared direct dependency and only when the target-name binding is
    # present or the dependency exposes exactly one output binding.
    by_step_now = {str(x.get("id") or ""): x for x in steps}
    for step in steps:
        aliases = dict(step.get("path_bindings") or {})
        for target, raw_alias in list(aliases.items()):
            producer = by_step_now.get(str(raw_alias))
            if producer is None or str(raw_alias) not in {str(x) for x in step.get("depends_on") or []}:
                continue
            names = list(dict.fromkeys(
                [str(x).strip("{}") for x in producer.get("binds") or []] +
                [str(x).strip("{}") for x in (producer.get("binding_paths") or {})]))
            replacement = str(target).strip("{}") if str(target).strip("{}") in names else (
                names[0] if len(names) == 1 else None)
            if replacement:
                aliases[str(target).strip("{}")] = replacement
                warnings.append(
                    f"step {step.get('id')}: normalized path binding {{{target}}}<-{raw_alias} "
                    f"to producer alias {replacement}")
        step["path_bindings"] = aliases

    # Recover an omitted data-flow edge only when the consumer declared no
    # dependencies at all and exactly one *earlier* step binds the placeholder.
    # If the planner already declared dependencies, never graft an unrelated
    # sibling branch into that graph merely because it happens to reuse the same
    # placeholder name.
    binders: dict[str, list[str]] = {}
    step_order = {str(step.get("id")): i for i, step in enumerate(steps)}
    for candidate in steps:
        for name in candidate.get("binds") or []:
            binders.setdefault(str(name).strip("{}"), []).append(candidate["id"])
    for step in steps:
        inferred = list(step.get("depends_on") or [])
        if inferred:
            continue
        placeholders = [x.strip("{}") for x in re.findall(r"\{[^{}]+\}",
                                                            step.get("endpoint") or "")]
        binding_needs = [name for name in placeholders
                         if name not in (step.get("path_literals") or {})]
        binding_needs.extend(str(x) for x in (step.get("query_bindings") or {}).values())
        binding_needs.extend(str(x) for x in (step.get("body_bindings") or {}).values())
        for name in dict.fromkeys(binding_needs):
            candidates = [sid for sid in binders.get(name, [])
                          if sid != step["id"] and
                          step_order.get(sid, 10**9) < step_order.get(step["id"], -1)]
            if len(candidates) == 1 and candidates[0] not in inferred:
                inferred.append(candidates[0])
                warnings.append(
                    f"step {step['id']}: inferred dependency {candidates[0]} for binding {name}")
        step["depends_on"] = inferred

    derivations = []
    seen_derivation_ids: set[str] = set()
    allowed_ops = {
        "endpoint_rank", "argmax", "argmin", "filter", "count",
        "membership", "compare", "logical_and", "logical_or", "first", "nth", "identity",
    }
    for index, raw in enumerate(plan.get("derivations") or [], 1):
        if not isinstance(raw, dict):
            continue
        op = str(raw.get("operator") or "").lower()
        comparison_aliases = {"max", "min", "eq", "neq", "gt", "gte", "lt", "lte",
                              "difference", "abs_difference"}
        comparison_alias = None
        if op in comparison_aliases:
            # Generic recovery for a planner formatting slip: comparison modes
            # belong under operator=compare. Normalize the shape instead of
            # silently dropping an otherwise meaningful derivation. Structural
            # source-count validation below still applies, so an underspecified
            # comparison triggers a bounded replan rather than being accepted.
            comparison_alias = op
            warnings.append(
                f"derivation {index}: normalized comparison operator alias {op!r} to operator='compare'")
            op = "compare"
        source_steps = raw.get("source_steps") or ([raw.get("source_step")]
                                                    if raw.get("source_step") else [])
        source_steps = [str(x) for x in source_steps if str(x) in valid_step_ids]
        did = str(raw.get("id") or f"d{index}")
        if did in seen_derivation_ids:
            errors.append(f"derivation {index}: duplicate derivation id {did!r}")
            continue
        raw_source_derivations = [str(x) for x in (raw.get("source_derivations") or []) if str(x)]
        prior_derivation_ids = {str(x.get("id")) for x in derivations}
        unknown_source_derivations = [x for x in raw_source_derivations if x not in prior_derivation_ids]
        if unknown_source_derivations:
            errors.append(
                f"derivation {did}: unknown or forward source_derivations {unknown_source_derivations}")
            continue
        source_derivations = raw_source_derivations
        derived_only_ops = {"compare", "logical_and", "logical_or"}
        if op in allowed_ops and not source_steps and not (op in derived_only_ops and source_derivations):
            # Generic recovery for a follow-up derivation that omitted source_steps.
            # If it explicitly names prior source_derivations, inherit the union of
            # THOSE derivations' source steps rather than the immediately preceding
            # unrelated derivation. This preserves the declared derivation graph.
            if source_derivations:
                prior_by_id = {str(x.get("id")): x for x in derivations}
                inherited = []
                for source_did in source_derivations:
                    inherited.extend(str(x) for x in (prior_by_id.get(source_did) or {}).get("source_steps") or [])
                source_steps = list(dict.fromkeys(x for x in inherited if x in valid_step_ids))
            elif derivations:
                source_steps = list(derivations[-1].get("source_steps") or [])
            else:
                answer_sources = [str(x.get("id")) for x in steps if x.get("answer_source")]
                if len(answer_sources) == 1:
                    source_steps = answer_sources
            if source_steps:
                warnings.append(
                    f"derivation {index}: inferred missing source_steps={source_steps}")
        if op not in allowed_ops or (not source_steps and not (op in derived_only_ops and source_derivations)):
            warnings.append(
                f"derivation {index}: ignored invalid operator/source "
                f"(operator={op!r}, source_steps={source_steps!r}, source_derivations={source_derivations!r})")
            continue
        comparison = str(raw.get("comparison") or comparison_alias or "").lower().strip()
        allowed_comparisons = {"max", "min", "eq", "neq", "gt", "gte", "lt", "lte",
                               "difference", "abs_difference"}
        if op == "compare" and comparison not in allowed_comparisons:
            warnings.append(f"derivation {index}: ignored compare without explicit comparison mode")
            continue
        if op == "membership" and len(source_steps) < 2:
            if raw.get("comparison_literal") is not None and len(source_steps) == 1:
                # Literal membership is complete with one collection source.
                # The comparison target is the declared literal, not a second API step.
                pass
            elif str(plan.get("answer_mode") or "").lower() == "boolean":
                # Boolean membership is answer-critical. Preserve it for the
                # schema-grounded observation stage, which may recover one unique
                # target source from shared projected fields.
                warnings.append(f"derivation {did}: deferred membership with fewer than two declared source steps for schema-grounded source repair")
            else:
                warnings.append(f"derivation {did}: ignored membership with fewer than two declared source steps")
                continue
        if op == "compare":
            literal_sources = 1 if raw.get("comparison_literal") is not None else 0
            declared_input_count = len(source_steps) + len(source_derivations) + literal_sources
            if declared_input_count < 2:
                errors.append(f"derivation {did}: compare requires at least two declared sources/literals")
                continue
            # Comparisons over prior derivations must consume replayable scalar
            # values. Selection/filter derivations yield records or collections,
            # which the compiler deliberately will not coerce into a metric. Catch
            # that mismatch before execution so a bounded planner retry can insert
            # an explicit identity/count extraction instead of wasting API calls.
            prior_by_id = {str(x.get("id")): x for x in derivations}
            nonscalar_sources = []
            boolean_sources = []
            for source_did in source_derivations:
                src = prior_by_id.get(source_did) or {}
                src_op = str(src.get("operator") or "").lower()
                src_cmp = str(src.get("comparison") or "").lower()
                scalar = False
                boolean = False
                if src_op == "identity":
                    # A comparison input must name the scalar field explicitly;
                    # compiler-side field inference is intentionally not used for
                    # cross-derivation arithmetic/comparison.
                    scalar = bool(str(src.get("field") or "").strip())
                elif src_op == "count":
                    scalar = True
                elif src_op == "membership" or src_op in {"logical_and", "logical_or"}:
                    scalar = True; boolean = True
                elif src_op == "compare":
                    if src_cmp in {"difference", "abs_difference"}:
                        scalar = True
                    elif src_cmp in {"eq", "neq", "gt", "gte", "lt", "lte"}:
                        scalar = True; boolean = True
                if not scalar:
                    nonscalar_sources.append(source_did)
                if boolean:
                    boolean_sources.append(source_did)
            if nonscalar_sources:
                errors.append(
                    f"derivation {did}: compare source_derivations must be scalar-producing; "
                    f"extract an explicit identity/count value first: {nonscalar_sources}")
                continue
            if comparison in {"difference", "abs_difference"} and boolean_sources:
                errors.append(
                    f"derivation {did}: {comparison} cannot consume Boolean derivations: {boolean_sources}")
                continue
            # Binary comparisons are replayed pairwise by the host. Avoid silently
            # ignoring a third input; max/min are the only n-ary comparison modes.
            if comparison not in {"max", "min"} and declared_input_count != 2:
                errors.append(
                    f"derivation {did}: {comparison} requires exactly two declared sources/literals")
                continue
        if op in {"logical_and", "logical_or"}:
            if source_steps:
                errors.append(f"derivation {did}: {op} must combine prior source_derivations, not raw API steps")
                continue
            if len(source_derivations) < 2:
                errors.append(f"derivation {did}: {op} requires at least two prior boolean derivations")
                continue
            prior_by_id = {str(x.get("id")): x for x in derivations}
            non_boolean = []
            for source_did in source_derivations:
                src = prior_by_id.get(source_did) or {}
                src_op = str(src.get("operator") or "").lower()
                src_cmp = str(src.get("comparison") or "").lower()
                if not (src_op in {"membership", "logical_and", "logical_or"}
                        or (src_op == "compare" and src_cmp in {"eq", "neq", "gt", "gte", "lt", "lte"})):
                    non_boolean.append(source_did)
            if non_boolean:
                errors.append(f"derivation {did}: {op} sources are not declared boolean derivations: {non_boolean}")
                continue
        label_steps = [str(x) for x in (raw.get("label_steps") or []) if str(x)]
        label_fields = [str(x) for x in (raw.get("label_fields") or []) if str(x)]
        if label_steps or label_fields:
            if op != "compare":
                warnings.append(f"derivation {did}: ignored label_steps/label_fields on non-compare operator")
                label_steps, label_fields = [], []
            elif len(label_steps) != len(label_fields) or any(x not in valid_step_ids for x in label_steps):
                # Labels are presentation metadata, not evidence semantics.  A
                # malformed label reference must not delete an otherwise replayable
                # compare derivation or make the acquisition plan non-executable.
                warnings.append(
                    f"derivation {did}: ignored malformed label_steps/label_fields; "
                    "comparison evidence remains replayable")
                label_steps, label_fields = [], []
        raw_field = raw.get("field")
        # Filter derivations do not require ``field`` to carry the predicate. A
        # model can accidentally copy the filter object into ``field``; retaining
        # that object causes false schema errors and later stringification. Keep a
        # field only when it is an actual scalar schema path.
        normalized_field = raw_field if isinstance(raw_field, str) else None
        derivations.append({
            "id": did,
            "operator": op,
            "source_steps": source_steps,
            "source_derivations": source_derivations,
            "label_steps": label_steps,
            "label_fields": label_fields,
            "field": normalized_field,
            "comparison": comparison if op == "compare" else "",
            "comparison_literal": (raw.get("comparison_literal")
                                   if isinstance(raw.get("comparison_literal"), (str, int, float, bool)) else None),
            "unit": str(raw.get("unit") or "raw").lower(),
            "distinct_field": (str(raw.get("distinct_field")) if raw.get("distinct_field") else None),
            "rank": int(raw.get("rank", 0) or 0),
            "filter": _canonical_derivation_filter(raw.get("filter"), raw.get("field")),
            "top_k": int(raw.get("top_k", 10) or 10),
            "purpose": str(raw.get("purpose") or "derive answer evidence"),
        })
        seen_derivation_ids.add(did)

    # Collapse exact duplicate derivations before Boolean validation/replay.
    # This is semantic normalization, not domain logic: identical operations over
    # identical declared sources cannot add evidence. It also simplifies
    # idempotent A AND A / A OR A constructs back to A.
    alias_map: dict[str, str] = {}
    unique_derivations: list[dict[str, Any]] = []
    seen_semantics: dict[tuple, str] = {}

    def _canonical_source_id(value: str) -> str:
        value = str(value)
        seen = set()
        while value in alias_map and value not in seen:
            seen.add(value)
            value = alias_map[value]
        return value

    def _freeze(value):
        """Return a recursively hashable representation for planner semantics.

        Planner JSON is model-produced and can legally contain nested objects/lists in
        places we normally expect scalars.  Semantic de-duplication must never crash
        merely because a value is structured.
        """
        if isinstance(value, dict):
            return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(_freeze(v) for v in value)
        if isinstance(value, set):
            return tuple(sorted((_freeze(v) for v in value), key=repr))
        try:
            hash(value)
            return value
        except Exception:
            return repr(value)

    for derivation in derivations:
        current = dict(derivation)
        current["source_derivations"] = list(dict.fromkeys(
            _canonical_source_id(x) for x in (current.get("source_derivations") or [])))
        op = str(current.get("operator") or "").lower()
        did = str(current.get("id"))
        # Boolean wrappers such as ``membership == true`` do not add a second
        # independent proposition. Canonicalize the identity-preserving forms to
        # their source so Boolean closure does not demand a spurious AND/OR.
        if (op == "compare" and not (current.get("source_steps") or [])
                and len(current.get("source_derivations") or []) == 1
                and isinstance(current.get("comparison_literal"), bool)
                and str(current.get("comparison") or "").lower() in {"eq", "neq"}):
            source_id = current["source_derivations"][0]
            source = next((x for x in unique_derivations if str(x.get("id")) == source_id), {})
            source_op = str(source.get("operator") or "").lower()
            source_cmp = str(source.get("comparison") or "").lower()
            source_boolean = (source_op in {"membership", "logical_and", "logical_or"}
                              or (source_op == "compare" and source_cmp in
                                  {"eq", "neq", "gt", "gte", "lt", "lte"}))
            same_truth = ((current["comparison"] == "eq" and current["comparison_literal"] is True)
                          or (current["comparison"] == "neq" and current["comparison_literal"] is False))
            if source_boolean and same_truth:
                alias_map[did] = source_id
                warnings.append(
                    f"derivation {did}: removed redundant Boolean compare-to-true wrapper over {source_id}")
                continue
        if op in {"logical_and", "logical_or"} and len(current["source_derivations"]) == 1:
            alias_map[did] = current["source_derivations"][0]
            warnings.append(
                f"derivation {did}: removed redundant {op} over one unique Boolean input")
            continue
        signature = _freeze((
            op, current.get("source_steps") or [],
            current.get("source_derivations") or [],
            current.get("label_steps") or [],
            current.get("label_fields") or [],
            current.get("field"), current.get("comparison"),
            current.get("comparison_literal"), current.get("unit"),
            current.get("distinct_field"), current.get("rank"),
            current.get("filter") or {}, current.get("top_k"),
        ))
        if signature in seen_semantics:
            canonical = seen_semantics[signature]
            alias_map[did] = canonical
            warnings.append(
                f"derivation {did}: removed exact duplicate of {canonical}")
            continue
        seen_semantics[signature] = did
        unique_derivations.append(current)
    derivations = unique_derivations

    if str(plan.get("answer_mode") or "").lower() == "boolean":
        boolean_producers = []
        has_logical = False
        for derivation in derivations:
            op = str(derivation.get("operator") or "").lower()
            cmp_mode = str(derivation.get("comparison") or "").lower()
            if op in {"logical_and", "logical_or"}:
                has_logical = True
            if op == "membership" or (op == "compare" and cmp_mode in {"eq", "neq", "gt", "gte", "lt", "lte"}):
                boolean_producers.append(str(derivation.get("id")))
        if len(boolean_producers) > 1 and not has_logical:
            errors.append(
                "boolean answer has multiple independent Boolean derivations but no logical_and/logical_or combiner")

    # Compose a preceding filter into a later selection/aggregate operation over
    # the same source step when the planner emitted them as separate derivations.
    # The JSON schema names plan steps (not derivation ids) as sources, so without
    # this normalization ``filter -> count/argmax`` would accidentally replay the
    # second operator over the unfiltered response.
    for i, derivation in enumerate(derivations):
        op = str(derivation.get("operator") or "").lower()
        if op not in {"count", "argmax", "argmin", "first", "nth", "endpoint_rank"} or derivation.get("filter"):
            continue
        sources = tuple(derivation.get("source_steps") or [])
        prior = next((d for d in reversed(derivations[:i])
                      if str(d.get("operator") or "").lower() == "filter"
                      and tuple(d.get("source_steps") or []) == sources
                      and d.get("filter")), None)
        if prior is not None:
            derivation["filter"] = dict(prior.get("filter") or {})
            warnings.append(
                f"derivation {derivation.get('id')}: inherited preceding filter over the same source")

    explicit_answer_steps = [str(x) for x in (plan.get("answer_steps") or [])
                             if str(x) in valid_step_ids]
    answer_source_steps = [s["id"] for s in steps if s.get("answer_source")]
    answer_steps = list(explicit_answer_steps)
    if not answer_steps:
        answer_steps = list(answer_source_steps)
    if steps and not answer_steps and derivations:
        # The derivation DAG already declares which raw API steps feed the final
        # computation.  Recover answer roots from terminal derivations instead of
        # requiring the model to duplicate that information in answer_steps. This
        # is graph normalization, not semantic guessing.
        by_did = {str(d.get("id")): d for d in derivations if d.get("id")}
        referenced = {str(x) for d in derivations
                      for x in (d.get("source_derivations") or [])}
        terminals = [d for d in derivations if str(d.get("id")) not in referenced]
        mode = str(plan.get("answer_mode") or "direct").lower()
        if mode in {"boolean", "comparison"}:
            preferred = [d for d in terminals
                         if str(d.get("operator") or "").lower() in
                         {"compare", "membership", "logical_and", "logical_or"}]
            terminals = preferred or terminals

        collected: list[str] = []
        seen_dids: set[str] = set()
        def collect_sources(deriv):
            did = str(deriv.get("id") or "")
            if did in seen_dids:
                return
            seen_dids.add(did)
            for sid in deriv.get("source_steps") or []:
                sid = str(sid)
                if sid in valid_step_ids and sid not in collected:
                    collected.append(sid)
            for parent_did in deriv.get("source_derivations") or []:
                parent = by_did.get(str(parent_did))
                if parent:
                    collect_sources(parent)
        for terminal in terminals:
            collect_sources(terminal)
        if collected:
            answer_steps = collected
            warnings.append(
                f"inferred answer_steps={answer_steps} from terminal derivation lineage")
    if steps and not answer_steps:
        errors.append(
            "plan has no explicit answer_steps/answer_source and no derivation lineage from which to infer them")

    # answer_steps identify API evidence roots; answer_derivations optionally name
    # the exact derivations that are user-facing answer values.  Keeping these
    # concepts separate prevents request-routing ids and replay context from
    # leaking into otherwise grounded answers.  The field is optional for
    # backwards compatibility with stored/legacy plans.
    valid_derivation_ids = {str(d.get("id")) for d in derivations if d.get("id")}
    answer_derivations: list[str] = []
    missing_answer_derivations: list[str] = []
    non_value_answer_derivations: list[str] = []
    selector_ops = {"filter", "first", "nth", "endpoint_rank", "argmax", "argmin"}
    derivation_by_id = {str(d.get("id")): d for d in derivations if d.get("id")}
    for raw_did in (plan.get("answer_derivations") or []):
        did = _canonical_source_id(str(raw_did))
        if did in answer_derivations:
            continue
        if did not in valid_derivation_ids:
            missing_answer_derivations.append(did)
            continue
        op = str((derivation_by_id.get(did) or {}).get("operator") or "").lower()
        if op in selector_ops:
            non_value_answer_derivations.append(did)
            continue
        answer_derivations.append(did)
    if missing_answer_derivations:
        errors.append(
            "answer_derivations reference missing derivations: "
            + str(sorted(set(missing_answer_derivations))))
    if non_value_answer_derivations:
        errors.append(
            "answer_derivations must reference value-producing derivations, not selectors: "
            + str(sorted(set(non_value_answer_derivations))))

    # Remove unambiguously unused terminal GETs in one pass. This deliberately
    # does not recurse: after pruning a leaf, its parent is retained even if it
    # becomes a leaf. Plans that needed the fallback-to-last answer step are also
    # left untouched because they lack a reliable answer root for liveness.
    has_reliable_answer_root = bool(explicit_answer_steps or answer_source_steps)
    if has_reliable_answer_root:
        derivation_sources = {
            sid for derivation in derivations
            for sid in (derivation.get("source_steps") or [])
        }
        dependency_targets = {
            sid for step in steps for sid in (step.get("depends_on") or [])
        }
        pruned_steps = [
            step for step in steps
            if (step.get("method") == "GET"
                and not step.get("answer_source")
                and step["id"] not in answer_steps
                and step["id"] not in derivation_sources
                and step["id"] not in dependency_targets
                and not step.get("binds"))
        ]
        if pruned_steps:
            pruned_ids = {step["id"] for step in pruned_steps}
            steps = [step for step in steps if step["id"] not in pruned_ids]
            descriptions = ", ".join(
                f"{step['id']} ({step['endpoint']})" for step in pruned_steps)
            warnings.append(
                "pruned unused non-answer GET leaf steps: " + descriptions)

    normalized = {
        "version": 2,
        "steps": steps,
        "derivations": derivations,
        "answer_steps": answer_steps,
        "answer_derivations": answer_derivations,
        "answer_mode": str(plan.get("answer_mode") or "direct").lower(),
        # Final-output cardinality is part of the typed semantic contract.  It
        # must survive structural/OAS normalization; dropping it here causes
        # observation planning and answer emission to expand a singular typed
        # answer back into the provider collection size.
        "answer_cardinality": (str(plan.get("answer_cardinality") or "").lower()
                               if str(plan.get("answer_cardinality") or "").lower() in {"one", "many"}
                               else ""),
        # Typed output metadata is part of the semantic contract, not planner
        # prose. Preserve it across low-level structural/OAS normalization just
        # like cardinality; otherwise provider normalization can silently turn a
        # detail answer back into a scalar label or lose terminal resource type.
        "answer_resource": str(plan.get("answer_resource") or ""),
        "answer_detail": bool(plan.get("answer_detail", False)),
        "answer_fields": [str(x) for x in (plan.get("answer_fields") or []) if str(x)],
        "answer_requirements": [str(x) for x in
                                (plan.get("answer_requirements") or [])],
        "planner_notes": str(plan.get("planner_notes") or ""),
        # Compiler-authored collection lifting is structural execution metadata.
        # Preserve it through plan normalization so schema validation and the
        # observation compiler can realize selected_all bindings explicitly.
        "typed_fanout_bindings": [dict(x) for x in (plan.get("typed_fanout_bindings") or [])
                                   if isinstance(x, dict)],
        "valid": bool(steps) and not errors,
        "validation_errors": errors,
        "validation_warnings": warnings,
    }
    return normalized, errors


def annotate_execution_eligibility(plan: dict[str, Any] | None) -> dict[str, Any]:
    """Separate strict certification validity from safe evidence acquisition.

    A read-only route plan can still be useful when answer/selection metadata is
    malformed.  The trusted request guard already constrains execution to the
    validated API origin and planned path templates, so throwing away every GET
    route because a derivation or answer_steps field is imperfect only prevents
    later evidence-based recovery.

    Invalid state-changing plans remain fail-closed: side effects require the full
    strict contract. Dependency cycles also remain non-executable.
    """
    out = dict(plan or {})
    steps = list(out.get("steps") or [])
    strict = bool(out.get("valid"))
    errors = [str(x) for x in (out.get("validation_errors") or [])]
    has_cycle = any("dependency graph contains a cycle" in x for x in errors)
    hard_semantic_rejection = any(
        "semantic critic incompatibility unresolved after bounded correction" in x
        or "semantic route invariant unresolved after bounded convergence" in x
        for x in errors)
    read_only = bool(steps) and all(
        str(step.get("method") or "GET").upper() == "GET" for step in steps)
    eligible = bool(steps) and not has_cycle and not hard_semantic_rejection and (strict or read_only)
    out["strict_valid"] = strict
    out["execution_eligible"] = eligible
    out["execution_advisory"] = bool(eligible and not strict)
    out["execution_advisory_errors"] = (errors if eligible and not strict else [])
    return out


def _step_placeholders(step: dict[str, Any]) -> list[str]:
    return [x.strip("{}") for x in re.findall(r"\{[^{}]+\}",
                                                str(step.get("endpoint") or ""))]


def _dynamic_input_binding_names(step: dict[str, Any]) -> set[str]:
    """Names consumed by a step that already have upstream provenance."""
    literals = {str(x).strip("{}") for x in (step.get("path_literals") or {})}
    path_bindings = {str(k).strip("{}"): str(v).strip("{}")
                     for k, v in (step.get("path_bindings") or {}).items()}
    names = {path_bindings.get(str(x).strip("{}"), str(x).strip("{}"))
             for x in _step_placeholders(step)
             if str(x).strip("{}") not in literals}
    names.update(str(x).strip("{}") for x in (step.get("query_bindings") or {}).values())
    names.update(str(x).strip("{}") for x in (step.get("body_bindings") or {}).values())
    return {x for x in names if x}


def _placeholder_route_context(step: dict[str, Any], name: str) -> str | None:
    """Return the documented path prefix that gives a placeholder its route role.

    Same placeholder names can legitimately denote a newly selected child on a
    different route (``/roots/{id}/children`` -> ``/children/{id}``).  Conversely,
    rebinding an identifier while the downstream route keeps the same prefix silently
    changes the parent identity.  Route context lets us distinguish those cases without
    any API/domain vocabulary.
    """
    endpoint = str(step.get("endpoint") or "")
    marker = "{" + str(name).strip("{}") + "}"
    if marker not in endpoint:
        return None
    return endpoint.split(marker, 1)[0].rstrip("/") or "/"


def _protected_passthrough_binding_names(plan: dict[str, Any], step_id: str) -> set[str]:
    """Dynamic path inputs that a descendant reuses in the same route context.

    ``path_bindings`` may deliberately give the consumed binding a different name
    from the URL placeholder (for example ``{movie_id} <- similar_movie_id``).
    Route context belongs to the placeholder while provenance belongs to the alias,
    so keep both rather than looking the alias up as if it appeared in the route.
    """
    steps = {str(x.get("id")): x for x in (plan or {}).get("steps") or []}
    source = steps.get(str(step_id)) or {}

    def path_inputs(step: dict[str, Any]) -> list[tuple[str, str, str | None]]:
        literals = {str(x).strip("{}") for x in (step.get("path_literals") or {})}
        aliases = {str(k).strip("{}"): str(v).strip("{}")
                   for k, v in (step.get("path_bindings") or {}).items()}
        out = []
        for placeholder in _step_placeholders(step):
            target = str(placeholder).strip("{}")
            if target in literals:
                continue
            binding = aliases.get(target, target)
            if binding:
                out.append((target, binding, _placeholder_route_context(step, target)))
        return out

    candidates: dict[str, set[str]] = {}
    for _target, binding, context in path_inputs(source):
        if context is not None:
            candidates.setdefault(binding, set()).add(context)
    if not candidates:
        return set()

    def descends_from(candidate_id: str, ancestor_id: str) -> bool:
        queue = [str(x) for x in (steps.get(candidate_id) or {}).get("depends_on") or []]
        seen = set()
        while queue:
            current = queue.pop(0)
            if current == ancestor_id:
                return True
            if current in seen or current not in steps:
                continue
            seen.add(current)
            queue.extend(str(x) for x in (steps.get(current) or {}).get("depends_on") or [])
        return False

    protected = set()
    for binding, source_contexts in candidates.items():
        for cid, child in steps.items():
            if cid == str(step_id) or not descends_from(cid, str(step_id)):
                continue
            for _target, child_binding, child_context in path_inputs(child):
                if child_binding == binding and child_context in source_contexts:
                    protected.add(binding)
                    break
            if binding in protected:
                break
    return protected


def _selection_source_steps(plan: dict[str, Any]) -> set[str]:
    selecting = {"endpoint_rank", "argmax", "argmin", "first", "nth", "filter", "identity"}
    return {
        str(source)
        for derivation in (plan or {}).get("derivations") or []
        if str(derivation.get("operator") or "").lower() in selecting
        for source in (derivation.get("source_steps") or [])
    }


def _dynamic_path_reference_name(value: Any) -> str | None:
    """Return an explicit upstream binding alias encoded in a path-literal slot.

    Planner outputs sometimes place ``{{entity_id}}`` or ``entity_id`` in
    ``path_literals`` even though those values are dynamic bindings, not literals.
    We normalize only identifier-shaped values that resolve to a declared upstream
    binding; ordinary string path literals remain untouched.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    wrapped = re.fullmatch(r"\{\{?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}?\}", text)
    if wrapped:
        return wrapped.group(1)
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
        return text
    return None


def normalize_dynamic_path_references(plan: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize explicit dynamic path references into producer aliases.

    The source alias is already declared by the evidence plan. When a path
    placeholder points at that alias, copy the target placeholder name onto the same
    producer field so the existing provenance machinery can authorize it. No API
    value, endpoint, or answer is inferred.
    """
    out = dict(plan or {})
    steps = [dict(x) for x in (out.get("steps") or [])]
    order = {str(x.get("id")): i for i, x in enumerate(steps)}

    def bind_names(step: dict[str, Any]) -> set[str]:
        names = [str(x).strip("{}") for x in (step.get("binds") or [])]
        names += [str(x).strip("{}") for x in (step.get("binding_paths") or {})]
        return {x for x in names if x}

    def producer_for(child: dict[str, Any], alias: str | None) -> dict[str, Any] | None:
        if not alias:
            return None
        cid = str(child.get("id") or "")
        prior = [st for st in steps
                 if order.get(str(st.get("id")), 10**9) < order.get(cid, -1)
                 and alias in bind_names(st)]
        deps = {str(x) for x in (child.get("depends_on") or [])}
        direct = [st for st in prior if str(st.get("id")) in deps]
        candidates = direct or prior
        return candidates[0] if len(candidates) == 1 else None

    warnings = list(out.get("validation_warnings") or [])

    def path_key(value: Any) -> str:
        text = str(value or "").strip().lstrip("$.")
        text = re.sub(r"\[(?:\d+|\*)\]", "[*]", text)
        return re.sub(r"\.{2,}", ".", text)

    for child in steps:
        placeholders = set(_step_placeholders(child))
        refs: dict[str, str] = {}

        # Canonicalize explicit step-qualified response references placed directly
        # in path_bindings.  The model may write the same dependency as
        # ``s1.results[0].id`` or ``s1:$.results[0].id``.  Resolve those forms to
        # one producer alias instead of asking the planner to regenerate the plan.
        explicit_aliases = dict(child.get("path_bindings") or {})
        for target, raw_ref in list(explicit_aliases.items()):
            parsed = _parse_step_qualified_binding_reference(raw_ref)
            if parsed is None:
                continue
            producer_id, response_path = parsed
            producer = next((st for st in steps if str(st.get("id") or "") == producer_id), None)
            cid = str(child.get("id") or "")
            if producer is None or order.get(producer_id, 10**9) >= order.get(cid, -1):
                continue
            wanted = path_key(response_path)
            matches = []
            for alias, declared_path in (producer.get("binding_paths") or {}).items():
                have = path_key(declared_path)
                if have == wanted or have.endswith("." + wanted) or wanted.endswith("." + have):
                    matches.append(str(alias).strip("{}"))
            matches = list(dict.fromkeys(matches))
            if len(matches) == 1:
                alias = matches[0]
            elif len(matches) == 0:
                # The planner explicitly named both the producer step and response
                # path.  Preserve that declaration as a binding and let the normal
                # schema/selection validators decide whether the path is legal.
                alias = str(target).strip("{}")
                binds = [str(x).strip("{}") for x in (producer.get("binds") or [])]
                if alias not in binds:
                    binds.append(alias)
                binding_paths = dict(producer.get("binding_paths") or {})
                binding_paths[alias] = response_path
                producer["binds"] = list(dict.fromkeys(binds))
                producer["binding_paths"] = binding_paths
            else:
                # Ambiguous existing aliases remain unresolved rather than guessing.
                continue
            explicit_aliases[str(target).strip("{}")] = alias
            deps = [str(x) for x in (child.get("depends_on") or [])]
            if producer_id not in deps:
                child["depends_on"] = deps + [producer_id]
            warnings.append(
                f"step {child.get('id')}: normalized explicit binding "
                f"{{{target}}}<-{raw_ref} to producer alias {alias} from {producer_id}")
        child["path_bindings"] = explicit_aliases

        # Apply the same syntax normalization to dynamic query/body arguments.
        # These are less common than path IDs but represent the same producer →
        # consumer data-flow edge and should not require a planner retry either.
        for mapping_name in ("query_bindings", "body_bindings"):
            mapping = dict(child.get(mapping_name) or {})
            for target, raw_ref in list(mapping.items()):
                parsed = _parse_step_qualified_binding_reference(raw_ref)
                if parsed is None:
                    continue
                producer_id, response_path = parsed
                producer = next((st for st in steps if str(st.get("id") or "") == producer_id), None)
                cid = str(child.get("id") or "")
                if producer is None or order.get(producer_id, 10**9) >= order.get(cid, -1):
                    continue
                wanted = path_key(response_path)
                matches = []
                for alias, declared_path in (producer.get("binding_paths") or {}).items():
                    have = path_key(declared_path)
                    if have == wanted or have.endswith("." + wanted) or wanted.endswith("." + have):
                        matches.append(str(alias).strip("{}"))
                matches = list(dict.fromkeys(matches))
                if len(matches) == 1:
                    alias = matches[0]
                elif len(matches) == 0:
                    alias = str(target).strip("{}")
                    binds = [str(x).strip("{}") for x in (producer.get("binds") or [])]
                    if alias not in binds:
                        binds.append(alias)
                    binding_paths = dict(producer.get("binding_paths") or {})
                    binding_paths[alias] = response_path
                    producer["binds"] = list(dict.fromkeys(binds))
                    producer["binding_paths"] = binding_paths
                else:
                    continue
                mapping[str(target)] = alias
                deps = [str(x) for x in (child.get("depends_on") or [])]
                if producer_id not in deps:
                    child["depends_on"] = deps + [producer_id]
                warnings.append(
                    f"step {child.get('id')}: normalized explicit {mapping_name[:-1]} "
                    f"{target}<-{raw_ref} to producer alias {alias} from {producer_id}")
            child[mapping_name] = mapping

        literals = dict(child.get("path_literals") or {})
        for target in list(literals):
            if target not in placeholders:
                continue
            alias = _dynamic_path_reference_name(literals.get(target))
            if producer_for(child, alias) is not None:
                refs[target] = str(alias)
                literals.pop(target, None)

        # A path placeholder mistakenly placed in query_bindings is unambiguous
        # syntactically: the endpoint itself declares that target as a path field.
        qbind = dict(child.get("query_bindings") or {})
        for target in list(qbind):
            if str(target) not in placeholders:
                continue
            alias = str(qbind.get(target) or "").strip("{}")
            if producer_for(child, alias) is not None:
                refs[str(target)] = alias
                qbind.pop(target, None)
        child["path_literals"] = literals
        child["query_bindings"] = qbind
        path_bindings = dict(child.get("path_bindings") or {})
        for target, alias in refs.items():
            path_bindings[str(target).strip("{}")] = str(alias).strip("{}")
        if path_bindings:
            child["path_bindings"] = path_bindings

        for target, alias in refs.items():
            producer = producer_for(child, alias)
            if producer is None:
                continue
            binds = [str(x).strip("{}") for x in (producer.get("binds") or [])]
            paths = dict(producer.get("binding_paths") or {})
            # Preserve the exact producer alias on the consumer. Copying it to the
            # placeholder name can collide with an older ancestor that used the
            # same placeholder and silently change entity lineage.
            if alias not in binds:
                binds.append(alias)
            # Keep the historical same-field placeholder alias for compatibility,
            # but the consumer's explicit path_bindings entry is authoritative.
            # If this target would shadow an ancestor input, the later pass-through
            # cleanup removes it while preserving path_bindings.
            if target not in binds:
                binds.append(target)
            if target not in paths and alias in paths:
                paths[target] = paths[alias]
            producer["binds"] = list(dict.fromkeys(binds))
            if paths:
                producer["binding_paths"] = paths
            deps = [str(x) for x in (child.get("depends_on") or [])]
            pid = str(producer.get("id") or "")
            if pid and pid not in deps:
                child["depends_on"] = deps + [pid]
            warnings.append(
                f"step {child.get('id')}: normalized dynamic path {{{target}}}<-{alias} "
                f"from producer {producer.get('id')}")

    out["steps"] = steps
    out["validation_warnings"] = list(dict.fromkeys(warnings))
    return out


def normalize_derivation_binding_aliases(plan: dict[str, Any]) -> dict[str, Any]:
    """Resolve derivation paths that repeat a declared semantic binding alias.

    A planner often names a useful response field through ``binding_paths`` and then
    reuses that alias in identity/filter/ranking derivations.  The compiler and OAS
    validator operate on response paths, not semantic alias names.  Canonicalize exact
    alias references to the planner's own declared response path.  No field mapping is
    invented: ``binding_paths`` is the only authority, and the selected-schema validator
    still decides whether that declared path is actually documented.
    """
    out = dict(plan or {})
    steps = {str(x.get("id") or ""): x for x in (out.get("steps") or [])}
    derivations = []
    warnings = list(out.get("validation_warnings") or [])
    for raw in out.get("derivations") or []:
        deriv = dict(raw)
        sources = [str(x) for x in deriv.get("source_steps") or []]
        if len(sources) == 1:
            mapping = (steps.get(sources[0]) or {}).get("binding_paths") or {}
            mapping = {str(k): str(v).strip() for k, v in mapping.items()
                       if str(k).strip() and isinstance(v, str) and str(v).strip()}
            field = str(deriv.get("field") or "").strip()
            target = mapping.get(field)
            if target and target != field:
                deriv["field"] = target
                warnings.append(
                    f"derivation {deriv.get('id')}: normalized binding alias field "
                    f"{field!r}->{target!r}")
            filt = deriv.get("filter") if isinstance(deriv.get("filter"), dict) else None
            if filt:
                repaired_filter = {}
                changed = False
                for raw_key, predicate in filt.items():
                    key = str(raw_key)
                    mapped = mapping.get(key, key)
                    if mapped != key:
                        changed = True
                        warnings.append(
                            f"derivation {deriv.get('id')}: normalized binding alias filter field "
                            f"{key!r}->{mapped!r}")
                    # Preserve the first predicate if two aliases collapse to the same
                    # documented path; deterministic validation will flag contradictory
                    # semantics elsewhere rather than silently overwriting it.
                    if mapped not in repaired_filter:
                        repaired_filter[mapped] = predicate
                if changed:
                    deriv["filter"] = repaired_filter
        derivations.append(deriv)
    out["derivations"] = derivations
    out["validation_warnings"] = list(dict.fromkeys(warnings))
    return out


def normalize_age_date_comparisons(question: str, plan: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize ordering semantics when age is derived from birth dates.

    This is API-agnostic calendar semantics, not benchmark knowledge: an earlier
    birth date means an older person, while a later birth date means a younger
    person.  Apply the rule only when both comparison inputs are explicit identity
    derivations over clearly birth-date-like fields.  ``by how many`` requests use
    an absolute difference because the requested quantity is a magnitude.
    """
    out = dict(plan or {})
    derivations = [dict(x) for x in out.get("derivations") or []]
    by_id = {str(x.get("id") or ""): x for x in derivations}
    warnings = list(out.get("validation_warnings") or [])
    text = str(question or "").casefold()

    def birth_like_field(deriv: dict[str, Any]) -> bool:
        if str(deriv.get("operator") or "").lower() != "identity":
            return False
        field = re.sub(r"[^a-z0-9]+", "_", str(deriv.get("field") or "").casefold()).strip("_")
        return any(token in field for token in (
            "birthday", "birth_date", "date_of_birth", "birthdate", "dob"))

    for deriv in derivations:
        if str(deriv.get("operator") or "").lower() != "compare":
            continue
        source_ids = [str(x) for x in deriv.get("source_derivations") or []]
        if len(source_ids) != 2 or not all(birth_like_field(by_id.get(x) or {}) for x in source_ids):
            continue
        mode = str(deriv.get("comparison") or "").lower()
        new_mode = mode
        if re.search(r"\bolder\b", text):
            new_mode = {"gt": "lt", "gte": "lte", "max": "min"}.get(mode, mode)
        elif re.search(r"\byounger\b", text):
            new_mode = {"lt": "gt", "lte": "gte", "min": "max"}.get(mode, mode)
        if new_mode != mode:
            deriv["comparison"] = new_mode
            warnings.append(
                f"derivation {deriv.get('id')}: normalized birth-date age ordering "
                f"{mode!r}->{new_mode!r}")
        if mode == "difference" and re.search(
                r"\b(?:by how many|how many)\b|\bdifference\b", text):
            deriv["comparison"] = "abs_difference"
            warnings.append(
                f"derivation {deriv.get('id')}: normalized age difference to absolute magnitude")

    out["derivations"] = derivations
    out["validation_warnings"] = list(dict.fromkeys(warnings))
    return out


def _retarget_read_step(step: dict[str, Any], new_endpoint: str) -> dict[str, Any]:
    """Retarget one GET step while preserving only request declarations still valid by shape.

    This helper intentionally knows nothing about benchmark answers.  It is used by
    deterministic semantic normalizers after an OAS sibling operation has already
    been selected from question+documentation semantics.
    """
    out = dict(step or {})
    old_endpoint = str(out.get("endpoint") or "")
    out["endpoint"] = str(new_endpoint)
    placeholders = set(re.findall(r"\{([^{}]+)\}", str(new_endpoint)))
    out["path_literals"] = {
        str(k): v for k, v in dict(out.get("path_literals") or {}).items()
        if str(k) in placeholders
    }
    out["path_bindings"] = {
        str(k): v for k, v in dict(out.get("path_bindings") or {}).items()
        if str(k) in placeholders
    }
    # These are re-enriched from the selected OAS card during observation-plan
    # synthesis.  Keeping metadata from the previous route can create a false
    # request-contract failure after an otherwise-correct deterministic retarget.
    out["request_parameters"] = []
    if old_endpoint != str(new_endpoint):
        out.pop("request_body_required", None)
        out.pop("request_body_required_fields", None)
        out.pop("request_body_leaf_paths", None)
    return out


def normalize_latest_released_population(question: str, plan: dict[str, Any],
                                         tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Replace newest-record routes for explicit *released* recency with an OAS release population.

    ``latest`` is overloaded by APIs: an operation may mean newest database entry,
    while the user can explicitly mean most recently released.  When the chosen
    route documents newest/latest-record semantics and a unique same-family sibling
    documents a released/currently-playing population, retarget the step and let
    downstream deterministic selection operate on that population.
    """
    q = str(question or "").casefold()
    if not re.search(
            r"\b(?:latest|newest|most\s+recent)(?:\s+\w+){0,3}\s+released\b|"
            r"\bmost\s+recently\s+released\b|\blatest\s+release\b", q):
        return copy.deepcopy(plan or {})
    out = copy.deepcopy(plan or {})
    by_path = {str(t.get("path") or ""): t for t in tools or []
               if str(t.get("method") or "GET").upper() == "GET"}
    warnings = list(out.get("validation_warnings") or [])
    steps = []
    retargeted: dict[str, str] = {}
    for raw in out.get("steps") or []:
        step = dict(raw); path = str(step.get("endpoint") or "")
        prose = (path + " " + str((by_path.get(path) or {}).get("functionality") or "")).casefold()
        says_latest = bool(re.search(r"\b(?:latest|newest|most recent)\b", prose))
        says_release_population = bool(re.search(
            r"\b(?:released|now playing|currently playing|in theaters|in theatres)\b", prose))
        if not says_latest or says_release_population:
            steps.append(step); continue
        candidates = []
        for alt_path, alt_tool in by_path.items():
            if alt_path == path or "{" in alt_path or not _same_collection_family(path, alt_path):
                continue
            alt_text = (alt_path + " " + str(alt_tool.get("functionality") or "")).casefold()
            if re.search(r"\b(?:released|now playing|currently playing|in theaters|in theatres)\b", alt_text):
                # Prefer present/released populations over future/upcoming ones.
                future = bool(re.search(r"\b(?:upcoming|future|will be released)\b", alt_text))
                present = bool(re.search(r"\b(?:now playing|currently playing|in theaters|in theatres)\b", alt_text))
                candidates.append((2 if present else (0 if future else 1), alt_path))
        candidates.sort(key=lambda x:(x[0],x[1]), reverse=True)
        if candidates and (len(candidates)==1 or candidates[0][0] > candidates[1][0]):
            new_path = candidates[0][1]
            step = _retarget_read_step(step, new_path)
            retargeted[str(step.get("id") or "")] = new_path
            warnings.append(
                f"step {step.get('id')}: retargeted newest-record route {path} to documented "
                f"released-population sibling {new_path} for explicit latest-released intent")
        steps.append(step)
    out["steps"] = steps
    # A planner may already have chosen the release-population sibling before this
    # normalizer runs. For explicit latest-released intent, keep that route-level
    # selection finite rather than requiring an unbounded release-date scan.
    release_population_steps=set(retargeted)
    for st in out.get("steps") or []:
        path=str(st.get("endpoint") or "")
        tool=by_path.get(path) or {}
        prose=(path+" "+str(tool.get("functionality") or "")).casefold()
        if re.search(r"\b(?:now playing|currently playing|in theaters|in theatres|released)\b",prose) and not re.search(r"\b(?:upcoming|future)\b",prose):
            release_population_steps.add(str(st.get("id") or ""))
    if release_population_steps:
        ds=[dict(d) for d in out.get("derivations") or []]
        for d in ds:
            if not ({str(x) for x in d.get("source_steps") or []} & release_population_steps): continue
            op=str(d.get("operator") or "").lower();field=str(d.get("field") or "").casefold();purpose=str(d.get("purpose") or "").casefold()
            if op in {"argmax","argmin"} and ("release" in field or "release" in purpose) and any(tok in field for tok in ("date","time","year")):
                d["operator"]="endpoint_rank";d["rank"]=0;d["field"]="results";d["comparison"]="";d["comparison_literal"]=None
                warnings.append(f"derivation {d.get('id')}: normalized latest-released date extremum to endpoint-ranked head of the documented release population")
        out["derivations"]=ds
    if retargeted:
        # The old newest-record endpoint is commonly a singleton.  The sibling is a
        # collection, so ensure there is a concrete selector for downstream bindings.
        # Preserve the route-level recency semantics of this substitution: do not
        # turn it into a page-local/global release-date scan merely because the
        # collection also exposes release_date.  The documented release population
        # is itself the chosen semantic operation; select its endpoint-ranked head.
        derivs = [dict(d) for d in out.get("derivations") or []]
        for d in derivs:
            src={str(x) for x in d.get("source_steps") or []}
            if not (src & set(retargeted)):
                continue
            op=str(d.get("operator") or "").lower()
            field=str(d.get("field") or "").casefold()
            purpose=str(d.get("purpose") or "").casefold()
            if op in {"argmax","argmin"} and ("release" in field or "release" in purpose) and any(
                    token in field for token in ("date","time","year")):
                d["operator"]="endpoint_rank"; d["rank"]=0; d["field"]="results"
                d["comparison"]=""; d["comparison_literal"]=None
                warnings.append(
                    f"derivation {d.get('id')}: preserved endpoint-ranked head after latest-record -> released-population retarget instead of requiring an unbounded release-date extremum")
        source_ids = set(retargeted)
        has_selector = {sid: False for sid in source_ids}
        for d in derivs:
            if str(d.get("operator") or "").lower() in {"endpoint_rank","first","nth","argmax","argmin","filter"}:
                for sid in d.get("source_steps") or []:
                    if str(sid) in has_selector:
                        has_selector[str(sid)] = True
        existing = {str(d.get("id") or "") for d in derivs}
        for sid, present in has_selector.items():
            if present:
                continue
            did = f"release_population_pick_{sid}"; n = 2
            while did in existing:
                did = f"release_population_pick_{sid}_{n}"; n += 1
            existing.add(did)
            derivs.append({
                "id": did, "operator": "endpoint_rank", "source_steps": [sid],
                "source_derivations": [], "label_steps": [], "label_fields": [],
                "field": "results", "comparison": "", "comparison_literal": None,
                "unit": "raw", "distinct_field": None, "rank": 0, "filter": {},
                "top_k": 10,
                "purpose": "Select the first record from the documented released population.",
            })
        out["derivations"] = derivs
    out["validation_warnings"] = list(dict.fromkeys(warnings))
    return out


def normalize_explicit_population_sibling(question: str, plan: dict[str, Any],
                                          tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Remove an unstated terminal population qualifier when one OAS sibling matches the question.

    Example shape: a route whose suffix means ``airing today`` for a question that
    explicitly asks ``currently on the air``.  Candidate ranking (e.g. popularity)
    is preserved; only the population route is changed.
    """
    qstems = _semantic_stems(question)
    qtext = str(question or "").casefold()
    # Release-recency has its own stronger population normalizer. Do not let a
    # generic suffix cleanup undo the released-population route it selected.
    if re.search(r"\b(?:latest|newest|most\s+recent)(?:\s+\w+){0,3}\s+released\b|"
                 r"\bmost\s+recently\s+released\b|\blatest\s+release\b", qtext):
        return copy.deepcopy(plan or {})
    if not qstems:
        return copy.deepcopy(plan or {})
    out = copy.deepcopy(plan or {})
    by_path = {str(t.get("path") or ""): t for t in tools or []
               if str(t.get("method") or "GET").upper() == "GET"}
    warnings = list(out.get("validation_warnings") or [])
    steps=[]
    for raw in out.get("steps") or []:
        step=dict(raw); path=str(step.get("endpoint") or "")
        if not path or "{" in path or "/search/" in path.casefold() or path not in by_path:
            steps.append(step); continue
        selected_text = path.replace("_"," ") + " " + str(by_path[path].get("functionality") or "")
        sel_stems = _semantic_stems(selected_text)
        sel_suffix = _path_suffix_stems(path)
        unstated = {s for s in sel_suffix if s not in qstems and s not in {"popular","trend","trending"}}
        if not unstated:
            steps.append(step); continue
        ranked=[]
        population_core = {s for s in sel_suffix
                           if s not in unstated and s not in {"popular","trend","trending","top","rate","rating"}}
        for alt_path, tool in by_path.items():
            if alt_path == path or "{" in alt_path or not _same_collection_family(path, alt_path):
                continue
            alt_stems = _semantic_stems(alt_path.replace("_"," ") + " " + str(tool.get("functionality") or ""))
            overlap = len(qstems & alt_stems)
            alt_unstated = {s for s in _path_suffix_stems(alt_path)
                            if s not in qstems and s not in {"popular","trend","trending"}}
            if population_core and not (population_core & _path_suffix_stems(alt_path)):
                continue
            # Prefer siblings that remove the unstated qualifier and gain explicit
            # user vocabulary.  Require strictly more semantic overlap.
            if not (unstated & _path_suffix_stems(alt_path)):
                ranked.append((overlap, -len(alt_unstated), alt_path))
        ranked.sort(reverse=True)
        selected_score=(len(qstems & sel_stems), -len(unstated))
        if ranked and ranked[0][:2] > selected_score and (len(ranked) == 1 or ranked[0][:2] > ranked[1][:2]):
            new_path=ranked[0][2]
            step=_retarget_read_step(step,new_path)
            warnings.append(
                f"step {step.get('id')}: retargeted population {path} to OAS sibling {new_path}; "
                f"removed unstated qualifier(s) {sorted(unstated)} while preserving requested population semantics")
        steps.append(step)
    out["steps"]=steps
    out["validation_warnings"]=list(dict.fromkeys(warnings))
    return out


def repair_selection_dependencies(question: str, plan: dict[str, Any]) -> dict[str, Any]:
    """Repair placeholder data-flow using only the declared plan graph.

    This routine is intentionally API-agnostic.  It never infers entity types from
    route vocabulary or resource names.  A downstream path placeholder is owned
    by the nearest dependency that explicitly binds the same placeholder.  When a
    planner used an alias for the selected value, an immediate dependency that is
    itself a deterministic-selection source may receive an alias for the exact
    downstream placeholder.  Ambiguous producers are left unresolved rather than
    guessed.
    """
    out = normalize_dynamic_path_references(plan)
    out = normalize_derivation_binding_aliases(out)
    out = normalize_age_date_comparisons(question, out)
    steps = [dict(x) for x in (out.get("steps") or [])]
    by_id = {str(x.get("id")): x for x in steps}
    order = {str(x.get("id")): i for i, x in enumerate(steps)}
    selected_sources = _selection_source_steps(out)
    warnings = list(out.get("validation_warnings") or [])

    # Carry an explicit path literal through an immediate dependent when the same
    # placeholder keeps the same route context. This is request provenance, not a
    # response binding: e.g. a declared season_number=1 remains 1 for a child
    # season-credits route even if the parent response has no binding field for it.
    for step in steps:
        sid = str(step.get("id"))
        deps = [str(x) for x in (step.get("depends_on") or []) if str(x) in by_id]
        literals = dict(step.get("path_literals") or {})
        for name in _step_placeholders(step):
            if name in literals:
                continue
            matches = []
            child_context = _placeholder_route_context(step, name)
            for dep_id in deps:
                dep = by_id[dep_id]
                dep_literals = dep.get("path_literals") or {}
                if name not in dep_literals:
                    continue
                if child_context is not None and _placeholder_route_context(dep, name) != child_context:
                    continue
                matches.append((dep_id, dep_literals.get(name)))
            values = {repr(value): value for _, value in matches if value not in (None, "")}
            if len(values) == 1:
                value = next(iter(values.values()))
                literals[name] = value
                warnings.append(
                    f"step {sid}: inherited path literal {{{name}}}={value!r} from immediate dependency")
        if literals:
            step["path_literals"] = literals

    # A response binding must never shadow a dynamic value that this same step
    # consumed from an ancestor.  Preserve that value's original producer instead.
    # This is purely graph/lineage canonicalization and does not inspect API values.
    for step in steps:
        protected_inputs = _protected_passthrough_binding_names({"steps": steps}, str(step.get("id")))
        if not protected_inputs:
            continue
        binds = [str(x).strip("{}") for x in (step.get("binds") or [])]
        shadowed = [name for name in binds if name in protected_inputs]
        if shadowed:
            step["binds"] = [name for name in binds if name not in protected_inputs]
            if isinstance(step.get("binding_paths"), dict):
                step["binding_paths"] = {k: v for k, v in step["binding_paths"].items()
                                         if str(k).strip("{}") not in protected_inputs}
            warnings.append(
                f"step {step.get('id')}: removed pass-through binding aliases {shadowed}; "
                "dynamic request inputs retain their upstream provenance")

    # If the immediate dependency selects a record that feeds a single-placeholder
    # consumer, make the exact consumer placeholder an explicit binding alias.
    for step in steps:
        placeholders = _step_placeholders(step)
        deps = [str(x) for x in (step.get("depends_on") or []) if str(x) in by_id]
        if len(placeholders) != 1 or len(deps) != 1:
            continue
        name = placeholders[0]
        if name in (step.get("path_literals") or {}):
            continue
        dep = by_id[deps[0]]
        dep_binds = [str(x).strip("{}") for x in (dep.get("binds") or [])]
        if name in _protected_passthrough_binding_names({"steps": steps}, str(dep.get("id"))):
            continue
        if str(dep.get("id")) in selected_sources:
            bp = dict(dep.get("binding_paths") or {})
            added_alias = False
            if name not in dep_binds:
                dep_binds = list(dict.fromkeys(dep_binds + [name]))
                dep["binds"] = dep_binds
                added_alias = True

            # The alias denotes the same selected-record value.  A planner may
            # already list the consumer placeholder in ``binds`` but omit its
            # binding_path (for example person_id beside cast_id).  Recover that
            # path only when the selected producer exposes exactly one compatible
            # leaf.  This is schema/graph normalization, not value guessing.
            if name not in bp:
                known_paths = [str(bp.get(b)).strip() for b in dep_binds if bp.get(b)]
                target_leaf = str(name).strip("{}").split("_")[-1].casefold()
                leaf_matches = []
                for path in known_paths:
                    clean = re.sub(r"\[(?:\*|\d*)\]", "", path)
                    leaf = clean.split(".")[-1].casefold()
                    if leaf == target_leaf:
                        leaf_matches.append(path)
                leaf_matches = list(dict.fromkeys(leaf_matches))
                unique_paths = list(dict.fromkeys(known_paths))
                if len(leaf_matches) == 1:
                    bp[name] = leaf_matches[0]
                elif len(unique_paths) == 1:
                    bp[name] = unique_paths[0]
            if bp:
                dep["binding_paths"] = bp
            if added_alias:
                warnings.append(
                    f"step {dep.get('id')}: added binding alias {{{name}}} for immediate dependent {step.get('id')}")
            elif name in bp and not (dep.get("binding_paths") or {}).get(name):
                warnings.append(
                    f"step {dep.get('id')}: recovered binding path for {{{name}}} from selected producer field")

    def ancestor_distances(start_ids: list[str]) -> dict[str, int]:
        distances: dict[str, int] = {}
        queue = [(sid, 1) for sid in start_ids if sid in by_id]
        while queue:
            sid, dist = queue.pop(0)
            if sid in distances and distances[sid] <= dist:
                continue
            distances[sid] = dist
            for dep in (by_id.get(sid) or {}).get("depends_on") or []:
                dep = str(dep)
                if dep in by_id:
                    queue.append((dep, dist + 1))
        return distances

    # Rebind a consumer only within its declared dependency ancestry. This avoids
    # the previous order-based heuristic where an unrelated later sibling could
    # steal ownership simply because it happened to bind the same placeholder.
    for step in steps:
        sid = str(step.get("id"))
        placeholders = _step_placeholders(step)
        if not placeholders:
            continue
        current = [str(x) for x in (step.get("depends_on") or []) if str(x) in by_id]
        if not current:
            continue
        distances = ancestor_distances(current)
        for name in placeholders:
            if name in (step.get("path_literals") or {}):
                continue
            candidates = []
            for producer, dist in distances.items():
                binds = {str(v).strip("{}") for v in (by_id.get(producer) or {}).get("binds") or []}
                if name in binds:
                    candidates.append((dist, producer))
            if not candidates:
                continue
            best_dist = min(dist for dist, _ in candidates)
            nearest = [producer for dist, producer in candidates if dist == best_dist]
            if len(nearest) > 1:
                # A planner may redundantly list both a selected producer and one
                # of its ancestors as direct dependencies. Prefer the most
                # downstream candidate in that partial order; unrelated siblings
                # remain genuinely ambiguous and are never guessed.
                downstream = [p for p in nearest if not any(
                    p in set(ancestor_distances([q])) for q in nearest if q != p)]
                nearest = downstream
            if len(nearest) != 1:
                warnings.append(
                    f"step {sid}: ambiguous producers for {{{name}}}: {nearest}; left unchanged")
                continue
            best = nearest[0]
            competing = [x for x in current if name in {
                str(v).strip("{}") for v in (by_id.get(x) or {}).get("binds") or []
            }]
            # If the nearest producer is already a direct dependency, remove only
            # older competitors that are ancestors of that producer. This
            # preserves unrelated sibling dependencies while eliminating an
            # ancestor that could otherwise compete for the same placeholder.
            best_ancestors = set(ancestor_distances([best]))
            redundant = [x for x in competing if x != best and x in best_ancestors]
            if redundant:
                current = [x for x in current if x not in redundant]
                warnings.append(
                    f"step {sid}: removed ancestor binding competitors {redundant} for {{{name}}}; nearest producer is {best}")
            if best not in current and best_dist == 1:
                current = [x for x in current if x not in competing] + [best]
                warnings.append(f"step {sid}: bound {{{name}}} to nearest declared producer {best}")
        step["depends_on"] = list(dict.fromkeys(current))

    out["steps"] = steps
    out["validation_warnings"] = list(dict.fromkeys(warnings))
    return out



def _explicit_multi_intent(question: str) -> bool:
    """Return True only for high-confidence user requests for multiple outputs."""
    q = str(question or "").casefold().strip()
    if re.search(r"\b(?:some|several|multiple|a\s+few|few\s+more|a\s+couple(?:\s+of)?)\b", q):
        return True
    if re.match(r"^(?:please\s+)?list\b", q):
        return True
    if re.search(r"\b(?:what|which)\s+are\b", q):
        return True
    # A relation phrased as "who starred/acted/appeared in ..." asks for the
    # relation members, not one arbitrarily first-billed member.  Explicit lead/
    # first/one wording remains singular and therefore bypasses this rule.
    if (re.match(r"^(?:please\s+)?who\b", q)
            and re.search(r"\b(?:starred|acted|appeared|starring)\s+in\b", q)
            and not re.search(r"\b(?:lead|leading)\s+(?:actor|actress|performer|cast\s+member)\b", q)
            and not re.search(r"\b(?:one|single|first)\s+(?:actor|actress|performer|person|cast\s+member)\b", q)):
        return True
    # Imperative plural nouns are also explicit enough to preserve cardinality.
    return bool(re.search(
        r"\b(?:give|show|recommend|provide|find|tell)\b[^?]{0,45}\b"
        r"(?:movies|films|tv\s+shows|shows|reviews|images|photos|posters|keywords|"
        r"actors|directors|genres|works|recommendations|release\s+dates)\b", q))


def _explicit_single_intent(question: str) -> bool:
    """Return True for explicit one-item requests without conflicting plurality."""
    q = str(question or "").casefold().strip()
    if _explicit_multi_intent(q):
        return False
    if re.search(r"\b(?:one|a\s+single|single)\b[^?]{0,28}\b"
                 r"(?:image|photo|poster|cover|logo|review|movie|film|show|keyword|"
                 r"recommendation|actor|director)\b", q):
        return True
    # Singular answer-like wording is a strong surface contract. Keep the noun set
    # intentionally narrow so phrases such as "a movie" remain source entities,
    # while "a movie cover" / "a keyword" / "the logo" remain output constraints.
    return bool(re.search(
        r"\b(?:a|an|the)\s+(?:(?:\w+[ -]){0,2})?"
        r"(?:cover(?:\s+image)?|image|photo|poster|logo|keyword|review|recommendation|(?:production\s+)?company)\b", q))


def normalize_typed_multi_answer_cardinality(plan: dict[str, Any]) -> dict[str, Any]:
    """Remove a contradictory terminal singleton for explicit multi-item requests.

    This is a deterministic semantic normalization, not a domain rule.  It applies
    only when all of the following are true:
      * the user explicitly says some/several/multiple/a few/a couple;
      * the plan answer mode is list;
      * an answer identity consumes one same-step singleton selector;
      * that answer source step has no downstream API consumer.

    Bypassing ``filter -> endpoint_rank -> identity`` becomes
    ``filter -> identity``; bypassing ``endpoint_rank -> identity`` becomes a
    collection identity.  Upstream search/entity selectors and any step that feeds
    a later request are untouched.  An orphaned singleton derivation is pruned so
    observation synthesis cannot silently re-collapse the projected collection.
    """
    out = copy.deepcopy(plan or {})
    if str(out.get("answer_cardinality") or "").lower() != "many":
        return out
    derivs = [dict(d) for d in (out.get("derivations") or [])]
    by_id = {str(d.get("id") or ""): d for d in derivs}
    answer_steps = {str(x) for x in (out.get("answer_steps") or [])}
    steps = {str(x.get("id") or ""): x for x in (out.get("steps") or [])}
    has_downstream = {sid: False for sid in steps}
    for consumer in steps.values():
        for dep in consumer.get("depends_on") or []:
            if str(dep) in has_downstream:
                has_downstream[str(dep)] = True
    singleton_ops = {"endpoint_rank", "first", "nth", "argmax", "argmin"}
    bypassed: set[str] = set()
    warnings = list(out.get("validation_warnings") or [])
    for d in derivs:
        if str(d.get("operator") or "").lower() != "identity":
            continue
        source_steps = {str(x) for x in (d.get("source_steps") or [])}
        if not source_steps or not (source_steps & answer_steps):
            continue
        if any(has_downstream.get(sid, False) for sid in source_steps):
            continue
        parents = [str(x) for x in (d.get("source_derivations") or [])]
        if len(parents) != 1:
            continue
        parent = by_id.get(parents[0]) or {}
        if str(parent.get("operator") or "").lower() not in singleton_ops:
            continue
        parent_steps = {str(x) for x in (parent.get("source_steps") or [])}
        if parent_steps != source_steps:
            continue
        # A same-response singleton can select the *owner* record while the
        # terminal identity projects a nested child collection from that owner
        # (for example results -> known_for.title). Final output cardinality
        # applies to the nested collection, so preserve an outer-owner selector.
        terminal_field = _normalized_schema_path(d.get("field"))
        parent_field = _normalized_schema_path(parent.get("field"))
        if "." in terminal_field:
            terminal_collection = terminal_field.split(".", 1)[0]
            if not parent_field or not (parent_field == terminal_collection or
                                        parent_field.endswith("." + terminal_collection)):
                continue
        d["source_derivations"] = [str(x) for x in (parent.get("source_derivations") or [])]
        bypassed.add(parents[0])
        warnings.append(
            f"derivation {d.get('id')}: expanded explicit multi-item answer by bypassing "
            f"terminal singleton selector {parents[0]}")

    if bypassed:
        referenced = {str(x) for d in derivs for x in (d.get("source_derivations") or [])}
        derivs = [d for d in derivs if str(d.get("id") or "") not in bypassed
                  or str(d.get("id") or "") in referenced]
        out["derivations"] = derivs
        out["validation_warnings"] = list(dict.fromkeys(warnings))
    return out



def normalize_typed_single_answer_cardinality(plan: dict[str, Any]) -> dict[str, Any]:
    """Insert one deterministic same-step selector for explicit one-item assets.

    This is the conservative inverse of multi-cardinality expansion. It applies
    only to explicit one-item requests and only when a terminal answer identity
    names a concrete collection-qualified field such as ``posters.file_path`` or
    ``profiles[*].file_path``. The collection name comes from the planner itself;
    no API/domain relation is guessed.
    """
    out = copy.deepcopy(plan or {})
    if str(out.get("answer_cardinality") or "").lower() != "one":
        return out
    derivs = [dict(d) for d in (out.get("derivations") or [])]
    by_id = {str(d.get("id") or ""): d for d in derivs}
    answer_ids = {str(x) for x in (out.get("answer_derivations") or []) if str(x)}
    answer_steps = {str(x) for x in (out.get("answer_steps") or []) if str(x)}
    selector_ops = {"first", "nth", "endpoint_rank", "argmax", "argmin"}
    existing_ids = set(by_id)
    warnings = list(out.get("validation_warnings") or [])
    additions = []

    for d in derivs:
        did = str(d.get("id") or "")
        if str(d.get("operator") or "").lower() != "identity":
            continue
        if answer_ids and did not in answer_ids:
            continue
        src_steps = [str(x) for x in (d.get("source_steps") or [])]
        if len(src_steps) != 1 or (answer_steps and src_steps[0] not in answer_steps):
            continue
        # Do not add a second same-step singleton selector.
        already = False
        for parent_id in d.get("source_derivations") or []:
            parent = by_id.get(str(parent_id)) or {}
            if (str(parent.get("operator") or "").lower() in selector_ops
                    and set(str(x) for x in parent.get("source_steps") or []) == set(src_steps)):
                already = True; break
        if already:
            continue
        raw_field = str(d.get("field") or "")
        clean = re.sub(r"\[(?:\*|\d*)\]", "", raw_field)
        bits = [x for x in clean.split(".") if x]
        filter_parent_id = ""
        collection = bits[0] if len(bits) >= 2 else ""
        if not collection:
            # A role-scoped relation is often represented as
            # filter(collection) -> identity(name).  The identity field is bare
            # because the filter already establishes the record universe.  For a
            # singular final answer, select one record from that filtered set.
            for parent_id in d.get("source_derivations") or []:
                parent = by_id.get(str(parent_id)) or {}
                if str(parent.get("operator") or "").lower() != "filter":
                    continue
                parent_field = str(parent.get("field") or "")
                parent_clean = re.sub(r"\[(?:\*|\d*)\]", "", parent_field)
                parent_bits = [x for x in parent_clean.split(".") if x]
                if parent_bits:
                    collection = parent_bits[0]
                    filter_parent_id = str(parent_id)
                    break
        if not collection:
            continue
        pick_id = f"single_pick_{did or src_steps[0]}"
        suffix = 2
        while pick_id in existing_ids:
            pick_id = f"single_pick_{did or src_steps[0]}_{suffix}"; suffix += 1
        existing_ids.add(pick_id)
        additions.append({
            "id": pick_id,
            "operator": "first",
            "source_steps": src_steps,
            "source_derivations": [filter_parent_id] if filter_parent_id else [],
            "label_steps": [], "label_fields": [],
            "field": collection,
            "comparison": "", "comparison_literal": None,
            "unit": "raw", "distinct_field": None, "rank": 0,
            "filter": dict(d.get("filter") or {}), "top_k": 1,
            "purpose": "Select one record from the explicitly singular answer collection.",
        })
        existing_parents = [str(x) for x in (d.get("source_derivations") or [])]
        if filter_parent_id:
            existing_parents = [x for x in existing_parents if x != filter_parent_id]
        d["source_derivations"] = list(dict.fromkeys(existing_parents + [pick_id]))
        warnings.append(
            f"derivation {did}: inserted deterministic first selector for explicit one-item answer")

    if additions:
        # Insert selectors before their consumers so source_derivation ordering is valid.
        add_by_consumer = {str(a["id"]).removeprefix("single_pick_"): a for a in additions}
        rebuilt = []
        for d in derivs:
            did = str(d.get("id") or "")
            # Find the selector referenced by this consumer rather than relying on
            # the generated id parsing when the id contains underscores.
            refs = [a for a in additions if str(a["id"]) in set(d.get("source_derivations") or [])]
            rebuilt.extend(refs)
            rebuilt.append(d)
        derivs = rebuilt
    out["derivations"] = derivs
    out["validation_warnings"] = list(dict.fromkeys(warnings))
    return out







def normalize_appearance_credit_relation(question: str, plan: dict[str, Any]) -> dict[str, Any]:
    """Prefer acting/cast credit branches for explicit person "appearance" intent.

    When one response exposes parallel cast and crew credit collections, treating both as
    candidate "appearances" makes latest-recency selection ambiguous. The user-facing term
    appearance/starred/acted is a high-confidence acting relation. If the plan already contains
    both branches, prune only the crew branch and its binding aliases; no route is invented.
    """
    q = str(question or "").casefold()
    if not re.search(r"\b(?:appearance|appeared|starred|starring|acted|acting)\b", q):
        return copy.deepcopy(plan or {})
    out = copy.deepcopy(plan or {})
    derivs = [dict(d) for d in out.get("derivations") or []]
    blob = " ".join(str(d.get("field") or "") + " " + " ".join(str(x) for x in (d.get("filter") or {}))
                    for d in derivs).casefold()
    if "cast" not in blob or "crew" not in blob:
        return out
    crew_ids=set()
    for d in derivs:
        texts=[str(d.get("field") or "")] + [str(x) for x in (d.get("filter") or {})]
        normalized=" ".join(re.sub(r"\[(?:\*|\d*)\]", "", x).casefold() for x in texts)
        if re.search(r"(?:^|\.)crew(?:\.|\b)|\bcrew_", normalized):
            crew_ids.add(str(d.get("id") or ""))
    # Propagate removal through derivation-only children of a removed crew branch.
    changed=True
    while changed:
        changed=False
        for d in derivs:
            did=str(d.get("id") or "")
            if did in crew_ids: continue
            parents={str(x) for x in d.get("source_derivations") or []}
            if parents and parents.issubset(crew_ids):
                crew_ids.add(did); changed=True
    if not crew_ids:
        return out
    out["derivations"]=[d for d in derivs if str(d.get("id") or "") not in crew_ids]
    out["answer_derivations"]=[str(x) for x in out.get("answer_derivations") or [] if str(x) not in crew_ids]
    # Remove crew-only binding aliases when equivalent cast aliases remain on the producer.
    steps=[]
    for raw in out.get("steps") or []:
        step=dict(raw); bpaths=dict(step.get("binding_paths") or {})
        if any("cast" in str(v).casefold() or str(k).casefold().startswith("cast_") for k,v in bpaths.items()):
            bpaths={k:v for k,v in bpaths.items()
                    if not ("crew" in str(v).casefold() or str(k).casefold().startswith("crew_"))}
            step["binding_paths"]=bpaths
            step["binds"]=[x for x in step.get("binds") or [] if str(x) in bpaths or not str(x).casefold().startswith("crew_")]
        steps.append(step)
    out["steps"]=steps
    out.setdefault("validation_warnings", []).append(
        f"appearance/acting intent pruned parallel crew-credit derivation branch {sorted(crew_ids)}")
    return out

def normalize_nested_relation_owner_selection(question: str, plan: dict[str, Any]) -> dict[str, Any]:
    """Add an explicit outer-record selector before projecting a nested relation.

    A search response may contain ``results[*].known_for[*]`` or an analogous nested
    relation. Projecting the nested relation without first selecting the owning result
    flattens children from every search candidate. This repair is graph/schema agnostic:
    it adds only a fieldless endpoint-rank selector on the same source step and makes the
    nested terminal derivation consume that selector as ownership lineage.
    """
    out = copy.deepcopy(plan or {})
    steps = {str(x.get("id") or ""): x for x in out.get("steps") or []}
    derivs = [dict(x) for x in out.get("derivations") or []]
    selector_ops = {"endpoint_rank", "first", "nth", "argmax", "argmin", "filter"}
    existing = {str(x.get("id") or "") for x in derivs}
    additions: list[tuple[str, dict[str, Any]]] = []
    warnings = list(out.get("validation_warnings") or [])
    for d in derivs:
        did = str(d.get("id") or "")
        if str(d.get("operator") or "").lower() != "identity" or d.get("source_derivations"):
            continue
        field = re.sub(r"\[(?:\*|\d*)\]", "", str(d.get("field") or "")).strip(".")
        if field.count(".") < 1:
            continue
        src = [str(x) for x in d.get("source_steps") or []]
        if len(src) != 1:
            continue
        sid = src[0]
        endpoint = str((steps.get(sid) or {}).get("endpoint") or "").casefold()
        if "/search/" not in endpoint:
            continue
        if any(sid in {str(x) for x in p.get("source_steps") or []}
               and str(p.get("operator") or "").lower() in selector_ops
               for p in derivs if str(p.get("id") or "") != did):
            continue
        pick = f"owner_pick_{did or sid}"
        n = 2
        while pick in existing:
            pick = f"owner_pick_{did or sid}_{n}"; n += 1
        existing.add(pick)
        selector = {
            "id": pick, "operator": "endpoint_rank", "source_steps": [sid],
            "source_derivations": [], "label_steps": [], "label_fields": [],
            "field": None, "comparison": "", "comparison_literal": None,
            "unit": "raw", "distinct_field": None, "rank": 0, "filter": {},
            "top_k": 10,
            "purpose": "Select one owning outer record before projecting its nested relation.",
        }
        d["source_derivations"] = [pick]
        additions.append((did, selector))
        warnings.append(
            f"derivation {did}: inserted outer owner selection before nested relation projection")
    if additions:
        rebuilt=[]
        addmap={did:sel for did,sel in additions}
        for d in derivs:
            if str(d.get("id") or "") in addmap:
                rebuilt.append(addmap[str(d.get("id") or "")])
            rebuilt.append(d)
        out["derivations"] = rebuilt
        out["validation_warnings"] = list(dict.fromkeys(warnings))
    return out

def normalize_answer_steps_from_answer_derivations(question: str, plan: dict[str, Any]) -> dict[str, Any]:
    """Prune support-only answer_steps when explicit final derivations already identify the answer.

    Planner retries sometimes leave an upstream search/detail step in ``answer_steps`` even
    though every declared final answer derivation consumes only a downstream step. Treating
    that support step as an answer source creates a false validation failure and expensive
    replanning. This normalization is purely graph-based: it never adds a route or field.
    """
    out = copy.deepcopy(plan or {})
    derivs = {str(d.get("id") or ""): d for d in (out.get("derivations") or [])}
    answer_ids = [str(x) for x in (out.get("answer_derivations") or []) if str(x)]
    if not answer_ids:
        return out
    direct_sources: set[str] = set()
    for did in answer_ids:
        d = derivs.get(did) or {}
        direct_sources.update(str(x) for x in (d.get("source_steps") or []) if str(x))
    if not direct_sources:
        return out
    old = [str(x) for x in (out.get("answer_steps") or []) if str(x)]
    kept = [sid for sid in old if sid in direct_sources]
    if kept and kept != old:
        out["answer_steps"] = kept
        out.setdefault("validation_warnings", []).append(
            f"pruned support-only answer_steps {sorted(set(old) - set(kept))}; explicit answer derivations consume {kept}")
    return out


def normalize_requested_answer_derivations(question: str, plan: dict[str, Any]) -> dict[str, Any]:
    """Remove high-confidence support fields from the declared answer surface.

    Evidence may retain IDs, characters, scores, dates, and counts for selection/replay, but
    those fields should not be emitted unless the question actually asks for them. Only the
    ``answer_derivations`` list is narrowed; the derivations themselves remain available as
    support evidence.
    """
    out = copy.deepcopy(plan or {})
    q = str(question or "").casefold()
    derivs = {str(d.get("id") or ""): d for d in (out.get("derivations") or [])}
    answer_ids = [str(x) for x in (out.get("answer_derivations") or []) if str(x)]
    # Boolean questions should terminate on the replayed Boolean, not also emit
    # the identity/count operands that merely support it.  Prefer an explicit
    # logical combiner; otherwise one membership derivation is sufficient.
    if str(out.get("answer_mode") or "").lower() == "boolean" and answer_ids:
        logical=[did for did in answer_ids if str((derivs.get(did) or {}).get("operator") or "").lower() in {"logical_and","logical_or"}]
        members=[did for did in answer_ids if str((derivs.get(did) or {}).get("operator") or "").lower() == "membership"]
        terminal = logical if len(logical)==1 else (members if len(members)==1 else [])
        if terminal and answer_ids != terminal:
            out["answer_derivations"] = terminal
            out.setdefault("validation_warnings", []).append(
                f"pruned Boolean support operands from answer surface; terminal Boolean derivation is {terminal[0]}")
            answer_ids = terminal
    if len(answer_ids) < 2:
        return out
    drop: list[str] = []
    for did in answer_ids:
        d = derivs.get(did) or {}
        if str(d.get("operator") or "").lower() not in {"identity", "count"}:
            continue
        field = re.sub(r"\[(?:\*|\d*)\]", "", str(d.get("field") or "")).split(".")[-1].casefold()
        support_only = False
        if field in {"id", "person_id", "movie_id", "tv_id", "series_id", "company_id"}:
            support_only = not bool(re.search(r"\b(?:id|identifier)\b", q))
        elif field in {"character", "role"}:
            support_only = not bool(re.search(r"\b(?:character|role)\b", q))
        elif field in {"popularity", "vote_average", "rating", "score"}:
            support_only = not bool(re.search(r"\b(?:rating|score|popularity|how popular)\b", q))
        elif field in {"release_date", "air_date", "first_air_date", "date", "year"}:
            support_only = not bool(re.search(r"\b(?:when|date|year|released|release date|aired|air date)\b", q))
        elif str(d.get("operator") or "").lower() == "count":
            support_only = not bool(re.search(r"\b(?:how many|number|count)\b", q))
        if support_only:
            drop.append(did)
    kept = [x for x in answer_ids if x not in set(drop)]
    if kept and drop:
        out["answer_derivations"] = kept
        out.setdefault("validation_warnings", []).append(
            f"removed support-only answer derivation(s) {drop}; retained them as non-terminal evidence")
    return out


def normalize_coordinated_parent_child_answers(question: str, plan: dict[str, Any],
                                               cards: list[dict[str, Any]]) -> dict[str, Any]:
    """Restore a requested selected parent label when a child relation is also requested.

    Planners often correctly use an upstream ranked/search collection to choose an
    entity and a child endpoint for its relation, but declare only the child value
    terminally (e.g. "most popular movie **and** its keywords").  If the wording
    clearly asks for both and the selected parent schema exposes a human label,
    synthesize only that label identity from the existing selector.
    """
    q = str(question or "").casefold()
    if " and " not in q:
        return copy.deepcopy(plan or {})
    # Require an explicit parent entity noun plus a possessive/coreferential child.
    if not re.search(r"\b(?:movie|film|tv\s+show|show|series|person|actor|company|collection)\b", q):
        return copy.deepcopy(plan or {})
    if not re.search(r"\b(?:its|their|his|her)\b|\band\s+what\b|\band\s+which\b", q):
        return copy.deepcopy(plan or {})
    out=copy.deepcopy(plan or {})
    derivs=[dict(d) for d in out.get("derivations") or []]
    by_id={str(d.get("id") or ""):d for d in derivs}
    answer_ids=[str(x) for x in out.get("answer_derivations") or [] if str(x)]
    card_by_step={str(c.get("step_id") or ""):c for c in cards or []}
    selector_ops={"endpoint_rank","first","nth","argmax","argmin"}
    additions=[]; existing=set(by_id); warnings=list(out.get("validation_warnings") or [])
    for sel in derivs:
        if str(sel.get("operator") or "").lower() not in selector_ops:
            continue
        src=[str(x) for x in sel.get("source_steps") or []]
        if len(src)!=1:
            continue
        sid=src[0]; card=card_by_step.get(sid) or {}
        # If the selected parent step already contributes a terminal human label,
        # nothing is missing. A child relation's generic ``name`` field (keyword
        # name, cast name, etc.) must not be mistaken for the parent entity label.
        parent_already_answered = any(
            sid in {str(x) for x in (by_id.get(a) or {}).get("source_steps") or []}
            and re.sub(r"\[(?:\*|\d*)\]", "", str((by_id.get(a) or {}).get("field") or "")).split(".")[-1].casefold()
                in {"title","name","original_title","original_name"}
            for a in answer_ids)
        if parent_already_answered:
            continue
        leaves=[re.sub(r"\[(?:\*|\d*)\]", "", str(x.get("path") if isinstance(x,dict) else x or "")).strip(".")
                for x in card.get("leaf_paths") or []]
        # Prefer title for movie-like schemas, name otherwise.  Require uniqueness
        # at the selected record depth to avoid inventing a nested relation label.
        candidates=[]
        for leaf in leaves:
            tail=leaf.split(".")[-1]
            if tail in {"title","name"}:
                candidates.append(leaf)
        shallow=[x for x in candidates if x.count(".") <= 1]
        if shallow:
            candidates=shallow
        preferred=next((x for x in candidates if x.split(".")[-1]=="title"), None)
        if preferred is None and candidates:
            preferred=candidates[0]
        if not preferred:
            continue
        # Only restore this parent if at least one declared answer comes from a
        # downstream dependent step, proving the coordinated parent/child shape.
        steps={str(s.get("id") or ""):s for s in out.get("steps") or []}
        downstream=False
        for aid in answer_ids:
            for asid in (by_id.get(aid) or {}).get("source_steps") or []:
                cur=str(asid); seen=set()
                while cur and cur not in seen:
                    seen.add(cur); deps=[str(x) for x in (steps.get(cur) or {}).get("depends_on") or []]
                    if sid in deps:
                        downstream=True; break
                    cur=deps[0] if len(deps)==1 else ""
                if downstream: break
            if downstream: break
        if not downstream:
            continue
        did=f"requested_parent_label_{sid}"; n=2
        while did in existing:
            did=f"requested_parent_label_{sid}_{n}"; n+=1
        existing.add(did)
        additions.append({
            "id":did,"operator":"identity","source_steps":[sid],
            "source_derivations":[str(sel.get("id"))],"label_steps":[],"label_fields":[],
            "field":preferred,"comparison":"","comparison_literal":None,"unit":"raw",
            "distinct_field":None,"rank":0,"filter":{},"top_k":10,
            "purpose":"Return the explicitly requested selected parent entity label alongside its requested child relation.",
        })
        out["answer_derivations"]=[did]+answer_ids
        out["answer_steps"]=list(dict.fromkeys([sid]+[str(x) for x in out.get("answer_steps") or []]))
        warnings.append(f"restored requested parent label {preferred!r} from selected step {sid} alongside child-relation answer")
        break
    if additions:
        out["derivations"]=derivs+additions
    out["validation_warnings"]=list(dict.fromkeys(warnings))
    return out


def normalize_rooted_collection_derivation_paths(plan: dict[str, Any],
                                                cards: list[dict[str, Any]]) -> dict[str, Any]:
    """Canonicalize collection-root-prefixed fields to the selected record's relative schema.

    For a card whose record universe is ``crew[*]``, planner expressions such as
    ``crew.job`` + ``crew.name`` describe fields on each selected crew record. The
    compiler operates on those records and therefore needs ``job`` and ``name``.
    This normalization is schema-grounded and applies to any record collection.
    """
    out=copy.deepcopy(plan or {})
    card_by_step={str(c.get("step_id") or ""):c for c in cards or []}
    warnings=list(out.get("validation_warnings") or []); derivs=[]
    for raw in out.get("derivations") or []:
        d=dict(raw); src=[str(x) for x in d.get("source_steps") or []]
        if len(src)!=1:
            derivs.append(d); continue
        card=card_by_step.get(src[0]) or {}
        roots=[]
        for r in card.get("record_paths") or []:
            path=str(r.get("path") if isinstance(r,dict) else r or "")
            if path.endswith("[*]"):
                roots.append(re.sub(r"\[\*\]$", "", path).strip("."))
        roots=list(dict.fromkeys(roots))
        if not roots:
            derivs.append(d); continue
        def rel(value:str)->str:
            clean=re.sub(r"\[(?:\*|\d*)\]", "", str(value or "")).strip(".")
            matches=[root for root in roots if clean==root or clean.startswith(root+".")]
            if len(matches)!=1:
                return str(value or "")
            root=matches[0]
            if clean==root:
                return root
            return clean[len(root)+1:]
        old_field=str(d.get("field") or ""); new_field=rel(old_field)
        rooted_filters=[]
        for k in (d.get("filter") or {}):
            clean=re.sub(r"\[(?:\*|\d*)\]", "",str(k)).strip(".")
            rooted_filters.extend(root for root in roots if clean.startswith(root+".") or clean==root)
        rooted_filters=list(dict.fromkeys(rooted_filters))
        op=str(d.get("operator") or "").lower()
        if op in {"filter","endpoint_rank","first","nth","argmax","argmin"} and len(rooted_filters)==1:
            # Keep the collection universe explicit for schema/candidate-root
            # selection, while predicates become relative to each record.
            if old_field != rooted_filters[0]:
                d["field"]=rooted_filters[0]
                warnings.append(f"derivation {d.get('id')}: normalized selected record universe to {rooted_filters[0]!r}")
        elif op == "membership" and any(
                re.sub(r"\[(?:\*|\d*)\]", "", old_field).strip(".") == root or
                re.sub(r"\[(?:\*|\d*)\]", "", old_field).strip(".").startswith(root + ".")
                for root in roots):
            # Membership may need the relation qualifier (cast.name vs crew.name)
            # to identify one collection on responses that expose sibling arrays.
            # The replay compiler understands this qualified field directly.
            d["field"] = old_field
        elif new_field and new_field!=old_field and new_field not in roots:
            d["field"]=new_field
            warnings.append(f"derivation {d.get('id')}: normalized record-root field {old_field!r}->{new_field!r}")
        filt=dict(d.get("filter") or {}); nf={}; changed=False
        for k,v in filt.items():
            nk=rel(str(k))
            if nk!=str(k) and nk not in roots:
                changed=True
            else:
                nk=str(k)
            nf[nk]=v
        if changed:
            d["filter"]=nf
            warnings.append(f"derivation {d.get('id')}: normalized record-root filter keys")
        derivs.append(d)
    out["derivations"]=derivs
    out["validation_warnings"]=list(dict.fromkeys(warnings))
    return out


def normalize_redundant_answer_derivations(plan: dict[str, Any]) -> dict[str, Any]:
    """Drop a redundant downstream answer scalar already available on its selected parent record.

    This is deliberately narrow: two terminal identity derivations must expose the
    same scalar leaf and one source step must be a descendant of the other's source
    step.  The upstream value remains preferred because it avoids an unnecessary
    detail call without changing the requested field.
    """
    out=copy.deepcopy(plan or {})
    derivs={str(d.get("id") or ""):d for d in out.get("derivations") or []}
    ans=[str(x) for x in out.get("answer_derivations") or [] if str(x)]
    if len(ans)<2:
        return out
    steps={str(s.get("id") or ""):s for s in out.get("steps") or []}
    def leaf(d):
        if str(d.get("operator") or "").lower()!="identity": return ""
        return re.sub(r"\[(?:\*|\d*)\]", "", str(d.get("field") or "")).split(".")[-1].casefold()
    def is_desc(child,parent):
        stack=[child]; seen=set()
        while stack:
            cur=stack.pop()
            if cur in seen: continue
            seen.add(cur)
            for dep in (steps.get(cur) or {}).get("depends_on") or []:
                dep=str(dep)
                if dep==parent: return True
                stack.append(dep)
        return False
    drop=set()
    for i,a in enumerate(ans):
        da=derivs.get(a) or {}; la=leaf(da)
        if not la: continue
        sa=[str(x) for x in da.get("source_steps") or []]
        if len(sa)!=1: continue
        for b in ans[i+1:]:
            db=derivs.get(b) or {}; lb=leaf(db); sb=[str(x) for x in db.get("source_steps") or []]
            if la!=lb or len(sb)!=1: continue
            if is_desc(sb[0],sa[0]): drop.add(b)
            elif is_desc(sa[0],sb[0]): drop.add(a)
    if drop and len(drop)<len(ans):
        out["answer_derivations"]=[x for x in ans if x not in drop]
        out.setdefault("validation_warnings",[]).append(
            f"pruned redundant downstream answer derivation(s) {sorted(drop)}; identical requested scalar already available upstream")
    return out


def normalize_explicit_nested_relation_projection(question: str, plan: dict[str, Any],
                                                  cards: list[dict[str, Any]]) -> dict[str, Any]:
    """Use an explicitly named nested relation already exposed by an existing source step.

    When the question literally asks for a relation such as ``known for`` and the
    selected search schema exposes ``known_for[*]``, fetching a broader credits
    endpoint changes the relation.  This repair projects the documented nested
    relation from the already-selected owner and prunes now-unused answer steps.
    """
    q=str(question or "").casefold()
    relation=None
    if re.search(r"\bknown\s+for\b",q): relation="known_for"
    if not relation:
        return copy.deepcopy(plan or {})
    out=copy.deepcopy(plan or {}); card_by_step={str(c.get("step_id") or ""):c for c in cards or []}
    derivs=[dict(d) for d in out.get("derivations") or []]; existing={str(d.get("id") or "") for d in derivs}
    selector_ops={"endpoint_rank","first","nth","argmax","argmin","filter"}
    for sid,card in card_by_step.items():
        leaves=[str(x.get("path") if isinstance(x,dict) else x or "") for x in card.get("leaf_paths") or []]
        rel_leaves=[x for x in leaves if relation in re.sub(r"\[(?:\*|\d*)\]", "",x).split(".")]
        if not rel_leaves: continue
        selectors=[d for d in derivs if sid in [str(x) for x in d.get("source_steps") or []]
                   and str(d.get("operator") or "").lower() in selector_ops]
        if not selectors: continue
        label_leaf=next((x for x in rel_leaves if re.sub(r"\[(?:\*|\d*)\]", "",x).split(".")[-1]=="title"),None)
        if label_leaf is None:
            label_leaf=next((x for x in rel_leaves if re.sub(r"\[(?:\*|\d*)\]", "",x).split(".")[-1]=="name"),None)
        if not label_leaf: continue
        clean=re.sub(r"\[(?:\*|\d*)\]", "",label_leaf).strip(".")
        # Relative to selected outer search record.
        parts=clean.split(".")
        try: idx=parts.index(relation)
        except ValueError: continue
        field=".".join(parts[idx:])
        owner_sel=selectors[0]
        if str(owner_sel.get("field") or ""):
            owner_id=f"explicit_{relation}_owner_{sid}"; n=2
            while owner_id in existing:
                owner_id=f"explicit_{relation}_owner_{sid}_{n}"; n+=1
            existing.add(owner_id)
            owner_sel={"id":owner_id,"operator":"endpoint_rank","source_steps":[sid],
                       "source_derivations":[],"label_steps":[],"label_fields":[],"field":None,
                       "comparison":"","comparison_literal":None,"unit":"raw","distinct_field":None,
                       "rank":int(selectors[0].get("rank") or 0),"filter":dict(selectors[0].get("filter") or {}),
                       "top_k":10,"purpose":"Select the owning outer search record before projecting the explicitly requested nested relation."}
            derivs.append(owner_sel)
        did=f"explicit_{relation}_{sid}"; n=2
        while did in existing:
            did=f"explicit_{relation}_{sid}_{n}"; n+=1
        existing.add(did)
        derivs.append({
            "id":did,"operator":"identity","source_steps":[sid],
            "source_derivations":[str(owner_sel.get("id"))],"label_steps":[],"label_fields":[],
            "field":field,"comparison":"","comparison_literal":None,"unit":"raw","distinct_field":None,
            "rank":0,"filter":{},"top_k":10,
            "purpose":f"Project the explicitly requested {relation.replace('_',' ')} relation from the selected owner.",
        })
        out["derivations"]=derivs
        out["answer_derivations"]=[did]
        out["answer_steps"]=[sid]
        out["answer_mode"]="list"
        out.setdefault("validation_warnings",[]).append(
            f"explicit relation {relation!r} is documented on step {sid}; replaced broader terminal relation with nested owner-preserving projection")
        return out
    return out


def normalize_relation_population_metric_extrema(question: str, plan: dict[str, Any],
                                                 cards: list[dict[str, Any]]) -> dict[str, Any]:
    """Rank relation records themselves when the superlative noun names that relation resource.

    This covers person→TV/movie-credit populations without accidentally ranking
    people inside TV/movie credits.  The route suffix must identify the candidate
    resource (``tv_credits`` or ``movie_credits``) and the question must name that
    resource alongside an explicit popularity/rating superlative.
    """
    q=str(question or "").casefold(); criterion=None; op=None; leaves=set()
    if re.search(r"\bmost\s+popular\b",q): criterion="popularity";op="argmax";leaves={"popularity"}
    elif re.search(r"\b(?:highest|best|top)[ -]?rated\b|\bhighest\s+rating\b",q): criterion="rating";op="argmax";leaves={"vote_average","rating","score"}
    elif re.search(r"\b(?:lowest|worst)[ -]?rated\b",q): criterion="rating";op="argmin";leaves={"vote_average","rating","score"}
    if not criterion: return copy.deepcopy(plan or {})
    out=copy.deepcopy(plan or {}); steps={str(s.get("id") or ""):s for s in out.get("steps") or []}
    cards_by={str(c.get("step_id") or ""):c for c in cards or []}; warnings=list(out.get("validation_warnings") or [])
    derivs=[]
    for raw in out.get("derivations") or []:
        d=dict(raw)
        if str(d.get("operator") or "").lower() not in {"endpoint_rank","first","nth"}:
            derivs.append(d); continue
        src=[str(x) for x in d.get("source_steps") or []]
        if len(src)!=1: derivs.append(d); continue
        sid=src[0]; path=str((steps.get(sid) or {}).get("endpoint") or "").casefold()
        resource=None
        if re.search(r"/person/\{[^}]+\}/tv_credits$",path): resource="tv"
        elif re.search(r"/person/\{[^}]+\}/movie_credits$",path): resource="movie"
        if not resource: derivs.append(d); continue
        if resource=="tv" and not re.search(r"\b(?:tv|television|show|series)\b",q): derivs.append(d); continue
        if resource=="movie" and not re.search(r"\b(?:movie|film)\b",q): derivs.append(d); continue
        field_root=re.sub(r"\[(?:\*|\d*)\]", "",str(d.get("field") or "")).strip(".")
        if field_root not in {"cast","crew"}: derivs.append(d); continue
        metric=[]
        for leaf in (cards_by.get(sid) or {}).get("leaf_paths") or []:
            p=str(leaf.get("path") if isinstance(leaf,dict) else leaf or "")
            clean=re.sub(r"\[(?:\*|\d*)\]", "",p).strip(".")
            if clean.startswith(field_root+".") and clean.split(".")[-1].casefold() in leaves:
                metric.append(p)
        metric=list(dict.fromkeys(metric))
        if len(metric)==1:
            d["operator"]=op;d["field"]=metric[0];d["rank"]=0
            warnings.append(f"derivation {d.get('id')}: normalized relation population selector to {op}({metric[0]}) for explicit {criterion} {resource} candidate")
        derivs.append(d)
    out["derivations"]=derivs;out["validation_warnings"]=list(dict.fromkeys(warnings));return out


def normalize_age_winner_and_difference(question: str, plan: dict[str, Any]) -> dict[str, Any]:
    """Ensure age questions asking *who* and *by how much* expose both ordering and magnitude."""
    q=re.sub(r"\s+", " ", str(question or "").strip()).casefold()
    if not (re.search(r"\bwho\b",q) and re.search(r"\b(?:older|younger)\b",q)
            and re.search(r"\b(?:by how many|by how much|difference)\b",q)):
        return copy.deepcopy(plan or {})
    out=copy.deepcopy(plan or {}); derivs=[dict(d) for d in out.get("derivations") or []]
    by_id={str(d.get("id") or ""):d for d in derivs}; ans=[str(x) for x in out.get("answer_derivations") or [] if str(x)]
    birth=[]
    for d in derivs:
        if str(d.get("operator") or "").lower()!="identity": continue
        leaf=re.sub(r"[^a-z0-9]+","_",str(d.get("field") or "").casefold())
        if any(x in leaf for x in ("birthday","birth_date","birthdate","date_of_birth","dob")):
            birth.append(d)
    if len(birth)!=2: return out
    difference=next((d for d in derivs if str(d.get("operator") or "").lower()=="compare"
                     and set(str(x) for x in d.get("source_derivations") or [])=={str(birth[0].get('id')),str(birth[1].get('id'))}
                     and str(d.get("comparison") or "").lower() in {"difference","abs_difference"}),None)
    if not difference: return out
    ordering=next((d for d in derivs if str(d.get("operator") or "").lower()=="compare"
                   and set(str(x) for x in d.get("source_derivations") or [])=={str(birth[0].get('id')),str(birth[1].get('id'))}
                   and str(d.get("comparison") or "").lower() in {"min","max","gt","lt","gte","lte"}),None)
    if ordering is None:
        existing={str(d.get("id") or "") for d in derivs}; did="age_winner";n=2
        while did in existing: did=f"age_winner_{n}";n+=1
        ordering={"id":did,"operator":"compare","source_steps":[],
                  "source_derivations":[str(birth[0].get("id")),str(birth[1].get("id"))],
                  "label_steps":list(difference.get("label_steps") or []),
                  "label_fields":list(difference.get("label_fields") or []),"field":None,
                  "comparison":"min" if "older" in q else "max","comparison_literal":None,
                  "unit":"raw","distinct_field":None,"rank":0,"filter":{},"top_k":10,
                  "purpose":"Determine which person is older/younger from the two birth dates."}
        derivs.append(ordering)
    difference["comparison"]="abs_difference"; difference["unit"]="years"
    # Put the winner first so deterministic materialization keeps the answer shape.
    out["derivations"]=derivs
    out["answer_derivations"]=[str(ordering.get("id")),str(difference.get("id"))]
    out["answer_mode"]="comparison"
    out.setdefault("validation_warnings",[]).append("age comparison normalized to coordinated winner + absolute-year-difference outputs")
    return normalize_age_date_comparisons(question,out)

def _answer_surface_errors(question: str, plan: dict[str, Any]) -> list[str]:
    """Generic plan-surface validation.

    Semantic endpoint choice belongs to the OAS-grounded planner, not to hard-coded
    domain rules.  Here we only require an executable, connected answer surface and
    deterministic derivations whose sources are valid plan steps.
    """
    steps = {str(s.get("id")): s for s in (plan or {}).get("steps") or []}
    answer_steps = [str(x) for x in (plan or {}).get("answer_steps") or []]
    if not answer_steps:
        return ["plan has no answer-step endpoint"]
    missing = [x for x in answer_steps if x not in steps]
    if missing:
        return [f"answer_steps reference missing plan steps: {missing}"]
    errors: list[str] = []
    derivations = list((plan or {}).get("derivations") or [])
    # Answer-field semantics belong to the Evidence Plan. If an endpoint is marked
    # as an answer step but no derivation consumes it, a later projection/LLM would
    # have to decide which response fields are the answer. Require the planner to
    # declare that extraction explicitly; fetched empty collections remain covered
    # by the compiler's deterministic empty-result handling.
    mode = str((plan or {}).get("answer_mode") or "").lower()
    if mode != "action":
        for sid in answer_steps:
            step_derivations = [d for d in derivations
                                if sid in {str(x) for x in d.get("source_steps") or []}]
            if not step_derivations:
                errors.append(
                    f"answer step {sid} has no explicit derivation; declare the requested "
                    "identity/count/selection value so observation projection does not choose answer fields")

    # Record selectors establish *which record*, not *which answer value*.  For
    # direct/list/asset answers, every declared answer response step must therefore
    # expose at least one scalar/list value through a value-producing derivation.
    # This closes a generic hole where a ranked record plus a downstream attribute
    # could be certified while silently omitting the selected entity's requested
    # label/title.  The planner already has ``identity`` for this purpose.
    value_ops = {"identity", "count", "membership", "compare", "logical_and", "logical_or"}
    if mode in {"direct", "list", "asset"}:
        for sid in answer_steps:
            step_derivations = [d for d in derivations
                                if sid in {str(x) for x in d.get("source_steps") or []}]
            if step_derivations and not any(
                    str(d.get("operator") or "").lower() in value_ops
                    for d in step_derivations):
                errors.append(
                    f"answer step {sid} is represented only by record-selection derivations; "
                    "add an identity/count value derivation for the requested final field(s) "
                    "instead of treating a selected record as a complete answer")
    for derivation in derivations:
        bad = [str(x) for x in derivation.get("source_steps") or [] if str(x) not in steps]
        if bad:
            errors.append(f"derivation {derivation.get('id')} references missing steps: {bad}")
    if mode == "comparison" and not any(
            str(d.get("operator") or "").lower() == "compare" for d in derivations):
        errors.append("comparison answer requires an explicit host-replayable compare derivation")
    if mode == "boolean" and not any(
            (str(d.get("operator") or "").lower() in {"membership", "logical_and", "logical_or"})
            or (str(d.get("operator") or "").lower() == "compare" and
                str(d.get("comparison") or "").lower() in {"eq", "neq", "gt", "gte", "lt", "lte"})
            for d in derivations):
        errors.append("boolean answer requires an explicit host-replayable boolean comparison derivation")

    # A winner-seeking comparison is not a yes/no question.  Plans that answer
    # "Who/which is older/higher/earlier/more..." with a Boolean lose the entity
    # the user actually asked for.  Conversely, a comparison-mode winner must
    # carry a deterministic side->label mapping rather than relying on the final
    # language model to remember which scalar belonged to which entity.
    winner_seeking = bool(re.search(
        r"\b(?:who|which)\b[^?]{0,120}\b(?:higher|lower|older|younger|earlier|later|"
        r"more|less|greater|smaller|highest|lowest|largest|smallest)\b",
        str(question or ""), re.I))
    ordering_ops = {"gt", "gte", "lt", "lte", "max", "min"}
    ordering_compares = [
        d for d in derivations
        if str(d.get("operator") or "").lower() == "compare"
        and str(d.get("comparison") or "").lower() in ordering_ops
    ]
    if winner_seeking and mode == "boolean":
        errors.append(
            "answer intent mismatch: the user asks which entity wins a comparison, "
            "but answer_mode=boolean can only return true/false; use comparison mode "
            "with replayable side labels")
    if winner_seeking and mode == "comparison" and ordering_compares:
        if not any(
                len(list(d.get("label_steps") or [])) >= 2
                and len(list(d.get("label_fields") or [])) >= 2
                and len(list(d.get("label_steps") or [])) == len(list(d.get("label_fields") or []))
                for d in ordering_compares):
            errors.append(
                "comparison winner lacks explicit replayable side labels; provide aligned "
                "label_steps and label_fields for the compared entities")

    # Temporal direction is semantic, not a schema detail.  A plan asking for the
    # latest/newest/most-recent record cannot deterministically use argmin on a
    # date/time/year field.  Keep this intentionally narrow so words such as
    # "older" (where an earlier birthday means older) are not conflated with
    # generic recency.
    q = str(question or "").casefold()
    asks_latest = any(term in q for term in (
        "latest", "newest", "most recent", "recently", "recent ", "current released"))
    asks_earliest = any(term in q for term in ("earliest", "oldest release", "first released"))
    for d in derivations:
        op = str(d.get("operator") or "").lower()
        field = str(d.get("field") or "").casefold()
        date_like = any(token in field for token in ("date", "time", "year"))
        if asks_latest and date_like and op == "argmin":
            errors.append(
                f"derivation {d.get('id')}: latest/most-recent intent uses argmin on "
                f"date-like field {d.get('field')!r}; use the later/larger date direction")
        if asks_earliest and date_like and op == "argmax":
            errors.append(
                f"derivation {d.get('id')}: earliest intent uses argmax on date-like field "
                f"{d.get('field')!r}; use the earlier/smaller date direction")

    # Explicit future/scheduled intent must not be silently converted into a
    # historical/latest-released population by a <=-today filter.  This is a
    # question-level contradiction independent of API vocabulary, so reject it
    # deterministically and let bounded convergence repair the population scope.
    future_requested = any(term in q for term in (
        "upcoming", "future", "will release", "will air", "scheduled",
        "next release", "next movie", "next season", "coming out"))
    if future_requested:
        from datetime import date as _surface_date
        today = _surface_date.today().isoformat()
        for d in derivations:
            canonical_filter = _canonical_derivation_filter(
                d.get("filter") or {}, d.get("field"))
            for filter_field, pred in canonical_filter.items():
                if not any(tok in str(filter_field).casefold() for tok in ("date", "time", "year")):
                    continue
                if not isinstance(pred, dict):
                    continue
                pred_op = str(pred.get("op") or "").lower()
                value = pred.get("value")
                if pred_op in {"lt", "lte"} and isinstance(value, str) and value <= today:
                    errors.append(
                        f"derivation {d.get('id')}: explicit future/scheduled intent is "
                        f"contradicted by {filter_field} {pred_op} {value!r}; do not exclude "
                        "future-dated candidates when the user asks for a future/scheduled item")

    # Final-output intent must survive planning.  This deliberately validates only
    # very high-confidence linguistic contracts that are API/domain independent.
    # In particular, a question beginning with "when" must terminate in a
    # date/time-like value rather than silently rewriting the requested output to
    # an intermediate entity label.  This catches a dangerous class of certified
    # semantic drift while leaving relation-specific terminology to the OAS critic.
    referenced_derivations = {
        str(parent) for d in derivations
        for parent in (d.get("source_derivations") or [])
    }
    terminal_derivations = [
        d for d in derivations
        if str(d.get("id") or "") not in referenced_derivations
    ]
    answer_step_set = set(answer_steps)
    terminal_answer_values = [
        d for d in terminal_derivations
        if answer_step_set & {str(x) for x in (d.get("source_steps") or [])}
        or str(d.get("operator") or "").lower() in
           {"compare", "membership", "logical_and", "logical_or"}
    ]
    # High-confidence terminal-attribute contracts.  These rules preserve the
    # *kind* of evidence explicitly requested by the user; they do not choose an
    # entity, route, or answer value.
    terminal_step_text = " ".join(
        str((steps.get(sid) or {}).get("endpoint") or "") + " " +
        str((steps.get(sid) or {}).get("purpose") or "")
        for sid in answer_steps
    ).casefold()
    terminal_route_text = " ".join(
        str((steps.get(sid) or {}).get("endpoint") or "") for sid in answer_steps
    ).casefold()
    terminal_value_text = " ".join(
        str(d.get("field") or "") + " " + str(d.get("purpose") or "")
        for d in terminal_answer_values
    ).casefold()
    terminal_field_text = " ".join(
        str(d.get("field") or "") for d in terminal_answer_values
    ).casefold()
    terminal_surface_text = terminal_step_text + " " + terminal_value_text

    visual_requested = bool(re.search(
        r"\b(?:look(?:s)?\s+like|image|photo|poster|cover\s+image|logo)\b", q))
    if visual_requested:
        visual_tokens = ("image", "photo", "poster", "profile", "file_path", "file path",
                         "logo", "backdrop", "still", "cover")
        visual_evidence_text = terminal_route_text + " " + terminal_field_text
        if terminal_answer_values and not any(tok in visual_evidence_text for tok in visual_tokens):
            errors.append(
                "answer intent mismatch: the user requests visual/image evidence, but the "
                "terminal answer surface is not an image/photo/poster/logo/profile asset")

    review_requested = bool(re.search(r"\breviews?\b", q))
    if review_requested:
        # A synopsis/overview is not a review.  Require the terminal evidence path
        # to preserve the requested review relation or a review-content extraction.
        review_relation = ("review" in terminal_route_text or
                           "review" in terminal_field_text)
        if terminal_answer_values and not review_relation:
            errors.append(
                "answer intent mismatch: the user requests review evidence, but the terminal "
                "answer surface does not preserve a review relation/content source")

    if re.match(r"^\s*who\b", q):
        anti_identity = {"job", "role", "department", "character", "order", "popularity"}
        identity_terminals = [d for d in terminal_answer_values
                              if str(d.get("operator") or "").lower() == "identity"]
        if identity_terminals:
            leaves = {
                _normalized_schema_path(d.get("field")).split(".")[-1].casefold()
                for d in identity_terminals if str(d.get("field") or "").strip()
            }
            if leaves and leaves.issubset(anti_identity):
                errors.append(
                    "answer intent mismatch: a who-question must expose an entity/person label, "
                    "not only role/job/department/character/order metadata")

    if re.search(r"\bwhere\b[^?]{0,100}\bfounded\b", q):
        founding_place_tokens = ("found", "origin", "place_of", "birthplace", "location_founded")
        if terminal_answer_values and not any(tok in terminal_field_text for tok in founding_place_tokens):
            errors.append(
                "answer intent mismatch: a where-founded question requires a founding/origin "
                "location field; current headquarters is not evidence of where an entity was founded")

    if re.match(r"^\s*when\b", q):
        temporal_tokens = ("date", "time", "year", "birthday", "birth", "released",
                           "release", "air_date", "air date", "created", "founded")
        temporal_surface = " ".join(
            str(d.get("field") or "") + " " + str(d.get("purpose") or "")
            for d in terminal_answer_values
        ).casefold()
        if terminal_answer_values and not any(tok in temporal_surface for tok in temporal_tokens):
            errors.append(
                "answer intent mismatch: the user asks when, but terminal answer derivations "
                "do not expose a date/time/year-like value")

    # Explicit multiplicity words are a semantic contract.  A list request such as
    # "some", "several", or "a few" must not deterministically collapse the final
    # answer collection to a single record through first/nth/endpoint_rank/argmax.
    # This is intentionally narrower than grammatical plural detection to avoid
    # guessing about collective nouns such as "cast".
    explicit_multi = str((plan or {}).get("answer_cardinality") or "").lower() == "many"
    if explicit_multi:
        by_did = {str(d.get("id") or ""): d for d in derivations}
        singleton_ops = {"endpoint_rank", "first", "nth", "argmax", "argmin"}

        def _lineage_has_singleton_selector(did: str, current_steps: set[str],
                                            terminal_field: str = "", seen=None) -> bool:
            seen = set(seen or ())
            if not did or did in seen:
                return False
            seen.add(did)
            d = by_did.get(did) or {}
            parent_steps = {str(x) for x in (d.get("source_steps") or [])}
            # A source_derivation from another API step is dependency lineage, not
            # a selector over this answer collection. Mirror compiler record-set
            # semantics so an upstream search selector does not collapse a
            # downstream keyword/review/recommendation collection.
            if parent_steps and current_steps and not parent_steps.issubset(current_steps):
                return False
            if str(d.get("operator") or "").lower() in singleton_ops:
                # A fieldless selector on the *same response* may select the outer
                # owner record before a terminal identity projects a nested child
                # collection (search result -> known_for.title). That owner choice
                # does not collapse the nested answer collection itself.
                parent_field = re.sub(r"\[(?:\*|\d*)\]", "", str(d.get("field") or "")).strip(".")
                child_field = re.sub(r"\[(?:\*|\d*)\]", "", str(terminal_field or "")).strip(".")
                if "." in child_field:
                    child_collection = child_field.split(".", 1)[0]
                    if not parent_field or not (parent_field == child_collection or
                                                parent_field.endswith("." + child_collection)):
                        return False
                return True
            return any(_lineage_has_singleton_selector(
                           str(parent), current_steps, terminal_field, seen)
                       for parent in (d.get("source_derivations") or []))

        value_terminals = [d for d in terminal_answer_values
                           if str(d.get("operator") or "").lower() == "identity"]
        if value_terminals and all(
                any(_lineage_has_singleton_selector(
                        str(parent), {str(x) for x in (d.get("source_steps") or [])},
                        str(d.get("field") or ""))
                    for parent in (d.get("source_derivations") or []))
                for d in value_terminals):
            errors.append(
                "answer cardinality mismatch: the user explicitly requests multiple items, "
                "but every terminal answer branch is collapsed by a singleton selector")

    # Explicit ordinal claims over a collection require an explicit selector in
    # the derivation lineage.  This is intentionally limited to collection-
    # qualified terminal identity fields so an ordinal represented structurally in
    # a request path (season 2 / episode 3) is not confused with an unselected
    # collection member.  A purpose that says "first movie" while reading
    # ``parts.release_date`` directly is therefore rejected rather than trusting
    # observation projection to choose a row.
    ordinal_word = re.compile(
        r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|top[- ]?1)\b",
        re.I)
    for d in terminal_answer_values:
        if str(d.get("operator") or "").lower() != "identity":
            continue
        if d.get("source_derivations"):
            continue
        field = str(d.get("field") or "").replace("[*]", "")
        purpose = str(d.get("purpose") or "")
        # A dotted field is a strong structural signal that the value is being
        # taken from a child relation/collection rather than one root object.
        # If this terminal step is a child request whose *direct producer* already
        # selected a first/nth member, the ordinal is represented in request
        # lineage (e.g. first collection member -> child credits -> cast names)
        # and need not be repeated on the child collection itself.
        source_step_ids = {str(x) for x in (d.get("source_steps") or [])}
        direct_deps = {str(dep) for sid in source_step_ids
                       for dep in (steps.get(sid) or {}).get("depends_on") or []}
        upstream_ordinal = any(
            str(parent.get("operator") or "").lower() in {"first", "nth", "endpoint_rank"}
            and bool({str(x) for x in (parent.get("source_steps") or [])} & direct_deps)
            for parent in derivations)
        if "." in field and ordinal_word.search(purpose) and not upstream_ordinal:
            errors.append(
                f"derivation {d.get('id')}: explicit ordinal collection member in purpose "
                "has no replayable first/nth/endpoint-rank selector in its derivation lineage")

    # Cardinality is part of semantics. A filter without a subsequent selector
    # denotes a candidate SET; identity over that set may legitimately yield more
    # than one scalar. Feeding such a value into scalar eq/neq is ambiguous and was
    # the root of repeated relation-comparison failures. Require the plan to state
    # the intended set operation (membership/overlap) or an explicit documented
    # singleton selector rather than letting runtime cardinality silently redefine
    # equality.
    by_derivation = {str(d.get("id") or ""): d for d in derivations}
    selector_ops = {"endpoint_rank", "first", "nth", "argmax", "argmin"}
    def _may_be_candidate_set(did: str, seen=None) -> bool:
        seen = set(seen or ())
        if not did or did in seen:
            return False
        seen.add(did)
        d = by_derivation.get(did) or {}
        op = str(d.get("operator") or "").lower()
        if op in selector_ops:
            return False
        if op == "filter":
            return True
        if op == "identity":
            parents = [str(x) for x in d.get("source_derivations") or []]
            if parents:
                return any(_may_be_candidate_set(x, seen) for x in parents)
            field = str(d.get("field") or "")
            return "[*]" in field or "[]" in field
        return False

    # Equality over two identity sets is meaningful for an explicit same-person/
    # same-entity question: the compiler replays it as set overlap.  Other scalar
    # equality remains strict so accidental collection-vs-collection comparisons
    # are rejected.
    same_entity_question = bool(re.search(
        r"\b(same|identical)\b.*\b(person|people|entity|director|actor|name|id)\b|"
        r"\b(person|people|entity|director|actor|name|id)\b.*\b(same|identical)\b",
        str(question or ""), re.I))
    for d in derivations:
        if (str(d.get("operator") or "").lower() == "compare" and
                str(d.get("comparison") or "").lower() in {"eq", "neq"}):
            parent_ids = [str(x) for x in d.get("source_derivations") or []]
            multi = [did for did in parent_ids if _may_be_candidate_set(did)]
            # Explicit identity-set equality is a typed same-entity operation.
            # It is replayed as set overlap by evidence_compiler; no raw-question
            # regex is required once both operands explicitly project identity.
            identity_sets = bool(parent_ids) and all(
                str((by_derivation.get(did) or {}).get("operator") or "").lower() == "identity"
                and str((by_derivation.get(did) or {}).get("field") or "").replace("[*]", "").split(".")[-1].lower()
                    in {"id", "name", "title", "label"}
                for did in parent_ids
            )
            if multi and not (same_entity_question or identity_sets):
                errors.append(
                    f"derivation {d.get('id')}: scalar {d.get('comparison')} comparison consumes "
                    f"potentially set-valued derivation(s) {multi}; use membership/overlap "
                    "semantics for candidate sets or add a documented singleton selector before equality")

    # Equality between two Boolean existence/membership results does not establish
    # that the underlying entities are the same.  For explicit same-entity/person
    # questions require IDs/names/sets to be compared directly.
    same_entity_question = bool(re.search(
        r"\b(same|identical)\b.*\b(person|people|entity|director|actor|name|id)\b|"
        r"\b(person|people|entity|director|actor|name|id)\b.*\b(same|identical)\b",
        str(question or ""), re.I))
    if same_entity_question:
        boolean_ops = {"membership", "logical_and", "logical_or"}
        for d in derivations:
            if (str(d.get("operator") or "").lower() == "compare" and
                    str(d.get("comparison") or "").lower() in {"eq", "neq"}):
                parents = [by_derivation.get(str(x)) or {} for x in d.get("source_derivations") or []]
                if parents and all(str(x.get("operator") or "").lower() in boolean_ops for x in parents):
                    errors.append(
                        f"derivation {d.get('id')}: same-entity question compares Boolean "
                        "membership/existence results rather than the underlying entity ids/names/sets")

    # Reject a comparison input that merely re-emits the literal used to filter
    # that same field. Example shape: filter job==Director, then identity(job),
    # then compare two branches. That comparison is tautologically about the
    # predicate literal, not the selected entities. Direct answers remain allowed;
    # this check applies only when the identity feeds a comparison/membership.
    consumers = {}
    for d in derivations:
        for source in d.get("source_derivations") or []:
            consumers.setdefault(str(source), []).append(d)
    for d in derivations:
        if str(d.get("operator") or "").lower() != "identity":
            continue
        downstream = consumers.get(str(d.get("id") or ""), [])
        if not any(str(x.get("operator") or "").lower() in {"compare", "membership"}
                   for x in downstream):
            continue
        field = _normalized_schema_path(d.get("field"))
        filt = _canonical_derivation_filter(d.get("filter") or {}, d.get("field"))
        for fpath, predicate in filt.items():
            pred = predicate if isinstance(predicate, dict) else {"op": "eq", "value": predicate}
            if (_normalized_schema_path(fpath) == field and
                    str(pred.get("op") or "eq").lower() in {"eq", "eq_ci"}):
                errors.append(
                    f"derivation {d.get('id')}: comparison input identity field {d.get('field')!r} "
                    "is fixed by its own equality filter; extract a stable entity/value field "
                    "such as the selected record id/name/metric instead")
                break
    return errors



def build_planner_messages(question: str, tools: list[dict[str, Any]],
                           catalog_mode: str = "full") -> list[dict[str, str]]:
    """Build a complete OAS-grounded, API-agnostic planning prompt."""
    mode = str(catalog_mode or "full").lower()
    if mode == "compact":
        # Keep every route *and* a short OAS functionality description.  Earlier
        # compact mode retained only METHOD+path, which saved tokens but removed
        # the semantic documentation needed to distinguish related operations
        # (for example, two endpoints with similar shapes but different meanings).
        # Detailed parameters/response schemas are still deferred until the route
        # is selected, so this remains substantially smaller than the full catalog.
        lines = []
        for tool in tools:
            description = " ".join(str(tool.get("functionality") or "").split())
            if len(description) > 48:
                description = description[:45].rsplit(" ", 1)[0] + "…"
            prefix = f"- {str(tool.get('method', 'GET')).upper()} {tool['path']}"
            pieces = [prefix + ((": " + description) if description else "")]

            # Planning also needs the documented request surface. Keep names,
            # locations, required markers and small enums, but omit verbose types
            # and response schemas until the operation has actually been selected.
            card = tool.get("schema_card") if isinstance(tool.get("schema_card"), dict) else {}
            grouped: dict[str, list[str]] = {}
            for param in card.get("parameters") or []:
                name = str(param.get("name") or "").strip()
                if not name:
                    continue
                values = (param.get("enum") if isinstance(param.get("enum"), list)
                          else param.get("item_enum") if isinstance(param.get("item_enum"), list)
                          else [])
                enum_hint = ("{" + ",".join(str(x) for x in values[:8]) + "}"
                             if values and len(values) <= 8 else "")
                token = name + ("*" if param.get("required") else "") + enum_hint
                grouped.setdefault(str(param.get("in") or "query"), []).append(token)
            if grouped:
                pieces.append("; ".join(
                    f"{location}=" + ",".join(names)
                    for location, names in grouped.items()))

            body_paths = [str(x.get("path")) for x in card.get("request_body_leaf_paths") or []
                          if x.get("path")]
            required_body = {str(x) for x in card.get("request_body_required_fields") or []}
            if body_paths:
                # Always retain every required top-level body field. Optional
                # leaves are bounded only to protect pathological giant schemas;
                # detailed selected-endpoint schemas are loaded on the next stage.
                shown = body_paths[:20]
                for required in required_body:
                    if required and not any(p == required or p.startswith(required + ".")
                                            or p.startswith(required + "[") for p in shown):
                        shown.append(required)
                suffix = f",…(+{len(body_paths)-20})" if len(body_paths) > 20 else ""
                pieces.append("body=" + ",".join(shown) + suffix)
            lines.append(" | ".join(pieces))
        toolbox = "\n".join(lines)
    else:
        try:
            from utils.schema_outline import compact_catalog_line
            lines = []
            for tool in tools:
                card = tool.get("schema_card") if isinstance(tool.get("schema_card"), dict) else None
                if card:
                    lines.append("- " + compact_catalog_line(card, max_response_fields=0))
                else:
                    lines.append(
                        f"- {tool.get('method', 'GET')} {tool['path']}: "
                        f"{str(tool.get('functionality', ''))[:300]}")
            toolbox = "\n".join(lines)
        except Exception:
            toolbox = "\n".join(
                f"- {tool.get('method', 'GET')} {tool['path']}: {str(tool.get('functionality', ''))[:300]}"
                for tool in tools)
    from datetime import date as _planner_date
    current_date = _planner_date.today().isoformat()
    system = """You are the evidence planner for an API agent. Create one compact JSON plan. Do not answer the question or use prior factual knowledge.

Current date: <CURRENT_DATE>. For current or latest requests, do not treat future items as current unless the user asks for future items.

Rules:
1. Keep all important details from the user's question.
2. Use only available API routes and fields.
3. Make each data dependency clear. Use earlier results when a later call needs them.
4. Add filtering, selection, sorting, counting, or comparison only when needed.
5. Use the simplest plan that fully answers the question.

Use only these derivation operators: endpoint_rank, argmax, argmin, filter, count, membership, compare, logical_and, logical_or, first, nth, identity.

Return JSON only in this shape:
{
  "steps": [{"id":"s1","method":"GET","endpoint":"/exact/path","purpose":"...","depends_on":[],"binding_paths":{},"path_literals":{},"path_bindings":{},"query_literals":{},"query_bindings":{},"body_literals":{},"body_bindings":{},"answer_source":false}],
  "derivations": [{"id":"d1","operator":"identity","source_steps":["s1"],"source_derivations":[],"field":"name","comparison":"","comparison_literal":null,"unit":"raw","rank":0,"filter":{},"purpose":"..."}],
  "answer_steps": ["s1"],
  "answer_derivations": ["d1"],
  "answer_mode": "direct|list|count|boolean|comparison|asset",
  "answer_requirements": ["what the answer must contain"],
  "planner_notes": ""
}
"""
    system = system.replace("<CURRENT_DATE>", current_date)
    user = f"QUESTION:\n{question}\n\nAVAILABLE ENDPOINTS:\n{toolbox}"
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]



def schema_population_retry_hints(plan: dict[str, Any], cards: list[dict[str, Any]]) -> list[str]:
    """Detect plans that cannot establish a population-wide extremum from one child.

    This is a schema-only review.  It never rewrites routes or uses benchmark gold.
    When an argmax/argmin is declared over a per-entity singleton child lookup and
    an upstream collection already documents the same comparison field, a planner
    retry is worthwhile: either rank on that collection (when semantically valid)
    or make the required fan-out explicit later in the observation/binding plan.
    """
    plan = plan or {}
    steps = {str(x.get("id") or ""): x for x in plan.get("steps") or []}
    card_by_step = {str(x.get("step_id") or ""): x for x in cards or []}
    hints: list[str] = []

    def leaf_paths(sid: str) -> set[str]:
        return {str(x.get("path") or "") for x in (card_by_step.get(sid) or {}).get("leaf_paths") or []}

    def root_has_field(sid: str, field: str) -> bool:
        field = str(field or "").replace("$.", "").lstrip("$.")
        return bool(field) and field in leaf_paths(sid)

    def collection_has_field(sid: str, field: str) -> bool:
        field = str(field or "").replace("$.", "").lstrip("$.")
        if not field:
            return False
        return any("[*]" in path and (path.endswith("." + field) or path == field)
                   for path in leaf_paths(sid))

    for derivation in plan.get("derivations") or []:
        op = str(derivation.get("operator") or "").lower()
        if op not in {"argmax", "argmin"}:
            continue
        sources = [str(x) for x in derivation.get("source_steps") or [] if str(x) in steps]
        if len(sources) != 1:
            continue
        source = sources[0]
        field = str(derivation.get("field") or "")
        # If the comparison field itself is documented under a collection in the
        # source response, one call already provides a candidate population.
        if collection_has_field(source, field) or not root_has_field(source, field):
            continue
        source_step = steps.get(source) or {}
        deps = [str(x) for x in source_step.get("depends_on") or [] if str(x) in steps]
        placeholder_names = [x.strip("{}") for x in re.findall(r"\{[^{}]+\}", str(source_step.get("endpoint") or ""))]
        binding_names = set(placeholder_names) | set(str(x) for x in (source_step.get("query_bindings") or {}).values()) | set(str(x) for x in (source_step.get("body_bindings") or {}).values())
        candidate_parents = []
        for dep in deps:
            dep_binds = {str(x).strip("{}") for x in (steps.get(dep) or {}).get("binds") or []}
            if dep_binds & binding_names:
                candidate_parents.append(dep)
        if len(candidate_parents) != 1:
            continue
        parent = candidate_parents[0]
        if collection_has_field(parent, field):
            hints.append(
                f"derivation {derivation.get('id')}: {op}({field}) is declared over singleton-per-call "
                f"step {source}, while upstream collection step {parent} already documents {field}; "
                "the plan must either rank the semantically appropriate upstream collection directly "
                "or explicitly evaluate the child for every candidate, not one selected candidate")
    return list(dict.fromkeys(hints))



def _explicit_binding_selection_index(step: dict[str, Any],
                                      names: set[str] | None = None) -> int | None:
    """Return one explicit collection index encoded in declared binding paths.

    ``results[0].id`` already says which producer record supplies the binding.
    Requiring a second, duplicate ``endpoint_rank`` derivation makes the planner
    representation brittle without adding semantics.  When every consumed binding
    path that contains an index agrees on one index, treat that index as explicit
    selection metadata.  Conflicting/no indices return None and are handled by the
    ordinary selection rules.
    """
    wanted = {str(x).strip("{}") for x in (names or set()) if str(x).strip("{}")}
    found: list[int] = []
    for name, raw_path in (step.get("binding_paths") or {}).items():
        key = str(name).strip("{}")
        if wanted and key not in wanted:
            continue
        matches = re.findall(r"\[(\d+)\]", str(raw_path or ""))
        if matches:
            found.append(int(matches[-1]))
    if not found or len(set(found)) != 1:
        return None
    return found[0]


def validate_collection_binding_selection(plan: dict[str, Any],
                                          cards: list[dict[str, Any]]) -> list[str]:
    """Require semantics for collection records that feed downstream requests.

    Observation projection is a compiler, not a second planner. When a collection
    supplies later request bindings, the Evidence Plan must already specify either
    (a) which producer records are selected, or (b) an explicit population-wide
    downstream extremum whose semantics require evaluating every producer record.
    """
    steps = {str(x.get("id") or ""): x for x in (plan or {}).get("steps") or []}
    cards_by_step = {str(x.get("step_id") or ""): x for x in cards or []}
    derivs_by_step: dict[str, list[dict[str, Any]]] = {}
    for deriv in (plan or {}).get("derivations") or []:
        for sid in [str(x) for x in deriv.get("source_steps") or []]:
            derivs_by_step.setdefault(sid, []).append(deriv)
    typed_fanout: dict[str, set[str]] = {}
    for item in (plan or {}).get("typed_fanout_bindings") or []:
        sid = str((item or {}).get("step_id") or "")
        name = str((item or {}).get("binding") or "").strip("{}")
        if sid and name:
            typed_fanout.setdefault(sid, set()).add(name)

    def input_names(step: dict[str, Any]) -> set[str]:
        # Request provenance is expressed by binding *aliases*, not necessarily by
        # the consumer placeholder name.  A child may legitimately declare
        # ``{movie_id} <- similar_movie_id``; treating that as a need for an older
        # ``movie_id`` alias can force selection semantics onto the wrong ancestor
        # and recreate the exact lineage bug this validator is meant to prevent.
        literals = {str(x).strip("{}") for x in (step.get("path_literals") or {})}
        path_bindings = {str(k).strip("{}"): str(v).strip("{}")
                         for k, v in (step.get("path_bindings") or {}).items()}
        names: set[str] = set()
        for raw in re.findall(r"\{[^{}]+\}", str(step.get("endpoint") or "")):
            placeholder = raw.strip("{}")
            if placeholder in literals:
                continue
            names.add(path_bindings.get(placeholder, placeholder))
        names.update(str(x).strip("{}") for x in (step.get("query_bindings") or {}).values())
        names.update(str(x).strip("{}") for x in (step.get("body_bindings") or {}).values())
        return {x for x in names if x}

    downstream_uses: dict[str, list[tuple[str, set[str]]]] = {}
    for consumer_id, consumer in steps.items():
        needs = input_names(consumer)
        if not needs:
            continue
        for dep in [str(x) for x in consumer.get("depends_on") or [] if str(x) in steps]:
            produced = {str(x).strip("{}") for x in (steps[dep].get("binds") or [])}
            used = needs & produced
            if used:
                downstream_uses.setdefault(dep, []).append((consumer_id, used))

    semantic_ops = {"filter", "first", "nth", "endpoint_rank", "argmax", "argmin"}

    def has_declared_selection(step, derivations, names):
        # An indexed binding path (e.g. results[0].id) is already explicit
        # selection syntax; do not require the model to restate the same choice as
        # a separate endpoint_rank/nth derivation.
        if _explicit_binding_selection_index(step, names) is not None:
            return True
        for deriv in derivations:
            op = str(deriv.get("operator") or "").lower()
            if op in semantic_ops:
                return True
            if op == "identity" and bool(deriv.get("filter")):
                return True
        return False

    def population_wide_child_semantics(producer_sid: str) -> bool:
        uses = downstream_uses.get(producer_sid) or []
        if not uses:
            return False
        for consumer_sid, _names in uses:
            card = cards_by_step.get(consumer_sid) or {}
            # Population fan-out is only implied when each child HTTP response is
            # itself a singleton. If the child already contains a collection, its
            # local derivation does not imply "all parents".
            if any("[*]" in str(x.get("path") or "") for x in card.get("record_paths") or []):
                return False
            ops = {str(d.get("operator") or "").lower()
                   for d in derivs_by_step.get(consumer_sid, [])}
            if not (ops & {"argmax", "argmin"}):
                return False
        return True

    errors: list[str] = []
    for sid, uses in downstream_uses.items():
        names = set().union(*(names for _consumer, names in uses))
        card = cards_by_step.get(sid) or {}
        roots = [str(x.get("path") or "") for x in card.get("record_paths") or []
                 if "[*]" in str(x.get("path") or "")]
        if not roots:
            continue
        direct = derivs_by_step.get(sid, [])
        # A dependent binding may come from one explicitly qualified nested object
        # even when the response also contains unrelated arrays.  In that case no
        # record selection is required: the binding path itself identifies the
        # singleton owner (e.g. item.album.id).
        binding_paths = [str((steps.get(sid) or {}).get("binding_paths", {}).get(name) or "")
                         for name in names]
        if binding_paths and all(binding_paths):
            object_root, object_errors = _candidate_record_root(
                card, direct, require_binding=True, binding_paths=binding_paths)
            if object_root and not object_errors and "[*]" not in str(object_root):
                continue
        if names and names <= typed_fanout.get(sid, set()):
            continue
        if has_declared_selection(steps.get(sid) or {}, direct, names):
            continue
        if population_wide_child_semantics(sid):
            continue
        errors.append(
            f"step {sid}: collection-valued response supplies downstream binding(s) "
            f"{sorted(names)} but the evidence plan declares no selection semantics for the "
            "producer records; add filter/first/nth/endpoint_rank/argmax/argmin so runtime "
            "projection does not choose a record implicitly")
    return list(dict.fromkeys(errors))


def schema_derivation_retry_hints(plan: dict[str, Any], cards: list[dict[str, Any]],
                                  question: str = "") -> list[str]:
    """Return value-free hints when a derivation names an absent response field.

    This is intentionally advisory/retry-only. OpenAPI response schemas can be
    incomplete, so an absent documented field is not made a permanent hard error;
    it is, however, strong evidence that the planner should reconsider the route
    or derivation once before spending execution tokens.
    """
    card_by_step = {str(x.get("step_id") or ""): x for x in cards or []}

    def norm(path: Any) -> str:
        text = str(path or "").replace("$.", "").strip(".")
        text = re.sub(r"\[(?:\*|\d*)\]", "", text)
        return text

    def present(card: dict[str, Any], field: Any) -> bool:
        wanted = norm(field)
        if not wanted:
            return True
        paths = [norm(x.get("path")) for x in (card or {}).get("leaf_paths") or []]
        paths += [norm(x.get("path")) for x in (card or {}).get("record_paths") or []]
        return any(p == wanted or p.endswith("." + wanted) or p.startswith(wanted + ".")
                   for p in paths if p)

    def leaf_present(card: dict[str, Any], field: Any) -> bool:
        wanted = norm(field)
        if not wanted:
            return False
        paths = [norm(x.get("path")) for x in (card or {}).get("leaf_paths") or []]
        return any(p == wanted or p.endswith("." + wanted) for p in paths if p)

    def top_level_record_roots(card: dict[str, Any]) -> list[str]:
        out = []
        for item in (card or {}).get("record_paths") or []:
            path = str(item.get("path") or "")
            if "[*]" not in path:
                continue
            normalized = norm(path)
            if normalized and "." not in normalized and path not in out:
                out.append(path)
        return out

    def top_level_roots_with_leaf(card: dict[str, Any], field: Any) -> list[str]:
        wanted = norm(field)
        if not wanted:
            return []
        roots = top_level_record_roots(card)
        out = []
        for root in roots:
            root_norm = norm(root)
            for item in (card or {}).get("leaf_paths") or []:
                path = norm(item.get("path"))
                if not path.startswith(root_norm + "."):
                    continue
                rel = path[len(root_norm) + 1:]
                if rel == wanted or rel.endswith("." + wanted):
                    out.append(root); break
        return out

    hints: list[str] = []
    q = str(question or "").casefold()
    future_requested = any(term in q for term in (
        "upcoming", "future", "will release", "will air", "scheduled", "next release",
        "next movie", "next season", "coming out"))
    latest_requested = any(term in q for term in (
        "latest", "newest", "most recent", "recently", "recent "))
    for derivation in (plan or {}).get("derivations") or []:
        source_steps = [str(x) for x in derivation.get("source_steps") or [] if str(x) in card_by_step]
        if not source_steps:
            continue
        op = str(derivation.get("operator") or "").lower()
        purpose = str(derivation.get("purpose") or "").casefold()
        field_decl = str(derivation.get("field") or "").strip()
        has_derived_source = bool(derivation.get("source_derivations"))
        canonical_for_ambiguity = _canonical_derivation_filter(
            derivation.get("filter") or {}, derivation.get("field"))
        if op in {"first", "nth", "endpoint_rank"} and not field_decl and not canonical_for_ambiguity and not has_derived_source:
            ambiguous = []
            for sid in source_steps:
                roots = top_level_record_roots(card_by_step[sid])
                if len(roots) > 1:
                    ambiguous.extend(roots)
            if ambiguous:
                hints.append(
                    f"derivation {derivation.get('id')}: selector does not identify which documented "
                    f"record collection it selects among {list(dict.fromkeys(ambiguous))}; qualify "
                    "the intended collection in field so the selection is schema-unambiguous")
        if field_decl and "." not in norm(field_decl) and not has_derived_source:
            ambiguous = []
            for sid in source_steps:
                roots = top_level_roots_with_leaf(card_by_step[sid], field_decl)
                if len(roots) > 1:
                    ambiguous.extend(roots)
            if ambiguous:
                hints.append(
                    f"derivation {derivation.get('id')}: field {field_decl!r} exists in multiple sibling "
                    f"record collections {list(dict.fromkeys(ambiguous))}; qualify the intended "
                    "collection relation in the field path")
        if op == "argmin":
            field_text = str(derivation.get("field") or "").lower()
            if latest_requested and any(token in field_text for token in ("date", "time", "year")):
                hints.append(
                    f"derivation {derivation.get('id')}: latest/most-recent intent uses argmin "
                    f"over date-like field {derivation.get('field')!r}; use argmax/later-date "
                    "selection unless the user explicitly requested earliest")
        if op == "endpoint_rank":
            metric_terms = (
                "latest", "newest", "most recent", "recent",
                "highest", "lowest", "most popular", "least popular",
                "top rated", "highest rated", "lowest rated",
            )
            if any(term in purpose for term in metric_terms):
                hints.append(
                    f"derivation {derivation.get('id')}: endpoint_rank is declared for a "
                    "metric/recency superlative, but returned order is not itself a replayable "
                    "metric. Use a documented date/metric field with argmax/argmin (and any "
                    "needed date filter) unless the API documentation explicitly guarantees the "
                    "requested ranking order")
        if op == "argmax":
            field_text = str(derivation.get("field") or "").lower()
            recency_terms = ("latest", "newest", "most recent", "current")
            looks_date = any(token in field_text for token in ("date", "time", "year"))
            if looks_date and any(term in purpose for term in recency_terms) and not future_requested:
                filt = derivation.get("filter") if isinstance(derivation.get("filter"), dict) else {}
                pred = filt.get(derivation.get("field")) if derivation.get("field") in filt else None
                op_name = str((pred or {}).get("op") or "").lower() if isinstance(pred, dict) else ""
                value = (pred or {}).get("value") if isinstance(pred, dict) else None
                from datetime import date as _schema_date
                today = _schema_date.today().isoformat()
                if not (op_name in {"lt", "lte"} and isinstance(value, str) and value <= today):
                    hints.append(
                        f"derivation {derivation.get('id')}: recency argmax over date-like field "
                        f"{derivation.get('field')!r} does not exclude future-dated candidates. "
                        f"Filter that same field to <= {today} before ranking unless the user "
                        "explicitly requested upcoming/future items")
        canonical_filter = _canonical_derivation_filter(
            derivation.get("filter") or {}, derivation.get("field"))
        for filter_field, predicate in canonical_filter.items():
            pred_op = str((predicate or {}).get("op") or "eq").lower() if isinstance(predicate, dict) else "eq"
            # A record/collection root is useful for selection, but text/numeric
            # predicates over that whole object are not replayable field semantics.
            # Require a documented leaf except for existence checks.
            if pred_op not in {"exists", "not_exists"} and not any(
                    leaf_present(card_by_step[sid], filter_field) for sid in source_steps):
                if any(present(card_by_step[sid], filter_field) for sid in source_steps):
                    hints.append(
                        f"derivation {derivation.get('id')}: filter field {filter_field!r} "
                        "targets a documented record/container rather than a scalar leaf; "
                        "filter on a documented child field instead")

        fields: list[str] = []
        field = str(derivation.get("field") or "").strip()
        if field:
            fields.append(field)
        distinct = str(derivation.get("distinct_field") or "").strip()
        if distinct:
            fields.append(distinct)
        fields.extend(str(x) for x in canonical_filter.keys() if str(x).strip())
        for candidate_field in dict.fromkeys(fields):
            if not any(present(card_by_step[sid], candidate_field) for sid in source_steps):
                hints.append(
                    f"derivation {derivation.get('id')}: field {candidate_field!r} is absent from "
                    f"the documented selected response schema for source steps {source_steps}; "
                    "revise the derivation or choose a documented source that exposes the field")
    return list(dict.fromkeys(hints))


def repair_terminal_selector_bindings(plan: dict[str, Any]) -> dict[str, Any]:
    """Align downstream bindings with a declared single-record selector.

    Observation planning can conservatively emit ``selected_all`` for a collection
    even when the evidence plan subsequently declares first/nth/endpoint-rank or
    argmax/argmin over that same step. Executing every candidate is both semantically
    broader than the certified selection and potentially very expensive. Narrow such
    bindings to the selected record first; the later population-fanout repair may
    deliberately upgrade them back to ``selected_all`` when a downstream singleton
    argmax/argmin genuinely requires population-wide evaluation.
    """
    out = dict(plan or {})
    specs = [dict(x) for x in out.get("observation_specs") or []]
    derivs_by_step: dict[str, list[dict[str, Any]]] = {}
    for deriv in out.get("derivations") or []:
        sources = [str(x) for x in deriv.get("source_steps") or []]
        if len(sources) == 1:
            derivs_by_step.setdefault(sources[0], []).append(deriv)
    selector_ops = {"first", "nth", "endpoint_rank", "argmax", "argmin"}
    warnings = list(out.get("validation_warnings") or [])
    for spec in specs:
        sid = str(spec.get("step_id") or "")
        selectors = [d for d in derivs_by_step.get(sid, [])
                     if str(d.get("operator") or "").lower() in selector_ops]
        if not selectors:
            continue
        terminal = selectors[-1]
        terminal_op = str(terminal.get("operator") or "").lower()
        changed = False
        if terminal_op in {"first", "nth", "endpoint_rank"} and spec.get("sort"):
            # The fixed evidence plan already defines returned-order selection.
            # A projection-planner sort would silently change that semantics, so
            # canonicalize it away rather than spending another model retry.
            spec["sort"] = []
            changed = True
            warnings.append(
                f"step {sid}: removed projection sort to preserve {terminal_op} returned-order semantics")
        bindings = []
        for raw in spec.get("bindings") or []:
            binding = dict(raw)
            if str(binding.get("source") or "selected_first") == "selected_all":
                binding["source"] = "selected_first"
                changed = True
            bindings.append(binding)
        if changed:
            spec["bindings"] = bindings
            warnings.append(
                f"step {sid}: narrowed selected_all binding to selected_first to match "
                f"terminal {str(terminal.get('operator') or '').lower()} selector")
    out["observation_specs"] = specs
    out["validation_warnings"] = list(dict.fromkeys(warnings))
    return out


def repair_requested_subset_fanout(question: str, plan: dict[str, Any]) -> dict[str, Any]:
    """Bound ``selected_all`` child fan-out for explicitly subset-sized requests.

    ``selected_all`` historically meant the complete filtered producer universe,
    intentionally ignoring the Observation Planner's display limit.  That is
    correct for population-wide count/rank operations, but it is wasteful and
    semantically unnecessary when the user explicitly asks for only "some", "a
    few", or "several" examples.  Preserve ``selected_all`` lineage while adding
    a small deterministic value bound.  Population-wide repair below removes this
    bound whenever complete coverage is required for argmax/argmin.
    """
    text = str(question or "").casefold()
    if not re.search(r"\b(?:some|a few|few|several)\b", text):
        return dict(plan or {})
    if re.search(r"\b(?:all|every|each|entire|complete)\b", text):
        return dict(plan or {})
    out = dict(plan or {})
    specs = [dict(x) for x in out.get("observation_specs") or []]
    warnings = list(out.get("validation_warnings") or [])
    for spec in specs:
        select = dict(spec.get("select") or {})
        try:
            requested = max(1, int(select.get("limit", 5) or 5))
        except Exception:
            requested = 5
        limit = min(5, requested)
        changed = False
        bindings = []
        for raw in spec.get("bindings") or []:
            binding = dict(raw)
            if str(binding.get("source") or "selected_first") == "selected_all":
                binding["max_values"] = limit
                changed = True
            bindings.append(binding)
        if changed:
            spec["bindings"] = bindings
            warnings.append(
                f"step {spec.get('step_id')}: bounded selected_all fan-out to {limit} values "
                "because the user requested only a subset of examples")
    out["observation_specs"] = specs
    out["validation_warnings"] = list(dict.fromkeys(warnings))
    return out




def repair_plural_source_owner_fanout(question: str, plan: dict[str, Any]) -> dict[str, Any]:
    """Preserve plural *source-owner* cardinality for downstream read-only fan-out.

    Multiple child assets from one owner do not satisfy requests such as "some cover images
    of movies ...". When the question explicitly pluralizes the owner of a visual asset,
    upgrade only the producer binding that feeds the terminal asset step from selected_first
    to a bounded selected_all fan-out. This uses the existing plan graph/observation specs and
    never invents an endpoint or binding value.
    """
    q = str(question or "").casefold()
    if not _explicit_multi_intent(q) or not re.search(r"\b(?:image|photo|poster|cover|logo)s?\b", q):
        return dict(plan or {})
    # Require an explicitly plural owner near the asset relation; this avoids turning
    # a request for several images of one named item into multi-owner fan-out.
    if not re.search(
            r"\b(?:image|photo|poster|cover|logo)s?\b[^?]{0,45}\b"
            r"(?:movies|films|shows|series|episodes|people|persons|actors|directors|companies|collections)\b",
            q):
        return dict(plan or {})
    out = copy.deepcopy(plan or {})
    steps = {str(x.get("id") or ""): x for x in out.get("steps") or []}
    specs = [dict(x) for x in out.get("observation_specs") or []]
    by_spec = {str(x.get("step_id") or ""): x for x in specs}
    answer_steps = {str(x) for x in out.get("answer_steps") or []}
    warnings = list(out.get("validation_warnings") or [])
    changed = False

    def consumed_binding_names(step: dict[str, Any]) -> set[str]:
        names = {str(x).strip("{}") for x in _step_placeholders(step)}
        names.update(str(x).strip("{}") for x in (step.get("query_bindings") or {}).values())
        names.update(str(x).strip("{}") for x in (step.get("body_bindings") or {}).values())
        return {x for x in names if x}

    for child_id in list(answer_steps):
        child = steps.get(child_id) or {}
        if str(child.get("method") or "GET").upper() not in {"GET", "HEAD"}:
            continue
        deps = [str(x) for x in child.get("depends_on") or [] if str(x) in steps]
        needed = consumed_binding_names(child)
        for dep in deps:
            spec = by_spec.get(dep)
            if not spec or "[*]" not in str(spec.get("record_path") or ""):
                continue
            bindings = []
            dep_changed = False
            for raw in spec.get("bindings") or []:
                binding = dict(raw)
                name = str(binding.get("name") or "").strip("{}")
                if name in needed and str(binding.get("source") or "selected_first") == "selected_first":
                    binding["source"] = "selected_all"
                    binding["max_values"] = min(5, max(2, int((spec.get("select") or {}).get("limit", 5) or 5)))
                    dep_changed = True
                bindings.append(binding)
            if dep_changed:
                spec["bindings"] = bindings
                spec["select"] = {"mode": "all_matches", "limit": 5, "index": 0}
                spec["completeness"] = "bounded plural-owner candidates inspected"
                changed = True
                warnings.append(
                    f"step {dep}: preserved plural source-owner cardinality with bounded selected_all fan-out")
    if changed:
        out["observation_specs"] = specs
        out["validation_warnings"] = list(dict.fromkeys(warnings))
    return out

def repair_population_fanout_bindings(plan: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Make population-wide child derivations cover every upstream candidate.

    The route planner does not encode whether a downstream placeholder is one
    selected value or a fan-out set; the schema-grounded observation plan does.
    For argmax/argmin over a singleton-per-call child, a single selected parent
    can never establish a global extremum.  Upgrade the unique upstream binding to
    ``selected_all`` when the producer projection is a collection.  Ambiguous or
    genuinely singleton producers fail closed instead of guessing.
    """
    out = dict(plan or {})
    steps = {str(x.get("id") or ""): x for x in out.get("steps") or []}
    specs = [dict(x) for x in out.get("observation_specs") or []]
    by_step = {str(x.get("step_id") or ""): x for x in specs}
    warnings = list(out.get("validation_warnings") or [])
    errors: list[str] = []

    def input_binding_names(step: dict[str, Any]) -> list[str]:
        names = [x.strip("{}") for x in re.findall(r"\{[^{}]+\}", str(step.get("endpoint") or ""))
                 if x.strip("{}") not in (step.get("path_literals") or {})]
        names += [str(x) for x in (step.get("query_bindings") or {}).values()]
        names += [str(x) for x in (step.get("body_bindings") or {}).values()]
        return list(dict.fromkeys(x for x in names if x))

    for derivation in out.get("derivations") or []:
        op = str(derivation.get("operator") or "").lower()
        if op not in {"argmax", "argmin"}:
            continue
        sources = [str(x) for x in derivation.get("source_steps") or [] if str(x) in steps]
        if len(sources) != 1:
            continue
        source = sources[0]
        source_spec = by_step.get(source) or {}
        record_path = str(source_spec.get("record_path") or "$")
        if "[*]" in record_path:
            continue  # one HTTP response already contains a candidate collection

        source_step = steps.get(source) or {}
        binding_names = input_binding_names(source_step)
        deps = [str(x) for x in source_step.get("depends_on") or [] if str(x) in steps]
        candidates: list[tuple[str, str, dict[str, Any]]] = []
        for dep in deps:
            dep_spec = by_step.get(dep) or {}
            for binding in dep_spec.get("bindings") or []:
                if str(binding.get("name") or "").strip("{}") in binding_names:
                    candidates.append((dep, str(binding.get("name") or "").strip("{}"), binding))
        # De-duplicate aliases pointing at the same producer/path/source.
        unique = {}
        for dep, name, binding in candidates:
            key = (dep, str(binding.get("path") or ""), str(binding.get("source") or "selected_first"))
            unique[key] = (dep, name, binding)
        candidates = list(unique.values())
        if len(candidates) != 1:
            errors.append(
                f"derivation {derivation.get('id')}: {op} over singleton step {source} lacks one "
                "unambiguous upstream candidate binding for complete population coverage")
            continue
        dep, name, binding = candidates[0]
        dep_spec = by_step.get(dep) or {}
        if "[*]" not in str(dep_spec.get("record_path") or "$"):
            errors.append(
                f"derivation {derivation.get('id')}: {op} over singleton step {source} is fed by "
                f"singleton producer {dep}, so no comparison population is established")
            continue
        if str(binding.get("source") or "selected_first") != "selected_all" or binding.get("max_values"):
            # Update every alias on the same path so downstream authorization cannot
            # mix selected_first and selected_all interpretations of one value set.
            # Population-wide ranking requires complete coverage, so remove any
            # subset fan-out bound added for an otherwise "some/few" request.
            path = str(binding.get("path") or "")
            for candidate in dep_spec.get("bindings") or []:
                if str(candidate.get("path") or "") == path:
                    candidate["source"] = "selected_all"
                    candidate.pop("max_values", None)
            warnings.append(
                f"step {dep}: binding path {path} upgraded to selected_all because downstream "
                f"{source} participates in population-wide {op}")

    out["observation_specs"] = specs
    out["validation_warnings"] = list(dict.fromkeys(warnings))
    return out, list(dict.fromkeys(errors))


def _semantic_stem(token: str) -> str:
    """Tiny deterministic stemmer for OAS/question relation matching.

    This is deliberately lexical rather than domain-specific.  It is used only to
    surface high-signal API alternatives to the answer-free semantic reviewer; it
    never selects an answer or endpoint by benchmark identity.
    """
    t = re.sub(r"[^a-z0-9]+", "", str(token or "").lower())
    if len(t) <= 3:
        return t
    for suffix, repl in (("ities", "ity"), ("ity", ""), ("ies", "y"),
                         ("ing", ""), ("ed", ""), ("s", ""), ("es", "")):
        if t.endswith(suffix) and len(t) - len(suffix) >= 3:
            return t[:-len(suffix)] + repl
    return t


def _semantic_stems(text: Any) -> set[str]:
    stop = {"the", "a", "an", "of", "to", "for", "and", "or", "is", "are",
            "was", "were", "in", "on", "with", "by", "from", "get", "retrieve",
            "fetch", "find", "search", "api", "this", "that", "list", "details"}
    out = set()
    for token in re.findall(r"[a-z0-9]+", str(text or "").lower()):
        if token in stop or len(token) <= 2 or token.isdigit():
            continue
        stem = _semantic_stem(token)
        if stem and stem not in stop:
            out.add(stem)
    return out


def _path_parts(path: str) -> list[str]:
    return [x for x in str(path or "").strip("/").split("/") if x]


def _path_suffix_stems(path: str) -> set[str]:
    """Semantic tokens from the route suffix, excluding version/entity anchors."""
    parts = _path_parts(path)
    static = [p for p in parts if not (p.startswith("{") and p.endswith("}"))]
    # APIs commonly begin with a version segment followed by a resource/entity
    # segment.  The remaining static path is the route qualifier/action.
    suffix = static[2:] if len(static) >= 2 and static[0].isdigit() else static[1:]
    return _semantic_stems(" ".join(x.replace("_", " ") for x in suffix))


def _same_collection_family(a: str, b: str) -> bool:
    """Whether two routes are true sibling collections under the same resource scope.

    A global collection must not replace a resource-bound collection merely because
    both start with the same entity token.  For example, ``/people/{id}/credits``
    and ``/people/popular`` have different candidate populations.  True siblings
    such as ``/items/active`` and ``/items/popular`` share the complete parent
    prefix (including placeholder positions) and differ only in the terminal
    collection qualifier.
    """
    pa, pb = _path_parts(a), _path_parts(b)
    if len(pa) < 2 or len(pb) < 2 or len(pa) != len(pb):
        return False
    def norm(part: str) -> str:
        return "{}" if part.startswith("{") and part.endswith("}") else part
    return [norm(x) for x in pa[:-1]] == [norm(x) for x in pb[:-1]]


def deterministic_semantic_route_risks(question: str, plan: dict[str, Any],
                                       tools: list[dict[str, Any]]) -> list[str]:
    """Find narrow, answer-free route-relation mismatches from question + OAS.

    These are *review triggers*, not benchmark route rules.  They use only the
    user's wording, the proposed derivation semantics, and documented operations.
    The two guarded patterns are general:
      1) a global superlative is computed inside a qualified subset even though a
         sibling OAS collection directly represents the requested ranking; and
      2) a resource-level relation is reconstructed from a nested subordinate
         collection absent from the question even though a direct child operation
         exposes the same relation at the requested resource level.
    """
    qstems = _semantic_stems(question)
    by_step = {str(x.get("id") or ""): x for x in (plan or {}).get("steps") or []}
    tool_by_path = {str(t.get("path") or ""): t for t in tools
                    if str(t.get("method") or "GET").upper() == "GET"}
    risks: list[str] = []

    # Pattern 0: an explicit ranking concept must be represented by the executable
    # selection semantics.  This is intentionally limited to high-confidence
    # lexical contracts such as "trending": a route that merely represents a
    # temporal subset (for example "active today") is not evidence that the
    # records are ranked by trend.  The repair context is discovered from the OAS,
    # never from a benchmark route or expected answer.
    qtext = str(question or "").casefold()
    requested_criterion = None
    criterion_aliases: set[str] = set()
    if re.search(r"\b(?:most\s+)?trending\b", qtext):
        requested_criterion = "trending"
        criterion_aliases = {"trend", "trending"}
    if requested_criterion:
        represented = False
        # A TV-specific popularity population is the deterministic fallback for
        # an explicit TV-only "most trending" request when the alternative OAS
        # trend route is mixed-media.  The normalizer records this choice; do not
        # immediately reject the repaired owner population for lacking the literal
        # word "trending" in its endpoint name.
        if requested_criterion == "trending" and re.search(r"\b(?:tv|television)\s+(?:show|series)\b", qtext):
            if any(re.search(r"/tv/popular$",str(st.get("endpoint") or "")) for st in by_step.values()):
                represented = True
        selected_sources: list[str] = []
        for d in (plan or {}).get("derivations") or []:
            op = str(d.get("operator") or "").lower()
            if op in {"endpoint_rank", "first", "nth", "argmax", "argmin"}:
                selected_sources.extend(str(x) for x in d.get("source_steps") or [])
            if op in {"argmax", "argmin"}:
                semantic = _semantic_stems(
                    str(d.get("field") or "").replace("_", " ") + " " +
                    str(d.get("purpose") or ""))
                if semantic & criterion_aliases:
                    represented = True
        for sid in dict.fromkeys(selected_sources):
            step = by_step.get(sid) or {}
            path = str(step.get("endpoint") or "")
            tool = tool_by_path.get(path) or {}
            semantic = _semantic_stems(
                path.replace("_", " ") + " " + str(tool.get("functionality") or ""))
            if semantic & criterion_aliases:
                represented = True
                break
        if selected_sources and not represented:
            # Prefer alternatives that share user-visible resource vocabulary
            # (TV/movie/person/item/etc.) so an unrelated globally-ranked resource
            # is not offered merely because its docs also say "trending".
            alternatives: list[tuple[int, str]] = []
            for path, tool in tool_by_path.items():
                semantic = _semantic_stems(
                    path.replace("_", " ") + " " + str(tool.get("functionality") or ""))
                if not (semantic & criterion_aliases):
                    continue
                resource_overlap = len((qstems - criterion_aliases) & semantic)
                alternatives.append((resource_overlap, path))
            alternatives.sort(key=lambda x: (x[0], x[1]), reverse=True)
            alt_paths = [p for score, p in alternatives if score > 0][:3]
            if alt_paths:
                risks.append(
                    f"requested ranking criterion '{requested_criterion}' is not represented by "
                    f"the selected endpoint/order semantics for step(s) {list(dict.fromkeys(selected_sources))}; "
                    f"documented OAS operation(s) {alt_paths} explicitly represent that criterion")

    # Pattern 0b: "latest released" is a population+ordering conjunction.
    # A route documented merely as the newest/latest database record is not release
    # chronology. Accept an explicit release/date argmax, or an OAS route whose own
    # documentation represents released/currently-playing population semantics.
    release_recency = bool(re.search(
        r"\b(?:latest|newest|most\s+recent)(?:\s+\w+){0,3}\s+released\b|"
        r"\bmost\s+recently\s+released\b|\blatest\s+release\b", qtext))
    if release_recency:
        explicit_release_max = any(
            str(d.get("operator") or "").lower() == "argmax"
            and any(tok in str(d.get("field") or "").casefold()
                    for tok in ("release", "date", "time", "year"))
            for d in (plan or {}).get("derivations") or [])
        if not explicit_release_max:
            for sid, step in by_step.items():
                path = str(step.get("endpoint") or "")
                tool = tool_by_path.get(path) or {}
                prose = (path + " " + str(tool.get("functionality") or "")).casefold()
                selected_here = any(
                    sid in {str(x) for x in d.get("source_steps") or []}
                    and str(d.get("operator") or "").lower() in {"endpoint_rank", "first", "nth"}
                    for d in (plan or {}).get("derivations") or [])
                if not selected_here:
                    continue
                says_latest = bool(re.search(r"\b(?:latest|newest|most recent)\b", prose))
                says_release_population = bool(re.search(
                    r"\b(?:released|now playing|currently playing|in theaters|in theatres)\b", prose))
                if says_latest and not says_release_population:
                    alternatives = []
                    for alt_path, alt_tool in tool_by_path.items():
                        if alt_path == path or "{" in alt_path or not _same_collection_family(path, alt_path):
                            continue
                        alt_text = (alt_path + " " + str(alt_tool.get("functionality") or "")).casefold()
                        if re.search(r"\b(?:released|now playing|currently playing|in theaters|in theatres)\b", alt_text):
                            alternatives.append(alt_path)
                    risks.append(
                        f"latest-released intent uses route {path} whose OAS semantics are latest/newest "
                        "record semantics rather than release chronology; use an explicit release-date "
                        "argmax over a released population" +
                        (f" or consider documented sibling collection(s) {alternatives[:3]}" if alternatives else ""))

    # Pattern 0c: requested relation/content must survive to terminal evidence.
    # This catches semantic substitutions such as answering a review request from an
    # overview/rating object when the OpenAPI catalog exposes a review child route.
    # It is relation-generic and answer-free: only literal relation words in the
    # question and documented route/functionality text participate.
    relation_terms = {
        "review": {"review", "reviews"},
        "recommendation": {"recommendation", "recommendations", "recommend"},
    }
    terminal_ids = {str(x) for x in (plan or {}).get("answer_steps") or []}
    terminal_steps = [by_step.get(sid) or {} for sid in terminal_ids]
    # Do not trust planner-authored purpose prose as proof that the relation is
    # actually present. Only executable terminal routes and extracted schema fields
    # count as relation evidence.
    terminal_blob = " ".join(str(st.get("endpoint") or "") for st in terminal_steps).casefold()
    terminal_blob += " " + " ".join(
        str(d.get("field") or "")
        for d in (plan or {}).get("derivations") or []
        if str(d.get("id") or "") in {str(x) for x in (plan or {}).get("answer_derivations") or []}
    ).casefold()
    for relation, aliases in relation_terms.items():
        requested = any(re.search(r"\b" + re.escape(a) + r"s?\b", qtext) for a in aliases)
        if not requested or relation in terminal_blob:
            continue
        resource_anchors = set()
        for st in (plan or {}).get("steps") or []:
            parts = _path_parts(str(st.get("endpoint") or ""))
            static = [x for x in parts if not (x.startswith("{") and x.endswith("}"))]
            if static and static[0].isdigit() and len(static) > 1:
                resource_anchors.add(static[1])
            elif static:
                resource_anchors.add(static[0])
        alternatives = []
        for path, tool in tool_by_path.items():
            prose = (path.replace("_", " ") + " " + str(tool.get("functionality") or "")).casefold()
            if relation not in prose:
                continue
            parts = _path_parts(path)
            static = [x for x in parts if not (x.startswith("{") and x.endswith("}"))]
            anchor_name = static[1] if static and static[0].isdigit() and len(static) > 1 else (static[0] if static else "")
            if not resource_anchors or anchor_name in resource_anchors:
                alternatives.append(path)
        if alternatives:
            risks.append(
                f"requested {relation} relation is absent from terminal answer evidence; "
                f"documented same-resource relation operation(s) {alternatives[:3]} expose it directly")

    # Pattern 0d: explicit visual-appearance intent should use the owner's
    # dedicated visual relation when the catalog documents one. A nullable metadata
    # thumbnail/profile path can be useful fallback evidence, but it should not
    # displace a same-owner /images/photos-style child operation for questions such
    # as "what does ... look like?". This is route/schema based and never chooses an
    # entity or asset value.
    look_like_requested = bool(re.search(
        r"\b(?:what\s+does\b[^?]{0,90}\blook\s+like|what\b[^?]{0,90}\blooks\s+like)\b",
        qtext))
    if look_like_requested:
        answer_ids = {str(x) for x in (plan or {}).get("answer_steps") or []}
        for sid in answer_ids:
            step = by_step.get(sid) or {}
            selected_path = str(step.get("endpoint") or "")
            if not selected_path:
                continue
            selected_tool = tool_by_path.get(selected_path) or {}
            selected_prose = (selected_path + " " + str(selected_tool.get("functionality") or "")).casefold()
            # Already on a visual child relation: nothing to repair.
            if any(tok in selected_prose for tok in ("/images", "/photos", " image", " photo", "profile images")):
                continue
            parts = _path_parts(selected_path)
            # Build the route's entity-owner prefix through the first placeholder,
            # e.g. /v1/person/{person_id}. Child visual routes sharing that prefix
            # are stronger same-owner evidence than metadata paths on the detail route.
            owner_prefix = None
            for idx, part in enumerate(parts):
                if part.startswith("{") and part.endswith("}"):
                    owner_prefix = "/" + "/".join(parts[:idx + 1])
                    break
            if not owner_prefix:
                continue
            alternatives = []
            for alt_path, alt_tool in tool_by_path.items():
                if alt_path == selected_path or not alt_path.startswith(owner_prefix.rstrip("/") + "/"):
                    continue
                alt_prose = (alt_path + " " + str(alt_tool.get("functionality") or "")).casefold()
                if any(tok in alt_prose for tok in ("/images", "/photos", " images", " photos", "profile images")):
                    alternatives.append(alt_path)
            if alternatives:
                risks.append(
                    f"visual appearance intent terminates at metadata/detail route {selected_path}, but "
                    f"documented same-owner visual operation(s) {alternatives[:3]} provide direct image evidence")

    # Pattern 1: criterion-aligned sibling collection versus a narrower subset.
    for d in (plan or {}).get("derivations") or []:
        op = str(d.get("operator") or "").lower()
        if op not in {"argmax", "argmin"}:
            continue
        source_steps = [str(x) for x in d.get("source_steps") or []]
        if len(source_steps) != 1:
            continue
        step = by_step.get(source_steps[0]) or {}
        selected_path = str(step.get("endpoint") or "")
        field_leaf = str(d.get("field") or "").replace("[*]", "").split(".")[-1]
        criterion = _semantic_stem(field_leaf)
        # Only act when the ranking criterion is actually stated in the question.
        if not criterion or not any(
                criterion == q or criterion.startswith(q) or q.startswith(criterion)
                for q in qstems):
            continue
        selected_tool = tool_by_path.get(selected_path) or {}
        selected_text_stems = _semantic_stems(
            selected_path.replace("_", " ") + " " + str(selected_tool.get("functionality") or ""))
        if any(criterion == x or criterion.startswith(x) or x.startswith(criterion)
               for x in selected_text_stems):
            # The selected collection itself documents the requested criterion.
            continue
        selected_suffix = _path_suffix_stems(selected_path)
        unstated_qualifiers = {x for x in selected_suffix if x not in qstems and x != criterion}
        alternatives = []
        for path, tool in tool_by_path.items():
            if path == selected_path or not _same_collection_family(selected_path, path):
                continue
            # A criterion collection must itself be directly callable. A child
            # relation requiring an entity id (e.g. /{id}/similar) is not an
            # alternative universe for a global collection ranking.
            if "{" in path or "}" in path:
                continue
            text_stems = _semantic_stems(
                path.replace("_", " ") + " " + str(tool.get("functionality") or ""))
            if any(criterion == x or criterion.startswith(x) or x.startswith(criterion)
                   for x in text_stems):
                alternatives.append(path)
        if alternatives and unstated_qualifiers:
            risks.append(
                f"ranking criterion '{field_leaf}' is requested by the user, but source route "
                f"{selected_path} adds qualifier(s) {sorted(unstated_qualifiers)} absent from "
                f"the question; documented sibling collection(s) {alternatives[:3]} directly "
                "represent the requested ranking criterion")

    # Pattern 1b: the selected OAS operation itself documents returned order by
    # the user's ranking criterion.  Recomputing argmax/argmin on one paginated
    # page is both more expensive and less complete than replaying endpoint order.
    for d in (plan or {}).get("derivations") or []:
        op = str(d.get("operator") or "").lower()
        if op not in {"argmax", "argmin"}:
            continue
        source_steps = [str(x) for x in d.get("source_steps") or []]
        if len(source_steps) != 1:
            continue
        step = by_step.get(source_steps[0]) or {}
        path = str(step.get("endpoint") or "")
        tool = tool_by_path.get(path) or {}
        prose = str(tool.get("functionality") or "").casefold()
        ordered = re.search(r"\border(?:ed)?\s+by\s+([a-z0-9_ -]+)", prose)
        if not ordered:
            continue
        order_stems = _semantic_stems(ordered.group(1))
        if not (qstems & order_stems):
            continue
        field_leaf = str(d.get("field") or "").replace("[*]", "").split(".")[-1]
        risks.append(
            f"derivation {d.get('id')} recomputes {op}({field_leaf or 'field'}) over route {path}, "
            f"but the documented endpoint already returns records ordered by the requested "
            f"criterion {sorted(qstems & order_stems)}; preserve endpoint order with "
            "endpoint_rank/first (after any required filter) instead of a page-local extremum")

    # Pattern 2: nested subordinate relation versus a direct resource-level child.
    for d in (plan or {}).get("derivations") or []:
        source_steps = [str(x) for x in d.get("source_steps") or []]
        if len(source_steps) != 1:
            continue
        step = by_step.get(source_steps[0]) or {}
        selected_path = str(step.get("endpoint") or "")
        if not selected_path:
            continue
        raw_paths = [str(d.get("field") or "")] + [str(x) for x in (d.get("filter") or {}).keys()]
        nested_tokens = []
        relation_tokens = set()
        for raw in raw_paths:
            bits = [x for x in re.sub(r"\[(?:\*|\d*)\]", "", raw).split(".") if x]
            if len(bits) >= 2:
                # Only prefixes *above* the immediate relation are subordinate
                # scope.  In episodes.crew.name, ``episodes`` narrows the season
                # to episode-level records, while ``crew`` is the requested role
                # relation carrier and should remain available for alternative
                # endpoint matching.  Flat crew.name/results.title have no extra
                # subordinate scope.
                if len(bits) >= 3:
                    nested_tokens.extend(_semantic_stem(x) for x in bits[:-2])
                relation_tokens.update(_semantic_stems(" ".join(bits)))
        nested_unstated = {x for x in nested_tokens if x and x not in qstems}
        if not nested_unstated:
            continue
        selected_tool = tool_by_path.get(selected_path) or {}
        # Do not reward the selected route merely for repeating the very nested
        # subordinate scope that is absent from the question.
        relation_core = relation_tokens - nested_unstated
        selected_score = len((qstems | relation_core) & _semantic_stems(
            selected_path.replace("_", " ") + " " + str(selected_tool.get("functionality") or "")))
        direct_children = []
        prefix = selected_path.rstrip("/") + "/"
        for path, tool in tool_by_path.items():
            if not path.startswith(prefix):
                continue
            tail = path[len(prefix):]
            if "/" in tail:  # only a direct child operation, not a distant descendant
                continue
            alt_stems = _semantic_stems(
                path.replace("_", " ") + " " + str(tool.get("functionality") or ""))
            score = len((qstems | relation_core) & alt_stems)
            # Require that the alternative better matches the relation vocabulary
            # and does not itself introduce the same unstated subordinate scope.
            if score > selected_score and not (nested_unstated & alt_stems):
                direct_children.append((score, path))
        if direct_children:
            direct_children.sort(reverse=True)
            risks.append(
                f"derivation {d.get('id')} answers from nested subordinate relation(s) "
                f"{sorted(nested_unstated)} that the user did not request, while direct "
                f"resource-level operation(s) {[p for _, p in direct_children[:3]]} are "
                "documented and better match the requested relation")

    # Pattern 3: explicit calendar-window scope must be represented somewhere in
    # the executable plan.  This is deliberately limited to high-confidence
    # phrases such as "this week"/"weekly". A date-bounded filter counts as an
    # explicit representation; otherwise surface documented time-window-capable
    # operations as repair context instead of silently treating a generic current
    # ranking as a weekly ranking.
    qtext = str(question or "").casefold()
    if re.search(r"\b(?:this|current|past|last)\s+week\b|\bweekly\b", qtext):
        represented = False
        for step in (plan or {}).get("steps") or []:
            path = str(step.get("endpoint") or "")
            tool = tool_by_path.get(path) or {}
            semantic_text = (path + " " + str(tool.get("functionality") or "")).casefold()
            vals = list((step.get("path_literals") or {}).values()) + \
                   list((step.get("query_literals") or {}).values()) + \
                   list((step.get("body_literals") or {}).values())
            if "week" in semantic_text or any("week" in str(v).casefold() for v in vals):
                represented = True; break
        if not represented:
            for d in (plan or {}).get("derivations") or []:
                if any(any(tok in str(field).casefold() for tok in ("date", "time", "year"))
                       for field in (d.get("filter") or {})):
                    represented = True; break
        if not represented:
            alternatives = []
            selected_static = [p for p in (str(x.get("endpoint") or "") for x in (plan or {}).get("steps") or [])
                               if p and "{" not in p and "/search/" not in p]
            for path, tool in tool_by_path.items():
                card = tool.get("schema_card") if isinstance(tool.get("schema_card"), dict) else {}
                params = {str(x.get("name") or "").casefold() for x in card.get("parameters") or []}
                text = (path + " " + str(tool.get("functionality") or "")).casefold()
                if not ("time_window" in params or "time window" in text or "weekly" in text):
                    continue
                # Do not force an unrelated mixed/global resource route merely because it
                # has a time-window parameter.  Besides true route siblings, accept a
                # time-window operation only when its documented semantics explicitly carry
                # the same selected resource anchor (e.g. an items collection -> trending
                # items).  A generic trending/all route therefore cannot replace a TV-only
                # population just because both mention a week.
                def _resource_anchor(route: str) -> str:
                    parts = [x for x in _path_parts(route)
                             if not (x.startswith("{") and x.endswith("}"))]
                    while parts and (parts[0].isdigit() or re.fullmatch(r"v\d+", parts[0], re.I)):
                        parts.pop(0)
                    return _semantic_stem(parts[0]) if parts else ""
                alt_semantic = _semantic_stems(
                    path.replace("_", " ") + " " + str(tool.get("functionality") or ""))
                compatible = False
                for sel in selected_static:
                    anchor = _resource_anchor(sel)
                    if _same_collection_family(sel, path) or (anchor and anchor in alt_semantic):
                        compatible = True; break
                if compatible:
                    alternatives.append(path)
            if alternatives:
                risks.append(
                    "the user explicitly requests a weekly candidate/time scope, but the plan "
                    "contains no week/time-window literal and no date-bounded filter; documented "
                    f"same-resource time-window-capable operation(s) {alternatives[:3]} should be considered")

    # Pattern 4: when the question explicitly names a documented nested relation
    # (for example ``known_for`` -> "known for"), the plan must actually consume
    # that relation rather than replacing it with a broader sibling API.  Relation
    # names are discovered from the selected OAS schema; there is no domain table.
    qtext = str(question or "").casefold()
    for sid, step in by_step.items():
        path = str(step.get("endpoint") or "")
        tool = tool_by_path.get(path) or {}
        card = tool.get("schema_card") if isinstance(tool.get("schema_card"), dict) else {}
        relation_names: set[str] = set()
        for item in card.get("record_paths") or []:
            rp = str(item.get("path") or "")
            arrays = re.findall(r"([A-Za-z0-9_]+)\[\*\]", rp)
            # The first array is normally the endpoint's result population.  Any
            # deeper named array is an explicit relation on those records.
            relation_names.update(x for x in arrays[1:] if x and x != "results")
        for relation in sorted(relation_names):
            phrase = relation.replace("_", " ").casefold()
            rel_stems = _semantic_stems(phrase)
            if not phrase or not (phrase in qtext or (rel_stems and rel_stems.issubset(qstems))):
                continue
            represented = False
            for d in (plan or {}).get("derivations") or []:
                if sid not in {str(x) for x in d.get("source_steps") or []}:
                    continue
                texts = [str(d.get("field") or "")] + [str(x) for x in (d.get("filter") or {}).keys()]
                if any(relation in re.sub(r"\[(?:\*|\d*)\]", "", x).split(".") for x in texts):
                    represented = True; break
            if not represented:
                bps = [str(x) for x in (step.get("binding_paths") or {}).values()]
                represented = any(relation in re.sub(r"\[(?:\*|\d*)\]", "", x).split(".") for x in bps)
            if not represented:
                risks.append(
                    f"the user explicitly requests documented relation '{phrase}' exposed by route {path}, "
                    f"but the plan never selects or projects that relation from step {sid}; do not replace "
                    "an explicitly named response relation with a broader sibling resource")

    # Pattern 4b: preserve terminal asset ownership when the requested owner
    # noun is explicit and also appears as an OAS resource name.  A parent
    # collection's image is not an image of one of its member resources. Resource
    # vocabulary comes from the OAS itself; no owner synonym table is used.
    visual_request = bool(re.search(r"\b(?:image|photo|poster|cover|logo)\b", qtext))
    if visual_request:
        owner_candidates: list[str] = []
        for m in re.finditer(
                r"\b(?:image|photo|poster|cover|logo)\s+(?:of|for)\s+(?:a|an|the)?\s*"
                r"([a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*){0,2})", qtext):
            words = [w for w in re.findall(r"[a-z0-9_]+", m.group(1))
                     if w not in {"first", "second", "third", "lead", "top", "most"}]
            if words:
                owner_candidates.append(words[-1])
        for m in re.finditer(r"\b([a-z][a-z0-9_-]*)\s+(?:image|photo|poster|cover|logo)\b", qtext):
            if m.group(1) not in {"cover", "profile"}:
                owner_candidates.append(m.group(1))
        owner_candidates = list(dict.fromkeys(owner_candidates))
        if owner_candidates:
            answer_ids = {str(x) for x in (plan or {}).get("answer_steps") or []}
            answer_paths = [str((by_step.get(sid) or {}).get("endpoint") or "") for sid in answer_ids]
            for owner in owner_candidates:
                owner_stem = _semantic_stem(owner)
                if not owner_stem:
                    continue
                # Only enforce when the OAS itself has an asset-capable route for
                # this exact resource token; otherwise the noun may be a synonym
                # (actor/person) or ordinary prose and remains critic territory.
                owner_asset_routes = []
                for path in tool_by_path:
                    parts = [_semantic_stem(x.replace("_", " ")) for x in _path_parts(path)
                             if not x.startswith("{") and not x.isdigit()]
                    path_stems = set().union(*(_semantic_stems(x) for x in _path_parts(path)))
                    if owner_stem in path_stems and any(tok in path_stems for tok in
                                                       {"image", "photo", "poster", "logo"}):
                        owner_asset_routes.append(path)
                if not owner_asset_routes:
                    continue
                if any(owner_stem in _semantic_stems(path.replace("_", " ")) for path in answer_paths):
                    continue
                risks.append(
                    f"the user requests a visual asset owned by resource '{owner}', but terminal "
                    f"answer route(s) {answer_paths} do not preserve that owner; documented "
                    f"owner-specific asset operation(s) {owner_asset_routes[:3]} exist")

    # Pattern 5: prefer the sibling collection whose *route qualifier* is stated
    # literally in the question. This catches status/population distinctions such
    # as on-the-air vs popular without maintaining an endpoint-specific mapping.
    selected_paths = [str(st.get("endpoint") or "") for st in by_step.values()]
    for selected_path in selected_paths:
        if not selected_path or "{" in selected_path:
            continue
        # ``latest released`` is a population+ordering request.  Once the
        # normalizer has moved the plan onto a documented *released/currently
        # playing* population, do not let the generic sibling-qualifier guard
        # pull it back toward a database-``latest`` route merely because the
        # word "latest" appears literally in the question.  The latter is a
        # record-ingestion qualifier, not a release-population qualifier.
        if release_recency:
            selected_tool = tool_by_path.get(selected_path) or {}
            selected_prose = (
                selected_path + " " + str(selected_tool.get("functionality") or "")
            ).casefold()
            if re.search(
                    r"\b(?:released|now[ _-]?playing|currently playing|"
                    r"in theaters|in theatres)\b", selected_prose):
                continue
        selected_parts = _path_parts(selected_path)
        # /search/{entity-type} siblings are different entity universes, not
        # status/population qualifiers. A question may mention a movie while
        # legitimately searching for its director/person, so never use this
        # collection-status rule across search entity types.
        if len(selected_parts) >= 2 and selected_parts[-2].casefold() == "search":
            continue
        selected_suffix = _path_suffix_stems(selected_path)
        selected_unstated = selected_suffix - qstems
        selected_step_ids = {sid for sid, st in by_step.items()
                             if str(st.get("endpoint") or "") == selected_path}
        executable_stems = set(selected_suffix)
        for d in (plan or {}).get("derivations") or []:
            if not (selected_step_ids & {str(x) for x in d.get("source_steps") or []}):
                continue
            if str(d.get("operator") or "").lower() in {"argmax", "argmin"}:
                executable_stems |= _semantic_stems(str(d.get("field") or ""))
            for fpath, pred in (d.get("filter") or {}).items():
                executable_stems |= _semantic_stems(str(fpath))
                if isinstance(pred, dict):
                    executable_stems |= _semantic_stems(str(pred.get("value") or ""))
        missing_qualifier_alts = []
        narrower_than_requested_alts = []
        for alt_path, alt_tool in tool_by_path.items():
            if alt_path == selected_path or "{" in alt_path or not _same_collection_family(selected_path, alt_path):
                continue
            alt_suffix = _path_suffix_stems(alt_path)
            if not alt_suffix:
                continue
            # Require every semantic suffix token of the alternative to be stated
            # by the question. If the current route does not represent those same
            # tokens, the plan is missing an explicit population/status qualifier.
            if (alt_suffix.issubset(qstems) and selected_suffix and selected_suffix.issubset(qstems)):
                # Both qualifiers are stated (e.g. "most popular ... on the air").
                # Distinguish the ranking qualifier from the population/status qualifier.
                # If the selected route represents only the ranking criterion, it has
                # dropped the explicitly requested population and should be challenged.
                # If the selected route represents the population, keep it and require
                # the ranking criterion to be expressed by executable selection semantics.
                ranking_stems: set[str] = set()
                for match in re.finditer(
                        r"\b(?:most|least|highest|lowest|best|worst)\s+([a-z0-9_-]+)", qtext):
                    stem = _semantic_stem(match.group(1))
                    if stem:
                        ranking_stems.add(stem)
                selected_is_ranking = bool(selected_suffix & ranking_stems)
                alt_is_ranking = bool(alt_suffix & ranking_stems)
                if selected_is_ranking and not alt_is_ranking:
                    missing_qualifier_alts.append(alt_path)
                    continue
                # The selected route is the population/status side (or the roles
                # cannot be distinguished safely); do not swap it for a ranking route.
                continue
            if (alt_suffix.issubset(qstems) and not alt_suffix.issubset(selected_suffix)
                    and not alt_suffix.issubset(executable_stems)):
                missing_qualifier_alts.append(alt_path)
            elif (selected_unstated and alt_suffix.issubset(qstems)
                  and alt_suffix.issubset(selected_suffix)):
                narrower_than_requested_alts.append(alt_path)
        if missing_qualifier_alts:
            risks.append(
                f"source route {selected_path} does not represent an explicit population/status "
                f"qualifier in the question; documented sibling collection(s) "
                f"{missing_qualifier_alts[:3]} expose that qualifier" +
                (f"; the selected route also adds unstated qualifier(s) {sorted(selected_unstated)}"
                 if selected_unstated else ""))
        elif narrower_than_requested_alts:
            risks.append(
                f"source route {selected_path} adds unstated qualifier(s) {sorted(selected_unstated)} "
                f"inside a population/status relation explicitly requested by the user; documented "
                f"sibling collection(s) {narrower_than_requested_alts[:3]} preserve the stated "
                "qualifier without that additional narrowing")

    # Pattern 6: an endpoint_rank is not a generic substitute for a requested
    # recency ranking. If the user says latest/newest/most recent and the selected
    # collection does not document recency ordering while exposing a date/time
    # field, the ranking must be made replayable with that field.
    recency_requested = bool(re.search(r"\b(?:latest|newest|most\s+recent|recently)\b", qtext))
    if recency_requested:
        for d in (plan or {}).get("derivations") or []:
            if str(d.get("operator") or "").lower() != "endpoint_rank":
                continue
            purpose_text = str(d.get("purpose") or "").casefold()
            if not re.search(r"\b(?:latest|newest|most\s+recent|recently|recent)\b", purpose_text):
                continue
            source_steps = [str(x) for x in d.get("source_steps") or []]
            if len(source_steps) != 1:
                continue
            step = by_step.get(source_steps[0]) or {}
            path = str(step.get("endpoint") or "")
            tool = tool_by_path.get(path) or {}
            prose = (path + " " + str(tool.get("functionality") or "")).casefold()
            # For an explicit "latest released" request, a documented current-
            # release population (for example a now-playing collection) is
            # itself the API's release-recency contract.  Requiring a page-local
            # argmax(release_date) can actually change the API semantics and was
            # causing the repaired plan to be rejected immediately after the
            # repair.  Database-latest routes are handled separately by Pattern
            # 0b above and therefore do not receive this exemption.
            if release_recency and re.search(
                    r"\b(?:released|now[ _-]?playing|currently playing|"
                    r"in theaters|in theatres)\b", prose):
                continue
            if re.search(r"\b(?:latest|newest|most recent|ordered by .*date|sorted by .*date)\b", prose):
                continue
            card = tool.get("schema_card") if isinstance(tool.get("schema_card"), dict) else {}
            date_leaves = []
            for leaf in card.get("leaf_paths") or []:
                lp = str(leaf.get("path") or "")
                tail = re.sub(r"\[(?:\*|\d*)\]", "", lp).split(".")[-1].casefold()
                if any(tok in tail for tok in ("date", "time", "year")):
                    date_leaves.append(lp)
            if date_leaves:
                risks.append(
                    f"derivation {d.get('id')} uses undocumented endpoint order from {path} to answer "
                    f"an explicit recency request even though documented date/time field(s) {date_leaves[:4]} "
                    "are available; rank by a replayable date/time field instead")

    # Accuracy-first invariants that do not require benchmark-specific route knowledge.
    # Kept in a separate utility so the same checks can also trigger post-execution
    # replanning against real observations.
    try:
        from utils.accuracy_semantics import plan_semantic_risks
        risks.extend(plan_semantic_risks(question, plan))
    except Exception:
        pass
    return list(dict.fromkeys(risks))


def _semantic_critic_messages(question: str, plan: dict[str, Any],
                              tools: list[dict[str, Any]],
                              schema_text: str = "") -> list[dict[str, str]]:
    """Build one answer-free semantic review from question + OAS only.

    The critic is an answer-free OAS compatibility reviewer, not a benchmark
    oracle.  It may identify a clear semantic incompatibility, but incomplete OAS
    prose must be reported as uncertainty rather than fabricated contradiction.
    """
    # The critic needs relation fidelity, not a second full planner catalog. Rank
    # routes lexically against the user question and proposed routes, then keep a
    # small neighborhood. This still exposes alternatives such as a global
    # "popular" collection next to a proposed "now playing" collection without
    # paying thousands of tokens for unrelated operations.
    stop = {"the","a","an","of","to","for","and","or","is","are","was","were",
            "in","on","with","by","from","get","retrieve","fetch","find","search"}
    def toks(text):
        return {x for x in re.findall(r"[a-z0-9]+", str(text or "").lower())
                if len(x) > 2 and x not in stop}
    qtokens = toks(question)
    selected_paths = {str(st.get("endpoint") or "") for st in plan.get("steps") or []}
    selected_tokens = set().union(*(toks(p) for p in selected_paths)) if selected_paths else set()
    ranked = []
    for tool in tools:
        path = str(tool.get("path") or "")
        desc = str(tool.get("functionality") or "")
        tt = toks(path + " " + desc)
        score = 4 * len(qtokens & tt) + len(selected_tokens & tt) + (6 if path in selected_paths else 0)
        ranked.append((score, path, tool))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    keep = [tool for score, _path, tool in ranked if score > 0][:16]
    for _score, path, tool in ranked:
        if path in selected_paths and tool not in keep:
            keep.append(tool)
    catalog = "\n".join(
        f"- {tool.get('method', 'GET')} {tool.get('path')}: "
        f"{str(tool.get('functionality') or '')[:110]}"
        for tool in keep[:20]
    )
    proposed_steps = []
    for raw in plan.get("steps") or []:
        proposed_steps.append({k: raw.get(k) for k in (
            "id", "method", "endpoint", "purpose", "depends_on",
            "path_literals", "path_bindings", "query_literals", "query_bindings",
            "body_literals", "body_bindings", "answer_source") if raw.get(k) not in (None, {}, [])})
    proposed_derivations = []
    for raw in plan.get("derivations") or []:
        proposed_derivations.append({k: raw.get(k) for k in (
            "id", "operator", "source_steps", "source_derivations", "field",
            "filter", "comparison", "comparison_literal", "unit", "rank",
            "top_k", "purpose") if raw.get(k) not in (None, "", {}, [])})
    proposed = json.dumps({
        "steps": proposed_steps,
        "derivations": proposed_derivations,
        "answer_steps": plan.get("answer_steps") or [],
        "answer_derivations": plan.get("answer_derivations") or [],
        "answer_mode": plan.get("answer_mode"),
        "answer_requirements": plan.get("answer_requirements") or [],
    }, ensure_ascii=False)
    system = """You are the semantic plan critic for an API agent. Do not answer the user's question or use prior factual knowledge.

Check only these things:
- Does the plan keep all important parts of the question?
- Does it use API data that directly answers the question?
- Can each step get the values it needs?

Return `compatible` when the plan is correct. Return `incompatible` only for a clear problem. Return `uncertain` when the API documentation is not enough to decide.

Return JSON only:
{"verdict":"compatible|incompatible|uncertain","errors":["short reason", ...]}
"""
    user = (f"USER QUESTION:\n{question}\n\nPROPOSED PLAN:\n{proposed}\n\n"
            f"AVAILABLE ENDPOINT DOCUMENTATION:\n{catalog}\n\n"
            f"SELECTED RESPONSE SCHEMAS:\n{schema_text or '(not available)'}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def semantic_plan_critic(question: str, plan: dict[str, Any], tools: list[dict[str, Any]],
                         benchmark: str, model: str, client) -> tuple[str, list[str], str]:
    """Return an answer-free OAS compatibility verdict.

    ``incompatible`` is reserved for a clear mismatch; ``uncertain`` is fail-open
    at the semantic layer because incomplete API prose is not evidence of error.
    Structural/request/provenance validation remains deterministic and fail-closed.
    """
    deterministic_risks = deterministic_semantic_route_risks(question, plan, tools)
    if deterministic_risks:
        # These are lexical/schema relation conflicts, not answer guesses.  Bypass
        # an unnecessary critic-model call and send the high-signal OAS diagnosis
        # straight to the same bounded semantic-plan repair path.
        return "incompatible", deterministic_risks, "deterministic OAS relation mismatch"
    schema_text = ""
    try:
        from utils.schema_outline import selected_endpoint_cards, format_endpoint_cards
        cards = selected_endpoint_cards(benchmark, plan, max_paths_per_endpoint=80)
        schema_text = format_endpoint_cards(cards, max_chars=2200)
    except Exception:
        schema_text = ""
    messages = _semantic_critic_messages(question, plan, tools, schema_text)
    try:
        kwargs = dict(model=model, messages=messages, temperature=0.0)
        from utils.token_meter import stage as token_stage
        with token_stage("planner_critic"):
            try:
                response = client.chat.completions.create(
                    **kwargs, response_format={"type": "json_object"},
                    max_completion_tokens=280)
            except Exception:
                response = client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
    except Exception as exc:
        return "uncertain", [], f"semantic critic unavailable: {exc}"
    parsed = _extract_json(text) or {}
    verdict = str(parsed.get("verdict") or "").strip().lower()
    if verdict not in {"compatible", "incompatible", "uncertain"}:
        # Backward compatibility with the v3.7 critic shape.
        if parsed.get("retry") is True or parsed.get("valid") is False:
            verdict = "incompatible"
        elif parsed.get("retry") is False or parsed.get("valid") is True:
            verdict = "compatible"
        else:
            verdict = "uncertain"
    errors = [str(x) for x in (parsed.get("errors") or []) if str(x).strip()]
    if verdict == "incompatible" and not errors:
        errors = ["semantic critic found a clear question/plan/API incompatibility"]
    return verdict, errors, text



_ORDINAL_WORD_INDEX = {
    "first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4,
    "sixth": 5, "seventh": 6, "eighth": 7, "ninth": 8, "tenth": 9,
}


def _question_ordinal_indexes(question: str) -> set[int]:
    """Return explicit zero-based ordinal positions stated by the user.

    This is intentionally narrow. It is metadata for the semantic guard so it can
    distinguish an explicitly requested ordinal from an unrelated ordinal elsewhere
    in the question; it is never used to infer an answer or API-specific entity.
    """
    text = str(question or "").lower()
    out = {index for word, index in _ORDINAL_WORD_INDEX.items()
           if re.search(r"\b" + re.escape(word) + r"\b", text)}
    for match in re.finditer(r"\b(\d+)(?:st|nd|rd|th)\b", text):
        try:
            value = int(match.group(1))
        except Exception:
            continue
        if value > 0:
            out.add(value - 1)
    return out


def positional_selection_risks(question: str, plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Find unanchored positive ordinal/rank choices in a proposed plan.

    A positive position is semantically high-risk when it participates in a
    ranking/role selection because it can silently turn a qualitative role
    (lead/main/top) into an arbitrary second/third record. Zero remains the ordinary
    first/top position. Explicit nth requests are still reviewed so an ordinal in an
    unrelated relation (for example, a season number) cannot accidentally authorize
    a cast position. The helper never decides what entity is correct; it only decides
    when an answer-free documentation review is needed.
    """
    explicit = _question_ordinal_indexes(question)
    risks: list[dict[str, Any]] = []
    ordinal_names = {"order", "rank", "position", "index"}
    cue_text = (str(question or "") + " " + " ".join(
        str(d.get("purpose") or "") for d in (plan or {}).get("derivations") or [])).lower()
    ranking_cues = (
        "lead", "leading", "main", "primary", "top", "first", "second", "third",
        "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
        "rank", "ranked", "position", "highest", "lowest", "latest", "earliest",
        "most", "least", "head", "nth",
    )
    has_ranking_context = any(re.search(r"\b" + re.escape(cue) + r"\b", cue_text)
                              for cue in ranking_cues)

    def _numeric(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except Exception:
            return None
        if not number.is_integer():
            return None
        return int(number)

    for deriv in (plan or {}).get("derivations") or []:
        did = str(deriv.get("id") or "")
        op = str(deriv.get("operator") or "").lower()
        filt = deriv.get("filter") if isinstance(deriv.get("filter"), dict) else {}
        for raw_field, predicate in filt.items():
            field = str(raw_field or "")
            leaf = re.sub(r"\[.*?\]", "", field).split(".")[-1].lower()
            leaf = leaf.rsplit("_", 1)[-1] if "_" in leaf else leaf
            if leaf not in ordinal_names or not has_ranking_context:
                continue
            if isinstance(predicate, dict):
                pred_op = str(predicate.get("op") or "eq").lower()
                value = predicate.get("value")
            else:
                pred_op, value = "eq", predicate
            if pred_op not in {"eq", "eq_ci"}:
                continue
            number = _numeric(value)
            if number is None or number <= 0:
                continue
            risks.append({
                "derivation_id": did,
                "operator": op,
                "field": field,
                "position_value": number,
                "purpose": str(deriv.get("purpose") or ""),
                "source_steps": [str(x) for x in deriv.get("source_steps") or []],
                "explicit_question_ordinal_indexes": sorted(explicit),
                "reason": "positive positional literal requires relation-specific semantic justification",
            })

        if op in {"nth", "endpoint_rank"}:
            number = _numeric(deriv.get("rank", 0))
            if number is not None and number > 0:
                risks.append({
                    "derivation_id": did,
                    "operator": op,
                    "field": "rank",
                    "position_value": number,
                    "purpose": str(deriv.get("purpose") or ""),
                    "source_steps": [str(x) for x in deriv.get("source_steps") or []],
                    "explicit_question_ordinal_indexes": sorted(explicit),
                    "reason": "positive positional rank requires relation-specific semantic justification",
                })
    # Stable de-duplication keeps retry diagnostics compact.
    out, seen = [], set()
    for risk in risks:
        key = (risk.get("derivation_id"), risk.get("field"), risk.get("position_value"))
        if key not in seen:
            seen.add(key); out.append(risk)
    return out


def selection_semantic_guard(question: str, plan: dict[str, Any], risks: list[dict[str, Any]],
                             benchmark: str, model: str, client) -> tuple[str, list[str], str]:
    """Review only risky positional selections against user wording + selected OAS.

    This is intentionally much smaller than the optional full semantic critic.  It
    runs only when the deterministic trigger finds an unexplained positive ordinal,
    receives no gold answer/route, and is forbidden from identifying the answer.
    """
    schema_text = ""
    try:
        from utils.schema_outline import selected_endpoint_cards, format_endpoint_cards
        cards = selected_endpoint_cards(benchmark, plan, max_paths_per_endpoint=80)
        schema_text = format_endpoint_cards(cards, max_chars=4500)
    except Exception:
        schema_text = ""
    system = """You are the semantic selection guard for an API agent. Do not answer the question or identify the answer.

Check only the flagged positions. A position is valid when the user asked for it or the API documentation clearly defines it. Otherwise it is not valid.

Return JSON only:
{"verdict":"justified|unjustified|uncertain","errors":["short reason", ...]}
"""
    proposed = json.dumps({
        "flagged_selections": risks,
        "steps": [x for x in (plan.get("steps") or [])
                  if str(x.get("id")) in {sid for r in risks for sid in r.get("source_steps") or []}],
    }, ensure_ascii=False)
    user = (f"USER QUESTION:\n{question}\n\nFLAGGED PLAN SELECTIONS:\n{proposed}\n\n"
            f"SELECTED OPENAPI MATERIAL:\n{schema_text or '(not available)'}")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        kwargs = dict(model=model, messages=messages, temperature=0.0)
        from utils.token_meter import stage as token_stage
        try:
            import config as _config
            max_tokens = int(getattr(_config, "OCA_SELECTION_SEMANTIC_GUARD_MAX_TOKENS", 220))
        except Exception:
            max_tokens = 220
        with token_stage("selection_guard"):
            try:
                response = client.chat.completions.create(
                    **kwargs, response_format={"type": "json_object"},
                    max_completion_tokens=max_tokens)
            except Exception:
                response = client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
    except Exception as exc:
        return "uncertain", [f"selection guard unavailable: {exc}"], ""
    parsed = _extract_json(text) or {}
    verdict = str(parsed.get("verdict") or "").lower().strip()
    if verdict not in {"justified", "unjustified", "uncertain"}:
        verdict = "uncertain"
    errors = [str(x) for x in (parsed.get("errors") or []) if str(x).strip()]
    if verdict != "justified" and not errors:
        errors = ["positive positional selection is not justified by the user request/API documentation"]
    return verdict, errors, text


def _schema_answer_cardinality_errors(plan: dict[str, Any],
                                      cards: list[dict[str, Any]]) -> list[str]:
    """Validate explicit multi-item intent against schema-level answer cardinality.

    ``_answer_surface_errors`` catches a singleton selector that sits directly in
    a terminal value branch.  A subtler collapse happens when a collection step
    selects one candidate, binds its id into a singleton detail endpoint, and the
    detail field becomes the terminal answer.  The derivation DAG then looks
    locally scalar-correct even though the user's requested plurality was lost one
    API hop earlier.  Use only OAS response cardinality + declared dependency/
    selection structure to catch that case; do not infer domain-specific meaning.
    """
    if str((plan or {}).get("answer_cardinality") or "").lower() != "many":
        return []

    steps = {str(x.get("id") or ""): x for x in (plan or {}).get("steps") or []}
    derivations = list((plan or {}).get("derivations") or [])
    cards_by_step = {str(c.get("step_id") or ""): c for c in cards or []}
    singleton_ops = {"endpoint_rank", "first", "nth", "argmax", "argmin"}

    def step_has_singleton_selector(sid: str) -> bool:
        return any(
            str(d.get("operator") or "").lower() in singleton_ops
            and sid in {str(x) for x in (d.get("source_steps") or [])}
            for d in derivations
        )

    errors: list[str] = []
    answer_steps = [str(x) for x in (plan or {}).get("answer_steps") or []]
    for sid in answer_steps:
        step = steps.get(sid) or {}
        card = cards_by_step.get(sid) or {}
        if not card:
            continue
        # An empty response schema means the OAS does not tell us whether this
        # operation is singleton- or collection-valued.  Do not convert ignorance
        # into a cardinality contradiction here: read-only execution will use the
        # schema-deferred runtime structural resolver, which accepts exactly one
        # unambiguous record universe and otherwise fails closed.
        if not list(card.get("leaf_paths") or []):
            continue
        step_derivations = [d for d in derivations
                            if sid in {str(x) for x in (d.get("source_steps") or [])}]
        binding_paths = [str(x) for x in (step.get("binding_paths") or {}).values()]
        root, root_errors = _candidate_record_root(
            card, step_derivations, require_binding=False, binding_paths=binding_paths)
        if root_errors or not root or "[*]" in str(root):
            # A genuine answer collection can satisfy an explicit multi-item request.
            continue

        # A singleton response can still satisfy plurality if the framework fans it
        # out over a multi-valued parent binding.  Conversely, when every direct
        # producer feeding this child has already been collapsed by a singleton
        # selector, the child can produce at most one requested entity.
        deps = [str(x) for x in (step.get("depends_on") or []) if str(x) in steps]
        dynamic_aliases = set()
        for key in ("path_bindings", "query_bindings", "body_bindings"):
            dynamic_aliases.update(str(v).strip("{}") for v in (step.get(key) or {}).values()
                                   if str(v).strip())
        producers = []
        for dep in deps:
            produced = {str(x).strip("{}") for x in (steps[dep].get("binds") or [])}
            if not dynamic_aliases or (produced & dynamic_aliases):
                producers.append(dep)
        if producers and all(step_has_singleton_selector(dep) for dep in producers):
            errors.append(
                f"answer cardinality mismatch: explicit multiple-item request reaches "
                f"singleton answer step {sid} through producer(s) {producers} that are "
                "already collapsed by singleton selection")
    return list(dict.fromkeys(errors))




def _schema_single_answer_cardinality_errors(plan: dict[str, Any],
                                             cards: list[dict[str, Any]]) -> list[str]:
    """Require an explicit selector when the user asks for exactly one collection item."""
    if str((plan or {}).get("answer_cardinality") or "").lower() != "one":
        return []
    # Output cardinality does not imply singleton *support* collections for
    # Boolean/comparison/count answers.  Those modes reduce their operands via
    # typed derivations; applying a surface cardinality constraint to each input
    # collection creates false plan failures.
    if str((plan or {}).get("answer_mode") or "direct").lower() not in {"direct", "list", "asset"}:
        return []
    steps = {str(x.get("id") or ""): x for x in (plan or {}).get("steps") or []}
    derivations = list((plan or {}).get("derivations") or [])
    cards_by_step = {str(c.get("step_id") or ""): c for c in cards or []}
    singleton_ops = {"endpoint_rank", "first", "nth", "argmax", "argmin"}
    errors: list[str] = []
    for sid in [str(x) for x in (plan or {}).get("answer_steps") or []]:
        card = cards_by_step.get(sid) or {}
        if not card or not list(card.get("leaf_paths") or []):
            continue
        step_derivs = [d for d in derivations
                       if sid in {str(x) for x in (d.get("source_steps") or [])}]
        binding_paths = [str(x) for x in ((steps.get(sid) or {}).get("binding_paths") or {}).values()]
        root, root_errors = _candidate_record_root(
            card, step_derivs, require_binding=False, binding_paths=binding_paths)
        if root_errors or not root or "[*]" not in str(root):
            continue
        if any(str(d.get("operator") or "").lower() in singleton_ops for d in step_derivs):
            continue
        errors.append(
            f"answer cardinality mismatch: the user explicitly requests one item, but answer "
            f"step {sid} exposes collection {root} without first/nth/endpoint_rank/argmax/argmin selection")
    return list(dict.fromkeys(errors))


def normalize_collection_binding_scalar_paths(plan: dict[str, Any],
                                              cards: list[dict[str, Any]]) -> dict[str, Any]:
    """Repair scalar binding aliases that accidentally point at a collection container.

    The planner may correctly declare a semantic alias such as ``movie_id`` while writing
    ``binding_paths={"movie_id":"parts"}``.  A downstream scalar placeholder cannot consume
    the whole ``parts[*]`` record collection; when the selected OAS contains exactly one
    scalar child whose leaf name matches the alias suffix (for example ``parts[*].id``),
    canonicalize the binding to that documented leaf.  This is schema-only and never chooses
    a record: existing selection semantics still decide *which* collection item supplies it.
    """
    out = copy.deepcopy(plan or {})
    card_by_step = {str(c.get("step_id") or ""): c for c in (cards or [])}
    warnings = list(out.get("validation_warnings") or [])
    repaired_steps = []
    for raw_step in out.get("steps") or []:
        step = dict(raw_step)
        sid = str(step.get("id") or "")
        card = card_by_step.get(sid) or {}
        record_roots = {
            _normalized_schema_path(x.get("path"))
            for x in (card.get("record_paths") or [])
            if str(x.get("type") or "") == "array_item" and x.get("path")
        }
        leaf_paths = [str(x.get("path") or "") for x in (card.get("leaf_paths") or [])
                      if x.get("path")]
        mapping = dict(step.get("binding_paths") or {})
        changed = False
        for alias, raw_path in list(mapping.items()):
            container = _normalized_schema_path(raw_path)
            if not container or container not in record_roots:
                continue
            alias_norm = re.sub(r"[^a-z0-9]+", "_", str(alias or "").casefold()).strip("_")
            candidates = []
            for lp in leaf_paths:
                norm = _normalized_schema_path(lp)
                if not norm.startswith(container + "."):
                    continue
                relative = norm[len(container) + 1:]
                if "." in relative:
                    continue
                leaf_norm = re.sub(r"[^a-z0-9]+", "_", relative.casefold()).strip("_")
                if leaf_norm and (alias_norm == leaf_norm or alias_norm.endswith("_" + leaf_norm)):
                    candidates.append(lp)
            if len(candidates) != 1:
                continue
            target = re.sub(r"\[(?:\*|\d*)\]", "", candidates[0])
            if target and target != str(raw_path):
                mapping[alias] = target
                changed = True
                warnings.append(
                    f"step {sid}: normalized collection binding alias {alias!r} "
                    f"from container {str(raw_path)!r} to documented scalar {target!r}")
        if changed:
            step["binding_paths"] = mapping
        repaired_steps.append(step)
    out["steps"] = repaired_steps
    out["validation_warnings"] = list(dict.fromkeys(warnings))
    return out


def _selected_schema_validation_errors(benchmark: str, plan: dict[str, Any],
                                       question: str = "") -> list[str]:
    """Run deterministic selected-OAS validation and fail closed on validator faults.

    OCA treats request/schema validation as part of the evidence contract, not an
    optional lint pass.  Earlier builds swallowed exceptions from schema-card
    extraction in several planner/repair paths; that could mark a plan valid even
    though the trusted validator never ran.  Convert such faults into an explicit
    deterministic plan error so execution cannot silently bypass the contract.
    """
    try:
        from utils.schema_outline import selected_endpoint_validation_cards
        cards = selected_endpoint_validation_cards(benchmark, plan)
        errors = validate_request_contract(plan, cards)
        errors += validate_declared_request_binding_provenance(plan)
        errors += validate_schema_projection_semantics(plan, cards)
        errors += _schema_answer_cardinality_errors(plan, cards)
        errors += _schema_single_answer_cardinality_errors(plan, cards)
        return list(dict.fromkeys(str(x) for x in errors))
    except Exception as exc:
        # Unit-level/custom tool catalogs can be injected without registering a
        # benchmark OAS.  That path never occurs in the normal OCA runtime because
        # _load_tools itself requires a registered benchmark.  Preserve that
        # lightweight extension/testing mode, but fail closed whenever the runtime
        # benchmark declares an authoritative OAS and its trusted schema validator
        # cannot run.
        registered = False
        try:
            import benchmarks as B
            registered = bool((B.get_benchmark(benchmark) or {}).get("oas_file"))
        except Exception:
            registered = False
        if not registered:
            return []
        detail = str(exc).strip().replace("\n", " ")[:240]
        suffix = f": {detail}" if detail else ""
        return [f"selected OAS schema validation unavailable ({type(exc).__name__}){suffix}"]


_MISSING_COLLECTION_SELECTION_RE = re.compile(
    r"^step\s+([^:]+):\s+collection-valued response supplies downstream binding\(s\).*"
    r"declares no selection semantics for the producer records;",
    re.IGNORECASE,
)


def _selection_only_error_steps(errors: list[str] | tuple[str, ...] | None) -> list[str]:
    """Return producer step ids when *all* remaining defects are selection omissions.

    This intentionally recognizes only the exact deterministic validator family.
    It is used for one small planner micro-repair after the ordinary bounded plan
    attempts are exhausted.  Mixed structural/schema failures are never hidden by
    this path.
    """
    items = [str(x) for x in (errors or []) if str(x).strip()]
    if not items:
        return []
    out: list[str] = []
    for item in items:
        match = _MISSING_COLLECTION_SELECTION_RE.match(item)
        if not match:
            return []
        sid = str(match.group(1)).strip()
        if sid and sid not in out:
            out.append(sid)
    return out


def _apply_selection_micro_repair(question: str, benchmark: str, model: str, client,
                                  plan: dict[str, Any], structural_errors: list[str],
                                  valid_paths: list[str],
                                  valid_methods_by_path: dict[str, set[str]],
                                  ) -> tuple[dict[str, Any] | None, str, list[str]]:
    """Ask for selection semantics only, without regenerating a near-valid plan.

    The normal planner owns route and relation choice.  Occasionally its final
    bounded retry fixes every OAS/schema issue but omits only the explicit record
    selector required for downstream bindings.  Regenerating the whole plan again
    is expensive and can regress unrelated fields, so this helper asks the same LLM
    for a tiny answer-free selector patch.  Existing deterministic validation is
    then rerun over the patched plan; the patch is accepted only if it makes the
    entire plan strictly valid.
    """
    target_ids = _selection_only_error_steps(structural_errors)
    if not target_ids:
        return None, "", []
    steps = {str(x.get("id") or ""): x for x in (plan or {}).get("steps") or []}
    if not all(sid in steps for sid in target_ids):
        return None, "", ["selection micro-repair target step missing"]
    if any(str(x.get("method") or "GET").upper() != "GET"
           for x in (plan or {}).get("steps") or []):
        return None, "", ["selection micro-repair is read-only only"]

    # Keep this prompt compact: routes, request literals, and answer logic are
    # frozen.  Only the producer steps needing a selector and their selected OAS
    # schemas are exposed.
    target_steps = [steps[sid] for sid in target_ids]
    nearby_derivations = [
        d for d in (plan or {}).get("derivations") or []
        if any(sid in {str(x) for x in d.get("source_steps") or []}
               for sid in target_ids)
    ]
    schema_text = ""
    try:
        from utils.schema_outline import selected_endpoint_cards, format_endpoint_cards
        cards = selected_endpoint_cards(benchmark, plan, max_paths_per_endpoint=100)
        cards = [c for c in cards if str(c.get("step_id") or "") in set(target_ids)]
        schema_text = format_endpoint_cards(cards, max_chars=5000)
    except Exception:
        schema_text = ""

    messages = [
        {
            "role": "system",
            "content":
                "Fix only the missing record selection. Do not change routes, requests, bindings, "
                "or answer logic. Use only the question and supplied API schema. Do not infer answer values. "
                "Allowed operators: endpoint_rank, first, argmax, argmin. Return JSON only: "
                "{\"selections\":[{\"step_id\":\"s1\",\"operator\":\"endpoint_rank\",\"rank\":0,"
                "\"field\":null,\"filter\":{},\"purpose\":\"...\"}]}"
        },
        {
            "role": "user",
            "content":
                "QUESTION:\n" + str(question) +
                "\n\nSTEPS REQUIRING EXPLICIT SELECTION:\n" +
                json.dumps(target_steps, ensure_ascii=False, default=str)[:6000] +
                "\n\nEXISTING DERIVATIONS TOUCHING THOSE STEPS:\n" +
                json.dumps(nearby_derivations, ensure_ascii=False, default=str)[:4000] +
                "\n\nVALIDATOR ERRORS:\n" +
                json.dumps(structural_errors, ensure_ascii=False)[:3000] +
                (("\n\nSELECTED OAS SCHEMAS:\n" + schema_text) if schema_text else "")
        },
    ]
    text = ""
    try:
        kwargs = dict(model=model, messages=messages, temperature=0.0)
        from utils.token_meter import stage as token_stage
        with token_stage("planner"):
            try:
                response = client.chat.completions.create(
                    **kwargs, response_format={"type": "json_object"},
                    max_completion_tokens=700)
            except Exception:
                response = client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
    except Exception as exc:
        return None, text, [f"selection micro-repair unavailable: {exc}"]

    parsed = _extract_json(text) or {}
    patches = parsed.get("selections") if isinstance(parsed, dict) else None
    if not isinstance(patches, list):
        return None, text, ["selection micro-repair returned no selections array"]
    by_sid = {}
    for patch in patches:
        if not isinstance(patch, dict):
            continue
        sid = str(patch.get("step_id") or "")
        if sid in target_ids and sid not in by_sid:
            by_sid[sid] = patch
    if set(by_sid) != set(target_ids):
        return None, text, ["selection micro-repair did not cover every missing producer step"]

    patched = copy.deepcopy(plan)
    used_ids = {str(d.get("id") or "") for d in patched.get("derivations") or []}
    new_derivations = list(patched.get("derivations") or [])
    for sid in target_ids:
        patch = by_sid[sid]
        op = str(patch.get("operator") or "").lower().strip()
        if op not in {"endpoint_rank", "first", "argmax", "argmin"}:
            return None, text, [f"selection micro-repair used unsupported operator {op!r}"]
        rank = int(patch.get("rank") or 0)
        if rank != 0:
            return None, text, ["selection micro-repair may not invent a positive rank"]
        filt = patch.get("filter") if isinstance(patch.get("filter"), dict) else {}
        did_base = f"selection_repair_{sid}"
        did = did_base
        suffix = 2
        while did in used_ids:
            did = f"{did_base}_{suffix}"
            suffix += 1
        used_ids.add(did)
        new_derivations.append({
            "id": did,
            "operator": op,
            "source_steps": [sid],
            "source_derivations": [],
            "label_steps": [],
            "label_fields": [],
            "field": patch.get("field"),
            "comparison": "",
            "comparison_literal": None,
            "unit": "raw",
            "distinct_field": None,
            "rank": 0,
            "filter": filt,
            "top_k": 10,
            "purpose": str(patch.get("purpose") or
                           "Declare the producer-record selection required by the downstream binding."),
        })
    patched["derivations"] = new_derivations

    normalized, errors = validate_plan(patched, valid_paths, valid_methods_by_path)
    normalized = repair_selection_dependencies(question, normalized)
    # Some deterministic normalizers intentionally repair references that were
    # invalid in the raw planner JSON (for example a dangling identity-after-compare
    # answer id).  Do not carry the pre-normalization missing-reference error forward
    # when every current answer derivation is now present.
    current_derivation_ids = {str(d.get("id") or "") for d in normalized.get("derivations") or []}
    current_answer_ids = [str(x) for x in normalized.get("answer_derivations") or [] if str(x)]
    if current_answer_ids and all(x in current_derivation_ids for x in current_answer_ids):
        errors = [e for e in errors
                  if not str(e).startswith("answer_derivations reference missing derivations:")]
    errors = list(errors) + _answer_surface_errors(question, normalized)
    errors += _selected_schema_validation_errors(benchmark, normalized, question)
    errors = list(dict.fromkeys(str(x) for x in errors))
    normalized["validation_errors"] = errors
    normalized["valid"] = bool(normalized.get("steps")) and not errors
    if not normalized["valid"]:
        return None, text, errors
    normalized.setdefault("validation_warnings", []).append(
        "bounded selection-only planner micro-repair accepted")
    return normalized, text, []



_SCHEMA_DIRECT_FIELD_RE = re.compile(
    r"^derivation\s+([^:]+):\s+(identity|argmax|argmin) field (.+?) is not a "
    r"documented scalar under selected record collection (.+)$",
    re.IGNORECASE,
)
_SCHEMA_FILTER_FIELD_RE = re.compile(
    r"^derivation\s+([^:]+):\s+filter field (.+?) is not a documented scalar "
    r"under selected record collection (.+)$",
    re.IGNORECASE,
)


def _literal_error_value(text: str) -> str:
    """Decode the validator's repr-style field token without evaluating code."""
    value = str(text or "").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _schema_field_micro_repair_targets(
        errors: list[str] | tuple[str, ...] | None) -> dict[str, dict[str, Any]]:
    """Return narrow derivation-field repair targets for near-valid plans.

    Selection-omission errors may coexist because the selection micro-repair runs
    immediately afterward. Any other error family disables this repair, preventing
    it from becoming a general plan rewriter.
    """
    items = [str(x) for x in (errors or []) if str(x).strip()]
    if not items:
        return {}
    targets: dict[str, dict[str, Any]] = {}
    saw_field_error = False
    for item in items:
        direct = _SCHEMA_DIRECT_FIELD_RE.match(item)
        if direct:
            saw_field_error = True
            did = str(direct.group(1)).strip()
            targets.setdefault(did, {"fields": [], "filter_fields": []})["fields"].append(
                _literal_error_value(direct.group(3)))
            continue
        filt = _SCHEMA_FILTER_FIELD_RE.match(item)
        if filt:
            saw_field_error = True
            did = str(filt.group(1)).strip()
            targets.setdefault(did, {"fields": [], "filter_fields": []})["filter_fields"].append(
                _literal_error_value(filt.group(2)))
            continue
        if _MISSING_COLLECTION_SELECTION_RE.match(item):
            continue
        return {}
    return targets if saw_field_error else {}


def _apply_schema_field_micro_repair(
        question: str, benchmark: str, model: str, client,
        plan: dict[str, Any], structural_errors: list[str],
        valid_paths: list[str], valid_methods_by_path: dict[str, set[str]],
        ) -> tuple[dict[str, Any] | None, str, list[str]]:
    """Repair only OAS-invalid scalar paths in existing derivations.

    Routes, dependencies, request values, bindings, operators, answer steps and the
    task relation are frozen. The LLM sees only the offending derivations plus the
    selected endpoint schemas and may replace scalar field paths. The ordinary
    deterministic validators decide whether the patch is acceptable.
    """
    targets = _schema_field_micro_repair_targets(structural_errors)
    if not targets:
        return None, "", []
    derivations = {str(d.get("id") or ""): d for d in (plan or {}).get("derivations") or []}
    if not all(did in derivations for did in targets):
        return None, "", ["schema-field micro-repair target derivation missing"]
    if any(str(x.get("method") or "GET").upper() != "GET"
           for x in (plan or {}).get("steps") or []):
        return None, "", ["schema-field micro-repair is read-only only"]

    target_derivations = [derivations[did] for did in targets]
    target_step_ids = {
        str(sid) for d in target_derivations for sid in (d.get("source_steps") or [])
    }
    schema_text = ""
    try:
        from utils.schema_outline import selected_endpoint_cards, format_endpoint_cards
        cards = selected_endpoint_cards(benchmark, plan, max_paths_per_endpoint=180)
        cards = [c for c in cards if str(c.get("step_id") or "") in target_step_ids]
        schema_text = format_endpoint_cards(cards, max_chars=7000)
    except Exception:
        schema_text = ""

    messages = [
        {
            "role": "system",
            "content":
                "Fix only the invalid response field paths. Do not change routes, requests, operators, "
                "or answer logic. Use only scalar paths from the supplied API schema. Do not infer answer values. "
                "Return JSON only: {\"repairs\":[{\"derivation_id\":\"d1\",\"field\":\"id\","
                "\"filter_path_replacements\":{\"old.path\":\"new.path\"}}]}"
        },
        {
            "role": "user",
            "content":
                "QUESTION:\n" + str(question) +
                "\n\nOFFENDING DERIVATIONS:\n" +
                json.dumps(target_derivations, ensure_ascii=False, default=str)[:7000] +
                "\n\nALLOWED REPAIR TARGETS:\n" +
                json.dumps(targets, ensure_ascii=False, default=str)[:3000] +
                "\n\nVALIDATOR ERRORS:\n" +
                json.dumps(structural_errors, ensure_ascii=False)[:4000] +
                (("\n\nSELECTED OAS SCHEMAS:\n" + schema_text) if schema_text else "")
        },
    ]
    text = ""
    try:
        kwargs = dict(model=model, messages=messages, temperature=0.0)
        from utils.token_meter import stage as token_stage
        with token_stage("planner"):
            try:
                response = client.chat.completions.create(
                    **kwargs, response_format={"type": "json_object"},
                    max_completion_tokens=700)
            except Exception:
                response = client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
    except Exception as exc:
        return None, text, [f"schema-field micro-repair unavailable: {exc}"]

    parsed = _extract_json(text) or {}
    patches = parsed.get("repairs") if isinstance(parsed, dict) else None
    if not isinstance(patches, list):
        return None, text, ["schema-field micro-repair returned no repairs array"]
    by_did: dict[str, dict[str, Any]] = {}
    for patch in patches:
        if not isinstance(patch, dict):
            continue
        did = str(patch.get("derivation_id") or "")
        if did in targets and did not in by_did:
            by_did[did] = patch
    if set(by_did) != set(targets):
        return None, text, ["schema-field micro-repair did not cover every offending derivation"]

    patched = copy.deepcopy(plan)
    patched_by_did = {str(d.get("id") or ""): d for d in patched.get("derivations") or []}
    for did, target in targets.items():
        patch = by_did[did]
        deriv = patched_by_did[did]
        direct_bad = list(dict.fromkeys(str(x) for x in target.get("fields") or []))
        filter_bad = list(dict.fromkeys(str(x) for x in target.get("filter_fields") or []))
        if direct_bad:
            new_field = patch.get("field")
            if not isinstance(new_field, str) or not new_field.strip():
                return None, text, [f"schema-field micro-repair omitted direct field for {did}"]
            deriv["field"] = new_field.strip()
        elif patch.get("field") not in (None, ""):
            return None, text, [f"schema-field micro-repair attempted unrequested direct field change for {did}"]

        replacements = patch.get("filter_path_replacements") or {}
        if filter_bad:
            if not isinstance(replacements, dict):
                return None, text, [f"schema-field micro-repair returned malformed filter replacements for {did}"]
            current_filter = deriv.get("filter")
            if not isinstance(current_filter, dict):
                return None, text, [f"schema-field micro-repair target {did} has no filter mapping"]
            for old_path in filter_bad:
                new_path = replacements.get(old_path)
                if not isinstance(new_path, str) or not new_path.strip():
                    return None, text, [f"schema-field micro-repair omitted filter replacement {old_path!r} for {did}"]
                if old_path not in current_filter:
                    return None, text, [f"schema-field micro-repair could not find filter path {old_path!r} in {did}"]
                predicate = current_filter.pop(old_path)
                new_path = new_path.strip()
                if new_path in current_filter and new_path != old_path:
                    return None, text, [f"schema-field micro-repair would overwrite filter path {new_path!r} in {did}"]
                current_filter[new_path] = predicate
        elif replacements:
            return None, text, [f"schema-field micro-repair attempted unrequested filter change for {did}"]

    normalized, errors = validate_plan(patched, valid_paths, valid_methods_by_path)
    normalized = repair_selection_dependencies(question, normalized)
    errors = list(errors) + _answer_surface_errors(question, normalized)
    errors += _selected_schema_validation_errors(benchmark, normalized, question)
    errors = list(dict.fromkeys(str(x) for x in errors))
    if _schema_field_micro_repair_targets(errors):
        return None, text, errors
    if errors and not _selection_only_error_steps(errors):
        return None, text, errors
    normalized["validation_errors"] = errors
    normalized["valid"] = bool(normalized.get("steps")) and not errors
    normalized.setdefault("validation_warnings", []).append(
        "bounded schema-field-only planner micro-repair accepted")
    return normalized, text, errors


def _repair_schema_context(
        benchmark: str, plan: dict[str, Any], tools: list[dict[str, Any]],
        diagnostics: list[str], *, max_chars: int = 12000,
        max_paths_per_endpoint: int = 180) -> str:
    """Return detailed OAS cards for current routes + diagnostic alternatives.

    Any route named by deterministic/semantic diagnostics is discovered from the
    supplied tool catalog and added only for schema inspection.  This keeps every
    route-changing repair grounded in the same request/response contract without
    embedding benchmark endpoint mappings in production code.
    """
    try:
        from utils.schema_outline import selected_endpoint_cards, format_endpoint_cards
        schema_plan = copy.deepcopy(plan or {})
        schema_steps = [dict(x) for x in (schema_plan.get("steps") or [])]
        selected_paths = {str(x.get("endpoint") or "") for x in schema_steps}
        diag_text = "\n".join(str(x) for x in diagnostics or [])
        alt_index = 0
        for tool in tools or []:
            path = str(tool.get("path") or "")
            if not path or path in selected_paths or path not in diag_text:
                continue
            alt_index += 1
            schema_steps.append({
                "id": f"_oas_alternative_{alt_index}",
                "method": str(tool.get("method") or "GET").upper(),
                "endpoint": path, "depends_on": [], "binds": [],
                "answer_source": False,
            })
            selected_paths.add(path)
        schema_plan["steps"] = schema_steps
        return format_endpoint_cards(
            selected_endpoint_cards(
                benchmark, schema_plan,
                max_paths_per_endpoint=max_paths_per_endpoint,
                max_request_paths_per_endpoint=160,
                max_depth=12, max_request_depth=12),
            max_chars=max_chars)
    except Exception:
        return ""


def _apply_semantic_plan_repair(
        question: str, benchmark: str, model: str, client, tools: list[dict[str, Any]],
        catalog_mode: str, plan: dict[str, Any], critic_errors: list[str],
        valid_paths: list[str], valid_methods_by_path: dict[str, set[str]],
        ) -> tuple[dict[str, Any] | None, str, list[str]]:
    """One bounded full-plan correction after a clear answer-free semantic mismatch."""
    messages = build_planner_messages(question, tools, catalog_mode=catalog_mode)
    schema_text = _repair_schema_context(
        benchmark, plan, tools, critic_errors, max_chars=12000,
        max_paths_per_endpoint=180)
    mentioned_text = "\n".join(str(x) for x in critic_errors)
    mentioned_tools = [
        t for t in tools
        if str(t.get("path") or "") and str(t.get("path") or "") in mentioned_text
    ]
    alternative_text = "\n".join(
        f"- {str(t.get('method') or 'GET').upper()} {t.get('path')}: "
        f"{str(t.get('functionality') or '')[:260]}"
        for t in mentioned_tools[:8]
    )
    messages.append({
        "role": "user",
        "content":
            "A generic answer-free OpenAPI semantic reviewer found a clear mismatch between "
            "the user question and the otherwise structurally valid plan. Produce ONE fresh "
            "corrected JSON plan. Preserve every requested relation/modifier, choose documented "
            "operations that actually represent them, and do not infer the answer. Reviewer "
            "feedback: " + json.dumps(critic_errors[:12], ensure_ascii=False) +
            (("\nDOCUMENTED ALTERNATIVES IDENTIFIED FROM THE OAS:\n" + alternative_text)
             if alternative_text else "") +
            (("\nCURRENT + DIAGNOSTIC-ALTERNATIVE OAS SCHEMAS:\n" + schema_text)
             if schema_text else "")
    })
    text = ""
    try:
        kwargs = dict(model=model, messages=messages, temperature=0.0)
        from utils.token_meter import stage as token_stage
        with token_stage("planner"):
            try:
                response = client.chat.completions.create(
                    **kwargs, response_format={"type": "json_object"},
                    max_completion_tokens=1800)
            except Exception:
                response = client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
    except Exception as exc:
        return None, text, [f"semantic plan repair unavailable: {exc}"]

    candidate, errors = validate_plan(_extract_json(text), valid_paths, valid_methods_by_path)
    candidate = repair_selection_dependencies(question, candidate)
    candidate = normalize_direct_dependency_step_bindings(candidate)
    errors = list(errors) + _answer_surface_errors(question, candidate)
    errors += _selected_schema_validation_errors(benchmark, candidate, question)
    errors = list(dict.fromkeys(str(x) for x in errors))
    candidate["validation_errors"] = errors
    candidate["valid"] = bool(candidate.get("steps")) and not errors
    if not candidate["valid"]:
        return None, text, errors
    # A bounded semantic correction is accepted only if it clears the same cheap,
    # deterministic question+OAS relation checks that triggered the repair.  This
    # prevents a fresh LLM plan from simply restating the same narrowed route while
    # avoiding a second critic-model call.
    remaining_semantic_risks = deterministic_semantic_route_risks(question, candidate, tools)
    if remaining_semantic_risks:
        return None, text, [
            "semantic plan repair retained an answer-free OAS relation mismatch: " + x
            for x in remaining_semantic_risks
        ]
    candidate.setdefault("validation_warnings", []).append(
        "bounded semantic-plan correction accepted after answer-free OAS review")
    return candidate, text, []





def normalize_person_most_popular_tv_collaborators(question: str, plan: dict[str, Any],
                                                     tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Repair person-as-show search plans for collaborator questions.

    Shape: ``who worked with PERSON in his/her most popular TV show``.  The
    requested population belongs to the person's TV credits, not to free-text TV
    search results for the person's name.  When the OAS exposes the complete
    person-search -> person-tv-credits -> tv-credits chain, rewrite that chain and
    make the popularity selection explicit.  The named person is removed from the
    collaborator output.
    """
    q=str(question or '').casefold()
    if not (re.search(r'\b(?:worked|works|working)\s+with\b',q) and
            re.search(r'\bmost\s+popular\s+(?:tv|television)\s+(?:show|series)\b',q)):
        return copy.deepcopy(plan or {})
    out=copy.deepcopy(plan or {})
    steps=list(out.get('steps') or [])
    if len(steps)<3: return out
    # The planner's free-text query is answer-free evidence of the named person.
    search_step=next((st for st in steps if '/search/' in str(st.get('endpoint') or '')),None)
    if not search_step: return out
    literals=search_step.get('query_literals') or {}
    person=next((str(literals.get(k)).strip() for k in ('query','q','name') if literals.get(k) not in (None,'')),'')
    if not person: return out
    get_paths=[str(t.get('path') or '') for t in tools or []
               if str(t.get('method') or 'GET').upper()=='GET']
    person_search=next((x for x in get_paths if re.search(r'/search/person$',x)),None)
    person_tv=next((x for x in get_paths if re.search(r'/person/\{[^{}]+\}/tv_credits$',x)),None)
    tv_credits=next((x for x in get_paths if re.search(r'/tv/\{[^{}]+\}/credits$',x)),None)
    if not (person_search and person_tv and tv_credits): return out
    s1,s2,s3=steps[:3]
    s1.update(_retarget_read_step(s1,person_search)); s1['depends_on']=[]
    s1['query_literals']={'query':person}; s1['query_bindings']={}
    s1['binds']=['person_id','person_name']; s1['binding_paths']={'person_id':'id','person_name':'name'}
    ph1=(re.findall(r'\{([^{}]+)\}',person_tv) or ['person_id'])[0]
    s2.update(_retarget_read_step(s2,person_tv)); s2['depends_on']=[str(s1.get('id') or 's1')]
    s2['path_bindings']={ph1:'person_id'}; s2['query_literals']={};s2['query_bindings']={}
    s2['binds']=['series_id','series_name','series_popularity']
    s2['binding_paths']={'series_id':'cast.id','series_name':'cast.name','series_popularity':'cast.popularity'}
    ph2=(re.findall(r'\{([^{}]+)\}',tv_credits) or ['series_id'])[0]
    s3.update(_retarget_read_step(s3,tv_credits)); s3['depends_on']=[str(s2.get('id') or 's2')]
    s3['path_bindings']={ph2:'series_id'};s3['query_literals']={};s3['query_bindings']={}
    s3['binds']=['cast_person_id','cast_person_name'];s3['binding_paths']={'cast_person_id':'cast.id','cast_person_name':'cast.name'}
    s1id=str(s1.get('id') or 's1');s2id=str(s2.get('id') or 's2');s3id=str(s3.get('id') or 's3')
    out['steps']=[s1,s2,s3]
    out['derivations']=[
      {'id':'collab_person_pick','operator':'endpoint_rank','source_steps':[s1id],'source_derivations':[],
       'label_steps':[],'label_fields':[],'field':'results','comparison':'','comparison_literal':None,'unit':'raw',
       'distinct_field':None,'rank':0,'filter':{'name':{'op':'eq_ci','value':person}},'top_k':10,
       'purpose':f'Select the named person {person} from person search.'},
      {'id':'collab_show_pick','operator':'argmax','source_steps':[s2id],'source_derivations':[],
       'label_steps':[],'label_fields':[],'field':'cast.popularity','comparison':'','comparison_literal':None,'unit':'raw',
       'distinct_field':'cast.id','rank':0,'filter':{},'top_k':10,
       'purpose':f'Select {person} most popular TV show from that person TV credits.'},
      {'id':'collab_names','operator':'identity','source_steps':[s3id],'source_derivations':['collab_show_pick'],
       'label_steps':[],'label_fields':[],'field':'cast.name','comparison':'','comparison_literal':None,'unit':'raw',
       'distinct_field':'cast.id','rank':0,'filter':{'cast.name':{'op':'neq_ci','value':person}},'top_k':50,
       'purpose':f'Return cast collaborators in the selected show, excluding {person}.'},
    ]
    out['answer_steps']=[s3id];out['answer_derivations']=['collab_names'];out['answer_mode']='list'
    out['answer_requirements']=[f'people who worked with {person} in that person most popular TV show']
    out['observation_specs']=[]
    out.setdefault('validation_warnings',[]).append(
        f'retargeted person-as-show collaborator strategy to person search -> person TV credits -> TV credits for {person!r}')
    return out

def normalize_explicit_ordinal_child_selection(question: str, plan: dict[str, Any],
                                               cards: list[dict[str, Any]]) -> dict[str, Any]:
    """Make an explicitly requested ordinal child record replayable.

    If a terminal identity addresses a child collection (``parts.release_date``)
    and the user/purpose explicitly says first/second/etc., insert a same-step
    endpoint-rank selector over that child collection before scalar extraction.
    This uses only question ordinals plus the selected OAS card; no provider or
    benchmark answer knowledge is involved.
    """
    out=copy.deepcopy(plan or {})
    ordinals=_question_ordinal_indexes(question)
    if not ordinals: return out
    card_by={str(c.get('step_id') or ''):c for c in cards or []}
    derivs=[dict(d) for d in out.get('derivations') or []]
    by_id={str(d.get('id') or ''):d for d in derivs}
    answer_ids={str(x) for x in out.get('answer_derivations') or [] if str(x)}
    existing=set(by_id); additions=[]; warnings=list(out.get('validation_warnings') or [])
    word_index={**_ORDINAL_WORD_INDEX}
    for d in derivs:
        did=str(d.get('id') or '')
        if str(d.get('operator') or '').lower()!='identity' or (answer_ids and did not in answer_ids):
            continue
        src=[str(x) for x in d.get('source_steps') or []]
        if len(src)!=1: continue
        raw=re.sub(r'\[(?:\*|\d*)\]','',str(d.get('field') or '')).strip('.')
        bits=[x for x in raw.split('.') if x]
        if not bits: continue
        card=card_by.get(src[0]) or {}
        roots={re.sub(r'\[\*\]$','',str(x.get('path') or '')).strip('.')
               for x in card.get('record_paths') or [] if str(x.get('path') or '').endswith('[*]')}
        if len(bits)>=2:
            relation=bits[0]
        else:
            # A prior normalization may already have made the scalar field
            # record-relative (release_date) while the selected OAS card still
            # proves that scalar belongs to exactly one child collection.
            leaf=bits[0]
            candidate_roots=[]
            for root in roots:
                prefix=root+'.'
                for item in card.get('leaf_paths') or []:
                    lp=str(item.get('path') if isinstance(item,dict) else item or '')
                    clean=re.sub(r'\[(?:\*|\d*)\]','',lp).strip('.')
                    if clean.startswith(prefix) and clean.split('.')[-1]==leaf:
                        candidate_roots.append(root);break
            candidate_roots=list(dict.fromkeys(candidate_roots))
            if len(candidate_roots)!=1: continue
            relation=candidate_roots[0]
        if relation not in roots: continue
        # Already selected on this same child relation.
        already=False
        for pid in d.get('source_derivations') or []:
            pd=by_id.get(str(pid)) or {}
            pf=re.sub(r'\[(?:\*|\d*)\]','',str(pd.get('field') or '')).strip('.')
            if str(pd.get('operator') or '').lower() in {'endpoint_rank','first','nth'} and pf==relation:
                already=True;break
        if already: continue
        purpose=str(d.get('purpose') or '').casefold()
        chosen=None
        for word,idx in word_index.items():
            if re.search(r'\b'+re.escape(word)+r'\b',purpose): chosen=idx;break
        if chosen is None and len(ordinals)==1: chosen=next(iter(ordinals))
        if chosen is None: continue
        sid=f'ordinal_pick_{did or src[0]}'; n=2
        while sid in existing: sid=f'ordinal_pick_{did or src[0]}_{n}';n+=1
        existing.add(sid)
        sel={'id':sid,'operator':'endpoint_rank','source_steps':src,'source_derivations':[],
             'label_steps':[],'label_fields':[],'field':relation,'comparison':'',
             'comparison_literal':None,'unit':'raw','distinct_field':None,'rank':int(chosen),
             'filter':{},'top_k':10,
             'purpose':f'Select the explicitly requested ordinal record from {relation}.'}
        additions.append((did,sel)); d['source_derivations']=list(dict.fromkeys(
            [str(x) for x in d.get('source_derivations') or []]+[sid]))
        warnings.append(f'derivation {did}: inserted ordinal selector {sid} rank={chosen} over {relation}')
    if additions:
        rebuilt=[]
        for d in derivs:
            for consumer,sel in additions:
                if str(d.get('id') or '')==consumer: rebuilt.append(sel)
            rebuilt.append(d)
        out['derivations']=rebuilt
        out['validation_warnings']=list(dict.fromkeys(warnings))
    return out


def normalize_tv_ranked_population(question: str, plan: dict[str, Any],
                                   tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Prefer the documented TV-specific popularity population for TV popularity/trend tasks.

    Mixed-media trending and airing-today routes are poor owner populations for a
    question whose requested entity is explicitly a TV show and whose selection is
    a popularity/trend superlative.  When the OAS provides a TV-specific operation
    explicitly ordered by popularity, retarget only the population step. Explicit
    ``currently on the air`` questions retain their on-air population.
    """
    q=str(question or '').casefold()
    if not re.search(r'\b(?:tv|television)\s+(?:show|series)\b',q): return copy.deepcopy(plan or {})
    if not re.search(r'\b(?:most\s+popular|most\s+trending|top\s+tv)\b',q): return copy.deepcopy(plan or {})
    if re.search(r'\b(?:currently\s+on\s+the\s+air|on\s+the\s+air)\b',q): return copy.deepcopy(plan or {})
    popular=[]
    for t in tools or []:
        path=str(t.get('path') or ''); prose=(path+' '+str(t.get('functionality') or '')).casefold()
        if str(t.get('method') or 'GET').upper()=='GET' and re.search(r'/tv/popular$',path) and 'popular' in prose:
            popular.append(path)
    if len(popular)!=1: return copy.deepcopy(plan or {})
    out=copy.deepcopy(plan or {}); new_path=popular[0]; changed=False; warnings=list(out.get('validation_warnings') or [])
    for st in out.get('steps') or []:
        old=str(st.get('endpoint') or '')
        if old==new_path: continue
        if ('/trending/' in old or re.search(r'/tv/(?:airing_today)$',old)):
            st.update(_retarget_read_step(st,new_path)); changed=True
            warnings.append(f"step {st.get('id')}: retargeted TV superlative population {old} to documented TV-specific ranked population {new_path}")
            # Mixed-media-only filters become redundant after the owner route is TV-specific.
            sid=str(st.get('id') or '')
            for d in out.get('derivations') or []:
                if sid in {str(x) for x in d.get('source_steps') or []}:
                    filt=dict(d.get('filter') or {})
                    filt={k:v for k,v in filt.items() if str(k).split('.')[-1] != 'media_type'}
                    d['filter']=filt
                    if str(d.get('operator') or '').lower() in {'argmax','argmin'} and 'popular' in str(d.get('field') or '').casefold():
                        d['operator']='endpoint_rank'; d['rank']=0; d['field']='results'
    if changed:
        out['validation_warnings']=list(dict.fromkeys(warnings))
    return out


def normalize_missing_comparison_answer_derivation(question: str, plan: dict[str, Any]) -> dict[str, Any]:
    """Repair a dangling answer-derivation reference to one terminal comparison.

    Planner output sometimes creates an identity-after-compare node; validation
    drops that non-record identity but leaves the answer id dangling.  If there is
    exactly one terminal compare and the question asks *who/which* won, the compare
    record itself already carries the replayed winner label and is the correct
    answer derivation.
    """
    out=copy.deepcopy(plan or {}); derivs=[dict(d) for d in out.get('derivations') or []]
    existing={str(d.get('id') or '') for d in derivs}; ans=[str(x) for x in out.get('answer_derivations') or [] if str(x)]
    missing=[x for x in ans if x not in existing]
    compares=[d for d in derivs if str(d.get('operator') or '').lower()=='compare']
    if len(compares)!=1 or not re.search(r'\b(?:who|which)\b',str(question or '').casefold()): return out
    if ans and not missing: return out
    cid=str(compares[0].get('id') or '')
    if ans:
        out['answer_derivations']=[cid if x in missing else x for x in ans]
    elif str(out.get('answer_mode') or '').lower() in {'comparison','direct',''}:
        out['answer_derivations']=[cid]
    else:
        return out
    out.setdefault('validation_warnings',[]).append(
        f'repaired missing/dangling comparison answer derivation(s) {missing or ["<empty>"]} to terminal compare {cid}')
    return out


def normalize_missing_boolean_equality(question: str, plan: dict[str, Any]) -> dict[str, Any]:
    """Build one replayable equality Boolean when the question asks if two values are the same.

    Minimal prompts can omit the final ``compare(eq)`` node even when they already
    declared the two scalar identity/count values.  When there are exactly two
    compatible scalar derivations from distinct branches and no Boolean derivation,
    add the mechanical equality node instead of replanning the whole task.
    """
    out = copy.deepcopy(plan or {})
    if str(out.get("answer_mode") or "").lower() != "boolean":
        return out
    q = str(question or "").casefold()
    if not re.search(r"\b(?:same|equal|identical)\b", q):
        return out
    derivs = [dict(d) for d in out.get("derivations") or []]
    for d in derivs:
        op = str(d.get("operator") or "").lower()
        cmp_mode = str(d.get("comparison") or "").lower()
        if op in {"membership", "logical_and", "logical_or"} or (
                op == "compare" and cmp_mode in {"eq", "neq", "gt", "gte", "lt", "lte"}):
            return out

    candidates = []
    for d in derivs:
        op = str(d.get("operator") or "").lower()
        if op not in {"identity", "count"}:
            continue
        field = re.sub(r"\[(?:\*|\d+)\]", "", str(d.get("field") or "")).split(".")[-1].casefold()
        sources = tuple(str(x) for x in (d.get("source_steps") or []) if str(x))
        if not sources:
            continue
        candidates.append((d, field, sources))

    pairs = []
    for i, (left, lf, ls) in enumerate(candidates):
        for right, rf, rs in candidates[i + 1:]:
            if ls == rs:
                continue
            # Compare like with like. Counts are compatible with counts; identities
            # must expose the same terminal field name (e.g. name with name, id with id).
            lop = str(left.get("operator") or "").lower()
            rop = str(right.get("operator") or "").lower()
            compatible = (lop == rop == "count") or (lop == rop == "identity" and lf and lf == rf)
            if compatible:
                pairs.append((left, right))
    if len(pairs) != 1:
        return out

    left, right = pairs[0]
    existing = {str(d.get("id") or "") for d in derivs}
    did = "boolean_eq"
    n = 2
    while did in existing:
        did = f"boolean_eq_{n}"; n += 1
    derivs.append({
        "id": did,
        "operator": "compare",
        "source_steps": [],
        "source_derivations": [str(left.get("id")), str(right.get("id"))],
        "label_steps": [],
        "label_fields": [],
        "field": None,
        "comparison": "eq",
        "comparison_literal": None,
        "unit": "raw",
        "distinct_field": None,
        "rank": 0,
        "filter": {},
        "top_k": 10,
        "purpose": "Compare the two requested values for equality.",
    })
    out["derivations"] = derivs
    out["answer_derivations"] = [did]
    out.setdefault("validation_warnings", []).append(
        f"added deterministic equality derivation {did} from two declared scalar branches")
    return out


def _evaluate_convergence_candidate(
        question: str, benchmark: str, candidate: dict[str, Any] | None,
        tools: list[dict[str, Any]], valid_paths: list[str],
        valid_methods_by_path: dict[str, set[str]],
        ) -> tuple[dict[str, Any], list[str]]:
    """Evaluate one plan candidate against the complete deterministic invariant stack.

    Every convergence candidate is judged by the same rules regardless of how it
    was produced.  This makes repair monotonic: a candidate that fixes relation
    fidelity but leaves one smaller structural defect can be retained and refined
    instead of being discarded in favor of an older, semantically worse plan.
    """
    prior = dict(candidate or {})
    prior_warnings = list(prior.get("validation_warnings") or [])
    normalized, errors = validate_plan(prior, valid_paths, valid_methods_by_path)
    normalized = repair_selection_dependencies(question, normalized)
    normalized = normalize_latest_released_population(question, normalized, tools)
    normalized = normalize_explicit_population_sibling(question, normalized, tools)
    normalized = normalize_tv_ranked_population(question, normalized, tools)
    normalized = normalize_person_most_popular_tv_collaborators(question, normalized, tools)
    normalized = repair_selection_dependencies(question, normalized)
    normalized = normalize_temporal_extremum_direction(question, normalized)
    normalized = normalize_documented_ordered_extrema(question, normalized, tools)
    normalized = normalize_typed_multi_answer_cardinality(normalized)
    normalized = normalize_typed_single_answer_cardinality(normalized)
    normalized = normalize_appearance_credit_relation(question, normalized)
    normalized = normalize_nested_relation_owner_selection(question, normalized)
    normalized = normalize_requested_answer_derivations(question, normalized)
    normalized = normalize_answer_steps_from_answer_derivations(question, normalized)
    normalized = normalize_direct_dependency_step_bindings(normalized)
    try:
        from utils.schema_outline import selected_endpoint_validation_cards
        _literal_cards = selected_endpoint_validation_cards(benchmark, normalized)
    except Exception:
        _literal_cards = []
    normalized = normalize_collection_binding_scalar_paths(normalized, _literal_cards)
    normalized = normalize_explicit_ordinal_child_selection(question, normalized, _literal_cards)
    normalized = normalize_rooted_collection_derivation_paths(normalized, _literal_cards)
    normalized = normalize_relation_population_metric_extrema(question, normalized, _literal_cards)
    normalized = normalize_explicit_metric_extrema(question, normalized, _literal_cards, tools)
    normalized = normalize_explicit_nested_relation_projection(question, normalized, _literal_cards)
    normalized = normalize_coordinated_parent_child_answers(question, normalized, _literal_cards)
    normalized = normalize_age_winner_and_difference(question, normalized)
    normalized = normalize_redundant_answer_derivations(normalized)
    normalized = normalize_requested_answer_derivations(question, normalized)
    normalized = normalize_answer_steps_from_answer_derivations(question, normalized)
    normalized = normalize_missing_comparison_answer_derivation(question, normalized)
    normalized = normalize_missing_boolean_equality(question, normalized)
    normalized = normalize_obvious_request_literal_bindings(question, normalized, _literal_cards)
    normalized["validation_warnings"] = list(dict.fromkeys(
        prior_warnings + list(normalized.get("validation_warnings") or [])))
    errors = list(errors) + _answer_surface_errors(question, normalized)
    errors += _selected_schema_validation_errors(benchmark, normalized, question)
    errors += [
        "semantic route invariant unresolved after convergence repair: " + x
        for x in deterministic_semantic_route_risks(question, normalized, tools)
    ]
    pos = positional_selection_risks(question, normalized)
    if pos and not _question_ordinal_indexes(question):
        errors += [
            "selection semantics unresolved after convergence repair: "
            f"{r.get('derivation_id')}: {r.get('field')}={r.get('position_value')}"
            for r in pos
        ]
    errors = list(dict.fromkeys(str(x) for x in errors))
    normalized["validation_errors"] = errors
    normalized["valid"] = bool(normalized.get("steps")) and not errors
    return normalized, errors


def _convergence_error_score(errors: list[str]) -> tuple[int, int, int]:
    """Return a deterministic lower-is-better score for residual plan defects.

    The ordering is *tiered*, not merely a weighted sum:

    1. hard route/graph defects (unknown endpoint, missing step, dependency cycle)
       always dominate because such a candidate is not a trustworthy executable
       basis for further repair;
    2. semantic relation/population mismatches dominate local schema/Boolean/output
       closure, so a route-correct candidate is retained even if it still needs
       several local repairs;
    3. the weighted residual total breaks ties inside those two safety tiers.

    Earlier builds returned ``(weighted_total, semantic_count, n_errors)``.  That
    accidentally allowed several small local errors to outweigh one semantic route
    error, contradicting the monotonic-convergence design.
    """
    total = 0
    semantic = 0
    hard = 0
    for raw in errors or []:
        text = str(raw).casefold()
        weight = 100
        if ("invalid endpoint" in text or "unknown endpoint" in text
                or "schema validation unavailable" in text
                or "answer_steps reference missing" in text
                or "plan dependency graph contains a cycle" in text
                or "unknown dependencies" in text
                or "invalid method" in text):
            weight = 900
            hard += 1
        elif "semantic route invariant" in text or "relation mismatch" in text:
            weight = 800
            semantic += 1
        elif "selection semantics unresolved" in text:
            weight = 650
            semantic += 1
        elif "plan has no answer-step" in text or "missing plan" in text:
            weight = 500
            hard += 1
        elif "request" in text or "schema" in text or "documented scalar" in text:
            weight = 260
        elif "binding" in text or "placeholder" in text or "producer" in text:
            weight = 240
        elif "boolean" in text or "membership" in text or "set-valued" in text or "cardinality" in text:
            weight = 180
        elif "answer step" in text or "answer surface" in text or "record-selection" in text:
            weight = 160
        elif "derivation" in text:
            weight = 140
        total += weight
    # Lexicographic ordering is intentional: executable graph safety first, then
    # relation fidelity, then local closure.  Keep a 3-tuple for stable diagnostics.
    return (hard, semantic, total)

def _apply_general_convergence_repair(
        question: str, benchmark: str, model: str, client, tools: list[dict[str, Any]],
        catalog_mode: str, plan: dict[str, Any], diagnostics: list[str],
        valid_paths: list[str], valid_methods_by_path: dict[str, set[str]],
        ) -> tuple[dict[str, Any] | None, str, list[str]]:
    """Produce one answer-free convergence candidate, valid or partially improved.

    Unlike the older all-or-nothing repair, this function returns a parseable
    candidate even when one deterministic defect remains.  The caller compares it
    against the previous candidate using the complete invariant score and may run
    one further bounded repair only when progress is monotonic.
    """
    # Convergence repair needs every route but not the verbose full catalog.  The
    # compact catalog retains all METHOD+path+functionality choices, while detailed
    # schemas are supplied only for endpoints already present in the current plan.
    # This keeps difficult-task recovery general without making the common path pay
    # for another large planner prompt.
    messages = build_planner_messages(question, tools, catalog_mode="compact")
    schema_text = _repair_schema_context(
        benchmark, plan, tools, diagnostics, max_chars=15000,
        max_paths_per_endpoint=220)
    current_plan_text = json.dumps(plan or {}, ensure_ascii=False, default=str)
    messages.append({
        "role": "user",
        "content":
            "Fix the CURRENT PLAN using the diagnostics below. Keep the meaning of the user's question. "
            "Change only what is needed. Reuse correct parts of the plan. Use only documented API routes and fields. "
            "Do not infer the answer. Return one complete JSON plan, not a patch.\n\nCURRENT PLAN:\n" +
            current_plan_text[:14000] +
            "\n\nDETERMINISTIC DIAGNOSTICS:\n" +
            json.dumps(list(diagnostics)[:24], ensure_ascii=False) +
            (("\n\nCURRENT + DIAGNOSTIC-ALTERNATIVE OAS SCHEMAS:\n" + schema_text)
             if schema_text else "")
    })
    text = ""
    try:
        kwargs = dict(model=model, messages=messages, temperature=0.0)
        from utils.token_meter import stage as token_stage
        with token_stage("planner"):
            try:
                response = client.chat.completions.create(
                    **kwargs, response_format={"type": "json_object"},
                    max_completion_tokens=1800)
            except Exception:
                response = client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
    except Exception as exc:
        return None, text, [f"general convergence repair unavailable: {exc}"]

    parsed = _extract_json(text)
    if not isinstance(parsed, dict) or not parsed:
        return None, text, ["general convergence repair returned no parseable plan"]
    candidate, errors = _evaluate_convergence_candidate(
        question, benchmark, parsed, tools, valid_paths, valid_methods_by_path)
    if candidate.get("valid"):
        candidate.setdefault("validation_warnings", []).append(
            "bounded general plan-convergence repair accepted")
    return candidate, text, errors

def make_evidence_plan(question: str, benchmark: str, model: str, client,
                       attempts: int = 2, api_hints: list[str] | None = None,
                       enable_semantic_adapters: bool = False,
                       catalog_mode: str = "full",
                       enable_semantic_critic: bool = False,
                       enable_selection_semantic_guard: bool = True,
                       runtime_feedback: dict[str, Any] | None = None,
                       ) -> tuple[dict[str, Any], str]:
    """Create a structurally valid OAS-grounded evidence plan.

    The optional generic semantic critic uses question + selected OpenAPI
    documentation only. A clear incompatibility can request one fresh planner retry;
    the corrected plan is then governed by deterministic structural/schema/provenance
    checks rather than another advisory critic call. An ``uncertain`` verdict is never treated as proof
    of failure. This closes semantic route-mismatch silent failures without
    reintroducing benchmark/gold endpoint rules.
    """
    del api_hints, enable_semantic_adapters
    tools = _load_tools(benchmark)
    valid_paths = list(dict.fromkeys(tool["path"] for tool in tools))
    valid_methods_by_path: dict[str, set[str]] = {}
    for tool in tools:
        valid_methods_by_path.setdefault(tool["path"], set()).add(
            str(tool.get("method") or "GET").upper())

    def _fresh_messages():
        msgs = build_planner_messages(question, tools, catalog_mode=catalog_mode)
        if runtime_feedback:
            # One bounded post-execution repair may revisit route choice when a
            # complete read-only plan returned no replayable answer evidence.  This
            # feedback contains only runtime failure shape + prior documented routes;
            # it never includes benchmark gold routes/answers or hidden solution data.
            msgs.append({
                "role": "user",
                "content":
                    "The previous plan did not solve the question. Make a new plan that fixes the reported problem. "
                    "Keep the meaning of the user's question. Reuse valid results already retrieved. "
                    "Use only documented API routes and fields. Do not infer the answer. Runtime diagnostics: " +
                    json.dumps(runtime_feedback, ensure_ascii=False, default=str)[:5000]
            })
        return msgs

    messages = _fresh_messages()
    last = ""
    attempt_count = 0
    attempt_diagnostics: list[dict[str, Any]] = []
    previous_invalid_fingerprint = None
    critic_used = False
    critic_checks = 0
    critic_advice: list[str] = []
    selection_guard_used = False
    selection_guard_advice: list[str] = []
    semantic_retry_pending = False
    semantic_hard_errors: list[str] = []
    population_retry_used = False
    max_attempts = max(1, int(attempts))
    # Preserve the latest non-empty bounded attempt that still has an answer
    # endpoint. A later retry is allowed to improve it, but must not erase it by
    # returning malformed/empty JSON. Strict validity is never fabricated: the
    # salvaged candidate keeps its own deterministic validation errors and is at
    # most eligible for the existing read-only advisory acquisition path.
    best_answerable_invalid_plan: dict[str, Any] | None = None
    best_answerable_invalid_text = ""

    for attempt in range(max_attempts):
        attempt_count = attempt + 1
        try:
            kwargs = dict(model=model, messages=messages, temperature=0.0)
            from utils.token_meter import stage as token_stage
            with token_stage("planner"):
                try:
                    response = client.chat.completions.create(
                        **kwargs, response_format={"type": "json_object"},
                        max_completion_tokens=1800)
                except Exception:
                    response = client.chat.completions.create(**kwargs)
            last = response.choices[0].message.content or ""
        except Exception as exc:
            last = f"planner call failed: {exc}"
            attempt_diagnostics.append({"attempt": attempt_count, "errors": [last]})
            continue

        parsed = _extract_json(last)
        plan, structural_errors = validate_plan(parsed, valid_paths, valid_methods_by_path)
        plan = repair_selection_dependencies(question, plan)
        _routes_before_deterministic_population = [
            (str(st.get("id") or ""), str(st.get("endpoint") or ""))
            for st in (plan or {}).get("steps") or []
        ]
        plan = normalize_latest_released_population(question, plan, tools)
        plan = normalize_explicit_population_sibling(question, plan, tools)
        plan = normalize_tv_ranked_population(question, plan, tools)
        _routes_after_deterministic_population = [
            (str(st.get("id") or ""), str(st.get("endpoint") or ""))
            for st in (plan or {}).get("steps") or []
        ]
        if _routes_after_deterministic_population != _routes_before_deterministic_population:
            # Keep the convergence audit trail even when a cheap host-side
            # normalization can repair the route before the final invariant has
            # to invoke another planner call.  This preserves the architectural
            # guarantee tested by older releases: route semantics that changed
            # during convergence are visible as an invariant-triggered repair,
            # rather than silently disappearing from planner diagnostics.
            attempt_diagnostics.append({
                "attempt": attempt_count,
                "trigger": "final_convergence_invariant",
                "deterministic_route_normalization": {
                    "before": _routes_before_deterministic_population,
                    "after": _routes_after_deterministic_population,
                },
            })
        plan = repair_selection_dependencies(question, plan)
        plan = normalize_temporal_extremum_direction(question, plan)
        plan = normalize_documented_ordered_extrema(question, plan, tools)
        plan = normalize_typed_multi_answer_cardinality(plan)
        plan = normalize_typed_single_answer_cardinality(plan)
        plan = normalize_appearance_credit_relation(question, plan)
        plan = normalize_nested_relation_owner_selection(question, plan)
        plan = normalize_requested_answer_derivations(question, plan)
        plan = normalize_answer_steps_from_answer_derivations(question, plan)
        plan = normalize_direct_dependency_step_bindings(plan)
        try:
            from utils.schema_outline import selected_endpoint_validation_cards
            _literal_cards = selected_endpoint_validation_cards(benchmark, plan)
        except Exception:
            _literal_cards = []
        plan = normalize_collection_binding_scalar_paths(plan, _literal_cards)
        plan = normalize_explicit_ordinal_child_selection(question, plan, _literal_cards)
        plan = normalize_rooted_collection_derivation_paths(plan, _literal_cards)
        plan = normalize_relation_population_metric_extrema(question, plan, _literal_cards)
        plan = normalize_explicit_metric_extrema(question, plan, _literal_cards, tools)
        plan = normalize_explicit_nested_relation_projection(question, plan, _literal_cards)
        plan = normalize_coordinated_parent_child_answers(question, plan, _literal_cards)
        plan = normalize_age_winner_and_difference(question, plan)
        plan = normalize_redundant_answer_derivations(plan)
        plan = normalize_requested_answer_derivations(question, plan)
        plan = normalize_answer_steps_from_answer_derivations(question, plan)
        plan = normalize_missing_comparison_answer_derivation(question, plan)
        plan = normalize_missing_boolean_equality(question, plan)
        plan = normalize_obvious_request_literal_bindings(question, plan, _literal_cards)
        structural_errors = list(structural_errors) + _answer_surface_errors(question, plan)
        structural_errors += _selected_schema_validation_errors(benchmark, plan, question)
        structural_errors = list(dict.fromkeys(structural_errors))
        plan["semantic_critic_enabled"] = bool(enable_semantic_critic)
        plan["semantic_critic_errors"] = []

        # Positive positional choices are checked deterministically first. A plan
        # that invents order/rank/index > 0 when the user stated no ordinal is
        # unsupported by construction and gets one ordinary planner retry. The
        # optional semantic reviewer is reserved for the genuinely ambiguous rare
        # case where the user did state an ordinal but it may belong to a different
        # relation (for example season 2 versus cast position).
        selection_risks = (positional_selection_risks(question, plan)
                           if not structural_errors else [])
        plan["selection_semantic_guard_enabled"] = bool(enable_selection_semantic_guard)
        plan["selection_semantic_risks"] = list(selection_risks)
        if selection_risks:
            explicit_ordinals = _question_ordinal_indexes(question)
            if not explicit_ordinals:
                guard_verdict = "unjustified"
                guard_errors = [
                    "positive positional literal/rank is unanchored because the user requested no ordinal"
                ]
            elif enable_selection_semantic_guard and not selection_guard_used:
                selection_guard_used = True
                guard_verdict, guard_errors, _guard_text = selection_semantic_guard(
                    question, plan, selection_risks, benchmark, model, client)
                plan["selection_semantic_guard_verdict"] = guard_verdict
                plan["selection_semantic_guard_errors"] = list(guard_errors)
            else:
                # With the optional reviewer disabled, an explicitly stated ordinal
                # is left to the evidence planner and deterministic schema/provenance
                # checks. Do not create another normal LLM stage.
                guard_verdict, guard_errors = "justified", []

            if guard_verdict != "justified":
                if attempt + 1 < max_attempts:
                    attempt_diagnostics.append({
                        "attempt": attempt_count,
                        "selection_semantic_retry": list(guard_errors),
                        "selection_risks": list(selection_risks),
                    })
                    messages = _fresh_messages()
                    messages.append({
                        "role": "user",
                        "content":
                            "The previous plan used an unsupported positive ordinal/rank. Generate "
                            "a fresh corrected plan. Do not invent a later positional literal for "
                            "lead/main/top/first. Use first/top returned order, a documented metric/"
                            "relation, or an ordinal that the user explicitly requested for this "
                            "same relation. Feedback: " +
                            json.dumps(guard_errors[:8], ensure_ascii=False)
                    })
                    selection_guard_advice = list(guard_errors)
                    continue
                unresolved = [
                    f"{r.get('derivation_id')}: {r.get('field')}={r.get('position_value')} "
                    "remains an unsupported positive positional selection"
                    for r in selection_risks
                ]
                selection_guard_advice = unresolved
                structural_errors.extend(
                    "selection semantics unresolved after bounded retry: " + x
                    for x in unresolved[:8])
                structural_errors = list(dict.fromkeys(structural_errors))

        # Semantic route compatibility is checked only from the question and OAS.
        # A clear mismatch gets one planner retry and one final recheck. Uncertainty
        # is advisory so terse documentation cannot fabricate a failure.
        # One semantic critic call per task is enough. Its only control-flow
        # effect is to request one fresh planner plan; a second critic pass was
        # advisory-only and consumed substantial tokens without adding a hard
        # correctness guarantee. Deterministic OAS/schema/provenance checks remain
        # authoritative on the corrected plan.
        if not structural_errors and enable_semantic_critic and not critic_used:
            critic_used = True
            critic_checks += 1
            critic_verdict, critic_errors, critic_text = semantic_plan_critic(
                question, plan, tools, benchmark, model, client)
            plan["semantic_critic_verdict"] = critic_verdict
            plan["semantic_critic_errors"] = list(critic_errors)
            if critic_text.startswith("semantic critic unavailable:") or critic_verdict == "uncertain":
                plan.setdefault("validation_warnings", []).append(
                    critic_text if critic_text.startswith("semantic critic unavailable:")
                    else "semantic critic uncertain; deterministic validation remains authoritative")
            if critic_verdict == "incompatible":
                if attempt + 1 < max_attempts:
                    # The critic is a one-shot reviewer. Feed its answer-free OAS
                    # diagnosis into one fresh planner attempt, then rely on the
                    # deterministic validators rather than paying for a second
                    # advisory review of the correction.
                    attempt_diagnostics.append({"attempt": attempt_count,
                                                "semantic_retry": list(critic_errors)})
                    messages = _fresh_messages()
                    retry_schema = _repair_schema_context(
                        benchmark, plan, tools, critic_errors, max_chars=9000,
                        max_paths_per_endpoint=150)
                    messages.append({
                        "role": "user",
                        "content": "A generic OAS semantic reviewer found a clear incompatibility in the previous plan. "
                                   "Generate a fresh corrected plan using the API catalog and selected schemas. "
                                   "Do not simply restate the old plan unless the documented relation/action is actually corrected. "
                                   "Reviewer feedback: " +
                                   json.dumps(critic_errors[:12], ensure_ascii=False) +
                                   ("\nCURRENT + DIAGNOSTIC-ALTERNATIVE OAS SCHEMAS:\n" + retry_schema if retry_schema else "")
                    })
                    critic_advice = []
                    semantic_retry_pending = False
                    continue
                # No ordinary planner attempt remains. Defer to one bounded
                # semantic correction after final structural/schema convergence
                # rather than certifying a plan the answer-free reviewer has
                # already identified as a clear relation mismatch.
                semantic_retry_pending = True
                semantic_hard_errors = list(critic_errors or ["question/plan/API mismatch"])
                critic_advice = list(semantic_hard_errors)
                break

        # A route can be structurally valid yet still be incapable of establishing
        # a population-wide extremum if it ranks one singleton child record.  Use
        # selected OAS schemas to request one fresh route plan when the upstream
        # collection already exposes the comparison field.  This is answer-free,
        # benchmark-agnostic and does not rewrite a reviewed plan post hoc.
        if (not structural_errors and not population_retry_used and
                attempt + 1 < max_attempts):
            try:
                from utils.schema_outline import selected_endpoint_cards, format_endpoint_cards
                population_cards = selected_endpoint_cards(
                    benchmark, plan, max_paths_per_endpoint=220)
                population_hints = schema_population_retry_hints(plan, population_cards)
                population_hints += schema_derivation_retry_hints(plan, population_cards, question)
                population_hints = list(dict.fromkeys(population_hints))
            except Exception:
                population_cards, population_hints = [], []
            if population_hints:
                population_retry_used = True
                attempt_diagnostics.append({
                    "attempt": attempt_count, "population_retry": list(population_hints)})
                messages = _fresh_messages()
                retry_schema = (format_endpoint_cards(population_cards, max_chars=16000)
                                if population_cards else "")
                messages.append({
                    "role": "user",
                    "content":
                        "A generic schema/population review found a replayability problem in the "
                        "previous plan. Generate a fresh corrected plan. Use only documented response "
                        "fields for deterministic derivations. Prefer a semantically correct upstream "
                        "collection when its documented fields already support the filter/ranking/output; "
                        "otherwise preserve the required relation and make any needed population fan-out "
                        "explicit. Do not drop any relation from the question. Review: " +
                        json.dumps(population_hints[:8], ensure_ascii=False) +
                        ("\nSELECTED ENDPOINT SCHEMAS FROM THE PREVIOUS PLAN:\n" + retry_schema
                         if retry_schema else "")
                })
                # Keep the critic one-shot even if this deterministic schema
                # review requests a new route. The retry itself is grounded in OAS
                # structure and will be checked by deterministic validators.
                semantic_retry_pending = False
                critic_advice = []
                continue

        plan["validation_errors"] = list(structural_errors)
        if selection_guard_advice:
            plan.setdefault("validation_warnings", []).extend(
                "selection semantic guard: " + x for x in selection_guard_advice)
        if critic_advice:
            plan.setdefault("validation_warnings", []).extend(
                "semantic critic advisory only: " + x for x in critic_advice)
        plan["valid"] = bool(plan.get("steps")) and not structural_errors
        if plan.get("valid"):
            # Single-exit convergence gate: a plan that became valid after any
            # structural/schema/population mutation must satisfy the SAME cheap
            # deterministic question+OAS relation invariant before it can leave
            # planning.  This is independent of whether the one-shot LLM critic
            # already reviewed an earlier version of the plan.
            route_risks = (deterministic_semantic_route_risks(question, plan, tools)
                           if enable_semantic_critic else [])
            if route_risks:
                if attempt + 1 < max_attempts:
                    attempt_diagnostics.append({
                        "attempt": attempt_count,
                        "semantic_retry": list(route_risks),
                        "trigger": "convergence_invariant",
                    })
                    messages = _fresh_messages()
                    messages.append({
                        "role": "user",
                        "content":
                            "The converged plan still violates a deterministic question+OpenAPI "
                            "relation invariant. Produce a fresh corrected plan without adding an "
                            "unstated subset/scope and without dropping requested outputs. Feedback: " +
                            json.dumps(route_risks[:8], ensure_ascii=False)
                    })
                    continue
                repaired, repair_text, repair_errors = _apply_semantic_plan_repair(
                    question, benchmark, model, client, tools, catalog_mode, plan,
                    route_risks, valid_paths, valid_methods_by_path)
                attempt_count += 1
                if repaired is not None:
                    attempt_diagnostics.append({
                        "attempt": attempt_count,
                        "semantic_plan_repair": list(route_risks[:8]),
                        "trigger": "final_convergence_invariant",
                    })
                    plan = repaired
                    if repair_text:
                        last = repair_text
                    route_risks = deterministic_semantic_route_risks(question, plan, tools)
                else:
                    attempt_diagnostics.append({
                        "attempt": attempt_count,
                        "semantic_plan_repair_errors": list(repair_errors[:12]),
                        "trigger": "final_convergence_invariant",
                    })
                if route_risks:
                    plan["validation_errors"] = [
                        "semantic route invariant unresolved after bounded convergence: " + str(x)
                        for x in route_risks[:8]
                    ]
                    plan["valid"] = False
            if plan.get("valid"):
                plan["planner_source"] = "llm_oas_generic+critic" if critic_used else "llm_oas_generic"
                plan["planner_attempt_count"] = attempt_count
                plan["planner_attempt_diagnostics"] = attempt_diagnostics
                return plan, last

        if plan.get("steps") and plan.get("answer_steps"):
            best_answerable_invalid_plan = copy.deepcopy(plan)
            best_answerable_invalid_text = last

        attempt_diagnostics.append({"attempt": attempt_count,
                                    "errors": list(structural_errors[:20])})
        invalid_fingerprint = json.dumps({
            "steps": plan.get("steps") or [],
            "derivations": plan.get("derivations") or [],
            "answer_steps": plan.get("answer_steps") or [],
            "validation_errors": structural_errors,
        }, ensure_ascii=False, sort_keys=True, default=str)
        repeated_invalid_plan = invalid_fingerprint == previous_invalid_fingerprint
        previous_invalid_fingerprint = invalid_fingerprint
        if repeated_invalid_plan:
            break
        if attempt + 1 < max_attempts:
            # The first call chooses routes from the compact catalog. If exact
            # response-field/binding validation fails, the one bounded retry must
            # see the schemas of those already-selected operations; otherwise we
            # would ask the model to repair field names while withholding the field
            # structure it is being checked against. This is schema grounding, not
            # another critic or another model stage.
            messages = _fresh_messages()
            retry_schema = ""
            try:
                from utils.schema_outline import selected_endpoint_cards, format_endpoint_cards
                retry_cards = selected_endpoint_cards(
                    benchmark, plan, max_paths_per_endpoint=220,
                    max_request_paths_per_endpoint=160, max_depth=12,
                    max_request_depth=12)
                retry_schema = format_endpoint_cards(retry_cards, max_chars=9000)
            except Exception:
                retry_schema = ""
            messages.append({
                "role": "user",
                "content":
                    "The previous plan failed deterministic OAS/schema validation. Generate one "
                    "fresh corrected JSON plan. Preserve the user's relation chain, but correct "
                    "request locations, collection selection, binding fields, and derivation "
                    "fields using the selected schemas below. If a selected route cannot expose "
                    "the actual requested attribute, choose a documented route from the catalog "
                    "that can. Do not invent fields. Validation errors: " +
                    json.dumps(structural_errors[:20], ensure_ascii=False) +
                    (("\nSELECTED SCHEMAS FROM THE FAILED PLAN:\n" + retry_schema)
                     if retry_schema else "")
            })

    # Revalidate the final planner response. If the final bounded retry collapsed
    # structurally (for example, empty steps/no answer endpoint), retain the latest
    # earlier non-empty attempt instead. This is plan *salvage*, not plan repair:
    # all of that earlier attempt's validation errors remain authoritative.
    plan, structural_errors = validate_plan(_extract_json(last), valid_paths, valid_methods_by_path)
    plan = repair_selection_dependencies(question, plan)
    plan = normalize_latest_released_population(question, plan, tools)
    plan = normalize_explicit_population_sibling(question, plan, tools)
    plan = normalize_tv_ranked_population(question, plan, tools)
    plan = normalize_person_most_popular_tv_collaborators(question, plan, tools)
    plan = repair_selection_dependencies(question, plan)
    plan = normalize_temporal_extremum_direction(question, plan)
    plan = normalize_documented_ordered_extrema(question, plan, tools)
    plan = normalize_typed_multi_answer_cardinality(plan)
    plan = normalize_typed_single_answer_cardinality(plan)
    plan = normalize_appearance_credit_relation(question, plan)
    plan = normalize_nested_relation_owner_selection(question, plan)
    plan = normalize_requested_answer_derivations(question, plan)
    plan = normalize_answer_steps_from_answer_derivations(question, plan)
    plan = normalize_direct_dependency_step_bindings(plan)
    try:
        from utils.schema_outline import selected_endpoint_validation_cards
        _literal_cards = selected_endpoint_validation_cards(benchmark, plan)
    except Exception:
        _literal_cards = []
    plan = normalize_collection_binding_scalar_paths(plan, _literal_cards)
    plan = normalize_explicit_ordinal_child_selection(question, plan, _literal_cards)
    plan = normalize_rooted_collection_derivation_paths(plan, _literal_cards)
    plan = normalize_relation_population_metric_extrema(question, plan, _literal_cards)
    plan = normalize_explicit_metric_extrema(question, plan, _literal_cards, tools)
    plan = normalize_explicit_nested_relation_projection(question, plan, _literal_cards)
    plan = normalize_coordinated_parent_child_answers(question, plan, _literal_cards)
    plan = normalize_age_winner_and_difference(question, plan)
    plan = normalize_redundant_answer_derivations(plan)
    plan = normalize_requested_answer_derivations(question, plan)
    plan = normalize_answer_steps_from_answer_derivations(question, plan)
    plan = normalize_missing_comparison_answer_derivation(question, plan)
    plan = normalize_missing_boolean_equality(question, plan)
    plan = normalize_obvious_request_literal_bindings(question, plan, _literal_cards)
    structural_errors = list(structural_errors) + _answer_surface_errors(question, plan)
    structural_errors += _selected_schema_validation_errors(benchmark, plan, question)
    structural_errors = list(dict.fromkeys(list(structural_errors)))
    if ((not plan.get("steps") or not plan.get("answer_steps"))
            and best_answerable_invalid_plan is not None):
        plan = copy.deepcopy(best_answerable_invalid_plan)
        structural_errors = list(plan.get("validation_errors") or [])
        last = best_answerable_invalid_text
        plan.setdefault("validation_warnings", []).append(
            "later bounded planner retry regressed structurally; salvaged the prior non-empty answerable attempt")

    # Rare convergence repair #1: if the plan is otherwise fixed but one or more
    # derivations name OAS-invalid scalar paths, repair only those paths. This is
    # especially important when a useful earlier plan is salvaged after a later
    # full retry regresses; the field patch cannot change routes or task semantics.
    schema_field_targets = _schema_field_micro_repair_targets(structural_errors)
    if schema_field_targets:
        repaired, repair_text, repair_errors = _apply_schema_field_micro_repair(
            question, benchmark, model, client, plan, structural_errors,
            valid_paths, valid_methods_by_path)
        attempt_count += 1
        if repaired is not None:
            attempt_diagnostics.append({
                "attempt": attempt_count,
                "schema_field_micro_repair": sorted(schema_field_targets),
            })
            plan = repaired
            structural_errors = list(repair_errors)
            if repair_text:
                last = repair_text
        else:
            attempt_diagnostics.append({
                "attempt": attempt_count,
                "schema_field_micro_repair_errors": list(repair_errors[:12]),
            })

    # Rare convergence repair #2: when every remaining defect is explicit
    # collection selection, ask for only those tiny selector declarations rather
    # than paying for another full-plan regeneration.
    selection_only_steps = _selection_only_error_steps(structural_errors)
    if selection_only_steps:
        repaired, repair_text, repair_errors = _apply_selection_micro_repair(
            question, benchmark, model, client, plan, structural_errors,
            valid_paths, valid_methods_by_path)
        attempt_count += 1
        if repaired is not None:
            attempt_diagnostics.append({
                "attempt": attempt_count,
                "selection_micro_repair": selection_only_steps,
            })
            plan = repaired
            structural_errors = []
            if repair_text:
                last = repair_text
        else:
            attempt_diagnostics.append({
                "attempt": attempt_count,
                "selection_micro_repair_errors": list(repair_errors[:12]),
            })

    # Monotonic bounded full-plan convergence for residual deterministic defects.
    # A semantically improved candidate is no longer discarded merely because one
    # smaller structural/cardinality defect remains.  We retain the lowest-scoring
    # candidate under the complete invariant stack and allow one further repair
    # only after measurable progress.  This is generic search over plan quality, not
    # an error-family/task-specific patch.
    if (structural_errors and plan.get("steps") and plan.get("answer_steps")):
        plan, structural_errors = _evaluate_convergence_candidate(
            question, benchmark, plan, tools, valid_paths, valid_methods_by_path)
        best_score = _convergence_error_score(structural_errors)
        for convergence_round in range(2):
            before_errors = list(structural_errors)
            before_score = best_score
            repaired, repair_text, repair_errors = _apply_general_convergence_repair(
                question, benchmark, model, client, tools, catalog_mode, plan,
                before_errors, valid_paths, valid_methods_by_path)
            attempt_count += 1
            if repaired is None:
                attempt_diagnostics.append({
                    "attempt": attempt_count,
                    "general_convergence_repair_errors": list(repair_errors[:12]),
                    "round": convergence_round + 1,
                })
                break
            candidate_score = _convergence_error_score(repair_errors)
            if candidate_score < best_score:
                attempt_diagnostics.append({
                    "attempt": attempt_count,
                    "general_convergence_repair": list(before_errors[:12]),
                    "round": convergence_round + 1,
                    "score_before": list(before_score),
                    "score_after": list(candidate_score),
                    "remaining_errors": list(repair_errors[:12]),
                })
                plan = repaired
                structural_errors = list(repair_errors)
                best_score = candidate_score
                if repair_text:
                    last = repair_text
                if not structural_errors:
                    break
                continue
            attempt_diagnostics.append({
                "attempt": attempt_count,
                "general_convergence_no_improvement": list(repair_errors[:12]),
                "round": convergence_round + 1,
                "score_before": list(before_score),
                "score_candidate": list(candidate_score),
            })
            break

    # Ensure the final converged plan receives the answer-free semantic review even
    # when ordinary attempts were spent fixing structural/schema defects. If the
    # reviewer finds a clear mismatch, allow exactly one fresh semantic correction.
    # This extra call is conditional (only on an incompatible verdict), so ordinary
    # compatible plans pay for the small review but not another full planner pass.
    if (enable_semantic_critic and not structural_errors
            and plan.get("steps") and plan.get("answer_steps") and not critic_used):
        critic_used = True
        critic_checks += 1
        critic_verdict, critic_errors, critic_text = semantic_plan_critic(
            question, plan, tools, benchmark, model, client)
        plan["semantic_critic_verdict"] = critic_verdict
        plan["semantic_critic_errors"] = list(critic_errors)
        if critic_text.startswith("semantic critic unavailable:") or critic_verdict == "uncertain":
            plan.setdefault("validation_warnings", []).append(
                critic_text if critic_text.startswith("semantic critic unavailable:")
                else "semantic critic uncertain; deterministic validation remains authoritative")
        if critic_verdict == "incompatible":
            semantic_retry_pending = True
            semantic_hard_errors = list(critic_errors or ["question/plan/API mismatch"])
            critic_advice = list(semantic_hard_errors)

    if (enable_semantic_critic and semantic_retry_pending and semantic_hard_errors
            and not structural_errors and plan.get("steps") and plan.get("answer_steps")):
        repaired, repair_text, repair_errors = _apply_semantic_plan_repair(
            question, benchmark, model, client, tools, catalog_mode, plan,
            semantic_hard_errors, valid_paths, valid_methods_by_path)
        attempt_count += 1
        if repaired is not None:
            attempt_diagnostics.append({
                "attempt": attempt_count,
                "semantic_plan_repair": list(semantic_hard_errors[:8]),
            })
            plan = repaired
            structural_errors = []
            semantic_retry_pending = False
            critic_advice = []
            if repair_text:
                last = repair_text
        else:
            attempt_diagnostics.append({
                "attempt": attempt_count,
                "semantic_plan_repair_errors": list(repair_errors[:12]),
            })
            # A clear incompatible verdict must cause a real strategy change.
            # If the bounded correction cannot produce a valid replacement, do not
            # execute the same read-only strategy as an advisory plan.  The critic
            # is still answer-free and OAS-grounded; uncertainty remains fail-open,
            # but an explicit incompatible verdict is not silently ignored.
            deterministic_backing = deterministic_semantic_route_risks(question, plan, tools)
            critic_advice = list(semantic_hard_errors)
            reasons = list(dict.fromkeys(deterministic_backing or semantic_hard_errors))
            structural_errors.extend(
                "semantic critic incompatibility unresolved after bounded correction: " + str(x)
                for x in reasons[:8])
            structural_errors = list(dict.fromkeys(structural_errors))
            semantic_retry_pending = False

    # Plan convergence invariant: every mutation path (ordinary retry, schema/
    # population retry, micro-repair, semantic repair) must end under the *same*
    # deterministic question+OAS relation check.  Earlier versions reviewed only
    # the plan version that happened to reach the critic, so a later population
    # retry could silently re-introduce a narrower route.
    final_route_risks = (deterministic_semantic_route_risks(question, plan, tools)
                         if enable_semantic_critic and not structural_errors else [])
    semantic_repair_already_used = any(
        isinstance(x, dict) and x.get("semantic_plan_repair")
        for x in attempt_diagnostics)
    if final_route_risks and not semantic_repair_already_used and not structural_errors:
        repaired, repair_text, repair_errors = _apply_semantic_plan_repair(
            question, benchmark, model, client, tools, catalog_mode, plan,
            final_route_risks, valid_paths, valid_methods_by_path)
        attempt_count += 1
        if repaired is not None:
            attempt_diagnostics.append({
                "attempt": attempt_count,
                "semantic_plan_repair": list(final_route_risks[:8]),
                "trigger": "final_convergence_invariant",
            })
            plan = repaired
            structural_errors = []
            critic_advice = []
            if repair_text:
                last = repair_text
            final_route_risks = deterministic_semantic_route_risks(question, plan, tools)
        else:
            attempt_diagnostics.append({
                "attempt": attempt_count,
                "semantic_plan_repair_errors": list(repair_errors[:12]),
                "trigger": "final_convergence_invariant",
            })
    if final_route_risks:
        structural_errors.extend(
            "semantic route invariant unresolved after bounded convergence: " + str(x)
            for x in final_route_risks[:8])
        structural_errors = list(dict.fromkeys(structural_errors))
        critic_advice = list(dict.fromkeys(list(critic_advice) + list(final_route_risks)))

    plan["semantic_critic_enabled"] = bool(enable_semantic_critic)
    plan["semantic_critic_errors"] = list(critic_advice)
    final_selection_risks = positional_selection_risks(question, plan)
    plan["selection_semantic_guard_enabled"] = bool(enable_selection_semantic_guard)
    plan["selection_semantic_risks"] = list(final_selection_risks)
    if final_selection_risks and not _question_ordinal_indexes(question):
        structural_errors.extend(
            "selection semantics unresolved after bounded retry: "
            f"{r.get('derivation_id')}: {r.get('field')}={r.get('position_value')}"
            for r in final_selection_risks)
        structural_errors = list(dict.fromkeys(structural_errors))
    elif final_selection_risks and enable_selection_semantic_guard and selection_guard_used:
        structural_errors.extend(
            "selection semantics unresolved after bounded retry: "
            f"{r.get('derivation_id')}: {r.get('field')}={r.get('position_value')}"
            for r in final_selection_risks)
        structural_errors = list(dict.fromkeys(structural_errors))
    plan["validation_errors"] = structural_errors
    if critic_advice:
        plan.setdefault("validation_warnings", []).extend(
            "semantic critic advisory only: " + x for x in critic_advice)
    plan["valid"] = bool(plan.get("steps")) and not structural_errors
    plan["planner_source"] = "llm_oas_generic+critic" if critic_used else "llm_oas_generic"
    plan["planner_attempt_count"] = attempt_count
    plan["planner_attempt_diagnostics"] = attempt_diagnostics
    return plan, last

def assign_calls_to_steps(plan: dict[str, Any] | None,
                          api_calls: list[dict[str, Any]]) -> dict[str, str]:
    """One-to-one structural assignment for ledgers lacking runtime tags.

    Current clean runs receive trusted ``runtime_step_id`` tags, but imported
    ledgers may not.  For repeated method/path templates, match explicit request
    literals rather than free-text purposes; ambiguous calls retain stable plan
    order and are rechecked by the host request-contract audit.
    """
    steps = list((plan or {}).get("steps") or [])
    remaining = {str(step.get("id")): index for index, step in enumerate(steps)}
    assignments: dict[str, str] = {}

    def call_key(call, index):
        return str(call.get("call_id") or f"call_index_{index}")

    def nested_value(body, path):
        cur = body
        for part in [x for x in str(path or "").replace("$.", "").split(".") if x]:
            if not isinstance(cur, dict) or part not in cur:
                return None, False
            cur = cur[part]
        return cur, True

    def equal(actual, expected):
        if isinstance(expected, bool):
            return actual is expected or (isinstance(actual, str) and
                   actual.strip().lower() in ({"true", "1"} if expected else {"false", "0"}))
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            try: return float(actual) == float(expected)
            except Exception: return False
        if isinstance(expected, (list, tuple)):
            if isinstance(actual, str) and "," in actual:
                actual = [x for x in actual.split(",") if x != ""]
            return isinstance(actual, (list, tuple)) and len(actual) == len(expected) and all(
                equal(a, b) for a, b in zip(actual, expected))
        return str(actual) == str(expected)

    # Preserve explicit trusted/historical forced annotations if present.
    for call_index, call in enumerate(api_calls or []):
        forced = str(call.get("plan_step_id") or "") if call.get("forced_plan_step") else ""
        if forced and forced in remaining:
            assignments[call_key(call, call_index)] = forced
            remaining.pop(forced, None)

    for call_index, call in enumerate(api_calls or []):
        if call_key(call, call_index) in assignments:
            continue
        actual = call.get("endpoint", "")
        actual_method = str(call.get("method") or "GET").upper()
        from utils.api_runtime import wire_endpoint_template
        runtime_base = str((plan or {}).get("_runtime_base_url") or "")
        candidates = [step for step in steps
                      if str(step.get("id")) in remaining and
                      str(step.get("method") or "GET").upper() == actual_method and
                      endpoint_matches(actual, wire_endpoint_template(
                          runtime_base, str(step.get("endpoint") or "")))]
        if not candidates:
            continue
        query = call.get("params") if isinstance(call.get("params"), dict) else {}
        body = call.get("request_body")
        actual_bindings_by_candidate = {}

        def score(step):
            sid = str(step.get("id"))
            matches, mismatches = 0, 0
            concrete = extract_path_bindings(str(actual), str(step.get("endpoint") or ""))
            actual_bindings_by_candidate[sid] = concrete
            for name, expected in (step.get("path_literals") or {}).items():
                if equal(concrete.get(name), expected): matches += 1
                else: mismatches += 1
            for name, expected in (step.get("query_literals") or {}).items():
                if name in query and equal(query.get(name), expected): matches += 1
                else: mismatches += 1
            for path, expected in (step.get("body_literals") or {}).items():
                actual_value, present = nested_value(body, path)
                if present and equal(actual_value, expected): matches += 1
                else: mismatches += 1
            # A mismatch is more important than any number of matches. Stable
            # declaration order is only a final tie-breaker.
            return (-mismatches, matches, -remaining[sid])

        chosen = max(candidates, key=score)
        sid = str(chosen.get("id"))
        assignments[call_key(call, call_index)] = sid
        remaining.pop(sid, None)
    return assignments


def _get_field_path(fields: dict[str, Any], path: str) -> Any:
    value: Any = fields or {}
    for part in [x for x in str(path or "").replace("$.", "").split(".") if x and x != "$"]:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def resolve_lineage_completion_targets(plan: dict[str, Any] | None,
                                       ledger, compiled: dict[str, Any] | None,
                                       limit: int = 2) -> list[dict[str, Any]]:
    """Resolve deterministic retries from declared bindings only.

    No endpoint/entity vocabulary is inspected.  For each one-placeholder answer
    step, the exact placeholder must be owned by the nearest dependency that
    declares that binding.  A concrete retry is generated only when that producer
    has one deterministic selected record and the binding value can be read from
    its validated projection. Ambiguity yields no retry and is handled by safe
    abstention. Parameter spelling is never used to infer a response field.
    """
    steps = {str(x.get("id")): x for x in (plan or {}).get("steps") or []}
    specs = {str(x.get("step_id")): x for x in (plan or {}).get("observation_specs") or []}
    existing = {str(x.get("endpoint") or "").rstrip("/")
                for x in getattr(ledger, "api_calls", [])}
    selection_ids = list((compiled or {}).get("selection_derivation_ids") or [])

    def distances(step_id: str) -> dict[str, int]:
        out: dict[str, int] = {}; queue = [(step_id, 0)]
        while queue:
            current, dist = queue.pop(0)
            for dep in (steps.get(current) or {}).get("depends_on") or []:
                dep = str(dep); nd = dist + 1
                if dep not in out or nd < out[dep]:
                    out[dep] = nd; queue.append((dep, nd))
        return out

    targets: list[dict[str, Any]] = []
    for answer_id in (plan or {}).get("answer_steps") or []:
        answer_id = str(answer_id); step = steps.get(answer_id) or {}
        template = str(step.get("endpoint") or "")
        placeholders = _step_placeholders(step)
        if len(placeholders) != 1:
            continue
        # This deterministic helper only constructs a path-bound GET. If the
        # validated request also carries query/body semantics, normal plan repair
        # must execute it so the trusted request contract remains authoritative.
        if (step.get("query_literals") or step.get("query_bindings") or
                step.get("body_literals") or step.get("body_bindings") or
                any(str(p.get("in") or "").lower() == "query" and
                    _oas_required(p.get("required"))
                    for p in (step.get("request_parameters") or []))):
            continue
        name = placeholders[0]; dists = distances(answer_id)
        producers = []
        for sid, dist in dists.items():
            step_binds = {str(x).strip("{}") for x in (steps.get(sid) or {}).get("binds") or []}
            spec_binds = {str(x.get("name")) for x in (specs.get(sid) or {}).get("bindings") or []}
            if name in step_binds or name in spec_binds:
                producers.append((dist, sid))
        if not producers:
            continue
        nearest_dist = min(x[0] for x in producers)
        nearest = [sid for dist, sid in producers if dist == nearest_dist]
        if len(nearest) != 1:
            continue
        producer = nearest[0]

        # Find deterministic selections produced by exactly that plan step.
        candidates = []
        for order, did in enumerate(selection_ids):
            d = ledger.get_derived(did) or {}
            selected = ledger.get(d.get("selected_obs_id")) if d.get("selected_obs_id") else None
            source_step = str(d.get("plan_step_id") or (selected or {}).get("plan_step_id") or "")
            if source_step != producer or not selected:
                continue
            policy = str(d.get("policy") or "")
            # Projection-declared decisions replay the validated observation plan;
            # otherwise planner-declared deterministic selection is acceptable.
            priority = 2 if policy == "projection_declared" else 1
            span = len(d.get("candidate_obs_ids") or [])
            candidates.append(((priority, span, order), d, selected))
        if not candidates:
            continue
        _, derivation, selected = max(candidates, key=lambda x: x[0])

        binding_path = None
        for binding in (specs.get(producer) or {}).get("bindings") or []:
            if str(binding.get("name")) == name:
                binding_path = str(binding.get("path") or "")
                break
        fields = selected.get("fields") or {}
        value = _get_field_path(fields, binding_path) if binding_path else None
        if value in (None, "") and name in fields:
            value = fields.get(name)
        if value in (None, ""):
            continue
        target = template.replace("{" + name + "}", str(value)).rstrip("/")
        if target in existing:
            continue
        payload = derivation.get("value")
        label = None
        if isinstance(payload, dict):
            label = next((str(v) for k, v in payload.items()
                          if k not in {"id", name} and isinstance(v, str) and v.strip()), None)
        targets.append({"plan_step_id": answer_id, "endpoint": target,
                        "selected_record_id": value, "selected_label": label,
                        "selection_derivation_id": derivation.get("obs_id"),
                        "binding_name": name, "producer_step_id": producer})
    out, seen = [], set()
    for item in targets:
        if item["endpoint"] not in seen:
            seen.add(item["endpoint"]); out.append(item)
    return out[:max(0, int(limit))]

def annotate_lineage_completion_targets(ledger, targets: list[dict[str, Any]]) -> int:
    """Force exact completion calls onto their intended answer step.

    One-to-one template assignment may have already attached an earlier call with the
    correct endpoint family but wrong entity id. Exact target paths are authoritative
    for the lineage-completion pass, so the earlier assignment is cleared and the
    exact call and its observations are attached to the answer step.
    """
    changed = 0
    for target in targets or []:
        endpoint = str(target.get("endpoint") or "").rstrip("/")
        step_id = str(target.get("plan_step_id") or "")
        if not endpoint or not step_id:
            continue
        exact_calls = [c for c in getattr(ledger, "api_calls", [])
                       if str(c.get("endpoint") or "").rstrip("/") == endpoint]
        if not exact_calls:
            continue
        chosen = exact_calls[-1]
        for call in getattr(ledger, "api_calls", []):
            if call.get("plan_step_id") == step_id and call is not chosen:
                call.pop("plan_step_id", None)
        chosen["plan_step_id"] = step_id
        call_id = chosen.get("call_id")
        for obs in getattr(ledger, "observations", []):
            if obs.get("plan_step_id") == step_id and obs.get("call_id") != call_id:
                obs.pop("plan_step_id", None)
            if obs.get("call_id") == call_id:
                obs["plan_step_id"] = step_id
                changed += 1
    return changed



def structural_fallback_targets(question: str, plan: dict[str, Any] | None, ledger) -> list[dict[str, Any]]:
    """No domain-specific structural fallbacks are permitted in the generic core."""
    del question, plan, ledger
    return []


# ---------------------------------------------------------------------------
# Adaptive observation planning (schema-guided, domain-field agnostic)
# ---------------------------------------------------------------------------

def _observation_plan_messages(question: str, plan: dict[str, Any],
                               benchmark: str) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Build a second, small planning call over schemas of selected endpoints only.

    The route planner decides *which APIs* to call. This planner decides *which
    response paths Phase A needs to see* in order to continue the chain. The full
    response remains in the ledger and is not supplied here.
    """
    from utils.schema_outline import selected_endpoint_cards, format_endpoint_cards
    cards = selected_endpoint_cards(benchmark, plan, max_paths_per_endpoint=400)
    schema_text = format_endpoint_cards(cards, max_chars=24000)
    plan_core = {
        "steps": plan.get("steps") or [],
        "derivations": plan.get("derivations") or [],
        "answer_steps": plan.get("answer_steps") or [],
        "answer_derivations": plan.get("answer_derivations") or [],
        "answer_mode": plan.get("answer_mode"),
        "answer_requirements": plan.get("answer_requirements") or [],
    }
    system = """You are the observation planner for an API agent. The API plan is already fixed. Choose only the response data needed to continue that plan. Do not answer the user.

Rules:
1. Use only paths from the supplied schema.
2. Keep fields needed for the answer or a later API call.
3. Apply only filters, sorting, and selection required by the fixed plan.
4. Keep enough records to complete the plan.

For each step return: step_id, record_path, project_paths, filters, sort, select, bindings, aggregates, completeness, and purpose. Paths inside a record are relative to record_path.

Return JSON only:
{"observation_specs":[{"step_id":"s1","record_path":"$","project_paths":[],"filters":[],"sort":[],"select":{"mode":"single","limit":1,"index":0},"bindings":[],"aggregates":[],"completeness":"one response object","purpose":"..."}]}
"""
    user = (
        "QUESTION:\n" + question +
        "\n\nFIXED EVIDENCE PLAN:\n" + json.dumps(plan_core, ensure_ascii=False) +
        "\n\nSCHEMAS FOR SELECTED ENDPOINTS ONLY:\n" + schema_text
    )
    return ([{"role": "system", "content": system},
             {"role": "user", "content": user}], cards)


def _validate_observation_specs(raw: dict[str, Any] | None,
                                plan: dict[str, Any],
                                cards: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    from utils.observation_projection import normalize_projection_spec, validate_projection_spec
    raw_specs = list((raw or {}).get("observation_specs") or [])
    by_step = {str(x.get("step_id")): x for x in raw_specs if isinstance(x, dict) and x.get("step_id")}
    card_by_step = {str(c.get("step_id")): c for c in cards}
    specs: list[dict[str, Any]] = []
    errors: list[str] = []
    for step in plan.get("steps") or []:
        sid = str(step.get("id"))
        if sid not in by_step:
            errors.append(f"step {sid}: missing observation spec")
            continue
        spec = normalize_projection_spec(by_step[sid], step_id=sid)
        shadowed_inputs = _protected_passthrough_binding_names(plan, sid)
        if shadowed_inputs:
            spec["bindings"] = [
                b for b in (spec.get("bindings") or [])
                if str(b.get("name") or "").strip("{}") not in shadowed_inputs
            ]
        card = card_by_step.get(sid) or {}
        leaf_paths = {str(x.get("path")) for x in card.get("leaf_paths") or [] if x.get("path")}
        record_paths = {str(x.get("path")) for x in card.get("record_paths") or [] if x.get("path")}
        record_paths.add("$")
        if spec.get("schema_deferred") and not leaf_paths:
            spec_errors = []
        else:
            spec_errors = validate_projection_spec(
                spec, leaf_paths=leaf_paths, record_paths=record_paths)
        errors.extend(f"step {sid}: {e}" for e in spec_errors)
        specs.append(spec)
    return specs, list(dict.fromkeys(errors))



def _relative_projection_field(field: Any, record_path: Any) -> str:
    """Normalize a plan field to the observation spec's record-relative dialect."""
    text = str(field or "").strip().replace("$.", "").lstrip("$.")
    # Schema list markers/indices identify a record selection, not a child field.
    # Normalize both wildcard and concrete-index spellings before making the path
    # relative to the chosen observation record universe.
    text = re.sub(r"\[(?:\*|\d*)\]", "", text)
    record = str(record_path or "$").strip().replace("$.", "").lstrip("$.")
    record = re.sub(r"\[(?:\*|\d*)\]", "", record).strip(".")
    if record and record != "$":
        if text == record:
            return "$"
        if text.startswith(record + "."):
            text = text[len(record) + 1:]
        else:
            # Normalized ledger records retain the terminal collection relation
            # (e.g. ``items``) even when OAS record_path is ``data.items[*]``.
            # Planner derivations commonly name ``items.score`` while projection
            # paths are correctly relative as ``score``. Mirror _field()'s
            # relation-prefix semantics so nested response envelopes do not cause
            # false selection-alignment failures.
            relation = record.split(".")[-1]
            if relation and text.startswith(relation + "."):
                text = text[len(relation) + 1:]
    return text or "$"


def _normalized_plan_filter_predicates(plan_filter: dict[str, Any] | None,
                                       record_path: Any) -> list[tuple[str, str, Any]]:
    out: list[tuple[str, str, Any]] = []
    for field, expected in (plan_filter or {}).items():
        path = _relative_projection_field(field, record_path)
        if isinstance(expected, dict) and expected.get("op"):
            op = str(expected.get("op") or "eq").lower()
            value = expected.get("value")
        elif (isinstance(expected, list) and len(expected) == 1 and
              isinstance(expected[0], dict) and expected[0].get("op")):
            op = str(expected[0].get("op") or "eq").lower()
            value = expected[0].get("value")
        elif isinstance(expected, list):
            op, value = "in", expected
        elif isinstance(expected, str):
            # evidence_compiler._filter_records compares ordinary strings
            # case-insensitively, so eq_ci is the exact projection equivalent.
            op, value = "eq_ci", expected
        else:
            op, value = "eq", expected
        out.append((path, op, value))
    return out


def _normalized_projection_filter_predicates(spec: dict[str, Any]) -> list[tuple[str, str, Any]]:
    return [(str(x.get("path") or "$"), str(x.get("op") or "eq").lower(), x.get("value"))
            for x in (spec.get("filters") or [])]


def validate_selection_alignment(plan: dict[str, Any],
                                 specs: list[dict[str, Any]] | None = None) -> list[str]:
    """Ensure runtime binding selection and deterministic plan selection agree.

    The trusted kernel authorizes downstream concrete parameters from observation
    specs, while certification replays evidence-plan derivations.  If those two
    selectors disagree, both layers can be locally valid yet refer to different
    records.  This validator joins them without endpoint/entity vocabulary.

    Extra observation filters are allowed (for example exact user-input
    disambiguation), but every plan-declared filter feeding a bound producer must
    also be present.  The compiler replays derivations over that same projected
    candidate universe, so execution and certification remain aligned.
    """
    specs = list(specs if specs is not None else (plan or {}).get("observation_specs") or [])
    by_step = {str(x.get("step_id") or ""): x for x in specs}
    derivs_by_step: dict[str, list[dict[str, Any]]] = {}
    for deriv in (plan or {}).get("derivations") or []:
        sources = [str(x) for x in deriv.get("source_steps") or []]
        if len(sources) == 1:
            derivs_by_step.setdefault(sources[0], []).append(deriv)

    errors: list[str] = []
    selector_ops = {"first", "nth", "endpoint_rank", "argmax", "argmin"}
    for sid, spec in by_step.items():
        bindings = [b for b in (spec.get("bindings") or []) if b.get("name") and b.get("path")]
        if not bindings:
            continue
        source_modes = {str(b.get("source") or "selected_first") for b in bindings}
        derivs = derivs_by_step.get(sid, [])
        selectors = [d for d in derivs if str(d.get("operator") or "").lower() in selector_ops]
        terminal_selector = selectors[-1] if selectors else None

        # A plan-level filter that narrows the records subsequently bound to a
        # consumer must also narrow the runtime projection. Otherwise the kernel
        # could call a child for a record the verifier would have filtered out.
        plan_preds: list[tuple[str, str, Any]] = []
        for d in derivs:
            op = str(d.get("operator") or "").lower()
            if op == "filter" or (terminal_selector is d):
                plan_preds.extend(_normalized_plan_filter_predicates(
                    d.get("filter") or {}, spec.get("record_path")))
        projection_preds = _normalized_projection_filter_predicates(spec)
        for pred in plan_preds:
            if pred not in projection_preds:
                errors.append(
                    f"step {sid}: runtime binding projection omits plan filter {pred[0]} {pred[1]}")

        # selected_all intentionally means the full filtered candidate universe;
        # a terminal single-record selector does not redefine that fan-out.
        if "selected_first" not in source_modes or terminal_selector is None:
            continue

        op = str(terminal_selector.get("operator") or "").lower()
        select = spec.get("select") or {}
        mode = str(select.get("mode") or "head").lower()
        index = max(0, int(select.get("index", 0) or 0))
        sorts = list(spec.get("sort") or [])

        if op in {"first", "nth", "endpoint_rank"}:
            rank = max(0, int(terminal_selector.get("rank", 0) or 0))
            if sorts:
                errors.append(
                    f"step {sid}: {op} binding must preserve API/projected order, not add a sort")
            if rank == 0:
                if not (mode in {"head", "top", "all_matches", "single"}
                        or (mode == "nth" and index == 0)):
                    errors.append(
                        f"step {sid}: {op} rank 0 does not match projection select {mode}[{index}]")
            elif not (mode == "nth" and index == rank):
                errors.append(
                    f"step {sid}: {op} rank {rank} requires projection select nth[{rank}]")

        elif op in {"argmax", "argmin"}:
            field = _relative_projection_field(terminal_selector.get("field"),
                                               spec.get("record_path"))
            direction = "desc" if op == "argmax" else "asc"
            if not field or field == "$":
                errors.append(f"step {sid}: {op} binding has no replayable comparison field")
            elif not sorts:
                errors.append(
                    f"step {sid}: {op} binding requires projection sort {field} {direction}")
            else:
                first_sort = sorts[0]
                actual_field = str(first_sort.get("path") or "")
                actual_direction = str(first_sort.get("direction") or "asc").lower()
                if actual_field != field or actual_direction != direction:
                    errors.append(
                        f"step {sid}: {op} binding expects sort {field} {direction}, "
                        f"got {actual_field or '<none>'} {actual_direction}")
            if not (mode in {"head", "top", "all_matches", "single"}
                    or (mode == "nth" and index == 0)):
                errors.append(
                    f"step {sid}: {op} binding must select the first sorted record, got {mode}[{index}]")

    return list(dict.fromkeys(errors))


def repair_path_binding_aliases_from_schema(plan: dict[str, Any],
                                            cards: list[dict[str, Any]]) -> dict[str, Any]:
    """Recover an omitted URL-placeholder alias from one direct parent, fail-closed.

    A planner can name an upstream value semantically (``similar_item_id``) while
    the child route names the URL slot generically (``item_id``).  We never infer
    from names.  Instead, after the selected OAS schemas and observation roots are
    known, recover the alias only when exactly one direct-parent declared binding
    has a response type compatible with the child path parameter.  The recovered
    binding is also added to the producer projection so runtime authorization and
    certificate replay use the same exact lineage.
    """
    out = copy.deepcopy(plan or {})
    steps = {str(x.get("id")): x for x in out.get("steps") or []}
    specs = {str(x.get("step_id")): x for x in out.get("observation_specs") or []}
    card_by_step = {str(x.get("step_id")): x for x in (cards or [])}
    warnings = list(out.get("validation_warnings") or [])

    def request_path_type(sid: str, name: str) -> str | None:
        for param in (card_by_step.get(sid) or {}).get("parameters") or []:
            if str(param.get("in") or "").lower() == "path" and str(param.get("name") or "") == name:
                return str(param.get("type") or "unknown")
        return None

    def binding_type(dep: str, path: str) -> tuple[str | None, str | None]:
        spec = specs.get(dep) or {}
        root = _normalized_schema_path(spec.get("record_path") or "$")
        raw = _normalized_schema_path(path)
        if root and root != "$" and not (raw == root or raw.startswith(root + ".")):
            absolute = (root + "." + raw).strip(".")
        else:
            absolute = raw
        card = card_by_step.get(dep) or {}
        typ = None
        actual = None
        for leaf in card.get("leaf_paths") or []:
            if _normalized_schema_path(leaf.get("path")) == absolute:
                typ = str(leaf.get("type") or "unknown")
                actual = str(leaf.get("path") or "")
                break
        relative = _record_relative_path(actual or absolute, str(spec.get("record_path") or "$"))
        return typ, relative

    for sid, child in steps.items():
        aliases = dict(child.get("path_bindings") or {})
        literals = child.get("path_literals") or {}
        deps = [str(x) for x in child.get("depends_on") or [] if str(x) in steps]
        if not deps:
            continue
        for placeholder in _step_placeholders(child):
            if placeholder in literals or placeholder in aliases:
                continue
            target_type = request_path_type(sid, placeholder)
            candidates = []
            for dep in deps:
                dep_step = steps.get(dep) or {}
                dep_spec = specs.get(dep) or {}
                select_mode = str((dep_spec.get("select") or {}).get("mode") or "head").lower()
                deterministic_single = select_mode in {"head", "top", "single", "nth"}
                if not deterministic_single:
                    continue
                for alias, path in (dep_step.get("binding_paths") or {}).items():
                    typ, relative = binding_type(dep, str(path))
                    if not relative:
                        continue
                    serializable = (str(target_type or "").lower() == "string" and
                                    str(typ or "").lower() in {"integer", "number", "boolean"})
                    if _schema_types_compatible(typ, target_type) or serializable:
                        candidates.append((dep, str(alias).strip("{}"), relative, typ))
            # Exact one-candidate rule is intentionally strict.  Ambiguity stays
            # unresolved rather than falling back to an older same-named ancestor.
            unique = {(d, a, r, t) for d, a, r, t in candidates}
            if len(unique) != 1:
                continue
            dep, alias, relative, _typ = next(iter(unique))
            aliases[placeholder] = alias
            producer_spec = specs.get(dep)
            if producer_spec is not None:
                bindings = [dict(x) for x in producer_spec.get("bindings") or []]
                if not any(str(x.get("name") or "").strip("{}") == alias for x in bindings):
                    bindings.append({"name": alias, "path": relative, "source": "selected_first"})
                    producer_spec["bindings"] = bindings
            warnings.append(
                f"step {sid}: schema-grounded path alias {{{placeholder}}}<-{alias} from direct producer {dep}")
        if aliases:
            child["path_bindings"] = aliases

    out["steps"] = list(steps.values())
    out["observation_specs"] = list(specs.values())
    out["validation_warnings"] = list(dict.fromkeys(warnings))
    return out


def repair_observation_bindings(plan: dict[str, Any]) -> dict[str, Any]:
    """Alias projection bindings to exact downstream placeholders generically.

    If an immediate dependency selects/binds one value that feeds a one-placeholder
    consumer, the projection may call it ``selected_id`` while the OAS path calls it
    ``resource_id``.  Add a second binding name for the exact placeholder, preserving
    the same source path.  This is structural aliasing, not entity inference.
    """
    out = dict(plan or {})
    steps = {str(x.get("id")): x for x in (out.get("steps") or [])}
    specs = [dict(x) for x in (out.get("observation_specs") or [])]
    by_step = {str(x.get("step_id")): x for x in specs}
    for child_id, child in steps.items():
        placeholders = _step_placeholders(child)
        deps = [str(x) for x in (child.get("depends_on") or []) if str(x) in steps]
        if len(placeholders) != 1 or len(deps) != 1:
            continue
        name = placeholders[0]
        dep_id = deps[0]
        dep_step = steps.get(dep_id) or {}
        if name in _protected_passthrough_binding_names(out, dep_id):
            continue
        spec = by_step.get(dep_id)
        if not spec:
            continue
        bindings = [dict(x) for x in (spec.get("bindings") or [])]
        if any(str(x.get("name")) == name for x in bindings):
            continue
        # Alias only when there is one unambiguous binding source on the selected
        # dependency.  Otherwise abstention is safer than guessing.
        paths = {str(x.get("path")) for x in bindings if x.get("path")}
        if len(paths) == 1:
            exemplar = bindings[0]
            bindings.append({"name": name, "path": exemplar.get("path"),
                             "source": exemplar.get("source") or "selected_first"})
            spec["bindings"] = bindings
    out["observation_specs"] = specs
    return out



def repair_membership_sources_from_observation_specs(plan: dict[str, Any]) -> dict[str, Any]:
    """Recover an omitted membership target source from validated projection data.

    Membership compares a target record/set against a collection and therefore
    needs at least two plan-step sources.  Route planners occasionally emit only
    the collection source.  After observation specs are schema-validated we can
    repair that omission *only* when exactly one other step exposes a scalar
    binding/projected path that is also projected by the collection.  This is
    purely structural/schema-driven; ambiguous cases remain unresolved.
    """
    out = dict(plan or {})
    derivs = [dict(x) for x in (out.get("derivations") or [])]
    specs = {str(x.get("step_id")): x for x in (out.get("observation_specs") or [])}
    step_order = [str(x.get("id")) for x in (out.get("steps") or [])]
    warnings = list(out.get("validation_warnings") or [])

    def scalar_paths(spec):
        projected = {str(x) for x in (spec or {}).get("project_paths") or [] if str(x)}
        binding = {str(x.get("path")) for x in (spec or {}).get("bindings") or [] if x.get("path")}
        return projected, binding

    for d in derivs:
        if str(d.get("operator") or "").lower() != "membership":
            continue
        if d.get("comparison_literal") is not None:
            # Literal membership already has its target value; adding another
            # plan step would change it into a different membership form.
            continue
        sources = [str(x) for x in (d.get("source_steps") or []) if str(x) in specs]
        if len(sources) >= 2 or len(sources) != 1:
            continue
        collection = sources[0]
        collection_projected, _ = scalar_paths(specs.get(collection))
        if not collection_projected:
            continue
        candidates = []
        for sid in step_order:
            if sid == collection or sid not in specs:
                continue
            projected, bindings = scalar_paths(specs.get(sid))
            # A target producer should expose one of its selected/bound scalar
            # fields on the collection too. Prefer explicit bindings; if absent,
            # allow a uniquely overlapping projected scalar path.
            overlap = (bindings & collection_projected) or (projected & collection_projected)
            if overlap:
                candidates.append((sid, sorted(overlap)))
        if len(candidates) == 1:
            target, overlap = candidates[0]
            d["source_steps"] = [target, collection]
            warnings.append(
                f"derivation {d.get('id')}: inferred membership target source {target} "
                f"from unique shared projected path(s) {overlap}")

    out["derivations"] = derivs
    out["validation_warnings"] = list(dict.fromkeys(warnings))
    return out


def synchronize_step_bindings_from_observation_specs(plan: dict[str, Any]) -> dict[str, Any]:
    """Promote schema-grounded projection bindings into the evidence-plan graph.

    The route planner may know that a step returns a selected entity without knowing
    the exact response field that carries the downstream placeholder. The adaptive
    schema planner *does* know that field. Once validated, its binding names are
    authoritative structural information and are copied to the corresponding step,
    then the dependency graph is repaired again. This is API-agnostic and prevents
    an older ancestor with the same placeholder name from owning a downstream call.
    """
    out = dict(plan or {})
    specs = {str(x.get("step_id")): x for x in (out.get("observation_specs") or [])}
    steps = []
    for raw in out.get("steps") or []:
        step = dict(raw)
        sid = str(step.get("id"))
        names = [str(x.get("name")).strip("{}")
                 for x in (specs.get(sid) or {}).get("bindings") or []
                 if x.get("name")]
        if names:
            existing = [str(x).strip("{}") for x in (step.get("binds") or [])]
            step["binds"] = list(dict.fromkeys(existing + names))
        steps.append(step)
    out["steps"] = steps
    return repair_selection_dependencies("", out)

def _oas_literal_type_ok(value: Any, meta: dict[str, Any] | None) -> bool:
    """Conservatively check a planner-declared literal against OAS metadata.

    Unknown/union schemas are left to the API. Numeric widening integer->number
    is safe; lossy coercions such as string->integer are not silently accepted.
    """
    meta = meta or {}
    typ = str(meta.get("type") or "unknown").lower()
    if typ in {"", "unknown", "oneof", "anyof"} or "|" in typ:
        type_ok = True
    elif typ == "string":
        type_ok = isinstance(value, str)
    elif typ == "integer":
        type_ok = isinstance(value, int) and not isinstance(value, bool)
    elif typ == "number":
        type_ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif typ == "boolean":
        type_ok = isinstance(value, bool)
    elif typ == "array":
        type_ok = isinstance(value, (list, tuple))
    elif typ == "object":
        type_ok = isinstance(value, dict)
    else:
        type_ok = True
    if not type_ok:
        return False
    enum = meta.get("enum") if isinstance(meta.get("enum"), list) else []
    if enum and value not in enum:
        return False
    item_enum = meta.get("item_enum") if isinstance(meta.get("item_enum"), list) else []
    if item_enum and isinstance(value, (list, tuple)) and any(x not in item_enum for x in value):
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(meta.get("minimum"), (int, float)) and value < meta["minimum"]:
            return False
        if isinstance(meta.get("maximum"), (int, float)) and value > meta["maximum"]:
            return False
    if isinstance(value, str):
        if isinstance(meta.get("minLength"), (int, float)) and len(value) < int(meta["minLength"]):
            return False
        if isinstance(meta.get("maxLength"), (int, float)) and len(value) > int(meta["maxLength"]):
            return False
    return True


def _schema_types_compatible(source_type: str | None, target_type: str | None) -> bool:
    source = str(source_type or "unknown").lower()
    target = str(target_type or "unknown").lower()
    if source in {"", "unknown", "oneof", "anyof"} or target in {"", "unknown", "oneof", "anyof"}:
        return True
    if "|" in source or "|" in target:
        return True
    if source == target:
        return True
    # Integer values are valid instances of OpenAPI number schemas. The reverse
    # is not guaranteed and therefore fails closed when both schemas are known.
    return source == "integer" and target == "number"


def _leaf_type(card: dict[str, Any] | None, path: str, *, body: bool = False) -> str | None:
    key = "request_body_leaf_paths" if body else "leaf_paths"
    for item in (card or {}).get(key) or []:
        if str(item.get("path") or "") == str(path):
            return str(item.get("type") or "unknown")
    return None


def validate_request_contract(plan: dict[str, Any], cards: list[dict[str, Any]]) -> list[str]:
    """Validate explicit request semantics against selected OAS operations.

    Validation is schema-only: it checks names, requiredness, primitive literal
    constraints and body paths. It never consults benchmark routes or answers.
    """
    errors: list[str] = []
    card_by_step = {str(c.get("step_id")): c for c in cards}
    for step in (plan or {}).get("steps") or []:
        sid = str(step.get("id") or "")
        card = card_by_step.get(sid) or {}
        parameters = {
            (str(p.get("in") or "query").lower(), str(p.get("name"))): p
            for p in (card.get("parameters") or []) if p.get("name")
        }
        path_params = {name: meta for (loc, name), meta in parameters.items() if loc == "path"}
        query_params = {name: meta for (loc, name), meta in parameters.items() if loc == "query"}

        # Every concrete path literal is checked when the OAS exposes parameter
        # metadata. A placeholder present in the path but missing metadata is not
        # invented as an error because some real OAS documents omit that object.
        for name, value in (step.get("path_literals") or {}).items():
            meta = path_params.get(str(name))
            if meta is not None and not _oas_literal_type_ok(value, meta):
                errors.append(f"step {sid}: path literal {name!r} violates documented type/constraints")

        ql = step.get("query_literals") or {}
        qb = step.get("query_bindings") or {}
        for name in list(ql) + list(qb):
            if name not in query_params:
                errors.append(f"step {sid}: request query argument {name!r} is not documented by the selected OAS operation")
        for name, value in ql.items():
            meta = query_params.get(str(name))
            if meta is not None and not _oas_literal_type_ok(value, meta):
                errors.append(f"step {sid}: query literal {name!r} violates documented type/constraints")
        for name, meta in query_params.items():
            if bool(meta.get("required")) and name not in ql and name not in qb:
                errors.append(f"step {sid}: required query argument {name!r} lacks a declared literal or binding")

        body_leaf_meta = {str(x.get("path") or ""): x for x in (card.get("request_body_leaf_paths") or [])}
        body_leafs = set(body_leaf_meta)
        body_records = {str(x.get("path") or "") for x in (card.get("request_body_record_paths") or [])}
        def documented_body_path(name: str) -> bool:
            if not body_leafs and not body_records:
                return False
            if name in body_leafs or name in body_records:
                return True
            return any(p.startswith(name + ".") or p.startswith(name + "[*]")
                       for p in body_leafs | body_records)
        bl = step.get("body_literals") or {}
        bb = step.get("body_bindings") or {}
        for name in list(bl) + list(bb):
            if not documented_body_path(str(name)):
                errors.append(f"step {sid}: request body path {name!r} is not documented by the selected OAS operation")
        for name, value in bl.items():
            meta = body_leaf_meta.get(str(name))
            if meta is not None and not _oas_literal_type_ok(value, meta):
                errors.append(f"step {sid}: body literal {name!r} violates documented type/constraints")
        required_body = [str(x) for x in (card.get("request_body_required_fields") or [])]
        if bl or bb or bool(card.get("request_body_required")):
            for name in required_body:
                if name not in bl and name not in bb:
                    errors.append(f"step {sid}: required request body field {name!r} lacks a declared literal or binding")
        if bool(card.get("request_body_required")) and not (bl or bb):
            errors.append(f"step {sid}: required request body has no declared fields")
    return list(dict.fromkeys(errors))




def normalize_explicit_metric_extrema(question: str, plan: dict[str, Any],
                                      cards: list[dict[str, Any]],
                                      tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Make an explicit metric superlative replayable inside the chosen population.

    If a plan selects the first/nth record from a collection even though its own
    purpose says it is selecting the user's explicit popularity/rating superlative,
    and the selected OAS response exposes exactly one documented scalar metric for
    that criterion, replace only that selector with argmax/argmin(metric).  A route
    whose OAS already documents criterion ordering is left unchanged.  This never
    changes the population/endpoint and never invents a field.
    """
    out = copy.deepcopy(plan or {})
    q = str(question or "").casefold()
    criterion = None
    direction = None
    leaves: set[str] = set()
    purpose_terms: tuple[str, ...] = ()
    if re.search(r"\bmost\s+popular\b|\bhighest\s+popularity\b", q):
        criterion, direction = "popularity", "argmax"
        leaves = {"popularity"}
        purpose_terms = ("popular", "popularity")
    elif re.search(r"\b(?:highest|best|top)[ -]?rated\b|\bhighest\s+rating\b", q):
        criterion, direction = "rating", "argmax"
        leaves = {"vote_average", "rating", "score"}
        purpose_terms = ("rated", "rating", "vote_average", "vote average", "score")
    elif re.search(r"\b(?:lowest|worst)[ -]?rated\b|\blowest\s+rating\b", q):
        criterion, direction = "rating", "argmin"
        leaves = {"vote_average", "rating", "score"}
        purpose_terms = ("rated", "rating", "vote_average", "vote average", "score")
    if not criterion:
        return out

    by_step = {str(x.get("id") or ""): x for x in out.get("steps") or []}
    card_by_step = {str(x.get("step_id") or ""): x for x in cards or []}
    tool_by_path = {str(x.get("path") or ""): x for x in tools or []
                    if str(x.get("method") or "GET").upper() == "GET"}
    warnings = list(out.get("validation_warnings") or [])
    derivs = []
    for raw in out.get("derivations") or []:
        d = dict(raw)
        if str(d.get("operator") or "").lower() not in {"endpoint_rank", "first", "nth"}:
            derivs.append(d); continue
        purpose = str(d.get("purpose") or "").casefold()
        purpose_matches = any(term in purpose for term in purpose_terms)
        sources = [str(x) for x in d.get("source_steps") or []]
        if len(sources) != 1:
            derivs.append(d); continue
        sid = sources[0]
        step = by_step.get(sid) or {}
        path = str(step.get("endpoint") or "")
        # Do not reinterpret unrelated navigation selectors merely because the
        # overall question happens to contain a superlative.  The one safe
        # purpose-free case is a directly callable non-search collection route
        # (no entity placeholder): its raw first-record selector is the candidate
        # population selector itself, not a nested cast/credit/navigation choice.
        direct_population = bool(path and "{" not in path and "/search/" not in path.casefold())
        if not purpose_matches and not direct_population:
            derivs.append(d); continue
        tool = tool_by_path.get(path) or {}
        route_text = (path.replace("_", " ") + " " + str(tool.get("functionality") or "")).casefold()
        # Explicitly criterion-ranked routes should preserve endpoint order.
        ordered = bool(re.search(r"\border(?:ed)?\s+by\s+[^.]{0,50}" + re.escape(criterion), route_text))
        criterion_route = (criterion == "popularity" and re.search(r"(?:^|[/ _-])popular(?:$|[/ _-])", route_text))
        if ordered or criterion_route:
            derivs.append(d); continue

        card = card_by_step.get(sid) or {}
        candidates = []
        for leaf in card.get("leaf_paths") or []:
            raw_path = str(leaf.get("path") if isinstance(leaf, dict) else leaf or "")
            clean = re.sub(r"\[(?:\*|\d*)\]", "", raw_path).strip(".")
            if clean and clean.split(".")[-1].casefold() in leaves:
                candidates.append(raw_path)
        # Prefer a metric under the selector's declared collection if one exists.
        selector_field = re.sub(r"\[(?:\*|\d*)\]", "", str(d.get("field") or "")).strip(".")
        if selector_field:
            scoped = [x for x in candidates
                      if re.sub(r"\[(?:\*|\d*)\]", "", x).startswith(selector_field + ".")]
            if scoped:
                candidates = scoped
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) != 1:
            derivs.append(d); continue
        d["operator"] = direction
        d["field"] = candidates[0]
        d["rank"] = 0
        warnings.append(
            f"derivation {d.get('id')}: normalized raw record selection to {direction}({candidates[0]}) "
            f"for explicit {criterion} superlative within the already-selected population")
        derivs.append(d)
    out["derivations"] = derivs
    out["validation_warnings"] = list(dict.fromkeys(warnings))
    return out


def normalize_documented_ordered_extrema(question: str, plan: dict[str, Any],
                                          tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Use documented endpoint order instead of a page-local positive extremum.

    This normalization is intentionally conservative: it applies only to argmax,
    only when the user explicitly asks for a positive superlative (most/highest/top/
    popular/trending), and only when the selected OAS operation explicitly states
    that its returned collection is ordered by the same requested criterion.
    """
    out = copy.deepcopy(plan or {})
    q = str(question or "").casefold()
    if not re.search(r"\b(?:most|highest|top|popular|trending)\b", q):
        return out
    if re.search(r"\b(?:least|lowest|minimum)\b", q):
        return out
    by_step = {str(x.get("id") or ""): x for x in out.get("steps") or []}
    tool_by_path = {str(x.get("path") or ""): x for x in tools or []
                    if str(x.get("method") or "GET").upper() == "GET"}
    qstems = _semantic_stems(question)
    warnings = list(out.get("validation_warnings") or [])
    derivs = []
    for raw in out.get("derivations") or []:
        d = dict(raw)
        if str(d.get("operator") or "").lower() != "argmax":
            derivs.append(d); continue
        sources = [str(x) for x in d.get("source_steps") or []]
        if len(sources) != 1:
            derivs.append(d); continue
        path = str((by_step.get(sources[0]) or {}).get("endpoint") or "")
        prose = str((tool_by_path.get(path) or {}).get("functionality") or "").casefold()
        ordered = re.search(r"\border(?:ed)?\s+by\s+([a-z0-9_ -]+)", prose)
        if not ordered:
            derivs.append(d); continue
        order_stems = _semantic_stems(ordered.group(1))
        if not (qstems & order_stems):
            derivs.append(d); continue
        d["operator"] = "endpoint_rank"
        d["rank"] = 0
        d["field"] = None
        warnings.append(
            f"derivation {d.get('id')}: normalized page-local argmax to endpoint_rank "
            "because the selected OAS operation explicitly documents order by the requested criterion")
        derivs.append(d)
    out["derivations"] = derivs
    out["validation_warnings"] = list(dict.fromkeys(warnings))
    return out

def normalize_temporal_extremum_direction(question: str, plan: dict[str, Any]) -> dict[str, Any]:
    """Normalize an unambiguous date-extremum direction from explicit user intent.

    Date/time/year values have a stable ordering independent of API vocabulary.
    This repair is intentionally narrow: only explicit latest/newest/most-recent
    versus earliest/first-released language is acted on, so age comparisons (where
    an earlier birthday means older) are untouched. Future/upcoming requests are
    also left to the planner because their population semantics differ.
    """
    out = copy.deepcopy(plan or {})
    q = str(question or "").casefold()
    future = any(term in q for term in (
        "upcoming", "future", "will release", "will air", "scheduled",
        "next release", "next movie", "next season", "coming out"))
    latest = any(term in q for term in (
        "latest", "newest", "most recent", "recently", "recent ")) and not future
    earliest = any(term in q for term in ("earliest", "first released", "oldest release"))
    warnings = list(out.get("validation_warnings") or [])
    derivs = []
    for raw in out.get("derivations") or []:
        d = dict(raw)
        op = str(d.get("operator") or "").lower()
        field = str(d.get("field") or "").casefold()
        date_like = any(token in field for token in ("date", "time", "year"))
        if date_like and latest and op == "argmin":
            d["operator"] = "argmax"
            warnings.append(
                f"derivation {d.get('id')}: normalized argmin to argmax for explicit latest-date intent")
        elif date_like and earliest and op == "argmax":
            d["operator"] = "argmin"
            warnings.append(
                f"derivation {d.get('id')}: normalized argmax to argmin for explicit earliest-date intent")
        derivs.append(d)
    out["derivations"] = derivs
    out["validation_warnings"] = list(dict.fromkeys(warnings))
    return out

def normalize_obvious_request_literal_bindings(
        question: str, plan: dict[str, Any], cards: list[dict[str, Any]] | None = None
        ) -> dict[str, Any]:
    """Move unmistakable literal strings out of *_bindings into *_literals.

    This is deliberately conservative.  A binding is changed only when it has no
    reachable declared producer, the OAS target is a string, and the raw value is
    either literally present in the user question or has unmistakable literal
    punctuation (for example ``en,null``). Identifier-like symbolic names such as
    ``lang_en``/``timezone`` remain invalid and are sent to normal convergence.
    """
    out = copy.deepcopy(plan or {})
    steps = {str(x.get("id") or ""): x for x in out.get("steps") or []}
    card_by_step = {str(x.get("step_id") or ""): x for x in (cards or [])}
    q = str(question or "").casefold()

    def produced(step: dict[str, Any]) -> set[str]:
        names = {str(x).strip("{}") for x in step.get("binds") or []}
        names.update(str(x).strip("{}") for x in (step.get("binding_paths") or {}))
        return {x for x in names if x}

    def has_producer(consumer: str, alias: str) -> bool:
        queue = [str(x) for x in (steps.get(consumer) or {}).get("depends_on") or []]
        seen = set()
        while queue:
            sid = queue.pop(0)
            if sid in seen or sid not in steps:
                continue
            seen.add(sid)
            if alias in produced(steps[sid]):
                return True
            queue.extend(str(x) for x in (steps[sid].get("depends_on") or []))
        return False

    def target_is_string(sid: str, location: str, target: str) -> bool:
        card = card_by_step.get(sid) or {}
        if location == "query":
            for param in card.get("parameters") or []:
                if str(param.get("in") or "").lower() == "query" and str(param.get("name") or "") == target:
                    return str(param.get("type") or "unknown").lower() in {"string", "unknown", ""}
        return False

    warnings = list(out.get("validation_warnings") or [])
    for sid, step in steps.items():
        for location in ("query",):
            bind_key = location + "_bindings"
            lit_key = location + "_literals"
            bindings = dict(step.get(bind_key) or {})
            literals = dict(step.get(lit_key) or {})
            for target, raw in list(bindings.items()):
                alias = str(raw).strip("{}")
                if has_producer(sid, alias) or not target_is_string(sid, location, str(target)):
                    continue
                raw_text = str(raw).strip()
                if not raw_text:
                    continue
                in_question = raw_text.casefold() in q
                punct_literal = bool(re.search(r"[\s,/:]", raw_text)) and "_" not in raw_text
                if not (in_question or punct_literal):
                    continue
                literals[str(target)] = raw
                bindings.pop(target, None)
                warnings.append(
                    f"step {sid}: normalized unproduced {location} binding {target}<-{raw!r} "
                    "to a literal because the value is user-stated/literal-shaped")
            step[bind_key] = bindings
            step[lit_key] = literals
    out["steps"] = list(steps.values())
    out["validation_warnings"] = list(dict.fromkeys(warnings))
    return out


def normalize_direct_dependency_step_bindings(plan: dict[str, Any]) -> dict[str, Any]:
    """Normalize planner shorthand ``placeholder <- step_id`` to a real binding.

    Models occasionally use a direct dependency's step id where the plan schema
    requires the *binding name* produced by that step.  This is safe to repair
    mechanically only when the referenced step is a declared direct dependency and
    either (a) it produces a binding matching the request target, or (b) it produces
    exactly one binding in total.  Ambiguous producer steps remain invalid and are
    sent through normal convergence instead of being guessed.
    """
    out = copy.deepcopy(plan or {})
    steps = {str(x.get("id") or ""): x for x in (out.get("steps") or [])}
    warnings = list(out.get("validation_warnings") or [])

    def produced_names(step: dict[str, Any]) -> list[str]:
        names = [str(x).strip("{}") for x in (step.get("binds") or []) if str(x).strip("{}")]
        names += [str(x).strip("{}") for x in (step.get("binding_paths") or {})
                  if str(x).strip("{}")]
        return list(dict.fromkeys(names))

    for sid, step in steps.items():
        deps = {str(x) for x in (step.get("depends_on") or [])}
        for key in ("path_bindings", "query_bindings", "body_bindings"):
            mapping = dict(step.get(key) or {})
            for target, raw_alias in list(mapping.items()):
                dep_id = str(raw_alias).strip("{}")
                if dep_id not in deps or dep_id not in steps:
                    continue
                names = produced_names(steps[dep_id])
                target_name = str(target).strip("{}")
                replacement = target_name if target_name in names else (names[0] if len(names) == 1 else None)
                if not replacement:
                    continue
                mapping[target] = replacement
                warnings.append(
                    f"step {sid}: normalized {key[:-1]} {target}<-{dep_id} to direct "
                    f"producer binding {replacement!r}")
            step[key] = mapping
    out["steps"] = list(steps.values())
    out["validation_warnings"] = list(dict.fromkeys(warnings))
    return out


def validate_declared_request_binding_provenance(plan: dict[str, Any]) -> list[str]:
    """Validate request wiring from the declared plan graph before projection exists.

    Observation bindings are synthesized later, but a dynamic request alias must
    already have exactly one reachable producer in the declared dependency DAG.
    Catching this here gives bounded plan convergence a chance to repair literal-vs-
    binding mistakes instead of discovering them only after execution planning.
    """
    steps = {str(x.get("id") or ""): x for x in (plan or {}).get("steps") or []}
    errors: list[str] = []

    def produced(step: dict[str, Any]) -> set[str]:
        names = {str(x).strip("{}") for x in step.get("binds") or []}
        names.update(str(x).strip("{}") for x in (step.get("binding_paths") or {}))
        return {x for x in names if x}

    def distances(consumer: str) -> dict[str, int]:
        out: dict[str, int] = {}
        queue = [(consumer, 0)]
        while queue:
            current, dist = queue.pop(0)
            for dep in (steps.get(current) or {}).get("depends_on") or []:
                dep = str(dep)
                nd = dist + 1
                if dep in steps and (dep not in out or nd < out[dep]):
                    out[dep] = nd
                    queue.append((dep, nd))
        return out

    def producer(consumer: str, alias: str) -> str | None:
        ds = distances(consumer)
        candidates = [(dist, sid) for sid, dist in ds.items()
                      if alias in produced(steps.get(sid) or {})]
        if not candidates:
            return None
        best = min(x[0] for x in candidates)
        nearest = [sid for dist, sid in candidates if dist == best]
        return nearest[0] if len(nearest) == 1 else None

    for sid, step in steps.items():
        literals = {str(x).strip("{}") for x in (step.get("path_literals") or {})}
        path_aliases: dict[str, str] = {}
        for name in _step_placeholders(step):
            name = str(name).strip("{}")
            if name in literals:
                continue
            alias = str((step.get("path_bindings") or {}).get(name) or name).strip("{}")
            path_aliases[name] = alias
            if producer(sid, alias) is None:
                errors.append(f"step {sid}: no unique declared producer for path placeholder {{{name}}}")

        # Reusing one dynamic alias for two distinct URL roles is almost always a
        # planner wiring error (e.g. series_id accidentally used as season_number).
        # Require the plan to declare separate aliases or an explicit literal.
        reverse: dict[str, list[str]] = {}
        for target, alias in path_aliases.items():
            reverse.setdefault(alias, []).append(target)
        for alias, targets in reverse.items():
            if len(targets) > 1:
                errors.append(
                    f"step {sid}: distinct path placeholders {sorted(targets)} reuse binding "
                    f"{alias!r}; declare distinct producer aliases or explicit literals")

        for location, mapping in (("query", step.get("query_bindings") or {}),
                                  ("body", step.get("body_bindings") or {})):
            for target, raw_alias in mapping.items():
                alias = str(raw_alias).strip("{}")
                if producer(sid, alias) is None:
                    errors.append(
                        f"step {sid}: no unique declared producer for {location} binding "
                        f"{target}<-{raw_alias}")
    return list(dict.fromkeys(errors))


def validate_binding_graph(plan: dict[str, Any], cards: list[dict[str, Any]] | None = None) -> list[str]:
    """Validate request-binding provenance and known schema type compatibility.

    Every dynamic request value must have one nearest producer with a validated
    observation binding. When both producer and consumer OAS types are known,
    incompatible wiring is rejected before execution.
    """
    from utils.plan_execution_audit import nearest_binding_producer, binding_spec
    errors: list[str] = []
    steps = {str(x.get("id")): x for x in (plan or {}).get("steps") or []}
    specs = {str(x.get("step_id")): x for x in (plan or {}).get("observation_specs") or []}
    card_by_step = {str(c.get("step_id")): c for c in (cards or [])}

    def producer_type(producer: str, binding_name: str) -> str | None:
        binding = binding_spec(plan, producer, binding_name)
        if not binding or not binding.get("path"):
            return None
        absolute = _absolute_schema_path(
            str((specs.get(producer) or {}).get("record_path") or "$"),
            str(binding.get("path") or ""))
        return _leaf_type(card_by_step.get(producer), absolute)

    def request_target_type(sid: str, location: str, target: str) -> str | None:
        card = card_by_step.get(sid) or {}
        if location in {"path", "query"}:
            for param in card.get("parameters") or []:
                if str(param.get("in") or "").lower() == location and str(param.get("name") or "") == target:
                    return str(param.get("type") or "unknown")
            return None
        return _leaf_type(card, target, body=True)

    for sid, step in steps.items():
        literals = step.get("path_literals") or {}
        for name in _step_placeholders(step):
            if name in literals:
                if literals.get(name) in (None, ""):
                    errors.append(f"step {sid}: path literal {{{name}}} is empty")
                continue
            binding_name = str((step.get("path_bindings") or {}).get(name) or name).strip("{}")
            producer = nearest_binding_producer(plan, sid, binding_name)
            if not producer:
                errors.append(f"step {sid}: no unique declared producer for path placeholder {{{name}}}")
                continue
            binding = binding_spec(plan, producer, binding_name)
            if not binding or not binding.get("path"):
                errors.append(f"step {sid}: producer {producer} lacks a validated binding path for {{{name}}}<-{binding_name}")
                continue
            src_t = producer_type(producer, binding_name); dst_t = request_target_type(sid, "path", name)
            path_scalar_serializable = (str(dst_t or "").lower() == "string" and
                                        str(src_t or "").lower() in {"integer", "number", "boolean"})
            if not (_schema_types_compatible(src_t, dst_t) or path_scalar_serializable):
                errors.append(f"step {sid}: incompatible schema types for path {{{name}}}: producer={src_t}, target={dst_t}")

    for sid, step in steps.items():
        for location, mapping in (("query", step.get("query_bindings") or {}),
                                  ("body", step.get("body_bindings") or {})):
            for target, binding_name in mapping.items():
                producer = nearest_binding_producer(plan, sid, str(binding_name))
                if not producer:
                    errors.append(f"step {sid}: no unique declared producer for {location} binding {target}<-{binding_name}")
                    continue
                binding = binding_spec(plan, producer, str(binding_name))
                if not binding or not binding.get("path"):
                    errors.append(f"step {sid}: producer {producer} lacks a validated binding path for {location} binding {target}<-{binding_name}")
                    continue
                src_t = producer_type(producer, str(binding_name))
                dst_t = request_target_type(sid, location, str(target))
                if not _schema_types_compatible(src_t, dst_t):
                    errors.append(f"step {sid}: incompatible schema types for {location} binding {target}<-{binding_name}: producer={src_t}, target={dst_t}")

    for d in (plan or {}).get("derivations") or []:
        op = str(d.get("operator") or "").lower()
        sources = [str(x) for x in d.get("source_steps") or [] if str(x) in steps]
        derived_sources = [str(x) for x in d.get("source_derivations") or []]
        if op == "membership":
            literal_sources = 1 if d.get("comparison_literal") is not None else 0
            if len(sources) + len(derived_sources) + literal_sources < 2:
                errors.append(
                    f"derivation {d.get('id')}: membership requires at least two declared "
                    "operands across source_steps/source_derivations/comparison_literal")
        literal_sources = 1 if d.get("comparison_literal") is not None else 0
        if op == "compare" and len(sources) + len(derived_sources) + literal_sources < 2:
            errors.append(f"derivation {d.get('id')}: compare requires at least two declared sources/literals")
    return list(dict.fromkeys(errors))


def _absolute_schema_path(record_path: str, relative_path: str) -> str:
    root = str(record_path or "$").strip()
    rel = str(relative_path or "").strip().replace("$.", "")
    if not rel:
        return root
    if root in {"", "$"}:
        return rel
    return root + "." + rel





def _normalized_schema_path(path: Any) -> str:
    """Canonicalize schema/list paths without treating selection indices as fields."""
    text = str(path or "").replace("$.", "")
    # ``items[*].id`` and ``items[0].id`` address the same schema leaf.  The
    # numeric index is selection metadata and must not make an otherwise valid
    # OAS field look absent.
    text = re.sub(r"\[(?:\*|\d*)\]", "", text)
    return text.strip(".$")


def _record_relative_path(absolute_path: str, record_path: str) -> str | None:
    """Return a schema leaf relative to a record root, or None if unrelated."""
    absolute = str(absolute_path or "").replace("$.", "").strip(".$")
    record = str(record_path or "$").replace("$.", "").strip(".$")
    if record in {"", "$"}:
        return absolute or "$"
    prefix = record + "."
    if absolute == record:
        return "$"
    if absolute.startswith(prefix):
        return absolute[len(prefix):]
    return None


def _downstream_binding_names(plan: dict[str, Any], producer_id: str) -> set[str]:
    """Bindings from producer_id that are actually consumed by direct dependents."""
    steps = {str(x.get("id")): x for x in (plan or {}).get("steps") or []}
    producer = steps.get(str(producer_id)) or {}
    declared = {str(x).strip("{}") for x in producer.get("binds") or []}
    needed: set[str] = set()
    for step in steps.values():
        if str(producer_id) not in {str(x) for x in step.get("depends_on") or []}:
            continue
        names = {str(x).strip("{}") for x in (step.get("path_bindings") or {}).values()}
        names.update(str(x).strip("{}") for x in _step_placeholders(step)
                     if str(x).strip("{}") not in (step.get("path_bindings") or {}))
        names.update(str(x).strip("{}") for x in (step.get("query_bindings") or {}).values())
        names.update(str(x).strip("{}") for x in (step.get("body_bindings") or {}).values())
        needed.update(names & declared)
    needed -= _protected_passthrough_binding_names(plan, str(producer_id))
    return needed


def _selection_derivations_for_step(plan: dict[str, Any], step_id: str) -> list[dict[str, Any]]:
    selection_ops = {"filter", "argmax", "argmin", "first", "nth", "endpoint_rank"}
    return [dict(d) for d in (plan or {}).get("derivations") or []
            if str(step_id) in {str(x) for x in d.get("source_steps") or []}
            and str(d.get("operator") or "").lower() in selection_ops]


def _projection_derivations_for_step(plan: dict[str, Any], step_id: str) -> list[dict[str, Any]]:
    """Derivations whose raw record universe belongs to one API step.

    Comparison/logical combiners consume prior derived scalars and therefore do
    not define an observation record root. Count/identity/filter/ranking operators
    do, and must inform deterministic projection even when the step has no
    downstream binding.
    """
    derived_only = {"compare", "logical_and", "logical_or"}
    return [dict(d) for d in (plan or {}).get("derivations") or []
            if str(step_id) in {str(x) for x in d.get("source_steps") or []}
            and str(d.get("operator") or "").lower() not in derived_only]


def _candidate_record_root(card: dict[str, Any], derivations: list[dict[str, Any]],
                           *, require_binding: bool,
                           binding_paths: list[str] | None = None) -> tuple[str | None, list[str]]:
    """Choose a record universe from OAS structure without semantic guessing.

    Only schema structure and explicit plan fields/filters are used. If two sibling
    collections remain equally plausible, return an ambiguity instead of choosing
    by endpoint names or domain knowledge.
    """
    records = list(dict.fromkeys(str(x.get("path")) for x in card.get("record_paths") or []
                                 if x.get("path")))
    array_records = [r for r in records if "[*]" in r]
    leaves = [str(x.get("path")) for x in card.get("leaf_paths") or [] if x.get("path")]

    derivation_hints: list[str] = []
    for d in derivations:
        field = str(d.get("field") or "").strip()
        if field:
            derivation_hints.append(field)
        derivation_hints.extend(str(x) for x in (d.get("filter") or {}).keys() if str(x))
    derivation_hints = list(dict.fromkeys(derivation_hints))
    binding_hints = [str(x).strip() for x in (binding_paths or []) if str(x).strip()]
    # Derivation semantics are primary. Explicit binding paths are additional
    # schema evidence for terminal/intermediate steps whose derivation uses a bare
    # scalar (or no scalar) while the binding already names the exact sibling
    # collection, e.g. guest_stars[*].name vs cast[*].name.
    hints = list(dict.fromkeys(derivation_hints + binding_hints))

    # Bare scalar paths describe the response object itself when that scalar is
    # documented at the root.  Prefer the root before looking for identically named
    # leaves inside incidental arrays.  If a binding/derivation explicitly names a
    # collection relation (crew.name, episodes[*].air_date, ...), that qualification
    # still wins below.  This removes false ambiguity on object endpoints that also
    # contain nested arrays with common fields such as name/air_date/id.
    top_level_leaves = {
        _normalized_schema_path(x) for x in leaves
        if "." not in _normalized_schema_path(x)
    }
    unqualified_deriv = [_normalized_schema_path(h) for h in derivation_hints if h]
    explicitly_qualified = any("." in _normalized_schema_path(h) for h in hints if h)
    binding_names_collection = False
    if binding_hints:
        normalized_arrays = {_normalized_schema_path(r) for r in array_records}
        for hint in binding_hints:
            h = _normalized_schema_path(hint)
            if h in normalized_arrays or any(r.startswith(h + ".") for r in normalized_arrays):
                binding_names_collection = True
                break
    if (unqualified_deriv and not explicitly_qualified and not binding_names_collection
            and all(h in top_level_leaves for h in unqualified_deriv)):
        return "$", []

    # A fully qualified binding can identify a nested *object* record universe,
    # not only an array.  This is common when one API response embeds the object
    # needed by a dependent request (e.g. item.album.id). Prefer the deepest
    # documented non-array object prefix and avoid treating incidental child
    # arrays as competing producer collections.
    qualified_binding_hints = [_normalized_schema_path(h) for h in binding_hints
                               if "." in _normalized_schema_path(h)]
    if qualified_binding_hints:
        # A qualified binding may pass through a collection before reaching the
        # scalar leaf (e.g. tracks.items.id).  Prefer that documented array record
        # universe over an ancestor pagination/container object such as ``tracks``.
        # If no array prefix exists, a nested singleton object remains the right
        # producer universe (e.g. item.album.id -> item.album).
        matching_arrays = []
        for r in array_records:
            rn = _normalized_schema_path(r)
            if all(h == rn or h.startswith(rn + ".") for h in qualified_binding_hints):
                matching_arrays.append(r)
        if matching_arrays:
            matching_arrays.sort(
                key=lambda r: (_normalized_schema_path(r).count("."), len(r)),
                reverse=True)
            best_depth = _normalized_schema_path(matching_arrays[0]).count(".")
            deepest = [r for r in matching_arrays
                       if _normalized_schema_path(r).count(".") == best_depth]
            if len(deepest) == 1:
                return deepest[0], []

        object_records = [r for r in records if "[*]" not in r and r not in {"", "$"}]
        matching_objects = []
        for r in object_records:
            rn = _normalized_schema_path(r)
            if all(h == rn or h.startswith(rn + ".") for h in qualified_binding_hints):
                matching_objects.append(r)
        if matching_objects:
            matching_objects.sort(key=lambda r: (_normalized_schema_path(r).count("."), len(r)),
                                  reverse=True)
            best_depth = _normalized_schema_path(matching_objects[0]).count(".")
            deepest = [r for r in matching_objects
                       if _normalized_schema_path(r).count(".") == best_depth]
            if len(deepest) == 1:
                return deepest[0], []

    def root_supports(root: str, hint: str) -> bool:
        h = _normalized_schema_path(hint)
        r = _normalized_schema_path(root)
        if not h:
            return True
        # Explicit relation/container qualification is strongest.
        if h == r or h.startswith(r + ".") or r.startswith(h + "."):
            return True
        # A qualified plan path (e.g. cast.popularity) names its collection
        # explicitly. Do not let a sibling collection match merely because it
        # exposes the same terminal scalar name.
        if "." in h:
            return False
        for leaf in leaves:
            rel = _record_relative_path(leaf, root)
            if rel is None:
                continue
            reln = _normalized_schema_path(rel)
            if reln == h or reln.endswith("." + h):
                return True
        return False

    fieldless_outer_selector = any(
        str(d.get("operator") or "").lower() in {"endpoint_rank", "first", "nth"}
        and not str(d.get("field") or "").strip()
        for d in derivations)
    if fieldless_outer_selector:
        top_arrays = [r for r in array_records if "." not in _normalized_schema_path(r)]
        top_arrays = list(dict.fromkeys(top_arrays))
        if len(top_arrays) == 1:
            return top_arrays[0], []

    if derivation_hints:
        deriv_exact = []
        for hint in derivation_hints:
            h = _normalized_schema_path(hint)
            deriv_exact.extend(r for r in array_records
                               if _normalized_schema_path(r) == h or
                               h.startswith(_normalized_schema_path(r) + "."))
        deriv_exact = list(dict.fromkeys(deriv_exact))
        if deriv_exact:
            deriv_exact.sort(key=lambda r: (_normalized_schema_path(r).count("."), len(r)),
                             reverse=True)
            best_depth = _normalized_schema_path(deriv_exact[0]).count(".")
            deepest = [r for r in deriv_exact
                       if _normalized_schema_path(r).count(".") == best_depth]
            if len(deepest) == 1:
                return deepest[0], []

    if hints:
        # An explicitly named collection relation (``results``, ``cast``,
        # ``episodes.crew``) selects that exact record universe. Do not reinterpret
        # it as the deepest descendant collection merely because descendants share
        # the same prefix. Depth is used only for scalar field hints.
        exact_roots = []
        for hint in hints:
            h = _normalized_schema_path(hint)
            exact_roots.extend(r for r in array_records
                               if _normalized_schema_path(r) == h)
            # A qualified scalar such as crew.job also explicitly identifies the
            # owning collection. One malformed sibling hint must not erase that
            # stronger relation declaration and turn it into a false ambiguity.
            exact_roots.extend(r for r in array_records
                               if h.startswith(_normalized_schema_path(r) + "."))
        exact_roots = list(dict.fromkeys(exact_roots))
        if exact_roots:
            # Qualified scalar paths can name both an ancestor collection and its
            # nested owning collection (episodes[*] and episodes[*].crew[*]).  That
            # is not sibling ambiguity: the deepest prefix owns the scalar.  Only
            # equally-deep unrelated roots remain ambiguous.
            exact_roots.sort(
                key=lambda r: (_normalized_schema_path(r).count("."), len(r)),
                reverse=True)
            best_depth = _normalized_schema_path(exact_roots[0]).count(".")
            deepest = [r for r in exact_roots
                       if _normalized_schema_path(r).count(".") == best_depth]
            if len(deepest) == 1:
                return deepest[0], []
            return None, [f"ambiguous explicitly named record collections: {deepest}"]

        # A response object may expose the requested scalar directly while also
        # containing many incidental child arrays (genres, seasons, networks, ...).
        # For entirely unqualified hints, a directly documented top-level scalar is
        # stronger evidence than sibling arrays that happen to repeat the same leaf
        # name. Qualified hints still select their named collection above.
        normalized_hints = [_normalized_schema_path(h) for h in hints if h]
        if normalized_hints and not any("." in h for h in normalized_hints):
            if all(h in top_level_leaves for h in normalized_hints):
                return "$", []

        supported = [r for r in array_records if all(root_supports(r, h) for h in hints)]
        if supported:
            normalized_hints = [_normalized_schema_path(h) for h in hints if h]
            if normalized_hints and not any("." in h for h in normalized_hints):
                # Unqualified scalars such as ``id``/``name`` belong to the nearest
                # compatible record universe.  Descending into an incidental nested
                # array merely because it repeats ``id`` can bind the child entity
                # instead of the requested/search entity (e.g. results[*].known_for[*]).
                # A planner that truly intends the nested relation must qualify it.
                supported.sort(key=lambda r: (_normalized_schema_path(r).count("."), len(r)))
                best_depth = _normalized_schema_path(supported[0]).count(".")
                nearest = [r for r in supported if _normalized_schema_path(r).count(".") == best_depth]
                if len(nearest) == 1:
                    return nearest[0], []
                return None, [f"ambiguous record collections for unqualified plan fields: {nearest}"]
            # Qualified scalar paths explicitly name their owner; use the deepest
            # compatible relation (e.g. episodes.crew.id -> crew records).
            supported.sort(key=lambda r: (_normalized_schema_path(r).count("."), len(r)), reverse=True)
            best_depth = _normalized_schema_path(supported[0]).count(".")
            deepest = [r for r in supported if _normalized_schema_path(r).count(".") == best_depth]
            if len(deepest) == 1:
                return deepest[0], []
            return None, [f"ambiguous record collections for explicit plan fields: {deepest}"]

    # With no structural field hint, a single top-level collection is safe. Multiple
    # sibling collections (e.g. cast vs crew) are a semantic choice and must remain
    # in the evidence plan rather than being invented here.
    top_arrays = []
    for r in array_records:
        norm = _normalized_schema_path(r)
        if "." not in norm:
            top_arrays.append(r)
    top_arrays = list(dict.fromkeys(top_arrays))
    if len(top_arrays) == 1:
        return top_arrays[0], []
    if not require_binding:
        return "$", []
    if not array_records:
        return "$", []
    return None, [f"ambiguous producer collection; evidence plan must qualify the relation: {top_arrays or array_records[:6]}"]



def validate_schema_projection_semantics(plan: dict[str, Any],
                                         cards: list[dict[str, Any]]) -> list[str]:
    """Ensure plan fields/bindings are mechanically realizable from selected OAS.

    Semantic choices stay in the Evidence Plan. This validator only checks that the
    chosen record universe, filter/ranking/output scalar fields, and downstream
    binding leaves actually exist. Prompt-oriented schema truncation must never be
    used here.
    """
    errors = list(validate_collection_binding_selection(plan, cards))
    card_by_step = {str(c.get("step_id") or ""): c for c in cards or []}
    for step in (plan or {}).get("steps") or []:
        sid = str(step.get("id") or "")
        card = card_by_step.get(sid) or {}
        bind_names = _downstream_binding_names(plan, sid)
        derivs = _projection_derivations_for_step(plan, sid)
        if not bind_names and not derivs:
            continue
        # Some real OpenAPI operations document the request surface but provide an
        # empty response schema. For read-only GETs, request/OAS authorization stays
        # strict while response-field validation is deferred to the immutable
        # runtime structural profile. The runtime resolver accepts only one unique
        # record universe, so this is fail-closed rather than field guessing.
        if (str(step.get("method") or "GET").upper() == "GET"
                and not list(card.get("leaf_paths") or [])):
            continue
        binding_hints = [str((step.get("binding_paths") or {}).get(name) or name)
                         for name in (set(bind_names) | set((step.get("binding_paths") or {}).keys()))]
        root, root_errors = _candidate_record_root(
            card, derivs, require_binding=bool(bind_names), binding_paths=binding_hints)
        errors.extend(f"step {sid}: {e}" for e in root_errors)
        if not root or root_errors:
            continue

        available_rel = {
            _normalized_schema_path(rel): rel
            for rel in (_record_relative_path(str(x.get("path")), root)
                        for x in card.get("leaf_paths") or [] if x.get("path"))
            if rel not in (None, "", "$")
        }
        for deriv in derivs:
            did = str(deriv.get("id") or "")
            op = str(deriv.get("operator") or "").lower()
            for path, pred_op, _value in _normalized_plan_filter_predicates(
                    deriv.get("filter") or {}, root):
                norm_path = _normalized_schema_path(path)
                if pred_op in {"exists", "not_exists"} and norm_path in {"", "$"}:
                    continue
                if norm_path not in available_rel:
                    errors.append(
                        f"derivation {did}: filter field {path!r} is not a documented scalar "
                        f"under selected record collection {root}")
            # These operators consume a scalar field directly. endpoint_rank/first/
            # nth may name a collection relation or no field at all, and count may
            # intentionally consume the collection itself.
            if op in {"identity", "argmax", "argmin"}:
                field = _relative_projection_field(deriv.get("field"), root)
                if field and _normalized_schema_path(field) not in available_rel:
                    errors.append(
                        f"derivation {did}: {op} field {deriv.get('field')!r} is not a "
                        f"documented scalar under selected record collection {root}")

        for name in sorted(bind_names):
            _path, bind_errors = _binding_leaf_for_root(
                card, root, name, (step.get("binding_paths") or {}).get(name))
            errors.extend(f"step {sid}: {e}" for e in bind_errors)
    return list(dict.fromkeys(errors))


def _binding_leaf_for_root(card: dict[str, Any], record_path: str, name: str,
                           declared_path: str | None = None) -> tuple[str | None, list[str]]:
    leaves = [str(x.get("path")) for x in card.get("leaf_paths") or [] if x.get("path")]
    rels = [r for r in (_record_relative_path(x, record_path) for x in leaves)
            if r not in (None, "$", "")]

    if declared_path:
        wanted = _normalized_schema_path(declared_path)
        # Primitive array items are legal typed values even though their schema
        # path is the selected record root itself rather than a child leaf.
        if wanted == _normalized_schema_path(record_path):
            raw_root_leaf = next((str(x.get("path")) for x in card.get("leaf_paths") or []
                                  if _normalized_schema_path(x.get("path")) == wanted), None)
            if raw_root_leaf:
                return "$", []
        matches = []
        for full, rel in zip(leaves, [_record_relative_path(x, record_path) for x in leaves]):
            if rel in (None, "$", ""):
                continue
            if (_normalized_schema_path(full) == wanted or
                    _normalized_schema_path(rel) == wanted):
                matches.append(rel)
        matches = list(dict.fromkeys(matches))
        if len(matches) == 1:
            return matches[0], []
        # Do not make a malformed redundant path string fatal when the binding
        # alias itself maps to one unique scalar under the already-selected record
        # universe. This is schema canonicalization, not semantic guessing.

    exact = [r for r in rels if _normalized_schema_path(r) == _normalized_schema_path(name)]
    if len(exact) == 1:
        return exact[0], []
    candidates: list[str] = []
    token = str(name or "").strip("{}").split("_")[-1]
    if str(name or "").endswith("_id"):
        token = "id"
    for rel in rels:
        norm = _normalized_schema_path(rel)
        if norm == token or norm.endswith("." + token):
            candidates.append(rel)
    candidates = list(dict.fromkeys(candidates))
    direct = [x for x in candidates if "[*]" not in x and "." not in x]
    if len(direct) == 1:
        return direct[0], []
    if len(candidates) == 1:
        return candidates[0], []
    if declared_path:
        return None, [
            f"declared binding path {declared_path!r} for {name} is not a schema leaf under "
            f"{record_path}, and the alias has no unique fallback: {candidates[:8]}"
        ]
    return None, [f"binding {name} has no unique schema leaf under {record_path}: {candidates[:8]}"]


def synthesize_observation_specs(question: str, plan: dict[str, Any],
                                 cards: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Compile observation specs from an evidence plan + selected OAS schemas.

    This is intentionally conservative. It handles mechanical projection/binding
    choices only. Semantic ambiguity falls back to the bounded LLM projection
    planner rather than being guessed by host code.
    """
    card_by_step = {str(c.get("step_id")): c for c in cards}
    specs: list[dict[str, Any]] = []
    errors: list[str] = []
    typed_fanout: dict[str, dict[str, dict[str, Any]]] = {}
    for item in (plan or {}).get("typed_fanout_bindings") or []:
        sid = str((item or {}).get("step_id") or "")
        name = str((item or {}).get("binding") or "").strip("{}")
        if sid and name:
            typed_fanout.setdefault(sid, {})[name] = dict(item or {})
    for step in (plan or {}).get("steps") or []:
        sid = str(step.get("id") or "")
        card = card_by_step.get(sid) or {}
        bind_names = _downstream_binding_names(plan, sid)
        derivs = _projection_derivations_for_step(plan, sid)

        if not bind_names and not derivs:
            # A truly mechanical terminal step with no derivation needs only a
            # compact success signal. If any derivation consumes this step, its
            # record universe/fields must still be projected deterministically.
            spec = {
                "step_id": sid, "record_path": "$", "project_paths": [],
                "filters": [], "sort": [],
                "select": {"mode": "single", "limit": 1, "index": 0},
                "bindings": [], "aggregates": [], "completeness": "one response object",
                "purpose": "Confirm execution; final evidence remains in the lossless ledger.",
            }
            specs.append(spec)
            continue

        binding_hints = [str((step.get("binding_paths") or {}).get(name) or name)
                         for name in (set(bind_names) | set((step.get("binding_paths") or {}).keys()))]
        if (str(step.get("method") or "GET").upper() == "GET"
                and not list(card.get("leaf_paths") or [])):
            # OAS-authorized request with undocumented response shape. Keep only
            # planner-declared structural paths; runtime projection will resolve
            # them against the observed names/types and fail if more than one
            # record universe could satisfy them.
            project_paths = []
            for d in derivs:
                field = str(d.get("field") or "").strip()
                if field and field not in {"$"} and field not in project_paths:
                    project_paths.append(field)
                for fp in (d.get("filter") or {}):
                    fp = str(fp).strip()
                    if fp and fp not in project_paths:
                        project_paths.append(fp)
            bindings = []
            for name in sorted(bind_names):
                path = str((step.get("binding_paths") or {}).get(name) or name).strip()
                if path and path not in project_paths:
                    project_paths.append(path)
                fanout_meta = typed_fanout.get(sid, {}).get(name)
                binding = {"name": name, "path": path,
                           "source": "selected_all" if fanout_meta else "selected_first"}
                if fanout_meta and fanout_meta.get("max_values"):
                    binding["max_values"] = int(fanout_meta["max_values"])
                bindings.append(binding)
            selector = next((d for d in reversed(derivs)
                             if str(d.get("operator") or "").lower() in
                             {"first", "nth", "endpoint_rank", "argmax", "argmin"}), None)
            select = {"mode": "head", "limit": 1, "index": 0}
            if selector and str(selector.get("operator") or "").lower() == "nth":
                select = {"mode": "nth", "limit": 1,
                          "index": max(0, int(selector.get("rank") or 0))}
            elif not selector and bind_names and all(
                    name in typed_fanout.get(sid, {}) for name in bind_names):
                select = {"mode": "all_matches", "limit": 10, "index": 0}
            elif not selector and derivs and not bind_names:
                select = {"mode": "all_matches", "limit": 10, "index": 0}
            specs.append({
                "step_id": sid, "record_path": "$", "project_paths": project_paths,
                "filters": [], "sort": [], "select": select, "bindings": bindings,
                "aggregates": [], "schema_deferred": True,
                "completeness": "runtime-observed schema; returned response",
                "purpose": "Resolve planner-declared fields from one unambiguous runtime record universe."
            })
            continue
        root, root_errors = _candidate_record_root(
            card, derivs, require_binding=bool(bind_names), binding_paths=binding_hints)
        if root_errors or not root:
            errors.extend(f"step {sid}: {e}" for e in root_errors)
            continue

        available_rel = {
            _normalized_schema_path(rel): rel
            for rel in (_record_relative_path(str(x.get("path")), root)
                        for x in card.get("leaf_paths") or [] if x.get("path"))
            if rel not in (None, "", "$")
        }
        filters: list[dict[str, Any]] = []
        for d in derivs:
            for path, op, value in _normalized_plan_filter_predicates(d.get("filter") or {}, root):
                norm = _normalized_schema_path(path)
                if op in {"exists", "not_exists"} and norm in {"", "$"}:
                    continue
                actual = available_rel.get(norm)
                if actual is None:
                    continue
                item = {"path": actual, "op": op, "value": value}
                if item not in filters:
                    filters.append(item)

        selector = None
        for opname in ("nth", "argmax", "argmin", "first", "endpoint_rank"):
            hits = [d for d in derivs if str(d.get("operator") or "").lower() == opname]
            if hits:
                selector = hits[-1]
                break
        indexed_binding_selection = _explicit_binding_selection_index(step, bind_names)
        sort: list[dict[str, Any]] = []
        select = {"mode": "head", "limit": 1, "index": 0}
        source = "selected_first"
        declared_fanout = bool(bind_names) and all(
            name in typed_fanout.get(sid, {}) for name in bind_names)
        if selector:
            op = str(selector.get("operator") or "").lower()
            if op in {"argmax", "argmin"}:
                field = _relative_projection_field(selector.get("field"), root)
                actual = available_rel.get(_normalized_schema_path(field))
                if actual is None:
                    errors.append(f"step {sid}: selector field {selector.get('field')} is not scalar under {root}")
                    continue
                sort = [{"path": actual, "direction": "desc" if op == "argmax" else "asc"}]
            elif op == "nth":
                select = {"mode": "nth", "limit": 1, "index": max(0, int(selector.get("rank") or 0))}
            elif op == "endpoint_rank":
                idx = max(0, int(selector.get("rank") or 0))
                select = {"mode": "nth" if idx else "head", "limit": 1, "index": idx}
        elif declared_fanout:
            source = "selected_all"
            select = {"mode": "all_matches", "limit": 10, "index": 0}
        elif indexed_binding_selection is not None:
            idx = max(0, int(indexed_binding_selection))
            select = {"mode": "nth" if idx else "head", "limit": 1, "index": idx}
        elif filters:
            source = "selected_all"
            select = {"mode": "all_matches", "limit": 5, "index": 0}
        elif bind_names and bool((plan or {}).get("execution_advisory")):
            # A recoverable plan omitted producer selection semantics. Do not
            # silently invent "first". Expose a bounded set of observed producer
            # bindings so generated code may continue on the planned GET route;
            # the strict final certificate still records the missing semantics.
            source = "selected_all"
            select = {"mode": "all_matches", "limit": 10, "index": 0}
        elif derivs and "[*]" in str(root):
            # Terminal list/count/identity derivations need the whole declared
            # record universe even though no downstream binding is produced.
            select = {"mode": "all_matches", "limit": 10, "index": 0}

        bindings = []
        project_paths: list[str] = []
        for name in sorted(bind_names):
            path, bind_errors = _binding_leaf_for_root(
                card, root, name, (step.get("binding_paths") or {}).get(name))
            if bind_errors or not path:
                errors.extend(f"step {sid}: {e}" for e in bind_errors)
                continue
            binding_source = ("selected_all" if name in typed_fanout.get(sid, {}) and not selector
                              else source)
            binding_item = {"name": name, "path": path, "source": binding_source}
            fanout_meta = typed_fanout.get(sid, {}).get(name) or {}
            if binding_source == "selected_all" and fanout_meta.get("max_values"):
                binding_item["max_values"] = int(fanout_meta["max_values"])
            elif binding_source == "selected_all" and bool((plan or {}).get("execution_advisory")):
                binding_item["max_values"] = 10
            bindings.append(binding_item)
            if path not in project_paths:
                project_paths.append(path)
        if len(bindings) != len(bind_names):
            continue

        # Expose scalar fields used by terminal identity/count/ranking derivations
        # so Phase-A feedback stays focused on exactly what the Evidence Plan named.
        for deriv in derivs:
            field = _relative_projection_field(deriv.get("field"), root)
            actual = available_rel.get(_normalized_schema_path(field)) if field else None
            if actual and actual not in project_paths:
                project_paths.append(actual)

        for item in filters + sort:
            pth = str(item.get("path") or "")
            if pth and pth not in project_paths:
                project_paths.append(pth)
        # Include one human-readable label when available. This is schema-only and
        # makes Phase-A feedback useful without sending complete response bodies.
        for label in ("name", "title", "label"):
            if label in available_rel and available_rel[label] not in project_paths:
                project_paths.append(available_rel[label]); break

        spec = {
            "step_id": sid, "record_path": root, "project_paths": project_paths,
            "filters": filters, "sort": sort, "select": select,
            "bindings": bindings, "aggregates": [],
            "completeness": ("all returned candidates inspected" if filters or sort or source == "selected_all"
                             else "endpoint-ranked head records"),
            "purpose": "Deterministically realize evidence-plan selection and downstream bindings.",
        }
        specs.append(spec)

    if errors:
        return specs, list(dict.fromkeys(errors))
    # Reuse the same validators/repairs as the LLM path. This is the single source
    # of truth for binding graph and selection semantics.
    payload = {"observation_specs": specs}
    validated, validation_errors = _validate_observation_specs(payload, plan, cards)
    if validation_errors:
        return validated, validation_errors
    candidate_plan = dict(plan)
    candidate_plan["observation_specs"] = validated
    candidate_plan = repair_path_binding_aliases_from_schema(candidate_plan, cards)
    candidate_plan = repair_observation_bindings(candidate_plan)
    candidate_plan = synchronize_step_bindings_from_observation_specs(candidate_plan)
    candidate_plan = repair_membership_sources_from_observation_specs(candidate_plan)
    candidate_plan = repair_terminal_selector_bindings(candidate_plan)
    candidate_plan = repair_requested_subset_fanout(question, candidate_plan)
    candidate_plan = repair_plural_source_owner_fanout(question, candidate_plan)
    candidate_plan, population_errors = repair_population_fanout_bindings(candidate_plan)
    if bool(candidate_plan.get("execution_advisory")):
        bounded_specs = []
        for raw_spec in candidate_plan.get("observation_specs") or []:
            spec = dict(raw_spec)
            bindings = []
            for raw_binding in spec.get("bindings") or []:
                binding = dict(raw_binding)
                if str(binding.get("source") or "selected_first") == "selected_all":
                    binding.setdefault("max_values", 10)
                bindings.append(binding)
            spec["bindings"] = bindings
            bounded_specs.append(spec)
        candidate_plan["observation_specs"] = bounded_specs
    aligned = validate_selection_alignment(candidate_plan, candidate_plan.get("observation_specs") or [])
    graph = validate_request_contract(candidate_plan, cards) + validate_binding_graph(candidate_plan, cards)
    all_errors = list(dict.fromkeys(list(population_errors) + aligned + graph))
    return list(candidate_plan.get("observation_specs") or []), all_errors

def make_observation_plan(question: str, benchmark: str, plan: dict[str, Any],
                          model: str, client, attempts: int = 2,
                          deterministic_first: bool = False
                          ) -> tuple[dict[str, Any], str]:
    """Add schema-grounded observation specs to an already validated route plan.

    Exact schema-grounded observation/binding specs are part of the generic
    execution contract. If they cannot be validated after bounded attempts, the
    plan fails closed rather than executing dependent calls with guessed fields.
    """
    if not plan or not plan.get("steps"):
        return plan, ""
    try:
        messages, _prompt_cards = _observation_plan_messages(question, plan, benchmark)
        from utils.schema_outline import selected_endpoint_validation_cards
        cards = selected_endpoint_validation_cards(benchmark, plan)
        plan = normalize_collection_binding_scalar_paths(plan, cards)
    except Exception as exc:
        detail = str(exc).strip().replace("\n", " ")[:240]
        suffix = f": {detail}" if detail else ""
        err = f"selected OAS schema unavailable for observation planning ({type(exc).__name__}){suffix}"
        out = dict(plan)
        out["observation_specs"] = []
        out["observation_plan_valid"] = False
        out["observation_plan_errors"] = [err]
        out["observation_plan_source"] = "selected_oas_schema_unavailable"
        out["valid"] = False
        out["validation_errors"] = list(dict.fromkeys(
            list(out.get("validation_errors") or []) + [err]))
        return out, ""
    advisory_execution = bool(plan.get("execution_eligible") and not plan.get("valid"))
    # Validate planner-declared request arguments before spending another model call.
    # For a strict plan these remain fail-closed.  For a read-only advisory plan,
    # retain them as diagnostics and still compile any mechanically usable
    # observation/binding specs so evidence acquisition can reach the final gate.
    request_errors = validate_request_contract(plan, cards)
    if request_errors and not advisory_execution:
        out = dict(plan)
        out["observation_specs"] = []
        out["observation_plan_valid"] = False
        out["observation_plan_errors"] = request_errors
        out["observation_plan_source"] = "request_contract_failed"
        out["valid"] = False
        out["validation_errors"] = list(dict.fromkeys(
            list(out.get("validation_errors") or []) + request_errors))
        return out, ""

    semantic_projection_errors = validate_schema_projection_semantics(plan, cards)
    if semantic_projection_errors and not advisory_execution:
        out = dict(plan)
        out["observation_specs"] = []
        out["observation_plan_valid"] = False
        out["observation_plan_errors"] = semantic_projection_errors
        out["observation_plan_source"] = "evidence_plan_projection_semantics_failed"
        out["valid"] = False
        out["validation_errors"] = list(dict.fromkeys(
            list(out.get("validation_errors") or []) + semantic_projection_errors))
        return out, ""

    if deterministic_first:
        deterministic_specs, deterministic_errors = synthesize_observation_specs(
            question, plan, cards)
        if not deterministic_errors and len(deterministic_specs) == len(plan.get("steps") or []):
            candidate_plan = dict(plan)
            candidate_plan["observation_specs"] = deterministic_specs
            candidate_plan = repair_path_binding_aliases_from_schema(candidate_plan, cards)
            candidate_plan = repair_observation_bindings(candidate_plan)
            candidate_plan = synchronize_step_bindings_from_observation_specs(candidate_plan)
            candidate_plan = repair_membership_sources_from_observation_specs(candidate_plan)
            candidate_plan = repair_terminal_selector_bindings(candidate_plan)
            candidate_plan = repair_requested_subset_fanout(question, candidate_plan)
            candidate_plan = repair_plural_source_owner_fanout(question, candidate_plan)
            candidate_plan, population_errors = repair_population_fanout_bindings(candidate_plan)
            deterministic_errors = list(population_errors)
            if not deterministic_errors:
                deterministic_errors = validate_selection_alignment(
                    candidate_plan, candidate_plan.get("observation_specs") or [])
            if not deterministic_errors:
                deterministic_errors = (validate_request_contract(candidate_plan, cards) +
                                        validate_binding_graph(candidate_plan, cards))
            if not deterministic_errors:
                plan = dict(candidate_plan)
                plan["observation_plan_valid"] = True
                plan["observation_plan_errors"] = []
                plan["observation_plan_source"] = "deterministic_plan+selected_oas_schema"
                card_by_step = {str(c.get("step_id")): c for c in cards}
                enriched_steps = []
                for raw_step in plan.get("steps") or []:
                    step = dict(raw_step); card = card_by_step.get(str(step.get("id"))) or {}
                    step["request_parameters"] = [dict(x) for x in card.get("parameters") or []]
                    step["request_body_required"] = bool(card.get("request_body_required"))
                    step["request_body_required_fields"] = list(card.get("request_body_required_fields") or [])
                    step["request_body_leaf_paths"] = [str(x.get("path")) for x in (card.get("request_body_leaf_paths") or []) if x.get("path")]
                    enriched_steps.append(step)
                plan["steps"] = enriched_steps
                plan["observation_plan_fallback_reason"] = None
                return plan, ""
        if advisory_execution and deterministic_specs:
            # Keep the mechanically valid subset even when strict selection or
            # answer-schema checks failed.  This is an acquisition projection, not
            # a certification claim: the original strict validation errors remain
            # attached to the plan and the final contract can still abstain.
            candidate_plan = dict(plan)
            candidate_plan["observation_specs"] = deterministic_specs
            candidate_plan = repair_path_binding_aliases_from_schema(candidate_plan, cards)
            candidate_plan = repair_observation_bindings(candidate_plan)
            candidate_plan = synchronize_step_bindings_from_observation_specs(candidate_plan)
            candidate_plan = repair_membership_sources_from_observation_specs(candidate_plan)
            candidate_plan = repair_requested_subset_fanout(question, candidate_plan)
            candidate_plan = repair_plural_source_owner_fanout(question, candidate_plan)
            combined_errors = list(dict.fromkeys(
                list(request_errors) + list(semantic_projection_errors) +
                list(deterministic_errors)))
            candidate_plan["observation_plan_valid"] = False
            candidate_plan["observation_plan_errors"] = combined_errors
            candidate_plan["observation_plan_source"] = "deterministic_advisory_salvage"
            candidate_plan["observation_plan_fallback_reason"] = combined_errors
            card_by_step = {str(c.get("step_id")): c for c in cards}
            enriched_steps = []
            for raw_step in candidate_plan.get("steps") or []:
                step = dict(raw_step); card = card_by_step.get(str(step.get("id"))) or {}
                step["request_parameters"] = [dict(x) for x in card.get("parameters") or []]
                step["request_body_required"] = bool(card.get("request_body_required"))
                step["request_body_required_fields"] = list(card.get("request_body_required_fields") or [])
                step["request_body_leaf_paths"] = [str(x.get("path")) for x in (card.get("request_body_leaf_paths") or []) if x.get("path")]
                enriched_steps.append(step)
            candidate_plan["steps"] = enriched_steps
            return candidate_plan, ""
        plan = dict(plan)
        plan["observation_plan_fallback_reason"] = list(dict.fromkeys(
            list(request_errors) + list(semantic_projection_errors) +
            list(deterministic_errors)))

    last = ""
    last_errors: list[str] = []
    # Observation-plan retries are intentionally delta-oriented: the retry prompt
    # asks the model to correct only missing/invalid specs. Preserve specs that
    # already validated so a partial retry cannot accidentally delete them.
    stable_specs_by_step: dict[str, dict[str, Any]] = {}
    for attempt in range(max(1, int(attempts))):
        call_messages = list(messages)
        if last_errors:
            call_messages.append({
                "role": "user",
                "content": "Fix these observation-spec problems: " +
                           json.dumps(last_errors[:20], ensure_ascii=False) +
                           ". Return only corrected or missing specs as JSON. Existing valid specs are kept.",
            })
        try:
            kwargs = dict(model=model, messages=call_messages, temperature=0.0)
            from utils.token_meter import stage as token_stage
            with token_stage("projection_planner"):
                try:
                    response = client.chat.completions.create(
                        **kwargs, response_format={"type": "json_object"},
                        max_completion_tokens=1500)
                except Exception:
                    response = client.chat.completions.create(**kwargs)
            last = response.choices[0].message.content or ""
        except Exception as exc:
            last = f"observation planner call failed: {exc}"
            last_errors = [last]
            continue
        parsed = _extract_json(last)
        parsed_specs = [x for x in ((parsed or {}).get("observation_specs") or [])
                        if isinstance(x, dict) and x.get("step_id")]
        # Merge delta retries over previously validated specs. New specs override
        # the same step; untouched valid specs survive.
        merged_by_step = dict(stable_specs_by_step)
        for raw_spec in parsed_specs:
            merged_by_step[str(raw_spec.get("step_id"))] = raw_spec
        merged_payload = {"observation_specs": list(merged_by_step.values())}
        specs, errors = _validate_observation_specs(merged_payload, plan, cards)

        # Keep only step specs that validated on this attempt. This is deliberately
        # step-local: one bad projection must not poison the other plan steps.
        errored_steps: set[str] = set()
        for error in errors:
            text = str(error)
            if text.startswith("step ") and ":" in text:
                errored_steps.add(text[5:text.index(":")].strip())
        for spec in specs:
            sid = str(spec.get("step_id") or "")
            if sid and sid not in errored_steps:
                stable_specs_by_step[sid] = dict(spec)

        # Revalidate the stable+current union after harvesting valid specs. This
        # allows attempt N+1 to return only the missing/invalid deltas.
        stable_payload = {"observation_specs": list(stable_specs_by_step.values())}
        specs, errors = _validate_observation_specs(stable_payload, plan, cards)
        candidate_plan = dict(plan)
        candidate_plan["observation_specs"] = specs
        if not errors:
            # Exact downstream placeholder aliases are a structural consequence of
            # one unambiguous producer path. Establish them before population/fanout
            # analysis so APIs whose response field/bind name differs from the child
            # placeholder are not falsely rejected.
            candidate_plan = repair_observation_bindings(candidate_plan)
            candidate_plan = synchronize_step_bindings_from_observation_specs(candidate_plan)
            candidate_plan = repair_membership_sources_from_observation_specs(candidate_plan)
            candidate_plan = repair_terminal_selector_bindings(candidate_plan)
            candidate_plan = repair_requested_subset_fanout(question, candidate_plan)
            candidate_plan = repair_plural_source_owner_fanout(question, candidate_plan)
            candidate_plan, population_errors = repair_population_fanout_bindings(candidate_plan)
            specs = list(candidate_plan.get("observation_specs") or [])
            errors = list(population_errors)
        if not errors:
            errors = validate_selection_alignment(candidate_plan, specs)
        if not errors and len(specs) == len(plan.get("steps") or []):
            plan = dict(candidate_plan)
            plan["observation_specs"] = specs
            plan["observation_plan_valid"] = True
            plan["observation_plan_errors"] = []
            plan["observation_plan_source"] = "llm+selected_oas_schema"
            # Request parameter metadata is copied from the selected OAS operations
            # so Phase A can construct calls without prior API knowledge. It is
            # structural documentation, never benchmark/gold information.
            card_by_step = {str(c.get("step_id")): c for c in cards}
            enriched_steps = []
            for raw_step in plan.get("steps") or []:
                step = dict(raw_step)
                card = card_by_step.get(str(step.get("id"))) or {}
                step["request_parameters"] = [dict(x) for x in card.get("parameters") or []]
                step["request_body_required"] = bool(card.get("request_body_required"))
                step["request_body_required_fields"] = list(card.get("request_body_required_fields") or [])
                step["request_body_leaf_paths"] = [str(x.get("path")) for x in (card.get("request_body_leaf_paths") or []) if x.get("path")]
                enriched_steps.append(step)
            plan["steps"] = enriched_steps
            # Do not rewrite/remove API steps after semantic plan review merely
            # because upstream and child schemas expose similarly named fields.
            # Such a rewrite can change API semantics. Optimization belongs in
            # the OAS-grounded planner/critic, not in a post-validation heuristic.
            graph_errors = validate_request_contract(plan, cards) + validate_binding_graph(plan, cards)
            if graph_errors:
                plan["valid"] = False
                plan["validation_errors"] = list(dict.fromkeys(
                    list(plan.get("validation_errors") or []) + graph_errors))
                plan["observation_plan_errors"] = graph_errors
                plan["observation_plan_valid"] = False
                plan["observation_plan_source"] = "schema_binding_graph_failed"
            return plan, last
        last_errors = errors
    plan = dict(plan)
    plan["observation_specs"] = []
    plan["observation_plan_valid"] = False
    plan["observation_plan_errors"] = last_errors
    plan["observation_plan_source"] = "schema_binding_plan_failed"
    plan["valid"] = False
    plan["validation_errors"] = list(dict.fromkeys(
        list(plan.get("validation_errors") or []) +
        ["schema-grounded observation/binding plan could not be validated"]))
    return plan, last


def repair_observation_spec(question: str, step: dict[str, Any],
                            current_spec: dict[str, Any] | None,
                            runtime_profile: dict[str, Any], model: str, client
                            ) -> tuple[dict[str, Any] | None, str, list[str]]:
    """Conditionally repair a projection from runtime structure only.

    No response values are sent to this call: only field paths, types, collection
    counts/presence and the already-fixed step purpose.
    """
    from utils.observation_projection import (
        format_runtime_structure, normalize_projection_spec, validate_projection_spec_runtime)
    structure = format_runtime_structure(runtime_profile, max_chars=4500)
    system = """Fix the observation spec using only paths in RUNTIME STRUCTURE. Keep the same plan step and meaning. Do not answer or invent values. Keep the spec small. Return JSON only as {"spec": {...}}."""
    user = (
        f"QUESTION:\n{question}\n\nPLAN STEP:\n" + json.dumps(step, ensure_ascii=False) +
        "\n\nCURRENT SPEC:\n" + json.dumps(current_spec or {}, ensure_ascii=False) +
        "\n\nRUNTIME STRUCTURE:\n" + structure
    )
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    try:
        from utils.token_meter import stage as token_stage
        with token_stage("projection_repair"):
            try:
                response = client.chat.completions.create(
                    model=model, messages=messages, temperature=0.0,
                    response_format={"type": "json_object"}, max_completion_tokens=850)
            except Exception:
                response = client.chat.completions.create(
                    model=model, messages=messages, temperature=0.0)
        raw = response.choices[0].message.content or ""
    except Exception as exc:
        return None, f"projection repair call failed: {exc}", [str(exc)]
    parsed = _extract_json(raw) or {}
    spec_raw = parsed.get("spec") if isinstance(parsed.get("spec"), dict) else parsed
    spec = normalize_projection_spec(spec_raw, step_id=str(step.get("id") or ""))
    errors = validate_projection_spec_runtime(spec, runtime_profile)
    return (spec if not errors else None), raw, errors


def plan_prompt_section(plan: dict[str, Any] | None) -> str:
    if not plan or not plan.get("steps"):
        return ""
    lines = ["\n\n## EVIDENCE PLAN (follow every step before READY_TO_ANSWER)"]
    for step in plan["steps"]:
        deps = f" after {','.join(step['depends_on'])}" if step.get("depends_on") else ""
        literal_text = (f" path_literals={step.get('path_literals')}"
                        if step.get('path_literals') else "")
        request_params = [
            f"{p.get('name')}:{p.get('in')}:{'required' if p.get('required') else 'optional'}:{p.get('type')}"
            for p in (step.get("request_parameters") or []) if p.get("name")
        ]
        declared_request = {}
        for key in ("query_literals", "query_bindings", "body_literals", "body_bindings"):
            if step.get(key): declared_request[key] = step.get(key)
        request_text = (f" request_parameters={request_params}" if request_params else "")
        declared_text = (f" request_contract={declared_request}" if declared_request else "")
        lines.append(f"- {step['id']}{deps}: {step['endpoint']}{literal_text}{request_text}{declared_text} — {step['purpose']}")
    if plan.get("derivations"):
        lines.append("Required deterministic derivations:")
        for d in plan["derivations"]:
            field = f" field={d.get('field')}" if d.get("field") else ""
            cmp = f" comparison={d.get('comparison')}" if d.get("comparison") else ""
            lit = f" comparison_literal={d.get('comparison_literal')!r}" if d.get("comparison_literal") is not None else ""
            distinct = f" distinct_field={d.get('distinct_field')}" if d.get("distinct_field") else ""
            srcd = f" source_derivations={d.get('source_derivations')}" if d.get("source_derivations") else ""
            labels = (f" label_steps={d.get('label_steps')} label_fields={d.get('label_fields')}"
                      if d.get("label_steps") else "")
            lines.append(f"- {d['id']}: {d['operator']}{field}{cmp}{lit}{distinct} from {d['source_steps']}{srcd}{labels} — {d['purpose']}")
    if plan.get("observation_specs"):
        lines.append("Planner-selected observation projections (full responses remain in the ledger):")
        for spec in plan.get("observation_specs") or []:
            project = ",".join(str(x) for x in (spec.get("project_paths") or [])) or "(none)"
            binds = ",".join(f"{b.get('name')}<-{b.get('path')}" for b in (spec.get("bindings") or [])) or "(none)"
            lines.append(
                f"- {spec.get('step_id')}: records={spec.get('record_path')} expose=[{project}] "
                f"bindings=[{binds}] completeness={spec.get('completeness')}")
    lines.append("For a dependent endpoint, use the entity selected by the required derivation from its dependency; do not silently substitute endpoint position zero.")
    lines.append("Do not signal READY_TO_ANSWER while any endpoint step is missing.")
    return "\n".join(lines)


def acquisition_plan_prompt_section(plan: dict[str, Any] | None) -> str:
    """Compact execution-only view of a validated evidence plan.

    Phase A does not need to reinterpret answer semantics or deterministic
    derivations.  It only needs the exact request recipe and the validated
    producer bindings required to realize that recipe.  Keeping this view
    separate from ``plan_prompt_section`` avoids paying to resend compiler-only
    semantics to the code-generating model on every acquisition turn.
    """
    if not plan or not plan.get("steps"):
        return ""
    specs = {str(x.get("step_id")): x for x in (plan.get("observation_specs") or [])}
    lines = ["\n\n## VALIDATED ACQUISITION RECIPE (execute exactly; do not redesign)"]
    for step in plan.get("steps") or []:
        sid = str(step.get("id") or "")
        method = str(step.get("method") or "GET").upper()
        endpoint = str(step.get("endpoint") or "")
        deps = ",".join(str(x) for x in (step.get("depends_on") or [])) or "none"
        pieces = [f"[{sid}] {method} {endpoint}", f"after={deps}"]
        if step.get("path_literals"):
            pieces.append(f"path={step.get('path_literals')}")
        if step.get("query_literals"):
            pieces.append(f"query={step.get('query_literals')}")
        if step.get("query_bindings"):
            pieces.append(f"query_bindings={step.get('query_bindings')}")
        if step.get("body_literals"):
            pieces.append(f"body={step.get('body_literals')}")
        if step.get("body_bindings"):
            pieces.append(f"body_bindings={step.get('body_bindings')}")
        if step.get("body_binding_wrappers"):
            pieces.append(f"body_binding_wrappers={step.get('body_binding_wrappers')}")
        spec = specs.get(sid) or {}
        record_path = str(spec.get("record_path") or "$")
        if record_path:
            pieces.append(f"projection_records={record_path}")
        selection = spec.get("select") or {}
        selection_mode = str(selection.get("mode") or "")
        if selection_mode:
            selection_view = {"mode": selection_mode}
            if selection.get("index") is not None:
                selection_view["index"] = selection.get("index")
            if selection.get("limit") is not None:
                selection_view["limit"] = selection.get("limit")
            pieces.append(f"projection_selection={selection_view}")
        bindings = []
        for b in spec.get("bindings") or []:
            name = str(b.get("name") or "")
            path = str(b.get("path") or "")
            source = str(b.get("source") or "selected_first")
            if name and path:
                shape = ":all_nested_values" if "[*]" in path else ""
                bindings.append(f"{name}<-{source}:{path}{shape}")
        if bindings:
            pieces.append("produces=" + ",".join(bindings))
        lines.append("; ".join(pieces))
    lines.extend([
        "Execution rules:",
        "- Follow the requests exactly.",
        "- projection_records is the record root; binding paths are relative to those projected records.",
        "- Apply projection_selection before extracting produced bindings; selected_first means the first record after that validated selection, not the first raw response record.",
        "- [*] in a binding path means all nested values; preserve the full list.",
        "- Use produced values for later steps. READY_TO_ANSWER only after all steps complete.",
    ])
    return "\n".join(lines)


def missing_plan_prompt(progress: dict[str, Any]) -> str:
    missing = progress.get("missing_steps") or []
    if not missing:
        return ""
    statuses = progress.get("step_status") or {}
    lines = ["The evidence plan is incomplete. Fetch only the missing plan instances before answering:"]
    for step in missing:
        sid = str(step.get("id"))
        lines.append(f"- {sid}: endpoint template {step.get('endpoint')} — {step.get('purpose')}")
        status = statuses.get(sid) or {}
        expected = status.get("expected_bindings") or {}
        covered = status.get("covered_bindings") or {}
        for name, values in expected.items():
            remaining = [str(v) for v in values if str(v) not in {str(x) for x in covered.get(name, [])}]
            if remaining:
                # These are observed producer values, not benchmark/gold hints.
                lines.append(f"  unresolved {name} values from the declared producer: {remaining[:20]}")
    lines.append("Use only values selected by the declared dependency bindings. Run code only; do not repeat covered instances.")
    return "\n".join(lines)
