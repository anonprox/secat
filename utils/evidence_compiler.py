"""Generic deterministic evidence compiler for OCA.

The compiler is deliberately API-agnostic.  It never branches on benchmark
names, endpoint tokens, task IDs, media/entity types, or benchmark answers.  It
replays only the validated evidence plan and (when available) its validated
observation-projection specification.
"""
from __future__ import annotations

import calendar as _calendar
import datetime as _dt
import re
from collections import defaultdict
from typing import Any

from utils.predicate_semantics import compare_predicate


def _stable_value_key(value: Any):
    """Return a hashable, type-preserving key for arbitrary JSON-like values.

    Evidence fields are usually scalars, but generic APIs may expose objects or
    arrays at a selected field. Host replay must fail closed or preserve them; it
    must never crash merely because a value is unhashable.
    """
    if isinstance(value, dict):
        return ("dict", tuple(sorted(
            ((str(k), _stable_value_key(v)) for k, v in value.items()),
            key=lambda item: item[0],
        )))
    if isinstance(value, (list, tuple)):
        return ("list", tuple(_stable_value_key(v) for v in value))
    if isinstance(value, set):
        frozen = [_stable_value_key(v) for v in value]
        return ("set", tuple(sorted(frozen, key=repr)))
    try:
        hash(value)
    except Exception:
        return ("repr", type(value).__name__, repr(value))
    return ("scalar", type(value).__name__, value)


def _unique_values(values: list[Any]) -> list[Any]:
    """Stable de-duplication for scalar or structured evidence values."""
    out: list[Any] = []
    seen = set()
    for value in values:
        key = _stable_value_key(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _date(value: Any):
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return _dt.date.fromisoformat(value.strip()[:10])
    except Exception:
        return None


def _date_bounds(value: Any):
    """Return the closed interval represented by an ISO date value.

    APIs commonly expose dates at day, month, or year precision.  Treating the
    partial forms as ordinary strings makes an otherwise homogeneous date
    population look mixed (``date`` versus ``str``), so a sound argmax/argmin
    cannot be replayed.  Bounds preserve the provider's stated precision: a
    year covers that whole year and a month covers that whole month.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    match = re.fullmatch(r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?", text[:10])
    if not match:
        return None
    try:
        year = int(match.group(1))
        month_text = match.group(2)
        day_text = match.group(3)
        if month_text is None:
            return _dt.date(year, 1, 1), _dt.date(year, 12, 31)
        month = int(month_text)
        if day_text is None:
            last_day = _calendar.monthrange(year, month)[1]
            return _dt.date(year, month, 1), _dt.date(year, month, last_day)
        day = int(day_text)
        exact = _dt.date(year, month, day)
        return exact, exact
    except Exception:
        return None


def _normalized_observation_path(path: Any) -> str:
    """Normalize schema list-index syntax to normalized-observation relations.

    Raw schemas/plans may spell a selected list item as ``cast[0].id`` while the
    observation ledger stores that item as relation=cast, fields={id,...}. Numeric
    indices are selection syntax, not field names, once list items are normalized.
    """
    text = str(path or "").strip().replace("$.", "")
    text = re.sub(r"\[(?:\*|\d*)\]", "", text)
    return text


def _field(obs: dict[str, Any], name: str):
    """Read a dotted field path from a normalized observation.

    Normalization lifts each list item into its own observation. If a schema/plan
    path includes that list relation as a prefix (e.g. ``entries.status``), drop
    only the exact observed relation prefix before traversing the child fields.
    """
    value: Any = obs.get("fields") or {}
    root_fields = value if isinstance(value, dict) else {}
    path = _normalized_observation_path(name)
    parts = [x for x in path.split(".") if x]
    relation = str(obs.get("relation") or "")
    if relation and relation in parts:
        # Nested arrays are normalized into child observations. A planner may
        # still name the full schema path (e.g. episodes.crew.name) while the
        # child observation is relation=crew with fields={name,...}. Anchor the
        # traversal at the deepest matching normalized relation rather than
        # requiring that relation to be the first path segment.
        idx = len(parts) - 1 - parts[::-1].index(relation)
        parts = parts[idx + 1:]
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value.get(part)
        else:
            # Normalization may already have lifted a nested schema record into a
            # flat observation whose fields contain only the terminal leaf. When
            # the full planner path cannot be traversed, the same-record terminal
            # leaf is a safe canonical fallback; it cannot switch records/entities.
            leaf = parts[-1] if parts else ""
            return root_fields.get(leaf) if leaf and leaf in root_fields else None
    return value


def _candidate_payload(obs: dict[str, Any]) -> dict[str, Any] | Any:
    fields = obs.get("fields") or {}
    out: dict[str, Any] = {}
    for key, value in fields.items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
        elif isinstance(value, list) and len(value) <= 30 and all(
                isinstance(x, (str, int, float, bool)) or x is None for x in value):
            out[key] = value
    return out or fields


def _projected_payload(obs: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Materialize only fields explicitly exposed by the validated projection."""
    out: dict[str, Any] = {}
    for path in spec.get("project_paths") or []:
        path = str(path)
        value = _field(obs, path)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (str, int, float, bool)):
            out[path] = value
        elif isinstance(value, list) and len(value) <= 30 and all(
                isinstance(x, (str, int, float, bool)) or x is None for x in value):
            out[path] = value
    return out


def _step_groups(ledger):
    groups = defaultdict(list)
    for obs in ledger.observations:
        if obs.get("plan_step_id"):
            groups[str(obs["plan_step_id"])].append(obs)
    return groups


def _select_by_field(records: list[dict[str, Any]], field: str, reverse: bool):
    valid = []
    date_values = []
    for obs in records:
        value = _field(obs, field)
        bounds = _date_bounds(value)
        if bounds is not None:
            date_values.append((bounds, obs))
            valid.append((bounds, obs))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            valid.append((float(value), obs))
        elif value not in (None, ""):
            valid.append((str(value), obs))
    if not valid:
        return None, []
    if len(date_values) == len(valid):
        # Rank only when the winning precision interval is distinguishable from
        # every differently-valued competitor.  Exact ties retain stable API
        # order; overlapping partial dates fail closed instead of inventing a day.
        if reverse:
            ordered = sorted(date_values, key=lambda x: x[0][0], reverse=True)
            winner_bounds = ordered[0][0]
            ambiguous = any(
                bounds != winner_bounds and bounds[1] >= winner_bounds[0]
                for bounds, _ in ordered[1:])
        else:
            ordered = sorted(date_values, key=lambda x: x[0][1])
            winner_bounds = ordered[0][0]
            ambiguous = any(
                bounds != winner_bounds and bounds[0] <= winner_bounds[1]
                for bounds, _ in ordered[1:])
        candidates = [obs for _, obs in ordered]
        return (None, candidates) if ambiguous else (candidates[0], candidates)
    # Values from heterogeneous APIs may mix numeric/date/string encodings.
    # Never let Python cross-type ordering choose or crash a winner. When the
    # comparison domain is mixed, fail closed so the certificate cannot claim
    # a deterministic argmax/argmin that was not well-defined.
    kinds = {type(x[0]) for x in valid}
    if len(kinds) != 1:
        return None, [x[1] for x in valid]
    try:
        valid.sort(key=lambda x: x[0], reverse=reverse)
    except Exception:
        return None, [x[1] for x in valid]
    return valid[0][1], [x[1] for x in valid]


def _filter_records(records: list[dict[str, Any]], filt: dict[str, Any]):
    if not filt:
        return records
    out = []
    for obs in records:
        ok = True
        for key, expected in filt.items():
            actual = _field(obs, key)
            if isinstance(expected, dict) and expected.get("op"):
                ok = _compare_projection(actual, str(expected.get("op")), expected.get("value"))
            elif isinstance(expected, list):
                ok = actual in expected
            elif isinstance(expected, str) and isinstance(actual, str):
                ok = expected.strip().casefold() == actual.strip().casefold()
            else:
                ok = actual == expected
            if not ok:
                break
        if ok:
            out.append(obs)
    return out


def _canonical_filter_for_replay(spec: dict[str, Any]) -> dict[str, Any]:
    """Use the planner's validated canonical predicate dialect for host replay."""
    from utils.evidence_plan import _canonical_derivation_filter
    filt = spec.get("filter") if isinstance(spec.get("filter"), dict) else {}
    return _canonical_derivation_filter(filt, spec.get("field"))


def _resolved_filter_for_replay(spec: dict[str, Any], plan: dict[str, Any],
                                by_step: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Resolve accidental symbolic predicate literals from declared upstream bindings.

    Route planners occasionally express a dynamic exclusion such as
    ``id != person_id`` using the binding alias string as the predicate value.  The
    plan has already declared which upstream step produces that alias, so host replay
    can resolve it without guessing or using benchmark knowledge.  Resolution is
    deliberately narrow: the value must exactly name a declared binding and that
    binding must resolve to one unique scalar in the selected upstream evidence.
    """
    filt = _canonical_filter_for_replay(spec)
    if not filt or not plan or not by_step:
        return filt

    producers: dict[str, tuple[str, str]] = {}
    for step in (plan or {}).get("steps") or []:
        sid = str(step.get("id") or "")
        bpaths = step.get("binding_paths") if isinstance(step.get("binding_paths"), dict) else {}
        obs_spec = _observation_spec(plan, sid)
        obs_bindings = {str(b.get("name") or ""): str(b.get("path") or "")
                        for b in (obs_spec or {}).get("bindings") or []
                        if isinstance(b, dict) and b.get("name")}
        for alias in set([str(x) for x in step.get("binds") or []] +
                         [str(x) for x in bpaths] + list(obs_bindings)):
            path = obs_bindings.get(alias) or str(bpaths.get(alias) or "")
            if alias and path:
                producers.setdefault(alias, (sid, path))

    def resolve_alias(value: Any) -> Any:
        if not isinstance(value, str) or value not in producers:
            return value
        sid, path = producers[value]
        rows = list(by_step.get(sid, []))
        obs_spec = _observation_spec(plan, sid)
        if obs_spec:
            _all, selected = _records_from_projection_spec(rows, obs_spec)
            if selected:
                rows = list(selected)
        vals = []
        for row in rows:
            v = _field(row, path)
            if v in (None, ""):
                # Observation-plan binding paths are usually normalized relative
                # to record_path, whereas planner binding_paths may retain the
                # collection prefix. Try the canonical selected-step alias.
                v = _field(row, _canonical_step_field(plan, sid, path))
            if v not in (None, "") and v not in vals:
                vals.append(v)
        return vals[0] if len(vals) == 1 else value

    out = {}
    for key, expected in filt.items():
        if isinstance(expected, dict) and expected.get("op"):
            e = dict(expected)
            e["value"] = resolve_alias(e.get("value"))
            out[key] = e
        else:
            out[key] = resolve_alias(expected)
    return out


def _add_selection(ledger, *, name: str, operation: str,
                   candidates: list[dict[str, Any]], chosen: dict[str, Any],
                   comparison_fields: list[str], policy: str,
                   plan_derivation_id: str | None = None,
                   value: Any = None) -> str:
    extra = {}
    if plan_derivation_id:
        extra["plan_derivation_id"] = str(plan_derivation_id)
    return ledger.add_derived_record(
        name=name,
        value=_candidate_payload(chosen) if value is None else value,
        operation=operation,
        input_obs_ids=[o["obs_id"] for o in candidates],
        selected_obs_id=chosen["obs_id"],
        candidate_obs_ids=[o["obs_id"] for o in candidates],
        comparison_fields=comparison_fields,
        policy=policy,
        confidence="high",
        source_call_id=chosen.get("call_id"),
        selected_record_id=chosen.get("record_id"),
        plan_step_id=chosen.get("plan_step_id"),
        **extra,
    )


def _relation_from_record_path(path: str) -> str | None:
    matches = re.findall(r"(?:^|\.)([A-Za-z_][A-Za-z0-9_]*)\[\*\]", str(path or ""))
    return matches[-1] if matches else None


def _compare_projection(value: Any, op: str, expected: Any) -> bool:
    """Compatibility wrapper around the canonical host predicate semantics."""
    return compare_predicate(value, op, expected)


def _filter_projection(records, filters):
    out = list(records)
    for item in filters or []:
        out = [obs for obs in out if _compare_projection(
            _field(obs, item.get("path")), item.get("op"), item.get("value"))]
    return out


def _sort_projection(records, sort_specs):
    ordered = list(records)
    # Missing values stay last in both directions.  A drifted payload that mixes
    # incompatible scalar domains must *not* be coerced to strings: doing so can
    # silently change a numeric/date ordering into lexicographic ordering and can
    # authorize the wrong downstream binding.  Fail closed by returning no ordered
    # records when an explicitly requested sort is not well-defined.
    for spec in reversed(sort_specs or []):
        path = str(spec.get("path") or "")
        reverse = str(spec.get("direction") or "asc").lower().startswith("d")
        present = [row for row in ordered if _field(row, path) is not None]
        missing = [row for row in ordered if _field(row, path) is None]
        domains = set()
        for row in present:
            value = _field(row, path)
            if isinstance(value, bool):
                domains.add("bool")
            elif isinstance(value, (int, float)):
                domains.add("number")
            elif isinstance(value, str):
                domains.add("string")
            else:
                domains.add("other")
        if len(domains) > 1 or "other" in domains:
            return []
        try:
            present.sort(key=lambda row: _field(row, path), reverse=reverse)
        except Exception:
            return []
        ordered = present + missing
    return ordered


def _select_projection(ordered, select):
    select = select or {}
    mode = str(select.get("mode") or "head").lower()
    index = max(0, int(select.get("index", 0) or 0))
    limit = max(1, int(select.get("limit", 5) or 5))
    if mode in {"head", "top", "all_matches"}: return list(ordered[:limit])
    if mode == "tail": return list(ordered[-limit:])
    if mode == "nth": return list(ordered[index:index + 1])
    if mode == "single": return list(ordered[:1])
    return list(ordered[:limit])


def _observation_spec(plan: dict[str, Any] | None, step_id: str) -> dict[str, Any]:
    for spec in (plan or {}).get("observation_specs") or []:
        if str(spec.get("step_id") or "") == str(step_id):
            return spec
    return {}


def _record_path_pointer_regex(record_path: str) -> re.Pattern | None:
    """Translate the small schema JSONPath dialect into an exact JSON pointer.

    The ledger keeps a precise ``json_pointer`` for every normalized record.  A
    relation name alone is insufficient because unrelated branches can legally
    contain arrays with the same key (for example ``left.items`` and
    ``right.items``).  Using the full pointer keeps host replay aligned with the
    trusted runtime, which evaluates the raw payload at the exact record_path.
    """
    text = str(record_path or "$").strip()
    if text in {"", "$"}:
        return re.compile(r"^/$")
    text = text.replace("$.", "")
    parts = [p for p in text.split(".") if p and p != "$"]
    chunks = ["^"]
    for part in parts:
        if part == "[*]":
            chunks.append(r"/\d+")
            continue
        if part.endswith("[*]"):
            key = part[:-3]
            chunks.append("/" + re.escape(key) + r"/\d+")
            continue
        chunks.append("/" + re.escape(part))
    chunks.append("$")
    try:
        return re.compile("".join(chunks))
    except re.error:
        return None


def _record_matches_projection_root(obs: dict[str, Any], record_path: str) -> bool:
    pattern = _record_path_pointer_regex(record_path)
    pointer = str(obs.get("json_pointer") or "")
    if pattern is not None and pointer:
        return bool(pattern.match(pointer))
    # Compatibility fallback for imported historical ledgers that predate exact
    # json_pointer capture.  New runs always use the pointer path above.
    relation = _relation_from_record_path(record_path)
    if relation:
        return str(obs.get("relation") or "") == relation
    if str(record_path or "$") == "$":
        return obs.get("kind") in {"record", "action_result"} and not obs.get("relation")
    return False


def _records_from_projection_spec(records: list[dict[str, Any]], spec: dict[str, Any]):
    records = list(records)
    record_path = str(spec.get("record_path") or "$")
    matched = [o for o in records if _record_matches_projection_root(o, record_path)]
    # A validated [*]/object record path that produces no normalized records is
    # an empty result, not permission to fall back to unrelated observations.
    records = matched
    records = _filter_projection(records, spec.get("filters") or [])
    ordered = _sort_projection(records, spec.get("sort") or [])
    selected = _select_projection(ordered, spec.get("select") or {})
    return ordered, selected





def _deferred_runtime_record_universe(records: list[dict[str, Any]],
                                      spec: dict[str, Any],
                                      derivation: dict[str, Any] | None = None
                                      ) -> list[dict[str, Any]]:
    """Resolve one runtime record universe for an OAS-undocumented response.

    Only normalized structural provenance and planner-declared field names are
    considered.  If the same field set exists under multiple structural roots,
    return no candidates so compilation fails closed.
    """
    if not (spec or {}).get("schema_deferred"):
        return list(records)
    wanted = [str(x) for x in (spec.get("project_paths") or []) if str(x) and str(x) != "$"]
    if derivation:
        field = str(derivation.get("field") or "").strip()
        if field and field != "$": wanted.append(field)
        wanted.extend(str(x) for x in (derivation.get("filter") or {}) if str(x))
    wanted = list(dict.fromkeys(wanted))

    def structural_key(row):
        pointer = str(row.get("json_pointer") or "/")
        pointer = re.sub(r"/\d+(?=/|$)", "/*", pointer)
        return pointer, str(row.get("relation") or "$")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in records:
        if row.get("kind") not in {"list_item", "nested_item", "record", "nested_object", "action_result"}:
            continue
        groups.setdefault(structural_key(row), []).append(row)

    compatible = []
    for key, rows in groups.items():
        if not wanted:
            continue
        ok = True
        for raw in wanted:
            # Full planner paths may include the runtime relation prefix. _field()
            # already supports that prefix convention, so use it directly.
            if not any(_field(row, raw) not in (None, "") for row in rows):
                ok = False; break
        if ok:
            compatible.append((key, rows))
    if len(compatible) != 1:
        return []
    return list(compatible[0][1])

def _records_for_derivation(records: list[dict[str, Any]], obs_spec: dict[str, Any],
                            derivation: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the normalized record universe addressed by a derivation path.

    Observation specs may root at an outer collection (``episodes[*]``) while
    projecting fields from a nested normalized child collection
    (``crew[*].name``).  The runtime evaluates that nested path over the raw
    payload; deterministic replay must therefore use the corresponding descendant
    child observations, but only beneath the declared outer record path.
    """
    records = list(records)
    if not obs_spec:
        return records
    if obs_spec.get("schema_deferred"):
        return _deferred_runtime_record_universe(records, obs_spec, derivation)
    relation_names = {str(o.get("relation") or "") for o in records if o.get("relation")}
    derivation_paths = [str(derivation.get("field") or "")] + [
        str(x) for x in (derivation.get("filter") or {}).keys()]
    parts: list[str] = []
    for raw in derivation_paths:
        bits = [x for x in _normalized_observation_path(raw).split(".") if x]
        if not bits:
            continue
        # Usually the final segment is a scalar field and the preceding segment
        # names the normalized child relation (episodes.crew.id -> crew).  But a
        # planner may also explicitly address the relation/container itself
        # (episodes.crew).  When that final segment is an observed relation, keep
        # it as the target instead of discarding it as if it were a scalar field.
        if bits[-1] in relation_names:
            parts.extend(bits)
        else:
            parts.extend(bits[:-1] if len(bits) > 1 else bits)
    matches = [x for x in parts if x in relation_names]
    root_relation = _relation_from_record_path(str(obs_spec.get("record_path") or "$"))
    target_relation = matches[-1] if matches else None
    def _projection_context(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply only projection predicates/sorts that belong to this row universe.

        A projection rooted at ``episodes[*]`` may expose/filter a descendant
        ``crew[*].job`` path for runtime binding.  Applying that descendant filter
        directly to the outer episode observations erases every candidate.  By
        contrast, a root-level ``name`` filter on search rows is part of the same
        candidate universe and must be preserved for first/endpoint-rank replay.
        """
        out = list(rows)
        applicable_filters = []
        for item in obs_spec.get("filters") or []:
            path = item.get("path")
            if any(_field(row, path) is not None for row in out):
                applicable_filters.append(item)
        if applicable_filters:
            out = _filter_projection(out, applicable_filters)
        applicable_sort = []
        for item in obs_spec.get("sort") or []:
            path = item.get("path")
            if any(_field(row, path) is not None for row in out):
                applicable_sort.append(item)
        if applicable_sort:
            out = _sort_projection(out, applicable_sort)
        return out

    if target_relation and target_relation != root_relation:
        pattern = _record_path_pointer_regex(str(obs_spec.get("record_path") or "$"))
        descendant = None
        if pattern is not None:
            text = pattern.pattern
            if text.endswith("$"):
                text = text[:-1] + r"(?:/.*)?$"
            try:
                descendant = re.compile(text)
            except re.error:
                descendant = None
        nested = [o for o in records
                  if str(o.get("relation") or "") == target_relation
                  and (descendant is None or descendant.match(str(o.get("json_pointer") or "")))]
        if nested:
            return _projection_context(nested)
    # The Observation Planner controls what Phase A needs to see/bind; it must not
    # silently redefine deterministic answer semantics.  Planner derivations are
    # replayed from the lossless ledger using their own declared filter/sort logic,
    # so restrict only to the schema-declared record universe here.
    return _projection_context(_records_at_observation_root(records, obs_spec))


def _infer_identity_field(plan: dict[str, Any] | None, step_id: str,
                          records: list[dict[str, Any]]) -> str | None:
    spec = _observation_spec(plan, step_id)
    # Identity is an answer-value extraction, not a path-binding operation.  Do not
    # silently choose a binding identifier merely because it is available: that can
    # turn a requested label/title/value into an unrelated numeric ID.  Infer only
    # when the validated projection exposes exactly one viable scalar field.
    projected = [str(x) for x in spec.get("project_paths") or []]
    viable = [x for x in projected if any(_field(o, x) not in (None, "") for o in records)]
    if len(viable) == 1:
        return viable[0]
    return None


def _infer_membership_field(plan: dict[str, Any] | None, source_steps: list[str],
                            groups: list[list[dict[str, Any]]]) -> str | None:
    if len(source_steps) < 2 or len(groups) < 2:
        return None
    first_spec = _observation_spec(plan, source_steps[0])
    other_specs = [_observation_spec(plan, sid) for sid in source_steps[1:]]
    preferred = [str(x.get("path") or "") for x in first_spec.get("bindings") or [] if x.get("path")]
    common = set(str(x) for x in first_spec.get("project_paths") or [])
    for spec in other_specs:
        common &= set(str(x) for x in spec.get("project_paths") or [])
    def viable(paths):
        return [path for path in dict.fromkeys(paths) if path and
                all(any(_field(o, path) not in (None, "") for o in group) for group in groups)]

    # A validated binding path is the strongest schema-grounded comparison key.
    # Use it only when unique. Otherwise require exactly one common projected path.
    preferred_viable = viable(preferred)
    if len(preferred_viable) == 1:
        return preferred_viable[0]
    common_viable = viable(sorted(x for x in common if x))
    if len(common_viable) == 1:
        return common_viable[0]
    return None


def _comparison_domain(value: Any):
    d = _date(value)
    if d is not None:
        return "date", d, d.toordinal()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number", value, float(value)
    if isinstance(value, bool):
        return "bool", value, bool(value)
    if value not in (None, ""):
        return "string", value, str(value)
    return None, value, None


def _date_year_difference(a: _dt.date, b: _dt.date) -> int:
    earlier, later = (a, b) if a <= b else (b, a)
    years = later.year - earlier.year
    if (later.month, later.day) < (earlier.month, earlier.day):
        years -= 1
    return max(0, years)


def _evaluate_comparison(entries: list[dict[str, Any]], mode: str, unit: str):
    """Replay a planner-declared comparison over typed scalar inputs."""
    if len(entries) < 2:
        return None
    parsed = []
    for entry in entries:
        domain, original, comparable = _comparison_domain(entry.get("value"))
        if domain is None:
            return None
        parsed.append((domain, original, comparable, entry))
    domains = {x[0] for x in parsed}
    if len(domains) != 1:
        return None
    domain = parsed[0][0]
    mode = str(mode or "").lower()
    if mode in {"max", "min"}:
        if domain == "bool":
            return None
        ordered = sorted(parsed, key=lambda x: x[2], reverse=(mode == "max"))
        winner = ordered[0]
        return {
            "comparison": mode, "result": winner[1],
            "winner_index": int(winner[3].get("entry_index", 0)),
            "winner_source_step": winner[3].get("source_step"),
            "winner_source_derivation": winner[3].get("source_derivation"),
            "values": [x[1] for x in parsed],
        }
    if mode in {"eq", "neq", "gt", "gte", "lt", "lte"}:
        left, right = parsed[0][2], parsed[1][2]
        if mode == "eq": return left == right
        if mode == "neq": return left != right
        if mode == "gt": return left > right
        if mode == "gte": return left >= right
        if mode == "lt": return left < right
        if mode == "lte": return left <= right
    if mode in {"difference", "abs_difference"}:
        if domain == "number":
            diff = float(parsed[0][2]) - float(parsed[1][2])
            return abs(diff) if mode == "abs_difference" else diff
        if domain == "date":
            a, b = parsed[0][1], parsed[1][1]
            if str(unit or "raw").lower() == "years":
                value = _date_year_difference(a, b)
                if mode == "difference" and a < b:
                    value = -value
                return value
            days = (a - b).days
            if str(unit or "raw").lower() in {"day", "days", "raw"}:
                return abs(days) if mode == "abs_difference" else days
        return None
    return None


def _canonical_step_field(plan: dict[str, Any] | None, step_id: str, field: str) -> str:
    """Resolve a planner label/value alias through that step's declared binding path.

    The planner may name a semantic alias (``director_birth_date``) while the
    response schema and observation record expose ``birthday``.  The mapping is
    not guessed: it comes only from the step's validated ``binding_paths``.
    """
    raw = str(field or "").strip()
    if not raw:
        return raw
    step = next((x for x in (plan or {}).get("steps") or []
                 if str(x.get("id") or "") == str(step_id)), None) or {}
    mapping = step.get("binding_paths") if isinstance(step.get("binding_paths"), dict) else {}
    target = mapping.get(raw)
    return str(target).strip() if isinstance(target, str) and target.strip() else raw


def _planner_derivation_spec(plan: dict[str, Any] | None, derivation_id: str) -> dict[str, Any]:
    return next((x for x in (plan or {}).get("derivations") or []
                 if str(x.get("id") or "") == str(derivation_id)), {}) or {}


def _human_label_field(record: dict[str, Any]) -> str | None:
    """Return a deterministic human-readable label field from one exact record."""
    for field in ("name", "title", "label"):
        if _field(record, field) not in (None, ""):
            return field
    return None


def _records_at_observation_root(records: list[dict[str, Any]], spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Restrict normalized observations to the schema-declared record universe.

    Planner derivations must never count/rank the response envelope together with
    its collection items. Filters/sorts/selects are intentionally *not* applied
    here; those are replayed by the declared derivation itself.
    """
    if not spec:
        return list(records)
    record_path = str(spec.get("record_path") or "$" )
    return [o for o in records if _record_matches_projection_root(o, record_path)]


def _planner_derivations(ledger, plan, result):
    by_step = _step_groups(ledger)
    # Record-valued derivations form a DAG.  Keep the exact record set produced
    # by each plan derivation so downstream source_derivations consume their
    # declared parent, not whichever filter/selection happened to run most recently
    # for the same API step.  ``last_*`` remains only as a compatibility fallback
    # for older plans that omit source_derivations.
    record_sets: dict[str, list[dict[str, Any]]] = {}
    last_filtered: dict[str, list[dict[str, Any]]] = {}
    last_selected: dict[str, list[dict[str, Any]]] = {}
    obs_by_id = {str(o.get("obs_id") or ""): o for rows in by_step.values() for o in rows
                 if o.get("obs_id")}

    def _descends_from(row: dict[str, Any], ancestor_ids: set[str]) -> bool:
        parent = str(row.get("parent_obs_id") or "")
        seen: set[str] = set()
        while parent and parent not in seen:
            if parent in ancestor_ids:
                return True
            seen.add(parent)
            parent_row = obs_by_id.get(parent) or {}
            parent = str(parent_row.get("parent_obs_id") or "")
        return False

    derivation_specs = {str(d.get("id") or ""): d for d in ((plan or {}).get("derivations") or [])}

    for spec in (plan or {}).get("derivations") or []:
        source_steps = [str(x) for x in spec.get("source_steps") or []]
        records = []
        for sid in source_steps:
            step_records = list(by_step.get(sid, []))
            obs_spec = _observation_spec(plan, sid)
            if obs_spec:
                # Runtime bindings are authorized from the projection root, but a
                # derivation may address a nested projected child relation. Replay
                # against that exact descendant relation when the schema path names
                # it; otherwise use the ordinary projection-root universe.
                step_records = _records_for_derivation(step_records, obs_spec, spec)
            else:
                step_records = _records_at_observation_root(step_records, obs_spec)
            records.extend(step_records)
        candidates = [o for o in records if o.get("kind") in {"list_item", "nested_item", "record", "action_result"}]
        candidates = candidates or records
        # Prefer the deepest normalized relation named by the derivation field
        # or filter paths. This lets full schema paths such as
        # episodes.crew.name replay over relation=crew child observations without
        # any endpoint/domain knowledge.
        relation_names = {str(o.get("relation") or "") for o in candidates if o.get("relation")}
        path_parts: list[str] = []
        derivation_paths = [str(spec.get("field") or "")] + [
            str(x) for x in (spec.get("filter") or {}).keys()]
        for raw_path in derivation_paths:
            parts = [x for x in raw_path.replace("[*]", "").split(".") if x]
            path_parts.extend(parts[:-1] if len(parts) > 1 else parts)
        matching_relations = [x for x in path_parts if x in relation_names]
        if matching_relations:
            relation = matching_relations[-1]
            related = [o for o in candidates if str(o.get("relation") or "") == relation]
            if related:
                candidates = related
        op = str(spec.get("operator") or "").lower()
        source_derivations = [str(x) for x in spec.get("source_derivations") or []]

        # Exact derivation lineage wins over procedural step-local state.  Record-
        # valued operators (filter/selection) publish their selected universe below;
        # a downstream identity/count/selection therefore consumes precisely the
        # declared record-set parent even when another branch over the same API step
        # appears in between.
        lineage_sets = [record_sets[x] for x in source_derivations if x in record_sets]
        scoped_by_parent = False
        if len(source_derivations) == 1 and len(lineage_sets) == 1:
            parent_rows = list(lineage_sets[0])
            # A source_derivation from another API step is dependency lineage, not
            # a replacement record universe. For a same-step nested relation, however,
            # the selected parent record scopes its normalized child observations.
            # Preserve those descendants instead of either flattening every parent's
            # children or replacing the child universe with the parent record itself.
            current_steps = set(source_steps)
            parent_steps = {str(row.get("plan_step_id") or "") for row in parent_rows}
            parent_steps.discard("")
            if parent_rows and parent_steps and parent_steps.issubset(current_steps):
                parent_ids = {str(row.get("obs_id") or "") for row in parent_rows if row.get("obs_id")}
                descendants = [row for row in candidates if _descends_from(row, parent_ids)]
                if descendants:
                    candidates = descendants
                    scoped_by_parent = True
                else:
                    candidates = parent_rows

        # Sequential deterministic transformations over the same plan source compose
        # only as a backward-compatible fallback when no explicit record-set parent
        # was declared.
        # This is important for generic filter -> first/argmax/identity chains.
        if (not source_derivations and len(source_steps) == 1 and not spec.get("filter") and op in {
                "endpoint_rank", "first", "nth", "argmax", "argmin", "identity", "count"}):
            prior = last_filtered.get(source_steps[0])
            if prior is not None:
                candidates = list(prior)
        resolved_filter = _resolved_filter_for_replay(spec, plan, by_step)
        # A common normalized chain is ``filter -> first -> identity`` on the
        # same API collection.  The downstream selector may intentionally carry
        # the same predicate as its filter parent for schema/validation clarity.
        # Once exact record-set lineage has already supplied the parent's filtered
        # rows, re-applying that predicate can be destructive: normalized ledger
        # rows selected from the parent relation may no longer expose every field
        # used by the original predicate.  Treat an identical same-step parent
        # filter as already satisfied.  Different/additional predicates are still
        # replayed normally.
        if len(source_derivations) == 1 and lineage_sets:
            parent_spec = derivation_specs.get(source_derivations[0]) or {}
            parent_filter = _resolved_filter_for_replay(parent_spec, plan, by_step)
            parent_steps = {str(x) for x in (parent_spec.get("source_steps") or [])}
            if (str(parent_spec.get("operator") or "").lower() == "filter"
                    and set(source_steps) == parent_steps
                    and resolved_filter == parent_filter):
                resolved_filter = {}
        candidates = _filter_records(candidates, resolved_filter)

        # Some deterministic operations have meaningful empty/fallback semantics.
        # - filter over zero matches is a valid empty filtered set;
        # - count over an empty set is exactly 0;
        # - identity may still be recoverable from the already validated
        #   observation projection when the derivation-path spelling and the
        #   normalized ledger root differ.  The identity branch below preserves
        #   an explicitly empty prior filter, so this fallback cannot bypass a
        #   semantic predicate that matched nothing.
        if not candidates and op not in {
                "membership", "compare", "logical_and", "logical_or",
                "filter", "count", "identity"}:
            result["warnings"].append(f"planner derivation {spec.get('id')} had no candidates")
            continue

        chosen = None
        comparison_fields: list[str] = []
        plan_did = str(spec.get("id") or "") or None

        if op in {"endpoint_rank", "first", "nth"}:
            rank = int(spec.get("rank", 0) or 0)
            ordered = sorted(candidates, key=lambda o: (o.get("position") is None,
                                                        o.get("position", 10**9)))
            if rank < len(ordered):
                chosen, candidates, comparison_fields = ordered[rank], ordered, ["position"]
        elif op in {"argmax", "argmin"} and spec.get("field"):
            chosen, candidates = _select_by_field(candidates, str(spec["field"]), op == "argmax")
            comparison_fields = [str(spec["field"])]
        elif op == "filter":
            did = ledger.add_derived_record(
                name=spec.get("purpose") or spec.get("id"),
                value=[_candidate_payload(o) for o in candidates],
                operation="filter", input_obs_ids=[o["obs_id"] for o in candidates],
                candidate_obs_ids=[o["obs_id"] for o in candidates],
                policy="planner_declared", confidence="high",
                plan_step_id=source_steps[0] if len(source_steps) == 1 else None,
                plan_derivation_id=plan_did)
            result["derivation_ids"].append(did)
            result["focus_obs_ids"].update(o["obs_id"] for o in candidates[:50])
            if plan_did:
                record_sets[plan_did] = list(candidates)
            if len(source_steps) == 1:
                last_filtered[source_steps[0]] = list(candidates)
                if len(candidates) == 1:
                    last_selected[source_steps[0]] = [candidates[0]]
            continue
        elif op == "count":
            distinct_field = str(spec.get("distinct_field") or "").strip()
            if distinct_field:
                distinct_keys = {_stable_value_key(_field(o, distinct_field)) for o in candidates
                                 if _field(o, distinct_field) not in (None, "")}
                count_value = len(distinct_keys)
            else:
                count_value = len(candidates)
            did = ledger.add_derived_record(
                name=spec.get("purpose") or spec.get("id"), value=count_value,
                operation="count", input_obs_ids=[o["obs_id"] for o in candidates],
                candidate_obs_ids=[o["obs_id"] for o in candidates],
                policy="planner_declared", confidence="high",
                plan_step_id=source_steps[0] if len(source_steps) == 1 else None,
                plan_derivation_id=plan_did)
            result["derivation_ids"].append(did)
            result["focus_obs_ids"].update(o["obs_id"] for o in candidates[:50])
            continue
        elif op == "identity":
            if len(source_steps) != 1:
                result["warnings"].append(f"planner derivation {spec.get('id')} identity requires one source step")
                continue
            sid = source_steps[0]
            exact_parent = []
            same_step_record_parent = False
            if len(source_derivations) == 1 and source_derivations[0] in record_sets:
                possible_parent = list(record_sets.get(source_derivations[0]) or [])
                parent_steps = {str(row.get("plan_step_id") or "") for row in possible_parent}
                parent_steps.discard("")
                if possible_parent and parent_steps and parent_steps.issubset({sid}) and not scoped_by_parent:
                    exact_parent = possible_parent
                    same_step_record_parent = True
            # source_derivations can also be dependency/value lineage from another
            # API step (for example, the search selector that authorized this
            # child request).  Only an exact same-step record-set parent changes
            # the identity record universe.  Foreign/scalar lineage must not block
            # the current step's validated projection from supplying records.
            use_current_universe = not same_step_record_parent
            had_prior_filter = sid in last_filtered
            selected = list(exact_parent)
            if not selected and scoped_by_parent:
                selected = list(candidates)
            if not selected and use_current_universe:
                selected = list(last_selected.get(sid) or [])
            if not selected and had_prior_filter and use_current_universe:
                # Identity after a filter extracts from the filtered universe. A
                # multi-match filter is not an error: list/Boolean comparison tasks
                # may intentionally need every matching scalar (e.g. all directors).
                # The old code remembered that a filter existed but then selected
                # zero rows unless exactly one record had matched.
                selected = list(last_filtered.get(sid) or [])
            if not selected and spec.get("filter") and use_current_universe:
                # Identity + filter means extract the requested field from the
                # filtered relation, including nested normalized child records.
                # Preserve all matches; duplicate scalar values collapse below.
                selected = list(candidates)
            if (not selected and not had_prior_filter and use_current_universe
                    and (_observation_spec(plan, sid) or {}).get("schema_deferred")
                    and candidates):
                # For an OAS-undocumented response, _records_for_derivation() has
                # already resolved one unique runtime record universe. Re-running
                # the unresolved projection spec rooted at "$" would jump back to
                # the response envelope and lose those rows. Apply only the
                # validated bounded selection to the resolved candidates.
                selected = _select_projection(
                    list(candidates), (_observation_spec(plan, sid).get("select") or {}))
            if not selected and not had_prior_filter and use_current_universe:
                # The validated observation projection already encodes the
                # deterministic record universe + bounded selection.  Honor the
                # entire selected set here rather than silently collapsing
                # all_matches to its first row.  This is important for list/asset
                # answers and also provides a safe host fallback when equivalent
                # path spellings caused the derivation-specific record lookup to
                # miss rows that the projection itself selected.
                _, projected = _records_from_projection_spec(
                    list(by_step.get(sid, [])), _observation_spec(plan, sid))
                selected = list(projected)
            if not selected and not had_prior_filter and use_current_universe:
                selected = list(candidates[:1])
            field = str(spec.get("field") or "").strip() or _infer_identity_field(plan, sid, selected)
            if not field:
                result["warnings"].append(f"planner derivation {spec.get('id')} identity field unresolved")
                continue
            vals = [_field(o, field) for o in selected]
            vals = [v for v in vals if v not in (None, "")]
            vals = _unique_values(vals)

            # Exact record-set lineage is authoritative even if ledger
            # normalization flattened away a leaf used only by the downstream
            # identity.  The parent filter/selector derivation stores a compact
            # payload for those exact records; extract the declared field from
            # that payload as a same-lineage fallback rather than declaring a
            # certificate gap.
            if not vals and len(source_derivations) == 1:
                parent_record = next((d for d in reversed(ledger.derived)
                                      if str(d.get("plan_derivation_id") or "") == source_derivations[0]), None)
                parent_value = (parent_record or {}).get("value")
                rows = parent_value if isinstance(parent_value, list) else [parent_value]
                normalized_path = _normalized_observation_path(field)
                parts = [x for x in normalized_path.split(".") if x]
                def _payload_leaf(row):
                    if not isinstance(row, dict):
                        return None
                    value = row
                    for part in parts:
                        if isinstance(value, dict) and part in value:
                            value = value.get(part)
                        else:
                            leaf = parts[-1] if parts else ""
                            return row.get(leaf) if leaf else None
                    return value
                vals = _unique_values([v for v in (_payload_leaf(row) for row in rows)
                                       if v not in (None, "")])
            if not vals:
                # An explicitly empty terminal collection is evidence, not a
                # replay failure.  This is intentionally narrow: only a simple
                # top-level array record_path whose raw response contains that
                # exact empty array qualifies.  Missing fields, malformed
                # projections, and non-empty collections still produce the
                # ordinary unavailable warning instead of being disguised as
                # "no results".
                obs_spec = _observation_spec(plan, sid)
                record_path = str(obs_spec.get("record_path") or "").strip().replace("$.", "")
                m_empty = re.fullmatch(r"([A-Za-z0-9_\-]+)\[\*\]", record_path)
                explicit_empty = False
                if m_empty:
                    collection_key = m_empty.group(1)
                    for row in by_step.get(sid, []):
                        fields = row.get("fields") or {}
                        if (row.get("json_pointer") in {None, "", "/"}
                                and isinstance(fields, dict)
                                and fields.get(collection_key) == []):
                            explicit_empty = True
                            break
                if explicit_empty and str((plan or {}).get("answer_mode") or "").lower() in {"list"}:
                    did = ledger.add_derived_record(
                        name=spec.get("purpose") or spec.get("id"), value=[],
                        operation="identity_empty_collection",
                        input_obs_ids=[o["obs_id"] for o in by_step.get(sid, []) if o.get("obs_id")],
                        candidate_obs_ids=[], policy="planner_declared", confidence="high",
                        plan_step_id=sid, plan_derivation_id=plan_did)
                    result["derivation_ids"].append(did)
                    continue
                result["warnings"].append(f"planner derivation {spec.get('id')} identity value unavailable")
                continue
            value = vals[0] if len(vals) == 1 else vals
            did = ledger.add_derived_record(
                name=spec.get("purpose") or spec.get("id"), value=value,
                operation="identity", input_obs_ids=[o["obs_id"] for o in selected],
                selected_obs_id=selected[0]["obs_id"] if len(selected) == 1 else None,
                selected_obs_ids=[o["obs_id"] for o in selected],
                candidate_obs_ids=[o["obs_id"] for o in selected],
                comparison_fields=[field], policy="planner_declared", confidence="high",
                plan_step_id=sid, plan_derivation_id=plan_did)
            result["derivation_ids"].append(did)
            result["focus_obs_ids"].update(o["obs_id"] for o in selected)
            continue
        elif op == "membership":
            # Literal membership form: one collection source plus an explicit
            # comparison literal.  This directly represents questions such as
            # "does this cast contain Alice?" without fabricating a second API
            # source.  The literal is compared only against the declared scalar
            # field of the validated source collection.
            if spec.get("comparison_literal") is not None and len(source_steps) == 1:
                sid = source_steps[0]
                obs_spec = _observation_spec(plan, sid)
                if obs_spec:
                    ordered, selected_rows = _records_from_projection_spec(
                        list(by_step.get(sid, [])), obs_spec)
                    collection_rows = ordered or selected_rows
                else:
                    collection_rows = [o for o in by_step.get(sid, [])
                                       if o.get("kind") in {"list_item", "nested_item", "record"}]
                field = str(spec.get("field") or "").strip() or _infer_membership_field(
                    plan, source_steps, [collection_rows])
                if field:
                    # For literal membership, rank>0 is an explicit returned-order
                    # prefix size (Top-N). Zero keeps the complete observed
                    # collection, which is required for ordinary cast/co-star
                    # membership questions.
                    membership_rank = max(0, int(spec.get("rank", 0) or 0))
                    if membership_rank:
                        collection_rows = collection_rows[:membership_rank]
                    # Schema validation may keep a relation qualifier such as
                    # ``cast.name`` to distinguish sibling collections. Once the
                    # observation spec has already scoped rows to ``cast[*]``, the
                    # normalized ledger record exposes the relative leaf ``name``.
                    # Fall back to that leaf only when the qualified lookup yields
                    # no values in the already-selected collection.
                    probe_field = field
                    probe_values = [_field(o, probe_field) for o in collection_rows]
                    if not any(v not in (None, "") for v in probe_values) and "." in field:
                        probe_field = field.split(".")[-1]
                    collection_values = {_stable_value_key(_field(o, probe_field)) for o in collection_rows
                                         if _field(o, probe_field) not in (None, "")}
                    field = probe_field
                    literal = spec.get("comparison_literal")
                    literal_values = (list(literal) if isinstance(literal, (list, tuple, set))
                                      else [literal])
                    targets = {_stable_value_key(v) for v in literal_values if v not in (None, "")}
                    value = bool(targets & collection_values)
                    inputs = [o["obs_id"] for o in collection_rows]
                    did = ledger.add_derived_record(
                        name=spec.get("purpose") or spec.get("id"), value=value,
                        operation="membership", input_obs_ids=inputs, candidate_obs_ids=inputs,
                        comparison_fields=[field], policy="planner_declared", confidence="high",
                        plan_step_id=sid, plan_derivation_id=plan_did,
                        membership_field=field, membership_literal=literal,
                        collection_obs_ids=inputs)
                    result["derivation_ids"].append(did)
                    result["boolean_derivation_ids"].append(did)
                    result["focus_obs_ids"].update(inputs[:50])
                    continue
            # One filtered record-set plus its raw collection is a replayable
            # membership/existence test.  Infer the scalar only from the exact
            # equality-filter field that produced the target set when the planner
            # named the container rather than the leaf.
            source_derivation_ids = [str(x) for x in spec.get("source_derivations") or []]
            if len(source_derivation_ids) == 1 and len(source_steps) == 1:
                parent_id = source_derivation_ids[0]
                target_rows = list(record_sets.get(parent_id) or [])
                parent_spec = _planner_derivation_spec(plan, parent_id)
                if str(parent_spec.get("operator") or "").lower() == "filter":
                    obs_spec = _observation_spec(plan, source_steps[0])
                    if obs_spec:
                        collection_rows, _ = _records_from_projection_spec(
                            list(by_step.get(source_steps[0], [])), obs_spec)
                    else:
                        collection_rows = list(by_step.get(source_steps[0], []))
                    field = str(spec.get("field") or "").strip()
                    # If the declared membership field is a container/non-scalar,
                    # use one unambiguous predicate leaf from the source filter.
                    probe_values = [_field(o, field) for o in collection_rows] if field else []
                    if not field or not any(v not in (None, "") for v in probe_values):
                        pf = _resolved_filter_for_replay(parent_spec, plan, by_step)
                        if len(pf) == 1:
                            field = str(next(iter(pf)))
                    if field:
                        targets = {_stable_value_key(_field(o, field)) for o in target_rows
                                   if _field(o, field) not in (None, "")}
                        collection = {_stable_value_key(_field(o, field)) for o in collection_rows
                                      if _field(o, field) not in (None, "")}
                        value = bool(targets & collection)
                        inputs = [o["obs_id"] for o in target_rows + collection_rows]
                        did = ledger.add_derived_record(
                            name=spec.get("purpose") or spec.get("id"), value=value,
                            operation="membership", input_obs_ids=inputs, candidate_obs_ids=inputs,
                            comparison_fields=[field], policy="planner_declared", confidence="high",
                            plan_step_id=source_steps[0], plan_derivation_id=plan_did,
                            membership_field=field,
                            target_obs_ids=[o["obs_id"] for o in target_rows],
                            collection_obs_ids=[o["obs_id"] for o in collection_rows])
                        result["derivation_ids"].append(did)
                        result["boolean_derivation_ids"].append(did)
                        result["focus_obs_ids"].update(inputs[:50])
                        continue

            # Hybrid modern form: one already-replayed scalar/set target plus
            # one raw collection source.  This is the natural representation for
            # "is person_id in cast.id?" and avoids requiring a fake second API
            # source. The target value comes from the exact declared derivation;
            # the collection comes from the validated projection for this step.
            source_derivation_ids = [str(x) for x in spec.get("source_derivations") or []]
            if len(source_derivation_ids) == 1 and len(source_steps) == 1:
                source_record = next((d for d in reversed(ledger.derived)
                                      if str(d.get("plan_derivation_id") or "") == source_derivation_ids[0]), None)
                parent_spec = _planner_derivation_spec(plan, source_derivation_ids[0])
                if source_record is not None and str(parent_spec.get("operator") or "").lower() != "filter":
                    sid = source_steps[0]
                    obs_spec = _observation_spec(plan, sid)
                    if obs_spec:
                        collection_rows, selected_rows = _records_from_projection_spec(
                            list(by_step.get(sid, [])), obs_spec)
                        collection_rows = collection_rows or selected_rows
                    else:
                        collection_rows = [o for o in by_step.get(sid, [])
                                           if o.get("kind") in {"list_item", "nested_item", "record"}]
                    field = str(spec.get("field") or "").strip() or _infer_membership_field(
                        plan, source_steps, [collection_rows])
                    if field:
                        def _as_set(value):
                            if isinstance(value, (list, tuple, set)):
                                return {_stable_value_key(v) for v in value if v not in (None, "")}
                            if value in (None, ""):
                                return set()
                            return {_stable_value_key(value)}
                        targets = _as_set(source_record.get("value"))
                        collection = {_stable_value_key(_field(o, field)) for o in collection_rows
                                      if _field(o, field) not in (None, "")}
                        value = bool(targets & collection)
                        inputs = [str(source_record.get("obs_id"))] + [o["obs_id"] for o in collection_rows]
                        did = ledger.add_derived_record(
                            name=spec.get("purpose") or spec.get("id"), value=value,
                            operation="membership", input_obs_ids=inputs, candidate_obs_ids=inputs,
                            comparison_fields=[field], policy="planner_declared", confidence="high",
                            plan_step_id=sid, plan_derivation_id=plan_did,
                            membership_field=field,
                            target_obs_ids=[str(source_record.get("obs_id"))],
                            collection_obs_ids=[o["obs_id"] for o in collection_rows])
                        result["derivation_ids"].append(did)
                        result["boolean_derivation_ids"].append(did)
                        result["focus_obs_ids"].update(inputs[:50])
                        continue

            # Preferred modern form: membership/equality over already-replayed
            # scalar derivations.  The planner may explicitly declare
            # source_derivations for same-entity checks; ignoring them and trying
            # to infer a common raw response field can make a fully grounded
            # Boolean impossible to replay.
            source_derivation_ids = [str(x) for x in spec.get("source_derivations") or []]
            if source_derivation_ids:
                source_records = []
                for source_did in source_derivation_ids:
                    record = next((d for d in reversed(ledger.derived)
                                   if str(d.get("plan_derivation_id") or "") == source_did), None)
                    if record is None:
                        source_records = []
                        break
                    source_records.append(record)
                if len(source_records) == len(source_derivation_ids) and len(source_records) >= 2:
                    def _value_set(value):
                        if isinstance(value, (list, tuple, set)):
                            return {_stable_value_key(v) for v in value if v not in (None, "")}
                        if value in (None, ""):
                            return set()
                        return {_stable_value_key(value)}
                    value_sets = [_value_set(record.get("value")) for record in source_records]
                    value = bool(value_sets) and all(value_sets) and bool(set.intersection(*value_sets))
                    inputs = [str(record.get("obs_id")) for record in source_records]
                    did = ledger.add_derived_record(
                        name=spec.get("purpose") or spec.get("id"), value=value,
                        operation="membership", input_obs_ids=inputs,
                        candidate_obs_ids=inputs, comparison_fields=["source_derivations"],
                        policy="planner_declared", confidence="high",
                        plan_step_id=source_steps[-1] if source_steps else None,
                        plan_derivation_id=plan_did,
                        membership_field="source_derivations",
                        target_obs_ids=inputs[:1], collection_obs_ids=inputs[1:])
                    result["derivation_ids"].append(did)
                    result["boolean_derivation_ids"].append(did)
                    result["focus_obs_ids"].update(inputs[:50])
                    continue
            groups_ordered: list[list[dict[str, Any]]] = []
            groups_selected: list[list[dict[str, Any]]] = []
            for sid in source_steps:
                obs_spec = _observation_spec(plan, sid)
                if obs_spec:
                    ordered, selected = _records_from_projection_spec(
                        list(by_step.get(sid, [])), obs_spec)
                else:
                    ordered = [o for o in by_step.get(sid, [])
                               if o.get("kind") in {"list_item", "nested_item", "record"}]
                    selected = list(ordered)
                groups_ordered.append(ordered)
                groups_selected.append(selected)
            field = str(spec.get("field") or "").strip() or _infer_membership_field(
                plan, source_steps, [g if g else o for g, o in zip(groups_selected, groups_ordered)])
            if not field or len(source_steps) < 2:
                result["warnings"].append(f"planner derivation {spec.get('id')} membership field unresolved")
                continue
            top_k = max(1, int(spec.get("top_k", 10) or 10))
            if (plan or {}).get("observation_specs"):
                # With schema-grounded projections, source 0 is the selected target
                # and source 1 is the ranked/filtered collection to test.
                target_records = groups_selected[0] or groups_ordered[0][:1]
                collection_records = (groups_ordered[1] or groups_selected[1])[:top_k]
                target_values = {_stable_value_key(_field(o, field)) for o in target_records
                                 if _field(o, field) not in (None, "")}
                collection_values = {_stable_value_key(_field(o, field)) for o in collection_records
                                     if _field(o, field) not in (None, "")}
                value = bool(target_values & collection_values)
                inputs = [o["obs_id"] for o in target_records + collection_records]
            else:
                # Legacy/no-projection generic semantics: membership means overlap
                # across the complete source sets.
                sets = []
                inputs = []
                for ordered in groups_ordered:
                    vals = {_stable_value_key(_field(o, field)) for o in ordered
                            if _field(o, field) not in (None, "")}
                    sets.append(vals); inputs.extend(o["obs_id"] for o in ordered)
                value = bool(sets) and (bool(set.intersection(*sets)) if len(sets) > 1 else bool(sets[0]))
                target_records = groups_ordered[0]
                collection_records = groups_ordered[1] if len(groups_ordered) > 1 else []
            did = ledger.add_derived_record(
                name=spec.get("purpose") or spec.get("id"), value=value,
                operation="membership", input_obs_ids=inputs, candidate_obs_ids=inputs,
                comparison_fields=[field], policy="planner_declared", confidence="high",
                plan_step_id=source_steps[-1], plan_derivation_id=plan_did,
                membership_field=field,
                target_obs_ids=[o["obs_id"] for o in target_records],
                collection_obs_ids=[o["obs_id"] for o in collection_records])
            result["derivation_ids"].append(did)
            result["boolean_derivation_ids"].append(did)
            result["focus_obs_ids"].update(inputs[:50])
            continue
        elif op in {"logical_and", "logical_or"}:
            inputs = []
            values = []
            for source_did in spec.get("source_derivations") or []:
                record = next((d for d in reversed(ledger.derived)
                               if str(d.get("plan_derivation_id") or "") == str(source_did)), None)
                if record is None or not isinstance(record.get("value"), bool):
                    continue
                values.append(bool(record.get("value")))
                inputs.append(str(record.get("obs_id")))
            if len(values) != len(spec.get("source_derivations") or []) or len(values) < 2:
                result["warnings"].append(
                    f"planner derivation {spec.get('id')} {op} could not replay every Boolean input")
                continue
            value = all(values) if op == "logical_and" else any(values)
            did = ledger.add_derived_record(
                name=spec.get("purpose") or spec.get("id"), value=value,
                operation=op, input_obs_ids=inputs, candidate_obs_ids=inputs,
                policy="planner_declared", confidence="high",
                plan_step_id=None, plan_derivation_id=plan_did,
                logical_inputs=list(spec.get("source_derivations") or []))
            result["derivation_ids"].append(did)
            result["boolean_derivation_ids"].append(did)
            result["focus_obs_ids"].update(inputs)
            continue
        elif op == "compare":
            field = str(spec.get("field") or "").strip()
            mode = str(spec.get("comparison") or "").strip().lower()
            unit = str(spec.get("unit") or "raw").strip().lower()
            entries: list[dict[str, Any]] = []
            inputs: list[str] = []
            # Raw API-step scalar inputs. The planner must name a field so the host
            # can replay the comparison without semantic guessing.
            if source_steps and field:
                for sid in source_steps:
                    _, selected = _records_from_projection_spec(
                        list(by_step.get(sid, [])), _observation_spec(plan, sid))
                    recs = selected or list(by_step.get(sid, []))[:1]
                    rec = next((o for o in recs if _field(o, field) not in (None, "")), None)
                    if rec is not None:
                        entries.append({"source_step": sid, "value": _field(rec, field)})
                        inputs.append(rec["obs_id"])
            # Scalar outputs of earlier deterministic derivations (e.g. compare two
            # counts). Only exact planner-derivation ids are accepted.
            for source_did in spec.get("source_derivations") or []:
                record = next((d for d in reversed(ledger.derived)
                               if str(d.get("plan_derivation_id") or "") == str(source_did)), None)
                if not record:
                    continue
                value = record.get("value")
                if isinstance(value, (str, int, float, bool, list, tuple, set)) and value not in (None, "", [], (), set()):
                    entries.append({"source_derivation": str(source_did),
                                    "source_step": str(record.get("plan_step_id") or "") or None,
                                    "value": value})
                    inputs.append(str(record.get("obs_id")))
            if spec.get("comparison_literal") is not None:
                entries.append({"source_literal": True, "value": spec.get("comparison_literal")})
            for entry_index, entry in enumerate(entries):
                entry["entry_index"] = entry_index
            same_entity_requirement = " ".join(str(x) for x in (plan or {}).get("answer_requirements") or []).casefold()
            identity_like_sources = True
            for source_did in spec.get("source_derivations") or []:
                src_spec = _planner_derivation_spec(plan, str(source_did))
                if str(src_spec.get("operator") or "").lower() != "identity":
                    identity_like_sources = False; break
                leaf = _normalized_observation_path(src_spec.get("field")).split(".")[-1]
                if leaf not in {"id", "name", "title", "label"}:
                    identity_like_sources = False; break
            if (mode in {"eq", "neq"} and str((plan or {}).get("answer_mode") or "").lower() == "boolean"
                    and any(isinstance(e.get("value"), (list, tuple, set)) for e in entries)
                    and (identity_like_sources or re.search(r"\bsame\b|\bequal|\bco[- ]?star|\bshared\b", same_entity_requirement))):
                def _as_value_set(v):
                    vals = list(v) if isinstance(v, (list, tuple, set)) else [v]
                    return {_stable_value_key(x) for x in vals if x not in (None, "")}
                sets = [_as_value_set(e.get("value")) for e in entries[:2]]
                overlap = bool(len(sets) >= 2 and sets[0] and sets[1] and (sets[0] & sets[1]))
                value = overlap if mode == "eq" else not overlap
            else:
                value = _evaluate_comparison(entries, mode, unit)
            label_steps = [str(x) for x in spec.get("label_steps") or []]
            raw_label_fields = [str(x) for x in spec.get("label_fields") or []]
            # Keep winner-label replay aligned with the same alias normalization
            # used for identity derivations.  A comparison must not fail merely
            # because the planner labels a value with its semantic binding alias
            # while the selected response record exposes the mapped schema field.
            label_fields = [
                _canonical_step_field(plan, sid, field)
                for sid, field in zip(label_steps, raw_label_fields)
            ] if len(label_steps) == len(raw_label_fields) else raw_label_fields
            # A planner may mistakenly label a count comparison with the support
            # field that was counted (job/id) rather than the compared entity's
            # human name. For ordering winner output, repair display lineage only
            # through declared step dependencies and already-selected observations.
            if mode in {"max", "min", "gt", "gte", "lt", "lte"} and len(entries) >= 2:
                dep_map = {str(st.get("id") or ""): [str(x) for x in st.get("depends_on") or []]
                           for st in (plan or {}).get("steps") or []}
                def _ancestor_human_label(sid: str):
                    queue = list(dep_map.get(sid, [])); seen=set()
                    while queue:
                        cur = queue.pop(0)
                        if cur in seen: continue
                        seen.add(cur)
                        _, selected_rows = _records_from_projection_spec(
                            list(by_step.get(cur, [])), _observation_spec(plan, cur))
                        rows = selected_rows or list(by_step.get(cur, []))
                        for row in rows:
                            human = _human_label_field(row)
                            if human and _field(row, human) not in (None, ""):
                                return cur, human
                        queue.extend(dep_map.get(cur, []))
                    return None
                repaired_steps=[]; repaired_fields=[]
                for i, entry in enumerate(entries):
                    sid = label_steps[i] if i < len(label_steps) else str(entry.get("source_step") or "")
                    fld = label_fields[i] if i < len(label_fields) else ""
                    humanish = _normalized_observation_path(fld).split(".")[-1] in {"name", "title", "label"}
                    if not humanish:
                        alt = _ancestor_human_label(str(entry.get("source_step") or sid))
                        if alt:
                            sid, fld = alt
                    repaired_steps.append(sid); repaired_fields.append(fld)
                if len(repaired_steps) == len(entries) and all(repaired_steps) and all(repaired_fields):
                    label_steps, label_fields = repaired_steps, repaired_fields
            # If an ordering comparison consumes scalar identity derivations and
            # the planner omitted display-label metadata, derive labels only from
            # the exact same selected source records. This is generic lineage
            # completion, not entity guessing: each scalar already carries its
            # source step, and the validated projection exposes name/title/label.
            if (not label_steps and not label_fields and len(entries) >= 2
                    and mode in {"max", "min", "gt", "gte", "lt", "lte"}):
                inferred_steps, inferred_fields = [], []
                for entry in entries:
                    sid = str(entry.get("source_step") or "")
                    if not sid:
                        inferred_steps, inferred_fields = [], []
                        break
                    _, selected_rows = _records_from_projection_spec(
                        list(by_step.get(sid, [])), _observation_spec(plan, sid))
                    rows = selected_rows or list(by_step.get(sid, []))
                    record = rows[0] if len(rows) == 1 else next(
                        (row for row in rows if _human_label_field(row)), None)
                    human = _human_label_field(record) if record is not None else None
                    if not human:
                        inferred_steps, inferred_fields = [], []
                        break
                    inferred_steps.append(sid); inferred_fields.append(human)
                if len(inferred_steps) == len(entries):
                    label_steps, label_fields = inferred_steps, inferred_fields
            # Ordering comparisons used as user-facing comparisons need a
            # replayable winner, not a bare True/False. When the planner supplied
            # aligned labels, turn gt/lt into the same structured winner form used
            # by max/min. Boolean tasks without labels remain plain booleans.
            if (isinstance(value, bool) and mode in {"gt", "gte", "lt", "lte"}
                    and str((plan or {}).get("answer_mode") or "").lower() != "boolean"
                    and len(entries) >= 2
                    and len(label_steps) == len(label_fields) == len(entries)):
                left = _comparison_domain(entries[0].get("value"))[2]
                right = _comparison_domain(entries[1].get("value"))[2]
                winner_index = None
                if left is not None and right is not None and left != right:
                    if mode in {"gt", "gte"}:
                        winner_index = 0 if left > right else 1
                    else:
                        winner_index = 0 if left < right else 1
                value = {
                    "comparison": mode,
                    "result": bool(value),
                    "winner_index": winner_index,
                    "values": [entry.get("value") for entry in entries],
                }
                # Equality is a legitimate deterministic outcome for an ordering
                # question. Older materialization treated ``winner_index=None`` as
                # missing evidence and fell back to an LLM/abstention. Preserve a
                # citable tie surface instead; no source entity is invented.
                if left is not None and right is not None and left == right:
                    value["tie"] = True
                    value["tie_label"] = "Tie"
            if isinstance(value, dict) and mode in {"max", "min", "gt", "gte", "lt", "lte"}:
                winner_raw = value.get("winner_index")
                winner_index = int(winner_raw) if isinstance(winner_raw, int) else -1
                if len(label_steps) == len(label_fields) == len(entries) and 0 <= winner_index < len(entries):
                    label_sid, label_field = label_steps[winner_index], label_fields[winner_index]
                    _, label_selected = _records_from_projection_spec(
                        list(by_step.get(label_sid, [])), _observation_spec(plan, label_sid))
                    label_rows = label_selected or list(by_step.get(label_sid, []))
                    label_record = next((o for o in label_rows
                                         if _field(o, label_field) not in (None, "")), None)

                    # ``label_fields`` is display lineage, not the compared value.
                    # If the planner accidentally repeats the exact scalar field
                    # being compared (for example birthday) as the winner label,
                    # prefer a human-readable label from that *same selected
                    # record* when one is already exposed by the validated
                    # projection.  This cannot switch entities because the label
                    # and compared scalar come from the identical step/record.
                    entry = entries[winner_index]
                    source_did = str(entry.get("source_derivation") or "")
                    source_spec = _planner_derivation_spec(plan, source_did)
                    source_field = _canonical_step_field(
                        plan, label_sid, str(source_spec.get("field") or ""))
                    if (label_record is not None and source_field
                            and _normalized_observation_path(label_field)
                                == _normalized_observation_path(source_field)):
                        human = _human_label_field(label_record)
                        if human:
                            label_field = human
                    if label_record is None or _field(label_record, label_field) in (None, ""):
                        label_record = next((o for o in label_rows
                                             if _field(o, label_field) not in (None, "")), None)
                    if label_record is not None and _field(label_record, label_field) not in (None, ""):
                        value["winner_label"] = _field(label_record, label_field)
                        value["winner_label_obs_id"] = label_record.get("obs_id")
                        value["winner_label_step"] = label_sid
                        value["winner_label_field"] = label_field
                        inputs.append(str(label_record.get("obs_id")))
            if value is None:
                result["warnings"].append(f"planner derivation {spec.get('id')} comparison could not be replayed")
                continue
            did = ledger.add_derived_record(
                name=spec.get("purpose") or spec.get("id"), value=value,
                operation="compare", input_obs_ids=inputs, candidate_obs_ids=inputs,
                comparison_fields=[field] if field else [], policy="planner_declared", confidence="high",
                plan_step_id=source_steps[-1] if source_steps else None,
                plan_derivation_id=plan_did, comparison_mode=mode, comparison_unit=unit,
                comparison_entries=entries)
            result["derivation_ids"].append(did)
            result["comparison_derivation_ids"].append(did)
            if isinstance(value, bool): result["boolean_derivation_ids"].append(did)
            result["focus_obs_ids"].update(inputs)
            continue

        if chosen is not None:
            did = _add_selection(
                ledger, name=spec.get("purpose") or spec.get("id"), operation=op,
                candidates=candidates, chosen=chosen, comparison_fields=comparison_fields,
                policy="planner_declared", plan_derivation_id=plan_did)
            result["derivation_ids"].append(did)
            result["selection_derivation_ids"].append(did)
            result["focus_obs_ids"].add(chosen["obs_id"])
            if plan_did:
                record_sets[plan_did] = [chosen]
            if len(source_steps) == 1:
                last_selected[source_steps[0]] = [chosen]
                # A selection after a filter preserves the filtered candidate set for
                # subsequent identity operations but not for unrelated source steps.
                if source_steps[0] not in last_filtered:
                    last_filtered[source_steps[0]] = list(candidates)

def _projection_derivations(ledger, plan, result):
    by_step = _step_groups(ledger)
    for spec in (plan or {}).get("observation_specs") or []:
        sid = str(spec.get("step_id") or "")
        records = list(by_step.get(sid, []))
        record_path = str(spec.get("record_path") or "$")
        records = [o for o in records if _record_matches_projection_root(o, record_path)]
        records = _filter_projection(records, spec.get("filters") or [])
        ordered = _sort_projection(records, spec.get("sort") or [])
        select = spec.get("select") or {}
        mode = str(select.get("mode") or "head").lower()
        selected = _select_projection(ordered, select)
        if selected:
            if len(selected) == 1:
                op = "projection_sort_select" if spec.get("sort") else (
                    "projection_nth" if mode == "nth" else "projection_select")
                did = _add_selection(
                    ledger, name=spec.get("purpose") or f"projection:{sid}",
                    operation=op, candidates=ordered, chosen=selected[0],
                    comparison_fields=[str(x.get("path")) for x in spec.get("sort") or []],
                    policy="projection_declared", value=_projected_payload(selected[0], spec))
                result["derivation_ids"].append(did); result["selection_derivation_ids"].append(did)
            else:
                did = ledger.add_derived_record(
                    name=spec.get("purpose") or f"projection:{sid}",
                    value=[_projected_payload(x, spec) for x in selected], operation="projection_list",
                    input_obs_ids=[x["obs_id"] for x in ordered],
                    candidate_obs_ids=[x["obs_id"] for x in ordered],
                    selected_obs_ids=[x["obs_id"] for x in selected],
                    policy="projection_declared", confidence="high", plan_step_id=sid)
                result["derivation_ids"].append(did)
            result["focus_obs_ids"].update(x["obs_id"] for x in selected)
        for agg in spec.get("aggregates") or []:
            op = str(agg.get("op") or "").lower(); path = str(agg.get("path") or "$")
            values = ordered if path == "$" else [_field(o, path) for o in ordered]
            values = [x for x in values if x not in (None, "")]
            value = None
            if op == "count": value = len(values)
            elif op in {"sum", "mean", "min", "max"}:
                nums = [float(x) for x in values if isinstance(x, (int, float)) and not isinstance(x, bool)]
                if nums:
                    value = {"sum": sum(nums), "mean": sum(nums)/len(nums), "min": min(nums), "max": max(nums)}[op]
            if value is not None:
                did = ledger.add_derived_record(
                    name=str(agg.get("as") or op), value=value, operation=f"aggregate:{op}",
                    input_obs_ids=[o["obs_id"] for o in ordered],
                    candidate_obs_ids=[o["obs_id"] for o in ordered],
                    policy="projection_declared", confidence="high", plan_step_id=sid)
                result["derivation_ids"].append(did); result["focus_obs_ids"].update(o["obs_id"] for o in ordered[:50])


def _payload_path(payload: Any, path: str) -> Any:
    """Resolve a raw response path while preserving empty-list evidence.

    Observation specs use schema paths such as ``items[*].album``.  The old
    helper simply removed ``[*]`` and then attempted dict-only traversal, so an
    authoritative payload ``{"items": []}`` became ``None`` instead of the
    meaningful empty collection.  This generic walker follows dict/list paths
    and returns ``[]`` when a wildcard collection is proven empty.
    """
    text = str(path or "$" ).strip().replace("$.", "")
    if text in {"", "$"}:
        return payload
    parts = [x for x in text.split(".") if x]

    def walk(value: Any, index: int) -> Any:
        if index >= len(parts):
            return value
        token = parts[index]
        wildcard = token.endswith("[*]")
        key = token[:-3] if wildcard else token

        if isinstance(value, dict):
            if key not in value:
                return None
            child = value.get(key)
        elif isinstance(value, list):
            # A previous wildcard produced a record collection. Apply the next
            # path segment to every member and preserve a proven empty set.
            if not value:
                return []
            collected = []
            for item in value:
                got = walk(item, index)
                if got is None:
                    continue
                if isinstance(got, list):
                    collected.extend(got)
                else:
                    collected.append(got)
            return collected
        else:
            return None

        if wildcard:
            if not isinstance(child, list):
                return None
            if not child:
                return []
            collected = []
            for item in child:
                got = walk(item, index + 1)
                if got is None:
                    continue
                if isinstance(got, list):
                    collected.extend(got)
                else:
                    collected.append(got)
            return collected
        return walk(child, index + 1)

    return walk(payload, 0)


def _empty_answer_derivations(ledger, plan, result):
    # Absence can be a meaningful direct/list result, but it is not an asset.
    # Returning "No results are available" for an image/logo request hides a
    # failed entity/asset selection and prevents accuracy recovery from running.
    if str((plan or {}).get("answer_mode") or "direct").lower() == "asset":
        return
    answer_steps = {str(x) for x in (plan or {}).get("answer_steps") or []}
    specs = {str(s.get("step_id")): s for s in (plan or {}).get("observation_specs") or []}
    call_step = {str(c.get("call_id")): str(c.get("plan_step_id")) for c in ledger.api_calls if c.get("plan_step_id")}
    for raw in ledger.raw_responses:
        sid = str(raw.get("plan_step_id") or call_step.get(str(raw.get("call_id"))) or "")
        if sid not in answer_steps: continue
        spec = specs.get(sid) or {}; surface = _payload_path(raw.get("payload"), spec.get("record_path") or "$")
        if isinstance(surface, list) and not surface:
            did = ledger.add_derived_record(
                name=f"empty_answer_surface:{sid}", value="No results are available.", operation="empty_result",
                input_obs_ids=[], candidate_obs_ids=[], policy="fetched_empty_answer_surface", confidence="high",
                source_call_id=raw.get("call_id"), source_endpoint=raw.get("endpoint"), plan_step_id=sid)
            result["derivation_ids"].append(did); result["answer_derivation_ids"].append(did)


def compile_evidence(question: str, ledger, plan=None) -> dict[str, Any]:
    """Replay validated plan/projection decisions and build the Phase-B focus packet."""
    del question
    ledger.derived = [d for d in ledger.derived if d.get("capture_method") != "auto_compiler"]
    ledger._dcounter = max([int(str(d["obs_id"]).split("_")[-1]) for d in ledger.derived] or [0])
    result = {"derivation_ids": [], "answer_derivation_ids": [], "selection_derivation_ids": [],
              "comparison_derivation_ids": [], "boolean_derivation_ids": [],
              "focus_obs_ids": set(), "warnings": []}
    _planner_derivations(ledger, plan, result)
    _projection_derivations(ledger, plan, result)
    _empty_answer_derivations(ledger, plan, result)

    answer_steps = {str(x) for x in (plan or {}).get("answer_steps") or []}
    result["focus_obs_ids"].update(o["obs_id"] for o in ledger.observations
                                   if str(o.get("plan_step_id") or "") in answer_steps)
    derivations = {did: (ledger.get_derived(did) or {}) for did in result["derivation_ids"]}
    mode = str((plan or {}).get("answer_mode") or "direct").lower()
    def terminal(d): return not answer_steps or str(d.get("plan_step_id") or "") in answer_steps

    # New plans may explicitly declare the exact user-facing derivations.  This
    # is intentionally separate from answer_steps: a step can contain both a
    # request-routing id and a requested title/value.  Prefer the explicit
    # derivation contract when present, while retaining legacy mode-based
    # inference for old plans that omit it.
    explicit_plan_answer_ids = [str(x) for x in ((plan or {}).get("answer_derivations") or [])]
    if explicit_plan_answer_ids:
        existing_empty = [did for did in result["answer_derivation_ids"]
                          if str((derivations.get(did) or {}).get("operation") or "") == "empty_result"]
        explicit_compiled: list[str] = []
        for plan_did in explicit_plan_answer_ids:
            matches = [did for did, d in derivations.items()
                       if str(d.get("plan_derivation_id") or "") == plan_did]
            if matches:
                explicit_compiled.append(matches[-1])
            else:
                result["warnings"].append(
                    f"explicit answer derivation {plan_did} could not be replayed")
        result["answer_derivation_ids"] = existing_empty + explicit_compiled
    elif mode == "count":
        ids = [did for did,d in derivations.items() if str(d.get("operation") or "") in {"count","aggregate:count"}]
        if ids: result["answer_derivation_ids"].append(ids[-1])
    elif mode == "boolean":
        logical_ids = [did for did, d in derivations.items()
                       if str(d.get("operation") or "") in {"logical_and", "logical_or"}
                       and isinstance(d.get("value"), bool)]
        if logical_ids:
            result["answer_derivation_ids"].append(logical_ids[-1])
        else:
            ids = [did for did,d in derivations.items()
                   if str(d.get("operation") or "") in {"membership", "compare"}
                   and isinstance(d.get("value"), bool)]
            if len(ids) == 1:
                result["answer_derivation_ids"].append(ids[0])
            elif len(ids) > 1:
                result["warnings"].append(
                    "multiple Boolean derivations exist without a host-replayed logical combiner")
    elif mode == "comparison":
        ids = [did for did,d in derivations.items() if str(d.get("operation") or "") == "compare"]
        result["answer_derivation_ids"].extend(ids)
    else:
        # If the plan explicitly extracts terminal fields with identity derivations,
        # those scalar extractions are the authoritative answer values. This supports
        # arbitrary field names after a prior first/argmax/argmin selection.
        identity_ids = [did for did, d in derivations.items()
                        if terminal(d) and str(d.get("operation") or "") == "identity"]
        if identity_ids:
            result["answer_derivation_ids"].extend(identity_ids)
        else:
            # A declared ranking selection is more authoritative than a later
            # projection-list helper over the same answer step. Prefer the exact
            # planner argmax/argmin result so deterministic emission cannot fall
            # back to an arbitrary projected row.
            ranked = [did for did, d in derivations.items()
                      if terminal(d) and str(d.get("operation") or "") in {"argmax", "argmin"}
                      and d.get("plan_derivation_id")]
            if ranked:
                result["answer_derivation_ids"].append(ranked[-1])
            else:
                ids = [did for did,d in derivations.items() if terminal(d) and str(d.get("operation") or "") in {
                    "projection_select","projection_sort_select","projection_nth","projection_list",
                    "endpoint_rank","first","nth","argmax","argmin","empty_result"}]
                if ids: result["answer_derivation_ids"].append(ids[-1])
    result["answer_derivation_ids"] = list(dict.fromkeys(result["answer_derivation_ids"]))
    for did in result["derivation_ids"]:
        d = ledger.get_derived(did) or {}; result["focus_obs_ids"].update(d.get("input_obs_ids") or [])
        if d.get("selected_obs_id"): result["focus_obs_ids"].add(d["selected_obs_id"])
    result["focus_obs_ids"] = sorted(result["focus_obs_ids"])
    return result
