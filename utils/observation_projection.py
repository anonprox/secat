"""Planner-guided, schema-agnostic observation projection for OCA.

Unlike ``observation_receipt.py``, this module has no whitelist of domain fields
such as ``id``, ``title``, ``release_date`` or ``rating``.  The LLM planner chooses
concrete response paths from the selected endpoint schemas.  Python then applies
that plan mechanically to the complete response already captured in OCA's ledger.

The full response is never discarded; projection only controls what is returned
to Phase A's chat context.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from utils.predicate_semantics import compare_predicate

from utils.observation_ledger import redact_secrets

_ALLOWED_FILTER_OPS = {
    "eq", "eq_ci", "neq", "neq_ci", "contains", "contains_ci", "startswith_ci",
    "exists", "not_exists", "gt", "gte", "lt", "lte", "in",
}
_ALLOWED_SELECT_MODES = {"head", "top", "tail", "nth", "all_matches", "single"}
_ALLOWED_AGG_OPS = {"count", "sum", "mean", "min", "max"}
_SECRET_RE = re.compile(r"(?:token|secret|password|api[_-]?key|authorization)", re.I)


def normalize_path(path: Any) -> str:
    text = str(path or "").strip()
    if text in {"", ".", "$"}:
        return "$"
    if text.startswith("$."):
        text = text[2:]
    elif text.startswith("$"):
        text = text[1:].lstrip(".")
    text = text.replace("/", ".") if text.startswith("/") else text
    text = re.sub(r"\.\.+", ".", text).strip(".")
    return text or "$"


def _tokens(path: Any) -> list[str]:
    path = normalize_path(path)
    if path == "$":
        return []
    out: list[str] = []
    for part in path.split("."):
        if not part:
            continue
        while "[*]" in part:
            before, after = part.split("[*]", 1)
            if before:
                out.append(before)
            out.append("*")
            part = after
        if part:
            out.append(part)
    return out


def extract_values(obj: Any, path: Any) -> list[Any]:
    """Resolve a small JSONPath-like path with ``[*]`` wildcards."""
    values = [obj]
    for token in _tokens(path):
        next_values: list[Any] = []
        if token == "*":
            for value in values:
                if isinstance(value, list):
                    next_values.extend(value)
            values = next_values
            continue
        for value in values:
            if isinstance(value, dict) and token in value:
                next_values.append(value[token])
        values = next_values
        if not values:
            break
    return values


def _scalar_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


def runtime_structure(payload: Any, *, max_depth: int = 7,
                      max_paths: int = 220) -> dict[str, Any]:
    """Describe runtime shape using names/types/counts only, never scalar values.

    Traversal is breadth-first so one very large early branch cannot consume the
    whole structural budget and hide later sibling fields. Arrays of objects union
    keys across the complete returned array before deeper traversal.
    """
    from collections import deque

    entries: dict[str, dict[str, Any]] = {}
    truncated = False

    def merge(path: str, typ: str, *, total: int | None = None,
              present: int | None = None):
        nonlocal truncated
        path = path or "$"
        if len(entries) >= max_paths and path not in entries:
            truncated = True
            return False
        item = entries.setdefault(path, {"path": path, "types": set()})
        item["types"].add(typ)
        if total is not None:
            item["count"] = max(int(item.get("count", 0)), int(total))
        if present is not None:
            item["present"] = max(int(item.get("present", 0)), int(present))
        return True

    root = redact_secrets(payload)
    # value, path, depth, collection_total, collection_presence
    queue = deque([(root, "", 0, None, None)])
    while queue:
        value, prefix, depth, total_hint, present_hint = queue.popleft()
        if depth > max_depth:
            truncated = True
            continue
        typ = _scalar_type(value)
        total = len(value) if isinstance(value, list) else total_hint
        if not merge(prefix or "$", typ, total=total, present=present_hint):
            continue

        if isinstance(value, dict):
            # Enqueue rather than fully expanding here: this is what gives true
            # breadth-first fairness across large sibling branches.
            for key, child in value.items():
                p = f"{prefix}.{key}" if prefix else str(key)
                queue.append((child, p, depth + 1, None, None))

        elif isinstance(value, list) and value:
            item_path = f"{prefix}[*]" if prefix else "[*]"
            merge(item_path, "array_item", total=len(value))
            if all(isinstance(x, dict) for x in value):
                presence: dict[str, int] = {}
                samples: dict[str, dict[str, Any]] = {}
                for row in value:
                    for key, child in row.items():
                        presence[key] = presence.get(key, 0) + 1
                        by_type = samples.setdefault(key, {})
                        typ0 = _scalar_type(child)
                        # One representative per structural type is enough to
                        # discover names/types; scalar values are never formatted.
                        if typ0 not in by_type:
                            by_type[typ0] = child
                for key, by_type in samples.items():
                    p = f"{item_path}.{key}"
                    for sample in by_type.values():
                        queue.append((sample, p, depth + 1, len(value), presence[key]))
            else:
                seen_types: set[str] = set()
                for child in value:
                    child_type = _scalar_type(child)
                    if child_type in seen_types:
                        continue
                    seen_types.add(child_type)
                    queue.append((child, item_path, depth + 1, len(value), None))

    normalized = []
    for item in entries.values():
        out = {"path": item["path"], "type": "|".join(sorted(item["types"]))}
        if "count" in item:
            out["count"] = item["count"]
        if "present" in item:
            out["present"] = item["present"]
        normalized.append(out)
    return {"paths": normalized, "truncated": truncated}

def format_runtime_structure(profile: dict[str, Any], *, max_chars: int = 1800) -> str:
    parts = []
    for item in profile.get("paths") or []:
        path = item.get("path")
        typ = item.get("type")
        extra = ""
        if item.get("count") is not None:
            extra += f" count={item['count']}"
        if item.get("present") is not None and item.get("count") is not None:
            extra += f" present={item['present']}/{item['count']}"
        parts.append(f"- {path}: {typ}{extra}")
    if profile.get("truncated"):
        parts.append("- … structural path budget reached; profile is incomplete")
    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 24)] + "\n… structure truncated"
    return text or "- $: empty/unknown"


def _path_set_from_profile(profile: dict[str, Any]) -> set[str]:
    return {normalize_path(x.get("path")) for x in (profile.get("paths") or []) if x.get("path")}


def _empty_collection_ancestor(profile: dict[str, Any], record_path: Any) -> str | None:
    """Return the empty runtime array that makes ``record_path`` unobservable.

    An empty array has no runtime item shape, so ``runtime_structure`` can report
    the collection itself but cannot report its ``[*]`` path or any child fields.
    That absence is a valid empty result, not evidence of schema drift. Only an
    array on the requested record path is accepted; unrelated empty arrays do not
    hide genuinely missing paths.
    """
    record = normalize_path(record_path)
    for item in profile.get("paths") or []:
        types = {part.strip() for part in str(item.get("type") or "").split("|")}
        if "array" not in types or item.get("count") != 0:
            continue
        collection = normalize_path(item.get("path"))
        if collection == "$":
            if record in {"$", "[*]"}:
                return collection
            continue
        item_path = f"{collection}[*]"
        if record == item_path or record.startswith(item_path + "."):
            return collection
    return None


def _combine(record_path: str, relative: str) -> str:
    record_path = normalize_path(record_path)
    relative = normalize_path(relative)
    if relative == "$":
        return record_path
    if record_path == "$":
        return relative
    # A planner may return an absolute path despite being asked for a relative one.
    if relative == record_path or relative.startswith(record_path + "."):
        return relative
    return f"{record_path}.{relative}"


def normalize_projection_spec(raw: dict[str, Any] | None,
                              *, step_id: str | None = None) -> dict[str, Any]:
    raw = dict(raw or {})
    spec: dict[str, Any] = {
        "step_id": str(raw.get("step_id") or step_id or ""),
        "record_path": normalize_path(raw.get("record_path") or "$"),
        "project_paths": [],
        "filters": [],
        "sort": [],
        "select": {"mode": "head", "limit": 5},
        "bindings": [],
        "aggregates": [],
        "completeness": str(raw.get("completeness") or "returned_response"),
        "purpose": str(raw.get("purpose") or "task-guided observation projection"),
        "schema_deferred": bool(raw.get("schema_deferred", False)),
    }
    for p in raw.get("project_paths") or raw.get("projection") or []:
        p = normalize_path(p)
        if p not in spec["project_paths"]:
            spec["project_paths"].append(p)
    for item in raw.get("filters") or []:
        if not isinstance(item, dict):
            continue
        op = str(item.get("op") or "eq").lower()
        if op not in _ALLOWED_FILTER_OPS:
            continue
        spec["filters"].append({
            "path": normalize_path(item.get("path")),
            "op": op,
            "value": item.get("value"),
        })
    for item in raw.get("sort") or raw.get("order_by") or []:
        if not isinstance(item, dict) or not item.get("path"):
            continue
        spec["sort"].append({
            "path": normalize_path(item.get("path")),
            "direction": "desc" if str(item.get("direction") or "asc").lower().startswith("d") else "asc",
        })
    select = raw.get("select") if isinstance(raw.get("select"), dict) else {}
    mode = str(select.get("mode") or raw.get("select_mode") or "head").lower()
    if mode not in _ALLOWED_SELECT_MODES:
        mode = "head"
    try:
        limit = min(20, max(1, int(select.get("limit", raw.get("limit", 5)) or 5)))
    except Exception:
        limit = 5
    try:
        index = max(0, int(select.get("index", raw.get("index", 0)) or 0))
    except Exception:
        index = 0
    spec["select"] = {"mode": mode, "limit": limit, "index": index}
    for item in raw.get("bindings") or []:
        if not isinstance(item, dict) or not item.get("name") or not item.get("path"):
            continue
        binding = {
            "name": str(item["name"]),
            "path": normalize_path(item["path"]),
            "source": str(item.get("source") or "selected_first"),
        }
        if item.get("max_values") is not None:
            try:
                binding["max_values"] = min(1000, max(1, int(item.get("max_values"))))
            except Exception:
                pass
        spec["bindings"].append(binding)
    for item in raw.get("aggregates") or []:
        if not isinstance(item, dict):
            continue
        op = str(item.get("op") or "").lower()
        if op not in _ALLOWED_AGG_OPS:
            continue
        spec["aggregates"].append({
            "op": op,
            "path": normalize_path(item.get("path")) if item.get("path") else "$",
            "as": str(item.get("as") or op),
        })
    return spec


def validate_projection_spec(spec: dict[str, Any], *, leaf_paths: set[str],
                             record_paths: set[str]) -> list[str]:
    """Validate a spec against either OAS paths or runtime structural paths."""
    errors: list[str] = []
    record = normalize_path(spec.get("record_path"))
    normalized_records = {normalize_path(x) for x in record_paths}
    normalized_leaves = {normalize_path(x) for x in leaf_paths}
    all_paths = normalized_records | normalized_leaves
    if record not in normalized_records and record not in all_paths:
        errors.append(f"record_path {record!r} not present")

    def check(rel: str, label: str):
        full = _combine(record, rel)
        # Objects/arrays can also be projected, so accept any structural path.
        if full not in all_paths:
            errors.append(f"{label} path {rel!r} -> {full!r} not present")

    for p in spec.get("project_paths") or []:
        check(p, "project")
    for item in spec.get("filters") or []:
        check(item.get("path"), "filter")
    for item in spec.get("sort") or []:
        check(item.get("path"), "sort")
    for item in spec.get("bindings") or []:
        check(item.get("path"), "binding")
    for item in spec.get("aggregates") or []:
        if item.get("path") and normalize_path(item.get("path")) != "$":
            check(item.get("path"), "aggregate")
    return list(dict.fromkeys(errors))


def validate_projection_spec_runtime(spec: dict[str, Any], profile: dict[str, Any]) -> list[str]:
    # Runtime data cannot expose item or leaf paths beneath an empty array. The
    # spec was already checked against the endpoint schema, so the collection's
    # presence is sufficient to certify a valid empty projection.
    if _empty_collection_ancestor(profile, spec.get("record_path")) is not None:
        return []
    paths = _path_set_from_profile(profile)
    records = {p for p in paths if p == "$" or p.endswith("[*]")}
    # Runtime object paths are valid record roots too.
    for item in profile.get("paths") or []:
        if item.get("type") in {"object", "array_item"}:
            records.add(normalize_path(item.get("path")))
    return validate_projection_spec(spec, leaf_paths=paths, record_paths=records)


def _first(values: list[Any]) -> Any:
    return values[0] if values else None


def _relative_value(record: Any, path: Any) -> Any:
    vals = extract_values(record, path)
    if not vals:
        return None
    return vals[0] if len(vals) == 1 else vals


def _compare(value: Any, op: str, expected: Any) -> bool:
    """Compatibility wrapper around the canonical host predicate semantics."""
    return compare_predicate(value, op, expected)


def _sort_key(value: Any):
    if value is None:
        return (1, "")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and math.isnan(value):
            return (1, "")
        return (0, value)
    return (0, str(value))


def filter_sort_select_records(records: list[Any], spec: dict[str, Any]
                               ) -> tuple[list[Any], list[Any]]:
    """Replay a normalized projection's filter, sort, and selection policy.

    Keeping this operation separate from display rendering lets the evidence
    compiler certify the exact same records Phase A saw without duplicating or
    subtly changing comparison semantics.
    """
    filtered = list(records)
    for filt in spec.get("filters") or []:
        next_rows = []
        for row in filtered:
            value = _relative_value(row, filt["path"])
            if _compare(value, filt["op"], filt.get("value")):
                next_rows.append(row)
        filtered = next_rows

    # Python's stable sort lets us apply lower-priority keys first. Missing
    # values stay last in BOTH directions; reversing a tuple key would otherwise
    # incorrectly put missing values first for descending/latest rankings.
    for sorter in reversed(spec.get("sort") or []):
        reverse = sorter.get("direction") == "desc"
        present, missing = [], []
        domains = set()
        for row in filtered:
            value = _relative_value(row, sorter["path"])
            if value is None:
                missing.append(row)
                continue
            present.append(row)
            if isinstance(value, bool): domains.add("bool")
            elif isinstance(value, (int, float)): domains.add("number")
            elif isinstance(value, str): domains.add("string")
            else: domains.add("other")
        if len(domains) > 1 or "other" in domains:
            return [], []
        try:
            present.sort(
                key=lambda row, p=sorter["path"]: _relative_value(row, p),
                reverse=reverse)
        except Exception:
            return [], []
        filtered = present + missing

    select = spec.get("select") or {"mode": "head", "limit": 5, "index": 0}
    mode = select.get("mode")
    limit = int(select.get("limit", 5))
    index = int(select.get("index", 0))
    if mode in {"head", "top", "all_matches"}:
        selected = filtered[:limit]
    elif mode == "tail":
        selected = filtered[-limit:]
    elif mode == "nth":
        selected = [filtered[index]] if 0 <= index < len(filtered) else []
    elif mode == "single":
        selected = filtered[:1]
    else:
        selected = filtered[:limit]
    return filtered, selected


def _short(value: Any, max_chars: int = 180) -> str:
    value = redact_secrets(value)
    if isinstance(value, str):
        text = value.replace("\n", " ").strip()
        if len(text) > max_chars:
            text = text[: max_chars - 1] + "…"
        return repr(text)
    if isinstance(value, (int, float, bool)) or value is None:
        return repr(value)
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        text = repr(value)
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    return text


@dataclass
class ProjectionResult:
    text: str
    errors: list[str]
    profile: dict[str, Any]
    stats: dict[str, Any]

    @property
    def needs_repair(self) -> bool:
        return bool(self.errors)



def _resolve_deferred_projection_spec(spec: dict[str, Any], profile: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Resolve an OAS-undocumented response record root from observed structure.

    The request route/arguments have already been OAS-validated.  This fallback is
    used only when the selected operation documents no response leaves at all.  It
    never guesses from endpoint names or values: a record universe is accepted only
    when exactly one runtime structural root exposes every planner-declared
    projection/filter/sort/binding path.  Ambiguity fails closed.
    """
    out = dict(spec or {})
    if not out.get("schema_deferred"):
        return out, []
    paths = {normalize_path(x.get("path")) for x in (profile.get("paths") or []) if x.get("path")}
    roots = []
    for item in profile.get("paths") or []:
        path = normalize_path(item.get("path"))
        typ = str(item.get("type") or "")
        if path == "$" or "array_item" in typ or typ == "object" or "object" in typ.split("|"):
            roots.append(path)
    roots = list(dict.fromkeys(roots))
    wanted = list(out.get("project_paths") or [])
    wanted += [str(x.get("path") or "") for x in (out.get("filters") or [])]
    wanted += [str(x.get("path") or "") for x in (out.get("sort") or [])]
    wanted += [str(x.get("path") or "") for x in (out.get("bindings") or [])]
    wanted = [normalize_path(x) for x in wanted if x and normalize_path(x) != "$"]

    def relative(root: str, raw: str) -> str:
        r = normalize_path(root).replace("[*]", "")
        x = normalize_path(raw).replace("[*]", "")
        if r not in {"", "$"} and (x == r or x.startswith(r + ".")):
            x = x[len(r):].lstrip(".")
        return x or "$"

    def full(root: str, raw: str) -> str:
        rel = relative(root, raw)
        if rel == "$":
            return normalize_path(root)
        if normalize_path(root) == "$":
            return normalize_path(rel)
        return normalize_path(str(root).rstrip(".") + "." + rel)

    candidates = []
    for root in roots:
        if all(full(root, w) in paths for w in wanted):
            candidates.append(root)
    # Prefer item/object roots over the response envelope when both satisfy an
    # unqualified scalar path; the envelope is not a candidate record collection.
    non_root = [r for r in candidates if r != "$"]
    if non_root:
        candidates = non_root
    if len(candidates) != 1:
        return out, [
            "runtime-observed response schema could not identify one unambiguous "
            f"record universe for deferred fields {wanted}: candidates={candidates[:8]}"
        ]
    root = candidates[0]
    out["record_path"] = root
    out["project_paths"] = [relative(root, x) for x in (out.get("project_paths") or [])]
    for key in ("filters", "sort", "bindings"):
        rows = []
        for raw in out.get(key) or []:
            row = dict(raw)
            if row.get("path"):
                row["path"] = relative(root, row["path"])
            rows.append(row)
        out[key] = rows
    return out, []

def project_payload(payload: Any, spec: dict[str, Any], *, max_chars: int = 2600) -> ProjectionResult:
    """Apply a validated planner spec to one full runtime payload."""
    payload = redact_secrets(payload)
    spec = normalize_projection_spec(spec, step_id=str(spec.get("step_id") or ""))
    profile = runtime_structure(payload)
    spec, deferred_errors = _resolve_deferred_projection_spec(spec, profile)
    errors = list(deferred_errors)
    if not errors:
        errors = validate_projection_spec_runtime(spec, profile)
    record_path = spec["record_path"]
    records = extract_values(payload, record_path)
    # ``extract_values`` on $ yields the whole root.  Flatten a direct list root.
    if record_path == "$" and len(records) == 1 and isinstance(records[0], list):
        records = list(records[0])
    total = len(records)
    empty_result = (
        not errors
        and total == 0
        and _empty_collection_ancestor(profile, record_path) is not None
    )

    filtered, selected = filter_sort_select_records(records, spec)

    empty_marker = " empty_result=true" if empty_result else ""
    lines = [
        f"projection_status={'mismatch' if errors else 'ok'}{empty_marker} "
        f"records={total} matched={len(filtered)} selected={len(selected)} "
        f"completeness={spec.get('completeness')}",
    ]
    for i, row in enumerate(selected):
        pieces = []
        paths = list(spec.get("project_paths") or [])
        # A binding path is automatically included in display even if the planner
        # forgot to repeat it under project_paths. This is plan-derived, not a key heuristic.
        for binding in spec.get("bindings") or []:
            if binding["path"] not in paths:
                paths.append(binding["path"])
        for path in paths:
            value = _relative_value(row, path)
            if value is not None:
                pieces.append(f"{path}={_short(value)}")
        lines.append(f"selected[{i}] " + (" ".join(pieces) if pieces else "(record selected)"))

    for binding in spec.get("bindings") or []:
        source = binding.get("source") or "selected_first"
        # ``selected_all`` is a data-flow contract, not a display limit.  The
        # select.limit field controls how many rows are shown to the model, while
        # a downstream fan-out binding must retain every filtered candidate from
        # the returned collection. Conflating the two silently truncates evidence.
        rows = filtered if source == "selected_all" else selected[:1]
        if source == "selected_all" and binding.get("max_values"):
            try:
                rows = rows[:max(1, int(binding.get("max_values")))]
            except Exception:
                pass
        vals = [_relative_value(row, binding["path"]) for row in rows]
        vals = [v for v in vals if v is not None]
        if vals:
            value = vals if source == "selected_all" else vals[0]
            lines.append(f"binding {binding['name']}={_short(value)}")

    for agg in spec.get("aggregates") or []:
        op = agg["op"]
        if op == "count" and normalize_path(agg.get("path")) == "$":
            value = len(filtered)
        else:
            vals = []
            for row in filtered:
                v = _relative_value(row, agg.get("path"))
                if isinstance(v, list):
                    vals.extend(v)
                elif v is not None:
                    vals.append(v)
            if op == "count":
                value = len(vals)
            else:
                nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
                if op == "sum":
                    value = sum(nums) if nums else None
                elif op == "mean":
                    value = (sum(nums) / len(nums)) if nums else None
                elif op == "min":
                    value = min(vals) if vals else None
                elif op == "max":
                    value = max(vals) if vals else None
                else:
                    value = None
        lines.append(f"aggregate {agg.get('as')}={_short(value)}")

    if errors:
        lines.append("projection_errors: " + "; ".join(errors[:8]))
        lines.append("runtime_structure (field names/types/counts only):")
        lines.append(format_runtime_structure(profile, max_chars=max(500, max_chars // 2)))

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 28)] + "\n… projection budget reached"
    return ProjectionResult(
        text=text,
        errors=errors,
        profile=profile,
        stats={
            "records": total,
            "matched": len(filtered),
            "selected": len(selected),
            "empty_result": empty_result,
        },
    )


def _safe_params(params: Any) -> dict[str, Any]:
    if not isinstance(params, dict):
        return {}
    return {str(k): redact_secrets(v) for k, v in params.items() if not _SECRET_RE.search(str(k))}


def _step_assignments(plan: dict[str, Any] | None, entries: list[dict[str, Any]]) -> dict[int, str]:
    try:
        from utils.evidence_plan import assign_calls_to_steps
        calls = [{"endpoint": e.get("endpoint", ""), "method": e.get("method", "GET"),
                  "params": e.get("params") or {}} for e in entries]
        mapped = assign_calls_to_steps(plan, calls)
        return {i: mapped.get(f"call_index_{i}") for i in range(len(calls)) if mapped.get(f"call_index_{i}")}
    except Exception:
        return {}


def adaptive_feedback(entries: list[dict[str, Any]], result: dict[str, Any], *,
                      question: str = "", plan: dict[str, Any] | None = None,
                      all_entries: list[dict[str, Any]] | None = None,
                      call_offset: int = 0, max_chars: int = 3500,
                      local_max_chars: int = 1200) -> tuple[str, list[dict[str, Any]]]:
    """Render planner-guided feedback plus machine-readable mismatch diagnostics."""
    all_entries = list(all_entries or entries)
    assignments = _step_assignments(plan, all_entries)
    specs = {str(s.get("step_id")): normalize_projection_spec(s)
             for s in ((plan or {}).get("observation_specs") or []) if s.get("step_id")}
    chunks: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    remaining = max(400, int(max_chars))

    for j, raw in enumerate(entries):
        global_index = call_offset + j
        entry = redact_secrets(raw)
        step_id = assignments.get(global_index) or str(entry.get("plan_step_id") or "")
        method = str(entry.get("method") or "GET").upper()
        endpoint = str(entry.get("endpoint") or "?")
        status = entry.get("status_code")
        head = f"call {global_index + 1}: {method} {endpoint} -> HTTP {status}"
        if step_id:
            head += f" [plan {step_id}]"
        params = _safe_params(entry.get("params") or {})
        if params:
            head += " params=" + _short(params, 240)
        chunks.append(head)
        remaining -= len(head) + 1
        if remaining <= 100:
            break

        spec = specs.get(step_id)
        if spec:
            projected = project_payload(entry.get("payload"), spec,
                                        max_chars=max(500, min(2600, remaining)))
            chunks.append(projected.text)
            remaining -= len(projected.text) + 1
            if projected.needs_repair:
                diagnostics.append({
                    "step_id": step_id,
                    "errors": projected.errors,
                    "profile": projected.profile,
                    "structure_text": format_runtime_structure(projected.profile),
                    "entry_index": global_index,
                })
        else:
            profile = runtime_structure(entry.get("payload"))
            structure = format_runtime_structure(profile, max_chars=max(500, min(1600, remaining)))
            text = ("projection_status=unplanned; no valid field projection exists for this call.\n"
                    "runtime_structure (field names/types/counts only):\n" + structure)
            chunks.append(text)
            remaining -= len(text) + 1
            if step_id:
                diagnostics.append({
                    "step_id": step_id,
                    "errors": ["no observation projection spec for executed plan step"],
                    "profile": profile,
                    "structure_text": structure,
                    "entry_index": global_index,
                })

    # Keep local computation/error output but suppress giant redundant JSON bodies.
    stderr = str(result.get("stderr") or "").strip()
    raw_output = str(result.get("auto_display") or result.get("combined") or
                     result.get("stdout") or "").strip()
    local = ""
    if stderr:
        local = ("Execution error/output:\n" + stderr)[:local_max_chars]
    elif not entries and raw_output:
        local = raw_output[:local_max_chars]
    elif raw_output and len(raw_output) <= min(local_max_chars, 700):
        local = raw_output
    if local and remaining > 80:
        chunks.append("Local execution output:\n" + local[:remaining])

    text = "\n".join(chunks) if chunks else local or "(execution produced no visible output)"
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 28)] + "\n… projection budget reached"
    return text, diagnostics
