"""Compact endpoint-schema outlines for adaptive OCA observation planning.

This module reads the benchmark's own OpenAPI/OAS description and exposes only
structural information (paths, types, requiredness, collection shape).  It never
uses benchmark answers or live response values.  The adaptive observation planner
uses these cards to choose which response fields are needed for the current task.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import benchmarks as B

_HTTP_METHODS = {"get", "post", "put", "delete", "patch"}


def _oas_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no", ""}:
            return False
    return default


def _oas_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            number = float(value.strip())
            return int(number) if number.is_integer() else number
        except Exception:
            return None
    return None


def _resolve_ref(root: dict[str, Any], value: Any, *, depth: int = 0,
                 seen: set[str] | None = None) -> Any:
    """Resolve local ``#/...`` references without mutating the source document."""
    if depth > 18:
        return value
    seen = set(seen or ())
    if isinstance(value, dict) and isinstance(value.get("$ref"), str):
        ref = value["$ref"]
        if not ref.startswith("#/") or ref in seen:
            return value
        node: Any = root
        try:
            for part in ref[2:].split("/"):
                part = part.replace("~1", "/").replace("~0", "~")
                node = node[part]
        except Exception:
            return value
        return _resolve_ref(root, node, depth=depth + 1, seen=seen | {ref})
    return value


def _merge_schema(root: dict[str, Any], schema: Any, *, depth: int = 0,
                  seen: set[str] | None = None) -> dict[str, Any]:
    """Return a shallowly dereferenced/merged schema suitable for traversal."""
    if not isinstance(schema, dict) or depth > 18:
        return schema if isinstance(schema, dict) else {}
    schema = _resolve_ref(root, schema, depth=depth, seen=seen)
    if not isinstance(schema, dict):
        return {}
    if "allOf" in schema and isinstance(schema["allOf"], list):
        merged: dict[str, Any] = {k: v for k, v in schema.items() if k != "allOf"}
        props = dict(merged.get("properties") or {})
        required = list(merged.get("required") or [])
        for part in schema["allOf"]:
            child = _merge_schema(root, part, depth=depth + 1, seen=seen)
            props.update(child.get("properties") or {})
            required.extend(child.get("required") or [])
            for key in ("type", "items", "additionalProperties"):
                if key not in merged and key in child:
                    merged[key] = child[key]
        if props:
            merged["properties"] = props
        if required:
            merged["required"] = list(dict.fromkeys(required))
        return merged
    return schema


def _schema_type(schema: dict[str, Any]) -> str:
    typ = schema.get("type")
    if isinstance(typ, list):
        typ = "|".join(str(x) for x in typ)
    if typ:
        return str(typ)
    if "properties" in schema:
        return "object"
    if "items" in schema:
        return "array"
    if "oneOf" in schema:
        return "oneOf"
    if "anyOf" in schema:
        return "anyOf"
    return "unknown"


def flatten_schema(root: dict[str, Any], schema: Any, *, max_depth: int = 7,
                   max_paths: int = 180) -> dict[str, Any]:
    """Flatten a response schema into leaf paths and record/container paths.

    Paths use a small JSONPath-like notation: ``results[*].id``.  ``$`` denotes
    the whole root object.  The output is intentionally compact enough to send to
    an LLM only for endpoints already selected by the evidence plan.
    """
    leaves: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = [{"path": "$", "type": "root"}]
    seen_pairs: set[tuple[int, str]] = set()
    truncated = False

    def add_leaf(path: str, typ: str, required: bool = False):
        nonlocal truncated
        if not path:
            return
        if len(leaves) >= max_paths:
            truncated = True
            return
        item = {"path": path, "type": typ}
        if required:
            item["required"] = True
        if item not in leaves:
            leaves.append(item)

    def add_record(path: str, typ: str):
        if not path:
            path = "$"
        item = {"path": path, "type": typ}
        if item not in records:
            records.append(item)

    def walk(raw: Any, prefix: str, depth: int, required: bool = False):
        nonlocal truncated
        if len(leaves) >= max_paths or depth > max_depth:
            truncated = True
            return
        schema0 = _merge_schema(root, raw, depth=depth)
        if not schema0:
            return
        marker = (id(raw), prefix)
        if marker in seen_pairs:
            return
        seen_pairs.add(marker)

        # Union alternatives are traversed into one structural union.  This does
        # not assert which branch will occur at runtime; it only lists legal paths.
        variants = schema0.get("oneOf") or schema0.get("anyOf")
        if isinstance(variants, list):
            for child in variants:
                walk(child, prefix, depth + 1, required=required)
            return

        typ = _schema_type(schema0)
        if typ == "array" or "items" in schema0:
            item_prefix = f"{prefix}[*]" if prefix else "[*]"
            add_record(item_prefix, "array_item")
            walk(schema0.get("items") or {}, item_prefix, depth + 1)
            return

        props = schema0.get("properties")
        if isinstance(props, dict):
            add_record(prefix or "$", "object")
            required_names = set(schema0.get("required") or [])
            for key, child in props.items():
                child_path = f"{prefix}.{key}" if prefix else str(key)
                walk(child, child_path, depth + 1, required=key in required_names)
            return

        add_leaf(prefix or "$", typ, required=required)

    walk(schema, "", 0)
    return {"leaf_paths": leaves, "record_paths": records, "schema_truncated": truncated}


def _parameter_card(root: dict[str, Any], raw: Any) -> dict[str, Any] | None:
    param = _resolve_ref(root, raw)
    if not isinstance(param, dict) or not param.get("name"):
        return None
    schema = _merge_schema(root, param.get("schema") or {})
    card = {
        "name": str(param.get("name")),
        "in": str(param.get("in") or "query"),
        "required": _oas_bool(param.get("required", False)),
        "type": _schema_type(schema),
    }
    if isinstance(schema.get("enum"), list) and schema.get("enum"):
        card["enum"] = list(schema.get("enum") or [])[:40]
    items = _merge_schema(root, schema.get("items") or {}) if isinstance(schema.get("items"), dict) else {}
    if isinstance(items.get("enum"), list) and items.get("enum"):
        card["item_enum"] = list(items.get("enum") or [])[:60]
        card["item_type"] = _schema_type(items)
    if param.get("style") is not None:
        card["style"] = str(param.get("style"))
    if param.get("explode") is not None:
        card["explode"] = _oas_bool(param.get("explode"), default=True)
    for key in ("minimum", "maximum", "minLength", "maxLength"):
        parsed = _oas_number(schema.get(key))
        if parsed is not None:
            card[key] = parsed
    description = str(param.get("description") or "").strip().replace("\n", " ")
    if description:
        card["description"] = description[:240]
    return card


def _openapi_request_body_schema(root: dict[str, Any], operation: dict[str, Any]) -> tuple[Any, bool]:
    """Return the JSON request-body schema and whether the body itself is required."""
    raw = _resolve_ref(root, operation.get("requestBody") or {})
    if not isinstance(raw, dict):
        return {}, False
    content = raw.get("content") or {}
    if isinstance(content, dict):
        media = (content.get("application/json") or content.get("application/*+json") or
                 next((v for k, v in content.items() if "json" in str(k).lower()), None))
        media = _resolve_ref(root, media or {})
        if isinstance(media, dict) and media.get("schema"):
            return media.get("schema"), _oas_bool(raw.get("required", False))
    return {}, _oas_bool(raw.get("required", False))


def _top_level_required_fields(root: dict[str, Any], schema: Any) -> list[str]:
    merged = _merge_schema(root, schema or {})
    if not isinstance(merged, dict):
        return []
    properties = set(str(x) for x in (merged.get("properties") or {}))
    # Some published OpenAPI documents contain stale required names that are not
    # actual properties. They cannot be constructed or validated safely, so only
    # enforce required names that the same schema documents structurally.
    return [str(x) for x in (merged.get("required") or []) if str(x) in properties]


def _openapi_response_schema(root: dict[str, Any], operation: dict[str, Any]) -> tuple[Any, str]:
    responses = operation.get("responses") or {}
    keys = sorted(
        [str(k) for k in responses if str(k).isdigit() and 200 <= int(str(k)) < 300],
        key=lambda x: int(x))
    if not keys:
        keys = [str(k) for k in responses if str(k).startswith("2")]
    if not keys:
        return {}, "unknown"
    code = keys[0]
    response = _resolve_ref(root, responses.get(code) or {})
    if not isinstance(response, dict):
        return {}, code
    content = response.get("content") or {}
    if isinstance(content, dict):
        media = (content.get("application/json") or content.get("application/*+json") or
                 next((v for k, v in content.items() if "json" in str(k).lower()), None))
        media = _resolve_ref(root, media or {})
        if isinstance(media, dict) and media.get("schema"):
            return media.get("schema"), code
    # 204/no-content and some action endpoints legitimately have no JSON schema.
    return {}, code


def selected_endpoint_cards(benchmark: str, plan: dict[str, Any] | None,
                            *, max_paths_per_endpoint: int = 160,
                            max_request_paths_per_endpoint: int = 80,
                            max_depth: int = 7,
                            max_request_depth: int = 7) -> list[dict[str, Any]]:
    """Return compact schema cards only for endpoints selected by ``plan``."""
    spec = B.get_benchmark(benchmark)
    path = Path(spec["oas_file"])
    root = json.loads(path.read_text(encoding="utf-8"))
    cards: list[dict[str, Any]] = []
    steps = list((plan or {}).get("steps") or [])

    if isinstance(root, list):
        by_path = {str(item.get("path")): item for item in root if isinstance(item, dict)}
        for step in steps:
            endpoint = str(step.get("endpoint") or "")
            item = by_path.get(endpoint) or {}
            outline = flatten_schema({}, item.get("schema") or {},
                                     max_paths=max_paths_per_endpoint, max_depth=max_depth)
            params = []
            for p in item.get("parameters") or []:
                if isinstance(p, dict) and p.get("name"):
                    params.append({
                        "name": str(p["name"]), "in": str(p.get("in") or "query"),
                        "required": _oas_bool(p.get("required", False)),
                        "type": _schema_type(p.get("schema") or {}),
                    })
            cards.append({
                "step_id": str(step.get("id")),
                "method": str(step.get("method") or "GET").upper(),
                "endpoint": endpoint,
                "parameters": params,
                "request_body_required": False,
                "request_body_required_fields": [],
                "request_body_leaf_paths": [],
                "request_body_record_paths": [],
                "response_status": "2xx",
                **outline,
            })
        return cards

    paths = root.get("paths") or {} if isinstance(root, dict) else {}
    for step in steps:
        endpoint = str(step.get("endpoint") or "")
        method = str(step.get("method") or "GET").lower()
        path_item = paths.get(endpoint) or {}
        operation = path_item.get(method) if isinstance(path_item, dict) else None
        operation = operation if isinstance(operation, dict) else {}
        params = []
        for p in list(path_item.get("parameters") or []) + list(operation.get("parameters") or []):
            card = _parameter_card(root, p)
            if card and card not in params:
                params.append(card)
        response_schema, code = _openapi_response_schema(root, operation)
        outline = flatten_schema(root, response_schema,
                                 max_paths=max_paths_per_endpoint, max_depth=max_depth)
        request_schema, request_required = _openapi_request_body_schema(root, operation)
        request_outline = flatten_schema(root, request_schema, max_paths=max_request_paths_per_endpoint,
                                         max_depth=max_request_depth) if request_schema else {
            "leaf_paths": [], "record_paths": [], "schema_truncated": False}
        cards.append({
            "step_id": str(step.get("id")),
            "method": method.upper(),
            "endpoint": endpoint,
            "parameters": params,
            "request_body_required": request_required,
            "request_body_required_fields": _top_level_required_fields(root, request_schema),
            "request_body_leaf_paths": request_outline.get("leaf_paths") or [],
            "request_body_record_paths": request_outline.get("record_paths") or [],
            "request_body_schema_truncated": bool(request_outline.get("schema_truncated")),
            "response_status": code,
            **outline,
        })
    return cards


def selected_endpoint_validation_cards(benchmark: str, plan: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return effectively untruncated selected-operation schemas for host validation.

    Prompt formatting may intentionally use compact schema outlines, but deterministic
    validation/compilation must never reject a valid plan merely because a required
    leaf sorted beyond a prompt-oriented path cap. Selected endpoints are few, so a
    generous structural cap is inexpensive and does not add model tokens.
    """
    return selected_endpoint_cards(
        benchmark, plan, max_paths_per_endpoint=5000,
        max_request_paths_per_endpoint=5000,
        max_depth=30, max_request_depth=30)


def endpoint_catalog_cards(benchmark: str, *, max_paths_per_endpoint: int = 40) -> list[dict[str, Any]]:
    """Return compact structural cards for every documented API operation.

    These cards contain schema names/types only: no examples, response values,
    benchmark solutions, or expected answers. They let the route planner avoid
    unnecessary downstream calls when an upstream response already exposes the
    fields required for a deterministic derivation.
    """
    spec = B.get_benchmark(benchmark)
    root = json.loads(Path(spec["oas_file"]).read_text(encoding="utf-8"))
    cards: list[dict[str, Any]] = []
    if isinstance(root, list):
        for index, item in enumerate(root):
            if not isinstance(item, dict) or not item.get("path"):
                continue
            params = []
            for p in item.get("parameters") or []:
                if isinstance(p, dict) and p.get("name"):
                    params.append({
                        "name": str(p["name"]), "in": str(p.get("in") or "query"),
                        "required": _oas_bool(p.get("required", False)),
                        "type": _schema_type(p.get("schema") or {}),
                    })
            outline = flatten_schema({}, item.get("schema") or {}, max_paths=max_paths_per_endpoint)
            cards.append({
                "step_id": f"catalog_{index}",
                "method": str(item.get("method") or "GET").upper(),
                "endpoint": str(item.get("path")),
                "description": str(item.get("functionality") or item.get("description") or ""),
                "parameters": params,
                "request_body_required": False,
                "request_body_required_fields": [],
                "request_body_leaf_paths": [],
                **outline,
            })
        return cards

    paths = root.get("paths") or {} if isinstance(root, dict) else {}
    for endpoint, path_item0 in paths.items():
        path_item = path_item0 if isinstance(path_item0, dict) else {}
        for method, operation0 in path_item.items():
            if str(method).lower() not in _HTTP_METHODS:
                continue
            operation = operation0 if isinstance(operation0, dict) else {}
            params = []
            for rawp in list(path_item.get("parameters") or []) + list(operation.get("parameters") or []):
                card = _parameter_card(root, rawp)
                if card and card not in params:
                    params.append(card)
            response_schema, _ = _openapi_response_schema(root, operation)
            outline = flatten_schema(root, response_schema, max_paths=max_paths_per_endpoint)
            request_schema, request_required = _openapi_request_body_schema(root, operation)
            request_outline = flatten_schema(root, request_schema, max_paths=40) if request_schema else {
                "leaf_paths": [], "record_paths": [], "schema_truncated": False}
            cards.append({
                "step_id": f"catalog_{len(cards)}",
                "method": str(method).upper(),
                "endpoint": str(endpoint),
                "description": str(operation.get("summary") or operation.get("description") or ""),
                "parameters": params,
                "request_body_required": request_required,
                "request_body_required_fields": _top_level_required_fields(root, request_schema),
                "request_body_leaf_paths": request_outline.get("leaf_paths") or [],
                "request_body_record_paths": request_outline.get("record_paths") or [],
                **outline,
            })
    return cards


def compact_catalog_line(card: dict[str, Any], *, max_response_fields: int = 45,
                         max_body_fields: int = 24, max_chars: int = 950) -> str:
    """Serialize one operation card for the initial route planner compactly."""
    params = []
    for p in card.get("parameters") or []:
        name = str(p.get("name") or "")
        if not name:
            continue
        enum = p.get("enum") if isinstance(p.get("enum"), list) else []
        item_enum = p.get("item_enum") if isinstance(p.get("item_enum"), list) else []
        values = enum or item_enum
        enum_hint = "{" + ",".join(str(x) for x in values[:8]) + "}" if values and len(values) <= 8 else ""
        params.append(f"{p.get('in','query')}:{name}:{p.get('type','?')}" +
                      ("*" if p.get("required") else "") + enum_hint)
    response_tree = _compact_path_tree((card.get("leaf_paths") or [])[:max_response_fields])
    body_tree = _compact_path_tree((card.get("request_body_leaf_paths") or [])[:max_body_fields])
    pieces = [f"{card.get('method','GET')} {card.get('endpoint')}"]
    desc = str(card.get("description") or "").strip().replace("\n", " ")
    if desc:
        pieces.append(desc[:200])
    if params:
        pieces.append("params=" + ",".join(params))
    if body_tree:
        pieces.append("body=" + body_tree)
    if response_tree:
        pieces.append("returns=" + response_tree)
    line = " | ".join(pieces)
    return line if len(line) <= max_chars else line[:max_chars-1] + "…"


def _compact_path_tree(leaf_paths: list[dict[str, Any]]) -> str:
    """Group repeated dotted prefixes so complete schemas cost fewer tokens."""
    tree: dict[str, Any] = {}
    for item in leaf_paths:
        path = str(item.get("path") or "$")
        typ = str(item.get("type") or "unknown")
        required = bool(item.get("required"))
        parts = path.split(".") if path != "$" else ["$"]
        node = tree
        for part in parts:
            node = node.setdefault(part, {})
        node["__leaf__"] = typ + ("!" if required else "")

    short_types = {
        "string": "str", "integer": "int", "number": "num",
        "boolean": "bool", "object": "obj", "array": "arr", "unknown": "?",
    }

    def emit(node: dict[str, Any]) -> str:
        pieces = []
        for key, child in node.items():
            if key == "__leaf__":
                continue
            leaf = child.get("__leaf__") if isinstance(child, dict) else None
            nested = {k: v for k, v in child.items() if k != "__leaf__"} if isinstance(child, dict) else {}
            label = key
            if leaf:
                req = "!" if str(leaf).endswith("!") else ""
                typ = str(leaf).rstrip("!")
                label += ":" + short_types.get(typ, typ) + req
            if nested:
                label += "{" + emit(nested) + "}"
            pieces.append(label)
        return ",".join(pieces)

    return emit(tree)


def format_endpoint_cards(cards: list[dict[str, Any]], *, max_chars: int = 24000) -> str:
    """Serialize selected cards compactly without dropping documented leaf paths."""
    blocks: list[str] = []
    remaining = max(1000, int(max_chars))
    for card in cards:
        params = ", ".join(
            f"{p['name']}:{p['type']}{'*' if p.get('required') else ''}" +
            (("{" + ",".join(str(x) for x in ((p.get('enum') or p.get('item_enum') or [])[:8])) + "}")
             if isinstance((p.get('enum') or p.get('item_enum')), list) and
                (p.get('enum') or p.get('item_enum')) and len(p.get('enum') or p.get('item_enum')) <= 8 else "")
            for p in card.get("parameters") or []) or "(none)"
        all_records = list(dict.fromkeys(
            str(x["path"]) for x in card.get("record_paths") or []))
        # A concise hint is enough: collection item roots plus direct root objects.
        # The complete object nesting remains visible in response_tree and is used
        # by the validator even when it is not repeated in this hint line.
        roots = [p for p in all_records if p == "$" or
                 (p.endswith("[*]") and p.count(".") <= 1) or
                 ("." not in p and "[*]" not in p)]
        record_paths = ", ".join(roots) or "$"
        leaves = _compact_path_tree(card.get("leaf_paths") or []) or "(no JSON response body documented)"
        schema_note = (
            "\nschema_note: structural outline hit its safety budget; runtime structure "
            "repair is allowed if a needed path is absent."
            if card.get("schema_truncated") else ""
        )
        body_leaves = _compact_path_tree(card.get("request_body_leaf_paths") or [])
        body_line = (f"\nrequest_body: {body_leaves}" if body_leaves else "")
        block = (
            f"STEP {card.get('step_id')} {card.get('method')} {card.get('endpoint')}\n"
            f"params: {params}{body_line}\n"
            f"record_root_hints: {record_paths}\n"
            f"response_tree: {leaves}{schema_note}"
        )
        if len(block) > remaining:
            block = block[: max(0, remaining - 20)] + "\n… schema budget"
        blocks.append(block)
        remaining -= len(block) + 2
        if remaining <= 100:
            break
    return "\n\n".join(blocks)

