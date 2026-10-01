"""Generic deterministic Phase-A receipts for OCA.

The lossless HTTP response remains in the observation ledger.  This module only
builds a bounded transcript view from structural properties plus fields named by
the validated plan.  It contains no API/domain/entity vocabulary.
"""
from __future__ import annotations

import json
import re
from typing import Any

from utils.observation_ledger import redact_secrets

_SECRET_HINTS = ("token", "secret", "password", "api_key", "authorization")


def _norm_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")


def _plan_interest(plan: dict | None) -> set[str]:
    """Return field/binding names explicitly declared by the validated plan."""
    out: set[str] = set()
    for step in (plan or {}).get("steps") or []:
        out.update(_norm_key(x) for x in step.get("binds") or [] if x)
        out.update(_norm_key(x) for x in re.findall(r"\{([^{}]+)\}", str(step.get("endpoint") or "")))
    for d in (plan or {}).get("derivations") or []:
        if d.get("field"): out.add(_norm_key(d["field"]))
        for key, value in (d.get("filter") or {}).items():
            out.add(_norm_key(key))
            if isinstance(value, (str, int, float)):
                out.update(_norm_key(x) for x in re.findall(r"[A-Za-z0-9_]+", str(value)) if x)
    for spec in (plan or {}).get("observation_specs") or []:
        out.update(_norm_key(x) for x in spec.get("project_paths") or [] if x)
        for item in spec.get("filters") or []:
            if item.get("path"): out.add(_norm_key(item["path"]))
        for item in spec.get("sort") or []:
            if item.get("path"): out.add(_norm_key(item["path"]))
        for binding in spec.get("bindings") or []:
            if binding.get("name"): out.add(_norm_key(binding["name"]))
            if binding.get("path"): out.add(_norm_key(binding["path"]))
    return {x for x in out if x}


def _is_secret_key(key: str) -> bool:
    norm = _norm_key(key)
    return any(h in norm for h in _SECRET_HINTS)


def _key_score(key: str, value: Any, interest: set[str]) -> int:
    norm = _norm_key(key)
    if _is_secret_key(norm):
        return -1000
    score = 0
    # Plan-declared fields dominate the receipt.
    if norm in interest or any(part in interest for part in norm.split("_")):
        score += 200
    # Structural chaining convention only; no resource/entity names.
    if norm == "id" or norm.endswith(("_id", "_uri", "_url", "_href", "_path")):
        score += 100
    if isinstance(value, bool): score += 35
    elif isinstance(value, (int, float)): score += 30
    elif isinstance(value, str):
        if len(value) > 180 and norm not in interest:
            score -= 40
        else:
            score += 25
    return score


def _short(value: Any, max_chars: int = 120) -> str:
    value = redact_secrets(value)
    if isinstance(value, str):
        text = value.replace("\n", " ").strip()
        if len(text) > max_chars: text = text[: max_chars - 1] + "…"
        return repr(text)
    if isinstance(value, (int, float, bool)) or value is None:
        return repr(value)
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, dict) else repr(value)
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def _project_record(record: dict[str, Any], interest: set[str], max_fields: int = 10,
                    max_chars: int = 360) -> str:
    scored = []
    for key, value in record.items():
        if isinstance(value, (dict, list)):
            continue
        scored.append((_key_score(key, value, interest), str(key), value))
    scored.sort(key=lambda x: (-x[0], x[1]))
    pieces = []
    for score, key, value in scored:
        if score < 0: continue
        piece = f"{key}={_short(value)}"
        if pieces and len(" ".join(pieces + [piece])) > max_chars: break
        pieces.append(piece)
        if len(pieces) >= max_fields: break
    return " ".join(pieces) if pieces else "(record present)"


def _value_for_field(item: dict[str, Any], field: str):
    # Support dotted paths selected by the plan.
    value: Any = item
    for part in [x for x in str(field or "").replace("$.", "").split(".") if x]:
        if not isinstance(value, dict): return None
        if part in value:
            value = value.get(part)
        else:
            match = next((k for k in value if _norm_key(k) == _norm_key(part)), None)
            if match is None: return None
            value = value.get(match)
    return value


def _select_indices(items: list[Any], question: str, plan: dict | None,
                    max_items: int = 12) -> list[int]:
    """Select transcript rows using only plan-declared ranking/position fields."""
    if len(items) <= max_items: return list(range(len(items)))
    chosen: list[int] = []
    def add(i):
        if 0 <= i < len(items) and i not in chosen: chosen.append(i)

    # Preserve a small API-order head because endpoint_rank is a generic operator.
    for i in range(min(3, len(items))): add(i)
    for d in (plan or {}).get("derivations") or []:
        op = str(d.get("operator") or "").lower()
        if op in {"endpoint_rank", "first", "nth"}:
            try: add(int(d.get("rank") or 0))
            except Exception: pass
        if op in {"argmax", "argmin"} and d.get("field"):
            vals = []
            for i, item in enumerate(items):
                if not isinstance(item, dict): continue
                v = _value_for_field(item, str(d["field"]))
                if v not in (None, ""): vals.append((v, i))
            try: vals.sort(key=lambda x: x[0])
            except TypeError: vals.sort(key=lambda x: str(x[0]))
            for _, i in (vals[-3:] if op == "argmax" else vals[:3]): add(i)
    # Generic late-match discovery: if the user wording overlaps scalar string
    # values in a record, expose the best few rows. This uses no field/entity names.
    q_words = {w for w in re.findall(r"[a-z0-9]+", str(question or "").casefold()) if len(w) > 1}
    if q_words:
        matches = []
        for i, item in enumerate(items):
            if not isinstance(item, dict): continue
            words = set()
            for value in item.values():
                if isinstance(value, str) and len(value) <= 240:
                    words.update(re.findall(r"[a-z0-9]+", value.casefold()))
            overlap = len(q_words & words)
            if overlap: matches.append((overlap, i))
        for _, i in sorted(matches, reverse=True)[:3]: add(i)
    add(len(items) - 1)
    return sorted(chosen[:max_items])


def _list_priority(key: str, items: list[Any], interest: set[str]) -> int:
    """Prioritize sibling arrays from plan field names and observed scalar keys only."""
    norm = _norm_key(key); score = 0
    if norm in interest or set(norm.split("_")) & interest: score += 200
    for item in list(items[:12]) + (list(items[-3:]) if len(items) > 12 else []):
        if not isinstance(item, dict): continue
        keys = {_norm_key(k) for k in item}
        score += 20 * len(keys & interest)
        if score >= 260: break
    return score


def _summarize_payload(payload: Any, question: str, plan: dict | None,
                       budget: int, max_items: int = 12) -> list[str]:
    interest = _plan_interest(plan); lines: list[str] = []
    def remaining(): return budget - sum(len(x) + 1 for x in lines)
    def add(text):
        if remaining() <= 20: return False
        text = str(text)
        if len(text) > remaining(): text = text[: max(0, remaining() - 1)] + "…"
        lines.append(text); return True

    payload = redact_secrets(payload)
    if isinstance(payload, dict):
        root = {k: v for k, v in payload.items() if not isinstance(v, (dict, list))}
        if root: add("  " + _project_record(root, interest, max_fields=12, max_chars=500))
        lists = [(k, v) for k, v in payload.items() if isinstance(v, list)]
        for key, value in lists:
            if remaining() <= 60: break
            add(f"  {key}[{len(value)}]")
        for key, value in [(k, v) for k, v in payload.items() if isinstance(v, dict)]:
            if remaining() <= 80: break
            add(f"  {key}: " + _project_record(value, interest, max_fields=8, max_chars=320))
        ranked = sorted(enumerate(lists), key=lambda p: (-_list_priority(p[1][0], p[1][1], interest), p[0]))
        for _, (key, value) in ranked:
            for i in _select_indices(value, question, plan, max_items=max_items):
                if remaining() <= 80: break
                item = value[i]
                add(f"    {key}[{i}] " + (_project_record(item, interest) if isinstance(item, dict) else _short(item, 180)))
            if remaining() <= 100: break
    elif isinstance(payload, list):
        add(f"  items[{len(payload)}]")
        for i in _select_indices(payload, question, plan, max_items=max_items):
            if remaining() <= 80: break
            item = payload[i]
            add(f"    [{i}] " + (_project_record(item, interest) if isinstance(item, dict) else _short(item, 180)))
    else:
        add(f"  value={_short(payload, min(500, budget))}")
    return lines


def build_receipt(entries: list[dict[str, Any]], question: str = "",
                  plan: dict | None = None, max_chars: int = 3500,
                  call_offset: int = 0) -> str:
    if not entries: return ""
    chunks: list[str] = []; budget_left = max(256, int(max_chars))
    for j, raw in enumerate(entries, start=1):
        entry = redact_secrets(raw)
        method = str(entry.get("method") or "GET").upper(); endpoint = str(entry.get("endpoint") or "?")
        effective = str(entry.get("effective_endpoint") or endpoint); status = entry.get("status_code")
        head = f"call {call_offset + j}: {method} {endpoint} -> HTTP {status}"
        if effective and effective != endpoint: head += f" (effective {effective})"
        if chunks: head = "\n" + head
        if len(head) > budget_left: break
        chunks.append(head); budget_left -= len(head)
        params = entry.get("params") or {}; safe = {k: v for k, v in params.items() if not _is_secret_key(k)}
        if safe and budget_left > 80:
            text = "  params=" + _short(safe, 300); chunks.append(text[:budget_left]); budget_left -= min(len(text), budget_left)
        for line in _summarize_payload(entry.get("payload"), question, plan, max(80, budget_left - 20), 12):
            if budget_left <= 20: break
            text = line[:budget_left]; chunks.append(text); budget_left -= len(text)
        if budget_left <= 80: break
    text = "\n".join(chunks)
    return text if len(text) <= max_chars else text[: max_chars - 28] + "\n… receipt budget reached"


def compact_local_output(result: dict[str, Any], has_api_receipt: bool,
                         max_chars: int = 1200) -> str:
    stderr = str(result.get("stderr") or "").strip()
    raw = str(result.get("auto_display") or result.get("combined") or result.get("stdout") or "").strip()
    if stderr: return ("Execution error/output:\n" + (stderr or raw))[:max_chars]
    if not raw: return ""
    if not has_api_receipt: return raw[:max_chars]
    return raw if len(raw) <= min(max_chars, 700) else ""


def feedback_text(entries: list[dict[str, Any]], result: dict[str, Any],
                  question: str = "", plan: dict | None = None,
                  max_chars: int = 3500, local_max_chars: int = 1200,
                  call_offset: int = 0) -> str:
    receipt = build_receipt(entries, question=question, plan=plan,
                            max_chars=max_chars, call_offset=call_offset)
    local = compact_local_output(result, bool(receipt), max_chars=local_max_chars)
    if receipt and local: return receipt + "\nLocal execution output:\n" + local
    return receipt or local or "(execution produced no visible output)"
