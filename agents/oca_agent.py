"""Observation-Carrying Answers (OCA) v2.

OCA v2 preserves the original two-phase isolation but fixes two empirically
observed weaknesses:

1. incomplete retrieval chains: a one-call OAS-grounded evidence plan is checked
   before Phase A may finish; and
2. wrong-but-grounded answers: deterministic evidence compilation plus a strict
   claim/lineage commit gate validates ranking, filtering, selection, count,
   membership, comparison, and positional decisions.

Verification is feedback, not an excuse to stop solving. Read-only failures trigger
bounded evidence-plan recovery and semantic re-checking. After all grounded recovery
is exhausted, OCA returns the strongest evidence-only candidate while preserving an
explicit uncertified marker in metadata rather than replacing it with a refusal.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agents.common import make_client, log_no_code_turn, finalize_task
from utils.logger import TaskLogger
from utils.observation_ledger import ObservationLedger, EvidenceGraph, redact_secrets

client = make_client()
MAX_PHASE_A_TURNS = getattr(config, "OCA_MAX_PHASE_A_TURNS", 8)
OCA_BUILD_ID = config.OCA_BUILD_ID


_LEDGER_BOOTSTRAP = r'''
import os as _os, requests as _requests_module, json as _json, re as _re, sys as _sys, urllib.parse as _urlparse
# The trusted execution backend installs the registry-configured transport before
# evaluation isolation removes project imports. OCA wraps that transport to
# capture evidence and enforce the validated plan/data-flow contract.
_real_requests = _requests_module
_SECAT_LEDGER = []
# Append-only minimal receipts for successful state-changing calls.  This is
# deliberately separate from rich ledger serialization so a later bookkeeping
# failure can never make a committed side effect look as if it never happened.
_SECAT_WRITE_COMMITS = []
# Requests that reached the transport but raised before a response was observed.
# Their provider-side outcome is unknown, so non-idempotent actions must never be
# semantically replanned/replayed as though nothing happened.
_SECAT_UNCERTAIN_WRITES = []
_SECAT_EXECUTION_RULES = []
_SECAT_EXECUTION_POLICY = "strict"
_SECAT_PLAN_CALL_COUNTS = {}
# Candidate replans may safely reuse earlier GET responses only after the host
# independently audits those call IDs against the candidate plan.  This map is
# populated by trusted host code; generated code cannot access _SECAT_* names.
_SECAT_REPLAY_CALL_IDS_BY_STEP = {}
_SECAT_ADVISORY_CALL_COUNTS = {}
# Safe GET producers refreshed once when a successful response yields no binding
# required by a still-missing downstream step. This closes transient empty-read
# dead ends without reopening writes or changing routes.
_SECAT_BINDING_REFRESHED_STEPS = []
_SECAT_TOTAL_CALLS = 0
# Monotonic response sequence shared by the rich ledger and minimal write journal.
# Transport exceptions do not consume a response id; every received HTTP response does.
_SECAT_RESPONSE_SEQ = 0
_SECAT_IDEMPOTENT_RETRY_COUNTS = {}
# When an idempotent write loses its response, only the exact same semantic
# request may use the reopened plan-step slot.
_SECAT_IDEMPOTENT_RETRY_REQUESTS = {}
_SECAT_MAX_IDEMPOTENT_WRITE_RETRIES = 1
_SECAT_MAX_TOTAL_CALLS = 30
_SECAT_MAX_ADVISORY_CALLS_PER_STEP = 2

def _secat_endpoint(url):
    m = _re.search(r"https?://[^/]+(/[^?]*)", str(url or ""))
    return (m.group(1) if m else str(url or "")).rstrip("/") or "/"

def _secat_origin(url):
    try:
        parts = _urlparse.urlsplit(str(url or ""))
        return (parts.scheme.lower() + "://" + parts.netloc.lower()) if parts.scheme and parts.netloc else ""
    except Exception:
        return ""

def _secat_template_match(actual, template):
    actual = str(actual or "").rstrip("/") or "/"
    template = str(template or "").rstrip("/") or "/"
    parts = _re.split(r"(\{[^{}]+\})", template)
    pattern = "^" + "".join("[^/]+" if p.startswith("{") and p.endswith("}") else _re.escape(p) for p in parts) + "$"
    return bool(_re.match(pattern, actual))

def _secat_path_bindings(actual, template):
    actual = str(actual or "").rstrip("/") or "/"
    template = str(template or "").rstrip("/") or "/"
    names = _re.findall(r"\{([^{}]+)\}", template)
    parts = _re.split(r"(\{[^{}]+\})", template)
    pattern = "^" + "".join("([^/]+)" if p.startswith("{") and p.endswith("}") else _re.escape(p) for p in parts) + "$"
    match = _re.match(pattern, actual)
    if not match:
        return {}
    out = {}
    for name, value in zip(names, match.groups()):
        value = _urlparse.unquote(value or "")
        if _re.fullmatch(r"-?\d+", value or ""):
            try: value = int(value)
            except Exception: pass
        out[name] = value
    return out

def _secat_get(value, path):
    path = str(path or "$").replace("$.", "")
    if path in {"", "$"}:
        return value
    parts = [p for p in path.split(".") if p and p != "$"]

    def walk(current, index):
        if index >= len(parts):
            return current
        part = parts[index]
        expand = part.endswith("[*]")
        key = part[:-3] if expand else part
        if expand:
            if key:
                if not isinstance(current, dict):
                    return None
                current = current.get(key)
            if not isinstance(current, list):
                return None
            values = []
            for item in current:
                child = walk(item, index + 1)
                if child is None:
                    continue
                if isinstance(child, list):
                    values.extend(child)
                else:
                    values.append(child)
            return values
        if not isinstance(current, dict):
            return None
        return walk(current.get(key), index + 1)

    return walk(value, 0)

def _secat_records(payload, record_path):
    path = str(record_path or "$").replace("$.", "")
    if path in {"", "$"}:
        return list(payload) if isinstance(payload, list) else [payload]
    values = [payload]
    for part in [p for p in path.split(".") if p and p != "$"]:
        expand = part.endswith("[*]")
        key = part[:-3] if expand else part
        nxt = []
        for value in values:
            child = value.get(key) if key and isinstance(value, dict) else value if not key else None
            if expand:
                if isinstance(child, list): nxt.extend(child)
            elif child is not None:
                nxt.append(child)
        values = nxt
    return values

def _secat_compare(value, op, expected):
    if isinstance(value, list):
        if op == "exists": return bool(value)
        if op == "not_exists": return not value
        if op == "neq": return all(_secat_compare(v, op, expected) for v in value)
        return any(_secat_compare(v, op, expected) for v in value)
    if op == "exists": return value is not None
    if op == "not_exists": return value is None
    if op in {"eq_ci", "neq_ci", "contains_ci", "startswith_ci"}:
        left = "" if value is None else str(value).casefold()
        right = "" if expected is None else str(expected).casefold()
        if op == "eq_ci": return left == right
        if op == "neq_ci": return left != right
        if op == "contains_ci": return right in left
        return left.startswith(right)
    if op == "contains":
        try: return expected in value
        except Exception: return str(expected) in str(value)
    if op == "in":
        try: return value in expected
        except Exception: return False
    if op == "eq": return value == expected
    if op == "neq": return value != expected
    try:
        if op == "gt": return value > expected
        if op == "gte": return value >= expected
        if op == "lt": return value < expected
        if op == "lte": return value <= expected
    except Exception:
        return False
    return False

def _secat_project_records(payloads, spec):
    rows = []
    for payload in payloads:
        rows.extend(_secat_records(payload, (spec or {}).get("record_path") or "$"))
    for filt in (spec or {}).get("filters") or []:
        rows = [row for row in rows if _secat_compare(
            _secat_get(row, filt.get("path")), str(filt.get("op") or "eq"), filt.get("value"))]
    for sorter in reversed((spec or {}).get("sort") or []):
        path = sorter.get("path")
        present = [row for row in rows if _secat_get(row, path) is not None]
        missing = [row for row in rows if _secat_get(row, path) is None]
        domains = set()
        for row in present:
            value = _secat_get(row, path)
            if isinstance(value, bool): domains.add("bool")
            elif isinstance(value, (int, float)): domains.add("number")
            elif isinstance(value, str): domains.add("string")
            else: domains.add("other")
        if len(domains) > 1 or "other" in domains:
            return [], []
        try:
            present.sort(key=lambda row: _secat_get(row, path),
                         reverse=str(sorter.get("direction") or "asc") == "desc")
        except Exception:
            return [], []
        rows = present + missing
    select = (spec or {}).get("select") or {}
    mode = str(select.get("mode") or "head")
    limit = max(1, int(select.get("limit", 5) or 5))
    index = max(0, int(select.get("index", 0) or 0))
    if mode in {"head", "top", "all_matches"}: selected = rows[:limit]
    elif mode == "tail": selected = rows[-limit:]
    elif mode == "nth": selected = [rows[index]] if index < len(rows) else []
    elif mode == "single": selected = rows[:1]
    else: selected = rows[:limit]
    return rows, selected

def _secat_payloads_for_step(step_id):
    """Return trusted payloads for a current plan step, including audited replay.

    Replanning namespaces step IDs, so an already-fetched GET from the original
    attempt cannot satisfy a recovered downstream binding merely by runtime tag.
    The host may whitelist exact call IDs only after auditing them against the
    recovered plan's method/path/request contract.  This preserves provenance
    while avoiding wasteful identical refetches during accuracy recovery.
    """
    sid = str(step_id or "")
    replay_ids = set(str(x) for x in (_SECAT_REPLAY_CALL_IDS_BY_STEP.get(sid) or []))
    return [entry.get("payload") for entry in _SECAT_LEDGER
            if str(entry.get("runtime_step_id") or "") == sid
            or str(entry.get("call_id") or "") in replay_ids]


def _secat_authorized_values(producer_step_id, binding):
    if not producer_step_id or not binding or not binding.get("path"):
        return set()
    producer = next((r for r in _SECAT_EXECUTION_RULES
                     if str(r.get("step_id")) == str(producer_step_id)), None)
    if not producer:
        return set()
    payloads = _secat_payloads_for_step(producer_step_id)
    if not payloads:
        return set()
    filtered, selected = _secat_project_records(payloads, producer.get("observation_spec") or {})
    source = str(binding.get("source") or "selected_first")
    records = filtered if source == "selected_all" else selected[:1]
    if source == "selected_all" and binding.get("max_values"):
        try: records = records[:max(1, int(binding.get("max_values")))]
        except Exception: pass
    out = set()
    for row in records:
        value = _secat_get(row, binding.get("path"))
        if value not in (None, ""):
            try: out.add(str(value))
            except Exception: pass
    return out

def _secat_query_keys(url, kw):
    keys = set()
    try:
        keys.update(_urlparse.parse_qs(_urlparse.urlsplit(str(url or "")).query, keep_blank_values=True).keys())
    except Exception:
        pass
    params = (kw or {}).get("params")
    if isinstance(params, dict):
        keys.update(str(k) for k in params.keys())
    elif isinstance(params, (list, tuple)):
        for item in params:
            if isinstance(item, (list, tuple)) and item:
                keys.add(str(item[0]))
    return keys

def _secat_query_map(url, kw):
    out = {}
    try:
        parsed = _urlparse.parse_qs(_urlparse.urlsplit(str(url or "")).query, keep_blank_values=True)
        for k, vals in parsed.items():
            out[str(k)] = vals if len(vals) != 1 else vals[0]
    except Exception:
        pass
    params = (kw or {}).get("params")
    if isinstance(params, dict):
        for key, value in params.items():
            k = str(key)
            if k in out:
                old = out[k] if isinstance(out[k], list) else [out[k]]
                out[k] = old + (value if isinstance(value, list) else [value])
            else:
                out[k] = value
    elif isinstance(params, (list, tuple)):
        for item in params:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                k, v = str(item[0]), item[1]
                if k in out:
                    old = out[k] if isinstance(out[k], list) else [out[k]]
                    out[k] = old + ([v] if not isinstance(v, list) else v)
                else:
                    out[k] = v
    return out

def _secat_body_map(kw):
    body = (kw or {}).get("json")
    if body is None:
        body = (kw or {}).get("data")
    if isinstance(body, str):
        try: body = _json.loads(body)
        except Exception: pass
    return body

def _secat_request_value(body, path):
    if not isinstance(body, (dict, list)):
        return None, False
    cur = body
    for part in str(path or "").replace("$.", "").split("."):
        if not part:
            continue
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None, False
    return cur, True

def _secat_body_leaf_paths(value, prefix=""):
    """Return concrete request-body leaf/container paths for contract checking."""
    if isinstance(value, dict):
        out = set()
        if not value and prefix:
            out.add(prefix)
        for key, item in value.items():
            path = (prefix + "." + str(key)) if prefix else str(key)
            out.update(_secat_body_leaf_paths(item, path))
        return out
    # Treat arrays as an atomic request value. A plan that owns the array path
    # owns its members as part of the exact literal/binding value.
    if isinstance(value, (list, tuple)):
        return {prefix} if prefix else {"$"}
    return {prefix} if prefix else ({"$"} if value is not None else set())

def _secat_body_path_covered(actual_path, declared_paths):
    for declared in declared_paths:
        declared = str(declared)
        if declared == "$":
            return True
        if actual_path == declared or actual_path.startswith(declared + "."):
            return True
    return False

def _secat_scalar_equal(actual, expected):
    """Compare semantic request values after harmless wire-shape normalization.

    A provider may accept a scalar and a singleton repeated/list parameter as the
    same request value.  Treat only singleton scalar/list forms as equivalent;
    multi-value order and cardinality remain strict.
    """
    if isinstance(expected, bool):
        if isinstance(actual, str):
            return actual.strip().lower() in ({"true", "1"} if expected else {"false", "0"})
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try: return float(actual) == float(expected)
        except Exception: return False
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(
            _secat_scalar_equal(actual[k], expected[k]) for k in expected)
    if isinstance(expected, (list, tuple)):
        seq = _secat_as_sequence(actual)
        if seq is None:
            if len(expected) == 1:
                return _secat_scalar_equal(actual, expected[0])
            return False
        return len(seq) == len(expected) and all(
            _secat_scalar_equal(a, b) for a, b in zip(seq, expected))
    if isinstance(actual, (list, tuple)):
        return len(actual) == 1 and _secat_scalar_equal(actual[0], expected)
    return str(actual) == str(expected)

def _secat_as_sequence(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str) and "," in value:
        return [x for x in value.split(",") if x != ""]
    return None

def _secat_authorized_sequence(producer_step_id, binding):
    if not producer_step_id or not binding or not binding.get("path"):
        return []
    producer = next((r for r in _SECAT_EXECUTION_RULES
                     if str(r.get("step_id")) == str(producer_step_id)), None)
    if not producer:
        return []
    payloads = _secat_payloads_for_step(producer_step_id)
    if not payloads:
        return []
    filtered, selected = _secat_project_records(payloads, producer.get("observation_spec") or {})
    source = str(binding.get("source") or "selected_first")
    records = filtered if source == "selected_all" else selected[:1]
    if source == "selected_all" and binding.get("max_values"):
        try: records = records[:max(1, int(binding.get("max_values")))]
        except Exception: pass
    out = []
    for row in records:
        value = _secat_get(row, binding.get("path"))
        values = value if isinstance(value, list) else [value]
        for item in values:
            if item not in (None, ""):
                out.append(str(item))
    return out

def _secat_correlated_binding_match(rule, concrete, query_map, body):
    """Keep multiple bindings from one producer on the same selected record.

    Independent set-membership checks can authorize an impossible cross-pair,
    e.g. producer rows (a=1,b=X),(a=2,b=Y) followed by /{a}/{b}=1/Y.  For scalar
    request values sourced from the same producer, require one projected producer
    record to satisfy the whole tuple.  List-valued selected_all requests are
    already checked as ordered sequences and are excluded here.
    """
    grouped = {}
    for name, info in (rule.get("placeholders") or {}).items():
        if str(info.get("source") or "") != "binding":
            continue
        actual = concrete.get(name)
        if _secat_as_sequence(actual) is not None:
            continue
        grouped.setdefault(str(info.get("producer_step_id") or ""), []).append(
            (info.get("binding") or {}, actual))
    for name, info in (rule.get("query_contract") or {}).items():
        if str(info.get("source") or "") != "binding" or name not in query_map:
            continue
        actual = query_map.get(name)
        if _secat_as_sequence(actual) is not None:
            continue
        grouped.setdefault(str(info.get("producer_step_id") or ""), []).append(
            (info.get("binding") or {}, actual))
    for name, info in (rule.get("body_contract") or {}).items():
        if str(info.get("source") or "") != "binding":
            continue
        actual, present = _secat_request_value(body, name)
        if not present or _secat_as_sequence(actual) is not None:
            continue
        grouped.setdefault(str(info.get("producer_step_id") or ""), []).append(
            (info.get("binding") or {}, actual))
    for producer_step_id, items in grouped.items():
        if not producer_step_id or len(items) < 2:
            continue
        producer = next((r for r in _SECAT_EXECUTION_RULES
                         if str(r.get("step_id")) == producer_step_id), None)
        if not producer:
            return False
        payloads = _secat_payloads_for_step(producer_step_id)
        if not payloads:
            return False
        filtered, selected = _secat_project_records(payloads, producer.get("observation_spec") or {})
        records = filtered
        if any(str(binding.get("source") or "selected_first") != "selected_all"
               for binding, _ in items):
            records = selected[:1]
        matched = False
        for row in records:
            if all(_secat_scalar_equal(actual, _secat_get(row, binding.get("path")))
                   for binding, actual in items):
                matched = True
                break
        if not matched:
            return False
    return True

def _secat_request_arg_matches(info, actual, present):
    if not present:
        return False
    if str(info.get("source") or "") == "literal":
        return _secat_scalar_equal(actual, info.get("literal_value"))
    binding = info.get("binding") or {}
    allowed = _secat_authorized_sequence(info.get("producer_step_id"), binding)
    if not allowed:
        return False
    wrapper = str(info.get("wrapper") or "")
    if wrapper == "uri_objects":
        if not isinstance(actual, (list, tuple)):
            return False
        seq = []
        for item in actual:
            if not isinstance(item, dict) or set(item) != {"uri"}:
                return False
            value = item.get("uri")
            if value in (None, ""):
                return False
            seq.append(value)
    else:
        seq = _secat_as_sequence(actual)
    if seq is not None:
        # Preserve the original selected_all semantics: list-valued requests must
        # match the complete authorized producer sequence in order.
        return len(seq) == len(allowed) and all(str(a) == str(b) for a, b in zip(seq, allowed))
    # Preserve the original scalar binding semantics: a later action may select
    # one authorized member from a multi-record producer result. Singleton-list
    # normalization applies to literal wire shapes, not to narrowing this set.
    return str(actual) in set(allowed)

def _secat_step_completed(step_id):
    """Return whether trusted receipts prove the whole plan step completed.

    For ordinary steps, one successful response/commit is sufficient. For a
    ``selected_all`` fan-out step, however, a single successful target must not
    unlock later side effects. In that case the successful receipts must cover
    the complete producer-authorized sequence. One batched request covering the
    whole sequence and several scalar fan-out requests are both accepted.
    """
    sid = str(step_id or "")
    if not sid:
        return False
    rule = next((r for r in _SECAT_EXECUTION_RULES
                 if str(r.get("step_id") or "") == sid), None)
    replay_ids = set(str(x) for x in (_SECAT_REPLAY_CALL_IDS_BY_STEP.get(sid) or []))
    successes = []
    for receipt in _SECAT_WRITE_COMMITS:
        if str(receipt.get("runtime_step_id") or "") != sid:
            continue
        try:
            if 200 <= int(receipt.get("status_code")) < 300:
                successes.append(receipt)
        except Exception:
            continue
    for entry in _SECAT_LEDGER:
        direct = str(entry.get("runtime_step_id") or "") == sid
        replay = str(entry.get("call_id") or "") in replay_ids
        if not (direct or replay):
            continue
        try:
            if 200 <= int(entry.get("status_code")) < 300:
                successes.append(entry)
        except Exception:
            continue
    if not successes:
        return False
    completion_bindings = list((rule or {}).get("completion_bindings") or [])
    if not completion_bindings:
        return True

    def _flatten_actual(value, wrapper):
        if wrapper == "uri_objects":
            if not isinstance(value, (list, tuple)):
                return []
            return [x.get("uri") for x in value
                    if isinstance(x, dict) and x.get("uri") not in (None, "")]
        seq = _secat_as_sequence(value)
        return list(seq) if seq is not None else ([] if value in (None, "") else [value])

    for info in completion_bindings:
        allowed = _secat_authorized_sequence(info.get("producer_step_id"), info.get("binding") or {})
        if not allowed:
            return False
        covered = []
        for receipt in successes:
            loc = str(info.get("location") or "")
            name = str(info.get("name") or "")
            actual = None
            present = False
            if loc == "path":
                concrete = _secat_path_bindings(
                    str(receipt.get("endpoint") or ""),
                    str((rule or {}).get("wire_endpoint") or (rule or {}).get("endpoint") or ""),
                )
                if name in concrete:
                    actual, present = concrete.get(name), True
            elif loc == "query":
                params = receipt.get("params") or {}
                if isinstance(params, dict) and name in params:
                    actual, present = params.get(name), True
            elif loc == "body":
                actual, present = _secat_request_value(receipt.get("request_body"), name)
            if present:
                covered.extend(_flatten_actual(actual, str(info.get("wrapper") or "")))
        allowed_s = [str(x) for x in allowed]
        covered_s = [str(x) for x in covered]
        if any(value not in covered_s for value in allowed_s):
            return False
    return True


def _secat_successful_get_replay(rule, url, kw):
    """Return an exact successful prior GET for this already-authorized plan step.

    This is local replay, not a new API action.  It is used only after the plan
    step's external-call budget is exhausted, allowing repair code to reconstruct
    local variables without repeating a network read or consuming task quota.
    """
    if str(rule.get("method") or "GET").upper() != "GET":
        return None
    sid = str(rule.get("step_id") or "")
    if not sid:
        return None
    query = _secat_query_map(url, kw or {})
    body = _secat_body_map(kw or {})
    replay_ids = set(str(x) for x in (_SECAT_REPLAY_CALL_IDS_BY_STEP.get(sid) or []))
    for entry in reversed(_SECAT_LEDGER):
        if (str(entry.get("runtime_step_id") or "") != sid
                and str(entry.get("call_id") or "") not in replay_ids):
            continue
        if str(entry.get("method") or "").upper() != "GET":
            continue
        status = entry.get("status_code")
        try:
            if not (200 <= int(status) < 300):
                continue
        except Exception:
            continue
        if not _secat_scalar_equal(entry.get("params") or {}, query or {}):
            continue
        if not _secat_scalar_equal(entry.get("request_body") or {}, body or {}):
            continue
        # Reconstruct only responses whose model-visible payload was actually
        # retained.  A 2xx GET with an unrecorded/non-JSON body must fall back to
        # the normal budget error rather than fabricate an empty response.
        if entry.get("payload") is None:
            continue
        return entry
    return None

class _SecatReplayResponse:
    """Minimal requests.Response-compatible view over a trusted ledger receipt."""
    def __init__(self, entry, url=""):
        self.status_code = int(entry.get("status_code") or 200)
        self._payload = entry.get("payload")
        self.url = str(url or "")
        self.headers = {}
        self.request = None
        self.reason = "REPLAY"
        self.encoding = "utf-8"
        self.ok = 200 <= self.status_code < 400
        if self._payload is None:
            self.text = ""
        elif isinstance(self._payload, str):
            self.text = self._payload
        else:
            try: self.text = _json.dumps(self._payload, ensure_ascii=False)
            except Exception: self.text = str(self._payload)
        self.content = self.text.encode("utf-8")
    def json(self):
        try: return _json.loads(_json.dumps(self._payload))
        except Exception: return self._payload
    def raise_for_status(self):
        if not (200 <= self.status_code < 400):
            raise RuntimeError("OCA_REPLAY_HTTP_STATUS: " + str(self.status_code))
        return None

def _secat_rule_for_call(method, endpoint, url=None, kw=None):
    m = str(method or "GET").upper()
    actual_origin = _secat_origin(url)
    allowed_origins = {str(r.get("allowed_origin") or "").lower() for r in _SECAT_EXECUTION_RULES if r.get("allowed_origin")}
    if allowed_origins and (not actual_origin or actual_origin.lower() not in allowed_origins):
        raise RuntimeError("OCA_ISOLATION_VIOLATION: unauthorized API origin " + str(actual_origin or url))
    contract_missing = []
    dependency_blocked = []
    advisory_matches = []
    candidates = [r for r in _SECAT_EXECUTION_RULES
                  if m == str(r.get("method") or "GET").upper() and
                  _secat_template_match(endpoint, r.get("wire_endpoint") or r.get("endpoint"))]
    authorized = []
    for rule in candidates:
        # The plan's declared dependencies are a runtime invariant even when a
        # child request has no dynamic binding.  For state-changing calls we also
        # preserve the validated plan's write order so a later side effect cannot
        # change provider state before an earlier requested side effect runs.
        required_prior = list(rule.get("depends_on") or [])
        if m in {"POST", "PUT", "DELETE", "PATCH"}:
            required_prior += list(rule.get("prior_write_step_ids") or [])
        unmet = [str(x) for x in dict.fromkeys(required_prior)
                 if not _secat_step_completed(str(x))]
        if unmet:
            dependency_blocked.extend(unmet)
            continue
        concrete = _secat_path_bindings(endpoint, rule.get("wire_endpoint") or rule.get("endpoint"))
        valid = True
        for name, info in (rule.get("placeholders") or {}).items():
            if str(info.get("source") or "") == "literal":
                literal = info.get("literal_value")
                allowed = {str(literal)} if literal not in (None, "") else set()
            else:
                allowed = _secat_authorized_values(info.get("producer_step_id"), info.get("binding") or {})
            if not allowed or str(concrete.get(name)) not in allowed:
                valid = False
                break
        if not valid:
            continue
        # Path/origin lineage is always a hard boundary. Request arguments on a
        # safe GET are different: in advisory mode the model may inspect a
        # semantically plausible variant, but it will be recorded as *not*
        # satisfying the plan unless the full request contract matches.
        request_diagnostics = []
        required_query = {str(x) for x in (rule.get("required_query_params") or [])}
        query_map = _secat_query_map(url, kw or {})
        if required_query and not required_query.issubset(set(query_map)):
            missing = sorted(required_query - set(query_map))
            contract_missing.extend(missing)
            request_diagnostics.append("missing query: " + ",".join(missing))
            valid = False
        declared_query = set(str(x) for x in (rule.get("query_contract") or {}))
        # Query arguments can materially change the returned/actioned resource.
        # Authentication is injected after this guard, so every query value visible
        # here must be part of the validated plan rather than an undeclared model
        # choice such as a different page, market, search term, or filter.
        if valid and (set(query_map) - declared_query):
            request_diagnostics.append("undeclared query: " + ",".join(sorted(set(query_map) - declared_query)))
            valid = False
        for name, info in (rule.get("query_contract") or {}).items():
            if not _secat_request_arg_matches(info, query_map.get(name), name in query_map):
                request_diagnostics.append("query mismatch: " + str(name))
                valid = False
        body = _secat_body_map(kw or {})
        declared_body = set(str(x) for x in (rule.get("body_contract") or {}))
        actual_body_paths = _secat_body_leaf_paths(body) if body is not None else set()
        if valid and any(not _secat_body_path_covered(path, declared_body)
                         for path in actual_body_paths):
            request_diagnostics.append("undeclared body field")
            valid = False
        for name, info in (rule.get("body_contract") or {}).items():
            value, present = _secat_request_value(body, name)
            if not _secat_request_arg_matches(info, value, present):
                request_diagnostics.append("body mismatch: " + str(name))
                valid = False
        for name in (rule.get("required_body_fields") or []):
            _, present = _secat_request_value(body, name)
            if not present:
                contract_missing.append("body:" + str(name))
                request_diagnostics.append("missing body: " + str(name))
                valid = False
        if not _secat_correlated_binding_match(rule, concrete, query_map, body):
            request_diagnostics.append("binding tuple mismatch")
            valid = False
        if not valid:
            if (m == "GET" and str(_SECAT_EXECUTION_POLICY).lower() == "advisory" and
                    request_diagnostics):
                soft = dict(rule)
                soft["_request_contract_advisory"] = True
                soft["_request_contract_diagnostic"] = "; ".join(dict.fromkeys(request_diagnostics))
                advisory_matches.append(soft)
            continue
        authorized.append(rule)
        key = str(rule.get("step_id") or (m + " " + str(rule.get("endpoint"))))
        if m == "GET":
            _replay = _secat_successful_get_replay(rule, url, kw or {})
            if _replay is not None and (
                    str(_replay.get("runtime_step_id") or "") != str(rule.get("step_id") or "")
                    or str(_replay.get("call_id") or "") in set(
                        str(x) for x in (_SECAT_REPLAY_CALL_IDS_BY_STEP.get(str(rule.get("step_id") or "")) or []))):
                _out = dict(rule)
                _out["_request_replay_entry"] = _replay
                return _out
        if int(_SECAT_PLAN_CALL_COUNTS.get(key, 0)) < max(1, int(rule.get("max_calls", 1) or 1)):
            return rule
    if authorized:
        # Repair code often regenerates a whole dataflow after an earlier read
        # succeeded but a dependent action was blocked. Replaying that exact GET
        # from the trusted ledger is safe, deterministic, and avoids both network
        # duplication and a false plan-call-budget dead end. Writes are never replayed.
        if m == "GET":
            for _rule in authorized:
                _entry = _secat_successful_get_replay(_rule, url, kw or {})
                if _entry is not None:
                    _out = dict(_rule)
                    _out["_request_replay_entry"] = _entry
                    return _out
        keys = [str(r.get("step_id") or (m + " " + str(r.get("endpoint")))) for r in authorized]
        raise RuntimeError("OCA_PLAN_CALL_BUDGET_EXCEEDED: " + ",".join(keys))
    if advisory_matches:
        # This is deliberately *not* a plan-completing authorization. It merely
        # allows a same-origin, planned-path GET to return evidence/context so the
        # model can correct itself. The host audit later enforces the exact plan.
        return advisory_matches[0]
    if candidates and dependency_blocked:
        raise RuntimeError("OCA_DEPENDENCY_NOT_READY: " + ",".join(sorted(set(dependency_blocked))))
    if candidates and contract_missing:
        raise RuntimeError("OCA_REQUEST_CONTRACT: missing required request field(s) " + ",".join(sorted(set(contract_missing))))
    if candidates:
        raise RuntimeError("OCA_UNAUTHORIZED_REQUEST_INSTANCE: " + m + " " + str(endpoint))
    raise RuntimeError("OCA_UNPLANNED_API_CALL: " + m + " " + str(endpoint))

class _LedgerRequests:
    exceptions = _real_requests.exceptions
    RequestException = _real_requests.RequestException
    def _call(self, method, url, **kw):
        global _SECAT_TOTAL_CALLS, _SECAT_RESPONSE_SEQ
        endpoint = _secat_endpoint(url)
        rule = _secat_rule_for_call(method, endpoint, url=url, kw=kw)
        advisory = bool(rule.get("_request_contract_advisory"))
        replay_entry = rule.get("_request_replay_entry")
        if replay_entry is not None:
            return _SecatReplayResponse(replay_entry, url=url)
        key = str(rule.get("step_id") or (str(method).upper() + " " + str(rule.get("endpoint"))))
        retry_expected = _SECAT_IDEMPOTENT_RETRY_REQUESTS.get(key)
        if retry_expected is not None:
            retry_actual = {
                "method": str(method).upper(),
                "endpoint": endpoint,
                "params": _secat_query_map(url, kw),
                "request_body": _secat_body_map(kw),
            }
            if not _secat_scalar_equal(retry_actual, retry_expected):
                raise RuntimeError("OCA_IDEMPOTENT_RETRY_MISMATCH: " + key)
        used = int(_SECAT_PLAN_CALL_COUNTS.get(key, 0))
        allowed = max(1, int(rule.get("max_calls", 1) or 1))
        if not advisory and used >= allowed:
            raise RuntimeError("OCA_PLAN_CALL_BUDGET_EXCEEDED: " + key + " allowed=" + str(allowed))
        advisory_used = int(_SECAT_ADVISORY_CALL_COUNTS.get(key, 0)) if advisory else 0
        if advisory:
            if advisory_used >= int(_SECAT_MAX_ADVISORY_CALLS_PER_STEP):
                raise RuntimeError("OCA_PLAN_CALL_BUDGET_EXCEEDED: advisory:" + key)
            _SECAT_ADVISORY_CALL_COUNTS[key] = advisory_used + 1
        if _SECAT_TOTAL_CALLS >= int(_SECAT_MAX_TOTAL_CALLS):
            raise RuntimeError("OCA_TASK_CALL_BUDGET_EXCEEDED: allowed=" + str(_SECAT_MAX_TOTAL_CALLS))
        if not advisory:
            _SECAT_PLAN_CALL_COUNTS[key] = used + 1
        _SECAT_TOTAL_CALLS += 1
        fn = getattr(_real_requests, str(method).lower())
        # Do not allow requests to leave the validated origin/path through an
        # implicit HTTP redirect after the guard has approved the initial URL.
        # Redirect responses remain observable as non-2xx/3xx execution evidence
        # and cannot satisfy plan completion.
        kw = dict(kw)
        kw["allow_redirects"] = False
        try:
            resp = fn(url, **kw)
        except Exception as _exc:
            # A transport exception is ambiguous only when the client did not
            # receive a trustworthy provider response (connection/timeout family).
            # Provider-declared HTTP/quota/auth errors are *known rejections* and
            # must not be journaled as possibly committed writes, otherwise a
            # harmless rejected request could unnecessarily lock strategy recovery.
            method_upper = str(method).upper()
            _request_exception_type = getattr(_real_requests, "RequestException", None)
            _transport_ambiguous = bool(
                (isinstance(_request_exception_type, type) and isinstance(_exc, _request_exception_type))
                or type(_exc).__name__ in {"ConnectionError", "Timeout", "ConnectTimeout", "ReadTimeout"}
            )
            if method_upper in {"POST", "PUT", "DELETE", "PATCH"} and _transport_ambiguous:
                try:
                    _SECAT_UNCERTAIN_WRITES.append({
                        "method": method_upper,
                        "endpoint": endpoint,
                        "runtime_step_id": "" if advisory else str(rule.get("step_id") or ""),
                        "request_origin": _secat_origin(url),
                        "params": _secat_query_map(url, kw),
                        "request_body": _secat_body_map(kw),
                        "exception_type": type(_exc).__name__,
                    })
                except Exception:
                    pass
            if str(method).upper() in {"GET", "HEAD"}:
                if advisory:
                    _SECAT_ADVISORY_CALL_COUNTS[key] = advisory_used
                else:
                    _SECAT_PLAN_CALL_COUNTS[key] = used
            # PUT and DELETE are idempotent HTTP methods. Permit at most one
            # transport-level retry of the exact same validated request. POST/PATCH
            # remain closed because replay could duplicate a non-idempotent action.
            elif (not advisory and method_upper in {"PUT", "DELETE"}
                  and _transport_ambiguous
                  and int(_SECAT_IDEMPOTENT_RETRY_COUNTS.get(key, 0))
                      < int(_SECAT_MAX_IDEMPOTENT_WRITE_RETRIES)):
                _SECAT_IDEMPOTENT_RETRY_COUNTS[key] = int(
                    _SECAT_IDEMPOTENT_RETRY_COUNTS.get(key, 0)) + 1
                _SECAT_IDEMPOTENT_RETRY_REQUESTS[key] = {
                    "method": method_upper,
                    "endpoint": endpoint,
                    "params": _secat_query_map(url, kw),
                    "request_body": _secat_body_map(kw),
                }
                _SECAT_PLAN_CALL_COUNTS[key] = used
            # Transport exceptions can embed the fully prepared URL/headers, which
            # may contain registry-injected credentials. Do not expose that raw
            # exception to generated code or model-visible logs. The exception type
            # is enough for recovery/classification; semantic request details are
            # already stored separately by the trusted plan guard.
            raise RuntimeError("OCA_HTTP_TRANSPORT_FAILURE: " + type(_exc).__name__) from None
        _SECAT_IDEMPOTENT_RETRY_REQUESTS.pop(key, None)
        _SECAT_RESPONSE_SEQ += 1
        _call_id = "call_%03d" % int(_SECAT_RESPONSE_SEQ)
        # Record a successful side effect immediately, before optional response
        # sanitization/JSON parsing/rich ledger bookkeeping.  This journal is the
        # no-replay safety boundary if anything later in the turn fails.
        try:
            _status = int(getattr(resp, "status_code", 0) or 0)
            if str(method).upper() in {"POST", "PUT", "DELETE", "PATCH"} and 200 <= _status < 300:
                _SECAT_WRITE_COMMITS.append({
                    "call_id": _call_id,
                    "method": str(method).upper(),
                    "endpoint": endpoint,
                    "runtime_step_id": "" if advisory else str(rule.get("step_id") or ""),
                    "status_code": _status,
                    "request_origin": _secat_origin(url),
                    # These are the credential-free semantic values already
                    # validated by the request guard. They make the minimal receipt
                    # sufficient for post-write certification if rich bookkeeping
                    # fails after the provider has committed the side effect.
                    "params": _secat_query_map(url, kw),
                    "request_body": _secat_body_map(kw),
                    "runtime_contract_authorized": bool(not advisory),
                })
        except Exception:
            # The guard validator protects every dependency used above.  If a
            # future runtime change violates that invariant, do not fabricate a
            # commit receipt; the rich ledger path below may still capture it.
            pass
        # The trusted transport may inject credentials after the plan guard. A raw
        # requests.Response exposes those credentials again through request metadata
        # (Authorization headers or auth query parameters). Sanitize that metadata
        # before returning the response to generated code; JSON/body/status remain
        # untouched and the ledger already captured the credential-free semantic
        # request surface.
        try:
            _safe_url = str(url or "")
            try:
                _parts = _urlparse.urlsplit(_safe_url)
                _semantic_query = _secat_query_map(url, kw)
                _safe_url = _urlparse.urlunsplit((_parts.scheme, _parts.netloc, _parts.path,
                                                  _urlparse.urlencode(_semantic_query, doseq=True),
                                                  _parts.fragment))
            except Exception:
                pass
            if getattr(resp, "request", None) is not None:
                try:
                    _headers = getattr(resp.request, "headers", None)
                    if hasattr(_headers, "keys"):
                        for _name in list(_headers.keys()):
                            _low = str(_name).casefold()
                            if (_low in {"authorization", "proxy-authorization", "x-api-key", "api-key"}
                                    or "token" in _low or "secret" in _low):
                                try: del _headers[_name]
                                except Exception: pass
                    resp.request.url = _safe_url
                except Exception:
                    pass
            resp.url = _safe_url
        except Exception:
            pass
        try:
            try: payload = resp.json()
            except Exception: payload = None
            adapt = getattr(resp, "secat_api_adaptation", None)
            effective = endpoint
            if isinstance(adapt, dict) and adapt.get("effective_url"):
                effective = _secat_endpoint(adapt["effective_url"])
            _SECAT_LEDGER.append({
                "call_id": _call_id,
                "method": str(method).upper(), "endpoint": endpoint,
                "effective_endpoint": effective,
                "request_origin": _secat_origin(url),
                # Persist the semantic request surface seen by the guard, including
                # query values embedded directly in the URL.  Authentication is
                # injected later by the trusted transport and therefore never
                # appears here or in the host evidence graph.
                "params": _secat_query_map(url, kw),
                "request_body": _secat_body_map(kw),
                "payload": payload,
                "status_code": getattr(resp, "status_code", None),
                "adaptation": adapt,
                "runtime_step_id": "" if advisory else str(rule.get("step_id") or ""),
                # Runtime authorization is a first-class certificate.  The live
                # guard validated the concrete origin/path/query/body against the
                # exact plan rule using the full trusted producer payload.  Keep
                # that fact so post-run certification does not have to reconstruct
                # the same binding from a deliberately compact observation view.
                "runtime_contract_authorized": bool(not advisory),
                "runtime_contract_step_id": "" if advisory else str(rule.get("step_id") or ""),
                "runtime_contract_version": 1 if not advisory else 0,
                "request_contract_advisory": advisory,
                "request_contract_diagnostic": str(rule.get("_request_contract_diagnostic") or ""),
                "advisory_candidate_step_id": str(rule.get("step_id") or "") if advisory else "",
            })
        except Exception:
            # If a successful write reached the provider but rich response
            # bookkeeping failed, retain a minimal plan-certified ledger receipt.
            # This prevents both false action gaps and unsafe side-effect replay.
            try:
                _status2 = int(getattr(resp, "status_code", 0) or 0)
                if str(method).upper() in {"POST", "PUT", "DELETE", "PATCH"} and 200 <= _status2 < 300:
                    if not any(str(e.get("call_id") or "") == _call_id for e in _SECAT_LEDGER):
                        _SECAT_LEDGER.append({
                            "call_id": _call_id,
                            "method": str(method).upper(),
                            "endpoint": endpoint,
                            "effective_endpoint": endpoint,
                            "request_origin": _secat_origin(url),
                            "params": _secat_query_map(url, kw),
                            "request_body": _secat_body_map(kw),
                            "payload": None,
                            "status_code": _status2,
                            "adaptation": None,
                            "runtime_step_id": "" if advisory else str(rule.get("step_id") or ""),
                            "runtime_contract_authorized": bool(not advisory),
                            "runtime_contract_step_id": "" if advisory else str(rule.get("step_id") or ""),
                            "runtime_contract_version": 1 if not advisory else 0,
                            "request_contract_advisory": advisory,
                            "request_contract_diagnostic": "minimal_post_write_receipt",
                            "advisory_candidate_step_id": str(rule.get("step_id") or "") if advisory else "",
                        })
            except Exception:
                pass
        return resp
    def get(self, url, **kw): return self._call("GET", url, **kw)
    def post(self, url, **kw): return self._call("POST", url, **kw)
    def put(self, url, **kw): return self._call("PUT", url, **kw)
    def delete(self, url, **kw): return self._call("DELETE", url, **kw)
    def patch(self, url, **kw): return self._call("PATCH", url, **kw)
    def request(self, method, url, **kw): return self._call(str(method).upper(), url, **kw)
    def __getattr__(self, n):
        # Do not expose the underlying requests module (sessions/adapters/api)
        # because that would bypass the trusted endpoint/binding guard. Only
        # non-transport compatibility attributes are delegated.
        if n in {"exceptions", "RequestException", "codes", "Response"}:
            return getattr(_real_requests, n)
        if n in {"__spec__", "__loader__", "__package__", "__name__", "__file__", "__path__"}:
            return getattr(_real_requests, n, None)
        raise RuntimeError("OCA_UNSUPPORTED_TRANSPORT: use requests.get/post/put/delete/patch/request")
_secat_req = _LedgerRequests()
requests = _secat_req
_sys.modules["requests"] = _secat_req

_SECAT_DERIVED = []
class _LedgerHelper:
    def derive_count(self, name, records, predicate=None):
        matched = [r for r in (records or []) if (predicate(r) if predicate else True)]
        _SECAT_DERIVED.append({"name": name, "value": len(matched), "operation": "count",
                               "capture": "explicit_helper"})
        return len(matched)
    def derive_selection(self, name, records, sort_key, descending=True):
        recs = [r for r in (records or []) if isinstance(r, dict) and
                r.get(sort_key) not in (None, "")]
        if not recs:
            _SECAT_DERIVED.append({"name": name, "value": None, "operation": "selection",
                                   "support_values": [], "capture": "explicit_helper"})
            return None
        chosen = sorted(recs, key=lambda r: r.get(sort_key), reverse=descending)[0]
        val = dict(chosen)
        _SECAT_DERIVED.append({"name": name, "value": val,
                               "operation": f"selection:{'max' if descending else 'min'}({sort_key})",
                               "record": chosen,
                               "support_values": [v for v in chosen.values()
                                                  if isinstance(v, (str, int, float, bool))],
                               "capture": "explicit_helper"})
        return val
    def derive_value(self, name, value, operation="value"):
        _SECAT_DERIVED.append({"name": name, "value": value, "operation": operation,
                               "capture": "explicit_helper"})
        return value
ledger = _LedgerHelper()
'''


def _benchmark(task: dict) -> str:
    """Return the explicitly configured API domain without task-id inference."""
    value = task.get("benchmark") or os.environ.get("SECAT_BENCHMARK")
    if not value:
        raise ValueError("OCA requires an explicit API/benchmark configuration")
    return str(value)


def _extract_code(text: str) -> str | None:
    match = re.search(r"<execute>(.*?)</execute>", text or "", re.S)
    return match.group(1).strip() if match else None


def _normalize_generated_phase_a_code(code: str) -> str:
    """Normalize harmless top-level ``requests`` boilerplate safely.

    The persistent kernel already contains OCA's guarded ``requests`` proxy.
    Generated code may redundantly write a *direct top-level* unaliased
    ``import requests`` (including semicolon or mixed-import forms).  Remove only
    that exact binding.  Nested/conditional imports, aliases, submodules, and
    from-imports are intentionally left in place so the isolation validator can
    reject them as proxy-bypass attempts.
    """
    import ast as _phase_ast

    text = str(code or "")
    try:
        tree = _phase_ast.parse(text)
    except SyntaxError:
        return text

    changed = False
    rebuilt = []
    for stmt in tree.body:
        if isinstance(stmt, _phase_ast.Import):
            kept = []
            removed = False
            for alias in stmt.names:
                if str(alias.name) == "requests" and alias.asname is None:
                    removed = True
                    changed = True
                else:
                    kept.append(alias)
            if removed:
                if kept:
                    stmt.names = kept
                    rebuilt.append(stmt)
                else:
                    # ``pass`` is syntax-safe even for an original semicolon form
                    # such as ``import requests; r = requests.get(...)``.
                    rebuilt.append(_phase_ast.Pass())
                continue
        rebuilt.append(stmt)

    if not changed:
        return text
    tree.body = rebuilt
    _phase_ast.fix_missing_locations(tree)
    try:
        return _phase_ast.unparse(tree)
    except Exception:
        # Fail closed by returning the original code; the validator will still
        # inspect it and reject any requests import that was not safely normalized.
        return text

def _validate_generated_phase_a_code(code: str) -> list[str]:
    """Reject execution primitives that can bypass OCA's trusted API boundary.

    Phase A needs ordinary Python data processing plus the injected ``requests``
    proxy. It never needs filesystem/process access or an alternate network stack.
    This validator is API-agnostic and runs before model-generated code reaches
    the persistent kernel. Host-generated ledger inspection code is not subject to
    this check.
    """
    import ast
    try:
        tree = ast.parse(code or "")
    except SyntaxError as exc:
        return [f"unsupported/non-Python execution syntax: {exc.msg}"]
    allowed_import_roots = {
        "requests", "json", "re", "math", "statistics", "datetime",
        "collections", "itertools", "functools", "operator", "decimal",
        "fractions", "typing",
    }
    blocked_modules = {
        "os", "sys", "pathlib", "subprocess", "socket", "http", "urllib",
        "urllib3", "httpx", "aiohttp", "requests_html", "glob", "shutil",
        "tempfile", "importlib", "ftplib", "telnetlib",
    }
    blocked_calls = {
        "open", "exec", "eval", "compile", "__import__", "globals", "locals",
        "vars", "getattr", "setattr", "delattr", "input", "breakpoint", "get_ipython",
    }
    blocked_names = {
        "_real_requests", "_requests_module", "_secat_req", "_SecatReplayResponse",
        "_LedgerRequests", "_LedgerHelper", "_os", "_sys",
        "_json", "_re", "_urlparse",
        "_SECAT_LEDGER", "_SECAT_WRITE_COMMITS", "_SECAT_UNCERTAIN_WRITES", "_SECAT_EXECUTION_RULES",
        "_SECAT_PLAN_CALL_COUNTS", "_SECAT_ADVISORY_CALL_COUNTS", "_SECAT_DERIVED",
        "_SECAT_RESPONSE_SEQ", "_SECAT_IDEMPOTENT_RETRY_COUNTS",
        "_SECAT_IDEMPOTENT_RETRY_REQUESTS", "_SECAT_MAX_IDEMPOTENT_WRITE_RETRIES",
    }
    # Trusted guard functions live in the same persistent kernel as generated
    # code.  Python resolves builtins through globals first, so rebinding one of
    # these names (e.g. ``dict = {...}``) can corrupt the control plane on a later
    # turn even though the generated snippet itself appears harmless.
    protected_global_bindings = {
        "dict", "list", "set", "tuple", "str", "int", "float", "bool",
        "len", "max", "min", "sum", "sorted", "isinstance", "issubclass",
        "any", "all", "next", "range", "enumerate", "zip", "map", "filter",
        "getattr", "hasattr", "reversed",
        "Exception", "RuntimeError", "ValueError", "TypeError", "object",
        "type", "print",
        "requests", "ledger",
    } | blocked_names
    errors: list[str] = []
    top_level_import_ids = {id(node) for node in tree.body if isinstance(node, ast.Import)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = str(alias.name).split(".", 1)[0]
                bound = str(alias.asname or root)
                if bound.startswith("_secat_") or bound.startswith("_Secat"):
                    errors.append(f"blocked trusted-runtime binding: {bound}")
                if root in blocked_modules or root not in allowed_import_roots:
                    errors.append(f"blocked import: {alias.name}")
                # The only safe requests import is the exact top-level
                # ``import requests`` boilerplate removed by the normalizer
                # before validation. Any alias/submodule import would expose the
                # real network module outside the injected trusted proxy.
                exact_requests = (root == "requests" and alias.asname is None
                                  and str(alias.name) == "requests")
                if root == "requests" and not exact_requests:
                    errors.append(f"blocked alternate requests import: {alias.name}")
                # Exact requests boilerplate is admissible only as a direct
                # module-level import.  The execution normalizer removes that
                # form before kernel execution.  A nested/conditional exact import
                # would create a real-module binding later and is therefore blocked.
                safe_top_level_requests = exact_requests and id(node) in top_level_import_ids
                if exact_requests and not safe_top_level_requests:
                    errors.append("blocked nested requests import")
                if bound in protected_global_bindings and not safe_top_level_requests:
                    errors.append(f"blocked protected binding: {bound}")
        elif isinstance(node, ast.ImportFrom):
            root = str(node.module or "").split(".", 1)[0]
            if root in blocked_modules or root not in allowed_import_roots:
                errors.append(f"blocked import: {node.module}")
            if root == "requests":
                errors.append(f"blocked from-import: {node.module}")
            for alias in node.names:
                bound = str(alias.asname or alias.name)
                if bound.startswith("_secat_") or bound.startswith("_Secat"):
                    errors.append(f"blocked trusted-runtime binding: {bound}")
                if bound in protected_global_bindings:
                    errors.append(f"blocked protected binding: {bound}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if str(node.name).startswith("_secat_") or str(node.name).startswith("_Secat"):
                errors.append(f"blocked trusted-runtime binding: {node.name}")
            if str(node.name) in protected_global_bindings:
                errors.append(f"blocked protected binding: {node.name}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in blocked_calls:
                errors.append(f"blocked call: {node.func.id}")
        elif isinstance(node, ast.Name):
            trusted_prefix = (node.id.startswith("_SECAT_") or node.id.startswith("_secat_")
                              or node.id.startswith("_Secat"))
            if isinstance(node.ctx, (ast.Store, ast.Del)) and (
                    node.id in protected_global_bindings or trusted_prefix):
                errors.append(f"blocked protected binding: {node.id}")
            elif node.id in blocked_names or trusted_prefix:
                errors.append(f"blocked trusted-runtime name: {node.id}")
            elif isinstance(node.ctx, ast.Load) and (
                    node.id in blocked_modules or node.id in blocked_calls or
                    node.id.startswith("__")):
                # Block aliases of pre-existing privileged globals as well as
                # direct use.  Without this, snippets such as ``x = os`` or
                # ``f = open`` could rename a blocked capability and use the
                # alias on a later line.  Phase A never needs these names.
                errors.append(f"blocked privileged name: {node.id}")
        elif isinstance(node, ast.Attribute):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                # Phase A never needs to monkey-patch an object/module.  Blocking
                # attribute writes closes proxy/module/class mutation routes such
                # as ``requests.get = ...`` or ``json.loads = ...``.
                errors.append(f"blocked attribute mutation: {node.attr}")
            if str(node.attr).startswith("__"):
                errors.append(f"blocked introspection attribute: {node.attr}")
            # Direct access through an already-present blocked module/global is
            # also forbidden even when the current snippet omits an import.
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and (
                    root.id in blocked_modules or root.id in blocked_names or
                    root.id.startswith("_SECAT_") or root.id.startswith("_secat_")
                    or root.id.startswith("_Secat")):
                errors.append(f"blocked runtime/module access: {root.id}.{node.attr}")
    return list(dict.fromkeys(errors))


def _execute_generated_code(interp, code: str) -> dict[str, Any]:
    code = _normalize_generated_phase_a_code(code)
    violations = _validate_generated_phase_a_code(code)
    if violations:
        message = "OCA_ISOLATION_VIOLATION: " + "; ".join(violations[:6])
        return {"stdout": "", "stderr": message, "combined": message,
                "auto_display": "", "exit_code": 1, "timed_out": False,
                "success": False, "var_trace": []}
    return interp.execute(code)


def _build_phase_a_prompt(benchmark: str, plan=None) -> str:
    from utils.evidence_plan import acquisition_plan_prompt_section
    import benchmarks as B
    spec = B.get_benchmark(benchmark)
    label = str(spec.get("api_label") or "the configured REST API")
    base_url = str(spec.get("base_url") or "")
    mode = str(getattr(config, "OCA_OBSERVATION_MODE", "raw")).lower()

    prompt = (
        "You execute a validated API plan in a persistent Python interpreter. "
        "Use Python only inside <execute>...</execute> blocks. Use the injected requests object directly; do not import requests because the injected object is the trusted API proxy. "
        "Do not answer from prior knowledge.\n"
        f"API: {label}\nBase URL: {base_url}\n"
        "Authentication is handled by the runtime. Do not read environment variables, add auth headers, or print secrets. "
        "Do not import privileged system or network modules.\n\n"
        "Rules:\n"
        "1. Follow the validated plan. Do not change routes or request values.\n"
        "2. Reuse values and variables from earlier steps. Run dependent steps together when possible.\n"
        "3. Do not make unrelated API calls.\n"
        "4. Keep printed output small. Full API responses are stored by the runtime.\n"
        "5. When the plan is complete, write exactly READY_TO_ANSWER.\n"
    )
    if mode == "raw":
        prompt += "Print only the records needed to inspect execution.\n"
    return prompt + acquisition_plan_prompt_section(plan)


def _chat_json(messages, model, max_tokens=1000, stage_name="phase_b"):
    kwargs = dict(model=model, messages=messages, temperature=0.0)
    from utils.token_meter import stage as token_stage
    with token_stage(stage_name):
        try:
            return client.chat.completions.create(
                **kwargs, response_format={"type": "json_object"},
                max_completion_tokens=max_tokens)
        except Exception:
            return client.chat.completions.create(**kwargs)


def _phase_b_focus_ids(ledger, plan, compiled=None):
    from utils.phase_b_focus import focus_obs_ids
    return focus_obs_ids(
        ledger, plan, compiled,
        enabled=bool(getattr(config, "OCA_PHASE_B_FOCUS_FALLBACK", False)),
        dep_context=int(getattr(config, "OCA_PHASE_B_DEP_CONTEXT", 6)))


def _phase_b(question, ledger, model, logger, contract, plan, compiled,
             turn_num=98, correction_notes=None, stage_name="phase_b"):
    from utils.evidence_contract import contract_prompt_section
    focus = _phase_b_focus_ids(ledger, plan, compiled)
    max_chars = int(getattr(config, "OCA_LEDGER_MAX_CHARS", 24000))
    ledger_text = ledger.serialize(max_chars=max_chars, focus_obs_ids=focus)
    plan_summary = json.dumps({
        "steps": (plan or {}).get("steps", []),
        "answer_steps": (plan or {}).get("answer_steps", []),
        "answer_mode": (plan or {}).get("answer_mode"),
    }, ensure_ascii=False)
    correction = ""
    if correction_notes:
        correction = (
            "\n\nA previous draft failed verification for these reasons:\n- " +
            "\n- ".join(str(x) for x in correction_notes[:12]) +
            "\nCreate a fresh answer from the ledger and deterministic derivations. "
            "Do not preserve the previous draft."
        )
    system = (
        "Answer the user's question using only the supplied evidence. "
        "Follow the validated plan and deterministic derivations. "
        "Include only what the user asked for. Do not guess. "
        "For image answers, return the supported image reference and do not describe what it shows unless the evidence says so. "
        "If a required value is missing, return an empty answer. "
        + contract_prompt_section(contract) + correction
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content":
            f"QUESTION:\n{question}\n\nVALIDATED PLAN:\n{plan_summary}\n\nLEDGER:\n{ledger_text}"},
    ]
    response = _chat_json(messages, model,
                          max_tokens=int(getattr(config, "OCA_PHASE_B_MAX_TOKENS", 900)),
                          stage_name=stage_name)
    output = response.choices[0].message.content or ""
    log_no_code_turn(logger=logger, turn_num=turn_num, messages=messages,
                     llm_output=output, error_type="phase_b_extract",
                     scope="S5_Integration", is_error=False)
    obj: dict[str, Any] = {}
    match = re.search(r"\{.*\}", output, re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
        except Exception:
            obj = {}
    final_answer = str(obj.get("final_answer") or "").strip()
    if not final_answer and not obj:
        final_answer = output.strip()
    cited = [str(x) for x in (obj.get("cited_observation_ids") or [])]
    cert = {
        "cited_observation_ids": cited,
        "answer_values": obj.get("answer_values") or [],
        "derivation_ids": obj.get("derivation_ids") or [],
        "selection_anchor_id": (obj.get("selection_anchor_id") or
                                obj.get("selected_observation_id")),
        "answer_observation_ids": obj.get("answer_observation_ids") or [],
    }
    return final_answer, cited, cert, output


def _semantic_commit_review(question, final_answer, cert, contract, plan,
                            obs_index, der_index, model, logger, turn_num=95,
                            stage_name="semantic_commit"):
    """One bounded, evidence-only semantic check for non-compiler answers.

    Structural certification proves provenance and replay, but a Phase-B model can
    still verbalize an upstream/adjacent relation while citing real evidence.  This
    reviewer sees only the task, declared answer requirements, candidate answer,
    certificate values, and the *cited* evidence.  It never sees gold answers or
    benchmark routes and is forbidden from proposing a replacement answer.
    """
    cited = [str(x) for x in (cert or {}).get("cited_observation_ids") or []]
    # The structural verifier certifies the complete declared derivation graph, but
    # Phase B is free to cite only the terminal value.  A semantic reviewer that
    # sees only that terminal value can therefore falsely reject a correct answer
    # because the upstream rank/filter/relation evidence is absent from its packet.
    # Include every host-replayed derivation required by the validated contract.
    required_plan_ids = {
        str(x.get("id") or "")
        for x in (contract or {}).get("required_plan_derivations") or []
        if str(x.get("id") or "")
    }
    required_der_ids = [
        str(oid) for oid, record in (der_index or {}).items()
        if str((record or {}).get("plan_derivation_id") or "") in required_plan_ids
    ]
    evidence_ids = list(dict.fromkeys(cited + required_der_ids))
    packet = []
    for oid in evidence_ids[:32]:
        record = (der_index or {}).get(oid) or (obs_index or {}).get(oid)
        if not isinstance(record, dict):
            continue
        compact = {
            "id": oid,
            "kind": record.get("kind") or record.get("type"),
            "plan_step_id": record.get("plan_step_id"),
            "relation": record.get("relation"),
            "operation": record.get("operation"),
            "value": record.get("value"),
            "fields": record.get("fields"),
            "selected_obs_id": record.get("selected_obs_id"),
            "candidate_obs_ids": (record.get("candidate_obs_ids") or [])[:12],
        }
        text = json.dumps(compact, ensure_ascii=False, default=str)
        packet.append(text[:1400])
    payload = {
        "question": str(question),
        "answer_mode": (contract or {}).get("answer_kind"),
        "answer_requirements": (contract or {}).get("answer_requirements") or [],
        "answer_steps": (plan or {}).get("answer_steps") or [],
        "candidate_answer": str(final_answer or ""),
        "certified_values": (cert or {}).get("answer_values") or [],
        "cited_evidence": packet,
    }
    messages = [
        {"role": "system", "content": (
            "Check whether the candidate answer directly answers the user's question using only the supplied evidence. "
            "Check the requested relation, owner, selection, number of items, and output values. "
            "Accept only when the answer is directly supported. For image requests, an image path or URL can be a valid answer. "
            "Do not propose a replacement answer. Return JSON only: {\"accept\":true|false,\"reason\":\"short reason\"}."
        )},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)[:12000]},
    ]
    response = _chat_json(
        messages, model,
        max_tokens=int(getattr(config, "OCA_SEMANTIC_COMMIT_MAX_TOKENS", 220)),
        stage_name=stage_name)
    output = response.choices[0].message.content or ""
    log_no_code_turn(logger=logger, turn_num=turn_num, messages=messages,
                     llm_output=output, error_type="semantic_commit",
                     scope="S5_Integration", is_error=False)
    obj = {}
    match = re.search(r"\{.*\}", output, re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
        except Exception:
            obj = {}
    return {
        "accept": obj.get("accept") is True,
        "reason": str(obj.get("reason") or "semantic commit did not return an affirmative evidence-only decision")[:800],
        "raw": output[:1600],
    }


def _deterministic_comparison_commit(final_answer, cert, contract, der_index):
    """Trust a host-replayed ordering winner when it is internally provable.

    A semantic LLM should not be allowed to overturn a deterministic comparison
    simply because, for example, ``32 > 45`` is false while the compiler correctly
    reports the *second* source as the winner.  This helper is intentionally
    narrow: it applies only to comparison answers with a unique numeric/date
    winner, an explicit compiler-supplied winner label, and that same label in the
    structurally certified answer surface.

    It does not decide which relation/population should have been compared; those
    remain the job of the deterministic semantic-risk layer and structural plan
    validation.  It only prevents a second model from misreading a comparison the
    host has already replayed exactly.
    """
    if str((contract or {}).get("answer_kind") or "").lower() != "comparison":
        return None

    def _domain(value):
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return ("number", float(value))
        if isinstance(value, str):
            text = value.strip()
            # ISO dates/datetimes are the only textual ordering domain trusted
            # here.  Arbitrary lexical string order is not a semantic metric.
            try:
                import datetime as _dt
                probe = text.replace("Z", "+00:00")
                if "T" in probe:
                    return ("date", _dt.datetime.fromisoformat(probe).timestamp())
                return ("date", float(_dt.date.fromisoformat(probe).toordinal()))
            except Exception:
                return None
        return None

    def _norm(value):
        return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))

    certified_values = [x.get("value") for x in (cert or {}).get("answer_values") or []
                        if isinstance(x, dict)]
    candidate_norm = _norm(final_answer)
    supported = []
    for did, record in (der_index or {}).items():
        if str((record or {}).get("operation") or "").lower() != "compare":
            continue
        value = (record or {}).get("value")
        if not isinstance(value, dict):
            continue
        mode = str(value.get("comparison") or (record or {}).get("comparison_mode") or "").lower()
        if mode not in {"max", "min", "gt", "gte", "lt", "lte"}:
            continue
        vals = list(value.get("values") or [])
        if len(vals) < 2:
            continue
        parsed = [_domain(x) for x in vals]
        if any(x is None for x in parsed) or len({x[0] for x in parsed}) != 1:
            continue
        comparables = [x[1] for x in parsed]
        if len(set(comparables)) != len(comparables):
            # A tie has no unique winner and must not be silently trusted.
            continue
        want_max = mode in {"max", "gt", "gte"}
        expected = (max if want_max else min)(range(len(comparables)),
                                               key=lambda i: comparables[i])
        winner = value.get("winner_index")
        if not isinstance(winner, int) or winner != expected:
            continue
        if mode in {"gt", "gte", "lt", "lte"} and "result" in value:
            left, right = comparables[0], comparables[1]
            expected_result = {
                "gt": left > right,
                "gte": left >= right,
                "lt": left < right,
                "lte": left <= right,
            }[mode]
            if bool(value.get("result")) != expected_result:
                continue
        label = value.get("winner_label")
        label_norm = _norm(label)
        if not label_norm or label_norm not in candidate_norm:
            continue
        if not any(_norm(v) == label_norm for v in certified_values):
            continue
        supported.append({
            "derivation_id": str(did),
            "winner_index": winner,
            "winner_label": label,
            "values": vals,
            "comparison": mode,
        })

    if not supported:
        return None
    return {
        "accept": True,
        "reason": "host-replayed deterministic winner comparison is internally consistent",
        "deterministic_comparison_trust": True,
        "comparisons": supported[:4],
    }


def _apply_semantic_commit(verification, question, final_answer, cert, contract,
                           plan, obs_index, der_index, model, logger,
                           *, deterministic=False, turn_num=95,
                           stage_name="semantic_commit"):
    """Apply semantic commit only after structural verification has succeeded."""
    vr = dict(verification or {})
    vr["checks"] = dict(vr.get("checks") or {})
    vr["missing_slots"] = list(vr.get("missing_slots") or [])
    vr["notes"] = list(vr.get("notes") or [])
    if not (vr.get("certificate_accepted") and vr.get("contract_satisfied")):
        return vr, None

    # Accuracy-first recovery keeps the old zero-cost deterministic fast path.  The LLM semantic
    # judge is invoked globally only when the legacy flag is explicitly enabled;
    # otherwise accuracy-first mode invokes it *selectively* for generic,
    # high-confidence risk shapes. This catches false-certified answers without
    # adding a redundant model call to ordinary clean deterministic tasks.
    force_global = bool(getattr(config, "OCA_SEMANTIC_COMMIT", False))
    selective = bool(getattr(config, "OCA_ACCURACY_SEMANTIC_REVIEW", True))
    selective_risks = []
    deterministic_plan_risks = []
    surface_risks = []
    if selective and not force_global:
        try:
            from utils.accuracy_semantics import plan_semantic_risks, answer_surface_risks
            deterministic_plan_risks = list(plan_semantic_risks(question, plan))
            selective_risks.extend(deterministic_plan_risks)
            surface_risks = list(answer_surface_risks(question, final_answer, contract))
            selective_risks.extend(surface_risks)
        except Exception:
            selective_risks = []
            surface_risks = []
        selective_risks.extend(str(x) for x in (plan or {}).get("semantic_critic_errors") or [])
        selective_risks.extend(
            str(x.get("reason") or x) for x in (plan or {}).get("selection_semantic_risks") or [])
        selective_risks = list(dict.fromkeys(x for x in selective_risks if str(x).strip()))

    # Some answer-surface failures are factual impossibilities for the requested
    # output type and do not need an LLM opinion. In particular, absence prose or
    # null placeholders are never an image/logo/file asset. Reject deterministically
    # so a semantic judge cannot accidentally certify a missing asset as an answer.
    hard_asset_surface = [
        x for x in surface_risks
        if str(x).startswith("asset task candidate is an empty/null placeholder")
    ]
    if hard_asset_surface:
        review = {
            "accept": False,
            "reason": hard_asset_surface[0],
            "trigger_risks": list(selective_risks)[:12],
            "deterministic_surface_rejection": True,
        }
        vr["checks"]["semantic_commit"] = False
        vr["contract_satisfied"] = False
        vr["verification_status"] = "contradiction"
        if "semantic_commit" not in vr["missing_slots"]:
            vr["missing_slots"].append("semantic_commit")
        vr["notes"].append("deterministic answer-surface rejection: " + hard_asset_surface[0])
        return vr, review

    # If the structural certificate is deterministic and the only reason we
    # would call the semantic judge is advisory/critic residue, prefer the exact
    # host-replayed ordering proof.  Never take this shortcut while a current
    # deterministic plan/surface risk remains unresolved.
    if (deterministic and not force_global and not deterministic_plan_risks
            and not surface_risks
            and not (plan or {}).get("selection_semantic_risks")):
        comparison_review = _deterministic_comparison_commit(
            final_answer, cert, contract, der_index)
        if comparison_review is not None:
            comparison_review["trigger_risks"] = list(selective_risks)[:12]
            vr["checks"]["semantic_commit"] = True
            return vr, comparison_review

    if not force_global and not (selective and selective_risks):
        return vr, None

    review = _semantic_commit_review(
        question, final_answer, cert, contract, plan, obs_index, der_index,
        model, logger, turn_num=turn_num, stage_name=stage_name)
    if selective_risks:
        review["trigger_risks"] = list(selective_risks)[:12]
    vr["checks"]["semantic_commit"] = bool(review.get("accept"))
    if not review.get("accept"):
        vr["contract_satisfied"] = False
        vr["verification_status"] = "contradiction"
        if "semantic_commit" not in vr["missing_slots"]:
            vr["missing_slots"].append("semantic_commit")
        vr["notes"].append("semantic commit rejected candidate: " + str(review.get("reason") or "unsupported relation"))
    return vr, review


def _kernel_json_list(interp, expression: str, marker: str, *, fail_closed: bool) -> list[dict[str, Any]]:
    """Serialize trusted kernel state without relying on mutable model globals."""
    begin = f"__SECAT_{marker}_BEGIN__"
    end = f"__SECAT_{marker}_END__"
    # Re-import under a protected _SECAT_* name in the same host statement.
    # Generated Phase-A code cannot read or bind _SECAT_* identifiers.
    code = (
        "import json as _SECAT_HOST_JSON\n"
        + "print(" + repr(begin) + " + _SECAT_HOST_JSON.dumps(" + expression + ") + " + repr(end) + ")"
    )
    try:
        result = interp.execute(code)
        raw = result.get("stdout") or result.get("combined") or ""
        match = re.search(re.escape(begin) + r"(.*?)" + re.escape(end), raw, re.S)
        if not match:
            raise RuntimeError(f"trusted kernel {marker.lower()} marker missing")
        value = json.loads(match.group(1))
        if not isinstance(value, list):
            raise RuntimeError(f"trusted kernel {marker.lower()} is not a list")
        return value
    except Exception as exc:
        if fail_closed:
            raise RuntimeError(
                f"OCA_TRUSTED_KERNEL_STATE_UNREADABLE: {marker.lower()}: {type(exc).__name__}") from exc
        print(f"[WARN] could not read kernel {marker.lower()}: {exc}")
        return []


def _kernel_entries(interp) -> list[dict[str, Any]]:
    # Never turn an unreadable trusted execution ledger into an empty ledger:
    # recovery could otherwise replay a side effect that actually succeeded.
    return _kernel_json_list(interp, "_SECAT_LEDGER", "LEDGER", fail_closed=True)


def _kernel_entries_since(interp, start: int) -> list[dict[str, Any]]:
    """Read only newly appended kernel-ledger entries after ``start``."""
    index = max(0, int(start))
    return _kernel_json_list(interp, f"_SECAT_LEDGER[{index}:]", "LEDGER", fail_closed=True)


def _kernel_write_commits(interp) -> list[dict[str, Any]]:
    """Read the append-only successful state-changing-call journal."""
    return _kernel_json_list(interp, "_SECAT_WRITE_COMMITS", "WRITE_COMMITS", fail_closed=True)


def _kernel_uncertain_writes(interp) -> list[dict[str, Any]]:
    """Read state-changing requests whose transport outcome is unknown."""
    return _kernel_json_list(interp, "_SECAT_UNCERTAIN_WRITES", "UNCERTAIN_WRITES", fail_closed=True)


def _kernel_derived_entries(interp) -> list[dict[str, Any]]:
    return _kernel_json_list(interp, "_SECAT_DERIVED", "DERIVED", fail_closed=False)


def _stable_fingerprint(value: Any) -> str:
    return json.dumps(redact_secrets(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=str)


def _request_fingerprint(entry: dict[str, Any]) -> str:
    return _stable_fingerprint({
        "method": str(entry.get("method") or "GET").upper(),
        "endpoint": entry.get("endpoint"),
        "effective_endpoint": entry.get("effective_endpoint") or entry.get("endpoint"),
        "params": entry.get("params") or {},
        "request_body": entry.get("request_body"),
    })


def _repair_progress_state(entries: list[dict[str, Any]],
                           derived_entries: list[dict[str, Any]]) -> dict[str, set[str]]:
    requests = {_request_fingerprint(entry) for entry in entries}
    observations = {
        _stable_fingerprint({
            "request": _request_fingerprint(entry),
            "status_code": entry.get("status_code"),
            "payload": entry.get("payload"),
            "adaptation": entry.get("adaptation"),
        })
        for entry in entries
    }
    return {
        "requests": requests,
        "observations": observations,
        "derivations": {_stable_fingerprint(item) for item in derived_entries},
    }


def _repair_made_no_progress(previous: dict[str, set[str]],
                             current: dict[str, set[str]],
                             new_entries: list[dict[str, Any]]) -> bool:
    repeated_request = any(
        _request_fingerprint(entry) in previous.get("requests", set())
        for entry in new_entries)
    new_observations = (current.get("observations", set()) -
                        previous.get("observations", set()))
    new_derivations = (current.get("derivations", set()) -
                       previous.get("derivations", set()))
    return bool(repeated_request and not new_observations and not new_derivations)


def _api_call_accounting(interp, committed_ledger) -> dict[str, int]:
    committed = len(committed_ledger.api_calls)
    attempted = max(committed, len(_kernel_entries(interp)))
    return {
        "api_calls": committed,
        "api_calls_committed": committed,
        "api_calls_attempted": attempted,
        "api_calls_uncommitted": max(0, attempted - committed),
    }


def _executed_route_templates(plan: dict[str, Any], ledger) -> list[str]:
    """Return route templates for committed calls without consulting evaluator gold."""
    by_id = {str(x.get("id") or ""): str(x.get("endpoint") or "")
             for x in (plan or {}).get("steps") or []}
    routes: list[str] = []
    for call in getattr(ledger, "api_calls", []) or []:
        sid = str(call.get("plan_step_id") or call.get("runtime_step_id") or "")
        endpoint = by_id.get(sid) or str(call.get("endpoint") or "")
        if endpoint:
            routes.append(endpoint)
    return routes


def _phase_a_message_view(messages):
    if str(getattr(config, "OCA_PHASE_A_HISTORY", "full")).lower() != "compact":
        return messages
    from utils.phase_a_context import compact_messages
    return compact_messages(messages, keep_recent_pairs=1)


def _classify_phase_a_execution(code: str, raw_output: str,
                                shown_feedback: str) -> dict[str, Any]:
    """Classify Phase-A output while honoring a proven empty projection.

    A loop over a legitimately empty collection produces no local stdout, but
    adaptive projection still gives OCA an explicit, validated observation. Do
    not count that one case as a perception failure. All other silent results,
    including mismatches and zero rows produced only by filtering, retain the
    shared classifier's ``unobserved_result`` error.
    """
    from utils.error_classifier import classify_error

    error_info = classify_error(code, raw_output)
    # In adaptive mode the trusted projection is the observation delivered to the
    # model. Local stdout may legitimately be empty even though the model received
    # a validated projected response. Any projection_status=ok therefore rules out
    # an S4 "unobserved result" classification, not only the empty-result case.
    observed_projection = any(
        line.startswith("projection_status=ok")
        for line in str(shown_feedback or "").splitlines()
    )
    if (error_info.get("error_type") == "unobserved_result"
            and observed_projection):
        error_info = dict(error_info)
        error_info.update({
            "error_type": "none",
            "scope": "none",
            "is_error": False,
            "is_silent": False,
            "raw_error": None,
        })
    return error_info


def _phase_a_feedback(new_entries, result, question, plan, call_offset=0,
                      all_entries=None, model=None):
    mode = str(getattr(config, "OCA_OBSERVATION_MODE", "raw")).lower()
    raw = (result.get("auto_display") or result.get("combined") or
           result.get("stdout") or "")
    advisory_notes = [
        "REQUEST CONTRACT ADVISORY: safe GET executed for context but does not satisfy "
        "the validated plan step " + str(x.get("advisory_candidate_step_id") or "?") +
        "; " + str(x.get("request_contract_diagnostic") or "request arguments differ")
        for x in (new_entries or []) if x.get("request_contract_advisory")]
    advisory_prefix = ("\n".join(advisory_notes) + "\n") if advisory_notes else ""
    if mode == "raw":
        return (advisory_prefix + raw)[:6000]
    if mode == "receipt":
        from utils.observation_receipt import feedback_text
        return (advisory_prefix + feedback_text(
            new_entries, result, question=question, plan=plan,
            max_chars=int(getattr(config, "OCA_RECEIPT_MAX_CHARS", 3500)),
            local_max_chars=int(getattr(config, "OCA_LOCAL_OUTPUT_MAX_CHARS", 1200)),
            call_offset=call_offset))[:6000]
    if mode != "adaptive":
        return (advisory_prefix + raw)[:6000]

    from utils.observation_projection import adaptive_feedback
    shown, diagnostics = adaptive_feedback(
        new_entries, result, question=question, plan=plan,
        all_entries=list(all_entries or new_entries), call_offset=call_offset,
        max_chars=int(getattr(config, "OCA_RECEIPT_MAX_CHARS", 3500)),
        local_max_chars=int(getattr(config, "OCA_LOCAL_OUTPUT_MAX_CHARS", 1200)))

    # Runtime response schemas sometimes drift from the documented OAS. Repair a
    # projection only when its concrete paths are missing, and give the repair LLM
    # field names/types/counts only -- never the response values. One repair per
    # plan step is enough to avoid a hidden retry loop.
    if (diagnostics and model and plan and bool(getattr(
            config, "OCA_ADAPTIVE_RUNTIME_REPAIR", True))):
        repaired_steps = set(plan.get("_projection_repaired_steps") or [])
        max_repairs = max(0, int(getattr(config, "OCA_ADAPTIVE_MAX_RUNTIME_REPAIRS", 2)))
        steps = {str(s.get("id")): s for s in (plan.get("steps") or [])}
        specs = {str(s.get("step_id")): s for s in (plan.get("observation_specs") or [])
                 if s.get("step_id")}
        changed = False
        for diag in diagnostics:
            sid = str(diag.get("step_id") or "")
            if (not sid or sid in repaired_steps or len(repaired_steps) >= max_repairs
                    or sid not in steps):
                continue
            from utils.evidence_plan import repair_observation_spec
            repaired, repair_raw, repair_errors = repair_observation_spec(
                question, steps[sid], specs.get(sid), diag.get("profile") or {},
                model, client)
            event = {
                "step_id": sid,
                "trigger_errors": list(diag.get("errors") or []),
                "accepted": bool(repaired),
                "validation_errors": list(repair_errors or []),
                "planner_output": str(repair_raw or "")[:3000],
            }
            plan.setdefault("projection_repair_events", []).append(event)
            repaired_steps.add(sid)
            if repaired:
                specs[sid] = repaired
                changed = True
                print(f"[PROJECTION] runtime schema repair accepted for {sid}")
            else:
                print(f"[PROJECTION] runtime schema repair failed for {sid}: "
                      f"{repair_errors[:2]}")
        plan["_projection_repaired_steps"] = sorted(repaired_steps)
        if changed:
            ordered = []
            for step in plan.get("steps") or []:
                sid = str(step.get("id"))
                if sid in specs:
                    ordered.append(specs[sid])
            plan["observation_specs"] = ordered
            shown, _ = adaptive_feedback(
                new_entries, result, question=question, plan=plan,
                all_entries=list(all_entries or new_entries), call_offset=call_offset,
                max_chars=int(getattr(config, "OCA_RECEIPT_MAX_CHARS", 3500)),
                local_max_chars=int(getattr(config, "OCA_LOCAL_OUTPUT_MAX_CHARS", 1200)))
    return (advisory_prefix + shown)[:6000]


def _read_ledger_from_kernel(interp) -> ObservationLedger:
    ledger = ObservationLedger()
    for entry in _kernel_entries(interp):
        created = ledger.record_response(
            entry.get("endpoint", ""), entry.get("params") or {}, entry.get("payload"),
            method=entry.get("method", "GET"),
            status_code=entry.get("status_code"),
            request_body=entry.get("request_body"),
            effective_endpoint=entry.get("effective_endpoint"),
            adaptation=entry.get("adaptation"),
            request_origin=entry.get("request_origin"))
        runtime_step = entry.get("runtime_step_id")
        forced_step = entry.get("plan_step_id")
        tagged_step = forced_step or runtime_step
        if tagged_step and ledger.api_calls:
            ledger.api_calls[-1]["plan_step_id"] = str(tagged_step)
            ledger.api_calls[-1]["runtime_step_id"] = str(tagged_step)
            # Preserve the live guard's authorization certificate.  This is
            # generated only by the trusted request wrapper after an exact rule
            # match; model code cannot mint it.
            ledger.api_calls[-1]["runtime_contract_authorized"] = bool(entry.get("runtime_contract_authorized", False))
            ledger.api_calls[-1]["runtime_contract_step_id"] = str(entry.get("runtime_contract_step_id") or "")
            ledger.api_calls[-1]["runtime_contract_version"] = int(entry.get("runtime_contract_version") or 0)
            if forced_step:
                ledger.api_calls[-1]["forced_plan_step"] = True
            if ledger.raw_responses:
                ledger.raw_responses[-1]["plan_step_id"] = str(tagged_step)
                ledger.raw_responses[-1]["runtime_step_id"] = str(tagged_step)
                ledger.raw_responses[-1]["runtime_contract_authorized"] = bool(entry.get("runtime_contract_authorized", False))
                ledger.raw_responses[-1]["runtime_contract_step_id"] = str(entry.get("runtime_contract_step_id") or "")
                ledger.raw_responses[-1]["runtime_contract_version"] = int(entry.get("runtime_contract_version") or 0)
                if forced_step:
                    ledger.raw_responses[-1]["forced_plan_step"] = True
            for oid in created:
                obs = ledger.get(oid)
                if obs is not None:
                    obs["plan_step_id"] = str(tagged_step)
                    obs["runtime_step_id"] = str(tagged_step)
                    if forced_step:
                        obs["forced_plan_step"] = True
    return ledger


def _read_derived_into(interp, ledger):
    before = len(ledger.derived)
    try:
        # Host-side trusted state reads must not depend on a mutable alias that
        # lives in the model's persistent kernel namespace. Re-import json under
        # a protected host-only name exactly as _kernel_json_list does.
        result = interp.execute(
            "import json as _SECAT_HOST_JSON\n"
            "print('__SECAT_DERIVED_BEGIN__' + _SECAT_HOST_JSON.dumps(_SECAT_DERIVED) + '__SECAT_DERIVED_END__')")
        raw = result.get("stdout") or result.get("combined") or ""
        match = re.search(r"__SECAT_DERIVED_BEGIN__(.*?)__SECAT_DERIVED_END__", raw, re.S)
        entries = json.loads(match.group(1)) if match else []
        for item in entries:
            ids = ledger._values_in_ledger(item.get("support_values") or [])
            extra = {"record": item.get("record")} if item.get("record") else None
            ledger._record_derived(
                item.get("name", "derived"), item.get("value"),
                item.get("operation", "value"), ids,
                item.get("capture", "explicit_helper"), extra=extra)
    except Exception as exc:
        print(f"[WARN] could not read explicit derivations: {exc}")
    return len(ledger.derived) - before


def _namespace_repair_plan(plan: dict[str, Any], prefix: str = "repair_") -> dict[str, Any]:
    """Give a post-execution repair plan disjoint step/derivation ids.

    The repair executes in the same trusted kernel/ledger so total call budgets
    and prior evidence remain visible.  Namespacing prevents the host audit from
    confusing old completed calls with the fresh repair plan when both use common
    planner ids such as s1/d1.
    """
    out = json.loads(json.dumps(plan or {}, ensure_ascii=False, default=str))
    steps = out.get("steps") or []
    derivs = out.get("derivations") or []
    step_map = {str(x.get("id") or ""): prefix + str(x.get("id") or "") for x in steps}
    deriv_map = {str(x.get("id") or ""): prefix + str(x.get("id") or "") for x in derivs}
    for step in steps:
        sid = str(step.get("id") or "")
        step["id"] = step_map.get(sid, sid)
        step["depends_on"] = [step_map.get(str(x), str(x)) for x in step.get("depends_on") or []]
    for deriv in derivs:
        did = str(deriv.get("id") or "")
        deriv["id"] = deriv_map.get(did, did)
        deriv["source_steps"] = [step_map.get(str(x), str(x)) for x in deriv.get("source_steps") or []]
        deriv["source_derivations"] = [deriv_map.get(str(x), str(x))
                                         for x in deriv.get("source_derivations") or []]
        deriv["label_steps"] = [step_map.get(str(x), str(x)) for x in deriv.get("label_steps") or []]
    out["answer_steps"] = [step_map.get(str(x), str(x)) for x in out.get("answer_steps") or []]
    for obligation in out.get("action_obligations") or []:
        if isinstance(obligation, dict):
            sid = str(obligation.get("step_id") or "")
            obligation["step_id"] = step_map.get(sid, sid)
    for spec in out.get("observation_specs") or []:
        sid = str(spec.get("step_id") or "")
        spec["step_id"] = step_map.get(sid, sid)
    out["runtime_repair_namespaced"] = True
    return out




def _dependent_plan_steps(plan, producer_step_ids):
    """Return transitive descendants of changed producer steps.

    Canonical search repair can change a selected entity after an earlier child GET
    already ran with the old binding.  The stale call remains in the ledger, but its
    per-step call-budget slot must not prevent the corrected descendant request.
    """
    changed = {str(x) for x in (producer_step_ids or []) if str(x)}
    if not changed:
        return []
    steps = [x for x in (plan or {}).get("steps") or [] if isinstance(x, dict)]
    descendants = set()
    progress = True
    while progress:
        progress = False
        parents = changed | descendants
        for step in steps:
            sid = str(step.get("id") or "")
            if not sid or sid in parents:
                continue
            deps = {str(x) for x in step.get("depends_on") or []}
            if deps & parents:
                descendants.add(sid)
                progress = True
    return sorted(descendants)


def _reset_kernel_plan_call_budget(interp, step_ids):
    """Reset only repaired descendant step budgets inside the trusted kernel.

    This does not delete evidence, reset the task-wide call ceiling, or authorize a
    new route.  It only lets the already-validated repaired plan execute one fresh
    request for a step whose previous request instance is now lineage-invalid.
    """
    ids = sorted({str(x) for x in (step_ids or []) if str(x)})
    if not ids or not hasattr(interp, "_run_silent"):
        return []
    code = (
        "for _sid in " + repr(ids) + ":\n"
        "    _SECAT_PLAN_CALL_COUNTS.pop(_sid, None)\n"
        "    _SECAT_ADVISORY_CALL_COUNTS.pop(_sid, None)"
    )
    interp._run_silent(code, check=True)
    return ids


def _apply_observed_canonical_search_repairs(interp, plan, ledger):
    """Apply unique low-risk API label canonicalizations before child binding."""
    try:
        from utils.accuracy_semantics import repair_observed_canonical_search_labels
        repaired, records = repair_observed_canonical_search_labels(plan, ledger)
    except Exception:
        return plan, []
    if records:
        _sync_kernel_execution_rules(interp, repaired)
        # A producer correction can make a previously executed child request
        # lineage-invalid (for example rank-0 owner A -> canonical owner B).
        # Preserve that stale call as evidence, but free only descendant per-step
        # budget slots so deterministic completion can execute the corrected GET.
        changed_producers = [str(r.get("step_id") or "") for r in records]
        reset_steps = _dependent_plan_steps(repaired, changed_producers)
        step_map = {str(x.get("id") or ""): x for x in (repaired or {}).get("steps") or []}
        # Never reopen the budget of a state-changing descendant. Accuracy recovery
        # may refetch corrected read evidence, but writes remain fail-closed.
        reset_steps = [sid for sid in reset_steps
                       if str((step_map.get(sid) or {}).get("method") or "GET").upper() in {"GET", "HEAD"}]
        reset_steps = _reset_kernel_plan_call_budget(interp, reset_steps)
        if reset_steps:
            for record in records:
                record["reset_descendant_call_budgets"] = list(reset_steps)
    return repaired, records


def _apply_observed_relation_repairs(interp, question, plan, ledger):
    """Apply same-response relation fallback and reopen only affected read descendants."""
    try:
        from utils.accuracy_semantics import repair_observed_actor_relation_siblings
        repaired, records = repair_observed_actor_relation_siblings(question, plan, ledger)
    except Exception:
        return plan, []
    if records:
        _sync_kernel_execution_rules(interp, repaired)
        changed = [str(r.get("step_id") or "") for r in records]
        reset_steps = _dependent_plan_steps(repaired, changed)
        step_map = {str(x.get("id") or ""): x for x in (repaired or {}).get("steps") or []}
        reset_steps = [sid for sid in reset_steps
                       if str((step_map.get(sid) or {}).get("method") or "GET").upper() in {"GET","HEAD"}]
        reset_steps = _reset_kernel_plan_call_budget(interp, reset_steps)
        for record in records:
            record["reset_descendant_call_budgets"] = list(reset_steps)
    return repaired, records



def _apply_observed_order_repairs(interp, question, plan, ledger):
    """Use strongly monotone observed endpoint order to close page-local extrema."""
    try:
        from utils.accuracy_semantics import repair_observed_ordered_extrema
        repaired, records = repair_observed_ordered_extrema(question, plan, ledger)
    except Exception:
        return plan, []
    if records:
        _sync_kernel_execution_rules(interp, repaired)
    return repaired, records

def _asset_answer_surface_is_empty(plan, ledger) -> bool:
    if str((plan or {}).get("answer_mode") or "").lower() != "asset":
        return False
    answer_steps = {str(x) for x in (plan or {}).get("answer_steps") or []}
    specs = {str(x.get("step_id") or ""): x for x in (plan or {}).get("observation_specs") or []}
    calls = {str(c.get("call_id") or ""): c for c in getattr(ledger, "api_calls", []) or []}
    saw_answer_call = False
    from utils.observation_projection import extract_values
    for raw in getattr(ledger, "raw_responses", []) or []:
        sid = str(raw.get("plan_step_id") or (calls.get(str(raw.get("call_id") or "")) or {}).get("plan_step_id") or "")
        if sid not in answer_steps:
            continue
        saw_answer_call = True
        spec = specs.get(sid) or {}
        values = extract_values(raw.get("payload"), spec.get("record_path") or "$")
        if len(values) == 1 and isinstance(values[0], list):
            values = list(values[0])
        if any(v not in (None, "", [], {}) for v in values):
            return False
    return saw_answer_call


def _asset_search_candidate_positions(plan, ledger, *, max_candidates=4):
    """Return plausible alternate endpoint ranks for an empty search-owned asset."""
    if str((plan or {}).get("answer_mode") or "").lower() != "asset":
        return []
    steps = {str(x.get("id") or ""): x for x in (plan or {}).get("steps") or []}
    answer_steps = [str(x) for x in (plan or {}).get("answer_steps") or []]
    if len(answer_steps) != 1 or answer_steps[0] not in steps:
        return []
    child = steps[answer_steps[0]]
    deps = [str(x) for x in child.get("depends_on") or [] if str(x) in steps]
    if len(deps) != 1:
        return []
    producer_id = deps[0]; producer = steps[producer_id]
    if "/search/" not in str(producer.get("endpoint") or "").lower():
        return []
    literals = producer.get("query_literals") or {}
    query = next((str(literals.get(k)).strip() for k in ("query", "q", "name", "title")
                  if literals.get(k) not in (None, "")), "")
    tokens = [x for x in re.findall(r"[a-z0-9]+", query.casefold()) if x not in {"the", "a", "an"}]
    if not tokens:
        return []
    candidates = []
    seen_pos = set()
    for obs in getattr(ledger, "observations", []) or []:
        if str(obs.get("plan_step_id") or "") != producer_id:
            continue
        fields = obs.get("fields") if isinstance(obs.get("fields"), dict) else {}
        label = next((fields.get(k) for k in ("name", "title", "label", "display_name")
                      if isinstance(fields.get(k), str) and fields.get(k).strip()), None)
        if not label:
            continue
        try:
            pos = int(obs.get("position"))
        except Exception:
            continue
        if pos <= 0 or pos in seen_pos:
            continue
        label_tokens = re.findall(r"[a-z0-9]+", str(label).casefold())
        if not all(tok in label_tokens for tok in tokens):
            continue
        seen_pos.add(pos)
        extra = max(0, len(label_tokens) - len(tokens))
        candidates.append((extra, pos, str(label)))
    candidates.sort(key=lambda x: (x[0], x[1]))
    return [{"rank": pos, "label": label} for _extra, pos, label in candidates[:max_candidates]]


def _asset_rank_fallback_plan(plan, producer_id: str, rank: int):
    """Clone a plan while selecting exactly one alternate search result by rank."""
    out = json.loads(json.dumps(plan or {}, ensure_ascii=False, default=str))
    changed_selector = False
    for deriv in out.get("derivations") or []:
        if str(producer_id) not in {str(x) for x in deriv.get("source_steps") or []}:
            continue
        op = str(deriv.get("operator") or "").lower()
        if op in {"endpoint_rank", "first", "nth"}:
            deriv["operator"] = "endpoint_rank"
            deriv["rank"] = int(rank)
            changed_selector = True
            break
    if not changed_selector:
        return None
    changed_spec = False
    for spec in out.get("observation_specs") or []:
        if str(spec.get("step_id") or "") == str(producer_id):
            spec["select"] = {"mode": "nth", "limit": 1, "index": int(rank)}
            for binding in spec.get("bindings") or []:
                binding["source"] = "selected_first"
            changed_spec = True
            break
    return out if changed_spec else None


def _run_asset_owner_fallback(interp, benchmark, instruction, logger, plan, ledger, compiled,
                              phase_stdout, *, base_turn=52):
    """Try alternate plausible search owners when the selected asset surface is empty.

    Every attempt remains a strict one-owner GET plan. The trusted host re-audits
    and reuses the original search call, changes only the selected endpoint rank,
    and executes the missing child asset request for that exact owner.
    """
    record = {"attempted": False, "attempts": [], "adopted": False}
    if not _asset_answer_surface_is_empty(plan, ledger):
        return plan, ledger, compiled, None, 0, [], record, None, None
    candidates = _asset_search_candidate_positions(
        plan, ledger, max_candidates=int(getattr(config, "OCA_ASSET_OWNER_FALLBACK_CANDIDATES", 4)))
    if not candidates:
        return plan, ledger, compiled, None, 0, [], record, None, None
    steps = {str(x.get("id") or ""): x for x in (plan or {}).get("steps") or []}
    answer_step = str(((plan or {}).get("answer_steps") or [""])[0])
    deps = [str(x) for x in (steps.get(answer_step) or {}).get("depends_on") or []]
    if len(deps) != 1:
        return plan, ledger, compiled, None, 0, [], record, None, None
    producer_id = deps[0]
    original_plan = plan
    working_ledger = ledger
    all_stdout = []
    total_runs = 0
    record["attempted"] = True
    record["candidate_positions"] = candidates
    from utils.evidence_compiler import compile_evidence
    for idx, candidate in enumerate(candidates, 1):
        rank = int(candidate["rank"])
        variant = _asset_rank_fallback_plan(original_plan, producer_id, rank)
        if not variant:
            break
        variant = _namespace_repair_plan(variant, prefix=f"assetfb{idx}_")
        replay_map = _sync_kernel_execution_rules(interp, variant, replay_ledger=working_ledger) or {}
        runs, extra_stdout, stop_reason = _run_deterministic_plan_completion(
            interp, benchmark, variant, logger,
            max_calls=int(getattr(config, "OCA_RUNTIME_REPLAN_MAX_CALLS", 12)),
            base_turn=base_turn + (idx - 1) * 3)
        total_runs += int(runs or 0); all_stdout.extend(extra_stdout or [])
        merged_stdout = list(phase_stdout) + list(all_stdout)
        variant_ledger, ne, nr = _rebuild_ledger(interp, variant, merged_stdout)
        variant_progress = _audit_ledger_execution(variant, variant_ledger)
        variant_compiled = compile_evidence(instruction, variant_ledger, variant)
        answer_ids = list(variant_compiled.get("answer_derivation_ids") or [])
        remaining = _runtime_plan_semantic_risks(instruction, variant, variant_ledger)
        attempt = {
            "rank": rank, "label": candidate.get("label"),
            "reused_prior_call_ids": sum(len(v) for v in replay_map.values()),
            "host_calls": int(runs or 0), "stop_reason": stop_reason,
            "plan_complete": bool(variant_progress.get("complete")),
            "answer_derivations": len(answer_ids),
            "remaining_semantic_risks": remaining[:6],
            "compiler_warnings": list(variant_compiled.get("warnings") or [])[:6],
        }
        record["attempts"].append(attempt)
        working_ledger = variant_ledger
        if variant_progress.get("complete") and answer_ids and not remaining:
            record.update({"adopted": True, "selected_rank": rank,
                           "selected_label": candidate.get("label")})
            return (variant, variant_ledger, variant_compiled, variant_progress,
                    total_runs, all_stdout, record, ne, nr)
    _sync_kernel_execution_rules(interp, original_plan, clear_replay=True)
    return original_plan, working_ledger, compiled, None, total_runs, all_stdout, record, None, None



def _run_season_episode_relation_fallback(interp, benchmark, instruction, model, logger,
                                           plan, ledger, compiled, phase_stdout, *, base_turn=54):
    """Recover omitted aggregate season roles from bounded episode-credit fan-out."""
    record={'attempted':False,'adopted':False,'attempts':[]}
    try:
        from utils.accuracy_semantics import observed_season_episode_relation_variant
        from utils.evidence_plan import (_load_tools,_evaluate_convergence_candidate,
                                         make_observation_plan,annotate_execution_eligibility)
        tools=_load_tools(benchmark)
        variant,meta=observed_season_episode_relation_variant(instruction,plan,ledger,tools)
    except Exception:
        return plan,ledger,compiled,None,0,[],record,None,None
    if not variant: return plan,ledger,compiled,None,0,[],record,None,None
    record['attempted']=True;record.update(meta)
    valid_paths=[str(t.get('path') or '') for t in tools if t.get('path')];methods={}
    for t in tools: methods.setdefault(str(t.get('path') or ''),set()).add(str(t.get('method') or 'GET').upper())
    candidate,errors=_evaluate_convergence_candidate(instruction,benchmark,variant,tools,valid_paths,methods)
    candidate=annotate_execution_eligibility(candidate)
    if not (candidate.get('valid') and candidate.get('execution_eligible')):
        record['attempts'].append({'validation_errors':list(errors)[:8],'adopted':False})
        return plan,ledger,compiled,None,0,[],record,None,None
    try:
        candidate,_=make_observation_plan(instruction,benchmark,candidate,model,client,attempts=1,deterministic_first=True)
        candidate=annotate_execution_eligibility(candidate)
    except Exception as exc:
        record['attempts'].append({'adopted':False,'reason':f'observation plan unavailable: {exc}'})
        return plan,ledger,compiled,None,0,[],record,None,None
    season_sid=str(meta.get('season_step_id') or '');episode_sid=str(meta.get('episode_step_id') or '')
    # Force bounded selected-all episode binding; this is what authorizes finite fan-out.
    for spec in candidate.get('observation_specs') or []:
        if str(spec.get('step_id') or '')==season_sid:
            spec['record_path']='episodes[*]';spec['project_paths']=list(dict.fromkeys(list(spec.get('project_paths') or [])+['episode_number']))
            spec['select']={'mode':'all_matches','limit':20,'index':0};spec['completeness']='all returned episode candidates inspected'
            found=False
            for b in spec.get('bindings') or []:
                if str(b.get('name') or '')=='episode_number':
                    b.update({'path':'episode_number','source':'selected_all','max_values':20});found=True
            if not found: spec.setdefault('bindings',[]).append({'name':'episode_number','path':'episode_number','source':'selected_all','max_values':20})
        elif str(spec.get('step_id') or '')==episode_sid:
            spec['record_path']='crew[*]';spec['project_paths']=list(dict.fromkeys(list(spec.get('project_paths') or [])+['id','name','job']))
    candidate=_namespace_repair_plan(candidate,prefix='seasonfb_')
    replay_map=_sync_kernel_execution_rules(interp,candidate,replay_ledger=ledger) or {}
    runs,extra_stdout,stop_reason=_run_deterministic_plan_completion(
        interp,benchmark,candidate,logger,max_calls=30,base_turn=base_turn)
    merged=list(phase_stdout)+list(extra_stdout or [])
    cand_ledger,ne,nr=_rebuild_ledger(interp,candidate,merged)
    progress=_audit_ledger_execution(candidate,cand_ledger)
    from utils.evidence_compiler import compile_evidence
    cand_compiled=compile_evidence(instruction,cand_ledger,candidate)
    remaining=_runtime_plan_semantic_risks(instruction,candidate,cand_ledger)
    answer_ids=list(cand_compiled.get('answer_derivation_ids') or [])
    attempt={'reused_prior_call_ids':sum(len(v) for v in replay_map.values()),'host_calls':int(runs or 0),
             'stop_reason':stop_reason,'plan_complete':bool(progress.get('complete')),
             'answer_derivations':len(answer_ids),'remaining_semantic_risks':remaining[:8],
             'compiler_warnings':list(cand_compiled.get('warnings') or [])[:8]}
    record['attempts'].append(attempt)
    if progress.get('complete') and answer_ids and not remaining:
        record['adopted']=True;attempt['adopted']=True
        return candidate,cand_ledger,cand_compiled,progress,int(runs or 0),extra_stdout or [],record,ne,nr
    _sync_kernel_execution_rules(interp,plan,clear_replay=True)
    return plan,cand_ledger,compiled,None,int(runs or 0),extra_stdout or [],record,None,None

def _run_cross_resource_search_fallback(interp, benchmark, instruction, model, logger,
                                        plan, ledger, compiled, phase_stdout, *, base_turn=56):
    """Try documented sibling resource searches when a named work search clearly missed.

    This is a bounded read-only evidence acquisition strategy. It does not use gold
    routes or answers: variants come only from the current plan, observed empty/
    unrelated search evidence, and sibling operations present in the OAS.
    """
    record={"attempted":False,"attempts":[],"adopted":False}
    try:
        from utils.accuracy_semantics import observed_cross_resource_search_variants
        from utils.evidence_plan import (_load_tools, _evaluate_convergence_candidate,
                                         make_observation_plan, annotate_execution_eligibility)
        tools=_load_tools(benchmark)
        variants=observed_cross_resource_search_variants(
            instruction,plan,ledger,tools,
            max_variants=int(getattr(config,"OCA_CROSS_RESOURCE_FALLBACK_VARIANTS",3)))
    except Exception:
        return plan,ledger,compiled,None,0,[],record,None,None
    if not variants:
        return plan,ledger,compiled,None,0,[],record,None,None
    valid_paths=[str(t.get("path") or "") for t in tools if t.get("path")]
    methods={}
    for t in tools:
        methods.setdefault(str(t.get("path") or ""),set()).add(str(t.get("method") or "GET").upper())
    record["attempted"]=True; total_runs=0; all_stdout=[]; working_ledger=ledger
    from utils.evidence_compiler import compile_evidence
    for idx,item in enumerate(variants,1):
        candidate,errors=_evaluate_convergence_candidate(
            instruction,benchmark,item["plan"],tools,valid_paths,methods)
        candidate=annotate_execution_eligibility(candidate)
        attempt={"search_endpoint":item.get("search_endpoint"),"new_resource":item.get("new_resource"),
                 "mapped_descendants":item.get("mapped_descendants"),"validation_errors":list(errors)[:6]}
        if not (candidate.get("valid") and candidate.get("execution_eligible") and
                all(str(s.get("method") or "GET").upper() in {"GET","HEAD"} for s in candidate.get("steps") or [])):
            attempt["adopted"]=False;attempt["reason"]="deterministic sibling variant did not validate"
            record["attempts"].append(attempt);continue
        try:
            candidate,_=make_observation_plan(
                instruction,benchmark,candidate,model,client,attempts=1,deterministic_first=True)
            candidate=annotate_execution_eligibility(candidate)
        except Exception as exc:
            attempt["adopted"]=False;attempt["reason"]=f"observation plan unavailable: {exc}"
            record["attempts"].append(attempt);continue
        if not candidate.get("observation_specs"):
            attempt["adopted"]=False;attempt["reason"]="no deterministic observation plan"
            record["attempts"].append(attempt);continue
        candidate=_namespace_repair_plan(candidate,prefix=f"xres{idx}_")
        replay_map=_sync_kernel_execution_rules(interp,candidate,replay_ledger=working_ledger) or {}
        runs,extra_stdout,stop_reason=_run_deterministic_plan_completion(
            interp,benchmark,candidate,logger,
            max_calls=int(getattr(config,"OCA_RUNTIME_REPLAN_MAX_CALLS",12)),
            base_turn=base_turn+(idx-1)*4)
        total_runs+=int(runs or 0);all_stdout.extend(extra_stdout or [])
        merged=list(phase_stdout)+list(all_stdout)
        cand_ledger,ne,nr=_rebuild_ledger(interp,candidate,merged)
        # Newly fetched sibling search evidence may now support canonical rank repair.
        candidate,canon=_apply_observed_canonical_search_repairs(interp,candidate,cand_ledger)
        progress=_audit_ledger_execution(candidate,cand_ledger)
        cand_compiled=compile_evidence(instruction,cand_ledger,candidate)
        remaining=_runtime_plan_semantic_risks(instruction,candidate,cand_ledger)
        answer_ids=list(cand_compiled.get("answer_derivation_ids") or [])
        if item.get("co_starring_variant") and answer_ids:
            derived_by_id={str(d.get("obs_id") or ""):d for d in getattr(cand_ledger,"derived",[]) or []}
            positive=any(isinstance((derived_by_id.get(str(aid)) or {}).get("value"),bool) and
                         (derived_by_id.get(str(aid)) or {}).get("value") is True for aid in answer_ids)
            if positive:
                # A positive two-person cast proof is stronger evidence for a
                # localized/alternate work title than lexical title overlap alone.
                remaining=[r for r in remaining if not str(r).startswith("named search ")]
                attempt["positive_costar_proof"] = True
        attempt.update({"reused_prior_call_ids":sum(len(v) for v in replay_map.values()),
                        "host_calls":int(runs or 0),"stop_reason":stop_reason,
                        "canonical_repairs":canon,"plan_complete":bool(progress.get("complete")),
                        "answer_derivations":len(answer_ids),"remaining_semantic_risks":remaining[:6],
                        "compiler_warnings":list(cand_compiled.get("warnings") or [])[:6]})
        record["attempts"].append(attempt);working_ledger=cand_ledger
        if progress.get("complete") and answer_ids and not remaining:
            attempt["adopted"]=True;record.update({"adopted":True,"selected_resource":item.get("new_resource"),
                                                    "selected_search_endpoint":item.get("search_endpoint")})
            return candidate,cand_ledger,cand_compiled,progress,total_runs,all_stdout,record,ne,nr
    return plan,working_ledger,compiled,None,total_runs,all_stdout,record,None,None


def _runtime_evidence_repair_feedback(plan: dict[str, Any], compiled: dict[str, Any],
                                      *, question: str = "", ledger=None,
                                      verification: dict[str, Any] | None = None) -> dict[str, Any]:
    """Answer-free diagnostics for bounded post-execution strategy recovery."""
    try:
        from utils.accuracy_semantics import recovery_feedback
        return recovery_feedback(question, plan, compiled, ledger=ledger, verification=verification)
    except Exception:
        endpoints = [
            {"method": str(x.get("method") or "GET").upper(),
             "endpoint": str(x.get("endpoint") or ""),
             "purpose": str(x.get("purpose") or "")[:240]}
            for x in (plan or {}).get("steps") or []
        ]
        return {
            "previous_routes": endpoints,
            "answer_mode": str((plan or {}).get("answer_mode") or "direct"),
            "answer_requirements": list((plan or {}).get("answer_requirements") or [])[:8],
            "compiler_warnings": list((compiled or {}).get("warnings") or [])[:10],
            "instruction": "Choose stronger evidence for the same requested relation; do not repeat a failed strategy.",
        }


def _runtime_plan_semantic_risks(question: str, plan: dict[str, Any], ledger=None) -> list[str]:
    try:
        from utils.accuracy_semantics import plan_semantic_risks, observed_semantic_risks
        risks = list(plan_semantic_risks(question, plan)) + list(
            observed_semantic_risks(question, plan, ledger))
        # The typed intent compiler deliberately keeps OAS route diagnostics as
        # advisories so acquisition may proceed.  Once evidence exists, however,
        # those same diagnostics are strong signals that a structurally valid plan
        # selected the wrong population/relation.  Feed them into bounded accuracy
        # recovery rather than silently certifying the wrong route.
        for warning in (plan or {}).get("validation_warnings") or []:
            text = str(warning or "")
            if text.casefold().startswith("typed-intent route advisory:"):
                risks.append(text.split(":", 1)[-1].strip() or text)
        return list(dict.fromkeys(x for x in risks if x))
    except Exception:
        return []


def _answer_evidence_lineage_complete(plan: dict[str, Any], progress: dict[str, Any],
                                     compiled: dict[str, Any], ledger=None) -> bool:
    """Return True when the concrete evidence ancestry for every answer value ran.

    Plan-wide completion is intentionally stronger than answer completion.  A
    planner may leave an explanatory/redundant branch unresolved even though the
    exact derivation used for the answer and every request that fed it already
    executed.  Use the ledger DAG first, then include declared step dependencies,
    so unrelated missing work cannot force a whole-route replan or abstention.
    """
    answer_ids = [str(x) for x in (compiled or {}).get("answer_derivation_ids") or []]
    if not answer_ids or ledger is None:
        return False
    derived = {str(d.get("obs_id") or ""): d for d in getattr(ledger, "derived", []) or []}
    observed = {str(o.get("obs_id") or ""): o for o in getattr(ledger, "observations", []) or []}
    if any(aid not in derived for aid in answer_ids):
        return False

    required_steps: set[str] = set()
    seen_evidence: set[str] = set()
    stack = list(answer_ids)
    while stack:
        oid = str(stack.pop() or "")
        if not oid or oid in seen_evidence:
            continue
        seen_evidence.add(oid)
        row = derived.get(oid) or observed.get(oid)
        if not row:
            # Compiler-derived evidence should never point outside the ledger.  If
            # it does, do not claim answer-lineage completion.
            return False
        sid = str(row.get("plan_step_id") or "")
        if sid:
            required_steps.add(sid)
        stack.extend(str(x) for x in row.get("input_obs_ids") or [])

    # A value observation from a child step semantically depends on its declared
    # request ancestors even if the low-level derived record did not list every
    # parent observation explicitly.
    by_id = {str(x.get("id") or ""): x for x in (plan or {}).get("steps") or []}
    step_stack = list(required_steps)
    while step_stack:
        sid = step_stack.pop()
        for parent in (by_id.get(sid) or {}).get("depends_on") or []:
            parent = str(parent or "")
            if parent and parent not in required_steps:
                required_steps.add(parent)
                step_stack.append(parent)

    completed = {str(x) for x in (progress or {}).get("completed_step_ids") or []}
    return bool(required_steps) and required_steps.issubset(completed)


def _needs_runtime_evidence_replan(plan: dict[str, Any], progress: dict[str, Any],
                                   compiled: dict[str, Any], *, question: str = "",
                                   ledger=None, verification=None) -> bool:
    """Accuracy-first trigger for a bounded read-only strategy change.

    Replanning is appropriate when a complete executable read attempt produced no
    answer, compiler/lineage closure failed, or generic question/plan/runtime
    semantics show a high-confidence mismatch.  It is never automatic for writes.
    """
    steps = list((plan or {}).get("steps") or [])
    if not steps:
        # A planner that failed to produce any route should get one whole-plan
        # recovery opportunity. The recovery helper will execute it only if the
        # replacement is a strict read-only plan, so this cannot synthesize writes.
        return bool(question and (plan or {}).get("valid") is False)
    if any(str(x.get("method") or "GET").upper() not in {"GET", "HEAD"}
           for x in steps):
        return False
    # A non-executable *invalid* read plan is itself a planning failure worth
    # replacing. Explicit non-executability on an otherwise-valid plan remains a
    # hard stop because it may encode a request-contract safety restriction.
    if (plan or {}).get("execution_eligible") is False and (plan or {}).get("valid") is not False:
        return False
    if _runtime_plan_semantic_risks(question, plan, ledger):
        return True
    warnings = " ".join(str(x) for x in (compiled or {}).get("warnings") or []).casefold()
    # Before contract verification, replan only on warnings that strongly indicate
    # a bad acquisition/population route. Compiler-representation gaps get their
    # normal deterministic closure first; if they still cause verification failure,
    # the verification-aware branch below can then request a whole strategy change.
    warning_terms = ("no candidates", "empty evidence set")
    answer_ids = list((compiled or {}).get("answer_derivation_ids") or [])
    if verification and not (
            verification.get("certificate_accepted") and verification.get("contract_satisfied")):
        # Verification failure is not synonymous with acquisition failure.  The
        # typed recovery diagnosis already distinguishes semantic/population gaps
        # from local plan/certificate closure.  Replanning the whole route for a
        # local closure defect wastes tokens and can discard correct evidence.
        feedback = _runtime_evidence_repair_feedback(
            plan, compiled, question=question, ledger=ledger, verification=verification)
        failure_type = str(feedback.get("failure_type") or "")
        if failure_type in {"semantic_strategy", "empty_or_wrong_population", "evidence_strategy"}:
            return True
        if failure_type in {"local_plan_closure", "certificate_or_lineage_closure"}:
            # A complete, semantically sound acquisition must not be discarded
            # because local replay/certificate closure failed. Route replanning is
            # for acquisition/semantic strategy errors, not derivation-engine bugs.
            if bool((progress or {}).get("complete")):
                return False
            return not bool(answer_ids)
        return True
    # Before verification, do not replace a compiler-replay problem with a route
    # change. Replan only when the current acquisition actually yielded no candidate
    # answer population. This preserves deterministic compiler-closure precedence.
    if not answer_ids and any(term in warnings for term in warning_terms):
        # Once every declared read step completed, "no candidates" is ambiguous:
        # it can be a legitimate empty relation or a local replay defect. Do not
        # restart language understanding or route selection without an independent
        # semantic-risk signal.
        return not bool((progress or {}).get("complete"))
    # An incomplete plan with known read-only missing work also deserves a route/
    # binding reconsideration after deterministic completion failed.
    if not (progress or {}).get("complete"):
        # Do not throw away a complete answer-producing evidence lineage merely
        # because an unrelated/redundant branch of the plan did not finish.
        if _answer_evidence_lineage_complete(plan, progress, compiled, ledger):
            return False
        return True
    return False


def _runtime_replan_attempt(interp, benchmark: str, instruction: str, model: str,
                            logger, plan: dict[str, Any], ledger, compiled: dict[str, Any],
                            phase_stdout: list[str], *, verification=None,
                            attempt_index: int = 1, base_turn: int = 60,
                            allow_actions: bool = False):
    """Try one whole strategy replacement in the same trusted kernel.

    By default this remains the historical read-only recovery.  ``allow_actions``
    is used only when the caller has established that no successful write has yet
    occurred, so a broken action strategy can be replaced without replaying an
    already-committed side effect.
    """
    original_plan = plan
    feedback = _runtime_evidence_repair_feedback(
        plan, compiled, question=instruction, ledger=ledger, verification=verification)
    record = {"attempted": True, "attempt_index": attempt_index,
              "semantic_risks": list(feedback.get("semantic_risks") or [])[:12]}
    try:
        from utils.evidence_plan import (
            make_evidence_plan, make_observation_plan, annotate_execution_eligibility,
        )
        if bool(getattr(config, "OCA_INTENT_COMPILER_ENABLED", False)):
            from utils.intent_compiler import make_intent_compiled_plan
            feedback = dict(feedback or {})
            feedback["prior_intent"] = (plan or {}).get("intent")
            repair_plan, repair_planner_output = make_intent_compiled_plan(
                instruction, benchmark, model, client,
                max_tokens=int(getattr(config, "OCA_INTENT_PLANNER_MAX_TOKENS", 900)),
                runtime_feedback=feedback, allow_legacy_fallback=bool(getattr(config, "OCA_INTENT_LEGACY_FALLBACK", False)),
                intent_repair_attempts=int(getattr(config, "OCA_INTENT_REPAIR_ATTEMPTS", 1)),
                legacy_attempts=int(getattr(config, "OCA_LEGACY_PLANNER_FALLBACK_ATTEMPTS", 1)),
                catalog_mode=str(getattr(config, "OCA_PLANNER_CATALOG_MODE", "full")))
        else:
            repair_plan, repair_planner_output = make_evidence_plan(
                instruction, benchmark, model, client,
                attempts=int(getattr(config, "OCA_RUNTIME_REPLAN_PLANNER_ATTEMPTS", 3)),
                api_hints=None, enable_semantic_adapters=False,
                catalog_mode=str(getattr(config, "OCA_PLANNER_CATALOG_MODE", "full")),
                enable_semantic_critic=True,
                enable_selection_semantic_guard=bool(getattr(config, "OCA_SELECTION_SEMANTIC_GUARD", True)),
                runtime_feedback=feedback)
        repair_plan = annotate_execution_eligibility(repair_plan)
        if repair_planner_output:
            log_no_code_turn(
                logger=logger, turn_num=92 + attempt_index,
                messages=[{"role": "system", "content": "OCA accuracy-first runtime re-plan"},
                          {"role": "user", "content": instruction}],
                llm_output=repair_planner_output, error_type="accuracy_recovery_plan",
                scope="S1_Intention", is_error=False)
        if repair_plan.get("steps") and repair_plan.get("execution_eligible"):
            repair_plan, _ = make_observation_plan(
                instruction, benchmark, repair_plan, model, client,
                attempts=int(getattr(config, "OCA_PROJECTION_PLANNER_ATTEMPTS", 1)),
                deterministic_first=bool(getattr(config, "OCA_DETERMINISTIC_OBSERVATION_PLAN", True)))
            repair_plan = annotate_execution_eligibility(repair_plan)
        try:
            import benchmarks as _benchmark_registry
            repair_plan = dict(repair_plan or {})
            repair_plan["_runtime_base_url"] = str(
                _benchmark_registry.get_benchmark(benchmark).get("base_url") or "")
            repair_plan["_execution_policy"] = "advisory"
        except Exception:
            repair_plan = dict(repair_plan or {})
            repair_plan["_execution_policy"] = "advisory"

        record.update({"planner_valid": bool(repair_plan.get("valid")),
                       "execution_eligible": bool(repair_plan.get("execution_eligible"))})
        _write_methods = {"POST", "PUT", "DELETE", "PATCH"}
        repair_has_write = any(str(x.get("method") or "GET").upper() in _write_methods
                               for x in repair_plan.get("steps") or [])
        if not (repair_plan.get("valid") and repair_plan.get("execution_eligible")):
            record.update({"adopted": False,
                           "reason": "recovery plan was not a strict executable plan"})
            return original_plan, ledger, compiled, None, 0, [], record, None, None
        if repair_has_write and not allow_actions:
            record.update({"adopted": False,
                           "reason": "read-only recovery rejected a state-changing plan"})
            return original_plan, ledger, compiled, None, 0, [], record, None, None

        try:
            from utils.accuracy_semantics import plan_signature
            same_strategy = plan_signature(repair_plan) == plan_signature(original_plan)
        except Exception:
            same_strategy = False
        if same_strategy:
            record.update({"adopted": False, "no_progress": True,
                           "reason": "planner repeated the same semantic strategy"})
            return original_plan, ledger, compiled, None, 0, [], record, None, None

        repair_plan = _namespace_repair_plan(repair_plan, prefix=f"replan{attempt_index}_")
        replay_map = (_sync_kernel_execution_rules(
            interp, repair_plan, replay_ledger=ledger) or {})
        record["reused_prior_call_ids"] = sum(len(v) for v in replay_map.values())
        record["reused_prior_steps"] = sorted(k for k, v in replay_map.items() if v)
        host_runs, extra_stdout, stop_reason = _run_deterministic_plan_completion(
            interp, benchmark, repair_plan, logger,
            max_calls=int(getattr(config, "OCA_RUNTIME_REPLAN_MAX_CALLS", 12)),
            base_turn=base_turn)
        merged_stdout = list(phase_stdout) + list(extra_stdout)
        repair_ledger, repair_explicit, repair_rpr = _rebuild_ledger(
            interp, repair_plan, merged_stdout)
        repair_progress = _audit_ledger_execution(repair_plan, repair_ledger)

        action_host_runs = 0
        action_host_stdout = []
        action_host_stop = None
        if allow_actions and repair_has_write and not repair_progress.get("complete"):
            action_host_runs, action_host_stdout, action_host_stop =                 _run_deterministic_action_completion(
                    interp, benchmark, repair_plan, logger,
                    max_calls=int(getattr(config, "OCA_ACTION_REPLAN_MAX_CALLS", 8)),
                    base_turn=base_turn + 2)
            merged_stdout.extend(action_host_stdout)
            repair_ledger, repair_explicit, repair_rpr = _rebuild_ledger(
                interp, repair_plan, merged_stdout)
            repair_progress = _audit_ledger_execution(repair_plan, repair_ledger)

        # If the new validated strategy is sound but host replay cannot resolve an
        # ambiguous join/binding, give the model a tiny plan-locked execution chance
        # in the same trusted kernel. This is execution recovery, not another route
        # invention; request guards still enforce the recovered plan exactly.
        model_runs = 0
        model_stdout = []
        model_meta = {}
        if not repair_progress.get("complete"):
            from utils.evidence_plan import missing_plan_prompt
            repair_system = _build_phase_a_prompt(benchmark, repair_plan)
            model_runs, model_stdout = _run_fetch_round(
                interp, repair_system,
                "Complete only the missing steps of this recovered evidence plan.\n" +
                missing_plan_prompt(repair_progress),
                model, logger, base_turn=base_turn + 4,
                max_steps=int(getattr(config, "OCA_RUNTIME_REPLAN_MODEL_STEPS", 2)),
                question=instruction, plan=repair_plan,
                stage_name="accuracy_replan_execution", progress_meta=model_meta)
            merged_stdout.extend(model_stdout)
            repair_ledger, repair_explicit, repair_rpr = _rebuild_ledger(
                interp, repair_plan, merged_stdout)
            repair_progress = _audit_ledger_execution(repair_plan, repair_ledger)

        from utils.evidence_compiler import compile_evidence
        repair_compiled = compile_evidence(instruction, repair_ledger, repair_plan)
        remaining_risks = _runtime_plan_semantic_risks(instruction, repair_plan, repair_ledger)
        answer_ids = list(repair_compiled.get("answer_derivation_ids") or [])
        successful_repair_write = any(
            str(c.get("method") or "GET").upper() in _write_methods
            and isinstance(c.get("status_code"), int) and 200 <= c.get("status_code") < 300
            for c in (getattr(repair_ledger, "api_calls", []) or []))
        if allow_actions and repair_has_write:
            # If a recovered action plan has committed a successful write, keep
            # that same plan even when a later action remains incomplete; rolling
            # back to the old strategy would lose side-effect state and risk replay.
            adopted = bool(not model_meta.get("fatal_error") and not remaining_risks
                           and (repair_progress.get("complete") or successful_repair_write))
        else:
            adopted = bool(
                repair_progress.get("complete") and answer_ids and not remaining_risks
                and not model_meta.get("fatal_error"))
        total_runs = int(host_runs or 0) + int(action_host_runs or 0) + int(model_runs or 0)
        all_stdout = list(extra_stdout) + list(action_host_stdout) + list(model_stdout)
        record.update({
            "plan_complete": bool(repair_progress.get("complete")),
            "host_calls": host_runs, "action_host_calls": action_host_runs,
            "action_host_stop": action_host_stop,
            "model_execution_runs": model_runs,
            "model_execution_stop": model_meta.get("stop_reason"),
            "stop_reason": stop_reason, "answer_derivations": len(answer_ids),
            "successful_repair_write": successful_repair_write,
            "compiler_warnings": list(repair_compiled.get("warnings") or [])[:8],
            "remaining_semantic_risks": remaining_risks[:8], "adopted": adopted,
        })
        if model_meta.get("fatal_error"):
            record["fatal_error"] = model_meta.get("fatal_error")
        if adopted:
            return (repair_plan, repair_ledger, repair_compiled, repair_progress,
                    total_runs, all_stdout, record, repair_explicit, repair_rpr)
        _sync_kernel_execution_rules(interp, original_plan, clear_replay=True)
        return original_plan, ledger, compiled, None, total_runs, all_stdout, record, None, None
    except Exception as exc:
        try:
            _sync_kernel_execution_rules(interp, original_plan, clear_replay=True)
        except Exception:
            pass
        record.update({"adopted": False,
                       "reason": "runtime accuracy re-plan failed safely: " + str(exc)[:500]})
        return original_plan, ledger, compiled, None, 0, [], record, None, None


def _sync_kernel_execution_rules(interp, plan, *, replay_ledger=None, clear_replay=False):
    """Refresh trusted binding/projection rules without resetting call budgets.

    ``replay_ledger`` is used only during whole-plan recovery.  The host audits
    prior calls against the recovered plan and sends the kernel an explicit
    step->call-id whitelist.  This lets a new strategy reuse already retrieved
    read evidence without trusting stale planner-local step IDs or repeating the
    same network request.
    """
    if not plan or not hasattr(interp, "_run_silent"):
        return {}
    from utils.plan_execution_audit import runtime_execution_rules
    rules = runtime_execution_rules(
        plan, max_fanout=int(getattr(config, "OCA_MAX_FANOUT_CALLS", 20)))
    replay_map = None
    if clear_replay:
        replay_map = {}
    elif replay_ledger is not None:
        try:
            from utils.plan_execution_audit import audit_execution
            audit = audit_execution(plan, replay_ledger)
            replay_map = {
                str(sid): [str(cid) for cid in (status.get("valid_call_ids") or []) if cid]
                for sid, status in (audit.get("step_status") or {}).items()
                if status.get("valid_call_ids")
            }
        except Exception:
            replay_map = {}
    from utils.kernel_executor import python_json_assignment
    code = (
        python_json_assignment("_SECAT_EXECUTION_RULES", rules) +
        "\n_SECAT_EXECUTION_POLICY = " + repr(str((plan or {}).get("_execution_policy") or "strict"))
    )
    if replay_map is not None:
        code += "\n" + python_json_assignment("_SECAT_REPLAY_CALL_IDS_BY_STEP", replay_map)
    interp._run_silent(code, check=True)
    return replay_map or {}


def _log_execution(logger, turn_num, messages, output, code, result,
                   error_type="none", scope="none", agent_output=None,
                   error_info=None):
    if agent_output is None:
        agent_output = (result.get("auto_display") or result.get("combined") or
                        result.get("stdout") or "")[:6000]
    if error_info is None:
        error_info = {"error_type": error_type, "scope": scope,
                      "is_error": False, "is_silent": False, "raw_error": None}
    logger.log_turn(
        turn_num=turn_num, llm_input=[dict(m) for m in messages], llm_output=output,
        code=code, exec_result=result, error_info=error_info,
        agent_output=agent_output)


def _run_fetch_round(interp, system_prompt, user_prompt, model, logger,
                     base_turn, max_steps=3, question="", plan=None,
                     stage_name="evidence_repair", progress_meta=None,
                     stop_on_plan_complete=False):
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    runs = 0
    stdout = []
    # One initial read establishes the incremental cursor. Subsequent feedback
    # fetches only newly captured responses rather than re-serializing the whole
    # kernel ledger on every turn.
    _existing_entries = _kernel_entries(interp)
    entry_cursor = len(_existing_entries)
    all_entries = list(_existing_entries)
    repair_progress = _repair_progress_state(
        all_entries, _kernel_derived_entries(interp))
    if progress_meta is not None:
        progress_meta.update({
            "stopped_no_progress": False,
            "stop_reason": None,
            "http_calls_observed": 0,
        })
    guard_signatures: dict[tuple[str, str, int], int] = {}
    execution_error_signatures: dict[tuple[str, str, int], int] = {}
    from utils.token_meter import stage as token_stage
    for offset in range(max_steps):
        sent_messages = _phase_a_message_view(messages)
        with token_stage(stage_name):
            response = client.chat.completions.create(
                model=model, messages=sent_messages, temperature=0.0)
        output = response.choices[0].message.content or ""
        code = _extract_code(output)
        if not code:
            log_no_code_turn(logger=logger, turn_num=base_turn + offset,
                             messages=sent_messages, llm_output=output,
                             error_type="fetch_round_signal", scope="S1_Intention",
                             is_error=False)
            if "REPAIR_DONE" in output or "READY_TO_ANSWER" in output:
                break
            messages.extend([
                {"role": "assistant", "content": output},
                {"role": "user", "content": "Run the missing API call in <execute> code, or write REPAIR_DONE."},
            ])
            continue
        result = _execute_generated_code(interp, code)
        new_entries = _kernel_entries_since(interp, entry_cursor)
        previous_cursor = entry_cursor
        entry_cursor += len(new_entries)
        all_entries.extend(new_entries)
        current_progress = _repair_progress_state(
            all_entries, _kernel_derived_entries(interp))
        made_no_progress = _repair_made_no_progress(
            repair_progress, current_progress, new_entries)
        repair_progress = current_progress
        if progress_meta is not None:
            progress_meta["http_calls_observed"] += len(new_entries)
        raw = (result.get("auto_display") or result.get("combined") or
               result.get("stdout") or "")
        # Preserve the original RPR input bound so token-only ablations do not
        # silently change evidence reconstruction semantics.
        stdout.append(raw[:6000])
        shown = _phase_a_feedback(
            new_entries, result, question, plan, call_offset=previous_cursor,
            all_entries=all_entries, model=model)
        _sync_kernel_execution_rules(interp, plan)
        try:
            error_info = _classify_phase_a_execution(code, raw, shown)
        except Exception:
            error_info = {"error_type": "none", "scope": "none",
                          "is_error": False, "is_silent": False,
                          "raw_error": None}
        _log_execution(logger, base_turn + offset, sent_messages, output, code, result,
                       agent_output=shown, error_info=error_info)
        runs += 1
        error_type = str(error_info.get("error_type") or "")
        fatal_types = {"HTTP_Auth", "task_call_budget", "unsupported_transport"}
        guard_types = {"unplanned_api_call", "unauthorized_binding",
                       "unauthorized_request_instance", "request_contract",
                       "dependency_not_ready", "idempotent_retry_mismatch",
                       "isolation_violation", "plan_call_budget"}
        if error_type in fatal_types:
            if progress_meta is not None:
                progress_meta.update({
                    "fatal_error": error_type,
                    "stop_reason": error_type,
                    "stopped_after_code_runs": runs,
                })
            print(f"[REPAIR] stopping on non-recoverable runtime failure: {error_type}")
            break
        # Generated repair code can also fail before making an HTTP call (for
        # example by referencing a host-only binding name).  Repeating the exact
        # same Python/runtime failure at the same ledger state cannot create new
        # evidence; stop that local loop early so the higher-level strategy replan
        # gets its chance instead of wasting every repair turn on the same NameError.
        if (error_info.get("is_error") and error_type not in fatal_types
                and error_type not in guard_types and not new_entries):
            raw_key = str(error_info.get("raw_error") or raw or shown)[:500]
            signature = (error_type or "execution_error", raw_key, entry_cursor)
            seen = execution_error_signatures.get(signature, 0) + 1
            execution_error_signatures[signature] = seen
            if seen >= 2:
                if progress_meta is not None:
                    progress_meta.update({
                        "stop_reason": "repeated_execution_error",
                        "stopped_after_code_runs": runs,
                    })
                print("[REPAIR] stopping after repeated execution error without new evidence")
                break
        if error_type in guard_types:
            # Stop only an *identical* guard failure at the same evidence state.
            # Different corrections, or the same request after new evidence was
            # obtained, are not the same no-progress action.
            signature = (error_type, str(error_info.get("raw_error") or shown)[:500], entry_cursor)
            seen = guard_signatures.get(signature, 0) + 1
            guard_signatures[signature] = seen
            if progress_meta is not None:
                progress_meta["guard_error_count"] = max(
                    int(progress_meta.get("guard_error_count", 0)), seen)
            if seen >= 2:
                if progress_meta is not None:
                    progress_meta.update({
                        "stop_reason": "repeated_" + error_type,
                        "stopped_after_code_runs": runs,
                    })
                print(f"[REPAIR] stopping after repeated runtime guard violation: {error_type}")
                break
        if stop_on_plan_complete and plan:
            live_ledger = _read_ledger_from_kernel(interp)
            live_progress = _audit_ledger_execution(plan, live_ledger)
            if live_progress.get("complete"):
                if progress_meta is not None:
                    progress_meta.update({
                        "plan_complete": True,
                        "stop_reason": "plan_complete",
                        "stopped_after_code_runs": runs,
                    })
                print("[REPAIR] evidence plan completed; stopping bounded repair")
                break
        if made_no_progress:
            if progress_meta is not None:
                progress_meta.update({
                    "stopped_no_progress": True,
                    "stop_reason": "exact_request_repeated_without_new_evidence",
                    "stopped_after_code_runs": runs,
                })
            print("[REPAIR] stopped after an exact repeated request added no new "
                  "observations or derivations")
            break
        messages.extend([
            {"role": "assistant", "content": output},
            {"role": "user", "content":
                f"Execution output:\n{shown}\n\nFetch another missing dependency if needed; otherwise write REPAIR_DONE."},
        ])
    return runs, stdout



def _binding_starved_get_refresh_target(interp, plan, ledger, progress):
    """Return one exact prior GET whose empty binding blocks a missing step.

    A 2xx GET can be structurally complete yet yield an empty collection.  If a
    downstream validated step requires a binding from that producer, action
    recovery would otherwise dead-end because the producer is no longer listed
    among ``missing_steps``.  Refresh only an already-successful GET, at most once
    per producer per task, and replay its exact credential-free endpoint/params.
    No new route, literal, or write is synthesized here.
    """
    from utils.plan_execution_audit import authorized_binding_sequence

    plan = plan or {}
    statuses = (progress or {}).get("step_status") or {}
    valid_map = {str(sid): set(str(x) for x in (st.get("valid_call_ids") or []))
                 for sid, st in statuses.items()}
    steps = {str(s.get("id") or ""): s for s in (plan.get("steps") or [])}
    calls = {str(c.get("call_id") or ""): c
             for c in (getattr(ledger, "api_calls", []) or [])}
    try:
        refreshed = set(str(x) for x in _kernel_json_list(
            interp, "_SECAT_BINDING_REFRESHED_STEPS", "BINDING_REFRESHED_STEPS",
            fail_closed=True))
    except Exception:
        return None

    for consumer in (progress or {}).get("missing_steps") or []:
        consumer_sid = str(consumer.get("id") or "")
        binding_names = []
        for mapping_name in ("path_bindings", "query_bindings", "body_bindings"):
            for binding in (consumer.get(mapping_name) or {}).values():
                name = str(binding or "")
                if name and name not in binding_names:
                    binding_names.append(name)
        for binding_name in binding_names:
            seq, producer, _source = authorized_binding_sequence(
                plan, ledger, consumer_sid, binding_name,
                valid_call_ids_by_step=valid_map)
            if seq or not producer:
                continue
            producer = str(producer)
            if producer in refreshed:
                continue
            producer_step = steps.get(producer) or {}
            if str(producer_step.get("method") or "GET").upper() != "GET":
                continue
            status = statuses.get(producer) or {}
            if (status.get("unresolved_placeholders") or
                    status.get("truncated_required_collection") or
                    status.get("incomplete_pagination")):
                continue
            valid_ids = [str(x) for x in (status.get("valid_call_ids") or []) if str(x)]
            if not valid_ids:
                continue
            # Reuse the most recent request that the execution audit already
            # accepted for this producer.  Its params are the semantic, redacted
            # request surface captured before auth injection.
            call = next((calls[cid] for cid in reversed(valid_ids) if cid in calls), None)
            if not call or str(call.get("method") or "GET").upper() != "GET":
                continue
            return {
                "plan_step_id": producer,
                "endpoint": str(call.get("endpoint") or producer_step.get("endpoint") or ""),
                "params": dict(call.get("params") or {}),
                "blocked_consumer_step_id": consumer_sid,
                "missing_binding": binding_name,
            }
    return None


def _run_binding_starvation_refresh(interp, benchmark, plan, logger, *, base_turn=40):
    """Perform at most one exact safe-GET refresh for a binding-starved action."""
    if not hasattr(interp, "execute"):
        return 0, [], "deterministic_host_unavailable"
    ledger = _read_ledger_from_kernel(interp)
    progress = _audit_ledger_execution(plan, ledger)
    target = _binding_starved_get_refresh_target(interp, plan, ledger, progress)
    if not target:
        return 0, [], "no_binding_starved_get"
    sid = str(target.get("plan_step_id") or "")
    if not sid:
        return 0, [], "no_binding_starved_get"
    _reset_kernel_plan_call_budget(interp, [sid])
    if hasattr(interp, "_run_silent"):
        interp._run_silent(
            "_SECAT_BINDING_REFRESHED_STEPS.append(" + repr(sid) + ") "
            "if " + repr(sid) + " not in _SECAT_BINDING_REFRESHED_STEPS else None",
            check=True)
    code = _deterministic_get_code(benchmark, target["endpoint"], target.get("params"))
    messages = [{"role": "system", "content":
                 "Deterministic refresh of one already validated safe GET whose "
                 "required downstream binding was empty."},
                {"role": "user", "content":
                 f"Repeat validated GET step {sid} exactly once to refresh binding "
                 f"{target.get('missing_binding')} for step "
                 f"{target.get('blocked_consumer_step_id')}."}]
    result = interp.execute(code)
    shown = (result.get("auto_display") or result.get("combined") or
             result.get("stdout") or "")[:1000]
    _log_execution(logger, base_turn, messages,
                   f"<execute>{code}</execute>", code, result,
                   error_type="binding_starvation_refresh", scope="S3_Execution")
    if not result.get("success", True):
        return 1, [shown], "binding_refresh_execution_failure"
    return 1, [shown], "binding_refresh_executed"


def _terminal_action_precondition(repair_record):
    """Return True when action repair proved execution is impossible in current state.

    This is deliberately narrow: only an explicit precondition_unsatisfied stop
    with a structured precondition record blocks whole-plan replanning.  Ordinary
    action gaps remain eligible for bounded strategy recovery.
    """
    record = repair_record or {}
    return bool(
        record.get("stop_reason") == "precondition_unsatisfied"
        and isinstance(record.get("precondition_unsatisfied"), dict)
        and record.get("precondition_unsatisfied")
    )


def _run_action_completion_repair(interp, benchmark: str, instruction: str, model: str,
                                  logger, plan: dict[str, Any], ledger, progress,
                                  phase_stdout: list[str], *, base_turn: int = 46):
    """Bounded recovery for an incomplete *already validated* action plan.

    This stage does not choose a new route, reinterpret the user request, or
    synthesize unplanned writes.  It reuses the existing trusted kernel/ledger and
    asks Phase A to execute only the plan instances that the host audit still marks
    missing.  The normal request guard remains authoritative, so a repair can only
    issue requests already licensed by the validated plan and observed bindings.

    Action tasks historically returned to certification before the generic
    evidence-repair block, which turned recoverable partial executions directly
    into ``action_gap``/safe abstention.  This helper closes that control-flow gap
    while keeping repair finite and fail-closed.
    """
    record = {
        "attempted": False,
        "mode": "remaining_action_steps",
        "code_runs": 0,
        "complete_before": bool((progress or {}).get("complete")),
        "complete_after": bool((progress or {}).get("complete")),
        "missing_before": [str(x.get("id") or "") for x in
                           ((progress or {}).get("missing_steps") or [])],
        "missing_after": [str(x.get("id") or "") for x in
                          ((progress or {}).get("missing_steps") or [])],
        "recovered": False,
        "stop_reason": None,
    }
    if not bool(getattr(config, "OCA_ACTION_COMPLETION_REPAIR", True)):
        record["stop_reason"] = "disabled"
        return ledger, progress, 0, [], 0, 0, record, None
    if not ((plan or {}).get("valid") and (plan or {}).get("execution_eligible")):
        record["stop_reason"] = "plan_not_executable"
        return ledger, progress, 0, [], 0, 0, record, None
    if (progress or {}).get("complete"):
        record["stop_reason"] = "already_complete"
        return ledger, progress, 0, [], 0, 0, record, None
    write_methods = {"POST", "PUT", "DELETE", "PATCH"}
    if not any(str(step.get("method") or "GET").upper() in write_methods
               for step in (plan or {}).get("steps") or []):
        record["stop_reason"] = "read_only_plan"
        return ledger, progress, 0, [], 0, 0, record, None

    # First complete any *read* prerequisite that is already uniquely determined
    # by the validated plan and trusted bindings, then materialize the write.  A
    # common partial-action shape is GET parent -> missing GET detail -> WRITE.
    # Asking the repair model to rediscover that detail call is unnecessary and
    # can trigger unplanned-call loops even though the host already has everything
    # needed to execute the validated route exactly.
    record["attempted"] = True
    preget_runs, preget_stdout, preget_stop = 0, [], None
    refresh_runs, refresh_stdout, refresh_stop = 0, [], None
    deterministic_stdout = []
    ne_pre = nr_pre = 0
    if hasattr(interp, "execute"):
        max_host = max(1, int(getattr(config, "OCA_ACTION_COMPLETION_REPAIR_STEPS", 3)))
        preget_runs, preget_stdout, preget_stop = _run_deterministic_plan_completion(
            interp, benchmark, plan, logger,
            max_calls=max_host,
            base_turn=max(1, base_turn - 5))
        if preget_runs:
            deterministic_stdout.extend(preget_stdout)
            phase_stdout = list(phase_stdout) + list(preget_stdout)
            ledger, ne_pre, nr_pre = _rebuild_ledger(interp, plan, phase_stdout)
            progress = _audit_ledger_execution(plan, ledger)

        # A successful safe GET can legitimately return a transient empty
        # collection. The audit then marks that producer call complete, while a
        # downstream path/query/body binding remains impossible to materialize.
        # Refresh only that exact already-authorized GET, once per producer, then
        # retry deterministic completion. Writes are never reopened here.
        refresh_limit = max_host
        while preget_stop == "no_unambiguous_host_target" and refresh_runs < refresh_limit:
            rr, rout, rstop = _run_binding_starvation_refresh(
                interp, benchmark, plan, logger,
                base_turn=max(1, base_turn - 4 + refresh_runs))
            refresh_stop = rstop
            if not rr:
                break
            refresh_runs += int(rr)
            refresh_stdout.extend(rout)
            deterministic_stdout.extend(rout)
            phase_stdout = list(phase_stdout) + list(rout)
            ledger, ne_pre, nr_pre = _rebuild_ledger(interp, plan, phase_stdout)
            progress = _audit_ledger_execution(plan, ledger)
            if progress.get("complete"):
                preget_stop = "plan_complete_after_binding_refresh"
                break
            more_runs, more_stdout, more_stop = _run_deterministic_plan_completion(
                interp, benchmark, plan, logger,
                max_calls=max_host,
                base_turn=max(1, base_turn - 2 + refresh_runs))
            preget_runs += int(more_runs)
            preget_stdout.extend(more_stdout)
            preget_stop = more_stop
            if more_runs:
                deterministic_stdout.extend(more_stdout)
                phase_stdout = list(phase_stdout) + list(more_stdout)
                ledger, ne_pre, nr_pre = _rebuild_ledger(interp, plan, phase_stdout)
                progress = _audit_ledger_execution(plan, ledger)
                if progress.get("complete"):
                    break
        record["deterministic_read_runs"] = int(preget_runs)
        record["deterministic_read_stop_reason"] = preget_stop
        record["binding_refresh_runs"] = int(refresh_runs)
        record["binding_refresh_stop_reason"] = refresh_stop
    else:
        preget_stop = "deterministic_host_unavailable"
        refresh_stop = "deterministic_host_unavailable"

    # Now finish a write whose exact route and request arguments are determined by
    # the validated plan + trusted ledger.  This avoids asking the repair model to
    # reconstruct host-side variables that it cannot reliably see.
    if hasattr(interp, "execute"):
        host_runs, host_stdout, host_stop = _run_deterministic_action_completion(
            interp, benchmark, plan, logger,
            max_calls=max(1, int(getattr(config, "OCA_ACTION_COMPLETION_REPAIR_STEPS", 3))),
            base_turn=max(1, base_turn - 3))
    else:
        # Some non-kernel/harness tests provide only the model-repair surface.
        # Production v2 uses KernelInterpreter; keep a graceful fallback rather
        # than treating absence of host execution introspection as ledger loss.
        host_runs, host_stdout, host_stop = 0, [], "deterministic_host_unavailable"
    if host_runs:
        phase_stdout = list(phase_stdout) + list(host_stdout)
        ledger, ne_host, nr_host = _rebuild_ledger(interp, plan, phase_stdout)
        progress = _audit_ledger_execution(plan, ledger)
        record["deterministic_action_runs"] = int(host_runs)
        record["deterministic_action_stop_reason"] = host_stop
        if progress.get("complete"):
            record.update({
                "complete_after": True,
                "missing_after": [],
                "recovered": True,
                "stop_reason": "deterministic_action_complete",
            })
            return (ledger, progress, int(preget_runs + refresh_runs + host_runs),
                    list(deterministic_stdout) + list(host_stdout),
                    int(ne_pre + ne_host), int(nr_pre + nr_host), record, None)

    # If a validated successful GET produced an actually empty collection that is
    # required by a downstream action, no amount of model repair can manufacture
    # the missing entity without hallucination.  Classify this as an environment/
    # state precondition instead of spending turns on unplanned-call loops.  We do
    # this only after the bounded exact refresh above, so transient emptiness still
    # gets one legitimate retry.
    try:
        from utils.plan_execution_audit import empty_required_binding_precondition
        precondition = empty_required_binding_precondition(plan, ledger, progress)
    except Exception:
        precondition = None
    if precondition:
        record.update({
            "complete_after": False,
            "missing_after": [str(x.get("id") or "") for x in
                              (progress.get("missing_steps") or [])],
            "recovered": False,
            "stop_reason": "precondition_unsatisfied",
            "precondition_unsatisfied": dict(precondition),
            "code_runs": int(preget_runs + refresh_runs + host_runs),
            "deterministic_action_runs": int(host_runs),
            "deterministic_action_stop_reason": host_stop,
        })
        return (ledger, progress, int(preget_runs + refresh_runs + host_runs),
                list(deterministic_stdout) + list(host_stdout), int(ne_pre), int(nr_pre),
                record, None)

    from utils.evidence_plan import missing_plan_prompt
    missing_prompt = missing_plan_prompt(progress)
    if not missing_prompt:
        record["stop_reason"] = host_stop or preget_stop or "no_missing_steps"
        return (ledger, progress, int(preget_runs + refresh_runs + host_runs),
                list(deterministic_stdout) + list(host_stdout), int(ne_pre), int(nr_pre), record, None)

    repair_meta: dict[str, Any] = {}
    system_prompt = (
        _build_phase_a_prompt(benchmark, plan)
        + "\n\nACTION COMPLETION REPAIR:\n"
          "The validated plan already partially executed. Preserve every trusted "
          "binding and completed plan instance. Execute ONLY unresolved plan "
          "instances. Do not restart the workflow, do not invent alternate routes, "
          "and do not repeat covered writes. Never introspect the interpreter namespace "
          "with globals(), locals(), vars(), dir(), inspect, or sys.modules; reference "
          "existing variables directly, or issue only the still-required validated GET. "
          "If a prerequisite GET appears in code, reuse the trusted replay supplied by "
          "the runtime rather than changing its arguments. A successful state-changing "
          "request must actually cross the "
          "guard and receive a success status before writing REPAIR_DONE.\n\n"
        + missing_prompt
    )
    max_steps = max(1, int(getattr(config, "OCA_ACTION_COMPLETION_REPAIR_STEPS", 3)))
    runs, extra_stdout = _run_fetch_round(
        interp, system_prompt,
        "Complete only the unresolved validated action-plan steps now.",
        model, logger, base_turn=base_turn, max_steps=max_steps,
        question=instruction, plan=plan, stage_name="action_completion_repair",
        progress_meta=repair_meta, stop_on_plan_complete=True)

    ledger2, ne2, nr2 = _rebuild_ledger(interp, plan, phase_stdout + list(extra_stdout))
    progress2 = _audit_ledger_execution(plan, ledger2)
    record.update({
        "code_runs": int(preget_runs) + int(refresh_runs) + int(host_runs) + int(runs),
        "deterministic_read_runs": int(preget_runs),
        "deterministic_read_stop_reason": preget_stop,
        "deterministic_action_runs": int(host_runs),
        "deterministic_action_stop_reason": host_stop,
        "complete_after": bool(progress2.get("complete")),
        "missing_after": [str(x.get("id") or "") for x in
                          (progress2.get("missing_steps") or [])],
        "recovered": bool(progress2.get("complete")),
        "stop_reason": repair_meta.get("stop_reason"),
        "stopped_no_progress": bool(repair_meta.get("stopped_no_progress")),
        "http_calls_observed": int(repair_meta.get("http_calls_observed", 0)),
        "guard_error_count": int(repair_meta.get("guard_error_count", 0)),
    })
    return (ledger2, progress2, int(preget_runs) + int(refresh_runs) + int(host_runs) + int(runs),
            list(deterministic_stdout) + list(host_stdout) + list(extra_stdout), ne2, nr2,
            record, repair_meta.get("fatal_error"))


def _lineage_completion_targets(plan, ledger, compiled):
    from utils.evidence_plan import resolve_lineage_completion_targets
    return resolve_lineage_completion_targets(plan, ledger, compiled)

def _deterministic_plan_step_targets(plan, ledger, progress, *, max_calls=6):
    """Construct exact missing GET requests from validated plan + observed bindings.

    This is deliberately conservative.  It does not choose routes or values.  A
    request is materialized only when the validated plan already fixes the
    method/endpoint/request literals and every dynamic value can be replayed
    unambiguously from the declared producer projection.  Ambiguous cases are
    left for the bounded model repair path.
    """
    import re
    import urllib.parse
    from utils.plan_execution_audit import authorized_binding_sequence, path_binding_name

    plan = plan or {}
    statuses = (progress or {}).get("step_status") or {}
    valid_map = {str(sid): set(str(x) for x in (st.get("valid_call_ids") or []))
                 for sid, st in statuses.items()}
    calls_by_id = {str(c.get("call_id") or ""): c
                   for c in getattr(ledger, "api_calls", []) or []}

    def request_signature(endpoint, params):
        # Ledger endpoints are stored on the wire surface. Convert the plan's
        # logical concrete path through the same generic base-path composer before
        # comparing, so already-covered fan-out instances are not proposed again.
        from utils.api_runtime import wire_endpoint_template
        wire = wire_endpoint_template(str((plan or {}).get("_runtime_base_url") or ""),
                                      str(endpoint or ""))
        return (str(wire), json.dumps(params or {}, sort_keys=True,
                                     ensure_ascii=False, default=str))

    def covered_signatures(step_id):
        out = set()
        status = statuses.get(str(step_id)) or {}
        for cid in status.get("valid_call_ids") or []:
            call = calls_by_id.get(str(cid))
            if not call:
                continue
            out.add((str(call.get("endpoint") or ""),
                     json.dumps(call.get("params") or {}, sort_keys=True,
                                ensure_ascii=False, default=str)))
        return out

    for step in (progress or {}).get("missing_steps") or []:
        sid = str(step.get("id") or "")
        if str(step.get("method") or "GET").upper() != "GET":
            continue
        # Read-only deterministic completion intentionally does not synthesize
        # request bodies.  State-changing/body-bearing actions remain model/
        # adapter controlled.
        if step.get("body_literals") or step.get("body_bindings"):
            continue
        template = str(step.get("endpoint") or "")
        placeholders = re.findall(r"\{([^{}]+)\}", template)
        values_by_name = {}
        producers = {}
        unresolved = False
        for name in placeholders:
            if name in (step.get("path_literals") or {}):
                val = (step.get("path_literals") or {}).get(name)
                seq, producer = ([str(val)] if val not in (None, "") else []), "__literal__"
            else:
                seq, producer, _source = authorized_binding_sequence(
                    plan, ledger, sid, path_binding_name(plan, sid, name), valid_call_ids_by_step=valid_map)
            if not seq:
                unresolved = True
                break
            values_by_name[name] = list(seq)
            producers[name] = producer
        if unresolved:
            continue

        # Query literals are already fixed by the plan. Query bindings are
        # accepted only when they replay to one value, or to the same correlated
        # fan-out length as the path values from the same producer.
        query_literals = dict(step.get("query_literals") or {})
        query_bound = {}
        query_meta = {}
        for target, binding_name in (step.get("query_bindings") or {}).items():
            seq, producer, _source = authorized_binding_sequence(
                plan, ledger, sid, str(binding_name), valid_call_ids_by_step=valid_map)
            if not seq:
                unresolved = True
                break
            query_bound[str(target)] = list(seq)
            query_meta[str(target)] = producer
        if unresolved:
            continue

        lengths = [len(v) for v in values_by_name.values()] + [len(v) for v in query_bound.values()]
        fanout = max(lengths or [1])
        multi_producers = set()
        for name, seq in values_by_name.items():
            if len(seq) > 1:
                multi_producers.add(producers.get(name))
        for target, seq in query_bound.items():
            if len(seq) > 1:
                multi_producers.add(query_meta.get(target))
        if len(multi_producers - {None}) > 1:
            # Multiple independently varying producers would require a Cartesian
            # product or a semantic join.  The host never guesses that relation.
            continue
        if any(len(seq) not in {1, fanout} for seq in list(values_by_name.values()) + list(query_bound.values())):
            continue

        # Required query parameters must already be represented by the validated
        # request contract; deterministic completion never invents defaults.
        def _required_flag(value):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"true", "1", "yes"}
            return bool(value) if isinstance(value, (int, float)) else False
        required_q = {str(p.get("name")) for p in (step.get("request_parameters") or [])
                      if str(p.get("in") or "").lower() == "query" and
                      _required_flag(p.get("required")) and p.get("name")}
        if required_q - (set(query_literals) | set(query_bound)):
            continue

        pending = []
        already_covered = covered_signatures(sid)
        for idx in range(fanout):
            endpoint = template
            concrete_bindings = {}
            for name, seq in values_by_name.items():
                value = seq[idx] if len(seq) > 1 else seq[0]
                endpoint = endpoint.replace("{" + name + "}", urllib.parse.quote(str(value), safe=""))
                concrete_bindings[name] = str(value)
            params = dict(query_literals)
            for target, seq in query_bound.items():
                params[target] = seq[idx] if len(seq) > 1 else seq[0]
            if request_signature(endpoint, params) in already_covered:
                continue
            pending.append({"plan_step_id": sid, "endpoint": endpoint,
                            "params": params, "source_bindings": concrete_bindings})

        if not pending:
            continue
        if len(pending) > max_calls:
            # Do not silently truncate a selected_all/completeness-sensitive set.
            # A later independent step may still be unambiguous, so continue the
            # search rather than emitting only a prefix of this one.
            continue
        # Return one dependency level at a time. All entries here belong to the
        # same already-authorized step and may be executed as one trusted host
        # batch; the ledger is re-audited before any dependent step is built.
        return pending
    return []


def _deterministic_action_step_targets(plan, ledger, progress, *, max_calls=8):
    """Materialize exact missing validated writes from trusted plan + ledger state.

    No route, entity, or argument is guessed here.  A target is emitted only when
    every dynamic binding can be replayed from a completed producer and all
    literals were already fixed by the validated plan.  This lets the host finish
    an authorized action even when the repair model cannot see/reconstruct a
    kernel variable name.
    """
    import re
    import urllib.parse
    from utils.plan_execution_audit import authorized_binding_sequence, path_binding_name

    plan = plan or {}
    statuses = (progress or {}).get("step_status") or {}
    valid_map = {str(sid): set(str(x) for x in (st.get("valid_call_ids") or []))
                 for sid, st in statuses.items()}
    completed = {str(x) for x in ((progress or {}).get("completed_steps") or [])}
    if not completed:
        completed = {str(sid) for sid, st in statuses.items()
                     if st.get("valid_call_ids") and not st.get("unresolved_placeholders")}
    write_methods = {"POST", "PUT", "DELETE", "PATCH"}
    plan_write_order = [str(x.get("id") or "") for x in (plan.get("steps") or [])
                        if str(x.get("method") or "GET").upper() in write_methods]
    missing_write_ids = {str(x.get("id") or "") for x in ((progress or {}).get("missing_steps") or [])
                         if str(x.get("method") or "GET").upper() in write_methods}

    def bound_sequence(sid, binding):
        seq, producer, _source = authorized_binding_sequence(
            plan, ledger, sid, str(binding), valid_call_ids_by_step=valid_map)
        return [str(x) for x in (seq or []) if x not in (None, "")], producer

    def query_value(step, target, seq):
        # Respect OAS type.  Legacy REST APIs commonly encode multi-ID query
        # bindings as one comma-separated string; true array parameters stay lists.
        param = next((x for x in (step.get("request_parameters") or [])
                      if str(x.get("in") or "").lower() == "query"
                      and str(x.get("name") or "") == str(target)), {})
        typ = str(param.get("type") or "").lower()
        if typ == "array":
            return list(seq)
        if len(seq) > 1:
            return ",".join(seq)
        return seq[0] if seq else None

    def body_value(step, target, seq):
        wrapper = str((step.get("body_binding_wrappers") or {}).get(target) or "")
        if wrapper == "uri_objects":
            return [{"uri": x} for x in seq]
        leafs = {str(x) for x in (step.get("request_body_leaf_paths") or [])}
        array_target = any(x == f"{target}[*]" or x.startswith(f"{target}[*].") for x in leafs)
        if array_target or len(seq) > 1:
            return list(seq)
        return seq[0] if seq else None

    def set_nested(obj, path, value):
        parts = [x for x in str(path).split(".") if x]
        if not parts:
            return
        cur = obj
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value

    targets = []
    for step in (progress or {}).get("missing_steps") or []:
        sid = str(step.get("id") or "")
        method = str(step.get("method") or "GET").upper()
        if method not in write_methods:
            continue
        if sid in plan_write_order:
            prior_writes = plan_write_order[:plan_write_order.index(sid)]
            # Do not have trusted host completion jump over an earlier missing
            # side effect merely because the later request happens to be easier
            # to materialize. The runtime guard enforces the same write order.
            if any(prev in missing_write_ids for prev in prior_writes):
                continue
        deps = [str(x) for x in (step.get("depends_on") or [])]
        # The auditor is authoritative; do not fire a side effect before all of
        # its explicit prerequisites have valid evidence.
        if any(dep not in completed and not (statuses.get(dep) or {}).get("valid_call_ids")
               for dep in deps):
            continue

        template = str(step.get("endpoint") or "")
        endpoint = template
        unresolved = False
        for name in re.findall(r"\{([^{}]+)\}", template):
            if name in (step.get("path_literals") or {}):
                seq = [str((step.get("path_literals") or {})[name])]
            else:
                seq, _producer = bound_sequence(sid, path_binding_name(plan, sid, name))
            if len(seq) != 1:
                unresolved = True; break
            endpoint = endpoint.replace("{" + name + "}",
                                        urllib.parse.quote(str(seq[0]), safe=""))
        if unresolved:
            continue

        params = dict(step.get("query_literals") or {})
        for target, binding in (step.get("query_bindings") or {}).items():
            seq, _producer = bound_sequence(sid, binding)
            if not seq:
                unresolved = True; break
            params[str(target)] = query_value(step, target, seq)
        if unresolved:
            continue

        body = {}
        for target, value in (step.get("body_literals") or {}).items():
            set_nested(body, str(target), value)
        for target, binding in (step.get("body_bindings") or {}).items():
            seq, _producer = bound_sequence(sid, binding)
            if not seq:
                unresolved = True; break
            set_nested(body, str(target), body_value(step, str(target), seq))
        if unresolved:
            continue

        # Required fields must already be represented. Never invent a missing
        # literal merely to make the endpoint accept the request.
        required_q = {str(x.get("name")) for x in (step.get("request_parameters") or [])
                      if str(x.get("in") or "").lower() == "query"
                      and bool(x.get("required")) and x.get("name")}
        if required_q - set(params):
            continue
        # Some historical/provider OAS cards expose leaf hints in
        # request_body_required_fields even when the action actually carries the
        # same semantic argument in the query string (for example provider ids).
        # Mirror runtime_execution_rules(): enforce required body fields only when
        # this validated step actually has a body contract or declares the body
        # itself required.  Never reject an otherwise exact query-bound action
        # merely because an unrelated schema leaf hint names the same field.
        has_body_contract = bool(step.get("body_literals") or step.get("body_bindings") or
                                 step.get("request_body_required"))
        required_body = (set(str(x) for x in (step.get("request_body_required_fields") or []))
                         if has_body_contract else set())
        if required_body and any(str(x).split(".",1)[0] not in body for x in required_body):
            continue

        targets.append({"plan_step_id": sid, "method": method,
                        "endpoint": endpoint, "params": params,
                        "json": body if body else None})
        if len(targets) >= max_calls:
            break
        # Re-audit after one side effect before materializing a dependent one.
        return targets
    return targets


def _deterministic_request_code(benchmark, target):
    """Build one exact host-owned request that still crosses the normal guard."""
    import urllib.parse
    import benchmarks as B
    from utils.api_runtime import wire_endpoint_template
    spec = B.get_benchmark(benchmark)
    base = spec["base_url"]
    parsed = urllib.parse.urlsplit(base)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    endpoint = str(target.get("endpoint") or "")
    url = origin + wire_endpoint_template(base, endpoint)
    method = str(target.get("method") or "GET").lower()
    params = dict(target.get("params") or {})
    body = target.get("json")
    lines = [f"_url = {url!r}", f"_params = {params!r}"]
    if body is not None:
        lines.append(f"_body = {body!r}")
        lines.append(f"_resp = requests.{method}(_url, params=_params, json=_body)")
    else:
        lines.append(f"_resp = requests.{method}(_url, params=_params)")
    lines += ["print(_resp.status_code)",
              "print((_resp.text or '')[:500] if hasattr(_resp, 'text') else '')"]
    return "\n".join(lines)


def _run_deterministic_action_completion(interp, benchmark, plan, logger, *,
                                         max_calls=8, base_turn=44):
    """Finish unambiguous validated action steps before asking an LLM to repair."""
    runs, stdout, stop_reason = 0, [], None
    attempted = set()
    while runs < max_calls:
        ledger = _read_ledger_from_kernel(interp)
        progress = _audit_ledger_execution(plan, ledger)
        if progress.get("complete"):
            break
        targets = _deterministic_action_step_targets(
            plan, ledger, progress, max_calls=max_calls-runs)
        if not targets:
            stop_reason = "no_unambiguous_action_target"; break
        target = targets[0]
        # A request rejected with a known 4xx did not satisfy the action step but
        # historically consumed its one-call guard slot, making a corrected
        # payload impossible. Permit one same-plan retry only when every prior
        # write for this step has a concrete 4xx status and none succeeded.
        _target_sid = str(target.get("plan_step_id") or "")
        _prior = [c for c in (ledger.api_calls or [])
                  if str(c.get("plan_step_id") or c.get("runtime_step_id") or "") == _target_sid
                  and str(c.get("method") or "GET").upper() in {"POST", "PUT", "DELETE", "PATCH"}]
        if (_prior and not any(isinstance(c.get("status_code"), int)
                               and 200 <= c.get("status_code") < 300 for c in _prior)
                and all(isinstance(c.get("status_code"), int)
                        and 400 <= c.get("status_code") < 500 for c in _prior)):
            _reset_kernel_plan_call_budget(interp, [_target_sid])
        signature = (str(target.get("plan_step_id") or ""),
                     str(target.get("method") or ""), str(target.get("endpoint") or ""),
                     json.dumps(target.get("params") or {}, sort_keys=True, default=str),
                     json.dumps(target.get("json"), sort_keys=True, default=str))
        if signature in attempted:
            stop_reason = "action_target_repeated_without_progress"; break
        attempted.add(signature)
        code = _deterministic_request_code(benchmark, target)
        messages = [{"role": "system", "content":
                     "Deterministic completion of one already validated action step."},
                    {"role": "user", "content":
                     f"Execute unresolved plan step {target.get('plan_step_id')} exactly as validated."}]
        result = _execute_generated_code(interp, code)
        shown = (result.get("auto_display") or result.get("combined") or
                 result.get("stdout") or "")[:1200]
        stdout.append(shown)
        _log_execution(logger, base_turn + runs, messages,
                       f"<execute>{code}</execute>", code, result,
                       error_type="deterministic_action_completion", scope="S3_Execution")
        runs += 1
        if not result.get("success", True):
            stop_reason = "action_completion_execution_failure"; break
    return runs, stdout, stop_reason


def _deterministic_get_code(benchmark, endpoint, params=None):
    """Build an exact GET action; runtime transport injects authentication."""
    import urllib.parse
    import benchmarks as B
    spec = B.get_benchmark(benchmark)
    base = spec["base_url"]
    parsed = urllib.parse.urlsplit(base)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    from utils.api_runtime import wire_endpoint_template
    wire_endpoint = wire_endpoint_template(base, endpoint)
    url = origin + wire_endpoint
    lines = [f"_url = {url!r}", f"_params = {dict(params or {})!r}",
             "_resp = requests.get(_url, params=_params)",
             "print(_resp.status_code)"]
    return "\n".join(lines)


def _deterministic_get_batch_code(benchmark, targets):
    """Build one trusted execution containing a finite set of exact GET requests.

    Every target was independently materialized from validated plan bindings by
    ``_deterministic_plan_step_targets``. Batching changes only interpreter/code-run
    overhead; each HTTP request still crosses the normal request guard, is captured
    separately in the ledger, and counts against the global API-call budget.
    """
    import urllib.parse
    import benchmarks as B
    from utils.api_runtime import wire_endpoint_template
    spec = B.get_benchmark(benchmark)
    base = spec["base_url"]
    parsed = urllib.parse.urlsplit(base)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    batch = []
    for target in targets or []:
        wire_endpoint = wire_endpoint_template(base, str(target.get("endpoint") or ""))
        batch.append({"url": origin + wire_endpoint,
                      "params": dict(target.get("params") or {})})
    lines = [f"_secat_batch = {batch!r}",
             "for _item in _secat_batch:",
             "    _resp = requests.get(_item['url'], params=_item['params'])",
             "    print(_resp.status_code)"]
    return "\n".join(lines)


def _run_deterministic_frontend_host_first(interp, benchmark, question, plan, logger, *,
                                           max_batches=30, max_fanout=20, base_turn=20):
    """Execute a deterministic-front-end plan one dependency level at a time.

    The important difference from generic host completion is *when* observed-data
    normalization runs: after each acquired dependency level and before any child
    request is materialized.  A named search therefore gets its canonical owner
    selection repaired before the selected ID can propagate into credits/details.

    Repairs are deliberately bounded to deterministic observed-data operations
    already used elsewhere in OCA; this helper never chooses a new route.
    """
    runs, stdout = 0, []
    stop_reason = None
    attempted: set[tuple[str, str, str]] = set()
    repair_log = {"canonical_search": [], "relation": [], "order": []}

    while runs < max_batches:
        ledger = _read_ledger_from_kernel(interp)

        # Normalize evidence-dependent selection semantics *before* computing the
        # next child binding. Mutate the caller-owned plan in place so every later
        # verifier/ledger stage sees exactly the graph that was executed.
        repaired, recs = _apply_observed_canonical_search_repairs(interp, plan, ledger)
        if repaired is not plan:
            plan.clear(); plan.update(repaired)
        repair_log["canonical_search"].extend(recs or [])

        repaired, recs = _apply_observed_relation_repairs(interp, question, plan, ledger)
        if repaired is not plan:
            plan.clear(); plan.update(repaired)
        repair_log["relation"].extend(recs or [])

        repaired, recs = _apply_observed_order_repairs(interp, question, plan, ledger)
        if repaired is not plan:
            plan.clear(); plan.update(repaired)
        repair_log["order"].extend(recs or [])

        progress = _audit_ledger_execution(plan, ledger)
        if progress.get("complete"):
            break
        if not plan.get("valid") or not plan.get("execution_eligible"):
            stop_reason = "frontend_plan_became_non_executable"
            break

        targets = _deterministic_plan_step_targets(
            plan, ledger, progress, max_calls=max_fanout)
        if not targets:
            stop_reason = "no_unambiguous_host_target"
            break

        batch, signatures = [], []
        for target in targets:
            signature = (str(target.get("plan_step_id") or ""),
                         str(target.get("endpoint") or ""),
                         json.dumps(target.get("params") or {}, sort_keys=True, default=str))
            if signature in attempted:
                continue
            batch.append(target); signatures.append(signature)
        if not batch:
            stop_reason = "host_completion_repeated_without_progress"
            break
        attempted.update(signatures)

        code = (_deterministic_get_code(benchmark, batch[0]["endpoint"], batch[0].get("params"))
                if len(batch) == 1 else _deterministic_get_batch_code(benchmark, batch))
        step_id = str(batch[0].get("plan_step_id") or "")
        messages = [{"role": "system", "content":
                     "Deterministic OCA execution of an already validated semantic plan."},
                    {"role": "user", "content":
                     f"Execute plan step {step_id} exactly as validated"
                     + (f" for {len(batch)} authorized binding values." if len(batch) > 1 else ".")}]
        result = interp.execute(code)
        shown = (result.get("auto_display") or result.get("combined") or
                 result.get("stdout") or "")[:1000]
        stdout.append(shown)
        _log_execution(logger, base_turn + runs, messages,
                       f"<execute>{code}</execute>", code, result,
                       error_type="deterministic_frontend_host", scope="S3_Execution")
        runs += 1
        if not result.get("success", True):
            stop_reason = "host_completion_execution_failure"
            break

    return runs, stdout, stop_reason, repair_log


def _run_deterministic_plan_completion(interp, benchmark, plan, logger, *, max_calls=6,
                                       base_turn=37):
    """Realize unambiguous missing validated GET steps without another LLM call."""
    runs, stdout = 0, []
    stop_reason = None
    attempted: set[tuple[str, str, str]] = set()
    while runs < max_calls:
        ledger = _read_ledger_from_kernel(interp)
        progress = _audit_ledger_execution(plan, ledger)
        if progress.get("complete"):
            break
        targets = _deterministic_plan_step_targets(
            plan, ledger, progress, max_calls=max_calls)
        if not targets:
            stop_reason = "no_unambiguous_host_target"
            break
        # Execute all still-unattempted requests for this one already-authorized
        # plan step, then re-audit before constructing any dependent request.
        batch = []
        batch_signatures = []
        for target in targets:
            signature = (str(target.get("plan_step_id") or ""),
                         str(target.get("endpoint") or ""),
                         json.dumps(target.get("params") or {}, sort_keys=True, default=str))
            if signature in attempted:
                continue
            batch.append(target)
            batch_signatures.append(signature)
        if not batch:
            stop_reason = "host_completion_repeated_without_progress"
            break
        attempted.update(batch_signatures)
        code = (_deterministic_get_code(benchmark, batch[0]["endpoint"], batch[0].get("params"))
                if len(batch) == 1 else _deterministic_get_batch_code(benchmark, batch))
        step_id = str(batch[0].get("plan_step_id") or "")
        messages = [{"role": "system", "content":
                     "Deterministic OCA completion of an already validated acquisition step."},
                    {"role": "user", "content":
                     f"Execute plan step {step_id} exactly as validated"
                     + (f" for {len(batch)} authorized binding values." if len(batch) > 1 else ".")}]
        result = interp.execute(code)
        shown = (result.get("auto_display") or result.get("combined") or
                 result.get("stdout") or "")[:1000]
        stdout.append(shown)
        _log_execution(logger, base_turn + runs, messages,
                       f"<execute>{code}</execute>", code, result,
                       error_type="deterministic_plan_completion", scope="S3_Execution")
        runs += 1
        # Guard/transport failures remain visible in the kernel output; stop
        # rather than trying alternate arguments.
        if not result.get("success", True):
            stop_reason = "host_completion_execution_failure"
            break
    return runs, stdout, stop_reason


def _run_lineage_completion(interp, benchmark, targets, logger, base_turn=35):
    runs, stdout = 0, []
    for offset, target in enumerate(targets):
        code = _deterministic_get_code(benchmark, target["endpoint"])
        messages = [{"role": "system", "content":
                     "Deterministic OCA lineage completion from a certified selection."},
                    {"role": "user", "content":
                     f"Fetch exact endpoint {target['endpoint']} for plan step "
                     f"{target['plan_step_id']}."}]
        result = interp.execute(code)
        # Do not mutate ledger provenance after execution. The trusted request
        # guard assigns runtime_step_id only when the concrete request satisfies
        # the validated plan; host audit then derives plan completion from that.
        shown = (result.get("auto_display") or result.get("combined") or
                 result.get("stdout") or "")[:6000]
        stdout.append(shown)
        _log_execution(logger, base_turn + offset, messages,
                       f"<execute>{code}</execute>", code, result,
                       error_type="lineage_completion", scope="S5_Integration")
        runs += 1
    return runs, stdout


def _structural_fallback_targets(question, plan, ledger):
    from utils.evidence_plan import structural_fallback_targets
    return structural_fallback_targets(question, plan, ledger)


def _rebuild_ledger(interp, plan, phase_stdout):
    ledger = _read_ledger_from_kernel(interp)
    ledger.annotate_plan_steps(plan)
    # A deterministic lineage-completion call supersedes any earlier call that
    # used the same endpoint template with the wrong entity id.
    forced_by_step = {
        str(call.get("plan_step_id")): call.get("call_id")
        for call in ledger.api_calls if call.get("forced_plan_step") and
        call.get("plan_step_id")
    }
    if forced_by_step:
        for call in ledger.api_calls:
            sid = str(call.get("plan_step_id") or "")
            if sid in forced_by_step and call.get("call_id") != forced_by_step[sid]:
                call.pop("plan_step_id", None)
        for obs in ledger.observations:
            sid = str(obs.get("plan_step_id") or "")
            if sid in forced_by_step and obs.get("call_id") != forced_by_step[sid]:
                obs.pop("plan_step_id", None)
        for raw in ledger.raw_responses:
            sid = str(raw.get("plan_step_id") or "")
            if sid in forced_by_step and raw.get("call_id") != forced_by_step[sid]:
                raw.pop("plan_step_id", None)
    explicit = _read_derived_into(interp, ledger)
    rpr = 0
    try:
        from utils.observation_ledger import reconstruct_provenance
        before = len(ledger.derived)
        reconstruct_provenance(ledger, "\n".join(phase_stdout))
        rpr = len(ledger.derived) - before
    except Exception as exc:
        print(f"[WARN] RPR failed: {exc}")
    return ledger, explicit, rpr


def _audit_ledger_execution(plan, ledger):
    """Host-side replay of plan completion and concrete binding lineage."""
    # Validity dominates coverage.  In particular, an invalid action plan that
    # compiled to zero executable steps must never become vacuously complete.
    if (plan or {}).get("valid") is False:
        return {"complete": False, "missing_steps": list((plan or {}).get("steps") or []),
                "completed_step_ids": [], "step_status": {}, "lineage_errors": [],
                "plan_invalid": True,
                "advisory": str((plan or {}).get("_execution_policy") or "strict").lower() == "advisory"}
    if not (plan or {}).get("steps"):
        return {"complete": True, "missing_steps": [], "completed_step_ids": [],
                "step_status": {}, "lineage_errors": []}
    from utils.plan_execution_audit import audit_execution, apply_audited_annotations
    audit = audit_execution(plan, ledger)
    apply_audited_annotations(plan, ledger, audit)
    # Execution coverage cannot turn a semantically invalid plan into a complete
    # one.  An invalid compiler output may contain a runnable read-only fragment;
    # reporting that fragment as plan_complete=True is misleading and caused
    # quality reports to hide missing typed action obligations.
    if (plan or {}).get("valid") is False:
        audit["complete"] = False
        audit["plan_invalid"] = True
    audit["advisory"] = str((plan or {}).get("_execution_policy") or "strict").lower() == "advisory"
    return audit


def _apply_plan_gate(verification, plan, progress, planner_enabled=True):
    """Keep strict plan semantics as a final certification obligation.

    Safe read-only advisory plans may acquire evidence and reach this gate, but
    unresolved strict plan semantics still prevent unsupported emission.
    """
    vr = dict(verification or {})
    vr["missing_slots"] = list(vr.get("missing_slots") or [])
    vr["notes"] = list(vr.get("notes") or [])
    vr["checks"] = dict(vr.get("checks") or {})
    if not planner_enabled:
        return vr
    vr["checks"]["plan_execution_eligible"] = bool((plan or {}).get("execution_eligible"))
    if not (plan or {}).get("valid"):
        vr["certificate_accepted"] = False
        vr["contract_satisfied"] = False
        vr["verification_status"] = "evidence_gap"
        if "plan_valid" not in vr["missing_slots"]:
            vr["missing_slots"].append("plan_valid")
        if (plan or {}).get("execution_eligible"):
            vr["notes"].append(
                "evidence acquisition continued under a read-only advisory plan, but "
                "strict plan semantics remain unresolved")
        else:
            vr["notes"].append("the OAS-grounded evidence plan was non-executable")
        vr["checks"]["plan_valid"] = False
        return vr
    vr["checks"]["plan_valid"] = True
    vr["checks"]["plan_advisory"] = str((plan or {}).get("_execution_policy") or "strict").lower() == "advisory"
    complete = bool((progress or {}).get("complete"))
    vr["checks"]["plan_complete"] = complete
    if not complete:
        vr["certificate_accepted"] = False
        vr["contract_satisfied"] = False
        vr["verification_status"] = "evidence_gap"
        if "answer_lineage" not in vr["missing_slots"]:
            vr["missing_slots"].append("answer_lineage")
        missing = [s.get("id") for s in (progress or {}).get("missing_steps", [])]
        vr["notes"].append(f"evidence plan remains incomplete: {missing}")
    return vr




def _finalize_action(task, instruction, model, logger, interp, ledger, plan,
                             progress, code_runs, n_explicit, n_rpr, token_read,
                             action_completion_repair=None, plan_repair_code_runs=0,
                             action_runtime_replan=None):
    """Finalize an observed state-changing API action through a configured adapter.

    Read-only requests never enter this path; they use the same generic OCA
    compiler/contract. The adapter contributes only action-result certification,
    not benchmark routes or expected answers.
    """
    from utils.api_runtime import action_certificate
    benchmark = _benchmark(task)
    repair_record = (dict(action_completion_repair or {})
                     if bool((action_completion_repair or {}).get("attempted")) else {})
    cert = action_certificate(benchmark, ledger, plan=plan)
    accepted = bool(cert["accepted"] and plan.get("valid") and progress.get("complete"))
    precondition = (repair_record.get("precondition_unsatisfied")
                    if isinstance(repair_record, dict) else None)
    final_answer = ("Completed the requested action."
                    if accepted else
                    "The requested action could not be completed because a required "
                    "source collection was empty in the current service state."
                    if precondition else
                    "Execution exhausted before every requested action step could "
                    "be completed and verified.")
    cited = cert["cited_observation_ids"] if accepted else []
    rejected_status = "precondition_unsatisfied" if precondition else "action_gap"
    verification = {
        "certificate_accepted": accepted,
        "contract_satisfied": accepted,
        "verification_status": cert["verification_status"] if accepted else rejected_status,
        "checks": {
            "plan_valid": bool(plan.get("valid")),
            "plan_complete": bool(progress.get("complete")),
            "observed_write_set_complete": cert["complete"],
            "all_write_calls_successful": not cert["failed_calls"],
            "action_observations_present": bool(cert["cited_observation_ids"]),
            "identifier_lineage": cert.get("identifier_lineage_ok", False),
        },
        "missing_slots": ([] if accepted else
                          ["runtime_precondition"] if precondition else
                          ["successful_observed_action"]),
        "notes": [],
    }
    log_no_code_turn(
        logger=logger, turn_num=98,
        messages=[{"role": "system", "content": "Deterministic observed-action certificate"}],
        llm_output=json.dumps({"final_answer": final_answer, **cert},
                              ensure_ascii=False, default=str),
        error_type="action_certificate", scope="S5_Integration", is_error=False)

    graph = EvidenceGraph(task.get("id", "?"), instruction, benchmark=benchmark)
    graph.finalize(ledger, final_answer, cited)
    try:
        graph_path = os.path.join(os.path.dirname(config.RUN_DIR), "evidence_graph.jsonl")
        graph.append_jsonl(ledger, graph_path)
    except Exception as exc:
        print(f"[WARN] evidence graph write failed: {exc}")

    api_call_accounting = _api_call_accounting(interp, ledger)
    try:
        uncertain_writes = _kernel_uncertain_writes(interp)
    except Exception:
        uncertain_writes = [{"state": "unreadable"}]
    logger.log["summary"].update({
        "oca_version": 2, "oca_build_id": OCA_BUILD_ID,
        "action_mode": True,
        "plan": plan, "plan_progress": progress,
        "executed_routes": _executed_route_templates(plan, ledger),
        "observations": len(ledger), **api_call_accounting,
        "derived_explicit": n_explicit, "derived_rpr": n_rpr,
        "cited_observation_ids": cited, "grounded": graph.grounded,
        "tokens": token_read(), "contract": {
            "version": 2, "build_id": OCA_BUILD_ID, "mode": "enforced",
            "task_kind": "action", "answer_kind": "action",
            "certificate_accepted": verification["certificate_accepted"],
            "contract_satisfied": verification["contract_satisfied"],
            "verification_status": verification["verification_status"],
            "checks": verification["checks"],
            "missing_slots": verification["missing_slots"],
            "certificate": cert,
            "repair": repair_record,
            "value_locked_emission": accepted,
        },
        "action_completion_repair": dict(action_completion_repair or {}),
        "action_runtime_replan": dict(action_runtime_replan or {}),
        "precondition_unsatisfied": dict(precondition or {}),
        "uncertain_write_outcomes": uncertain_writes,
        "plan_repair_code_runs": int(plan_repair_code_runs or 0),
        # Compatibility fields remain present for old analysis scripts, but an
        # exhausted action attempt is not represented as an abstention/refusal.
        "safe_abstention": False, "explicit_abstention": False,
        "action_incomplete": not accepted,
        "recovery_exhausted": bool(not accepted and not precondition),
        "uncertified_answer_emitted": False,
    })
    logger.finalize(final_answer=final_answer, success=bool(accepted),
                    silent_failure=False)
    logger.save()
    try:
        interp.shutdown()
    except Exception:
        pass
    return logger.log["summary"]


def _answer_repair_is_useful(verification):
    """Retry wording only when the evidence/derivation graph is already sound.

    Re-verbalization cannot manufacture a missing deterministic derivation or fix a
    lineage contradiction. Restricting the retry to surface/certificate formatting
    avoids an expensive second Phase-B call for structural failures.
    """
    slots = set((verification or {}).get("missing_slots") or [])
    if not slots:
        return False
    structural = {"derivations", "answer_lineage", "count_derivation",
                  "boolean_derivation", "comparison_derivation",
                  "plan_valid", "plan_complete", "transport_auth"}
    if slots & structural:
        return False
    # A semantic-commit rejection is not a wording/certificate formatting gap.
    # Re-running Phase B with exactly the same evidence usually repeats the same
    # disagreement while adding thousands of tokens.  Surface repair remains for
    # genuine certificate/value/citation formatting defects only.
    return bool(slots & {"answer_values", "answer_surface", "citations",
                         "answer_observation_ids"})

def _deterministic_evidence_salvage(question, ledger, plan, progress, compiled, verification):
    """Recover a strong evidence-backed answer from a locally broken read plan.

    This is deliberately *not* strict certification.  It exists for the recurring
    case where acquisition and deterministic replay succeeded but plan/certificate
    representation remained locally invalid.  We preserve that distinction in the
    run metadata instead of turning a representation failure into a user-visible
    refusal or pretending the original plan was valid.

    Salvage is intentionally conservative:
      * read-only evidence only;
      * only local-plan/certificate closure failure types;
      * no high-confidence semantic/observed route risk;
      * the answer-producing derivation and all of its step ancestors completed;
      * no known terminal-owner contradiction;
      * a fresh question-shape contract must verify the deterministic certificate;
      * question-only answer-surface checks must pass.
    """
    record = {"attempted": False, "adopted": False, "reason": None}
    steps = list((plan or {}).get("steps") or [])
    if not steps or any(str(x.get("method") or "GET").upper() not in {"GET", "HEAD"}
                        for x in steps):
        record["reason"] = "not_read_only"
        return None, [], None, record
    answer_ids = [str(x) for x in (compiled or {}).get("answer_derivation_ids") or []]
    if not answer_ids:
        record["reason"] = "no_compiled_answer_derivations"
        return None, [], None, record
    record["attempted"] = True

    feedback = _runtime_evidence_repair_feedback(
        plan, compiled, question=question, ledger=ledger, verification=verification)
    failure_type = str(feedback.get("failure_type") or "")
    record["failure_type"] = failure_type
    if failure_type not in {"local_plan_closure", "certificate_or_lineage_closure"}:
        record["reason"] = "strategy_or_evidence_failure"
        return None, [], None, record

    semantic_risks = _runtime_plan_semantic_risks(question, plan, ledger)
    if semantic_risks:
        record["reason"] = "semantic_risk"
        record["semantic_risks"] = semantic_risks[:8]
        return None, [], None, record

    # A known owner contradiction is semantic, not formatting/closure. Never
    # bypass it.  Other closure checks can be regenerated from the compiled DAG.
    checks = dict((verification or {}).get("checks") or {})
    if checks.get("terminal_entity_ownership") is False:
        record["reason"] = "terminal_owner_contradiction"
        return None, [], None, record

    obs_index = {str(o.get("obs_id")): o for o in getattr(ledger, "observations", []) or []}
    der_index = {str(d.get("obs_id")): d for d in getattr(ledger, "derived", []) or []}
    if any(did not in der_index for did in answer_ids):
        record["reason"] = "answer_derivation_missing_from_ledger"
        return None, [], None, record

    # Require completion of the concrete answer-producing evidence DAG, even
    # when an unrelated/redundant plan step remains incomplete.
    if not _answer_evidence_lineage_complete(plan, progress, compiled, ledger):
        record["reason"] = "answer_lineage_execution_incomplete"
        return None, [], None, record

    try:
        from utils.evidence_contract import (
            classify_task, certificate_from_compiled, verify_contract,
            materialize_certified_answer,
        )
        from utils.accuracy_semantics import answer_surface_risks

        # Question shape wins whenever it is explicit. The plan may refine a
        # coarse direct question into comparison/asset only after semantic-risk
        # screening above has accepted that interpretation.
        salvage_contract = classify_task(question)
        q_mode = str(salvage_contract.get("answer_kind") or "direct").lower()
        plan_mode = str((plan or {}).get("answer_mode") or "").lower()
        if q_mode == "direct" and plan_mode in {
                "direct", "list", "count", "boolean", "comparison", "asset"}:
            salvage_contract["answer_kind"] = plan_mode
            salvage_contract["task_kind"] = plan_mode

        answer, cert = certificate_from_compiled(
            salvage_contract, compiled, obs_index, der_index, plan=plan)
        if not answer or not cert:
            record["reason"] = "deterministic_materialization_unavailable"
            return None, [], None, record
        vr = verify_contract(
            salvage_contract, cert, obs_index, der_index, answer, plan=None)
        if not (vr.get("certificate_accepted") and vr.get("contract_satisfied")):
            record["reason"] = "question_contract_rejected"
            record["verification"] = vr
            return None, [], None, record
        surface_risks = answer_surface_risks(question, answer, salvage_contract)
        if surface_risks:
            record["reason"] = "answer_surface_risk"
            record["surface_risks"] = surface_risks[:8]
            return None, [], None, record

        locked = materialize_certified_answer(
            salvage_contract, cert, der_index, fallback=answer,
            obs_index=obs_index, plan=None)
        vr2 = verify_contract(
            salvage_contract, cert, obs_index, der_index, locked, plan=None)
        if not locked or not (vr2.get("certificate_accepted") and vr2.get("contract_satisfied")):
            record["reason"] = "value_lock_rejected"
            return None, [], None, record
        surface_risks = answer_surface_risks(question, locked, salvage_contract)
        if surface_risks:
            record["reason"] = "locked_answer_surface_risk"
            record["surface_risks"] = surface_risks[:8]
            return None, [], None, record

        cited = [str(x) for x in cert.get("cited_observation_ids") or []]
        record.update({
            "adopted": True,
            "reason": "local_closure_bypassed_with_question_verified_deterministic_evidence",
            "answer_kind": salvage_contract.get("answer_kind"),
            "cited_observation_ids": cited,
            "strict_missing_slots": list((verification or {}).get("missing_slots") or []),
        })
        return locked, cited, cert, record
    except Exception as exc:
        record["reason"] = f"salvage_exception:{type(exc).__name__}"
        return None, [], None, record


def _best_effort_evidence_answer(question, ledger, plan, compiled, model, logger, *, turn_num=99):
    """Last-resort evidence-only answer after all structured recovery is exhausted.

    This deliberately does not pretend to be certified. It sees no benchmark gold
    and is forbidden from using prior factual knowledge. Its purpose is to avoid
    replacing a plausible grounded answer with a generic refusal on answer-centric
    benchmarks where abstention is always scored wrong.
    """
    # This path runs only after normal focused compilation/recovery failed. Give
    # it a broader compact ledger so an earlier mistaken selector cannot hide an
    # already-retrieved alternative candidate. Derived evidence is still placed
    # first by ObservationLedger.serialize().
    ledger_text = ledger.serialize(
        max_chars=int(getattr(config, "OCA_BEST_EFFORT_LEDGER_CHARS", 12000)),
        focus_obs_ids=None)
    payload = {
        "question": question,
        "answer_mode": (plan or {}).get("answer_mode"),
        "answer_requirements": (plan or {}).get("answer_requirements") or [],
        "plan_steps": (plan or {}).get("steps") or [],
        "derivations": (plan or {}).get("derivations") or [],
        "compiler_warnings": list((compiled or {}).get("warnings") or [])[:12],
    }
    messages = [
        {"role": "system", "content": (
            "Give the best answer supported by the supplied evidence. Do not use prior factual knowledge or guess. "
            "Keep the meaning of the user's question and include only supported information. Keep the answer short. "
            "Return JSON only: {\"final_answer\":\"minimal answer\",\"evidence_ids\":[\"obs_...\"]}."
        )},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)[
            :int(getattr(config, "OCA_BEST_EFFORT_PAYLOAD_CHARS", 5000))] +
         "\n\nLEDGER:\n" + ledger_text},
    ]
    response = _chat_json(messages, model,
                          max_tokens=int(getattr(config, "OCA_BEST_EFFORT_MAX_TOKENS", 300)),
                          stage_name="best_effort_answer")
    output = response.choices[0].message.content or ""
    log_no_code_turn(logger=logger, turn_num=turn_num, messages=messages,
                     llm_output=output, error_type="best_effort_answer",
                     scope="S5_Integration", is_error=False)
    obj = {}
    match = re.search(r"\{.*\}", output, re.S)
    if match:
        try: obj = json.loads(match.group(0))
        except Exception: obj = {}
    answer = str(obj.get("final_answer") or "").strip()
    ids = [str(x) for x in obj.get("evidence_ids") or []]
    valid_ids = {o.get("obs_id") for o in ledger.observations} | {d.get("obs_id") for d in ledger.derived}
    ids = [x for x in ids if x in valid_ids]
    # Never emit a model-only answer with zero evidence anchors.
    if not answer or not ids:
        return "", []
    try:
        from utils.accuracy_semantics import answer_surface_risks
        if answer_surface_risks(question, answer, {"answer_kind": (plan or {}).get("answer_mode")}):
            return "", []
    except Exception:
        pass
    return answer, ids


def run(task: dict, model: str = config.DEFAULT_MODEL) -> dict:
    version = str(getattr(config, "OCA_VERSION", "v2")).lower()
    if version != "v2":
        raise ValueError("This minimal artifact includes only OCA v2.")

    instruction = task.get("instruction") or task.get("query") or ""
    if not task.get("instruction"):
        task = {**task, "instruction": instruction}
    benchmark = _benchmark(task)
    logger = TaskLogger(agent_name="oca", task=task, model=model)
    # The isolated task intentionally omits provider/benchmark identity; keep it
    # in experiment metadata from trusted runtime configuration only.
    logger.log["meta"]["benchmark"] = benchmark
    logger.log["meta"]["oca_build_id"] = OCA_BUILD_ID
    from utils.token_meter import reset as token_reset, read as token_read
    token_reset()

    print("\n" + "=" * 68)
    print(f"[OCA v2 / {OCA_BUILD_ID}] {task.get('id', '?')} | {instruction}")
    print("=" * 68)

    # --------------------------------------------------------------
    # Phase 0: one-call OAS-grounded evidence plan
    # --------------------------------------------------------------
    plan = {}
    planner_output = ""
    planner_enabled = bool(getattr(config, "OCA_PLANNER_ENABLED", True))
    if planner_enabled:
        from utils.evidence_plan import make_evidence_plan, annotate_execution_eligibility
        if bool(getattr(config, "OCA_INTENT_COMPILER_ENABLED", False)):
            from utils.intent_compiler import make_intent_compiled_plan
            plan, planner_output = make_intent_compiled_plan(
                instruction, benchmark, model, client,
                max_tokens=int(getattr(config, "OCA_INTENT_PLANNER_MAX_TOKENS", 900)),
                allow_legacy_fallback=bool(getattr(config, "OCA_INTENT_LEGACY_FALLBACK", False)),
                intent_repair_attempts=int(getattr(config, "OCA_INTENT_REPAIR_ATTEMPTS", 1)),
                legacy_attempts=int(getattr(config, "OCA_LEGACY_PLANNER_FALLBACK_ATTEMPTS", 1)),
                catalog_mode=str(getattr(config, "OCA_PLANNER_CATALOG_MODE", "full")))
        else:
            plan, planner_output = make_evidence_plan(
                instruction, benchmark, model, client,
                attempts=int(getattr(config, "OCA_PLANNER_ATTEMPTS", 2)),
                api_hints=None,
                enable_semantic_adapters=False,
                catalog_mode=str(getattr(config, "OCA_PLANNER_CATALOG_MODE", "full")),
                enable_semantic_critic=bool(getattr(config, "OCA_GENERIC_PLAN_CRITIC", False)),
                enable_selection_semantic_guard=bool(
                    getattr(config, "OCA_SELECTION_SEMANTIC_GUARD", False)))
        plan = annotate_execution_eligibility(plan)
        planner_messages = [{"role": "system", "content": "OCA evidence planner"},
                            {"role": "user", "content": instruction}]
        log_no_code_turn(logger=logger, turn_num=90, messages=planner_messages,
                         llm_output=planner_output, error_type="evidence_plan",
                         scope="S1_Intention", is_error=False)
        print(f"[PLAN] valid={plan.get('valid')} execution_eligible={plan.get('execution_eligible')} "
              f"steps={len(plan.get('steps', []))} derivations={len(plan.get('derivations', []))}")
        if plan.get("validation_errors"):
            print(f"[PLAN ERROR] {plan['validation_errors'][:3]}")
        if plan.get("validation_warnings"):
            print(f"[PLAN WARN] {plan['validation_warnings'][:3]}")

        if plan.get("steps") and plan.get("execution_eligible"):
            # Exact response-field bindings are a correctness requirement for
            # generic dependent calls, not an adaptive-display optimization.
            # Build the schema-grounded observation/binding plan in every mode;
            # adaptive mode merely uses it to reduce what Phase A sees.
            from utils.evidence_plan import make_observation_plan
            plan, projection_output = make_observation_plan(
                instruction, benchmark, plan, model, client,
                attempts=int(getattr(config, "OCA_PROJECTION_PLANNER_ATTEMPTS", 1)),
                deterministic_first=bool(getattr(
                    config, "OCA_DETERMINISTIC_OBSERVATION_PLAN", True)))
            plan = annotate_execution_eligibility(plan)
            if projection_output:
                log_no_code_turn(
                    logger=logger, turn_num=91,
                    messages=[{"role": "system", "content": "OCA schema-grounded observation/binding fallback planner"},
                              {"role": "user", "content": instruction}],
                    llm_output=projection_output, error_type="observation_plan",
                    scope="S1_Intention", is_error=False)
            print(f"[OBS PLAN] valid={plan.get('observation_plan_valid')} "
                  f"specs={len(plan.get('observation_specs') or [])}")
            if plan.get("observation_plan_errors"):
                print(f"[OBS PLAN WARN] {plan['observation_plan_errors'][:3]}")

    try:
        import benchmarks as _benchmark_registry
        plan = dict(plan or {})
        plan["_runtime_base_url"] = str(_benchmark_registry.get_benchmark(benchmark).get("base_url") or "")
        plan["_execution_policy"] = "advisory"
    except Exception:
        plan = dict(plan or {})
        plan["_execution_policy"] = "advisory"

    from utils.kernel_executor import KernelInterpreter, python_json_assignment
    from utils.plan_execution_audit import runtime_execution_rules
    execution_rules = runtime_execution_rules(
        plan, max_fanout=int(getattr(config, "OCA_MAX_FANOUT_CALLS", 20)))
    bootstrap_code = (
        _LEDGER_BOOTSTRAP + "\n" +
        python_json_assignment("_SECAT_EXECUTION_RULES", execution_rules) +
        "\n_SECAT_EXECUTION_POLICY = " + repr(str(plan.get("_execution_policy") or "strict")) +
        "\n_SECAT_PLAN_CALL_COUNTS = {}" +
        "\n_SECAT_ADVISORY_CALL_COUNTS = {}" +
        "\n_SECAT_TOTAL_CALLS = 0" +
        "\n_SECAT_MAX_TOTAL_CALLS = " + str(int(getattr(config, "OCA_MAX_API_CALLS_PER_TASK", 30))))
    # Kernel startup is infrastructure, not model reasoning. A transient Jupyter
    # startup/bootstrap failure should not consume a benchmark task. Retry once
    # from a fresh kernel before surfacing the exception; semantic/runtime errors
    # after Phase A begins are never retried this way.
    interp = None
    for _kernel_attempt in range(2):
        try:
            interp = KernelInterpreter()
            interp._run_silent(bootstrap_code, check=True)
            break
        except Exception:
            if interp is not None:
                try:
                    interp.shutdown()
                except Exception:
                    pass
            interp = None
            if _kernel_attempt >= 1:
                raise
            print("[KERNEL] startup/bootstrap failed once; retrying with a fresh kernel")
    assert interp is not None

    # --------------------------------------------------------------
    # Phase A: CodeAct-style evidence acquisition, plan-completion gated
    # --------------------------------------------------------------
    messages = [
        {"role": "system", "content": _build_phase_a_prompt(benchmark, plan)},
        {"role": "user", "content": instruction},
    ]
    phase_stdout = []
    code_runs = 0
    captured_entries = []
    entry_cursor = 0
    fatal_transport_error = None
    phase_guard_counts: dict[tuple[str, str, int], int] = {}
    phase_guard_stop_reason = None

    # v4 high-confidence path: a deterministic semantic front-end has already
    # fixed the user meaning, API route graph, bindings, and derivations.  Do not
    # ask a second model to reinterpret that exact executable graph into requests.
    # Execute only mechanically materializable GETs through the same guarded kernel
    # transport.  If any step remains ambiguous, the normal CodeAct phase below is
    # still available as a bounded fallback.
    deterministic_frontend_host_runs = 0
    deterministic_frontend_host_stop = None
    deterministic_frontend_host_complete = False
    if (bool(getattr(config, "OCA_DETERMINISTIC_FRONTEND_HOST_FIRST", True))
            and plan.get("deterministic_semantic_frontend")
            and plan.get("valid") and plan.get("execution_eligible")
            and all(str(x.get("method") or "GET").upper() in {"GET", "HEAD"}
                    for x in plan.get("steps") or [])):
        runs, extra_stdout, stop_reason, frontend_repairs = _run_deterministic_frontend_host_first(
            interp, benchmark, instruction, plan, logger,
            max_batches=int(getattr(config, "OCA_DETERMINISTIC_FRONTEND_HOST_MAX_CALLS", 30)),
            max_fanout=int(getattr(config, "OCA_MAX_FANOUT_CALLS", 20)),
            base_turn=20)
        plan["deterministic_frontend_repairs"] = frontend_repairs
        deterministic_frontend_host_runs += runs
        deterministic_frontend_host_stop = stop_reason
        code_runs += runs
        phase_stdout.extend(extra_stdout)
        host_ledger = _read_ledger_from_kernel(interp)
        host_progress = _audit_ledger_execution(plan, host_ledger)
        deterministic_frontend_host_complete = bool(host_progress.get("complete"))
        if runs:
            print(f"[HOST FIRST] executed {runs} deterministic plan batch(es); "
                  f"complete={deterministic_frontend_host_complete}")

    from utils.evidence_plan import missing_plan_prompt
    from utils.token_meter import stage as token_stage
    phase_a_turn_limit = (0 if deterministic_frontend_host_complete else
                          (MAX_PHASE_A_TURNS if (not planner_enabled or plan.get("execution_eligible")) else 0))
    if planner_enabled and plan.get("steps") and not plan.get("valid"):
        if plan.get("execution_eligible"):
            print("[PLAN] strict certification structure is incomplete; continuing safe GET acquisition in advisory mode")
        else:
            print("[PLAN] non-executable plan after bounded planning; skipping Phase A execution")
    for turn in range(phase_a_turn_limit):
        sent_messages = _phase_a_message_view(messages)
        with token_stage("phase_a"):
            response = client.chat.completions.create(
                model=model, messages=sent_messages, temperature=0.0)
        output = response.choices[0].message.content or ""
        code = _extract_code(output)
        if code:
            result = _execute_generated_code(interp, code)
            new_entries = _kernel_entries_since(interp, entry_cursor)
            previous_cursor = entry_cursor
            entry_cursor += len(new_entries)
            captured_entries.extend(new_entries)
            raw = (result.get("auto_display") or result.get("combined") or
                   result.get("stdout") or "")
            # Keep the established 6k RPR input bound so prompt-only ablations do
            # not change provenance reconstruction semantics. The TaskLogger still
            # stores complete stdout/stderr in execution.*.
            phase_stdout.append(raw[:6000])
            shown = _phase_a_feedback(
                new_entries, result, instruction, plan, call_offset=previous_cursor,
                all_entries=captured_entries, model=model)
            _sync_kernel_execution_rules(interp, plan)
            try:
                error_info = _classify_phase_a_execution(code, raw, shown)
            except Exception:
                error_info = {"error_type": "none", "scope": "none",
                              "is_error": False, "is_silent": False,
                              "raw_error": None}
            logger.log_turn(turn_num=turn + 1, llm_input=[dict(m) for m in sent_messages],
                            llm_output=output, code=code, exec_result=result,
                            error_info=error_info, agent_output=shown)
            code_runs += 1
            error_type = str(error_info.get("error_type") or "")
            fatal_types = {"HTTP_Auth", "task_call_budget", "unsupported_transport"}
            guard_types = {"unplanned_api_call", "unauthorized_binding",
                           "unauthorized_request_instance", "request_contract",
                           "isolation_violation", "plan_call_budget"}
            if error_type in fatal_types:
                fatal_transport_error = error_type
                print(f"[PHASE A] non-recoverable runtime guard failure "
                      f"({fatal_transport_error}); stopping execution and repair loops")
                break
            if error_type in guard_types:
                signature = (error_type, str(error_info.get("raw_error") or shown)[:500], entry_cursor)
                phase_guard_counts[signature] = phase_guard_counts.get(signature, 0) + 1
                if phase_guard_counts[signature] >= 2:
                    phase_guard_stop_reason = "repeated_" + error_type
                    print(f"[PHASE A] repeated runtime guard violation "
                          f"({error_type}); stopping this generated-code loop")
                    break
            messages.extend([
                {"role": "assistant", "content": output},
                {"role": "user", "content":
                    f"Execution output:\n{shown}\n\nContinue the evidence plan, or write READY_TO_ANSWER only when every plan step is complete."},
            ])
            if (plan.get("steps") and bool(getattr(
                    config, "OCA_EARLY_STOP_ON_PLAN_COMPLETE", True))):
                live_ledger = _read_ledger_from_kernel(interp)
                live_progress = _audit_ledger_execution(plan, live_ledger)
                if live_progress.get("complete"):
                    print("[PHASE A] evidence plan complete after execution; "
                          "skipping an extra READY_TO_ANSWER model turn")
                    break
            continue

        log_no_code_turn(logger=logger, turn_num=turn + 1, messages=sent_messages,
                         llm_output=output, error_type="phase_a_signal",
                         scope="S1_Intention", is_error=False)
        if "READY_TO_ANSWER" in output:
            live_ledger = _read_ledger_from_kernel(interp)
            progress = _audit_ledger_execution(plan, live_ledger)
            if progress.get("complete"):
                break
            messages.extend([
                {"role": "assistant", "content": output},
                {"role": "user", "content": missing_plan_prompt(progress)},
            ])
        else:
            messages.extend([
                {"role": "assistant", "content": output},
                {"role": "user", "content": "Run <execute> code to gather evidence, or write READY_TO_ANSWER."},
            ])

    ledger, n_explicit, n_rpr = _rebuild_ledger(interp, plan, phase_stdout)
    progress = _audit_ledger_execution(plan, ledger)

    # Normalize only one uniquely observed collection/company label decoration
    # before resolving child bindings. This is a deterministic observed-data
    # repair, not a new API route or model guess.
    plan, canonical_search_repairs = _apply_observed_canonical_search_repairs(
        interp, plan, ledger)
    plan, observed_relation_repairs = _apply_observed_relation_repairs(
        interp, instruction, plan, ledger)
    plan, observed_order_repairs = _apply_observed_order_repairs(
        interp, instruction, plan, ledger)
    if canonical_search_repairs:
        progress = _audit_ledger_execution(plan, ledger)
        print(f"[CANONICAL SEARCH] normalized {len(canonical_search_repairs)} observed label selection(s)")
    if observed_relation_repairs:
        progress = _audit_ledger_execution(plan, ledger)
        print(f"[RELATION RECOVERY] normalized {len(observed_relation_repairs)} observed sibling relation(s)")

    # First realize any missing validated GET steps mechanically when their
    # concrete bindings are already unambiguous in the ledger. This removes a
    # redundant LLM reinterpretation of the same plan while preserving the
    # initial code-as-action acquisition turn.
    deterministic_plan_completion_runs = 0
    deterministic_plan_completion_stop = None
    if (not fatal_transport_error and plan.get("execution_eligible") and not progress.get("complete") and
            bool(getattr(config, "OCA_DETERMINISTIC_PLAN_COMPLETION", True))):
        runs, extra_stdout, stop_reason = _run_deterministic_plan_completion(
            interp, benchmark, plan, logger,
            max_calls=int(getattr(config, "OCA_DETERMINISTIC_PLAN_COMPLETION_MAX_CALLS", 6)))
        deterministic_plan_completion_runs += runs
        deterministic_plan_completion_stop = stop_reason
        code_runs += runs
        phase_stdout.extend(extra_stdout)
        if runs:
            print(f"[PLAN COMPLETE] host executed {runs} unambiguous missing plan step(s)")
        ledger, n_explicit, n_rpr = _rebuild_ledger(interp, plan, phase_stdout)
        progress = _audit_ledger_execution(plan, ledger)
        # Host completion may itself have produced the first search evidence.
        # Re-run observed search selection repair now, before deciding the chain
        # is complete, so a wrong rank-0 owner cannot lock in its child request.
        plan, post_host_repairs = _apply_observed_canonical_search_repairs(
            interp, plan, ledger)
        plan, post_host_relation_repairs = _apply_observed_relation_repairs(
            interp, instruction, plan, ledger)
        plan, post_host_order_repairs = _apply_observed_order_repairs(
            interp, instruction, plan, ledger)
        if post_host_repairs:
            canonical_search_repairs.extend(post_host_repairs)
            progress = _audit_ledger_execution(plan, ledger)
            print(f"[CANONICAL SEARCH] repaired {len(post_host_repairs)} selection(s) after host acquisition")
        if post_host_relation_repairs:
            observed_relation_repairs.extend(post_host_relation_repairs)
            progress = _audit_ledger_execution(plan, ledger)
            print(f"[RELATION RECOVERY] repaired {len(post_host_relation_repairs)} relation(s) after host acquisition")
        if post_host_order_repairs:
            observed_order_repairs.extend(post_host_order_repairs)
            progress = _audit_ledger_execution(plan, ledger)
            print(f"[ORDER RECOVERY] repaired {len(post_host_order_repairs)} observed extremum selector(s) after host acquisition")

    # Accuracy-first finite completion extension. The legacy fast path deliberately
    # keeps a small six-request construction budget. If the validated read-only plan
    # still has *unambiguous* finite work, exhaust that declared work before changing
    # strategy. The trusted per-task/fan-out guards remain the hard upper bound.
    accuracy_completion_runs = 0
    accuracy_completion_stop = None
    if (not fatal_transport_error and plan.get("execution_eligible") and
            not progress.get("complete") and
            all(str(x.get("method") or "GET").upper() in {"GET", "HEAD"}
                for x in plan.get("steps") or [])):
        runs, extra_stdout, stop_reason = _run_deterministic_plan_completion(
            interp, benchmark, plan, logger,
            max_calls=int(getattr(config, "OCA_ACCURACY_COMPLETION_MAX_CALLS", 30)),
            base_turn=28)
        accuracy_completion_runs += runs
        accuracy_completion_stop = stop_reason
        code_runs += runs
        phase_stdout.extend(extra_stdout)
        if runs:
            print(f"[ACCURACY COMPLETE] host exhausted {runs} additional unambiguous plan batch(es)")
        ledger, n_explicit, n_rpr = _rebuild_ledger(interp, plan, phase_stdout)
        progress = _audit_ledger_execution(plan, ledger)
        plan, post_accuracy_repairs = _apply_observed_canonical_search_repairs(
            interp, plan, ledger)
        plan, post_accuracy_relation_repairs = _apply_observed_relation_repairs(
            interp, instruction, plan, ledger)
        plan, post_accuracy_order_repairs = _apply_observed_order_repairs(
            interp, instruction, plan, ledger)
        if post_accuracy_repairs:
            canonical_search_repairs.extend(post_accuracy_repairs)
            progress = _audit_ledger_execution(plan, ledger)
            print(f"[CANONICAL SEARCH] repaired {len(post_accuracy_repairs)} selection(s) after accuracy completion")
        if post_accuracy_relation_repairs:
            observed_relation_repairs.extend(post_accuracy_relation_repairs)
            progress = _audit_ledger_execution(plan, ledger)
            print(f"[RELATION RECOVERY] repaired {len(post_accuracy_relation_repairs)} relation(s) after accuracy completion")
        if post_accuracy_order_repairs:
            observed_order_repairs.extend(post_accuracy_order_repairs)
            progress = _audit_ledger_execution(plan, ledger)
            print(f"[ORDER RECOVERY] repaired {len(post_accuracy_order_repairs)} observed extremum selector(s) after accuracy completion")

    # One bounded model repair remains only for cases the host cannot construct
    # without semantic guessing (for example an ambiguous producer/join).
    plan_repair_runs = 0
    if (not fatal_transport_error and plan.get("valid") and not progress.get("complete") and
            bool(getattr(config, "OCA_PLAN_COMPLETION_REPAIR", False))):
        prompt = _build_phase_a_prompt(benchmark, plan) + "\n\n" + missing_plan_prompt(progress)
        plan_repair_meta = {}
        runs, extra_stdout = _run_fetch_round(
            interp, prompt, "Complete only the missing evidence-plan steps.",
            model, logger, base_turn=40,
            max_steps=int(getattr(config, "OCA_PLAN_REPAIR_STEPS", 3)),
            question=instruction, plan=plan, stage_name="phase_a_repair",
            progress_meta=plan_repair_meta)
        plan_repair_runs += runs
        code_runs += runs
        phase_stdout.extend(extra_stdout)
        if plan_repair_meta.get("fatal_error"):
            fatal_transport_error = str(plan_repair_meta["fatal_error"])
        ledger, n_explicit, n_rpr = _rebuild_ledger(interp, plan, phase_stdout)
        progress = _audit_ledger_execution(plan, ledger)

    write_methods = {"POST", "PUT", "DELETE", "PATCH"}
    plan_requires_write = any(str(step.get("method") or "GET").upper() in write_methods
                              for step in (plan.get("steps") or []))
    action_request_recognized = bool(
        plan_requires_write or str(plan.get("answer_mode") or "").lower() == "action"
        or any(str(n.get("op") or "") == "action"
               for n in ((plan.get("intent") or {}).get("nodes") or [])))
    action_completion_repair = None
    action_completion_runs = 0
    if (not fatal_transport_error and plan_requires_write and plan.get("valid") and
            plan.get("execution_eligible") and not progress.get("complete")):
        (ledger, progress, action_completion_runs, action_stdout, ne_action, nr_action,
         action_completion_repair, action_fatal_error) = _run_action_completion_repair(
            interp, benchmark, instruction, model, logger, plan, ledger, progress,
            phase_stdout, base_turn=46)
        code_runs += action_completion_runs
        phase_stdout.extend(action_stdout)
        n_explicit, n_rpr = ne_action, nr_action
        if action_fatal_error:
            fatal_transport_error = str(action_fatal_error)
        if action_completion_repair and action_completion_repair.get("attempted"):
            print("[ACTION REPAIR] "
                  f"runs={action_completion_runs} "
                  f"complete={progress.get('complete')} "
                  f"missing={action_completion_repair.get('missing_after')}")

    # If the validated action strategy is still broken and no successful side
    # effect has occurred, spend one bounded whole-plan recovery attempt instead
    # of terminating.  Once any write succeeds, whole-plan replanning is disabled
    # to prevent duplicate or conflicting side effects; only plan-locked completion
    # is allowed from that point onward.
    action_runtime_replan = None
    successful_write_before_replan = any(
        str(call.get("method") or "GET").upper() in write_methods
        and isinstance(call.get("status_code"), int)
        and 200 <= call.get("status_code") < 300
        for call in (ledger.api_calls or []))
    # The append-only commit journal is written before rich response bookkeeping.
    # If a write succeeded and later ledger/certificate work failed, semantic
    # replanning must still remain locked to prevent duplicate side effects.
    try:
        successful_write_before_replan = (
            successful_write_before_replan or bool(_kernel_write_commits(interp))
            or bool(_kernel_uncertain_writes(interp)))
    except Exception:
        # Fail closed: inability to prove that no write happened must never open a
        # whole-plan replan that could replay a side effect.
        successful_write_before_replan = True
    if (not fatal_transport_error and action_request_recognized
            and not progress.get("complete") and not successful_write_before_replan
            and not _terminal_action_precondition(action_completion_repair)
            and bool(getattr(config, "OCA_ACTION_RUNTIME_REPLAN", True))):
        (replan_plan, replan_ledger, _replan_compiled, replan_progress,
         replan_runs, replan_stdout, action_runtime_replan,
         replan_explicit, replan_rpr) = _runtime_replan_attempt(
            interp, benchmark, instruction, model, logger, plan, ledger, {},
            phase_stdout, attempt_index=1, base_turn=52, allow_actions=True)
        code_runs += int(replan_runs or 0)
        phase_stdout.extend(replan_stdout or [])
        if action_runtime_replan and action_runtime_replan.get("adopted"):
            plan, ledger = replan_plan, replan_ledger
            if replan_progress is not None:
                progress = replan_progress
            if replan_explicit is not None:
                n_explicit = replan_explicit
            if replan_rpr is not None:
                n_rpr = replan_rpr
            plan_requires_write = any(
                str(step.get("method") or "GET").upper() in write_methods
                for step in (plan.get("steps") or []))
            action_request_recognized = True
            print("[ACTION REPLAN] adopted bounded replacement strategy; "
                  f"complete={progress.get('complete')}")
            # If the recovered strategy has committed only a prefix, keep the
            # same strategy and exhaust its remaining validated actions.
            if (plan_requires_write and plan.get("valid") and plan.get("execution_eligible")
                    and not progress.get("complete")):
                (ledger, progress, extra_runs, extra_stdout, ne_action2, nr_action2,
                 completion2, action_fatal2) = _run_action_completion_repair(
                    interp, benchmark, instruction, model, logger, plan, ledger,
                    progress, phase_stdout, base_turn=58)
                code_runs += int(extra_runs or 0)
                phase_stdout.extend(extra_stdout or [])
                n_explicit, n_rpr = ne_action2, nr_action2
                if completion2:
                    if action_completion_repair:
                        action_completion_repair = {
                            "initial": action_completion_repair,
                            "after_replan": completion2,
                        }
                    else:
                        action_completion_repair = completion2
                if action_fatal2:
                    fatal_transport_error = str(action_fatal2)

    has_observed_write = any(str(call.get("method") or "GET").upper() in write_methods
                             for call in (ledger.api_calls or []))
    try:
        has_observed_write = has_observed_write or bool(_kernel_write_commits(interp))
    except Exception:
        # An unreadable commit journal is treated conservatively as possible write
        # state, routing finalization through action handling rather than replanning.
        has_observed_write = True
    if has_observed_write or plan_requires_write or action_request_recognized:
        print(f"[LEDGER] calls={len(ledger.api_calls)} raw_obs={len(ledger)} "
              f"derived={len(ledger.derived)} plan_complete={progress.get('complete')}")
        return _finalize_action(
            task, instruction, model, logger, interp, ledger, plan, progress,
            code_runs, n_explicit, n_rpr, token_read,
            action_completion_repair=action_completion_repair,
            plan_repair_code_runs=plan_repair_runs + action_completion_runs,
            action_runtime_replan=action_runtime_replan)

    # --------------------------------------------------------------
    # Deterministic evidence compilation
    # --------------------------------------------------------------
    from utils.evidence_compiler import compile_evidence
    compiled = compile_evidence(instruction, ledger, plan)

    # Bounded schema-level structural fallback. This is separate from generic
    # answer repair and cannot alter tasks whose required role evidence exists.
    structural_fallback_runs = 0
    structural_fallback_targets = []
    if bool(getattr(config, "OCA_STRUCTURAL_FALLBACK", False)):
        structural_fallback_targets = _structural_fallback_targets(
            instruction, plan, ledger)
        if structural_fallback_targets:
            runs, extra_stdout = _run_lineage_completion(
                interp, benchmark, structural_fallback_targets, logger, base_turn=33)
            structural_fallback_runs += runs
            code_runs += runs
            phase_stdout.extend(extra_stdout)
            ledger, n_explicit, n_rpr = _rebuild_ledger(interp, plan, phase_stdout)
            progress = _audit_ledger_execution(plan, ledger)
            compiled = compile_evidence(instruction, ledger, plan)

    # Complete an answer-surface call when Phase A used the right endpoint family
    # with an entity id that contradicts the compiler's authoritative selection.
    lineage_completion_runs = 0
    lineage_completion_targets = []
    if bool(getattr(config, "OCA_DETERMINISTIC_LINEAGE_COMPLETION", False)):
        lineage_completion_targets = _lineage_completion_targets(plan, ledger, compiled)
        if lineage_completion_targets:
            runs, extra_stdout = _run_lineage_completion(
                interp, benchmark, lineage_completion_targets, logger)
            lineage_completion_runs += runs
            code_runs += runs
            phase_stdout.extend(extra_stdout)
            ledger, n_explicit, n_rpr = _rebuild_ledger(interp, plan, phase_stdout)
            progress = _audit_ledger_execution(plan, ledger)
            compiled = compile_evidence(instruction, ledger, plan)

    # Before spending another planner call, recover a singular visual asset from
    # alternate plausible search owners when the selected owner's dedicated asset
    # collection is empty. Each attempt stays one-owner and lineage-preserving.
    asset_owner_fallback_record = {"attempted": False, "attempts": [], "adopted": False}
    if not fatal_transport_error and plan.get("execution_eligible"):
        asset_outcome = _run_asset_owner_fallback(
            interp, benchmark, instruction, logger, plan, ledger, compiled, phase_stdout)
        asset_owner_fallback_record = asset_outcome[6]
        code_runs += int(asset_outcome[4] or 0)
        phase_stdout.extend(asset_outcome[5] or [])
        if asset_owner_fallback_record.get("adopted"):
            plan, ledger, compiled, progress = asset_outcome[0], asset_outcome[1], asset_outcome[2], asset_outcome[3]
            n_explicit, n_rpr = asset_outcome[7], asset_outcome[8]
            print(f"[ASSET RECOVERY] adopted alternate owner rank {asset_owner_fallback_record.get('selected_rank')}")
        elif asset_owner_fallback_record.get("attempted"):
            ledger = asset_outcome[1]
            progress = _audit_ledger_execution(plan, ledger)
            compiled = compile_evidence(instruction, ledger, plan)

    # Aggregate season credits may omit episodic roles. Before whole-plan
    # replanning, expand that one relation through documented episode credits.
    season_episode_fallback_record = {"attempted": False, "attempts": [], "adopted": False}
    if not fatal_transport_error and plan.get("execution_eligible"):
        seasonfb = _run_season_episode_relation_fallback(
            interp, benchmark, instruction, model, logger, plan, ledger, compiled, phase_stdout)
        season_episode_fallback_record = seasonfb[6]
        code_runs += int(seasonfb[4] or 0); phase_stdout.extend(seasonfb[5] or [])
        if season_episode_fallback_record.get("adopted"):
            plan, ledger, compiled, progress = seasonfb[0], seasonfb[1], seasonfb[2], seasonfb[3]
            n_explicit, n_rpr = seasonfb[7], seasonfb[8]
            print(f"[SEASON RELATION RECOVERY] adopted episode-credit fan-out for {season_episode_fallback_record.get('role')}")
        elif season_episode_fallback_record.get("attempted"):
            ledger = seasonfb[1]; progress = _audit_ledger_execution(plan, ledger); compiled = compile_evidence(instruction, ledger, plan)

    # If a named work search returned no plausible candidate, try documented
    # sibling resource searches deterministically before asking the planner to
    # invent another whole strategy. This is particularly valuable for APIs that
    # split movies/TV/etc. into separate search families.
    cross_resource_fallback_record = {"attempted": False, "attempts": [], "adopted": False}
    if not fatal_transport_error and plan.get("execution_eligible"):
        xres = _run_cross_resource_search_fallback(
            interp, benchmark, instruction, model, logger, plan, ledger, compiled, phase_stdout)
        cross_resource_fallback_record = xres[6]
        code_runs += int(xres[4] or 0)
        phase_stdout.extend(xres[5] or [])
        if cross_resource_fallback_record.get("adopted"):
            plan, ledger, compiled, progress = xres[0], xres[1], xres[2], xres[3]
            n_explicit, n_rpr = xres[7], xres[8]
            print(f"[RESOURCE RECOVERY] adopted sibling search {cross_resource_fallback_record.get('selected_search_endpoint')}")
        elif cross_resource_fallback_record.get("attempted"):
            ledger = xres[1]
            progress = _audit_ledger_execution(plan, ledger)
            compiled = compile_evidence(instruction, ledger, plan)

    # Accuracy-first post-execution strategy recovery. A risky or incomplete
    # read-only attempt can be replaced by a genuinely different OAS-grounded plan.
    runtime_evidence_replan_attempted = False
    runtime_evidence_replan_adopted = False
    runtime_evidence_replan_record = None
    runtime_evidence_replan_records = []
    max_runtime_replans = int(getattr(config, "OCA_RUNTIME_REPLAN_ATTEMPTS", 2))
    for recovery_idx in range(1, max_runtime_replans + 1):
        if fatal_transport_error or str(getattr(config, "OCA_REPAIR_MODE", "typed")).lower() != "typed":
            break
        if not _needs_runtime_evidence_replan(
                plan, progress, compiled, question=instruction, ledger=ledger):
            break
        runtime_evidence_replan_attempted = True
        outcome = _runtime_replan_attempt(
            interp, benchmark, instruction, model, logger, plan, ledger, compiled,
            phase_stdout, attempt_index=recovery_idx, base_turn=60 + (recovery_idx - 1) * 12)
        record = outcome[6]
        runtime_evidence_replan_records.append(record)
        runtime_evidence_replan_record = record
        code_runs += int(outcome[4] or 0)
        phase_stdout.extend(outcome[5] or [])
        if record.get("adopted"):
            plan, ledger, compiled, progress = outcome[0], outcome[1], outcome[2], outcome[3]
            n_explicit, n_rpr = outcome[7], outcome[8]
            runtime_evidence_replan_adopted = True
            print(f"[ACCURACY RECOVERY] adopted runtime re-plan {recovery_idx}")
            # The adopted plan is already semantically clean by the helper. Stop
            # pre-contract replanning and let certification/semantic commit evaluate it.
            break
        if record.get("no_progress"):
            print("[ACCURACY RECOVERY] planner repeated the same strategy; stopping no-progress loop")
            break

    print(f"[LEDGER] calls={len(ledger.api_calls)} raw_obs={len(ledger)} "
          f"derived={len(ledger.derived)} plan_complete={progress.get('complete')}")

    from utils.evidence_contract import (
        classify_task, augment_contract_with_plan, verify_contract, repair_certificate,
        materialize_certified_answer, certificate_from_compiled,
        evidence_gap, repair_prompt_section, delta_check,
    )
    contract = augment_contract_with_plan(classify_task(instruction), plan)
    contract_mode = str(getattr(config, "OCA_CONTRACT_MODE", "enforced")).lower()
    if contract_mode == "off":
        # OCA v2's defining safety property requires the commit gate. Keep the old
        # behavior available through --oca-version v1 instead of silently disabling it.
        contract_mode = "enforced"
    obs_index = {o["obs_id"]: o for o in ledger.observations}
    der_index = {d["obs_id"]: d for d in ledger.derived}

    # Deterministic Phase B fast path.  When the compiler already has the exact
    # requested output (count/comparison/selected entity/complete role set), avoid
    # another model call and certify the replayable value directly.
    deterministic_answer = False
    semantic_commit_record = None
    skip_answer_generation = bool(planner_enabled and not plan.get("execution_eligible"))
    if skip_answer_generation:
        # Only plans that are unsafe/non-executable skip evidence selection. A
        # read-only advisory plan is allowed to reach compilation/Phase B; the
        # final evidence/plan gate still prevents unsupported certification.
        final_answer, cert, cited = "", {}, []
        certificate_changed = False
        verification = verify_contract(
            contract, cert, obs_index, der_index, final_answer, plan=plan)
        verification = _apply_plan_gate(
            verification, plan, progress, planner_enabled=planner_enabled)
        accepted = False
    elif fatal_transport_error:
        final_answer, cert, cited = "", {}, []
        certificate_changed = False
        verification = {
            "certificate_accepted": False,
            "contract_satisfied": False,
            "verification_status": "evidence_gap",
            "missing_slots": ["runtime_execution"],
            "checks": {
                "plan_valid": bool(plan.get("valid")),
                "plan_complete": False,
            },
            "notes": [f"non-recoverable runtime failure: {fatal_transport_error}"],
        }
        accepted = False
    else:
        final_answer, cert = certificate_from_compiled(
            contract, compiled, obs_index, der_index, plan=plan)
    if (not fatal_transport_error and final_answer and cert and bool(getattr(
            config, "OCA_DETERMINISTIC_ANSWER_FIRST", True))):
        verification = verify_contract(
            contract, cert, obs_index, der_index, final_answer, plan=plan)
        verification = _apply_plan_gate(
            verification, plan, progress, planner_enabled=planner_enabled)
        accepted = (verification["certificate_accepted"] and
                    verification["contract_satisfied"])
        if accepted:
            verification, semantic_commit_record = _apply_semantic_commit(
                verification, instruction, final_answer, cert, contract, plan,
                obs_index, der_index, model, logger, deterministic=True,
                turn_num=95, stage_name="semantic_commit_deterministic")
            accepted = (verification["certificate_accepted"] and
                        verification["contract_satisfied"])
        if accepted:
            deterministic_answer = True
            cited = cert.get("cited_observation_ids") or []
            log_no_code_turn(
                logger=logger, turn_num=98,
                messages=[{"role": "system", "content":
                           "Deterministic OCA Phase B from compiled evidence."}],
                llm_output=json.dumps({"final_answer": final_answer, **cert},
                                      ensure_ascii=False),
                error_type="phase_b_deterministic", scope="S5_Integration",
                is_error=False)
        else:
            final_answer, cert = None, None

    if not fatal_transport_error and not skip_answer_generation and not deterministic_answer:
        final_answer, cited, cert, _ = _phase_b(
            instruction, ledger, model, logger, contract, plan, compiled,
            turn_num=98)
        # Deterministic certificate repair fixes formatting/citation defects only.
        cert_repaired = repair_certificate(
            contract, cert, obs_index, der_index, final_answer, plan=plan)
        certificate_changed = cert_repaired != cert
        cert = cert_repaired
        cited = cert.get("cited_observation_ids") or []
        verification = verify_contract(
            contract, cert, obs_index, der_index, final_answer, plan=plan)
        verification = _apply_plan_gate(
            verification, plan, progress, planner_enabled=planner_enabled)
        verification, semantic_commit_record = _apply_semantic_commit(
            verification, instruction, final_answer, cert, contract, plan,
            obs_index, der_index, model, logger, deterministic=False,
            turn_num=95, stage_name="semantic_commit")
        accepted = (verification["certificate_accepted"] and
                    verification["contract_satisfied"])
    elif not fatal_transport_error and not skip_answer_generation:
        certificate_changed = False

    answer_repaired = False
    if (not fatal_transport_error and not accepted and verification.get("verification_status") in
            {"certificate_gap", "contradiction"} and
            _answer_repair_is_useful(verification) and
            bool(getattr(config, "OCA_ANSWER_REPAIR", False))):
        # Re-verbalize from the same evidence. No new factual source is introduced.
        fa2, cited2, cert2, _ = _phase_b(
            instruction, ledger, model, logger, contract, plan, compiled,
            turn_num=97, correction_notes=verification.get("notes") or [],
            stage_name="answer_repair")
        cert2 = repair_certificate(
            contract, cert2, obs_index, der_index, fa2, plan=plan)
        vr2 = verify_contract(contract, cert2, obs_index, der_index, fa2, plan=plan)
        vr2 = _apply_plan_gate(vr2, plan, progress, planner_enabled=planner_enabled)
        vr2, semantic2 = _apply_semantic_commit(
            vr2, instruction, fa2, cert2, contract, plan,
            obs_index, der_index, model, logger, deterministic=False,
            turn_num=94, stage_name="semantic_commit_repair")
        accepted2 = vr2["certificate_accepted"] and vr2["contract_satisfied"]
        if accepted2:
            final_answer, cert, cited, verification = (
                fa2, cert2, cert2.get("cited_observation_ids") or [], vr2)
            accepted = True
            answer_repaired = True
            semantic_commit_record = semantic2
        elif semantic2 is not None:
            semantic_commit_record = semantic2

    # If certification/semantic commit still rejects a complete read-only
    # attempt, make one additional whole-strategy recovery using the verification
    # diagnosis. This is the key accuracy-first difference from fail-closed v3.8.34.
    if (not fatal_transport_error and not accepted and
            _needs_runtime_evidence_replan(
                plan, progress, compiled, question=instruction, ledger=ledger,
                verification=verification)):
        post_idx = len(runtime_evidence_replan_records) + 1
        if post_idx <= int(getattr(config, "OCA_RUNTIME_REPLAN_ATTEMPTS", 2)):
            runtime_evidence_replan_attempted = True
            outcome = _runtime_replan_attempt(
                interp, benchmark, instruction, model, logger, plan, ledger, compiled,
                phase_stdout, verification=verification, attempt_index=post_idx,
                base_turn=72 + (post_idx - 1) * 12)
            record = outcome[6]
            runtime_evidence_replan_records.append(record)
            runtime_evidence_replan_record = record
            code_runs += int(outcome[4] or 0)
            phase_stdout.extend(outcome[5] or [])
            if record.get("adopted"):
                plan, ledger, compiled, progress = outcome[0], outcome[1], outcome[2], outcome[3]
                n_explicit, n_rpr = outcome[7], outcome[8]
                runtime_evidence_replan_adopted = True
                obs_index = {o["obs_id"]: o for o in ledger.observations}
                der_index = {d["obs_id"]: d for d in ledger.derived}
                contract = augment_contract_with_plan(classify_task(instruction), plan)
                final_answer, cert = certificate_from_compiled(
                    contract, compiled, obs_index, der_index, plan=plan)
                if final_answer and cert:
                    verification = verify_contract(
                        contract, cert, obs_index, der_index, final_answer, plan=plan)
                    verification = _apply_plan_gate(
                        verification, plan, progress, planner_enabled=planner_enabled)
                    verification, semantic_post = _apply_semantic_commit(
                        verification, instruction, final_answer, cert, contract, plan,
                        obs_index, der_index, model, logger, deterministic=True,
                        turn_num=93, stage_name="semantic_commit_runtime_replan")
                    accepted = bool(verification["certificate_accepted"] and
                                    verification["contract_satisfied"])
                    cited = (cert.get("cited_observation_ids") or []) if accepted else []
                    deterministic_answer = bool(accepted)
                    if semantic_post is not None:
                        semantic_commit_record = semantic_post
                if not accepted:
                    final_answer, cited, cert, _ = _phase_b(
                        instruction, ledger, model, logger, contract, plan, compiled,
                        turn_num=96, correction_notes=verification.get("notes") or [],
                        stage_name="runtime_replan_answer")
                    cert = repair_certificate(
                        contract, cert, obs_index, der_index, final_answer, plan=plan)
                    cited = cert.get("cited_observation_ids") or []
                    verification = verify_contract(
                        contract, cert, obs_index, der_index, final_answer, plan=plan)
                    verification = _apply_plan_gate(
                        verification, plan, progress, planner_enabled=planner_enabled)
                    verification, semantic_post = _apply_semantic_commit(
                        verification, instruction, final_answer, cert, contract, plan,
                        obs_index, der_index, model, logger, deterministic=False,
                        turn_num=93, stage_name="semantic_commit_runtime_replan_answer")
                    accepted = bool(verification["certificate_accepted"] and
                                    verification["contract_satisfied"])
                    if semantic_post is not None:
                        semantic_commit_record = semantic_post

    # --------------------------------------------------------------
    # One bounded typed evidence repair for genuine evidence gaps
    # --------------------------------------------------------------
    repair_record = None
    repair_mode = str(getattr(config, "OCA_REPAIR_MODE", "typed")).lower()
    if (not fatal_transport_error and plan.get("valid") and not accepted and
            repair_mode in {"generic", "typed", "shuffled"}):
        gap = evidence_gap(
            contract, verification, instruction,
            include_endpoint_family=bool(getattr(config, "OCA_REPAIR_ENDPOINT_HINT", False)),
            plan_progress=progress)
        if repair_mode == "generic":
            gap = {"required_slot": "generic", "needed_relation": "more relevant evidence",
                   "source_entity_hint": "the entities in the question",
                   "reason": "the first answer did not pass verification"}
        elif repair_mode == "shuffled":
            gap = {"required_slot": "asset_path", "needed_relation": "an unrelated image path",
                   "source_entity_hint": "an unrelated entity", "reason": "shuffled control"}
        if gap:
            before = {o["obs_id"]: o for o in ledger.observations}
            repair_system = _build_phase_a_prompt(benchmark, plan) + repair_prompt_section(gap)
            repair_progress_meta = {}
            runs, extra_stdout = _run_fetch_round(
                interp, repair_system, "Fetch the missing evidence now.", model,
                logger, base_turn=50,
                max_steps=int(getattr(config, "OCA_EVIDENCE_REPAIR_STEPS", 3)),
                question=instruction, plan=plan, stage_name="evidence_repair",
                progress_meta=repair_progress_meta)
            code_runs += runs
            phase_stdout.extend(extra_stdout)
            repair_fatal_error = repair_progress_meta.get("fatal_error")
            if repair_fatal_error:
                fatal_transport_error = str(repair_fatal_error)
            ledger2, ne2, nr2 = _rebuild_ledger(interp, plan, phase_stdout)
            compiled2 = compile_evidence(instruction, ledger2, plan)
            progress2 = _audit_ledger_execution(plan, ledger2)
            after = {o["obs_id"]: o for o in ledger2.observations}
            der_after = {d["obs_id"]: d for d in ledger2.derived}
            filled, delta_note = delta_check(gap, before, after, der_after)
            repair_record = {
                "mode": repair_mode, "gap": gap, "repair_code_runs": runs,
                "obs_before": len(before), "obs_after": len(after),
                "api_calls_before": len(ledger.api_calls),
                "api_calls_after": len(ledger2.api_calls),
                "delta_filled": filled, "delta_note": delta_note,
                "recovered": False,
                "stopped_no_progress": bool(
                    repair_progress_meta.get("stopped_no_progress")),
                "stop_reason": repair_progress_meta.get("stop_reason"),
                "repair_http_calls_observed": int(
                    repair_progress_meta.get("http_calls_observed", 0)),
            }
            if filled and not repair_fatal_error:
                fa3, cert3 = certificate_from_compiled(
                    contract, compiled2, after, der_after, plan=plan)
                repair_deterministic = bool(fa3 and cert3 and bool(getattr(
                    config, "OCA_DETERMINISTIC_ANSWER_FIRST", True)))
                if not repair_deterministic:
                    fa3, cited3, cert3, _ = _phase_b(
                        instruction, ledger2, model, logger, contract, plan, compiled2,
                        turn_num=96, correction_notes=[
                            "New evidence was fetched; answer only from the updated ledger."],
                        stage_name="evidence_repair_answer")
                    cert3 = repair_certificate(
                        contract, cert3, after, der_after, fa3, plan=plan)
                vr3 = verify_contract(contract, cert3, after, der_after, fa3, plan=plan)
                vr3 = _apply_plan_gate(vr3, plan, progress2, planner_enabled=planner_enabled)
                vr3, semantic3 = _apply_semantic_commit(
                    vr3, instruction, fa3, cert3, contract, plan,
                    after, der_after, model, logger,
                    deterministic=repair_deterministic,
                    turn_num=93, stage_name="semantic_commit_evidence_repair")
                accepted3 = vr3["certificate_accepted"] and vr3["contract_satisfied"]
                if accepted3:
                    ledger, compiled = ledger2, compiled2
                    progress = progress2
                    obs_index, der_index = after, der_after
                    n_explicit, n_rpr = ne2, nr2
                    final_answer, cert, cited, verification = (
                        fa3, cert3, cert3.get("cited_observation_ids") or [], vr3)
                    accepted = True
                    repair_record["recovered"] = True
                    if semantic3 is not None:
                        semantic_commit_record = semantic3

    phase_b_draft = final_answer
    value_locked_emission = False
    if accepted:
        final_answer = materialize_certified_answer(
            contract, cert, der_index, fallback=final_answer,
            obs_index=obs_index, plan=plan)
        # Defensive invariant: the value-locked emission must still satisfy the
        # same certificate that authorized the draft.
        emission_vr = verify_contract(
            contract, cert, obs_index, der_index, final_answer, plan=plan)
        emission_vr = _apply_plan_gate(
            emission_vr, plan, progress, planner_enabled=planner_enabled)
        if not (emission_vr["certificate_accepted"] and
                emission_vr["contract_satisfied"]):
            verification = emission_vr
            accepted = False
        else:
            verification = emission_vr
            value_locked_emission = True

    rejected_answer = None
    evidence_salvage_emitted = False
    evidence_salvage_record = {"attempted": False, "adopted": False, "reason": None}
    best_effort_emitted = False
    uncertified_fallback_emitted = False
    exhausted_execution_failure = False
    if contract_mode == "enforced" and not accepted:
        rejected_answer = final_answer
        if not fatal_transport_error:
            salvage_answer, salvage_ids, _salvage_cert, evidence_salvage_record = (
                _deterministic_evidence_salvage(
                    instruction, ledger, plan, progress, compiled, verification))
            if salvage_answer and salvage_ids:
                final_answer, cited = salvage_answer, salvage_ids
                evidence_salvage_emitted = True
                print("[EVIDENCE SALVAGE] emitted deterministic question-verified answer "
                      "despite unresolved strict plan closure")
        if (not evidence_salvage_emitted and
                bool(getattr(config, "OCA_EMIT_BEST_EFFORT_AFTER_RECOVERY", True)) and
                not fatal_transport_error):
            best_answer, best_ids = _best_effort_evidence_answer(
                instruction, ledger, plan, compiled, model, logger, turn_num=99)
            if best_answer and best_ids:
                final_answer, cited = best_answer, best_ids
                best_effort_emitted = True
        if not evidence_salvage_emitted and not best_effort_emitted:
            reason = ", ".join(verification.get("missing_slots") or []) or "verification failed"
            # Accuracy-first final fallback: certification failure must not erase a
            # non-empty answer candidate after all bounded recovery paths have been
            # exhausted. Emit the strongest attempted answer and mark it explicitly
            # uncertified. This preserves the user's requested answer attempt without
            # weakening the certificate gate itself.
            if isinstance(rejected_answer, str) and rejected_answer.strip():
                final_answer = rejected_answer.strip()
                cited = [str(x) for x in ((cert or {}).get("cited_observation_ids") or [])
                         if str(x) in obs_index or str(x) in der_index]
                uncertified_fallback_emitted = True
                print("[ACCURACY FALLBACK] emitted strongest uncertified answer after recovery exhaustion")
            else:
                # No answer candidate exists at all (for example an unrecoverable
                # transport failure or genuinely empty execution). Report an exhausted
                # execution failure rather than converting it into an abstention/refusal.
                final_answer = f"Execution exhausted without a usable answer ({reason})."
                cited = []
                exhausted_execution_failure = True

    contract_record = {
        "version": 2,
        "build_id": OCA_BUILD_ID,
        "mode": contract_mode,
        "task_kind": contract.get("task_kind"),
        "answer_kind": contract.get("answer_kind"),
        "answer_relation": contract.get("answer_relation"),
        "selection_policies": contract.get("selection_policies"),
        "certificate_repaired": certificate_changed,
        "deterministic_answer": deterministic_answer,
        "answer_repaired": answer_repaired,
        "semantic_commit": semantic_commit_record,
        "value_locked_emission": value_locked_emission,
        "phase_b_draft": phase_b_draft,
        "certificate_accepted": verification.get("certificate_accepted"),
        "contract_satisfied": verification.get("contract_satisfied"),
        "verification_status": verification.get("verification_status"),
        "missing_slots": verification.get("missing_slots"),
        "checks": verification.get("checks"),
        "notes": verification.get("notes"),
        "enforced_rejection": bool(contract_mode == "enforced" and not accepted),
        "rejected_answer": rejected_answer,
        "repair": repair_record,
        "runtime_evidence_replan": runtime_evidence_replan_record,
        "runtime_evidence_replans": runtime_evidence_replan_records,
        "evidence_salvage_emitted": evidence_salvage_emitted,
        "evidence_salvage": evidence_salvage_record,
        "best_effort_emitted": best_effort_emitted,
        "uncertified_fallback_emitted": uncertified_fallback_emitted,
        "exhausted_execution_failure": exhausted_execution_failure,
    }

    print(f"[COMMIT] accepted={accepted} status={verification.get('verification_status')} "
          f"missing={verification.get('missing_slots')}")
    print(f"[ANSWER] {final_answer[:500]}")

    api_call_accounting = _api_call_accounting(interp, ledger)
    if api_call_accounting["api_calls_uncommitted"]:
        print(f"[API CALLS] committed={api_call_accounting['api_calls_committed']} "
              f"attempted={api_call_accounting['api_calls_attempted']}")

    graph = EvidenceGraph(task.get("id", "?"), instruction, benchmark=benchmark)
    graph.finalize(ledger, final_answer, cited)
    try:
        graph_path = os.path.join(os.path.dirname(config.RUN_DIR), "evidence_graph.jsonl")
        graph.append_jsonl(ledger, graph_path)
    except Exception as exc:
        print(f"[WARN] evidence graph write failed: {exc}")

    # Accuracy-first OCA never relabels an exhausted attempt as an abstention.
    # Certification status remains separately visible in the contract record.
    explicit_abstention = False
    summary = finalize_task(
        logger=logger, interpreter=interp, final_answer=final_answer,
        # A visible abstention is an explicit failure, not a silent wrong answer.
        ground_truth=None,
        code_runs=code_runs, explicit_abstention=explicit_abstention,
        extra_summary={
            "oca_version": 2,
            "oca_build_id": OCA_BUILD_ID,
            "plan": plan,
            "plan_progress": progress,
            "phase_a_guard_stop_reason": phase_guard_stop_reason,
            "deterministic_frontend_host_runs": deterministic_frontend_host_runs,
            "deterministic_frontend_host_stop": deterministic_frontend_host_stop,
            "deterministic_frontend_host_complete": deterministic_frontend_host_complete,
            "deterministic_plan_completion_runs": deterministic_plan_completion_runs,
            "deterministic_plan_completion_stop": deterministic_plan_completion_stop,
            "accuracy_completion_runs": accuracy_completion_runs,
            "accuracy_completion_stop": accuracy_completion_stop,
            "canonical_search_repairs": canonical_search_repairs,
            "observed_relation_repairs": observed_relation_repairs,
            "asset_owner_fallback": asset_owner_fallback_record,
            "cross_resource_fallback": cross_resource_fallback_record,
            "runtime_evidence_replan_attempted": runtime_evidence_replan_attempted,
            "runtime_evidence_replan_adopted": runtime_evidence_replan_adopted,
            "runtime_evidence_replan": runtime_evidence_replan_record,
            "runtime_evidence_replans": runtime_evidence_replan_records,
            "plan_repair_code_runs": plan_repair_runs,
            "structural_fallback_code_runs": structural_fallback_runs,
            "structural_fallback_targets": structural_fallback_targets,
            "lineage_completion_code_runs": lineage_completion_runs,
            "lineage_completion_targets": lineage_completion_targets,
            "executed_routes": _executed_route_templates(plan, ledger),
            "observations": len(ledger),
            **api_call_accounting,
            "derived_explicit": n_explicit,
            "derived_rpr": n_rpr,
            "derived_auto": len([d for d in ledger.derived
                                 if d.get("capture_method") == "auto_compiler"]),
            "compiled": compiled,
            "cited_observation_ids": cited,
            "grounded": graph.grounded,
            "tokens": token_read(),
            "contract": contract_record,
            "uncertified_answer_emitted": bool(
                (contract_mode != "enforced" and not accepted) or
                evidence_salvage_emitted or best_effort_emitted),
            "evidence_salvage_emitted": evidence_salvage_emitted,
            "evidence_salvage": evidence_salvage_record,
            "best_effort_emitted": best_effort_emitted,
            "uncertified_fallback_emitted": uncertified_fallback_emitted,
            "exhausted_execution_failure": exhausted_execution_failure,
            "safe_abstention": False,
            "explicit_abstention": False,
        })
    if explicit_abstention:
        # finalize_task() already saved the complete task record. Keep the returned
        # in-memory summary consistent without writing the exact same JSON twice.
        summary["success"] = False
        summary["silent_failure"] = False
        logger.log["summary"]["success"] = False
        logger.log["summary"]["silent_failure"] = False
        logger.log["summary"]["explicit_abstention"] = True
    return summary
