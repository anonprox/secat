"""
Observation ledger and evidence graph for OCA.

OCA v2 keeps two representations of every API response:
  * a lossless in-memory response trace, used for deterministic compilation; and
  * normalized citable observations, used by Phase B and the verifier.

The normalization is structural rather than task-specific. Root primitives are
preserved, list items become independently citable observations, and provenance
(call id, endpoint, JSON relation, position, parent record) is retained. This
prevents the v1 failure where useful nested evidence (nested arrays, objects, and relation records) was fetched but discarded before Phase B.
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any, Iterable



_SECRET_KEYS = {
    "api_key", "apikey", "access_token", "refresh_token", "token",
    "authorization", "client_secret", "secret", "password"
}

def redact_secrets(value: Any) -> Any:
    """Recursively redact credentials before they enter persistent logs.

    Redaction happens at ingestion and again at serialization so a future caller
    cannot accidentally reintroduce a credential into the evidence graph.
    """
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            norm = str(key).lower().replace("-", "_")
            if norm in _SECRET_KEYS or norm.endswith("_api_key") or norm.endswith("_token"):
                out[key] = "[REDACTED]"
            else:
                out[key] = redact_secrets(item)
        return out
    if isinstance(value, list):
        return [redact_secrets(x) for x in value]
    if isinstance(value, tuple):
        return tuple(redact_secrets(x) for x in value)
    if isinstance(value, str):
        text = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._~-]+", "Bearer [REDACTED]", value)
        text = re.sub(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}", "[REDACTED_JWT]", text)
        return text
    return value

def _is_primitive(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool)) or v is None


def prune_record(obj: Any, max_list: int = 30) -> Any:
    """Compact a record without knowledge of any API/domain vocabulary."""
    if not isinstance(obj, dict):
        return obj
    out: dict[str, Any] = {}
    for key, value in obj.items():
        if _is_primitive(value):
            out[key] = value
        elif isinstance(value, list):
            if len(value) <= max_list and all(_is_primitive(x) for x in value):
                out[key] = value
            # Arrays of objects are normalized into their own child observations
            # below. Do not synthesize conventional ``*_names``/``*_ids`` fields:
            # those conventions are API-specific and they also change the schema's
            # original field paths.
        elif isinstance(value, dict):
            # Preserve bounded nested scalar structure under the API's original
            # field name so a schema path such as ``meta.label`` remains replayable.
            nested = {k: v for k, v in value.items()
                      if _is_primitive(v) and v not in (None, "")}
            if nested:
                out[key] = dict(list(nested.items())[:12])
    return out


def _endpoint_source(endpoint: str) -> dict[str, Any]:
    """Deprecated compatibility hook; endpoint typing is assigned from plan templates."""
    del endpoint
    return {}


def _nested_limit(relation: str) -> int:
    """A uniform structural safety cap, independent of relation names."""
    del relation
    return 500


class ObservationLedger:
    """Immutable citable facts plus replayable derived observations."""

    def __init__(self):
        self.observations: list[dict[str, Any]] = []
        self.derived: list[dict[str, Any]] = []
        self.api_calls: list[dict[str, Any]] = []
        self.raw_responses: list[dict[str, Any]] = []
        self._counter = 0
        self._dcounter = 0
        self._call_counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"obs_{self._counter:03d}"

    def _next_did(self) -> str:
        self._dcounter += 1
        return f"der_{self._dcounter:03d}"

    def _next_call_id(self) -> str:
        self._call_counter += 1
        return f"call_{self._call_counter:03d}"

    def _values_in_ledger(self, values: Iterable[Any]) -> list[str]:
        """Find observations containing each supplied value, including list members."""
        ids: list[str] = []
        for wanted in values:
            for obs in self.observations:
                found = False
                for value in obs.get("fields", {}).values():
                    if value == wanted:
                        found = True
                    elif isinstance(value, (list, tuple, set)) and wanted in value:
                        found = True
                    elif isinstance(value, dict) and wanted in value.values():
                        found = True
                    if found:
                        break
                if found:
                    ids.append(obs["obs_id"])
                    break
        return list(dict.fromkeys(ids))

    # ------------------------------------------------------------------
    # Explicit derived-observation helpers (kept for backwards compatibility)
    # ------------------------------------------------------------------
    def derive_count(self, name, records, predicate=None,
                     capture="explicit_helper"):
        matched = [r for r in (records or [])
                   if predicate is None or predicate(r)]
        # Link to observations only when caller-supplied records contain values
        # that can be matched exactly; the count itself is API/domain agnostic.
        primitive = [v for r in matched if isinstance(r, dict)
                     for v in r.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool)]
        ids = self._values_in_ledger(primitive[:50])
        return self._record_derived(name, len(matched), "count", ids, capture)

    def derive_selection(self, name, records, sort_key, descending=True,
                         capture="explicit_helper"):
        recs = [r for r in (records or []) if isinstance(r, dict)]
        valid = [r for r in recs if r.get(sort_key) not in (None, "")]
        if not valid:
            return self._record_derived(name, None, "selection", [], capture)
        valid.sort(key=lambda r: r.get(sort_key), reverse=descending)
        chosen = valid[0]
        value = dict(chosen)
        primitive = [v for v in chosen.values()
                     if isinstance(v, (str, int, float)) and not isinstance(v, bool)]
        ids = self._values_in_ledger(primitive[:20])
        op = f"selection:{'max' if descending else 'min'}({sort_key})"
        return self._record_derived(name, value, op, ids, capture,
                                    extra={"record": chosen,
                                           "selected_obs_id": ids[0] if ids else None,
                                           "candidate_obs_ids": ids,
                                           "comparison_fields": [sort_key]})

    def derive_value(self, name, value, input_obs_ids=None, operation="value",
                     capture="explicit_helper"):
        return self._record_derived(name, value, operation,
                                    list(input_obs_ids or []), capture)

    def _record_derived(self, name, value, operation, input_obs_ids, capture,
                        extra=None):
        did = self._next_did()
        item = {
            "obs_id": did,
            "type": "derived",
            "name": name,
            "value": value,
            "operation": operation,
            "input_obs_ids": list(dict.fromkeys(input_obs_ids or [])),
            "capture_method": capture,
        }
        if extra:
            item.update(extra)
        self.derived.append(item)
        return value

    def add_derived_record(self, name: str, value: Any, operation: str,
                           input_obs_ids: list[str], **extra) -> str:
        """Record a host-side deterministic derivation and return its id."""
        self._record_derived(name, value, operation, input_obs_ids,
                             "auto_compiler", extra=extra)
        return self.derived[-1]["obs_id"]

    def get_derived(self, did):
        return next((d for d in self.derived if d["obs_id"] == did), None)

    def get(self, obs_id):
        return next((o for o in self.observations if o["obs_id"] == obs_id), None)

    # ------------------------------------------------------------------
    # Response normalization
    # ------------------------------------------------------------------
    def record_response(self, endpoint, params, payload, method="GET",
                        status_code=None, request_body=None, effective_endpoint=None,
                        adaptation=None, request_origin=None):
        """Record one API response and return the observation ids created.

        ``method`` and the request body are essential for action benchmarks: a
        successful 204 write has no response JSON, but still needs a citable,
        replayable action result.
        """
        call_id = self._next_call_id()
        method = str(method or "GET").upper()
        call = {"call_id": call_id, "endpoint": endpoint,
                "effective_endpoint": effective_endpoint or endpoint,
                "method": method, "status_code": status_code,
                "request_origin": request_origin,
                "params": redact_secrets(dict(params or {})),
                "request_body": redact_secrets(request_body)}
        if adaptation:
            call["adaptation"] = redact_secrets(adaptation)
        self.api_calls.append(call)
        raw_call = {**call, "payload": redact_secrets(payload)}
        self.raw_responses.append(raw_call)
        source = _endpoint_source(endpoint)

        def mark_truncated(pointer: str, relation: str, total: int, retained: int):
            marker = {"json_pointer": pointer or "/", "relation": relation,
                      "total": int(total), "retained": int(retained)}
            for target in (call, raw_call):
                target.setdefault("truncated_collections", []).append(dict(marker))
        created: list[str] = []

        def add_record(rec, kind="record", *, relation=None, position=None,
                       parent_obs_id=None, parent_record_id=None,
                       json_pointer="/"):
            if not isinstance(rec, dict):
                return None
            oid = self._next_id()
            obs = {
                "obs_id": oid, "call_id": call_id, "endpoint": endpoint,
                "kind": kind, "relation": relation, "position": position,
                "json_pointer": json_pointer, "record_id": rec.get("id"),
                "parent_obs_id": parent_obs_id,
                "parent_record_id": parent_record_id,
                "fields": prune_record(rec), **source,
            }
            self.observations.append(obs); created.append(oid); return oid

        if method != "GET" or payload is None:
            action_fields = {
                "method": method, "status_code": status_code,
                "successful": isinstance(status_code, int) and 200 <= status_code < 300,
                "request_params": redact_secrets(dict(params or {})),
                "request_body": redact_secrets(request_body),
                "effective_endpoint": effective_endpoint or endpoint,
            }
            if adaptation:
                action_fields["adaptation"] = redact_secrets(adaptation)
            add_record(action_fields, "action_result", relation="http_action", json_pointer="/")

        def expand(parent: dict, parent_oid: str | None, parent_record_id: Any,
                   pointer="", depth=0):
            if depth >= 2:
                return
            for key, value in parent.items():
                ptr = f"{pointer}/{key}"
                if isinstance(value, list) and all(isinstance(x, dict) for x in value):
                    limit = _nested_limit(key)
                    if len(value) > limit:
                        mark_truncated(ptr, str(key), len(value), limit)
                    for pos, item in enumerate(value[:limit]):
                        kind = "list_item" if depth == 0 else "nested_item"
                        child = add_record(item, kind, relation=key, position=pos,
                                           parent_obs_id=parent_oid,
                                           parent_record_id=parent_record_id,
                                           json_pointer=f"{ptr}/{pos}")
                        if child:
                            expand(item, child, item.get("id"), f"{ptr}/{pos}", depth + 1)
                elif isinstance(value, dict):
                    child = add_record(value, "nested_object", relation=key, position=0,
                                       parent_obs_id=parent_oid,
                                       parent_record_id=parent_record_id,
                                       json_pointer=ptr)
                    if child:
                        expand(value, child, value.get("id"), ptr, depth + 1)

        if isinstance(payload, dict):
            root = add_record(payload, "record", json_pointer="/")
            expand(payload, root, payload.get("id"), pointer="", depth=0)
        elif isinstance(payload, list):
            if len(payload) > 500 and all(isinstance(x, dict) for x in payload):
                mark_truncated("/", "$", len(payload), 500)
            for pos, item in enumerate(payload[:500]):
                if isinstance(item, dict):
                    oid = add_record(item, "list_item", relation="$", position=pos,
                                     json_pointer=f"/{pos}")
                    if oid:
                        expand(item, oid, item.get("id"), f"/{pos}", depth=1)
        return created

    def annotate_plan_steps(self, plan, matcher=None):
        """Attach plan-step ids and OAS-template path bindings to evidence."""
        if not plan:
            return
        steps = {str(x.get("id")): x for x in (plan.get("steps") or [])}
        if matcher is None:
            try:
                from utils.evidence_plan import assign_calls_to_steps
                assignments = assign_calls_to_steps(plan, self.api_calls)
            except Exception:
                assignments = {}
        else:
            assignments = {}
            remaining = list(plan.get("steps") or [])
            for call in self.api_calls:
                match = next((step for step in remaining
                              if matcher(call.get("endpoint", ""), step.get("endpoint", ""))), None)
                if match:
                    assignments[call["call_id"]] = match.get("id")
                    remaining.remove(match)
        call_to_step: dict[str, str] = {}
        call_bindings: dict[str, dict[str, Any]] = {}
        for call in self.api_calls:
            # Trusted runtime lineage is authoritative when available. The
            # one-to-one matcher is retained only for historical/imported logs.
            sid = call.get("runtime_step_id") or call.get("plan_step_id") or assignments.get(call["call_id"])
            if not sid:
                continue
            sid = str(sid); call_to_step[call["call_id"]] = sid
            call["plan_step_id"] = sid
            try:
                from utils.evidence_plan import extract_path_bindings
                bindings = extract_path_bindings(
                    str(call.get("endpoint") or ""),
                    str((steps.get(sid) or {}).get("endpoint") or ""))
            except Exception:
                bindings = {}
            if bindings:
                call["source_bindings"] = dict(bindings)
                call_bindings[call["call_id"]] = dict(bindings)
        for obs in self.observations:
            cid = obs.get("call_id")
            if cid in call_to_step:
                obs["plan_step_id"] = call_to_step[cid]
                if cid in call_bindings:
                    obs["source_bindings"] = dict(call_bindings[cid])
        for raw in self.raw_responses:
            cid = raw.get("call_id")
            if cid in call_to_step:
                raw["plan_step_id"] = call_to_step[cid]
                if cid in call_bindings:
                    raw["source_bindings"] = dict(call_bindings[cid])

    # ------------------------------------------------------------------
    # Compact view
    # ------------------------------------------------------------------
    def serialize(self, max_chars=24000, focus_obs_ids=None,
                  include_derived=True):
        """Serialize a provenance-preserving compact view for Phase B."""
        focus = set(focus_obs_ids or [])
        observations = self.observations
        if focus:
            selected = [o for o in observations if o["obs_id"] in focus]
            # Keep a small amount of context from each call even when focused.
            call_ids = {o.get("call_id") for o in selected}
            context = [o for o in observations
                       if o.get("call_id") in call_ids and o not in selected][:20]
            observations = selected + context

        lines: list[str] = []
        # Derived decisions are the authoritative compact representation. Put them
        # first so a large raw response cannot truncate them out of Phase B's view.
        if include_derived and self.derived:
            lines.append("DERIVED OBSERVATIONS (deterministically replayable):")
            for d in self.derived:
                value = d.get("value")
                text = repr(value)
                if len(text) > 800:
                    text = text[:800] + "..."
                lines.append(
                    f"{d['obs_id']} operation={d.get('operation')} name={d.get('name')} "
                    f"value={text} selected={d.get('selected_obs_id')} "
                    f"from={d.get('input_obs_ids', [])[:20]}")
            lines.append("")
            lines.append("RAW OBSERVATIONS:")
        for obs in observations:
            f = obs.get("fields", {})
            relation = f" relation={obs.get('relation')}" if obs.get("relation") else ""
            position = (f" position={obs.get('position')}"
                        if obs.get("position") is not None else "")
            step = (f" step={obs.get('plan_step_id')}"
                    if obs.get("plan_step_id") else "")

            def fmt(k, v):
                if isinstance(v, list):
                    suffix = "..." if len(v) > 15 else ""
                    return f"{k}={v[:15]!r}{suffix}"
                if isinstance(v, dict):
                    return f"{k}={json.dumps(v, ensure_ascii=False)}"
                return f"{k}={v!r}"

            shown = ", ".join(fmt(k, v) for k, v in f.items())
            lines.append(
                f"{obs['obs_id']} [{obs['endpoint']}] call={obs.get('call_id')}"
                f"{step}{relation}{position} {shown}".rstrip())

        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... (view truncated)"
        return text

    def __len__(self):
        return len(self.observations)


def reconstruct_provenance(ledger, stdout_text):
    """Conservatively promote fully supported printed lists/selections."""
    if not stdout_text:
        return 0
    raw_values = set()
    for obs in ledger.observations:
        for value in obs.get("fields", {}).values():
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                raw_values.add(value)
            elif isinstance(value, list):
                raw_values.update(x for x in value
                                  if isinstance(x, (str, int, float)))
    minted = 0
    for match in re.finditer(r"\[[^\[\]]*\]", stdout_text):
        try:
            value = ast.literal_eval(match.group(0))
        except Exception:
            continue
        if (isinstance(value, list) and len(value) >= 2 and
                all(isinstance(x, str) for x in value) and
                all(x in raw_values for x in value)):
            ids = ledger._values_in_ledger(value)
            ledger._record_derived("validated_list", value, "validated_list",
                                   ids, "validated_stdout")
            ledger._record_derived("validated_count", len(value),
                                   "validated_count", ids,
                                   "validated_stdout")
            minted += 2
    for line in stdout_text.splitlines():
        match = re.match(r"\s*(.+?)\s+(\d+(?:\.\d+)?)\s*$", line)
        if match:
            label = match.group(1).strip().strip("'\"")
            if label in raw_values:
                ids = ledger._values_in_ledger([label])
                ledger._record_derived("validated_selection", label,
                                       "validated_selection", ids,
                                       "validated_stdout")
                minted += 1
    return minted


class LedgerRequests:
    """Drop-in requests wrapper that records successful JSON GET responses."""

    def __init__(self, ledger, real_requests):
        self._ledger = ledger
        self._real = real_requests
        self.exceptions = real_requests.exceptions
        self.RequestException = real_requests.RequestException

    def get(self, url, **kwargs):
        response = self._real.get(url, **kwargs)
        try:
            match = re.search(r"https?://[^/]+(/[^?]*)", url)
            endpoint = match.group(1) if match else url
            payload = response.json()
            self._ledger.record_response(endpoint, kwargs.get("params"), payload)
        except Exception:
            pass
        return response

    def __getattr__(self, name):
        return getattr(self._real, name)


class EvidenceGraph:
    """Per-task evidence lineage written as JSONL."""

    def __init__(self, task_id, query, benchmark=None):
        self.task_id = task_id
        self.query = query
        self.benchmark = benchmark
        self.final_answer = None
        self.cited_observation_ids = []
        self.grounded = None
        self.coverage_note = None

    def finalize(self, ledger, final_answer, cited_ids):
        self.final_answer = final_answer
        self.cited_observation_ids = list(cited_ids or [])
        existing = {o["obs_id"] for o in ledger.observations}
        existing |= {d["obs_id"] for d in ledger.derived}
        self.grounded = (all(c in existing for c in self.cited_observation_ids)
                         if self.cited_observation_ids else None)
        return self

    def to_dict(self, ledger):
        return redact_secrets({
            "task_id": self.task_id,
            "benchmark": self.benchmark,
            "query": self.query,
            "final_answer": self.final_answer,
            "cited_observation_ids": self.cited_observation_ids,
            "grounded": self.grounded,
            "api_calls": ledger.api_calls,
            "observations": ledger.observations,
            "derived": ledger.derived,
        })

    def append_jsonl(self, ledger, path):
        import os
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.to_dict(ledger), ensure_ascii=False) + "\n")
