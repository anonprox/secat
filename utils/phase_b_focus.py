"""Deterministic observation selection for OCA Phase B."""
from __future__ import annotations


def focus_obs_ids(ledger, plan, compiled=None, *, enabled=False, dep_context=6):
    """Return a provenance-safe focused observation id list, or ``None``.

    Compiler focus wins. When fallback is enabled, keep all answer-step facts,
    deterministic derivation anchors, and a bounded identity prefix from dependency
    steps. No factual values are synthesized or re-ranked here.
    """
    focus = list((compiled or {}).get("focus_obs_ids") or [])
    if focus or not enabled:
        return focus or None

    steps = {str(s.get("id")): s for s in (plan or {}).get("steps") or [] if s.get("id")}
    answer_steps = set(str(x) for x in ((plan or {}).get("answer_steps") or []) if x)
    wanted = set(answer_steps)
    frontier = list(answer_steps)
    while frontier:
        sid = frontier.pop()
        step = steps.get(sid) or {}
        for dep in step.get("depends_on") or []:
            dep = str(dep)
            if dep and dep not in wanted:
                wanted.add(dep)
                frontier.append(dep)
    if not wanted:
        return None

    ids = []
    for obs in ledger.observations:
        if str(obs.get("plan_step_id") or "") in answer_steps and obs.get("obs_id"):
            ids.append(obs["obs_id"])

    for derived in ledger.derived:
        selected = derived.get("selected_obs_id")
        if selected:
            ids.append(str(selected))
        for oid in (derived.get("input_obs_ids") or [])[:12]:
            ids.append(str(oid))

    limit = max(0, int(dep_context))
    for sid in sorted(wanted - answer_steps):
        step_obs = [o for o in ledger.observations
                    if str(o.get("plan_step_id") or "") == sid and o.get("obs_id")]
        ids.extend(o["obs_id"] for o in step_obs[:limit])

    return list(dict.fromkeys(ids)) or None
