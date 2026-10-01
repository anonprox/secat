"""Generic plan-progress diagnostics for OCA.

v4.1.62 stabilization note
-------------------------
The v4.1.59 experimental binding-consistency guard attempted to turn a
previously-complete step back into a missing step by appending the step *id*
to ``progress['missing_steps']``. The native audit schema stores full step
dictionaries in that list. Mixing strings into that list caused deterministic
plan completion to crash when it later evaluated ``step.get(...)``.

Until binding invalidation is reimplemented at a layer that has access to the
full plan-step objects, the mutation is deliberately disabled. Mismatch
detection is retained for diagnostics/tests, but ``enforce_binding_consistency``
is a strict identity/no-op. This restores pre-v4.1.59 runtime semantics while
keeping the independently-tested terminal-selector preservation from v4.1.60.
"""
from __future__ import annotations
from typing import Any


def _canon_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if value is None:
        return "<none>"
    return str(value)


def _canon_values(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set, frozenset)):
        return {_canon_scalar(v) for v in value}
    return {_canon_scalar(value)}


def binding_mismatches(progress: Any) -> dict[str, dict[str, dict[str, list[str]]]]:
    """Detect concrete expected bindings not covered by executed calls."""
    if not isinstance(progress, dict):
        return {}
    step_status = progress.get("step_status")
    if not isinstance(step_status, dict):
        return {}
    out: dict[str, dict[str, dict[str, list[str]]]] = {}
    for step_id, status in step_status.items():
        if not isinstance(status, dict):
            continue
        expected = status.get("expected_bindings")
        covered = status.get("covered_bindings")
        if not isinstance(expected, dict) or not expected:
            continue
        covered = covered if isinstance(covered, dict) else {}
        step_mismatches: dict[str, dict[str, list[str]]] = {}
        for name, raw_expected in expected.items():
            exp = _canon_values(raw_expected)
            if not exp:
                continue
            cov = _canon_values(covered.get(name))
            if not exp.issubset(cov):
                step_mismatches[str(name)] = {
                    "expected": sorted(exp),
                    "covered": sorted(cov),
                }
        if step_mismatches:
            out[str(step_id)] = step_mismatches
    return out


def enforce_binding_consistency(progress: Any) -> Any:
    """Return *progress* unchanged; v4.1.59 mutation is intentionally disabled."""
    return progress
