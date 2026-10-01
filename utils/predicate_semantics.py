"""Canonical predicate semantics shared by OCA host-side components.

The isolated kernel embeds a small source copy of this logic because it cannot
import project modules after isolation.  `tests/test_oca_semantic_parity.py`
keeps that unavoidable copy behaviorally aligned with this implementation.
"""
from __future__ import annotations

from typing import Any

SUPPORTED_PREDICATE_OPS = frozenset({
    "eq", "eq_ci", "neq", "neq_ci", "contains", "contains_ci",
    "startswith_ci", "exists", "not_exists", "gt", "gte", "lt", "lte", "in",
})


def compare_predicate(value: Any, op: str, expected: Any) -> bool:
    """Evaluate one projection/filter predicate with fail-closed semantics."""
    op = str(op or "eq").lower()
    if op not in SUPPORTED_PREDICATE_OPS:
        return False
    if isinstance(value, list):
        if op == "exists":
            return bool(value)
        if op == "not_exists":
            return not value
        if op in {"neq", "neq_ci"}:
            return all(compare_predicate(v, op, expected) for v in value)
        return any(compare_predicate(v, op, expected) for v in value)
    if op == "exists":
        return value is not None
    if op == "not_exists":
        return value is None
    if op in {"eq_ci", "neq_ci", "contains_ci", "startswith_ci"}:
        left = "" if value is None else str(value).casefold()
        right = "" if expected is None else str(expected).casefold()
        if op == "eq_ci":
            return left == right
        if op == "neq_ci":
            return left != right
        if op == "contains_ci":
            return right in left
        return left.startswith(right)
    if op == "contains":
        try:
            return expected in value
        except Exception:
            return str(expected) in str(value)
    if op == "in":
        try:
            return value in expected
        except Exception:
            return False
    if op == "eq":
        return value == expected
    if op == "neq":
        return value != expected
    try:
        if op == "gt":
            return value > expected
        if op == "gte":
            return value >= expected
        if op == "lt":
            return value < expected
        if op == "lte":
            return value <= expected
    except Exception:
        return False
    return False
