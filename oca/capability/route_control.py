"""Capability sentinel helpers used by native intent compilation."""
from __future__ import annotations
from typing import Any

SENTINEL = "__cap__"


def sentinel_for(semantic_name: str) -> str:
    return f"{SENTINEL}{semantic_name}"


def capability_from(relation: Any) -> str:
    text = str(relation or "")
    return text[len(SENTINEL):] if text.startswith(SENTINEL) else ""
