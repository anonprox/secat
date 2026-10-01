"""Evaluation-boundary helpers.

The benchmark loader may retain gold-only metadata for post-run scoring.  Agents
must not receive that metadata in an evaluation-isolated run.  This module keeps
the two concerns explicit and makes the historical reported-95 configuration
reproducible without silently using it as the default.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any

# Fields derived from a benchmark solution, expected answer, or evaluator.  They
# may remain in the runner's oracle copy, but never cross the isolated agent
# boundary.
GOLD_ONLY_FIELDS = frozenset({
    "api_list", "solution", "solution_steps", "http_methods", "has_writes",
    "read_only_task", "gold_path", "gold_answer", "ground_truth", "expected", "expected_answer",
    "reference_answer", "answer", "oracle", "verify_status", "exclude_accuracy",
})


def runtime_benchmark_name(name: str | None) -> str:
    """Return the configured runtime API domain without prefix heuristics."""
    if not name:
        raise ValueError("isolated evaluation requires an explicit benchmark/API configuration")
    value = str(name).strip()
    try:
        import benchmarks as B
        spec = B.get_benchmark(value)
        return str(spec.get("runtime_api") or spec.get("name") or value)
    except Exception:
        return value


def opaque_task_id(task: dict[str, Any], ordinal: int | None = None) -> str:
    """Stable, answer-free id that does not expose benchmark prefixes/numbers."""
    instruction = str(task.get("instruction") or task.get("query") or "")
    seed = f"{ordinal or 0}\0{instruction}".encode("utf-8", errors="replace")
    return "eval_" + hashlib.sha256(seed).hexdigest()[:12]


def sanitize_task_for_agent(task: dict[str, Any], *, profile: str = "isolated",
                            ordinal: int | None = None,
                            runtime_benchmark: str | None = None
                            ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an agent-visible task and a non-secret boundary audit record.

    ``reported95`` intentionally preserves the uploaded v3.3.1 agent-visible
    benchmark-assistance boundary for compatibility.  It still uses the current
    validators; the bundled historical source snapshot is the exact old code.  ``isolated`` is the defensible evaluation
    mode and exposes only an opaque id plus the natural-language request. The
    runtime API/domain is infrastructure configuration supplied out-of-band.
    """
    profile = str(profile or "isolated").strip().lower()
    if profile not in {"isolated", "reported95"}:
        raise ValueError(f"unknown evaluation profile: {profile!r}")

    if profile == "reported95":
        visible = copy.deepcopy(task)
        return visible, {
            "profile": profile,
            "runtime_task_id": str(visible.get("id") or ""),
            "agent_visible_fields": sorted(visible),
            "gold_fields_removed": [],
            "historical_nonisolated": True,
        }

    instruction = str(task.get("instruction") or task.get("query") or "").strip()
    runtime_id = opaque_task_id(task, ordinal)
    # Runtime API/auth configuration is intentionally *not* part of the task
    # object. The runner supplies it out-of-band through the trusted runtime.
    # This keeps benchmark/provider labels from becoming accidental prompt hints.
    runtime_benchmark_name(runtime_benchmark or task.get("benchmark"))  # fail closed if absent
    visible: dict[str, Any] = {
        "id": runtime_id,
        "instruction": instruction,
        "query": instruction,
    }

    removed = sorted(k for k in task if k in GOLD_ONLY_FIELDS)
    return visible, {
        "profile": profile,
        "runtime_task_id": runtime_id,
        "agent_visible_fields": sorted(visible),
        "gold_fields_removed": removed,
        "historical_nonisolated": False,
    }


def assert_isolated_task(task: dict[str, Any]) -> None:
    leaked = sorted(GOLD_ONLY_FIELDS.intersection(task))
    if leaked:
        raise RuntimeError(f"gold-only fields crossed the agent boundary: {leaked}")
