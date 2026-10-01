"""Run-level Spotify benchmark policy for the 2026 Development Mode surface.

This module is trusted benchmark orchestration metadata.  It is never exposed to
an agent and is deliberately separate from OCA/CodeAct/ToolCoder reasoning.
The classifications document whether a historical RestBench task can be made
reproducible on a fresh 2026 Development Mode account.
"""
from __future__ import annotations

# Historical routes removed from 2026 Development Mode with no faithful route
# replacement that preserves the RestBench solution semantics.
DEV2026_UNSUPPORTED_ENDPOINT = frozenset({23, 31, 36})

# Spotify exposes these read surfaces, but the benchmark fixture cannot create
# the required user history. /me/top/* is an affinity product of accumulated
# listening behavior, and /me/player/recently-played is playback history; neither
# has an API setter that can reproduce the task precondition deterministically.
DEV2026_NONFIXTURABLE_HISTORY = frozenset({10, 24, 25, 34})

# These tasks name a particular playback device ("My PC"). The Web API can list
# devices but cannot create/rename a device as benchmark setup.
DEV2026_NONFIXTURABLE_DEVICE = frozenset({28, 29})

# Historical task 21 assumes genre information can be recovered from current
# playback and used for same-genre discovery. On current Dev Mode, the playback
# object carries simplified artists, recommendation access is restricted, and
# artist genre metadata is not reliably populated. Keep it as a diagnostic task,
# not in the reproducible comparison set.
DEV2026_SEMANTICALLY_DEGRADED = frozenset({21})

DEV2026_STRICT_EXCLUDED = frozenset().union(
    DEV2026_UNSUPPORTED_ENDPOINT,
    DEV2026_NONFIXTURABLE_HISTORY,
    DEV2026_NONFIXTURABLE_DEVICE,
    DEV2026_SEMANTICALLY_DEGRADED,
)

DEV2026_ROUTE_COMPATIBLE_TASKS = tuple(
    i for i in range(1, 58) if i not in DEV2026_UNSUPPORTED_ENDPOINT
)
DEV2026_STRICT_COMPATIBLE_TASKS = tuple(
    i for i in range(1, 58) if i not in DEV2026_STRICT_EXCLUDED
)
DEV2026_DIAGNOSTIC_READONLY_TASKS = (17, 19, 21, 24, 25, 27, 30, 56)
DEV2026_STRICT_READONLY_TASKS = tuple(
    i for i in DEV2026_DIAGNOSTIC_READONLY_TASKS if i not in DEV2026_STRICT_EXCLUDED
)


def task_spec(ids) -> str:
    """Return the runner's comma-separated exact task specification."""
    return ",".join(str(int(x)) for x in ids)


def classification(task_id: int) -> str:
    i = int(task_id)
    if i in DEV2026_UNSUPPORTED_ENDPOINT:
        return "unsupported_endpoint"
    if i in DEV2026_NONFIXTURABLE_HISTORY:
        return "nonfixturable_history"
    if i in DEV2026_NONFIXTURABLE_DEVICE:
        return "nonfixturable_device"
    if i in DEV2026_SEMANTICALLY_DEGRADED:
        return "semantically_degraded"
    return "strict_compatible"
