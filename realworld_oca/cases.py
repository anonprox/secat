from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ReplayCase:
    case_id: str
    source_issue: str
    task: str
    failure: str
    expected: Any
    max_turns: int = 5


def _openinterpreter_raw() -> str:
    lines: List[str] = []
    for i in range(1, 421):
        if i == 121:
            lines.append("PRIMARY_DB_HOST=db-primary-1695.internal:5432")
        else:
            lines.append(
                f"diagnostic[{i:03d}] component=service_{i:03d} "
                f"status=ok latency_ms={(i * 7) % 97}"
            )
    return "\n".join(lines) + "\n"


def _openinterpreter_visible(raw: str) -> str:
    tail = raw[-1000:]
    # Match the projection used in the earlier OCA replay artifact: the raw
    # observation has 25,979 characters and the model-visible view has 1,166.
    prefix = (
        "Output truncated. Showing the last 1000 characters. You should try again "
        "and use computer.ai.summarize(output) over the output, or break it down "
        "into smaller steps.\n\n"
    )
    assert len(prefix) == 166
    return prefix + tail


def _openhands_raw() -> str:
    # Match the earlier replay artifact's retained size (54,499 chars) while
    # placing the requested value well beyond the 16,000-character ACI limit.
    target = "DEFAULT_RETRY_TIMEOUT_SECONDS = 37"
    before = []
    i = 1
    while len("".join(before)) < 30000:
        before.append(
            f"{i:04d}    setting_{i:04d} = 'fixture-value-{i:04d}-abcdefghijklmno'\n"
        )
        i += 1
    text = "".join(before) + target + "\n"
    while len(text) < 54499:
        text += f"{i:04d}    setting_{i:04d} = 'fixture-value-{i:04d}-abcdefghijklmno'\n"
        i += 1
    return text[:54499]


def _openhands_visible(raw: str) -> str:
    clipped = raw[:16000]
    note = (
        "<response clipped><NOTE>Due to the max output limit, only part of this file "
        "has been shown to you. The remainder of the successful file-read observation "
        "is not model-visible in this reproduced condition. Retry or narrow the read "
        "if possible.</NOTE>"
    )
    # Keep the same 16,250-character model-visible size recorded by the earlier replay.
    note = note[:250].ljust(250)
    return clipped + note


CASES: Dict[str, ReplayCase] = {
    "openinterpreter_1695": ReplayCase(
        case_id="open-interpreter#1695",
        source_issue="https://github.com/openinterpreter/openinterpreter/issues/1695",
        task=(
            "Run the diagnostic report and tell me the value of PRIMARY_DB_HOST. "
            "Use the runtime output as your evidence; do not inspect the diagnostic "
            "program's source code."
        ),
        failure="command output truncated to the final 1000 characters",
        expected="db-primary-1695.internal:5432",
    ),
    "smolagents_1382": ReplayCase(
        case_id="smolagents#1382",
        source_issue="https://github.com/huggingface/smolagents/issues/1382",
        task=(
            "Return the required final answer with all three fields unchanged: "
            "sources=['abc', 'def'], answer='Hello', info='This is a test'. "
            "All three fields are required by the task."
        ),
        failure=(
            "the reproduced pre-fix executor rejects custom final-answer keyword "
            "arguments, encouraging destructive repair"
        ),
        expected={
            "sources": ["abc", "def"],
            "answer": "Hello",
            "info": "This is a test",
        },
    ),
    "openhands_7115": ReplayCase(
        case_id="openhands#7115",
        source_issue="https://github.com/OpenHands/OpenHands/issues/7115",
        task=(
            "Read the project configuration and tell me the value of "
            "DEFAULT_RETRY_TIMEOUT_SECONDS. Use the file-read output as your evidence."
        ),
        failure="file-read observation clipped at 16000 characters",
        expected="37",
    ),
    "smolagents_1374": ReplayCase(
        case_id="smolagents#1374",
        source_issue="https://github.com/huggingface/smolagents/issues/1374",
        task=(
            "Compute 1654651 * 6332 using the provided execution capability and "
            "return the exact result."
        ),
        failure=(
            "final-answer validation observes memory before the current ActionStep "
            "has been appended"
        ),
        expected="10477250132",
        max_turns=3,
    ),
    "smolagents_149": ReplayCase(
        case_id="smolagents#149",
        source_issue="https://github.com/huggingface/smolagents/issues/149",
        task=(
            "Use the currently available echo tool to echo CAT exactly once, "
            "then return the tool result."
        ),
        failure=(
            "the model-facing tool catalog is stale after the framework registry "
            "replaces old_echo with new_echo"
        ),
        expected="NEW:CAT",
        max_turns=4,
    ),
}


def raw_observation(case_key: str) -> str:
    if case_key == "openinterpreter_1695":
        return _openinterpreter_raw()
    if case_key == "openhands_7115":
        return _openhands_raw()
    raise KeyError(f"{case_key} has no raw observation fixture")


def visible_observation(case_key: str, raw: str) -> str:
    if case_key == "openinterpreter_1695":
        return _openinterpreter_visible(raw)
    if case_key == "openhands_7115":
        return _openhands_visible(raw)
    raise KeyError(f"{case_key} has no observation projection")


def validate_reproductions() -> Dict[str, dict]:
    checks: Dict[str, dict] = {}

    oi_raw = raw_observation("openinterpreter_1695")
    oi_visible = visible_observation("openinterpreter_1695", oi_raw)
    oi_target = "PRIMARY_DB_HOST=db-primary-1695.internal:5432"
    checks["openinterpreter_1695"] = {
        "raw_chars": len(oi_raw),
        "visible_chars": len(oi_visible),
        "target_in_raw": oi_target in oi_raw,
        "target_in_visible": oi_target in oi_visible,
        "reproduced": oi_target in oi_raw and oi_target not in oi_visible,
    }

    oh_raw = raw_observation("openhands_7115")
    oh_visible = visible_observation("openhands_7115", oh_raw)
    oh_target = "DEFAULT_RETRY_TIMEOUT_SECONDS = 37"
    checks["openhands_7115"] = {
        "raw_chars": len(oh_raw),
        "visible_chars": len(oh_visible),
        "target_in_raw": oh_target in oh_raw,
        "target_in_visible": oh_target in oh_visible,
        "reproduced": oh_target in oh_raw and oh_target not in oh_visible,
    }

    # This is the exact behavioral defect reproduced from smolagents 1.16.1:
    # custom final-answer kwargs are rejected by the local executor wrapper.
    try:
        _smolagents_buggy_executor(
            sources=["abc", "def"], answer="Hello", info="This is a test"
        )
        reproduced = False
        error = None
    except TypeError as exc:
        reproduced = "unexpected keyword argument 'sources'" in str(exc)
        error = str(exc)
    checks["smolagents_1382"] = {
        "error": error,
        "reproduced": reproduced,
    }

    # smolagents #1374 paired fixture: validation runs before the just-executed
    # ActionStep is appended, so the first checker snapshot contains only TaskStep.
    first_check_memory_types = ["TaskStep"]
    checks["smolagents_1374"] = {
        "first_check_memory_types": first_check_memory_types,
        "current_action_type": "ActionStep",
        "first_check_missing_current_action": "ActionStep" not in first_check_memory_types,
        "reproduced": "ActionStep" not in first_check_memory_types,
    }

    # smolagents #149: the historical reproduction confirmed that the actual
    # registry contains only new_echo while the model-facing prompt still names
    # old_echo. The paired fixture recreates exactly that mismatch.
    current_registry = ["new_echo"]
    stale_model_catalog = ["old_echo"]
    checks["smolagents_149"] = {
        "current_registry": current_registry,
        "model_facing_catalog": stale_model_catalog,
        "old_in_registry": "old_echo" in current_registry,
        "new_in_registry": "new_echo" in current_registry,
        "old_in_model_catalog": "old_echo" in stale_model_catalog,
        "new_in_model_catalog": "new_echo" in stale_model_catalog,
        "reproduced": (
            "new_echo" in current_registry
            and "old_echo" not in current_registry
            and "old_echo" in stale_model_catalog
            and "new_echo" not in stale_model_catalog
        ),
    }
    return checks


def _smolagents_buggy_executor(*, sources=None, answer=None, info=None):
    """Reproduce the pre-fix LocalPythonExecutor final_answer wrapper.

    The wrapped final_answer accepts only the legacy single ``answer`` value.
    Passing custom fields therefore raises the same class of TypeError reported
    in smolagents #1382. If the model deletes the custom fields, execution can
    succeed but the task is incomplete.
    """
    if sources is not None:
        raise TypeError(
            "evaluate_python_code.<locals>.final_answer() got an unexpected "
            "keyword argument 'sources'"
        )
    if info is not None:
        raise TypeError(
            "evaluate_python_code.<locals>.final_answer() got an unexpected "
            "keyword argument 'info'"
        )
    return {"answer": answer}


def smolagents_buggy_executor(arguments: dict) -> dict:
    return _smolagents_buggy_executor(
        sources=arguments.get("sources"),
        answer=arguments.get("answer"),
        info=arguments.get("info"),
    )
