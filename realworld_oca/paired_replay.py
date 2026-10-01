from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from realworld_oca.cases import (
    CASES,
    raw_observation,
    smolagents_buggy_executor,
    validate_reproductions,
    visible_observation,
)


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except Exception:
        pass


def _tool(name: str, description: str, properties: dict, required: Optional[List[str]] = None) -> dict:
    params = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        params["required"] = required
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": params,
        },
    }


def _tools(case_key: str, mode: str) -> List[dict]:
    if case_key == "openinterpreter_1695":
        tools = [
            _tool("run_diagnostic", "Run the diagnostic report.", {}),
        ]
        if mode == "oca":
            tools.append(_tool(
                "search_observation",
                "Search the host-retained complete execution observation for exact evidence.",
                {
                    "query": {"type": "string"},
                },
                ["query"],
            ))
        tools.append(_tool(
            "submit_answer",
            "Submit the final answer.",
            {"answer": {"type": "string"}},
            ["answer"],
        ))
        return tools

    if case_key == "openhands_7115":
        tools = [
            _tool("read_project_file", "Read the project configuration file.", {}),
        ]
        if mode == "oca":
            tools.append(_tool(
                "search_observation",
                "Search the host-retained complete file-read observation for exact evidence.",
                {"query": {"type": "string"}},
                ["query"],
            ))
        tools.append(_tool(
            "submit_answer",
            "Submit the final answer.",
            {"answer": {"type": "string"}},
            ["answer"],
        ))
        return tools

    if case_key == "smolagents_1382":
        if mode == "baseline":
            # Keep custom fields optional at the protocol boundary, as in the
            # failure: after an execution TypeError the model can delete them.
            return [_tool(
                "final_answer",
                "Submit the final answer. The user task states which fields are required.",
                {
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "answer": {"type": "string"},
                    "info": {"type": "string"},
                },
            )]
        return [_tool(
            "commit_answer",
            "Submit the final answer while preserving every required task field.",
            {
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "answer": {"type": "string"},
                "info": {"type": "string"},
            },
            ["sources", "answer", "info"],
        )]

    if case_key == "smolagents_1374":
        # Same model-facing tool in both modes. The difference is only when the
        # framework validates completion relative to the just-executed action.
        return [_tool(
            "calculate_and_submit",
            "Multiply the supplied integers in the runtime and submit the computed result.",
            {
                "left": {"type": "integer"},
                "right": {"type": "integer"},
            },
            ["left", "right"],
        )]

    if case_key == "smolagents_149":
        # Keep the protocol surface identical in baseline and OCA. The model
        # chooses a tool name as data; the host executes only against the actual
        # current registry, which contains new_echo and not old_echo.
        return [
            _tool(
                "call_tool",
                "Execute a named capability from the model-facing capability catalog.",
                {
                    "name": {"type": "string"},
                    "text": {"type": "string"},
                },
                ["name", "text"],
            ),
            _tool(
                "submit_answer",
                "Submit the final answer after the required tool has executed.",
                {"answer": {"type": "string"}},
                ["answer"],
            ),
        ]

    raise KeyError(case_key)


def _system_prompt(case_key: str, mode: str) -> str:
    common = (
        "Complete the user's task using only the supplied tools and observations. "
        "Do not use prior knowledge to invent a value that is not visible in the tool evidence. "
        "When you have the answer, submit it with the final-answer tool. "
        "If an execution error occurs, inspect the error before deciding what to do next."
    )
    if case_key == "smolagents_1374":
        # Keep baseline and OCA prompts identical; only the framework state used
        # by final validation differs.
        return common
    if case_key == "smolagents_149":
        if mode == "baseline":
            return common + (
                " The model-facing tool catalog says the available echo capability is old_echo. "
                "Use a capability named in this catalog when the user asks for the echo tool."
            )
        return common + (
            " The framework has validated the current executable capability registry. "
            "The currently available echo capability is new_echo; old_echo has been removed. "
            "Only plan or execute capabilities present in this current registry."
        )
    if mode == "baseline":
        return common
    return common + (
        " The host retains complete execution observations even when the model-visible view is clipped. "
        "If required evidence is missing because an observation is truncated, use search_observation. "
        "Do not repair an execution error by deleting a field that the user explicitly requires."
    )


def _message_to_dict(message: Any) -> dict:
    content = getattr(message, "content", None)
    out: Dict[str, Any] = {"role": "assistant", "content": content or ""}
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        out["tool_calls"] = []
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            out["tool_calls"].append({
                "id": getattr(tc, "id", ""),
                "type": "function",
                "function": {
                    "name": getattr(fn, "name", ""),
                    "arguments": getattr(fn, "arguments", "{}") or "{}",
                },
            })
    return out


def _usage_dict(resp: Any) -> dict:
    usage = getattr(resp, "usage", None)
    if usage is None:
        raise RuntimeError(
            "Provider response omitted token usage. RQ4 token comparison requires provider-reported usage; "
            "the runner will not substitute an estimate."
        )
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", prompt + completion) or (prompt + completion))
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _normalize_answer(value: Any) -> str:
    return str(value or "").strip().strip("`\"'")


def _evaluate(case_key: str, submitted: Any, state: Optional[dict] = None) -> dict:
    case = CASES[case_key]
    if case_key == "smolagents_1382":
        got = submitted if isinstance(submitted, dict) else {}
        passed = (
            got.get("sources") == case.expected["sources"]
            and got.get("answer") == case.expected["answer"]
            and got.get("info") == case.expected["info"]
        )
        return {"passed": bool(passed), "expected": case.expected, "observed": got}

    if case_key == "smolagents_1374":
        observed = _normalize_answer(submitted)
        state = state or {}
        first_missing = bool(state.get("first_check_missing_current_action", False))
        passed = case.expected in observed and not first_missing
        return {
            "passed": bool(passed),
            "expected": case.expected,
            "observed": observed,
            "first_check_missing_current_action": first_missing,
            "check_events": state.get("final_check_events", []),
        }

    if case_key == "smolagents_149":
        observed = _normalize_answer(submitted)
        state = state or {}
        executed = state.get("executed_capabilities", [])
        passed = (
            case.expected in observed
            and executed == ["new_echo"]
        )
        return {
            "passed": bool(passed),
            "expected": case.expected,
            "observed": observed,
            "executed_capabilities": executed,
            "required_execution": ["new_echo"],
        }

    observed = _normalize_answer(submitted)
    expected = str(case.expected)
    # Accept the expected scalar embedded in a concise answer such as
    # "PRIMARY_DB_HOST=db-primary..." or "DEFAULT... = 37".
    passed = expected in observed
    return {"passed": bool(passed), "expected": expected, "observed": observed}


def _execute_tool(case_key: str, mode: str, name: str, args: dict, state: dict) -> dict:
    if case_key in {"openinterpreter_1695", "openhands_7115"}:
        if name in {"run_diagnostic", "read_project_file"}:
            raw = raw_observation(case_key)
            state["retained_raw"] = raw
            visible = visible_observation(case_key, raw)
            return {
                "truncated": True,
                "output": visible,
                "raw_chars_retained_by_host": len(raw) if mode == "oca" else 0,
            }
        if name == "search_observation":
            if mode != "oca":
                return {"error": "search_observation is unavailable"}
            raw = state.get("retained_raw") or ""
            query = str(args.get("query") or "")
            hits = [line for line in raw.splitlines() if query.lower() in line.lower()]
            return {"query": query, "matches": hits[:10]}
        if name == "submit_answer":
            state["submitted"] = args.get("answer")
            state["done"] = True
            return {"accepted": True}

    if case_key == "smolagents_1382":
        if name == "final_answer" and mode == "baseline":
            try:
                result = smolagents_buggy_executor(args)
                state["submitted"] = result
                state["done"] = True
                return {"accepted": True, "result": result}
            except TypeError as exc:
                return {"accepted": False, "error": f"TypeError: {exc}"}
        if name == "commit_answer" and mode == "oca":
            submitted = {
                "sources": args.get("sources"),
                "answer": args.get("answer"),
                "info": args.get("info"),
            }
            state["submitted"] = submitted
            state["done"] = True
            return {"accepted": True, "result": submitted}

   
    if case_key == "smolagents_1374":
        if name == "calculate_and_submit":
            left = int(args.get("left"))
            right = int(args.get("right"))
            computed = str(left * right)
            check_no = int(state.get("final_check_count", 0)) + 1
            state["final_check_count"] = check_no

            if mode == "baseline":
                # Faithful failure ordering: the checker sees committed memory
                # before the current action is appended. On the first attempt it
                # therefore sees only TaskStep. A later retry can be accepted
                # because a previous ActionStep is then present, matching the
                # historical behavior observed in the reproduction.
                memory_types = ["TaskStep"] if check_no == 1 else ["TaskStep", "ActionStep(previous)"]
                current_visible = False
            else:
                # OCA-style verification checks the just-executed evidence/state.
                memory_types = ["TaskStep", "ActionStep(current)"]
                current_visible = True

            state.setdefault("final_check_events", []).append({
                "check": check_no,
                "memory_types": memory_types,
                "current_action_visible": current_visible,
                "computed_result": computed,
            })

            if check_no == 1:
                state["first_check_missing_current_action"] = not current_visible

            if mode == "baseline" and check_no == 1:
                return {
                    "accepted": False,
                    "computed_result": computed,
                    "error": (
                        "final-answer critic rejected completion because the "
                        "current ActionStep is absent from checker memory"
                    ),
                }

            state["submitted"] = computed
            state["done"] = True
            return {"accepted": True, "result": computed}

    if case_key == "smolagents_149":
        if name == "call_tool":
            requested = str(args.get("name") or "")
            text = str(args.get("text") or "")
            if requested != "new_echo":
                return {
                    "accepted": False,
                    "error": f"unavailable capability: {requested}",
                }
            result = f"NEW:{text}"
            state.setdefault("executed_capabilities", []).append("new_echo")
            state["last_tool_result"] = result
            return {"accepted": True, "result": result}
        if name == "submit_answer":
            state["submitted"] = args.get("answer")
            state["done"] = True
            return {"accepted": True}

    return {"error": f"unknown tool {name}"}


def _make_client():
    from agents.common import make_client
    return make_client()


def run_one(case_key: str, mode: str, model: str, output_dir: Path) -> dict:
    if case_key not in CASES:
        raise KeyError(f"unknown case {case_key}")
    if mode not in {"baseline", "oca"}:
        raise ValueError("mode must be baseline or oca")

    from utils.model_provider import provider_for_model
    if provider_for_model(model) != "openai":
        raise RuntimeError(
            "The paired RQ4 runner currently uses function-tool calls and is intentionally restricted "
            "to OpenAI models. Use gpt-5.4-mini for the paper comparison."
        )

    from utils.token_meter import reset as reset_tokens, read as read_tokens, stage as token_stage

    case = CASES[case_key]
    reset_tokens()
    client = _make_client()
    tools = _tools(case_key, mode)
    messages: List[dict] = [
        {"role": "system", "content": _system_prompt(case_key, mode)},
        {"role": "user", "content": case.task},
    ]
    state: Dict[str, Any] = {"done": False, "submitted": None, "retained_raw": None, "executed_capabilities": []}
    trace: List[dict] = []

    for turn in range(1, case.max_turns + 1):
        with token_stage(f"rq4_{mode}"):
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_completion_tokens=2048,
            )
        usage = _usage_dict(resp)
        message = resp.choices[0].message
        assistant = _message_to_dict(message)
        messages.append(assistant)
        turn_entry = {
            "turn": turn,
            "usage": usage,
            "assistant_text": assistant.get("content") or "",
            "tool_calls": [],
        }

        tool_calls = assistant.get("tool_calls") or []
        if not tool_calls:
            # If the model stops without a tool call, preserve its text as the
            # attempted answer; the evaluator decides whether it is correct.
            if assistant.get("content"):
                state["submitted"] = assistant["content"]
            trace.append(turn_entry)
            break

        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _execute_tool(case_key, mode, name, args, state)
            turn_entry["tool_calls"].append({"name": name, "arguments": args, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result, ensure_ascii=False),
            })
        trace.append(turn_entry)
        if state.get("done"):
            break

    tokens = read_tokens()
    evaluation = _evaluate(case_key, state.get("submitted"), state)
    result = {
        "case_key": case_key,
        "case_id": case.case_id,
        "source_issue": case.source_issue,
        "model": model,
        "model_source": (
            "fallback_no_original_model" if case_key == "smolagents_149" else "controlled_paired_replay"
        ),
        "original_issue_model": (
            "claude-3-7-sonnet-20250219"
            if case_key == "smolagents_1374"
            else None
        ),
        "historical_reproduction_model": (
            "anthropic/claude-sonnet-4-6"
            if case_key == "smolagents_1374"
            else None
        ),
        "mode": mode,
        "failure": case.failure,
        "task": case.task,
        "max_turns": case.max_turns,
        "turns": len(trace),
        "tokens": {
            "prompt_tokens": int(tokens.get("prompt_tokens", 0)),
            "completion_tokens": int(tokens.get("completion_tokens", 0)),
            "total_tokens": int(tokens.get("total_tokens", 0)),
            "calls": int(tokens.get("calls", 0)),
        },
        "evaluation": evaluation,
        "trace": trace,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{case_key}__{mode}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _summary(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    by_case: Dict[str, dict] = {}
    for row in rows:
        case = by_case.setdefault(row["case_key"], {"case_id": row["case_id"]})
        case[row["mode"]] = {
            "passed": row["evaluation"]["passed"],
            "prompt_tokens": row["tokens"]["prompt_tokens"],
            "completion_tokens": row["tokens"]["completion_tokens"],
            "total_tokens": row["tokens"]["total_tokens"],
            "calls": row["tokens"]["calls"],
            "turns": row["turns"],
        }
    for case in by_case.values():
        if "baseline" in case and "oca" in case:
            bt = case["baseline"]["total_tokens"]
            ot = case["oca"]["total_tokens"]
            case["token_delta_oca_minus_baseline"] = ot - bt
            case["token_ratio_oca_over_baseline"] = (ot / bt) if bt else None
    return {
        "comparison": "same-model paired replay: reproduced baseline vs OCA",
        "cases": by_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RQ4 paired real-world replay with provider-reported token accounting."
    )
    generic_cases = [k for k in CASES if k != "smolagents_1374"]
    parser.add_argument(
        "--case",
        default="all",
        choices=["all", *generic_cases],
        help=(
            "Replay one generic paired case or all generic cases. "
            "smolagents_1374 uses the dedicated same-Claude runner: "
            "python -m realworld_oca.paired_replay_1374_claude"
        ),
    )
    parser.add_argument("--mode", choices=["baseline", "oca", "both"], default="both")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument(
        "--output-dir",
        default="results/realworld_rq4_paired",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all reproduced failure conditions without any LLM call.",
    )
    args = parser.parse_args()

    _load_env()
    checks = validate_reproductions()
    if not all(v.get("reproduced") for v in checks.values()):
        raise SystemExit("Reproduction validation failed:\n" + json.dumps(checks, indent=2))

    if args.dry_run:
        print(json.dumps({"reproductions": checks, "llm_calls": 0}, indent=2))
        return

    case_keys = generic_cases if args.case == "all" else [args.case]
    modes = ["baseline", "oca"] if args.mode == "both" else [args.mode]
    out_dir = Path(args.output_dir)
    rows: List[dict] = []
    for case_key in case_keys:
        for mode in modes:
            print(f"[RQ4] {case_key} | {mode} | {args.model}")
            row = run_one(case_key, mode, args.model, out_dir)
            rows.append(row)
            print(
                f"  {'PASS' if row['evaluation']['passed'] else 'FAIL'} | "
                f"tokens={row['tokens']['total_tokens']} "
                f"(prompt={row['tokens']['prompt_tokens']}, completion={row['tokens']['completion_tokens']}) | "
                f"calls={row['tokens']['calls']} turns={row['turns']}"
            )

    summary = _summary(rows)
    summary["model"] = args.model
    summary["reproduction_checks"] = checks
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "paired_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Human-readable comparison for direct inspection / paper transcription.
    import csv
    csv_path = out_dir / "paired_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "case_key", "case_id",
            "baseline_passed", "baseline_tokens", "baseline_prompt_tokens", "baseline_completion_tokens",
            "oca_passed", "oca_tokens", "oca_prompt_tokens", "oca_completion_tokens",
            "oca_minus_baseline_tokens", "oca_over_baseline_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case_key, case in summary["cases"].items():
            baseline = case.get("baseline", {})
            oca = case.get("oca", {})
            writer.writerow({
                "case_key": case_key,
                "case_id": case.get("case_id"),
                "baseline_passed": baseline.get("passed"),
                "baseline_tokens": baseline.get("total_tokens"),
                "baseline_prompt_tokens": baseline.get("prompt_tokens"),
                "baseline_completion_tokens": baseline.get("completion_tokens"),
                "oca_passed": oca.get("passed"),
                "oca_tokens": oca.get("total_tokens"),
                "oca_prompt_tokens": oca.get("prompt_tokens"),
                "oca_completion_tokens": oca.get("completion_tokens"),
                "oca_minus_baseline_tokens": case.get("token_delta_oca_minus_baseline"),
                "oca_over_baseline_ratio": case.get("token_ratio_oca_over_baseline"),
            })
    print(f"[RQ4] summary: {summary_path}")
    print(f"[RQ4] csv:     {csv_path}")


if __name__ == "__main__":
    main()
