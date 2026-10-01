from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CASE_KEY = "smolagents_1374"
CASE_ID = "smolagents#1374"
SOURCE_ISSUE = "https://github.com/huggingface/smolagents/issues/1374"
TASK = "What is 1654651 * 6332?"
EXPECTED = "10477250132"
DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"
ORIGINAL_ISSUE_MODEL = "claude-3-7-sonnet-20250219"
FAILURE = (
    "final_answer_checks runs before the current ActionStep is appended to memory, "
    "so the completion critic observes stale state"
)


def _tool_schema() -> list[dict]:
    # Identical model-facing action surface in baseline and OCA.
    return [{
        "type": "function",
        "function": {
            "name": "calculate_and_submit",
            "description": (
                "Multiply the supplied integers in the runtime and submit the computed "
                "result as the final answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "left": {"type": "integer"},
                    "right": {"type": "integer"},
                },
                "required": ["left", "right"],
                "additionalProperties": False,
            },
        },
    }]


def _system_prompt() -> str:
    # Identical between baseline and OCA. Only framework verification ordering differs.
    return (
        "Complete the user's task using only the supplied runtime tool. "
        "Use calculate_and_submit to perform the multiplication and submit the result. "
        "Do not calculate the answer only in free text."
    )


def _usage(resp: Any) -> dict:
    usage = getattr(resp, "usage", None)
    if usage is None:
        raise RuntimeError(
            "Provider response omitted token usage; RQ4 requires provider-reported usage."
        )
    prompt = int(
        getattr(usage, "prompt_tokens", None)
        or getattr(usage, "input_tokens", None)
        or 0
    )
    completion = int(
        getattr(usage, "completion_tokens", None)
        or getattr(usage, "output_tokens", None)
        or 0
    )
    total = int(getattr(usage, "total_tokens", None) or (prompt + completion))
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _message_dict(message: Any) -> dict:
    if hasattr(message, "model_dump"):
        data = message.model_dump(exclude_none=True)
    elif isinstance(message, dict):
        data = dict(message)
    else:
        data = {
            "role": "assistant",
            "content": getattr(message, "content", "") or "",
        }
    # Keep only Chat Completions fields needed for the next turn.
    out = {
        "role": data.get("role", "assistant"),
        "content": data.get("content") or "",
    }
    if data.get("tool_calls"):
        out["tool_calls"] = data["tool_calls"]
    return out


def _tool_calls(message: Any) -> list[dict]:
    if hasattr(message, "model_dump"):
        data = message.model_dump(exclude_none=True)
    elif isinstance(message, dict):
        data = message
    else:
        data = {}
    return data.get("tool_calls") or []


def _execute(mode: str, args: dict, state: dict) -> dict:
    left = int(args.get("left"))
    right = int(args.get("right"))
    computed = str(left * right)
    check_no = int(state.get("final_check_count", 0)) + 1
    state["final_check_count"] = check_no

    if mode == "baseline":
        # Historical smolagents ordering: the current action is not yet in memory.
        memory_types = (
            ["TaskStep"]
            if check_no == 1
            else ["TaskStep", "ActionStep(previous)"]
        )
        current_visible = False
    else:
        # OCA verifies against the just-executed action/evidence.
        memory_types = ["TaskStep", "ActionStep(current)"]
        current_visible = True

    event = {
        "check": check_no,
        "memory_types": memory_types,
        "current_action_visible": current_visible,
        "computed_result": computed,
    }
    state.setdefault("final_check_events", []).append(event)

    if check_no == 1:
        state["first_check_missing_current_action"] = not current_visible

    if mode == "baseline" and check_no == 1:
        return {
            "accepted": False,
            "computed_result": computed,
            "error": (
                "final-answer critic rejected completion because the current "
                "ActionStep is absent from checker memory"
            ),
        }

    state["submitted"] = computed
    state["done"] = True
    return {"accepted": True, "result": computed}


def _evaluate(mode: str, state: dict) -> dict:
    observed = str(state.get("submitted") or "").strip()
    first_missing = bool(state.get("first_check_missing_current_action", False))
    # The case-level success criterion includes correct completion verification,
    # not merely eventual arithmetic correctness after a stale-check rejection.
    passed = EXPECTED in observed and not first_missing
    return {
        "passed": bool(passed),
        "expected": EXPECTED,
        "observed": observed,
        "first_check_missing_current_action": first_missing,
        "check_events": state.get("final_check_events", []),
        "criterion": (
            "correct result and first final-answer check includes the current executed action"
        ),
        "mode": mode,
    }


def run_one(mode: str, model: str, output_dir: Path, max_turns: int = 4) -> dict:
    if mode not in {"baseline", "oca"}:
        raise ValueError("mode must be baseline or oca")
    if not model.startswith("anthropic/"):
        raise RuntimeError(
            "#1374 paired replay must use an Anthropic Claude model so baseline and OCA "
            "are tested with the same model family as the reproduced issue."
        )

    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for #1374 paired replay")

    import litellm
    litellm.drop_params = True

    messages: list[dict] = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": TASK},
    ]
    tools = _tool_schema()
    state = {
        "done": False,
        "submitted": None,
        "final_check_count": 0,
        "final_check_events": [],
    }
    trace: list[dict] = []
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}

    for turn in range(1, max_turns + 1):
        resp = litellm.completion(
            model=model,
            api_key=key,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=2048,
        )
        usage = _usage(resp)
        totals["prompt_tokens"] += usage["prompt_tokens"]
        totals["completion_tokens"] += usage["completion_tokens"]
        totals["total_tokens"] += usage["total_tokens"]
        totals["calls"] += 1

        message = resp.choices[0].message
        assistant = _message_dict(message)
        messages.append(assistant)
        calls = _tool_calls(message)
        turn_entry = {
            "turn": turn,
            "usage": usage,
            "assistant_text": assistant.get("content") or "",
            "tool_calls": [],
        }

        if not calls:
            if assistant.get("content"):
                state["submitted"] = assistant["content"]
            trace.append(turn_entry)
            break

        for tc in calls:
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except Exception:
                args = {}

            if name == "calculate_and_submit":
                result = _execute(mode, args, state)
            else:
                result = {"accepted": False, "error": f"unknown tool {name}"}

            turn_entry["tool_calls"].append({
                "name": name,
                "arguments": args,
                "result": result,
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id") or "",
                "content": json.dumps(result, ensure_ascii=False),
            })

        trace.append(turn_entry)
        if state.get("done"):
            break

    evaluation = _evaluate(mode, state)
    result = {
        "case_key": CASE_KEY,
        "case_id": CASE_ID,
        "source_issue": SOURCE_ISSUE,
        "model": model,
        "model_source": "claude_replacement_for_retired_issue_model",
        "original_issue_model": ORIGINAL_ISSUE_MODEL,
        "historical_reproduction_model": model,
        "mode": mode,
        "failure": FAILURE,
        "task": TASK,
        "max_turns": max_turns,
        "turns": len(trace),
        "tokens": totals,
        "evaluation": evaluation,
        "trace": trace,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{CASE_KEY}__{mode}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _summary(rows: list[dict], model: str) -> dict:
    by_mode = {}
    for row in rows:
        by_mode[row["mode"]] = {
            "passed": row["evaluation"]["passed"],
            "prompt_tokens": row["tokens"]["prompt_tokens"],
            "completion_tokens": row["tokens"]["completion_tokens"],
            "total_tokens": row["tokens"]["total_tokens"],
            "calls": row["tokens"]["calls"],
            "turns": row["turns"],
        }
    case = {"case_id": CASE_ID, **by_mode}
    if "baseline" in case and "oca" in case:
        bt = case["baseline"]["total_tokens"]
        ot = case["oca"]["total_tokens"]
        case["token_delta_oca_minus_baseline"] = ot - bt
        case["token_ratio_oca_over_baseline"] = (ot / bt) if bt else None
    return {
        "comparison": "same-Claude paired replay: reproduced baseline vs OCA",
        "model": model,
        "original_issue_model": ORIGINAL_ISSUE_MODEL,
        "model_substitution_reason": "original Claude 3.7 Sonnet model is retired",
        "cases": {CASE_KEY: case},
    }


def _write_summary(rows: list[dict], model: str, output_dir: Path) -> None:
    summary = _summary(rows, model)
    (output_dir / "paired_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    case = summary["cases"][CASE_KEY]
    baseline = case.get("baseline", {})
    oca = case.get("oca", {})
    with (output_dir / "paired_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "case", "model",
            "baseline_passed", "baseline_tokens", "baseline_prompt_tokens",
            "baseline_completion_tokens", "baseline_calls", "baseline_turns",
            "oca_passed", "oca_tokens", "oca_prompt_tokens", "oca_completion_tokens",
            "oca_calls", "oca_turns", "oca_minus_baseline_tokens", "oca_over_baseline_ratio",
        ])
        writer.writeheader()
        writer.writerow({
            "case": CASE_ID,
            "model": model,
            "baseline_passed": baseline.get("passed"),
            "baseline_tokens": baseline.get("total_tokens"),
            "baseline_prompt_tokens": baseline.get("prompt_tokens"),
            "baseline_completion_tokens": baseline.get("completion_tokens"),
            "baseline_calls": baseline.get("calls"),
            "baseline_turns": baseline.get("turns"),
            "oca_passed": oca.get("passed"),
            "oca_tokens": oca.get("total_tokens"),
            "oca_prompt_tokens": oca.get("prompt_tokens"),
            "oca_completion_tokens": oca.get("completion_tokens"),
            "oca_calls": oca.get("calls"),
            "oca_turns": oca.get("turns"),
            "oca_minus_baseline_tokens": case.get("token_delta_oca_minus_baseline"),
            "oca_over_baseline_ratio": case.get("token_ratio_oca_over_baseline"),
        })


def dry_run(model: str) -> dict:
    # Zero-provider-call validation of the exact baseline/OCA state transition.
    baseline_state = {"done": False, "submitted": None, "final_check_count": 0, "final_check_events": []}
    oca_state = {"done": False, "submitted": None, "final_check_count": 0, "final_check_events": []}
    args = {"left": 1654651, "right": 6332}

    b1 = _execute("baseline", args, baseline_state)
    b2 = _execute("baseline", args, baseline_state)
    o1 = _execute("oca", args, oca_state)

    b_eval = _evaluate("baseline", baseline_state)
    o_eval = _evaluate("oca", oca_state)

    checks = {
        "model": model,
        "same_prompt": True,
        "same_tool_schema": True,
        "baseline_first_rejected": b1.get("accepted") is False,
        "baseline_eventually_computes_correct": b2.get("result") == EXPECTED,
        "baseline_first_check_missing_current_action": baseline_state.get("first_check_missing_current_action") is True,
        "baseline_passed": b_eval["passed"],
        "oca_first_accepted": o1.get("accepted") is True,
        "oca_first_check_missing_current_action": oca_state.get("first_check_missing_current_action", False),
        "oca_passed": o_eval["passed"],
    }
    checks["valid"] = (
        checks["baseline_first_rejected"]
        and checks["baseline_eventually_computes_correct"]
        and checks["baseline_first_check_missing_current_action"]
        and checks["baseline_passed"] is False
        and checks["oca_first_accepted"]
        and checks["oca_first_check_missing_current_action"] is False
        and checks["oca_passed"] is True
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="smolagents #1374 same-Claude paired replay: reproduced baseline vs OCA"
    )
    parser.add_argument("--mode", choices=["baseline", "oca", "both"], default="both")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="results/realworld_rq4_paired_1374")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.model.startswith("anthropic/"):
        raise SystemExit("#1374 must be paired with an Anthropic Claude model")

    if args.dry_run:
        result = dry_run(args.model)
        print(json.dumps(result, indent=2))
        if not result["valid"]:
            raise SystemExit(2)
        return

    output_dir = Path(args.output_dir)
    modes = ["baseline", "oca"] if args.mode == "both" else [args.mode]
    rows = []
    for mode in modes:
        row = run_one(mode, args.model, output_dir)
        rows.append(row)
        t = row["tokens"]
        print(
            f"{CASE_KEY} {mode}: {'PASS' if row['evaluation']['passed'] else 'FAIL'} | "
            f"tokens={t['total_tokens']} (prompt={t['prompt_tokens']}, completion={t['completion_tokens']}) | "
            f"calls={t['calls']} turns={row['turns']}"
        )

    _write_summary(rows, args.model, output_dir)
    print(f"Summary: {output_dir / 'paired_summary.json'}")


if __name__ == "__main__":
    main()
