#!/usr/bin/env python3
"""Run SECAT agents on API-Bank Level-1/Level-2 with the official local evaluator."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from importlib import metadata as importlib_metadata
from typing import Iterable, List


_PROTOCOL_VERSION = "secat-apibank-lv1-oca-action-r13-v1"
_LV2_PROTOCOL_VERSION = "secat-apibank-lv2-efficient-r33-v2"
_COMPATIBLE_RESUME_PROTOCOLS = {"secat-apibank-lv1-audit5-v1"}


def _benchmark_level(value: str) -> int:
    name = str(value or "").strip().lower().replace("-", "_")
    return 2 if name in {"apibank_lv2", "api_bank_lv2"} else 1


def summarize_records(records: Iterable[dict], *, expected_total: int | None = None) -> dict:
    rows = list(records)
    total = len(rows)
    expected = total if expected_total is None else int(expected_total)
    correct = sum(1 for row in rows if bool(row.get("correct")))
    agent_errors = sum(1 for row in rows if row.get("status") == "agent_error")
    infrastructure_errors = sum(1 for row in rows if row.get("status") == "infrastructure_error")
    evaluated_total = total - infrastructure_errors
    incorrect = sum(
        1 for row in rows
        if row.get("status") != "infrastructure_error" and not bool(row.get("correct"))
    )
    complete = total == expected and infrastructure_errors == 0
    observed_accuracy = (correct / evaluated_total) if evaluated_total else 0.0
    return {
        "expected_total": expected,
        "total": total,
        "evaluated_total": evaluated_total,
        "correct": correct,
        "incorrect": incorrect,
        "agent_errors": agent_errors,
        "infrastructure_errors": infrastructure_errors,
        # Backwards-readable aggregate; infrastructure errors never make the run reportable.
        "crashes": agent_errors + infrastructure_errors,
        "run_status": "complete" if complete else "incomplete",
        "reportable": bool(complete),
        "observed_accuracy": observed_accuracy,
        "accuracy": observed_accuracy if complete else None,
    }


def _safe_tag(value: str) -> str:
    value = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,239}", value):
        raise SystemExit("result tag must use only letters, digits, '.', '_' or '-'")
    return value


def _select_samples(samples, tasks="5", sample=None, start=0, exact_selection=False):
    if sample is not None:
        text = str(sample).strip()
        for item in samples:
            if item.task_id.lower() == text.lower():
                return [item]
        if text.isdigit() and 1 <= int(text) <= len(samples):
            return [samples[int(text) - 1]]
        raise SystemExit(f"--sample {sample!r} matched no API-Bank task (1..{len(samples)})")

    start = int(start or 0)
    if start < 0:
        raise SystemExit("--start must be >= 0 for API-Bank")
    spec = str(tasks or "all").strip().lower()
    if spec in {"", "all"}:
        chosen = list(samples)
        if not chosen:
            raise SystemExit("API-Bank dataset contains no tasks")
        return chosen
    if spec.isdigit() and not exact_selection:
        count = int(spec)
        if count <= 0:
            raise SystemExit("--tasks count must be > 0 for API-Bank")
        available = len(samples) - start
        if start >= len(samples) or count > available:
            raise SystemExit(
                f"requested {count} API-Bank tasks from --start {start}, but only "
                f"{max(available, 0)} are available in a dataset of {len(samples)}")
        return list(samples)[start: start + count]

    from utils.task_select import parse_task_spec
    nums = parse_task_spec(spec)
    if nums is None:
        chosen = list(samples)
    else:
        invalid = sorted(n for n in nums if n < 1 or n > len(samples))
        if invalid:
            raise SystemExit(
                f"--tasks contains task IDs outside the valid range 1..{len(samples)}: {invalid}")
        chosen = [samples[n - 1] for n in sorted(nums)]
    if not chosen:
        raise SystemExit(f"--tasks {tasks!r} selected no API-Bank tasks")
    return chosen


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temp, path)


def _implementation_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    files = [
        root / "run_apibank.py",
        root / "run_experiment.py",
        root / "benchmarks" / "apibank_runtime.py",
        root / "agents" / "apibank_agents.py",
        root / "bootstrap_apibank.py",
        root / "config.py",
        root / "agents" / "common.py",
        root / "benchmarks" / "__init__.py",
        root / "benchmarks" / "apibank_support.py",
    ]
    files.extend(sorted((root / "utils").rglob("*.py")))
    files.extend(sorted((root / "oca").rglob("*.py")))
    digest = hashlib.sha256()
    for path in files:
        rel = str(path.relative_to(root)).replace(os.sep, "/")
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _lv2_environment() -> dict:
    """Record retrieval-stack versions without importing heavyweight packages."""
    packages = ("sentence-transformers", "transformers", "torch", "huggingface-hub")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": ".".join(str(x) for x in sys.version_info[:3]),
        "packages": versions,
        "hf_hub_offline": str(os.environ.get("HF_HUB_OFFLINE", "")).strip() or None,
        "transformers_offline": str(os.environ.get("TRANSFORMERS_OFFLINE", "")).strip() or None,
    }


def build_run_identity(*, runtime, agent: str, model: str, model_settings: dict,
                       state_mode: str) -> dict:
    from agents.apibank_agents import public_runtime_settings
    level = int(getattr(runtime, "level", 1) or 1)
    payload = {
        "protocol_version": _LV2_PROTOCOL_VERSION if level == 2 else _PROTOCOL_VERSION,
        "benchmark": f"apibank_lv{level}",
        "agent": str(agent),
        "model": str(model),
        "model_settings": dict(model_settings or {}),
        "runtime_settings": public_runtime_settings(agent, model),
        "state_mode": str(state_mode),
        "benchmark_sha256": runtime.benchmark_fingerprint(),
        "implementation_sha256": _implementation_fingerprint(),
    }
    if level == 2:
        payload["environment"] = _lv2_environment()
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    payload["identity_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def ensure_manifest(path: Path, identity: dict, *, fresh: bool = False) -> dict:
    path = Path(path)
    if path.exists() and not fresh:
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SystemExit(f"Could not read API-Bank run manifest {path}: {exc}") from exc
        if previous.get("identity_sha256") != identity.get("identity_sha256"):
            raise SystemExit(
                "API-Bank result identity mismatch; refusing to mix/resume runs. "
                "Use --fresh or a different --result-tag.")
        if list(previous.get("selected_task_ids") or []) != list(identity.get("selected_task_ids") or []):
            raise SystemExit(
                "API-Bank selected-task identity mismatch; refusing to resume a different task set. "
                "Use --fresh or a different --result-tag.")
        return previous
    _write_json(path, identity)
    return identity


def _read_existing_record(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _merge_token_usage(previous: dict | None, current: dict | None) -> dict:
    """Add token-meter snapshots, including per-stage counters."""
    previous = dict(previous or {})
    current = dict(current or {})
    result = {}
    for key in set(previous) | set(current):
        if key == "by_stage":
            stages = {}
            for source in (previous.get("by_stage") or {}, current.get("by_stage") or {}):
                for stage, values in source.items():
                    bucket = stages.setdefault(str(stage), {})
                    for name, value in dict(values or {}).items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            bucket[name] = bucket.get(name, 0) + value
            result[key] = stages
        else:
            left, right = previous.get(key, 0), current.get(key, 0)
            if (isinstance(left, (int, float)) and not isinstance(left, bool) and
                    isinstance(right, (int, float)) and not isinstance(right, bool)):
                result[key] = left + right
            else:
                result[key] = right if key in current else left
    return result


def _attempt_snapshot(record: dict) -> dict:
    return {
        "status": record.get("status"),
        "error_type": record.get("error_type"),
        "error": record.get("error"),
        "tokens": dict(record.get("attempt_tokens") or record.get("tokens") or {}),
        "prediction_saved": isinstance(record.get("prediction"), dict),
    }


def _prediction_from_saved(data: dict, prediction_cls):
    if not isinstance(data, dict):
        return None
    return prediction_cls(
        prediction_text=str(data.get("prediction_text") or ""),
        strategy=str(data.get("strategy") or ""),
        trace=list(data.get("trace") or []),
        executed_call=data.get("executed_call"),
        action=data.get("action"),
    )


def _evaluate_prediction(runtime, sample_obj, prediction):
    if prediction.executed_call is not None:
        executed = prediction.executed_call
        return runtime.evaluate_executed(
            sample_obj,
            api_name=str(executed.get("api_name")),
            params=dict(executed.get("params") or {}),
            result=executed.get("result"),
            replayed_calls=int(executed.get("replayed_calls") or 0),
        )
    if prediction.action is not None:
        return runtime.evaluate_action(
            sample_obj, api_name=prediction.action["api_name"],
            params=prediction.action["params"])
    return runtime.evaluate(sample_obj, prediction.prediction_text)


def _resume_compatibility_fields(identity: dict) -> dict:
    keys = (
        "benchmark", "agent", "model", "model_settings", "runtime_settings",
        "state_mode", "benchmark_sha256", "selected_task_ids",
    )
    return {key: identity.get(key) for key in keys}


def import_resume_source(source_dir: Path, target_runs_dir: Path, identity: dict) -> dict:
    """Import an audit5-v1 incomplete run into v2 without reusing its identity.

    Only the resume/persistence implementation changed between these protocols.
    Benchmark/model/runtime/selection identity must otherwise match exactly.
    """
    source_dir = Path(source_dir).expanduser().resolve()
    target_runs_dir = Path(target_runs_dir).resolve()
    target_dir = target_runs_dir.parent
    if source_dir == target_dir:
        raise SystemExit("--resume-from must point to a different API-Bank result directory")
    manifest_path = source_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"--resume-from has no run_manifest.json: {source_dir}")
    try:
        source_identity = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Could not read --resume-from manifest: {exc}") from exc
    if source_identity.get("protocol_version") not in _COMPATIBLE_RESUME_PROTOCOLS:
        raise SystemExit(
            "--resume-from is supported only for the immediately preceding audit5-v1 protocol")
    if _resume_compatibility_fields(source_identity) != _resume_compatibility_fields(identity):
        raise SystemExit(
            "--resume-from benchmark/model/runtime/task selection does not match this run")
    source_hash = source_identity.get("identity_sha256")
    imported = 0
    for task_id in identity.get("selected_task_ids") or []:
        source_record_path = source_dir / "runs" / f"{task_id}.json"
        if not source_record_path.is_file():
            continue
        source_record = _read_existing_record(source_record_path)
        if source_record is None:
            raise SystemExit(f"Could not read resume record: {source_record_path}")
        if (source_record.get("task_id") != task_id or
                source_record.get("run_identity_sha256") != source_hash):
            raise SystemExit(f"Unsafe/mismatched resume record: {source_record_path}")
        target_path = target_runs_dir / f"{task_id}.json"
        if target_path.exists():
            continue
        migrated = dict(source_record)
        migrated["source_run_identity_sha256"] = source_hash
        migrated["run_identity_sha256"] = identity["identity_sha256"]
        migrated["resume_migration"] = {
            "from_protocol": source_identity.get("protocol_version"),
            "from_result_dir": str(source_dir),
        }
        _write_json(target_path, migrated)
        imported += 1
    return {"source_dir": str(source_dir), "source_identity_sha256": source_hash,
            "imported_records": imported}


def _llm_failure_is_agent_generation_failure(exc: Exception) -> bool:
    """True only for bounded model-generation failures, not provider outages."""
    text = str(exc).lower()
    markers = (
        "thinking exhausted the api-bank oca completion budget",
        "json mode returned empty content on both bounded attempts",
        "oca api-bank task deadline exceeded",
    )
    return any(marker in text for marker in markers)


def run(args) -> dict:
    from benchmarks.apibank_runtime import APIBankRuntime
    from agents import apibank_agents
    from agents.apibank_agents import AGENT_RUNNERS, LLMRunError, AgentBehaviorError
    import config
    from utils.model_provider import (
        model_run_slug, public_model_settings, require_provider_key,
    )
    from utils import token_meter

    agent_name = str(getattr(args, "agent", "codeact")).strip().lower()
    if agent_name not in AGENT_RUNNERS:
        raise SystemExit("API-Bank supports --agent oca, codeact, or toolcoder")

    state_mode = str(getattr(args, "state_mode", "published") or "published").strip().lower()
    if state_mode not in {"published", "replay"}:
        raise SystemExit("--state-mode must be published or replay")

    vendor_root = getattr(args, "apibank_root", None) or os.getenv("APIBANK_ROOT") or None
    bootstrap = bool(getattr(args, "bootstrap", False))
    if vendor_root is None:
        default_root = Path(__file__).resolve().parent / "benchmarks" / "api_bank_vendor"
        if default_root.exists():
            vendor_root = default_root
        elif bootstrap:
            from bootstrap_apibank import install_apibank
            vendor_root = install_apibank(destination=default_root)

    explicit_level = getattr(args, "level", None)
    level = int(explicit_level) if explicit_level is not None else _benchmark_level(
        getattr(args, "benchmark", "apibank_lv1"))
    if level not in {1, 2}:
        raise SystemExit("API-Bank level must be 1 or 2")
    benchmark_name = f"apibank_lv{level}"
    try:
        runtime = APIBankRuntime(vendor_root=vendor_root, state_mode=state_mode, level=level)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\nRun `python bootstrap_apibank.py` once, or pass "
            "`--apibank-root /path/to/api-bank` (or set APIBANK_ROOT).") from exc

    all_samples = runtime.load_samples()
    exact = bool(getattr(args, "expect_task_ids", None))
    selected = _select_samples(
        all_samples, getattr(args, "tasks", "5"), getattr(args, "sample", None),
        getattr(args, "start", 0), exact_selection=exact)

    expected_spec = getattr(args, "expect_task_ids", None)
    if expected_spec:
        from utils.task_select import parse_task_spec
        expected = parse_task_spec(str(expected_spec).strip().lower())
        actual = {int(s.task_id.rsplit("_", 1)[-1]) for s in selected}
        if expected is not None and set(expected) != actual:
            raise SystemExit(
                "TASK SELECTION SAFETY CHECK FAILED: "
                f"expected={sorted(expected)}, actual={sorted(actual)}. No LLM/tool call was made.")
        print(f"[SAFETY] Exact API-Bank selection confirmed: {sorted(actual)}")

    model = str(getattr(args, "model", config.DEFAULT_MODEL))
    # Critical ordering: provider configuration/key is validated before any result
    # directory or task artifact is created.
    settings = public_model_settings(model)
    if not bool(getattr(args, "preflight_only", False)):
        require_provider_key(model)
    # Adapter/vendor compatibility is a setup concern, not a benchmark failure.
    # Validate it before creating result artifacts or making any model call.
    apibank_agents.validate_agent_environment(agent_name, runtime)
    preflight = runtime.preflight(selected)
    if bool(getattr(args, "preflight_only", False)):
        summary = {"preflight": "passed", "reportable": True, "selected_tasks": len(selected),
                   "dataset_tasks": len(all_samples), "runtime": preflight}
        if int(getattr(runtime, "level", 1) or 1) == 2:
            summary["environment"] = _lv2_environment()
        print("[PREFLIGHT] " + json.dumps(summary, sort_keys=True))
        return summary

    identity = build_run_identity(
        runtime=runtime, agent=agent_name, model=model,
        model_settings=settings, state_mode=state_mode)
    identity["selected_task_ids"] = [s.task_id for s in selected]

    base = Path(config.RESULT_DIR) / benchmark_name
    explicit_tag = str(getattr(args, "result_tag", "") or "").strip()
    oca_tag = str(getattr(args, "oca_result_tag", "") or "").strip() if agent_name == "oca" else ""
    if oca_tag and explicit_tag and oca_tag != explicit_tag:
        raise SystemExit("API-Bank --result-tag and --oca-result-tag disagree; use one tag")
    explicit_tag = explicit_tag or oca_tag
    if explicit_tag:
        out_dir = base / _safe_tag(explicit_tag)
    else:
        out_dir = base / agent_name / model_run_slug(model) / state_mode
    if bool(getattr(args, "fresh", False)) and out_dir.exists():
        shutil.rmtree(out_dir)

    ensure_manifest(out_dir / "run_manifest.json", identity, fresh=False)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    resume_from = str(getattr(args, "resume_from", "") or "").strip()
    if resume_from and level == 2:
        raise SystemExit("--resume-from migration is not supported across API-Bank LV2 runs; resume the same result-tag directly")
    if resume_from:
        migration = import_resume_source(Path(resume_from), runs_dir, identity)
        manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
        manifest["resume_migration"] = migration
        _write_json(out_dir / "run_manifest.json", manifest)
        print(f"[RESUME MIGRATION] imported={migration['imported_records']} from {migration['source_dir']}")

    runner = AGENT_RUNNERS[agent_name]
    records: List[dict] = []
    print(f"[INFO] API-Bank LV{level}: agent={agent_name} model={model} state={state_mode} tasks={len(selected)}")
    print(f"[INFO] API-Bank root: {runtime.root}")
    print(f"[INFO] Results: {out_dir}")

    infrastructure_failure = None
    for index, sample_obj in enumerate(selected, 1):
        run_path = runs_dir / f"{sample_obj.task_id}.json"
        previous = _read_existing_record(run_path)
        retry_prediction = None
        prior_tokens = {}
        prior_attempts = []
        if previous is not None:
            if (previous.get("run_identity_sha256") != identity["identity_sha256"] or
                    previous.get("task_id") != sample_obj.task_id):
                raise SystemExit(
                    f"API-Bank record identity mismatch for {run_path}; refusing unsafe resume. "
                    "Use --fresh or a different --result-tag.")
            if previous.get("status") != "infrastructure_error":
                records.append(previous)
                print(f"[{index}/{len(selected)}] {sample_obj.task_id} [RESUME]")
                continue
            prior_tokens = dict(previous.get("tokens") or {})
            prior_attempts = list(previous.get("attempts") or [])
            prior_attempts.append(_attempt_snapshot(previous))
            retry_prediction = _prediction_from_saved(
                previous.get("prediction"), apibank_agents.AgentPrediction)

        print(f"[{index}/{len(selected)}] {sample_obj.task_id}")
        token_meter.reset()
        record = {
            "task_id": sample_obj.task_id,
            "source_file": sample_obj.source_file,
            "source_api_index": sample_obj.source_api_index,
            "agent": agent_name,
            "model": model,
            "model_settings": settings,
            "state_mode": state_mode,
            "run_identity_sha256": identity["identity_sha256"],
            "status": "ok",
            "correct": False,
        }
        if previous is not None:
            for provenance_key in ("source_run_identity_sha256", "resume_migration"):
                if provenance_key in previous:
                    record[provenance_key] = previous[provenance_key]
        if prior_attempts:
            record["attempts"] = prior_attempts
        prediction = retry_prediction
        try:
            if prediction is None:
                view = runtime.agent_view(sample_obj)
                safe_sample = runtime.redacted_sample(sample_obj)
                prediction = runner(
                    view=view, sample=safe_sample, runtime=runtime, model=model)
            else:
                print("  [RESUME] reusing saved model prediction; retrying evaluator/tool only")
            # Persist the completed model decision even if subsequent tool/checker
            # infrastructure fails. This makes service retries deterministic.
            record["prediction"] = prediction.as_dict()
            evaluation = _evaluate_prediction(runtime, sample_obj, prediction)
            record.update({
                "evaluation": evaluation.as_dict(),
                "correct": bool(evaluation.correct),
                # Saved only after the agent boundary for auditability.
                "expected_api": sample_obj.ground_truth.get("api_name"),
            })
            if evaluation.error_type in {"checker_error", "replay_error", "tool_timeout", "tool_service_error", "tool_dependency_error"}:
                record.update({
                    "status": "infrastructure_error",
                    "correct": False,
                    "error_type": evaluation.error_type,
                    "error": evaluation.error or "unknown evaluator error",
                })
                infrastructure_failure = RuntimeError(record["error"])
                print(f"  [INFRASTRUCTURE] {evaluation.error_type}: {record['error']}")
        except AgentBehaviorError as exc:
            record.update({
                "status": "agent_error",
                "correct": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            agent_trace = getattr(exc, "agent_trace", None)
            if agent_trace:
                record["agent_trace"] = agent_trace
            print(f"  [AGENT ERROR] {type(exc).__name__}: {exc}")
        except LLMRunError as exc:
            if _llm_failure_is_agent_generation_failure(exc):
                record.update({
                    "status": "agent_error",
                    "correct": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                print(f"  [AGENT ERROR] bounded model generation failed: {exc}")
            else:
                record.update({
                    "status": "infrastructure_error",
                    "correct": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                infrastructure_failure = exc
                print(f"  [INFRASTRUCTURE] {type(exc).__name__}: {exc}")
        except Exception as exc:
            # Until an exception is explicitly classified as model/agent behavior,
            # fail closed: SECAT/framework defects must never become benchmark 0s.
            record.update({
                "status": "infrastructure_error",
                "correct": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            infrastructure_failure = exc
            print(f"  [INFRASTRUCTURE] {type(exc).__name__}: {exc}")
        attempt_tokens = token_meter.read()
        record["attempt_tokens"] = attempt_tokens
        record["tokens"] = _merge_token_usage(prior_tokens, attempt_tokens)
        records.append(record)
        _write_json(run_path, record)
        if record["status"] == "infrastructure_error":
            if bool(getattr(args, "continue_infrastructure", False)):
                print("  FAIL [INFRASTRUCTURE; continuing by request]")
                continue
            break
        if record["status"] == "agent_error" and bool(getattr(args, "fail_fast_crash", False)):
            print("  FAIL [FAIL-FAST]")
            break
        print("  " + ("PASS" if record["correct"] else "FAIL"))

    summary = summarize_records(records, expected_total=len(selected))
    token_totals = {
        "prompt_tokens": sum(int((r.get("tokens") or {}).get("prompt_tokens", 0)) for r in records),
        "completion_tokens": sum(int((r.get("tokens") or {}).get("completion_tokens", 0)) for r in records),
        "total_tokens": sum(int((r.get("tokens") or {}).get("total_tokens", 0)) for r in records),
        "calls": sum(int((r.get("tokens") or {}).get("calls", 0)) for r in records),
    }
    summary.update({
        "benchmark": benchmark_name,
        "agent": agent_name,
        "model": model,
        "model_settings": settings,
        "state_mode": state_mode,
        "run_identity_sha256": identity["identity_sha256"],
        "selected_task_ids": [s.task_id for s in selected],
        "completed_task_ids": [
            r.get("task_id") for r in records if r.get("status") != "infrastructure_error"
        ],
        "infrastructure_task_ids": [
            r.get("task_id") for r in records if r.get("status") == "infrastructure_error"
        ],
        "tokens": token_totals,
        "result_dir": str(out_dir),
    })
    _write_json(out_dir / "summary.json", summary)
    print("[SUMMARY] " + json.dumps(summary, sort_keys=True))
    return summary


def run_from_namespace(namespace) -> dict:
    """Entry point used by the top-level SECAT runner."""
    summary = run(namespace)
    if not summary.get("reportable"):
        raise SystemExit(2)
    return summary


def parse_args(argv=None):
    import config
    parser = argparse.ArgumentParser(description="SECAT API-Bank Level-1/Level-2 runner")
    parser.add_argument("--agent", choices=["oca", "codeact", "toolcoder"], default="codeact")
    parser.add_argument("--level", type=int, choices=[1, 2], default=1,
                        help="API-Bank level when invoking run_apibank.py directly")
    parser.add_argument("--model", default=config.DEFAULT_MODEL)
    parser.add_argument("--tasks", default="5",
                        help="Count (e.g. 5), selection (1,3,5 or 20-25), or all")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--sample", default=None,
                        help="Exactly one API-call sample by ID (apibank_lv1_001/apibank_lv2_001) or 1-based number")
    parser.add_argument("--expect-task-ids", default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--fail-fast-crash", action="store_true")
    parser.add_argument("--continue-infrastructure", action="store_true",
                        help="Continue later tasks after infrastructure errors; run remains non-reportable until retry")
    parser.add_argument("--result-tag", default=None)
    parser.add_argument("--resume-from", default=None,
                        help="Import a compatible incomplete audit5-v1 result directory into this corrected run")
    parser.add_argument("--state-mode", choices=["published", "replay"], default="published",
                        help="published=fresh ToolManager per target (paper-compatible default); replay=explicit stateful sensitivity mode")
    parser.add_argument("--apibank-root", default=None,
                        help="Path to official API-Bank root; overrides APIBANK_ROOT")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Download/install API-Bank if the default vendor tree is missing")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Validate selected real tools/resources without provider calls or result files")
    return parser.parse_args(argv)


def main(argv=None):
    summary = run(parse_args(argv))
    if not summary.get("reportable"):
        raise SystemExit(2)
    return summary


if __name__ == "__main__":
    main()
