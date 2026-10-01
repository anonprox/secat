"""
logger.py
Saves EVERYTHING to files — full trajectory, errors only, summary.
One JSON file per task run. Never lose data.
"""
import json, os
from datetime import datetime


def _write_json_durable(path, payload):
    """Atomically persist JSON and verify the final artifact exists.

    Result artifacts are experiment evidence.  Write them beneath the configured
    absolute result root, fsync the temporary file, atomically replace the target,
    then verify the target is present and non-empty before reporting success.
    """
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise OSError(f"result artifact was not persisted: {path}")
    return path

def _ensure_dirs():
    """Create log directories (honors config.set_output_base)."""
    import config
    os.makedirs(config.RUN_DIR,     exist_ok=True)
    os.makedirs(config.ERROR_DIR,   exist_ok=True)
    os.makedirs(config.SUMMARY_DIR, exist_ok=True)


class TaskLogger:
    """
    Logger for a single task run.
    Call log_turn() after each agent turn.
    Call save() at the end.
    """

    def __init__(self, agent_name: str, task: dict, model: str):
        _ensure_dirs()
        self.agent_name = agent_name
        self.model      = model
        self.task_id    = task.get("id", "unknown")
        self.task       = task
        self.timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log = {
            "meta": {
                "agent":     agent_name,
                "model":     model,
                "task_id":   self.task_id,
                "timestamp": self.timestamp,
                "benchmark": task.get("benchmark"),
                "instruction": task.get("instruction", ""),
                "api_list":    task.get("api_list", []),
                "ground_truth": task.get("ground_truth", None),
            },
            "turns": [],
            "summary": {
                "total_turns":           0,
                "num_errors":            0,
                "error_types":           [],
                "scopes_hit":            [],
                "first_error_turn":      None,
                "turns_after_first_error": 0,
                "final_answer":          None,
                "success":               None,
                "silent_failure":        False,
                "log_anomaly_types":     [],   # from log_analyzer
            }
        }

    def log_turn(self,
                 turn_num:    int,
                 llm_input:   list,   # full messages list sent to LLM
                 llm_output:  str,    # raw LLM response
                 code:        str,    # extracted original code
                 exec_result: dict,   # from executor.execute()
                 error_info:  dict,   # from error_classifier.classify_error()
                 log_findings:dict = None,  # from log_analyzer.analyze_logs()
                 instrumented_code: str = "",  # code after inject_logs()
                 agent_output:      str = "",  # clean output the agent sees
                 logged_output:     str = "",  # [Sx] lines only (display/analysis)
                 ):
        turn = {
            "turn":        turn_num,
            "llm_input":   llm_input,
            "llm_output":  llm_output,
            "code":              code,               # 1. original python code
            "instrumented_code": instrumented_code,  # 2. code with injected logs
            "execution": {
                "stdout":    exec_result.get("stdout", ""),
                "stderr":    exec_result.get("stderr", ""),
                "combined":  exec_result.get("combined", ""),
                "exit_code": exec_result.get("exit_code"),
                "timed_out": exec_result.get("timed_out", False),
                "success":   exec_result.get("success", False),
            },
            "agent_output":  agent_output,   # 3. output fed back to agent (no logs)
            "logged_output": logged_output,  # 4. injected [Sx] log lines only
            "var_trace":     exec_result.get("var_trace", []),  # 5. runtime variable state (wrapper)
            "error": {
                "is_error":   error_info.get("is_error", False),
                "is_silent":  error_info.get("is_silent", False),
                "error_type": error_info.get("error_type", "none"),
                "scope":      error_info.get("scope", "none"),
                "raw_error":  error_info.get("raw_error", None),
            },
            "log_analysis": log_findings or {},
        }
        self.log["turns"].append(turn)

        # Update summary
        if error_info.get("is_error"):
            self.log["summary"]["num_errors"] += 1
            self.log["summary"]["error_types"].append(error_info["error_type"])
            self.log["summary"]["scopes_hit"].append(error_info["scope"])

            if self.log["summary"]["first_error_turn"] is None:
                self.log["summary"]["first_error_turn"] = turn_num

        # Collect anomaly types from log_analyzer findings
        if log_findings:
            for anomaly in log_findings.get("anomalies", []):
                self.log["summary"]["log_anomaly_types"].append(
                    anomaly.get("type", "unknown")
                )

    def finalize(self, final_answer: str, success: bool = None,
                 silent_failure: bool = False):
        """Call at end of task with the agent's final answer."""
        total = len(self.log["turns"])
        first_err = self.log["summary"]["first_error_turn"]

        self.log["summary"]["total_turns"]      = total
        self.log["summary"]["final_answer"]     = final_answer
        self.log["summary"]["success"]          = success
        self.log["summary"]["silent_failure"]   = silent_failure
        self.log["summary"]["turns_after_first_error"] = (
            max(0, total - first_err) if first_err is not None else 0
        )

    def save(self):
        """Save full trajectory JSON and append to errors file."""
        # Full run log
        fname = f"{self.agent_name}_{self.task_id}_{self.timestamp}.json"
        import config; run_path = os.path.join(config.RUN_DIR, fname)
        run_path = _write_json_durable(run_path, self.log)

        # Append errors to shared errors file
        if self.log["summary"]["num_errors"] > 0 or \
           self.log["summary"]["silent_failure"]:
            err_path = os.path.join(__import__("config").ERROR_DIR,
                                    f"{self.agent_name}_errors.jsonl")
            with open(err_path, "a", encoding="utf-8") as f:
                error_entry = {
                    "task_id":     self.task_id,
                    "instruction": self.log["meta"]["instruction"],
                    "timestamp":   self.timestamp,
                    "summary":     self.log["summary"],
                }
                f.write(json.dumps(error_entry, ensure_ascii=False) + "\n")

        print(f"  [LOG] Saved → {run_path}")
        return run_path


def save_experiment_summary(agent_name: str, all_results: list):
    """
    Save aggregated summary across all tasks.
    Call after all tasks are done.
    """
    _ensure_dirs()
    from collections import Counter

    total   = len(all_results)
    success = sum(1 for r in all_results if r.get("success") is True)
    failed  = sum(1 for r in all_results if r.get("success") is False)
    unverified = sum(1 for r in all_results if r.get("success") is None)
    silent  = sum(1 for r in all_results if r.get("silent_failure"))
    no_tool = sum(1 for r in all_results if r.get("answered_without_tool_use"))

    all_errors   = [e for r in all_results for e in r.get("error_types", [])]
    all_scopes   = [s for r in all_results for s in r.get("scopes_hit",  [])]
    all_anomalies= [a for r in all_results for a in r.get("log_anomaly_types", [])]

    turns_wasted = [r.get("turns_after_first_error", 0) for r in all_results
                    if r.get("first_error_turn") is not None]

    # token totals across tasks (per-task usage recorded by the token meter)
    tok_total = sum((r.get("tokens") or {}).get("total_tokens", 0) for r in all_results)
    tok_prompt = sum((r.get("tokens") or {}).get("prompt_tokens", 0) for r in all_results)
    tok_compl = sum((r.get("tokens") or {}).get("completion_tokens", 0) for r in all_results)
    tok_calls = sum((r.get("tokens") or {}).get("calls", 0) for r in all_results)
    tok_cache_hit = sum((r.get("tokens") or {}).get("prompt_cache_hit_tokens", 0) for r in all_results)
    tok_cache_miss = sum((r.get("tokens") or {}).get("prompt_cache_miss_tokens", 0) for r in all_results)
    tok_reasoning = sum((r.get("tokens") or {}).get("reasoning_tokens", 0) for r in all_results)
    n_tok = sum(1 for r in all_results if r.get("tokens"))

    # error-recovery (ToolCoder's code-review etc.): how often an initial error was fixed
    recovered = sum(1 for r in all_results if r.get("recovered_from_error"))
    recovered_scopes = [r.get("initial_error_scope") for r in all_results
                        if r.get("recovered_from_error") and r.get("initial_error_scope")]

    # Spotify's RestBench tasks are dynamic and do not have stable semantic gold
    # answers in this harness. The comparable automatic signal is the post-run,
    # oracle-only method/path/status evaluation attached after the isolated agent
    # boundary. Keep it separate from generic semantic ``success``.
    spotify_evals = [r.get("spotify_oracle_path_evaluation") for r in all_results
                     if isinstance(r.get("spotify_oracle_path_evaluation"), dict)]
    spotify_metrics = None
    if spotify_evals:
        complete_ok = sum(1 for e in spotify_evals if e.get("score") is True)
        recalls = []
        for e in spotify_evals:
            required = len(e.get("required_steps") or [])
            matched = int(e.get("matched_steps") or 0)
            recalls.append((matched / required) if required else 0.0)
        spotify_metrics = {
            "tasks_scored": len(spotify_evals),
            "oracle_route_http_success": complete_ok,
            "oracle_route_http_success_rate": round(complete_ok / len(spotify_evals) * 100, 1),
            "mean_gold_route_recall": round(sum(recalls) / len(recalls) * 100, 1),
        }

    summary = {
        "agent":      agent_name,
        "timestamp":  datetime.now().isoformat(),
        "tasks_run":  total,
        "success":    success,
        "failed":     failed,
        "unverified": unverified,
        "success_rate": round(success / total * 100, 1) if total else 0,
        "verified_tasks": success + failed,
        "silent_failures": silent,
        "answered_without_tool_use": no_tool,
        "error_type_distribution": dict(Counter(all_errors)),
        "scope_distribution":      dict(Counter(all_scopes)),
        "avg_turns_wasted_after_first_error":
            round(sum(turns_wasted) / len(turns_wasted), 2) if turns_wasted else 0,
        "log_anomaly_distribution": dict(Counter(all_anomalies)),
        "tokens": {
            "total": tok_total, "prompt": tok_prompt, "completion": tok_compl,
            "calls": tok_calls, "tasks_with_usage": n_tok,
            "avg_total_per_task": round(tok_total / n_tok, 1) if n_tok else 0,
            "prompt_cache_hit_tokens": tok_cache_hit,
            "prompt_cache_miss_tokens": tok_cache_miss,
            "reasoning_tokens": tok_reasoning,
        },
        "error_recovery": {
            "recovered_count": recovered,
            "recovered_initial_scopes": dict(Counter(recovered_scopes)),
        },
        "spotify_restbench": spotify_metrics,
        "tasks": all_results,
    }

    path = os.path.join(__import__("config").SUMMARY_DIR, f"{agent_name}_summary.json")
    path = _write_json_durable(path, summary)

    print(f"\n[SUMMARY] Saved → {path}")
    print(f"  Tasks run:      {total}")
    print(f"  Success:        {success} ({summary['success_rate']}% of all; verified={success+failed})")
    print(f"  Failed:         {failed}")
    print(f"  Unverified:     {unverified} (answer given, no ground truth)")
    print(f"  Silent failures:{silent}")
    print(f"  Answered w/o tool use: {no_tool}")
    print(f"  Error types:    {dict(Counter(all_errors))}")
    print(f"  Scopes hit:     {dict(Counter(all_scopes))}")
    if spotify_metrics:
        print("  Spotify oracle route+HTTP: "
              f"{spotify_metrics['oracle_route_http_success']}/{spotify_metrics['tasks_scored']} "
              f"({spotify_metrics['oracle_route_http_success_rate']}%)")
        print("  Spotify mean gold-route recall: "
              f"{spotify_metrics['mean_gold_route_recall']}%")
    return summary
