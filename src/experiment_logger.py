from datetime import datetime, timezone
import json
import time
from pathlib import Path


def create_experiment_log(
    experiment_id,
    dataset,
    method,
    condition,
    agent_version=None,
    change="",
    reason_for_change="",
    llm_seed=None,
    leiden_seed=None,
    max_iterations=None,
    run_index=None,
    initial_parameters=None,
):
    """Create a record with explicit seeds and run-budget metadata."""
    return {
        "experiment_id": str(experiment_id),
        "date": datetime.now(timezone.utc).isoformat(),
        # Internal timer; removed before JSON serialization.
        "start_time": time.perf_counter(),
        "dataset": dataset,
        "method": method,
        "condition": condition,
        "agent_version": agent_version,
        "llm_seed": llm_seed,
        "leiden_seed": leiden_seed,
        "max_iterations": max_iterations,
        "run_index": run_index,
        "initial_parameters": (
            dict(initial_parameters) if initial_parameters is not None else None
        ),
        "change": change,
        "reason_for_change": reason_for_change,
        "iterations": [],
    }


def save_experiment_log(log, output_dir="experiments"):
    """Save experiment record as JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{log['experiment_id']}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False, allow_nan=False)
    return path


def add_final_result(
    log,
    best_experiment,
    final_selection_reason=None,
    final_selection_time=None,
):
    """Add the selected experiment and total runtime to the log."""
    log["total_runtime_seconds"] = time.perf_counter() - log["start_time"]
    termination_reason = log.get("termination_reason")
    log.pop("start_time", None)

    if best_experiment is None:
        log["final_result"] = {
            "termination_reason": termination_reason,
            "best_experiment": None,
            "final_selection_reason": final_selection_reason,
            "final_selection_time_seconds": final_selection_time,
        }
        return log

    log["final_result"] = {
        "termination_reason": termination_reason,
        "best_iteration": best_experiment["iteration"],
        "best_parameters": dict(best_experiment["parameters"]),
        "best_metrics": dict(best_experiment["metrics"]),
        "final_selection_reason": final_selection_reason,
        "final_selection_time_seconds": final_selection_time,
    }
    return log
