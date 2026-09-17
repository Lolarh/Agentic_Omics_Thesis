from datetime import datetime
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
    reason_for_change="",):
    """Create a new experiment record."""

    return {
        "experiment_id": experiment_id,
        "date": datetime.now().isoformat(),

        # Used internally to calculate total runtime
        "start_time": time.perf_counter(),

        "dataset": dataset,
        "method": method,
        "condition": condition,
        "agent_version": agent_version,
        "change": change,
        "reason_for_change": reason_for_change,
        "iterations": [],}

def save_experiment_log(
    log,
    output_dir="experiments",):
    """Save experiment record as JSON."""

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,)

    path = output_dir / (
        f"{log['experiment_id']}.json")

    with open(path, "w") as f:
        json.dump(log,f,indent=2,)
    
    return path


def add_final_result(
    log,
    best_experiment,
    final_selection_reason=None,
    final_selection_time=None,):
    """
    Add the final experiment selection and
    total runtime to the experiment log.
    """

    # --------------------------------------------------
    # Calculate total runtime
    # --------------------------------------------------

    total_runtime = (time.perf_counter() - log["start_time"])

    log["total_runtime_seconds"] = total_runtime

    # --------------------------------------------------
    # Retrieve exploration termination reason
    # --------------------------------------------------

    termination_reason = log.get("termination_reason")

    # --------------------------------------------------
    # Remove internal timer before saving
    # --------------------------------------------------

    log.pop("start_time", None)

    # --------------------------------------------------
    # No experiment was selected
    # --------------------------------------------------

    if best_experiment is None:

        log["final_result"] = {
            "termination_reason": termination_reason,
            "best_experiment": None,
            "final_selection_reason": (
                final_selection_reason
            ),
            "final_selection_time_seconds": (
                final_selection_time
            ),
        }

        return log

    # --------------------------------------------------
    # Store selected experiment
    # --------------------------------------------------

    log["final_result"] = {
        "termination_reason": termination_reason,

        "best_iteration": (best_experiment["iteration"]),

        "best_parameters": (best_experiment["parameters"].copy()),

        "best_metrics": (best_experiment["metrics"].copy()),


        "final_selection_reason": (final_selection_reason),

        "final_selection_time_seconds": (final_selection_time),
    }

    return log