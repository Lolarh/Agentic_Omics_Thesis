#!/usr/bin/env python3
"""Smoke-test the biology-informed PBMC3k agent with candidate-specific PCA metrics.

Run from the Agentic_Thesis project root:
    python 01_01_agent_repeatability_candidate_pcs_smoke_test.py

Default smoke-test design:
- Two independent executions, with at most two completed evaluations each
- GPT-5 LLM seed = 42 (best-effort reproducibility, not a guarantee)
- Leiden random_state = 42 for every clustering iteration in every run
- Candidate-specific evaluation uses the first n_pcs PCs for silhouette, DBI,
  and within-cluster dispersion; metric_n_pcs is logged for interpretation
- Cell IDs are saved beside selected and per-iteration labels for alignment

After the smoke test succeeds, increase N_REPEATED_RUNS and
MAX_AGENT_ITERATIONS below for the full repeatability experiment.

This script creates a new timestamped output folder under experiments/ and
never overwrites existing experiment records. It makes LLM API calls and may
incur API costs.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Load environment variables before importing modules that instantiate ChatOpenAI.
load_dotenv(override=True)

from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from src import data_load as dl
from src import preprocessing_pipeline as prp
from src import agent as agent_module
from src.graph import graph
from src.decision import OptimizationDecision, FinalExperimentSelection
from src.experiment_logger import create_experiment_log, save_experiment_log
from src.wt_biology_prompts import PBMC3K_BIOLOGICAL_CONTEXT


# -------------------------
# Fixed experimental design
# -------------------------
N_REPEATED_RUNS = 8 
LLM_SEED = 42
LEIDEN_SEED = 42
MAX_AGENT_ITERATIONS = 16  
MODEL_NAME = "gpt-5"
DATASET_NAME = "PBMC3k"
CONDITION = "biology"
AGENT_VERSION = "v0.10-candidate-pcs-llm-seed42-repeatability"
INITIAL_PARAMETERS = {
    "n_neighbors": 15,
    "resolution": 1.0,
    "n_pcs": 30,
}
OUTPUT_ROOT = Path("experiments")
TEMPERATURE = 0.0


class FingerprintLoggingHandler(BaseCallbackHandler):
    """Capture response metadata exposed by LangChain for each LLM call."""

    def __init__(self):
        self.current_run_index = None
        self.records = []

    def on_llm_end(self, response, **kwargs):
        # Chat generations normally expose the raw provider metadata on the message.
        for generation_group in getattr(response, "generations", []) or []:
            for generation in generation_group or []:
                message = getattr(generation, "message", None)
                metadata = getattr(message, "response_metadata", {}) or {}
                generation_info = getattr(generation, "generation_info", {}) or {}
                llm_output = getattr(response, "llm_output", {}) or {}
                fingerprint = (
                    metadata.get("system_fingerprint")
                    or generation_info.get("system_fingerprint")
                    or llm_output.get("system_fingerprint")
                )
                self.records.append({
                    "run_index": self.current_run_index,
                    "system_fingerprint": fingerprint,
                    "model_name": metadata.get("model_name") or llm_output.get("model_name"),
                    "response_metadata_keys": sorted(str(key) for key in metadata.keys()),
                    "generation_info_keys": sorted(str(key) for key in generation_info.keys()),
                    "llm_output_keys": sorted(str(key) for key in llm_output.keys()),
                })


def configure_agent_for_this_run(fingerprint_handler) -> None:
    """Configure the LLM client and agent budget without editing src files."""
    client = ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        seed=LLM_SEED,
        callbacks=[fingerprint_handler],
    )
    agent_module.llm = client
    agent_module.structured_llm = client.with_structured_output(OptimizationDecision)
    agent_module.final_selection_llm = client.with_structured_output(FinalExperimentSelection)
    agent_module.MAX_ITERATIONS = MAX_AGENT_ITERATIONS


def _json_default(value):
    """Convert common NumPy and Path objects to JSON-compatible values."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=_json_default)
        handle.write("\n")


def main() -> dict:
    fingerprint_handler = FingerprintLoggingHandler()
    configure_agent_for_this_run(fingerprint_handler)

    if N_REPEATED_RUNS < 2:
        raise ValueError("N_REPEATED_RUNS must be at least 2 for a repeatability experiment.")
    if MAX_AGENT_ITERATIONS < 1:
        raise ValueError("MAX_AGENT_ITERATIONS must be at least 1.")

    run_id = datetime.now().strftime("llm_seed42_agent_repeatability_%Y%m%d_%H%M%S_%f")
    output_dir = OUTPUT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    started_at = time.time()
    successful_runs = []
    failed_runs = []

    print("Loading PBMC3k and running the project preprocessing pipeline...")
    pbmc_data = dl.load_pbmc3k()
    pbmc_data = prp.run_preprocessing_pipeline(pbmc_data)

    if "X_pca" not in pbmc_data.obsm:
        raise RuntimeError("Preprocessing did not create X_pca.")
    available_pcs = int(pbmc_data.obsm["X_pca"].shape[1])
    if available_pcs < 50:
        raise ValueError(
            f"The agent search space permits up to 50 PCs, but preprocessing created {available_pcs}."
        )
    cell_ids = pbmc_data.obs_names.astype(str).tolist()
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("Preprocessed PBMC3k contains duplicate cell IDs.")

    run_config = {
        "run_id": run_id,
        "dataset": DATASET_NAME,
        "condition": CONDITION,
        "agent_version": AGENT_VERSION,
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "llm_seed": LLM_SEED,
        "llm_seed_note": "Best-effort reproducibility only; identical model outputs are not guaranteed.",
        "n_repeated_runs": N_REPEATED_RUNS,
        "leiden_seed_fixed_for_all_runs": LEIDEN_SEED,
        "max_completed_clustering_evaluations_per_run": MAX_AGENT_ITERATIONS,
        "initial_parameters": INITIAL_PARAMETERS,
        "n_cells": int(pbmc_data.n_obs),
        "n_genes_after_preprocessing": int(pbmc_data.n_vars),
        "cell_ids_in_order": cell_ids,
        "preprocessing": {
            "pipeline": "src.preprocessing_pipeline.run_preprocessing_pipeline",
            "n_pcs_available": available_pcs,
        },
        "metric_evaluation": {
            "embedding": "X_pca",
            "pc_selection": "First n_pcs PCs for each candidate",
            "n_pcs_source": "Candidate clustering parameters recorded in adata.uns['clustering_params']",
            "logged_metric_fields": ["metric_embedding", "metric_n_pcs"],
            "comparability_note": (
                "Silhouette, Davies-Bouldin, and within-cluster dispersion are evaluated "
                "in each candidate's selected PCA space. Scores from candidates using "
                "different n_pcs are not strictly dimension-matched; interpret them cautiously."
            ),
        },
        "label_alignment": {
            "cell_ids_saved_with_selected_labels": True,
            "cell_ids_saved_with_each_iteration_labels": True,
            "validation": (
                "Check label count and final AnnData observation order against the preprocessed "
                "cell IDs; the pipeline is expected to preserve observation order throughout."
            ),
        },
        "interpretation": (
            "This experiment tests repeatability of the full adaptive agent workflow while holding the "
            "configured LLM seed, Leiden seed, initial state, preprocessing, and evaluation setup fixed. "
            "It does not guarantee bitwise-identical results across API/backend changes."
        ),
    }
    _write_json(output_dir / "run_config.json", run_config)

    print(f"LLM: {MODEL_NAME}; configured seed={LLM_SEED} (best-effort)")
    print(f"Fixed Leiden seed: {LEIDEN_SEED}")
    print(f"Repeated runs: {N_REPEATED_RUNS}; maximum evaluations per run: {MAX_AGENT_ITERATIONS}")
    print("Metric evaluation: candidate-specific leading n_pcs PCs (logged as metric_n_pcs)")
    print(f"Cells: {pbmc_data.n_obs}; genes after preprocessing: {pbmc_data.n_vars}")
    print(f"Output directory: {output_dir.resolve()}")
    print("\nWARNING: this script makes LLM API calls and may incur API costs.")

    for run_index in range(1, N_REPEATED_RUNS + 1):
        fingerprint_handler.current_run_index = run_index
        experiment_id = f"{run_id}_PBMC3k_AGENT_BIOLOGY_REPEAT_{run_index:02d}_LSEED_{LEIDEN_SEED:02d}"
        print("\n" + "=" * 72)
        print(f"Starting repeat {run_index}/{N_REPEATED_RUNS}: {experiment_id}")
        print("=" * 72)

        experiment_log = create_experiment_log(
            experiment_id=experiment_id,
            dataset=DATASET_NAME,
            method="agentic",
            condition=CONDITION,
            agent_version=AGENT_VERSION,
            change="Agent repeatability with fixed seeds and candidate-specific PCA metric evaluation",
            reason_for_change=(
                "Smoke-test the adaptive agent workflow while keeping the configured LLM seed, "
                "Leiden seed, initial parameters, dataset, and preprocessing fixed. Evaluation "
                "metrics use the first n_pcs PCs specified by each candidate and log metric_n_pcs."
            ),
            llm_seed=LLM_SEED,
            leiden_seed=LEIDEN_SEED,
            max_iterations=MAX_AGENT_ITERATIONS,
            run_index=run_index,
            initial_parameters=INITIAL_PARAMETERS,
        )

        state_run = {
            "messages": [],
            "adata": pbmc_data.copy(),
            "parameters": INITIAL_PARAMETERS.copy(),
            "condition": CONDITION,
            "dataset_context": PBMC3K_BIOLOGICAL_CONTEXT,
            "leiden_seed": LEIDEN_SEED,
            "metrics": {},
            "history": [],
            "cluster_assignments": {},
            "experiment_log": experiment_log,
            "iteration": 0,
            "decision": "continue",
            "reason": "",
            "evaluated_configs": [],
            "clustering_time": 0.0,
            "evaluation_time": 0.0,
            "llm_decision_time": 0.0,
            "selected_iteration": None,
            "best_experiment": None,
            "final_selection_reason": None,
            "final_selection_time": 0.0,
        }

        try:
            fingerprint_handler.current_run_index = run_index
            result_run = graph.invoke(state_run)
            save_experiment_log(result_run["experiment_log"], output_dir=str(output_dir))

            best = result_run.get("best_experiment")
            selected_iteration = result_run.get("selected_iteration")
            if best is None or selected_iteration is None:
                raise RuntimeError("Agent completed without selecting a best iteration.")

            result_cell_ids = result_run["adata"].obs_names.to_numpy(dtype=str)
            expected_cell_ids = pbmc_data.obs_names.to_numpy(dtype=str)
            if not np.array_equal(result_cell_ids, expected_cell_ids):
                raise ValueError("Output cell IDs/order differ from the preprocessed input.")

            labels_by_iteration = result_run.get("cluster_assignments", {})
            if selected_iteration not in labels_by_iteration:
                # State implementations may use stringified iteration keys; accept only exact equivalent.
                if str(selected_iteration) in labels_by_iteration:
                    selected_labels = labels_by_iteration[str(selected_iteration)]
                else:
                    raise RuntimeError(f"Selected iteration {selected_iteration} has no saved labels.")
            else:
                selected_labels = labels_by_iteration[selected_iteration]
            selected_labels = np.asarray(selected_labels, dtype=str).reshape(-1)
            if len(selected_labels) != len(expected_cell_ids):
                raise ValueError("Selected cluster labels do not match the cell count.")

            # Record the selected partition's cluster-size distribution during the run.
            # String keys keep the mapping JSON/CSV-friendly and stable across serializers.
            selected_cluster_sizes = {
                str(cluster): int(size)
                for cluster, size in sorted(
                    zip(*np.unique(selected_labels, return_counts=True)),
                    key=lambda item: str(item[0]),
                )
            }

            np.save(output_dir / f"{experiment_id}_labels.npy", selected_labels)
            np.save(output_dir / f"{experiment_id}_cell_ids.npy", expected_cell_ids)

            saved_iterations = []
            for iteration, iteration_labels in sorted(
                labels_by_iteration.items(), key=lambda item: int(item[0])
            ):
                iteration_labels = np.asarray(iteration_labels, dtype=str).reshape(-1)

                # cluster_assignments are captured from adata.obs["leiden"] in observation
                # order. We validate label count and the final AnnData cell order here;
                # the clustering pipeline should preserve observation order at every iteration.
                # Saving the expected IDs beside each label vector makes downstream alignment
                # explicit, but is not by itself proof of an intermediate reorder never occurring.
                if len(iteration_labels) != len(expected_cell_ids):
                    raise ValueError(
                        f"Iteration {iteration} labels ({len(iteration_labels)}) do not match "
                        f"the cell count ({len(expected_cell_ids)})."
                    )
                if not np.array_equal(
                    result_run["adata"].obs_names.to_numpy(dtype=str), expected_cell_ids
                ):
                    raise ValueError(
                        f"Iteration {iteration}: AnnData cell IDs/order differ from the input."
                    )

                iteration_number = int(iteration)
                labels_path = output_dir / f"{experiment_id}_ITER_{iteration_number:02d}_labels.npy"
                cell_ids_path = output_dir / f"{experiment_id}_ITER_{iteration_number:02d}_cell_ids.npy"
                np.save(labels_path, iteration_labels)
                # Save the cell order explicitly beside every iteration's labels.
                np.save(cell_ids_path, expected_cell_ids)
                saved_iterations.append({
                    "iteration": iteration_number,
                    "labels_file": labels_path.name,
                    "cell_ids_file": cell_ids_path.name,
                    "n_cells": int(len(iteration_labels)),
                    "n_clusters": int(len(np.unique(iteration_labels))),
                    "is_selected": iteration_number == int(selected_iteration),
                })

            run_record = {
                "experiment_id": experiment_id,
                "run_index": run_index,
                "leiden_seed": LEIDEN_SEED,
                "llm_seed": LLM_SEED,
                "selected_iteration": int(selected_iteration),
                "selected_parameters": best.get("parameters", {}),
                "selected_metrics": best.get("metrics", {}),
                "n_cells": int(len(expected_cell_ids)),
                "n_clusters": int(len(np.unique(selected_labels))),
                "selected_cluster_sizes": selected_cluster_sizes,
                "n_evaluated_iterations": len(saved_iterations),
                "saved_iterations": saved_iterations,
                "termination_reason": result_run.get("experiment_log", {}).get("termination_reason"),
                "status": "success",
            }
            successful_runs.append(run_record)

            print("SUCCESS")
            print("Selected iteration:", selected_iteration)
            print("Selected parameters:", run_record["selected_parameters"])
            print("Evaluated iterations:", len(saved_iterations))
            print("Clusters in selected partition:", run_record["n_clusters"])

        except Exception as exc:
            failed_record = {
                "experiment_id": experiment_id,
                "run_index": run_index,
                "leiden_seed": LEIDEN_SEED,
                "llm_seed": LLM_SEED,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failed_runs.append(failed_record)
            print(f"FAILED ({type(exc).__name__}): {exc}")
            print("Continuing to the next repeated run.")

    summary = {
        "run_id": run_id,
        "dataset": DATASET_NAME,
        "condition": CONDITION,
        "agent_version": AGENT_VERSION,
        "model": MODEL_NAME,
        "llm_seed": LLM_SEED,
        "llm_seed_best_effort_only": True,
        "leiden_seed_fixed": LEIDEN_SEED,
        "max_completed_clustering_evaluations_per_run": MAX_AGENT_ITERATIONS,
        "n_requested": N_REPEATED_RUNS,
        "n_successful": len(successful_runs),
        "n_failed": len(failed_runs),
        "elapsed_seconds": round(time.time() - started_at, 3),
        "successful_runs": successful_runs,
        "failed_runs": failed_runs,
    }
    # Save all fingerprint metadata exposed by the installed LangChain integration.
    # A null fingerprint means it was not present in the callback-visible response metadata.
    fingerprint_records = fingerprint_handler.records
    _write_json(output_dir / "llm_response_metadata.json", {
        "note": (
            "system_fingerprint is recorded only when exposed by LangChain response metadata. "
            "A null value does not prove the provider omitted it; the installed integration may "
            "not expose it through this callback."
        ),
        "n_llm_responses_observed": len(fingerprint_records),
        "n_responses_with_fingerprint": sum(
            record.get("system_fingerprint") is not None for record in fingerprint_records
        ),
        "records": fingerprint_records,
    })
    _write_json(output_dir / "run_summary.json", summary)

    print("\n" + "=" * 72)
    print("REPEATABILITY EXPERIMENT SUMMARY")
    print("=" * 72)
    print(f"Successful runs: {len(successful_runs)}/{N_REPEATED_RUNS}")
    print(f"Failed runs: {len(failed_runs)}")
    print(f"Elapsed time: {summary['elapsed_seconds']} seconds")
    print(f"Results saved to: {output_dir.resolve()}")
    if failed_runs:
        print("\nFailed runs:")
        for record in failed_runs:
            print(
                f"  Repeat {record['run_index']}: {record['error_type']}: {record['error']}"
            )

    return {"output_dir": output_dir, "summary": summary}


if __name__ == "__main__":
    main()
