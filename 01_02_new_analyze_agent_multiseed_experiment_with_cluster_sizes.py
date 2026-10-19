#!/usr/bin/env python3
"""
NEW: Reusable, no-LLM analysis for a saved multi-seed agent experiment.

Run from the project root:
    python new_analyze_agent_multiseed_experiment.py experiments/<experiment-folder> --k 9

Outputs are written to <experiment-folder>/new_analysis_k9/.
This script does not rerun the agent, clustering, or LLM.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


PARAMETERS = ["n_neighbors", "n_pcs", "resolution"]
METRIC_ALIASES = {
    "silhouette": ["silhouette_score", "silhouette"],
    "davies_bouldin": ["davies_bouldin_score", "davies_bouldin", "dbi"],
    "wc_dispersion": ["wc_dispersion_score", "wc_dispersion"],
}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def metric_value(metrics: dict, alias: str):
    for key in METRIC_ALIASES[alias]:
        if metrics.get(key) is not None:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                pass
    return np.nan


def consensus_partition(partitions: list[np.ndarray], k: int):
    if len(partitions) < 2:
        raise ValueError("Consensus requires at least two partitions.")
    n_cells = len(partitions[0])
    if any(len(x) != n_cells for x in partitions):
        raise ValueError("Partitions have inconsistent cell counts.")
    if not 2 <= k < n_cells:
        raise ValueError(f"k must be between 2 and {n_cells - 1}.")
    coassoc = np.zeros((n_cells, n_cells), dtype=np.float32)
    for labels in partitions:
        coassoc += (labels[:, None] == labels[None, :])
    coassoc /= len(partitions)
    distance = np.clip(1.0 - coassoc, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    try:
        model = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="average"
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=k, affinity="precomputed", linkage="average"
        )
    return model.fit_predict(distance).astype(str), coassoc


def load_labels(path: Path, expected_n: int) -> np.ndarray:
    labels = np.asarray(np.load(path, allow_pickle=False)).astype(str).reshape(-1)
    if len(labels) != expected_n:
        raise ValueError(f"{path.name}: {len(labels)} labels; expected {expected_n}")
    if len(np.unique(labels)) < 2:
        raise ValueError(f"{path.name}: fewer than two clusters")
    return labels


def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    path: Path,
    vmin=0.0,
    vmax=1.0,
    label_map: dict | None = None,
    annotate: bool = True,
):
    """Save a readable heatmap; optionally replace long run IDs with short labels."""
    n_rows, n_cols = matrix.shape
    fig, ax = plt.subplots(
        figsize=(max(5.5, 1.35 * n_cols), max(4.5, 1.0 * n_rows))
    )
    values = matrix.to_numpy(dtype=float)
    im = ax.imshow(values, vmin=vmin, vmax=vmax, aspect="auto", cmap="viridis")

    x_labels = [label_map.get(str(x), str(x)) if label_map else str(x) for x in matrix.columns]
    y_labels = [label_map.get(str(x), str(x)) if label_map else str(x) for x in matrix.index]
    ax.set_xticks(range(n_cols), x_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows), y_labels, fontsize=9)
    ax.set_title(title, pad=12)

    if annotate:
        midpoint = (vmin + vmax) / 2
        for i in range(n_rows):
            for j in range(n_cols):
                value = values[i, j]
                if np.isfinite(value):
                    ax.text(
                        j, i, f"{value:.2f}", ha="center", va="center",
                        fontsize=9, color="black" if value > midpoint else "white"
                    )

    fig.colorbar(im, ax=ax, label="Similarity", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path, help="Folder containing run_summary.json")
    parser.add_argument("--k", type=int, default=9, help="Consensus cluster count (default: 9)")
    args = parser.parse_args()
    exp_dir = args.experiment_dir.expanduser().resolve()
    summary_path = exp_dir / "run_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Cannot find {summary_path}")
    summary = read_json(summary_path)
    successful = [r for r in summary.get("successful_runs", []) if r.get("status") == "success"]
    if not successful:
        raise ValueError("run_summary.json contains no successful runs.")

    out_dir = exp_dir / f"new_analysis_k{args.k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_cells = int(successful[0]["n_cells"])
    ref_cell_ids = np.load(
        exp_dir / f"{successful[0]['experiment_id']}_cell_ids.npy", allow_pickle=False
    ).astype(str).reshape(-1)
    if len(ref_cell_ids) != n_cells or len(set(ref_cell_ids)) != n_cells:
        raise ValueError("Reference cell IDs have an invalid length or duplicates.")

    trajectory_rows = []
    run_objects = []
    best_labels = []
    best_run_ids = []
    run_consensus_labels = []
    run_consensus_rows = []
    within_pair_rows = []
    within_consensus_rows = []
    cross_iteration_rows = []
    final_rows = []

    for run in successful:
        eid = run["experiment_id"]
        if int(run["n_cells"]) != n_cells:
            raise ValueError(f"{eid}: cell count differs from first successful run.")
        cell_path = exp_dir / f"{eid}_cell_ids.npy"
        cell_ids = np.load(cell_path, allow_pickle=False).astype(str).reshape(-1)
        if not np.array_equal(cell_ids, ref_cell_ids):
            raise ValueError(f"{eid}: cell IDs/order differ from the first run.")

        log_path = exp_dir / f"{eid}.json"
        log = read_json(log_path) if log_path.exists() else {}
        log_iterations = {
            int(item.get("iteration", idx)): item
            for idx, item in enumerate(log.get("iterations", []))
        }
        saved = sorted(run.get("saved_iterations", []), key=lambda x: int(x["iteration"]))
        labels_by_iter = {}
        rows_by_iter = []

        for item in saved:
            it = int(item["iteration"])
            labels_path = exp_dir / item["labels_file"]
            labels = load_labels(labels_path, n_cells)
            actual_k = len(np.unique(labels))
            if actual_k != int(item["n_clusters"]):
                raise ValueError(f"{labels_path.name}: cluster count does not match run_summary.json.")
            labels_by_iter[it] = labels
            log_item = log_iterations.get(it, {})
            params = log_item.get("parameters") or {}
            metrics = log_item.get("metrics") or {}
            row = {
                "experiment_id": eid,
                "run_index": run.get("run_index"),
                "leiden_seed": run.get("leiden_seed"),
                "llm_seed": run.get("llm_seed", summary.get("llm_seed")),
                "iteration": it,
                "n_clusters": actual_k,
                "is_selected_best": it == int(run["selected_iteration"]),
                "n_neighbors": params.get("n_neighbors"),
                "n_pcs": params.get("n_pcs"),
                "resolution": params.get("resolution"),
                "silhouette": metric_value(metrics, "silhouette"),
                "davies_bouldin": metric_value(metrics, "davies_bouldin"),
                "wc_dispersion": metric_value(metrics, "wc_dispersion"),
            }
            # Use summary-selected values as fallback for selected iteration only.
            if row["is_selected_best"]:
                for p in PARAMETERS:
                    if row[p] is None:
                        row[p] = (run.get("selected_parameters") or {}).get(p)
                selected_metrics = run.get("selected_metrics") or {}
                for name in METRIC_ALIASES:
                    if pd.isna(row[name]):
                        row[name] = metric_value(selected_metrics, name)
            trajectory_rows.append(row)
            rows_by_iter.append(row)

        selected_it = int(run["selected_iteration"])
        if selected_it not in labels_by_iter:
            raise ValueError(f"{eid}: selected iteration {selected_it} has no saved labels.")
        selected_labels = labels_by_iter[selected_it]
        best_labels.append(selected_labels)
        best_run_ids.append(eid)

        # Carry the experiment-time cluster-size distribution into the final CSV.
        # For older experiment summaries without this field, derive it from the
        # already-saved labels for the selected iteration (no reclustering).
        selected_cluster_sizes = run.get("selected_cluster_sizes")
        if selected_cluster_sizes is None:
            selected_cluster_sizes = {
                str(label): int(count)
                for label, count in pd.Series(selected_labels).value_counts().sort_index().items()
            }
        else:
            selected_cluster_sizes = {
                str(label): int(count)
                for label, count in selected_cluster_sizes.items()
            }

        final_rows.append({
            "experiment_id": eid,
            "run_index": run.get("run_index"),
            "leiden_seed": run.get("leiden_seed"),
            "llm_seed": run.get("llm_seed", summary.get("llm_seed")),
            "selected_iteration": selected_it,
            "selected_parameters": json.dumps(run.get("selected_parameters", {}), sort_keys=True),
            "n_clusters": int(len(np.unique(selected_labels))),
            "selected_cluster_sizes": json.dumps(selected_cluster_sizes, sort_keys=True),
            **{f"selected_{k}": v for k, v in (run.get("selected_metrics") or {}).items()},
        })

        iter_nums = sorted(labels_by_iter)
        if len(iter_nums) >= 2:
            parts = [labels_by_iter[i] for i in iter_nums]
            cons, coassoc = consensus_partition(parts, args.k)
            run_consensus_labels.append(cons)
            sizes = pd.Series(cons).value_counts().sort_index()
            run_out = out_dir / "within_run_consensus" / eid
            run_out.mkdir(parents=True, exist_ok=True)
            np.save(run_out / "new_trajectory_consensus_labels.npy", cons)
            np.save(run_out / "new_trajectory_coassociation_matrix.npy", coassoc)
            pd.DataFrame({
                "cell_id": ref_cell_ids,
                "trajectory_consensus_label": cons,
                "selected_best_label": selected_labels,
            }).to_csv(run_out / "new_consensus_and_selected_labels.csv", index=False)
            pd.DataFrame({"cluster": sizes.index, "n_cells": sizes.values}).to_csv(
                run_out / "new_trajectory_consensus_cluster_sizes.csv", index=False
            )
            run_consensus_rows.append({
                "experiment_id": eid,
                "run_index": run.get("run_index"),
                "leiden_seed": run.get("leiden_seed"),
                "n_iterations": len(iter_nums),
                "iterations": ",".join(map(str, iter_nums)),
                "consensus_k": args.k,
                "consensus_vs_selected_ari": adjusted_rand_score(cons, selected_labels),
                "consensus_vs_selected_nmi": normalized_mutual_info_score(cons, selected_labels),
                "cluster_sizes": json.dumps({str(k): int(v) for k, v in sizes.items()}),
            })
            for it in iter_nums:
                lab = labels_by_iter[it]
                within_consensus_rows.append({
                    "experiment_id": eid,
                    "run_index": run.get("run_index"),
                    "leiden_seed": run.get("leiden_seed"),
                    "iteration": it,
                    "n_clusters": len(np.unique(lab)),
                    "is_selected_best": it == selected_it,
                    "ari_vs_trajectory_consensus": adjusted_rand_score(cons, lab),
                    "nmi_vs_trajectory_consensus": normalized_mutual_info_score(cons, lab),
                })
            for ia, ib in combinations(iter_nums, 2):
                within_pair_rows.append({
                    "experiment_id": eid,
                    "run_index": run.get("run_index"),
                    "leiden_seed": run.get("leiden_seed"),
                    "iteration_a": ia,
                    "iteration_b": ib,
                    "ari": adjusted_rand_score(labels_by_iter[ia], labels_by_iter[ib]),
                    "nmi": normalized_mutual_info_score(labels_by_iter[ia], labels_by_iter[ib]),
                })

        run_objects.append({
            "run": run, "labels_by_iter": labels_by_iter, "selected_labels": selected_labels
        })

    # Cross-run comparisons at matching iteration numbers. These are descriptive only:
    # adaptive iteration numbers do not imply identical search decisions.
    for ra, rb in combinations(run_objects, 2):
        a, b = ra["run"], rb["run"]
        shared = sorted(set(ra["labels_by_iter"]) & set(rb["labels_by_iter"]))
        for it in shared:
            la, lb = ra["labels_by_iter"][it], rb["labels_by_iter"][it]
            cross_iteration_rows.append({
                "experiment_id_a": a["experiment_id"],
                "run_index_a": a.get("run_index"),
                "leiden_seed_a": a.get("leiden_seed"),
                "experiment_id_b": b["experiment_id"],
                "run_index_b": b.get("run_index"),
                "leiden_seed_b": b.get("leiden_seed"),
                "iteration_position": it,
                "n_clusters_a": len(np.unique(la)),
                "n_clusters_b": len(np.unique(lb)),
                "ari": adjusted_rand_score(la, lb),
                "nmi": normalized_mutual_info_score(la, lb),
                "interpretation": "descriptive same-iteration-position comparison; paths are adaptive",
            })

    # Final-best pairwise agreement.
    final_ari = pd.DataFrame(index=best_run_ids, columns=best_run_ids, dtype=float)
    final_nmi = pd.DataFrame(index=best_run_ids, columns=best_run_ids, dtype=float)
    final_pair_rows = []
    for i, j in combinations(range(len(best_labels)), 2):
        ari = float(adjusted_rand_score(best_labels[i], best_labels[j]))
        nmi = float(normalized_mutual_info_score(best_labels[i], best_labels[j]))
        final_ari.iloc[i, j] = final_ari.iloc[j, i] = ari
        final_nmi.iloc[i, j] = final_nmi.iloc[j, i] = nmi
        final_pair_rows.append({
            "experiment_id_a": best_run_ids[i],
            "leiden_seed_a": successful[i].get("leiden_seed"),
            "experiment_id_b": best_run_ids[j],
            "leiden_seed_b": successful[j].get("leiden_seed"),
            "ari": ari, "nmi": nmi,
        })
    np.fill_diagonal(final_ari.values, 1.0)
    np.fill_diagonal(final_nmi.values, 1.0)

    # Consensus across final-best partitions.
    final_cons, final_coassoc = consensus_partition(best_labels, args.k) if len(best_labels) >= 2 else (None, None)
    if final_cons is not None:
        np.save(out_dir / "new_final_best_coassociation_matrix.npy", final_coassoc)
        np.save(out_dir / "new_final_best_consensus_labels.npy", final_cons)
        pd.DataFrame({
            "cell_id": ref_cell_ids,
            "final_best_consensus_label": final_cons,
        }).to_csv(out_dir / "new_final_best_consensus_labels.csv", index=False)
        final_cons_sizes = pd.Series(final_cons).value_counts().sort_index()
        pd.DataFrame({"cluster": final_cons_sizes.index, "n_cells": final_cons_sizes.values}).to_csv(
            out_dir / "new_final_best_consensus_cluster_sizes.csv", index=False
        )
        final_cons_compare = []
        for eid, run, labels in zip(best_run_ids, successful, best_labels):
            final_cons_compare.append({
                "experiment_id": eid,
                "run_index": run.get("run_index"),
                "leiden_seed": run.get("leiden_seed"),
                "n_clusters_in_selected_partition": len(np.unique(labels)),
                "ari_vs_final_best_consensus": adjusted_rand_score(final_cons, labels),
                "nmi_vs_final_best_consensus": normalized_mutual_info_score(final_cons, labels),
            })
        pd.DataFrame(final_cons_compare).to_csv(
            out_dir / "new_final_best_consensus_vs_runs.csv", index=False
        )

    # Tables.
    trajectory_df = pd.DataFrame(trajectory_rows).sort_values(["run_index", "iteration"])
    trajectory_df.to_csv(out_dir / "new_iteration_trajectory_table.csv", index=False)
    pd.DataFrame(final_rows).to_csv(out_dir / "new_final_best_selection_table.csv", index=False)
    pd.DataFrame(run_consensus_rows).to_csv(out_dir / "new_within_run_trajectory_consensus_summary.csv", index=False)
    pd.DataFrame(within_consensus_rows).to_csv(out_dir / "new_trajectory_consensus_vs_iterations.csv", index=False)
    pd.DataFrame(within_pair_rows).to_csv(out_dir / "new_within_run_pairwise_iteration_similarity.csv", index=False)
    pd.DataFrame(cross_iteration_rows).to_csv(out_dir / "new_cross_run_same_iteration_position_similarity.csv", index=False)
    pd.DataFrame(final_pair_rows).to_csv(out_dir / "new_final_best_pairwise_similarity.csv", index=False)
    final_ari.to_csv(out_dir / "new_final_best_ari_matrix.csv")
    final_nmi.to_csv(out_dir / "new_final_best_nmi_matrix.csv")

    # Figures: trajectories, agreement heatmaps, and selected cluster counts.
    # Short labels keep figures readable; full experiment IDs remain in saved tables.
    run_labels = {
        r["experiment_id"]: f"Run {r.get('run_index')}"
        for r in successful
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for ax, p in zip(axes, PARAMETERS):
        for eid, group in trajectory_df.groupby("experiment_id", sort=False):
            group = group.sort_values("iteration")
            ax.plot(group["iteration"], group[p], marker="o", linewidth=1.2,
                    label=run_labels.get(eid, eid))
            best = group[group["is_selected_best"]]
            if not best.empty and best[p].notna().any():
                ax.scatter(best["iteration"], best[p], marker="*", s=110,
                           edgecolors="black", linewidths=0.5, zorder=5)
        ax.set_title(p.replace("_", " ").title())
        ax.set_xlabel("Iteration position")
        ax.set_ylabel("Selected value")
        ax.grid(alpha=0.25)
    axes[-1].legend(fontsize=6, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Parameter trajectories (star = selected best iteration)")
    fig.tight_layout()
    fig.savefig(out_dir / "new_fig_parameter_trajectories.png", dpi=250, bbox_inches="tight")
    fig.savefig(out_dir / "new_fig_parameter_trajectories.pdf", bbox_inches="tight")
    plt.close(fig)

    metric_specs = [
        ("silhouette", "Silhouette score"),
        ("davies_bouldin", "Davies–Bouldin index"),
        ("wc_dispersion", "WC-dispersion"),
    ]
    # Three metrics remain in the analysis; use three panels and no empty fourth axis.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (metric, title) in zip(axes, metric_specs):
        for eid, group in trajectory_df.groupby("experiment_id", sort=False):
            group = group.sort_values("iteration").dropna(subset=[metric])
            if group.empty:
                continue
            ax.plot(group["iteration"], group[metric], marker="o", linewidth=1.1,
                    markersize=4, label=run_labels.get(eid, eid))
            best = group[group["is_selected_best"]]
            if not best.empty:
                ax.scatter(best["iteration"], best[metric], marker="*", s=110,
                           edgecolors="black", linewidths=0.5, zorder=5)
        ax.set_title(title)
        ax.set_xlabel("Iteration position")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Metric value")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Clustering metric trajectories (star = selected best iteration)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "new_fig_metric_trajectories.png", dpi=250, bbox_inches="tight")
    fig.savefig(out_dir / "new_fig_metric_trajectories.pdf", bbox_inches="tight")
    plt.close(fig)

    plot_heatmap(
        final_ari,
        "Pairwise ARI: selected best partitions",
        out_dir / "new_fig_final_best_ari_heatmap.png",
        label_map=run_labels,
    )
    plot_heatmap(
        final_nmi,
        "Pairwise NMI: selected best partitions",
        out_dir / "new_fig_final_best_nmi_heatmap.png",
        label_map=run_labels,
    )

    if not pd.DataFrame(within_pair_rows).empty:
        # A matrix for each run, preserving iteration numbers on both axes.
        for obj in run_objects:
            run = obj["run"]
            labels_by_iter = obj["labels_by_iter"]
            its = sorted(labels_by_iter)
            if len(its) < 2:
                continue
            mat = pd.DataFrame(index=its, columns=its, dtype=float)
            for ia in its:
                for ib in its:
                    mat.loc[ia, ib] = adjusted_rand_score(labels_by_iter[ia], labels_by_iter[ib])
            plot_heatmap(
                mat, f"Within-run iteration ARI: Run {run.get('run_index')} / Leiden {run.get('leiden_seed')}",
                out_dir / f"new_fig_within_run_iteration_ari_run_{int(run.get('run_index', 0)):02d}.png"
            )

    readme = {
        "experiment_directory": str(exp_dir),
        "analysis_output_directory": str(out_dir),
        "n_successful_runs": len(successful),
        "n_failed_runs": int(summary.get("n_failed", len(summary.get("failed_runs", [])))),
        "n_cells": n_cells,
        "consensus_k": args.k,
        "llm_seed_configured_in_experiment": summary.get("llm_seed"),
        "interpretation": [
            "Consensus k is a chosen analysis setting, not ground truth.",
            "ARI/NMI measure partition agreement, not biological accuracy.",
            "Iteration positions are adaptive and need not represent equivalent decisions across runs.",
            "Final-best comparisons use the selected partition saved for each run.",
            "A configured LLM seed is best-effort reproducibility, not a guarantee of identical outputs.",
        ],
    }
    (out_dir / "new_analysis_readme.json").write_text(json.dumps(readme, indent=2), encoding="utf-8")
    print(f"Analysis complete: {out_dir}")
    print(f"Runs analyzed: {len(successful)}; cells: {n_cells}; consensus k: {args.k}")
    print("Key outputs:")
    for name in [
        "new_iteration_trajectory_table.csv",
        "new_final_best_selection_table.csv",
        "new_within_run_trajectory_consensus_summary.csv",
        "new_cross_run_same_iteration_position_similarity.csv",
        "new_final_best_pairwise_similarity.csv",
        "new_fig_parameter_trajectories.png",
        "new_fig_metric_trajectories.png",
        "new_fig_final_best_ari_heatmap.png",
        "new_fig_final_best_nmi_heatmap.png",
    ]:
        print(" -", out_dir / name)


if __name__ == "__main__":
    main()
