#!/usr/bin/env python3
"""
Leiden-seed reproducibility experiment using the exact project preprocessing.

This script:
1. Loads PBMC3K via src.data_load.load_pbmc3k().
2. Runs src.preprocessing_pipeline.run_preprocessing_pipeline().
3. Builds ONE neighborhood graph using the selected Run 8 reference settings.
4. Runs Leiden on that same graph with different random seeds.
5. Saves per-seed labels, cluster sizes, internal metrics, pairwise ARI/NMI,
   plots, and metadata.

Reference configuration (Run 8):
    n_neighbors = 40
    n_pcs        = 20
    resolution   = 0.85

Default Leiden seeds (10): 1 2 3 6 23 45 7 11 17 29

Run from the project root:
    python 02_01_clean_leiden_seed_reproducibility.py

Optional:
    python 02_01_clean_leiden_seed_reproducibility.py --seeds 1 2 3 6 23 45 7 11 17 29

Silhouette and Davies–Bouldin are calculated using the same leading PCs as
the graph configuration (the value of --n-pcs). This keeps the representation
consistent with the selected configuration.

This tests Leiden-seed reproducibility conditional on the preprocessed data and
fixed graph. It does not rerun the LLM agent. PCA is recomputed by the project's
existing preprocessing pipeline; small numerical differences across software
environments may be possible, so use the same environment as the agent runs.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
)

from src import data_load as dl
from src import preprocessing_pipeline as prp


DEFAULT_SEEDS = [1, 2, 3, 6, 23, 45, 7, 11, 17, 29]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--n-neighbors", type=int, default=40)
    parser.add_argument("--n-pcs", type=int, default=20)
    parser.add_argument("--resolution", type=float, default=0.85)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Default: timestamped folder under experiments/.",
    )
    return parser.parse_args()


def save_heatmap(matrix: pd.DataFrame, title: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(matrix.columns)), labels=matrix.columns)
    ax.set_yticks(range(len(matrix.index)), labels=matrix.index)
    ax.set_xlabel("Leiden seed")
    ax.set_ylabel("Leiden seed")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            ax.text(
                j, i, f"{value:.2f}", ha="center", va="center",
                color="white" if value < 0.55 else "black", fontsize=9
            )
    fig.colorbar(im, ax=ax, label="Agreement (0–1)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Provide at least two unique Leiden seeds.")
    if args.n_neighbors < 2 or args.n_pcs < 2:
        raise ValueError("Invalid n_neighbors/n_pcs/metric-pcs values.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (Path("experiments") / f"leiden_seed_reproducibility_{timestamp}").resolve()
    )
    labels_dir = output_dir / "labels"
    figures_dir = output_dir / "figures"
    labels_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading PBMC3K using src.data_load.load_pbmc3k()...")
    adata = dl.load_pbmc3k()

    print("Running the same preprocessing pipeline as the agent experiment...")
    adata = prp.run_preprocessing_pipeline(adata)

    if "X_pca" not in adata.obsm:
        raise RuntimeError("Preprocessing did not create adata.obsm['X_pca'].")

    X_pca = np.asarray(adata.obsm["X_pca"])
    if X_pca.ndim != 2 or X_pca.shape[0] != adata.n_obs:
        raise ValueError(f"Unexpected PCA shape: {X_pca.shape}")
    if X_pca.shape[1] < args.n_pcs:
        raise ValueError(
            f"Preprocessing produced {X_pca.shape[1]} PCs, but the selected "
            f"configuration requires {args.n_pcs}."
        )
    if adata.n_obs <= args.n_neighbors:
        raise ValueError("n_neighbors must be smaller than the number of cells.")

    cell_ids = adata.obs_names.astype(str).tolist()
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("Preprocessed AnnData contains duplicate cell IDs.")

    print(
        f"Preprocessed data: {adata.n_obs:,} cells x {adata.n_vars:,} genes; "
        f"PCA shape={X_pca.shape}"
    )
    print(
        f"Fixed configuration: n_neighbors={args.n_neighbors}, "
        f"n_pcs={args.n_pcs}, resolution={args.resolution}"
    )
    print(f"Leiden seeds: {args.seeds}")

    # Construct the neighborhood graph exactly once and reuse it for every seed.
    neighbors_key = "fixed_leiden_seed_graph"
    sc.pp.neighbors(
        adata,
        n_neighbors=args.n_neighbors,
        n_pcs=args.n_pcs,
        use_rep="X_pca",
        random_state=42,
        key_added=neighbors_key,
    )
    connectivities_key = adata.uns[neighbors_key]["connectivities_key"]

    labels_by_seed = {}
    summary_rows, size_rows = [], []
    X_metrics = X_pca[:, :args.n_pcs]

    for seed in args.seeds:
        cluster_key = f"leiden_seed_{seed}"
        print(f"Running Leiden with random_state={seed}...")
        sc.tl.leiden(adata,
                     resolution=args.resolution,
                     random_state=seed,
                     key_added=cluster_key,
                     neighbors_key=neighbors_key,
                     flavor="leidenalg",)
        labels = adata.obs[cluster_key].astype(str).to_numpy()
        labels_by_seed[seed] = labels
        label_path = labels_dir / f"seed_{seed}_labels.npy"
        np.save(label_path, labels)

        counts = pd.Series(labels).value_counts().sort_values(ascending=False)
        small = counts[counts < 30]
        n_clusters = int(len(counts))

        if 1 < n_clusters < adata.n_obs:
            silhouette = float(silhouette_score(X_metrics, labels))
            dbi = float(davies_bouldin_score(X_metrics, labels))
        else:
            silhouette, dbi = float("nan"), float("nan")

        summary_rows.append({
            "seed": seed,
            "n_cells": int(adata.n_obs),
            "n_clusters": n_clusters,
            "silhouette_score": silhouette,
            "davies_bouldin_index": dbi,
            "largest_cluster_cells": int(counts.iloc[0]),
            "largest_cluster_fraction_pct": 100 * counts.iloc[0] / adata.n_obs,
            "clusters_under_30_cells": int(len(small)),
            "cells_in_clusters_under_30": int(small.sum()),
            "labels_file": str(label_path.relative_to(output_dir)),
        })

        for rank, (cluster_id, count) in enumerate(counts.items(), start=1):
            size_rows.append({
                "seed": seed,
                "cluster_size_rank": rank,
                "cluster_id": str(cluster_id),
                "n_cells": int(count),
                "percentage_of_cells": 100 * count / adata.n_obs,
            })

    seed_names = [str(seed) for seed in args.seeds]
    ari = pd.DataFrame(index=seed_names, columns=seed_names, dtype=float)
    nmi = pd.DataFrame(index=seed_names, columns=seed_names, dtype=float)
    for seed_a in args.seeds:
        for seed_b in args.seeds:
            ari.loc[str(seed_a), str(seed_b)] = adjusted_rand_score(
                labels_by_seed[seed_a], labels_by_seed[seed_b]
            )
            nmi.loc[str(seed_a), str(seed_b)] = normalized_mutual_info_score(
                labels_by_seed[seed_a], labels_by_seed[seed_b]
            )

    summary_df = pd.DataFrame(summary_rows)
    sizes_df = pd.DataFrame(size_rows)
    summary_df.to_csv(output_dir / "leiden_seed_summary.csv", index=False)
    sizes_df.to_csv(output_dir / "cluster_sizes.csv", index=False)
    ari.to_csv(output_dir / "pairwise_ari.csv")
    nmi.to_csv(output_dir / "pairwise_nmi.csv")

    # Compare ranked sizes; numeric cluster IDs need not align across seeds.
    size_pivot = sizes_df.pivot(
        index="cluster_size_rank", columns="seed", values="n_cells"
    )
    ax = size_pivot.plot(kind="bar", figsize=(11, 5), width=0.82)
    ax.set_title("Cluster-size distributions across Leiden seeds")
    ax.set_xlabel("Cluster size rank (1 = largest)")
    ax.set_ylabel("Number of cells")
    ax.legend(title="Leiden seed")
    ax.tick_params(axis="x", rotation=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(figures_dir / "cluster_sizes_by_seed.png", dpi=200, bbox_inches="tight")
    plt.close()

    save_heatmap(ari, "Pairwise ARI across Leiden seeds", figures_dir / "pairwise_ari_heatmap.png")
    save_heatmap(nmi, "Pairwise NMI across Leiden seeds", figures_dir / "pairwise_nmi_heatmap.png")

    metadata = {
        "experiment": "Leiden-seed reproducibility with fixed graph",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "dataset_loader": "src.data_load.load_pbmc3k",
        "preprocessing_pipeline": "src.preprocessing_pipeline.run_preprocessing_pipeline",
        "n_cells": int(adata.n_obs),
        "n_genes_after_preprocessing": int(adata.n_vars),
        "cell_ids_in_order": cell_ids,
        "pca_shape": list(X_pca.shape),
        "parameters": {
            "n_neighbors": args.n_neighbors,
            "n_pcs_for_graph": args.n_pcs,
            "resolution": args.resolution,
            "metric_pcs": args.n_pcs,
            "leiden_seeds": args.seeds,
            "graph_constructed_once": True,
            "graph_random_state": 42,
        },
        "graph_keys": {
            "neighbors_key": neighbors_key,
            "connectivities_key": connectivities_key,
        },
        "software": {
            "python": platform.python_version(),
            "scanpy": sc.__version__,
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "notes": [
            "Uses the same project loader and preprocessing pipeline as the agent experiment.",
            "The neighborhood graph is constructed once and reused for all Leiden seeds.",
            "Only the random_state passed to sc.tl.leiden varies between runs.",
            f"Silhouette and DBI use the same first {args.n_pcs} PCs used by the fixed graph configuration.",
            "Cluster-size ranks are compared; numeric cluster IDs are not assumed to align.",
            "This tests Leiden-seed reproducibility conditional on this graph/configuration, not full agent repeatability.",
            "PCA is recomputed by the project preprocessing pipeline in this run; use the same software environment as the agent experiment.",
        ],
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"\nExperiment complete. Outputs saved to:\n{output_dir}")
    print("\nPer-seed summary:")
    print(summary_df.round(4).to_string(index=False))
    print("\nPairwise ARI:")
    print(ari.round(3).to_string())
    print("\nPairwise NMI:")
    print(nmi.round(3).to_string())


if __name__ == "__main__":
    main()
