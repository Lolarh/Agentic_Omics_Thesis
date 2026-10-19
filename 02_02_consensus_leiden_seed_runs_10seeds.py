"""
Consensus from ten saved Leiden random-seed partitions.

Run from project root:
    python consensus_leiden_seed_runs.py

This script does not rerun the agent or Leiden. It loads the saved labels,
builds a cell co-association matrix, and creates a k=9 consensus with
average-linkage agglomerative clustering on distance = 1 - co-association.
"""
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


PROJECT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = PROJECT_DIR / "experiments" / "leiden_seed_reproducibility_20261019_181654" 
SEEDS = [1, 2, 3, 6, 23, 45, 7, 11, 17, 29]
CONSENSUS_K = 9
HEATMAP_SUBSET_SIZE = 250
HEATMAP_RANDOM_SEED = 42
MAKE_UMAP = True


def load_partitions():
    if not EXPERIMENT_DIR.exists():
        raise FileNotFoundError(
            f"Input experiment directory not found: {EXPERIMENT_DIR}\n"
            "Update EXPERIMENT_DIR near the top of this script."
        )
    summary_path = EXPERIMENT_DIR / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    summary = json.loads(summary_path.read_text())
    if "cell_ids_in_order" not in summary:
        raise KeyError("run_summary.json lacks cell_ids_in_order.")
    cell_ids = np.asarray(summary["cell_ids_in_order"], dtype=str)
    if len(np.unique(cell_ids)) != len(cell_ids):
        raise ValueError("Duplicate cell IDs found in run_summary.json.")

    labels_by_seed = {}
    for seed in SEEDS:
        path = EXPERIMENT_DIR / "labels" / f"seed_{seed}_labels.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing labels for seed {seed}: {path}")
        labels = np.load(path, allow_pickle=True).astype(str)
        if len(labels) != len(cell_ids):
            raise ValueError(f"Seed {seed}: {len(labels)} labels vs {len(cell_ids)} cell IDs.")
        labels_by_seed[seed] = labels

    print(f"Loaded {len(SEEDS)} saved partitions for {len(cell_ids)} cells.")
    return summary, cell_ids, labels_by_seed


def build_consensus(labels_by_seed):
    seeds = list(labels_by_seed)
    label_matrix = np.vstack([labels_by_seed[s] for s in seeds])
    n_runs, n_cells = label_matrix.shape
    coassoc = np.zeros((n_cells, n_cells), dtype=np.float32)
    for labels in label_matrix:
        coassoc += (labels[:, None] == labels[None, :]).astype(np.float32)
    coassoc /= n_runs

    distance = np.clip(1.0 - coassoc, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    try:
        model = AgglomerativeClustering(
            n_clusters=CONSENSUS_K, metric="precomputed", linkage="average"
        )
    except TypeError:  # older scikit-learn compatibility
        model = AgglomerativeClustering(
            n_clusters=CONSENSUS_K, affinity="precomputed", linkage="average"
        )
    consensus = model.fit_predict(distance).astype(str)
    return seeds, label_matrix, coassoc, consensus


def save_tables(out_dir, cell_ids, seeds, label_matrix, consensus):
    pd.DataFrame({
        "cell_id": cell_ids,
        "consensus_cluster": consensus,
    }).to_csv(out_dir / "consensus_labels.csv", index=False)


    sizes = (
        pd.Series(consensus, name="consensus_cluster")
        .value_counts()
        .sort_index()
        .rename_axis("consensus_cluster")
        .reset_index(name="cell_count")
)
    sizes["cell_fraction"] = sizes["cell_count"] / len(cell_ids)
    sizes["cell_percentage"] = 100 * sizes["cell_fraction"]
    sizes.to_csv(out_dir / "consensus_cluster_sizes.csv", index=False)

    pair_rows = []
    for i, j in combinations(range(len(seeds)), 2):
        pair_rows.append({
            "seed_a": seeds[i], "seed_b": seeds[j],
            "ARI": adjusted_rand_score(label_matrix[i], label_matrix[j]),
            "NMI": normalized_mutual_info_score(label_matrix[i], label_matrix[j]),
            "clusters_seed_a": len(np.unique(label_matrix[i])),
            "clusters_seed_b": len(np.unique(label_matrix[j])),
        })
    pairwise = pd.DataFrame(pair_rows)
    pairwise.to_csv(out_dir / "pairwise_seed_similarity.csv", index=False)

    compare = pd.DataFrame([{
        "seed": seed,
        "ARI_vs_consensus": adjusted_rand_score(consensus, labels),
        "NMI_vs_consensus": normalized_mutual_info_score(consensus, labels),
        "n_clusters_in_seed_partition": len(np.unique(labels)),
    } for seed, labels in zip(seeds, label_matrix)])
    compare.to_csv(out_dir / "consensus_vs_seed_partitions.csv", index=False)

    summary = {
        "analysis": f"Co-association consensus from {len(seeds)} saved Leiden seed partitions",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_experiment_dir": str(EXPERIMENT_DIR),
        "n_runs": len(seeds), "seeds": seeds, "n_cells": len(cell_ids),
        "consensus_method": (
            "Cell co-association fraction across runs; distance = 1 - co-association; "
            "average-linkage agglomerative clustering"
        ),
        "consensus_k": CONSENSUS_K,
        "consensus_k_is_analysis_choice_not_ground_truth": True,
        "mean_pairwise_ari": float(pairwise["ARI"].mean()),
        "median_pairwise_ari": float(pairwise["ARI"].median()),
        "sd_pairwise_ari": float(pairwise["ARI"].std(ddof=1)),
        "min_pairwise_ari": float(pairwise["ARI"].min()),
        "max_pairwise_ari": float(pairwise["ARI"].max()),
        "mean_pairwise_nmi": float(pairwise["NMI"].mean()),
        "mean_ari_vs_consensus": float(compare["ARI_vs_consensus"].mean()),
        "mean_nmi_vs_consensus": float(compare["NMI_vs_consensus"].mean()),
        "interpretation_note": (
            f"Consensus is a derived summary of {len(seeds)} partitions, not ground truth. "
            "Agreement with it is descriptive, not independent validation."
        ),
    }
    (out_dir / "consensus_summary.json").write_text(json.dumps(summary, indent=2))
    return pairwise, compare, sizes, summary


def save_figures(out_dir, coassoc, cell_ids, consensus, seeds, label_matrix):
    rng = np.random.default_rng(HEATMAP_RANDOM_SEED)
    n = min(HEATMAP_SUBSET_SIZE, len(cell_ids))
    subset = np.sort(rng.choice(len(cell_ids), size=n, replace=False))
    subset = subset[np.argsort(consensus[subset], kind="stable")]

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        coassoc[np.ix_(subset, subset)], vmin=0, vmax=1, cmap="mako",
        xticklabels=False, yticklabels=False, ax=ax,
        cbar_kws={"label": "Fraction of runs co-clustering each cell pair"},
    )
    ax.set_title(f"Leiden-seed consensus co-association (k={CONSENSUS_K})")
    ax.set_xlabel("Sampled cells, ordered by consensus cluster")
    ax.set_ylabel("Sampled cells, ordered by consensus cluster")
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_coassociation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    ari, nmi = np.eye(len(seeds)), np.eye(len(seeds))
    for i, j in combinations(range(len(seeds)), 2):
        ari[i, j] = ari[j, i] = adjusted_rand_score(label_matrix[i], label_matrix[j])
        nmi[i, j] = nmi[j, i] = normalized_mutual_info_score(label_matrix[i], label_matrix[j])
    for matrix, title, filename in [
        (ari, "Pairwise ARI between Leiden seeds", "fig2_pairwise_ari_heatmap.png"),
        (nmi, "Pairwise NMI between Leiden seeds", "fig3_pairwise_nmi_heatmap.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(matrix, annot=True, fmt=".3f", vmin=0, vmax=1, cmap="mako",
                    xticklabels=seeds, yticklabels=seeds, square=True, ax=ax)
        ax.set_xlabel("Leiden random seed")
        ax.set_ylabel("Leiden random seed")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    compare = pd.read_csv(out_dir / "consensus_vs_seed_partitions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(compare["seed"].astype(str), compare["ARI_vs_consensus"])
    axes[0].set(title="Agreement with consensus", xlabel="Leiden seed", ylabel="ARI", ylim=(0, 1))
    axes[1].bar(compare["seed"].astype(str), compare["NMI_vs_consensus"])
    axes[1].set(title="Agreement with consensus", xlabel="Leiden seed", ylabel="NMI", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_consensus_agreement.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_umap(out_dir, cell_ids, consensus, n_seeds):
    """Visualization only: preprocess PBMC3k and create a shared UMAP."""
    if not MAKE_UMAP:
        return
    try:
        from src import data_load as dl
        from src import preprocessing_pipeline as prp
    except ImportError as exc:
        warnings.warn(f"Project preprocessing modules unavailable; skipping UMAP: {exc}")
        return

    print("Preparing PBMC3k for consensus UMAP visualization (no Leiden rerun)...")
    adata = prp.run_preprocessing_pipeline(dl.load_pbmc3k())
    fresh_ids = adata.obs_names.astype(str).to_numpy()
    if len(fresh_ids) != len(cell_ids) or not np.array_equal(fresh_ids, cell_ids):
        warnings.warn("Cell IDs/order mismatch; skipping UMAP to avoid incorrect labels.")
        return

    adata.obs["consensus_cluster"] = pd.Categorical(consensus)
    n_pcs = min(30, adata.obsm["X_pca"].shape[1])
    sc.pp.neighbors(
        adata, n_neighbors=40, n_pcs=n_pcs, use_rep="X_pca",
        random_state=42, key_added="consensus_visualization_graph",
    )
    sc.tl.umap(adata, neighbors_key="consensus_visualization_graph", random_state=42)
    sc.pl.umap(
        adata, color="consensus_cluster",
        title=f"Consensus partition from {n_seeds} Leiden seeds (k={CONSENSUS_K})",
        legend_loc="on data", frameon=False, show=False,
    )
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    fig.savefig(out_dir / "fig5_consensus_umap.png", dpi=300, bbox_inches="tight")
    plt.close("all")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_DIR / "experiments" / f"leiden_seed_consensus_k{CONSENSUS_K}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    print("Loading saved labels; no agent or Leiden rerun will occur.")
    _, cell_ids, labels_by_seed = load_partitions()
    seeds, label_matrix, coassoc, consensus = build_consensus(labels_by_seed)
    np.save(out_dir / "coassociation_matrix.npy", coassoc)
    np.save(out_dir / "consensus_labels.npy", consensus)

    pairwise, compare, sizes, summary = save_tables(
        out_dir, cell_ids, seeds, label_matrix, consensus
    )
    save_figures(out_dir, coassoc, cell_ids, consensus, seeds, label_matrix)
    save_umap(out_dir, cell_ids, consensus, n_seeds=len(seeds))

    print("\nConsensus cluster sizes:")
    print(sizes.to_string(index=False))
    print("\nPairwise seed similarity summary:")
    print(pairwise[["ARI", "NMI"]].agg(["mean", "median", "std", "min", "max"]).round(4))
    print("\nConsensus agreement by seed:")
    print(compare.round(4).to_string(index=False))
    print("\nSaved outputs to:", out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
