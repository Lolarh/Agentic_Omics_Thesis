import numpy as np
from itertools import combinations
from sklearn.metrics import adjusted_rand_score
from src.clustering import run_clustering


def run_stability_analysis(
    adata,
    n_neighbors,
    resolution,
    n_pcs,
    seeds,):
    """
    Run the same clustering configuration across
    multiple random seeds.

    Returns
    -------
    dict
        Cluster assignments and cluster counts for each run.
    """

    assignments = []
    cluster_counts = []

    for seed in seeds:

        adata_run = adata.copy()

        adata_run = run_clustering(
            adata_run,
            n_neighbors=n_neighbors,
            resolution=resolution,
            n_pcs=n_pcs,
            random_state=seed,)

        labels = adata_run.obs["leiden"].to_numpy()

        assignments.append(labels)

        cluster_counts.append(
            len(np.unique(labels)))

    return {
        "assignments": assignments,
        "cluster_counts": cluster_counts,}


def calculate_pairwise_ari(assignments):
    """
    Calculate pairwise Adjusted Rand Index (ARI)
    between repeated clustering runs.
    """

    ari_scores = []

    for i, j in combinations(range(len(assignments)), 2):

        ari = adjusted_rand_score(
            assignments[i],
            assignments[j],)

        ari_scores.append(ari)

    return ari_scores



def calculate_cluster_jaccard(assignments, reference_run=0):
    """
    Match clusters from other runs to clusters in a reference run
    using Jaccard similarity of cell memberships.
    """

    reference_labels = assignments[reference_run]
    reference_clusters = np.unique(reference_labels)

    results = []

    for run_index, labels in enumerate(assignments):

        if run_index == reference_run:
            continue

        for reference_cluster in reference_clusters:

            reference_cells = set(np.where(reference_labels == reference_cluster)[0])

            best_match = None
            best_jaccard = 0.0

            for candidate_cluster in np.unique(labels):

                candidate_cells = set(np.where(labels == candidate_cluster)[0])

                intersection = len(reference_cells & candidate_cells)

                union = len(reference_cells | candidate_cells)

                jaccard = intersection / union

                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_match = candidate_cluster

            results.append({
                "run": run_index,
                "reference_cluster": reference_cluster,
                "matched_cluster": best_match,
                "jaccard": best_jaccard,})

    return results