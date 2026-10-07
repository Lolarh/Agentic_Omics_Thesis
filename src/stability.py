import numpy as np
from itertools import combinations
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from src.clustering import run_clustering
from scipy.optimize import linear_sum_assignment


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

def calculate_pairwise_nmi(assignments):
    """
    Calculate pairwise Normalized Mutual Information (NMI)
    between repeated clustering runs.
    """

    nmi_scores = []

    for i, j in combinations(range(len(assignments)), 2):

        nmi = normalized_mutual_info_score(
            assignments[i],
            assignments[j],
        )

        nmi_scores.append(nmi)

    return nmi_scores
    

def calculate_cluster_jaccard(assignments, reference_run=0):
    """
    Calculate cluster-level Jaccard similarity between
    a reference clustering and all other runs.

    Clusters are matched one-to-one using the Hungarian
    assignment algorithm to maximize total Jaccard similarity.
    """

    reference_labels = assignments[reference_run]
    reference_clusters = np.unique(reference_labels)

    results = []

    for run_index, labels in enumerate(assignments):

        if run_index == reference_run:
            continue

        candidate_clusters = np.unique(labels)

        # Jaccard similarity matrix
        jaccard_matrix = np.zeros(
            (len(reference_clusters), len(candidate_clusters))
        )

        for i, reference_cluster in enumerate(reference_clusters):

            reference_cells = set(
                np.where(reference_labels == reference_cluster)[0]
            )

            for j, candidate_cluster in enumerate(candidate_clusters):

                candidate_cells = set(
                    np.where(labels == candidate_cluster)[0]
                )

                intersection = len(
                    reference_cells & candidate_cells
                )

                union = len(
                    reference_cells | candidate_cells
                )

                jaccard_matrix[i, j] = (
                    intersection / union
                    if union > 0 else 0
                )

        # Hungarian algorithm maximizes Jaccard
        row_ind, col_ind = linear_sum_assignment(
            -jaccard_matrix
        )

        for i, j in zip(row_ind, col_ind):

            results.append({
                "run": run_index,
                "reference_cluster": reference_clusters[i],
                "matched_cluster": candidate_clusters[j],
                "jaccard": jaccard_matrix[i, j],
            })

    return results


def calculate_ari_vs_reference(
    assignments,
    reference_index=0
):
    """
    Calculate ARI of each clustering relative to
    a reference clustering.
    """

    reference_labels = assignments[reference_index]

    ari_scores = []

    for labels in assignments:

        ari = adjusted_rand_score(
            reference_labels,
            labels
        )

        ari_scores.append(ari)

    return ari_scores


def calculate_nmi_vs_reference(
    assignments,
    reference_index=0
):
    """
    Calculate NMI of each clustering relative to
    a reference clustering.
    """

    reference_labels = assignments[reference_index]

    nmi_scores = []

    for labels in assignments:

        nmi = normalized_mutual_info_score(
            reference_labels,
            labels
        )

        nmi_scores.append(nmi)

    return nmi_scores