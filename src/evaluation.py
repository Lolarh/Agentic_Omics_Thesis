import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score


def get_cluster_summary(adata):
    """
    Return the number of cells in each Leiden cluster
    as a regular dictionary.
    """
    return (
        adata.obs["leiden"]
        .value_counts()
        .sort_index()
        .to_dict()
    )


def get_cluster_size_summary(adata):
    """
    Calculate summary statistics describing the cluster-size distribution.
    """

    cluster_sizes = get_cluster_summary(adata)

    total_cells = sum(cluster_sizes.values())
    largest_cluster_size = max(cluster_sizes.values())

    small_clusters = {
        cluster: size
        for cluster, size in cluster_sizes.items()
        if size < 50
    }

    return {
        "largest_cluster_size": largest_cluster_size,
        "largest_cluster_fraction": largest_cluster_size / total_cells,
        "number_of_small_clusters": len(small_clusters),
        "small_clusters": small_clusters,
    }


def get_number_of_clusters(adata):
    """
    Return the total number of Leiden clusters.
    """
    return adata.obs["leiden"].nunique()


def calculate_silhouette_score(adata, embedding="X_pca"):
    """
    Calculate the Silhouette Score using PCA coordinates.
    """
    return silhouette_score(
        adata.obsm[embedding],
        adata.obs["leiden"],
    )


def calculate_davies_bouldin_score(adata, embedding="X_pca"):
    """
    Calculate the Davies-Bouldin Index using the specified embedding.
    """
    return davies_bouldin_score(
        adata.obsm[embedding],
        adata.obs["leiden"],
    )


def calculate_wc_dispersion_score(adata, embedding="X_pca"):
    """
    Calculate within-cluster dispersion using PCA coordinates.

    Lower values indicate less within-cluster dispersion.
    """
    X = adata.obsm[embedding]
    labels = adata.obs["leiden"].to_numpy()

    total_dispersion = 0.0

    for cluster in np.unique(labels):

        cluster_points = X[labels == cluster]

        # Calculate the centroid of the cluster
        centroid = cluster_points.mean(axis=0)

        # Calculate squared distances from cells to centroid
        squared_distances = np.sum(
            (cluster_points - centroid) ** 2,
            axis=1,
        )

        # Add within-cluster dispersion
        total_dispersion += np.sum(squared_distances)

    return float(total_dispersion)


def has_reasonable_cluster_structure(experiment):
    """
    Check cluster-size diagnostics for obvious structural problems.

    These are warning signals, not biological rejection rules.
    """

    metrics = experiment["metrics"]

    largest_fraction = metrics.get(
        "largest_cluster_fraction",
        0
    )

    number_of_small_clusters = metrics.get(
        "number_of_small_clusters",
        0
    )

    # Conservative structural warnings
    highly_dominant = largest_fraction > 0.70
    excessive_small_clusters = number_of_small_clusters >= 5

    return not (
        highly_dominant or
        excessive_small_clusters
    )


def is_meaningful_improvement(
    current_experiment,
    best_experiment,
    min_silhouette_improvement=0.001,
    min_dbi_improvement=0.01,
    max_silhouette_drop=0.02,
    max_dbi_increase=0.10,
):
    """
    Determine whether the current experiment is better
    than the best experiment found so far.

    Silhouette:
        Higher is better.

    Davies-Bouldin:
        Lower is better.

    Cluster structure:
        Acts as a supporting constraint.
    """

    current = current_experiment["metrics"]
    best = best_experiment["metrics"]

    silhouette_change = (
        current["silhouette_score"]
        - best["silhouette_score"]
    )

    dbi_change = (
        best["davies_bouldin_score"]
        - current["davies_bouldin_score"]
    )

    silhouette_improved = (
        silhouette_change >= min_silhouette_improvement
    )

    dbi_improved = (
        dbi_change >= min_dbi_improvement
    )

    silhouette_not_badly_worse = (
        silhouette_change >= -max_silhouette_drop
    )

    dbi_not_badly_worse = (
        dbi_change >= -max_dbi_increase
    )

    metric_improvement = (
        (silhouette_improved or dbi_improved)
        and silhouette_not_badly_worse
        and dbi_not_badly_worse
    )

    structural_quality = has_reasonable_cluster_structure(
        current_experiment
    )

    return metric_improvement and structural_quality


def get_best_experiment(history):
    """
    Return the best experiment found so far.

    This function is used for final best-parameter selection.
    It does not perform convergence detection.
    """

    if not history:
        return None

    best_experiment = history[0]

    for experiment in history[1:]:

        if is_meaningful_improvement(
            experiment,
            best_experiment,
        ):
            best_experiment = experiment

    return best_experiment


def evaluate_clustering(adata):
    """
    Evaluate one Leiden clustering result.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing Leiden cluster labels
        and clustering parameters.

    Returns
    -------
    dict
        Clustering parameters and evaluation metrics.
    """

    params = adata.uns.get(
        "clustering_params",
        {}
    )

    cluster_size_summary = get_cluster_size_summary(
        adata
    )

    return {
        **params,

        "n_clusters": get_number_of_clusters(
            adata
        ),

        "cluster_sizes": get_cluster_summary(
            adata
        ),

        "largest_cluster_size":
            cluster_size_summary[
                "largest_cluster_size"
            ],

        "largest_cluster_fraction":
            cluster_size_summary[
                "largest_cluster_fraction"
            ],

        "number_of_small_clusters":
            cluster_size_summary[
                "number_of_small_clusters"
            ],

        "small_clusters":
            cluster_size_summary[
                "small_clusters"
            ],

        "silhouette_score":
            calculate_silhouette_score(adata),

        "davies_bouldin_score":
            calculate_davies_bouldin_score(adata),

        "wc_dispersion_score":
            calculate_wc_dispersion_score(adata),
    }