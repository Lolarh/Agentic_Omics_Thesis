import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
import scanpy as sc


def get_cluster_summary(adata):
    """Return the number of cells in each Leiden cluster as a dictionary."""
    return adata.obs["leiden"].value_counts().sort_index().to_dict()


def get_cluster_size_summary(adata):
    """Calculate summary statistics describing the cluster-size distribution."""
    cluster_sizes = get_cluster_summary(adata)
    total_cells = sum(cluster_sizes.values())
    largest_cluster_size = max(cluster_sizes.values())
    small_clusters = {
        cluster: size for cluster, size in cluster_sizes.items() if size < 50
    }
    return {
        "largest_cluster_size": largest_cluster_size,
        "largest_cluster_fraction": largest_cluster_size / total_cells,
        "number_of_small_clusters": len(small_clusters),
        "small_clusters": small_clusters,
    }


def get_number_of_clusters(adata):
    """Return the total number of Leiden clusters."""
    return adata.obs["leiden"].nunique()


def _get_evaluation_embedding(adata, embedding="X_pca", n_pcs=None):
    """Return the requested embedding, optionally restricted to its first n_pcs PCs.

    If n_pcs is omitted for X_pca, use the candidate's recorded clustering
    parameter. If that parameter is unavailable, retain the full embedding for
    backwards compatibility. Non-PCA embeddings are returned unchanged.
    """
    X = adata.obsm[embedding]

    if embedding != "X_pca":
        return X

    if n_pcs is None:
        n_pcs = adata.uns.get("clustering_params", {}).get("n_pcs")

    if n_pcs is None:
        return X

    if not isinstance(n_pcs, (int, np.integer)) or isinstance(n_pcs, (bool, np.bool_)):
        raise ValueError(f"n_pcs must be an integer; got {n_pcs!r}.")
    if n_pcs < 1 or n_pcs > X.shape[1]:
        raise ValueError(
            f"n_pcs must be between 1 and the available number of PCs ({X.shape[1]}); got {n_pcs}."
        )

    return X[:, :n_pcs]


def calculate_silhouette_score(adata, embedding="X_pca", n_pcs=None):
    """Calculate the Silhouette Score in the candidate's PCA space by default."""
    X = _get_evaluation_embedding(adata, embedding=embedding, n_pcs=n_pcs)
    return silhouette_score(X, adata.obs["leiden"])


def calculate_davies_bouldin_score(adata, embedding="X_pca", n_pcs=None):
    """Calculate the Davies-Bouldin Index in the candidate's PCA space by default."""
    X = _get_evaluation_embedding(adata, embedding=embedding, n_pcs=n_pcs)
    return davies_bouldin_score(X, adata.obs["leiden"])


def calculate_wc_dispersion_score(adata, embedding="X_pca", n_pcs=None):
    """Calculate total within-cluster squared dispersion in the selected embedding.

    Lower values indicate less within-cluster dispersion, but values depend on
    the dimensionality and scale of the embedding used.
    """
    X = _get_evaluation_embedding(adata, embedding=embedding, n_pcs=n_pcs)
    labels = adata.obs["leiden"].to_numpy()
    total_dispersion = 0.0

    for cluster in np.unique(labels):
        cluster_points = X[labels == cluster]
        centroid = cluster_points.mean(axis=0)
        squared_distances = np.sum((cluster_points - centroid) ** 2, axis=1)
        total_dispersion += np.sum(squared_distances)

    return float(total_dispersion)


def get_cluster_marker_genes(adata, groupby="leiden", n_genes=5):
    """Get the top marker genes for each cluster."""
    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        method="wilcoxon",
        n_genes=n_genes,
    )

    marker_genes = {}
    groups = adata.obs[groupby].cat.categories
    for cluster in groups:
        genes = (
            sc.get.rank_genes_groups_df(adata, group=cluster)["names"]
            .head(n_genes)
            .tolist()
        )
        marker_genes[str(cluster)] = genes

    return marker_genes


def evaluate_clustering(adata, include_markers=False):
    """Evaluate one Leiden clustering result using its recorded n_pcs.

    Silhouette, Davies-Bouldin, and within-cluster dispersion are computed on
    X_pca[:, :n_pcs], where n_pcs is read from adata.uns['clustering_params'].
    If n_pcs is absent, these functions fall back to the full X_pca embedding.
    Scores from candidates using different numbers of PCs should be interpreted
    cautiously because the evaluation spaces differ in dimensionality.
    """
    params = adata.uns.get("clustering_params", {})
    n_pcs = params.get("n_pcs")
    cluster_size_summary = get_cluster_size_summary(adata)

    if include_markers:
        marker_genes = get_cluster_marker_genes(adata)
    else:
        marker_genes = None

    results = {
        **params,
        "n_clusters": get_number_of_clusters(adata),
        "cluster_sizes": get_cluster_summary(adata),
        "largest_cluster_size": cluster_size_summary["largest_cluster_size"],
        "largest_cluster_fraction": cluster_size_summary["largest_cluster_fraction"],
        "number_of_small_clusters": cluster_size_summary["number_of_small_clusters"],
        "small_clusters": cluster_size_summary["small_clusters"],
        "metric_embedding": "X_pca",
        "metric_n_pcs": n_pcs if n_pcs is not None else adata.obsm["X_pca"].shape[1],
        "silhouette_score": calculate_silhouette_score(adata, n_pcs=n_pcs),
        "davies_bouldin_score": calculate_davies_bouldin_score(adata, n_pcs=n_pcs),
        "wc_dispersion_score": calculate_wc_dispersion_score(adata, n_pcs=n_pcs),
    }

    if include_markers:
        results["marker_genes"] = marker_genes

    return results
