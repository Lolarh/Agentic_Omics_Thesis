import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
import scanpy as sc


def get_cluster_summary(adata):
    """
    Return the number of cells in each Leiden cluster
    as a regular dictionary.
    """
    return (adata.obs["leiden"].value_counts().sort_index().to_dict())


def get_cluster_size_summary(adata):
    """
    Calculate summary statistics describing the cluster-size distribution.
    """

    cluster_sizes = get_cluster_summary(adata)

    total_cells = sum(cluster_sizes.values())
    largest_cluster_size = max(cluster_sizes.values())

    small_clusters = {cluster: size for cluster, size in cluster_sizes.items() if size < 50}

    return {
        "largest_cluster_size": largest_cluster_size,
        "largest_cluster_fraction": largest_cluster_size / total_cells,
        "number_of_small_clusters": len(small_clusters),
        "small_clusters": small_clusters,}


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
        adata.obs["leiden"],)


def calculate_davies_bouldin_score(adata, embedding="X_pca"):
    """
    Calculate the Davies-Bouldin Index using the specified embedding.
    """
    return davies_bouldin_score(
        adata.obsm[embedding],
        adata.obs["leiden"],)


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
        squared_distances = np.sum((cluster_points - centroid) ** 2,
                                   axis=1,)

        # Add within-cluster dispersion
        total_dispersion += np.sum(squared_distances)

    return float(total_dispersion)

def calculate_banfield_raftery_score(adata, embedding="X_pca",):
    """
    Calculate the Banfield-Raftery clustering criterion. The score is based on within-cluster dispersion.
    
    Lower values generally indicate more compact clusters, but the metric should be interpreted alongside the number
    of clusters and other clustering quality metrics.

    Parameters
    ----------
    adata : AnnData
    AnnData object containing cluster labels.

    embedding : str
    Name of the embedding stored in ``adata.obsm``.

    Returns
    -------
    float
    Banfield-Raftery criterion value.
    """

    X = adata.obsm[embedding]
    labels = adata.obs["leiden"].to_numpy()

    total_score = 0.0

    for cluster in np.unique(labels):

        cluster_points = X[labels == cluster]

        # Number of cells in the cluster
        n_k = len(cluster_points)

        # Skip clusters with fewer than two cells
        if n_k < 2:
            continue

        # Calculate cluster centroid
        centroid = cluster_points.mean(axis=0)
        
        # Within-cluster sum of squares
        w_k = np.sum((cluster_points - centroid) ** 2)

        # Avoid log(0)
        w_k = max(w_k, np.finfo(float).eps)

        total_score += (n_k * np.log(w_k / n_k))
        
    return float(total_score)


def get_cluster_marker_genes(
    adata,
    groupby="leiden",
    n_genes=10,):
    """
    Get the top marker genes for each cluster.
    """

    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        method="wilcoxon",
        n_genes=n_genes,)

    marker_genes = {}

    groups = adata.obs[groupby].cat.categories

    for cluster in groups:
        genes = sc.get.rank_genes_groups_df(adata, group=cluster,)["names"].head(n_genes).tolist()

        marker_genes[str(cluster)] = genes

    return marker_genes

    
def evaluate_clustering(
    adata,
    include_markers=False,):
    """
    Evaluate one Leiden clustering result.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing Leiden cluster labels
        and clustering parameters.

    include_markers : bool, default=False
        Whether to calculate the top marker genes for each cluster.
        Marker genes are calculated after clustering and are used as
        biological evidence only in the biology-informed condition.

    Returns
    -------
    dict
        Clustering parameters and evaluation metrics.
    """

    params = adata.uns.get(
        "clustering_params",
        {},
    )

    cluster_size_summary = get_cluster_size_summary(
        adata
    )

    if include_markers:
        marker_genes = get_cluster_marker_genes(
            adata
        )
    else:
        marker_genes = None

    results = {
        **params,

        "n_clusters":
            get_number_of_clusters(adata),

        "cluster_sizes":
            get_cluster_summary(adata),

        "largest_cluster_size":
            cluster_size_summary["largest_cluster_size"],

        "largest_cluster_fraction":
            cluster_size_summary["largest_cluster_fraction"],

        "number_of_small_clusters":
            cluster_size_summary["number_of_small_clusters"],

        "small_clusters":
            cluster_size_summary["small_clusters"],

        "silhouette_score":
            calculate_silhouette_score(adata),

        "davies_bouldin_score":
            calculate_davies_bouldin_score(adata),

        "wc_dispersion_score":
            calculate_wc_dispersion_score(adata),

        "banfield_raftery_score":
            calculate_banfield_raftery_score(adata),
    }

    if include_markers:
        results["marker_genes"] = marker_genes

    return results
    
