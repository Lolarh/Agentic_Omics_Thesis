import scanpy as sc
from sklearn.metrics import silhouette_score, davies_bouldin_score

def get_cluster_summary(adata):
    """
    Return the number of cells in each Leiden cluster in regular dictionary format not the usaual Pandas series.
    """
    return (
        adata.obs["leiden"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    # return adata.obs["leiden"].value_counts().sort_index()


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

def evaluate_clustering(adata):
    """
    Evaluate Leiden clustering results.
    
    Parameters
    ----------
    adata : AnnData
    AnnData object containing Leiden cluster labels and clustering parameters.

    Returns
    -------
    dict
    Clustering parameters and evaluation metrics.
    
    """

    params = adata.uns.get("clustering_params", {})
    return {
        **params,
        "n_clusters": get_number_of_clusters(adata),
        "cluster_sizes": get_cluster_summary(adata),
        "silhouette_score": calculate_silhouette_score(adata),
        "davies_bouldin_score": calculate_davies_bouldin_score(adata),}
    

