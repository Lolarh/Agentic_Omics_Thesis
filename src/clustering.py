import scanpy as sc


def run_clustering(
    adata,
    n_pcs=20,
    n_neighbors=15,
    resolution=1.0,
    random_state=42,):
    """
    Construct the neighborhood graph and perform Leiden clustering.

    Parameters
    ----------
    n_pcs : int
        Number of principal components used to construct the
        neighborhood graph. Experimental range: 10–50.

    n_neighbors : int
        Number of nearest neighbors used to construct the
        neighborhood graph. Experimental range: 10–50.

    resolution : float
        Leiden clustering resolution. Experimental range: 0.1–2.0.

    random_state : int
        Random seed used for reproducibility.
    """

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,)

    sc.tl.leiden(
        adata,
        resolution=resolution,
        random_state=random_state,
        flavor="igraph",
        directed=False,
        n_iterations=2,)

    sc.tl.umap(adata)

    adata.uns["clustering_params"] = {
        "n_neighbors": n_neighbors,
        "resolution": resolution,
        "n_pcs": n_pcs,
        "random_state": random_state,}

    return adata