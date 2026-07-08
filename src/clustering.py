import scanpy as sc

def run_clustering(
    adata,
    n_pcs=20,
    n_neighbors=15,
    resolution=1.0,
    random_state=42,
):
    """
    Construct the neighborhood graph and perform Leiden clustering.
    """

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
    )

    sc.tl.leiden(
        adata,
        resolution=resolution,
        random_state=random_state,
        flavor="igraph",
        directed=False,
        n_iterations=2,
    )

    # Compute UMAP coordinates
    sc.tl.umap(adata)

    # Store clustering parameters for reproducibility
    adata.uns["clustering_params"] = {
        "n_neighbors": n_neighbors,
        "resolution": resolution,
        "n_pcs": n_pcs,
        "random_state": random_state,
    }

    return adata