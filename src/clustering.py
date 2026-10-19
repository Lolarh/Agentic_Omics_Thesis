import scanpy as sc


def run_clustering(
    adata,
    n_pcs=30,
    n_neighbors=15,
    resolution=1.0,
    random_state=42,
):
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
        Random seed used for Leiden clustering.
    """

    # Validate parameter types and values
    if not isinstance(n_pcs, int) or isinstance(n_pcs, bool) or n_pcs < 1:
        raise ValueError(f"n_pcs must be a positive integer; got {n_pcs!r}.")

    if not isinstance(n_neighbors, int) or isinstance(n_neighbors, bool) or n_neighbors < 1:
        raise ValueError(
            f"n_neighbors must be a positive integer; got {n_neighbors!r}."
        )

    if not isinstance(resolution, (int, float)) or isinstance(resolution, bool) or resolution <= 0:
        raise ValueError(
            f"resolution must be a positive number; got {resolution!r}."
        )

    if not isinstance(random_state, int) or isinstance(random_state, bool):
        raise TypeError("random_state must be an integer.")

    # Construct neighborhood graph
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
    )

    # Perform Leiden clustering with a controlled seed
    sc.tl.leiden(
        adata,
        resolution=resolution,
        random_state=random_state,
        flavor="igraph",
        directed=False,
        n_iterations=2,
    )

    # removed UMAP computation to avoid additional runtime
    # Record parameters used
    adata.uns["clustering_params"] = {
        "n_neighbors": n_neighbors,
        "resolution": resolution,
        "n_pcs": n_pcs,
        "random_state": random_state,
    }

    return adata
