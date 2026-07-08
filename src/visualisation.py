import scanpy as sc
import matplotlib.pyplot as plt

# PCA visualization. 
# This lets you inspect the cells in principal component space before nonlinear embedding.

def plot_pca(
    adata,
    color="leiden",
    title="PCA Projection",
    show=True,
):
    """
    Visualize cells in principal component (PCA) space.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing PCA coordinates.
    color : str, default="leiden"
        Observation or gene used to colour the cells.
    title : str, default="PCA Projection"
        Figure title.
    show : bool, default=True
        Whether to display the figure immediately.

    Returns
    -------
    None
    """

    sc.pl.pca(
        adata,
        color=color,
        title=title,
        show=show,)

# UMAP
# Probably the most important figure in your thesis.

def plot_umap(
    adata,
    color="leiden",):
    """
    Visualize cells using UMAP.
    """

    sc.pl.umap(
        adata,
        color=color,)

# Cluster proportions
# This is something Scanpy doesn't directly provide, but it's useful.

def plot_cluster_sizes(adata):
    """
    Plot the number of cells in each Leiden cluster.
    """

    cluster_sizes = (
        adata.obs["leiden"]
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(8,5))

    cluster_sizes.plot(
        kind="bar"
    )

    plt.xlabel("Cluster")
    plt.ylabel("Number of cells")
    plt.title("Cluster sizes")

    plt.tight_layout()
    plt.show()

# Explaines variance: Shows How many principal components explain meaningful variation.

def plot_pca_variance(adata):
    """
    Plot PCA explained variance ratio.
    """

    sc.pl.pca_variance_ratio(
        adata,
        log=True,)

