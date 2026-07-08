import scanpy as sc

def find_marker_genes(
    adata,
    groupby="leiden",
    method="wilcoxon",
):
    """
    Identify marker genes for each cluster.

    Parameters
    ----------
    adata : AnnData
        Clustered AnnData object.
    groupby : str, default="leiden"
        Observation column containing cluster labels.
    method : str, default="wilcoxon"
        Differential expression test.

    Returns
    -------
    AnnData
        AnnData object containing marker gene results.
    """

    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        method=method,
    )

    return adata


def plot_marker_genes(
    adata,
    n_genes=20,
):
    """
    Plot the top marker genes for each cluster.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing marker gene results.
    n_genes : int, default=20
        Number of marker genes to display.
    """

    sc.pl.rank_genes_groups(
        adata,
        n_genes=n_genes,
        sharey=False,
    )


def plot_marker_heatmap(
    adata,
    n_genes=10,
):
    """
    Display a heatmap of the top marker genes.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing marker gene results.
    n_genes : int, default=10
        Number of marker genes to display.
    """

    sc.pl.rank_genes_groups_heatmap(
        adata,
        n_genes=n_genes,
    )


def plot_marker_dotplot(
    adata,
    n_genes=5,
):
    """
    Display a dot plot of the top marker genes.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing marker gene results.
    n_genes : int, default=5
        Number of marker genes to display.
    """

    sc.pl.rank_genes_groups_dotplot(
        adata,
        n_genes=n_genes,
    )