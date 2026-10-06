import scanpy as sc
import scipy.sparse as sp


def basic_qc(adata):
    """
    Perform basic quality control for the Muraro dataset.

    Muraro does not contain mitochondrial genes, so mitochondrial
    percentage is not used as a QC criterion.

    Low-complexity cells are filtered based on the number of
    detected genes.
    """

    # Ensure unique gene names
    adata.var_names_make_unique()

    # Filter genes expressed in too few cells
    sc.pp.filter_genes(
        adata,
        min_cells=3,
    )

    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(
        adata,
        inplace=True,
    )

    # Remove very low-complexity cells
    adata = adata[
        adata.obs.n_genes_by_counts > 200
    ].copy()  # No n_genes_by_counts < 2500 as in PBMC3k because Muraro median is already ~4,510 genes/cell

    return adata


def normalize_and_scale(adata):
    """
    Normalize expression, identify highly variable genes,
    regress out technical effects, and scale the data.

    Raw expression is preserved in the 'counts' layer.
    """

    # Store original expression
    adata.layers["counts"] = adata.X.copy()

    # Normalize total counts per cell
    sc.pp.normalize_total(
        adata,
        target_sum=1e4,
    )

    # Log-transform
    sc.pp.log1p(adata)

    # Identify highly variable genes
    sc.pp.highly_variable_genes(
    adata,
    n_top_genes=2000,
    flavor="cell_ranger",)

    # Keep highly variable genes
    adata = adata[:,adata.var.highly_variable].copy()

    # Create scaled layer
    if sp.issparse(adata.X):
        adata.layers["scaled"] = adata.X.toarray()
    else:
        adata.layers["scaled"] = adata.X.copy()

    # Regress technical effects
    sc.pp.regress_out(
        adata,
        keys=["total_counts"], # there are no mitochondrial genes in this dataset.
        layer="scaled",
    )

    # Scale
    sc.pp.scale(
        adata,
        max_value=10,
        layer="scaled",
    )

    return adata


def run_pca(
    adata,
    n_comps=50,
    layer="scaled",
    svd_solver="arpack",
):
    """
    Perform PCA using the scaled expression layer.
    """

    sc.pp.pca(
        adata,
        n_comps=n_comps,
        layer=layer,
        svd_solver=svd_solver,
    )

    return adata


def run_preprocessing_pipeline(adata):
    """
    Run the complete Muraro preprocessing pipeline.
    """

    print("Running Muraro QC...")
    adata = basic_qc(adata)

    print("Running normalization...")
    adata = normalize_and_scale(adata)

    print("Running PCA...")
    adata = run_pca(adata)

    print("Finished Muraro preprocessing.")

    return adata