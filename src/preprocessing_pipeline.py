import scanpy as sc
import scipy.sparse as sp

def basic_qc(adata):
    """
    Perform quality control filtering on cells and genes.

    This step:
    - Filters low-quality cells and genes
    - Calculates QC metrics
    - Removes cells with excessive mitochondrial expression
    """

    # Ensure unique gene names
    adata.var_names_make_unique()

    # Filter low-quality cells
    sc.pp.filter_cells(adata, min_genes=200)

    # Filter genes expressed in too few cells
    sc.pp.filter_genes(adata, min_cells=3)

    # Identify mitochondrial genes
    adata.var["mt"] = adata.var_names.str.startswith("MT-")

    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        inplace=True,
    )

    # Apply QC filtering
    adata = adata[
        (adata.obs.n_genes_by_counts > 200)
        & (adata.obs.n_genes_by_counts < 2500)
        & (adata.obs.pct_counts_mt < 5)
    ].copy()

    return adata


def normalize_and_scale(adata):
    """
    Normalize gene expression, identify highly variable genes,
    regress out technical effects, and scale the data.

    The original raw counts are preserved in the 'counts' layer,
    while regression and scaling are performed on a separate
    'scaled' layer for downstream PCA.
    """

    # Store raw counts before normalization
    adata.layers["counts"] = adata.X.copy()

    # Normalize total counts per cell
    sc.pp.normalize_total(
        adata,
        target_sum=1e4,
    )

    # Log-transform normalized counts
    sc.pp.log1p(adata)

    # Identify highly variable genes using raw counts
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=2000,
        flavor="seurat_v3",
        layer="counts",
    )

    # Keep only highly variable genes
    adata = adata[:, adata.var.highly_variable].copy()

    # Create a scaled layer
    if sp.issparse(adata.X):
        adata.layers["scaled"] = adata.X.toarray()
    else:
        adata.layers["scaled"] = adata.X.copy()

    # Regress out unwanted technical effects
    sc.pp.regress_out(
        adata,
        keys=["total_counts", "pct_counts_mt"],
        layer="scaled",
    )

    # Scale the data
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
    Perform Principal Component Analysis (PCA) using the scaled layer.
    """

    sc.pp.pca(
        adata,
        n_comps=n_comps,
        layer=layer,
        svd_solver=svd_solver,
    )

    return adata


def run_preprocessing_pipeline(adata):

    print("Running QC...")
    adata = basic_qc(adata)

    print("Running normalization...")
    adata = normalize_and_scale(adata)

    print("Running PCA...")
    adata = run_pca(adata)

    print("Finished preprocessing.")

    return adata