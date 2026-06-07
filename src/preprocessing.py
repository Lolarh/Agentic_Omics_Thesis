import scanpy as sc

def preprocess(adata):

    # store raw counts BEFORE normalization
    adata.layers["counts"] = adata.X.copy()

    # normalize
    sc.pp.normalize_total(adata, target_sum=1e4)

    # log transform
    sc.pp.log1p(adata)

    # HVG selection using raw counts
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=2000,
        flavor="seurat_v3",
        layer="counts",
        min_mean=0.0125, # Filters genes by average expression:remove extremely rare genes
        max_mean=3, # remove extremely highly expressed “boring” genes
        min_disp=0.5, # dispersion” = variability relative to mean; removes genes that don’t vary enough
    )

    # subset HVGs
    adata = adata[:, adata.var.highly_variable]

    # scaling + regression
    adata.layers["scaled"] = adata.X.toarray()
    sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"], layer="scaled") # Removes effects of: total_counts (sequencing depth differences)
    # pct_counts_mt: mitochondrial contamination (cell stress / death signal)
    sc.pp.scale(adata, max_value=10, layer="scaled")

    return adata