import scanpy as sc

def basic_qc(adata):
    # filter cells with very low gene counts
    sc.pp.filter_cells(adata, min_genes=200)

    # filter genes expressed in few cells that is, if the genes are only present in 1-3 cells (?)
    sc.pp.filter_genes(adata, min_cells=3)

    # mitochondrial gene percentage
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    # Other filtering using QC metrics
    adata = adata[(adata.obs.n_genes_by_counts < 2500) & (adata.obs.n_genes_by_counts > 200) & (adata.obs.pct_counts_mt < 5), :,].copy()
  

    return adata