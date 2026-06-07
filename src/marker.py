import scanpy as sc

def compute_markers(adata):
    sc.tl.rank_genes_groups(
        adata,
        groupby="leiden",
        method="wilcoxon"
    )

    return adata