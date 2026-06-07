import scanpy as sc

def run_clustering(adata, n_pcs=20, resolution=1.0):
    sc.pp.neighbors(adata, n_pcs=n_pcs)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=resolution)
    return adata


adata_10 = pbmc_data.copy()
adata_20 = pbmc_data.copy()

run_clustering(adata_10, n_pcs=10)
run_clustering(adata_20, n_pcs=20)

run_clustering(pbmc_data.copy(), n_pcs=10, resolution=0.5)
run_clustering(pbmc_data.copy(), n_pcs=20, resolution=1.0)
run_clustering(pbmc_data.copy(), n_pcs=30, resolution=1.2)

sc.pl.umap(pbmc_data.copy(), n_pcs=10, color="leiden")
sc.pl.umap(pbmc_data.copy(), n_pcs=20, color="leiden")