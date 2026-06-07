import scanpy as sc

def load_pbmc3k():
    adata = sc.datasets.pbmc3k()
    return adata