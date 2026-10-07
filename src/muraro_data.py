from pathlib import Path
import urllib.request

import anndata as ad
import pandas as pd
import scanpy as sc


MURARO_GEO_ACCESSION = "GSE85241"

MURARO_GEO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE85nnn/"
    "GSE85241/suppl/"
    "GSE85241_cellsystems_dataset_4donors_updated.csv.gz"
)

MURARO_REFERENCE_SOURCE = (
    "Bioconductor scRNAseq::MuraroPancreasData()"
)

MURARO_SCRNASEQ_VERSION = "2.18.0"


def download_muraro_expression(
    output_dir="data/muraro"
):
    """Download the Muraro expression matrix from GEO."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        output_dir
        / "GSE85241_cellsystems_dataset_4donors_updated.csv.gz"
    )

    if not output_file.exists():
        urllib.request.urlretrieve(
            MURARO_GEO_URL,
            output_file
        )

    return output_file


def create_muraro_h5ad(
    input_file,
    output_file="data/muraro/muraro_raw.h5ad"
):
    """Convert the Muraro GEO expression matrix to AnnData."""

    muraro_df = pd.read_csv(
        input_file,
        sep="\t",
        compression="gzip",
        index_col=0
    )

    muraro = ad.AnnData(
        X=muraro_df.T
    )

    muraro.obs_names = muraro_df.columns
    muraro.var_names = muraro_df.index

    muraro.write_h5ad(output_file)

    return muraro


def load_muraro_expression(
    path="data/muraro/muraro_raw.h5ad"
):
    """Load the locally saved Muraro AnnData object."""

    return sc.read_h5ad(path)


def load_muraro_annotations(
    path="data/muraro/muraro_cell_annotations.csv"
):
    """Load published Muraro cell-type annotations."""

    return pd.read_csv(path)


def attach_reference_annotations(
    adata,
    annotations,
    column_name="cell_type_reference"
):
    """Attach published reference labels using cell IDs."""

    annotation_map = (
        annotations
        .set_index("cell_id")["label"]
    )

    adata.obs[column_name] = (
        annotation_map
        .reindex(adata.obs_names)
        .values
    )

    return adata


def get_evaluation_subset(
    adata,
    reference_column="cell_type_reference"
):
    """Keep cells with usable published reference labels."""

    mask = (
        adata.obs[reference_column].notna()
        & (adata.obs[reference_column] != "unclear")
    )

    return adata[mask].copy()