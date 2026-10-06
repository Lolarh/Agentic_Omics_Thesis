# src/muraro_data.py

from pathlib import Path
import pandas as pd
import scanpy as sc
import pandas as pd


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