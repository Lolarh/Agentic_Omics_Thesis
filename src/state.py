from typing import Annotated, TypedDict

from anndata import AnnData
from langgraph.graph.message import add_messages


class Parameters(TypedDict):
    n_neighbors: int
    resolution: float
    n_pcs: int


class Metrics(TypedDict, total=False):
    silhouette_score: float
    davies_bouldin_score: float
    wc_dispersion_score: float
    n_clusters: int
    cluster_sizes: dict
    largest_cluster_size: int
    largest_cluster_fraction: float
    number_of_small_clusters: int
    small_clusters: dict


class Experiment(TypedDict, total=False):
    iteration: int
    parameters: Parameters
    metrics: Metrics
    reason: str


class State(TypedDict):
    """Centralized state."""

    messages: Annotated[list, add_messages]

    adata: AnnData

    parameters: Parameters

    metrics: Metrics

    # Stores every experiment performed by the agent
    history: list[Experiment]

    # Current optimization iteration
    iteration: int

    # Agent decision ("continue" or "stop")
    decision: str

    # Agent's reason for update or stop
    reason: str

    # Prevent duplicate configurations
    evaluated_configs: list[tuple[int, float, int]]