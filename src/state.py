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
    n_clusters: int
    cluster_sizes: dict


class Experiment(TypedDict):
    iteration: int
    parameters: Parameters
    metrics: Metrics


class State(TypedDict):
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

    # Agents reason for update or stop
    reason: str