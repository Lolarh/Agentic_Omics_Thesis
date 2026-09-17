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
    banfield_raftery_score: float
    n_clusters: int
    cluster_sizes: dict
    largest_cluster_size: int
    largest_cluster_fraction: float
    number_of_small_clusters: int
    small_clusters: dict
    marker_genes: dict

class Experiment(TypedDict, total=False):
    iteration: int
    parameters: Parameters
    metrics: Metrics
    reason: str

    # Agent decision
    decision: str
    next_parameters: Parameters

    # Stability analysis
    stability: dict



class State(TypedDict):
    """Centralized state."""

    messages: Annotated[list, add_messages]

    adata: AnnData

    parameters: Parameters

    # Experimental condition
    # "no_biology" = optimization without biological evidence
    # "biology" = optimization with biological evidence
    condition: str

    metrics: Metrics

    # Stores every experiment performed by the agent
    history: list[Experiment]

    # Persistent experimental record
    experiment_log: dict

    # Current optimization iteration
    iteration: int

    # Agent decision ("continue" or "stop")
    decision: str

    # Agent's reason for update or stop
    reason: str

    # Prevent duplicate configurations
    evaluated_configs: list[tuple[int, float, int]]

    # Computational timing
    clustering_time: float

    evaluation_time: float

    llm_decision_time: float

    # Final experiment selected after exploration
    selected_iteration: int | None

    # Complete experiment selected after exploration
    best_experiment: Experiment | None

    # Reason for selecting the final experiment
    final_selection_reason: str | None

    # Time required for final LLM selection
    final_selection_time: float