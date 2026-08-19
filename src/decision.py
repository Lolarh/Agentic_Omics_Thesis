from typing import Literal

from pydantic import BaseModel, Field


class SuggestedParameters(BaseModel):
    """Suggested clustering parameters."""

    n_neighbors: int = Field(
        description=(
            "Suggested number of nearest neighbors "
            "for the clustering graph."))

    resolution: float = Field(
        description=(
            "Suggested Leiden clustering resolution. "
            "Higher values generally produce more clusters."))

    n_pcs: int = Field(
        description=(
            "Suggested number of principal components "
            "used for neighborhood graph construction."))


class OptimizationDecision(BaseModel):
    """Structured decision returned by the reflection LLM."""

    parameters: SuggestedParameters = Field(
        description=(
            "Suggested parameters for the next clustering "
            "experiment. If decision is stop, these parameters "
            "are not used."))

    decision: Literal["continue", "stop"] = Field(
        description=(
            "Choose continue if another parameter configuration "
            "should be evaluated. Choose stop if the current "
            "optimization should terminate."))

    reason: str = Field(
        description=(
            "Brief explanation of why the optimization should "
            "continue or stop."))