from typing import Literal

from pydantic import BaseModel, Field


class SuggestedParameters(BaseModel):
    """Suggested clustering parameters."""

    n_neighbors: int = Field(
        ge=10,
        le=50,
        description=(
            "Suggested number of nearest neighbors for the clustering "
            "graph. Must be between 10 and 50."
        ),
    )

    resolution: float = Field(
        ge=0.1,
        le=2.0,
        description=(
            "Suggested Leiden clustering resolution. Must be between "
            "0.1 and 2.0. Higher values generally produce finer-grained "
            "clustering and may result in more clusters."
        ),
    )

    n_pcs: int = Field(
        ge=10,
        le=50,
        description=(
            "Suggested number of principal components used for "
            "neighborhood graph construction. Must be between 10 and 50."
        ),
    )


class OptimizationDecision(BaseModel):
    """Structured decision returned by the reflection LLM."""

    parameters: SuggestedParameters | None = Field(
        default=None,
        description=(
            "Suggested parameters for the next clustering experiment. "
            "Required when decision is 'continue'. Not required when "
            "decision is 'stop'."
        ),
    )

    decision: Literal["continue", "stop"] = Field(
        description=(
            "Choose continue if another parameter configuration "
            "should be evaluated. Choose stop if the current "
            "optimization should terminate."
        ),
    )

    reason: str = Field(
        description=(
            "Brief explanation of why the optimization should "
            "continue or stop."
        ),
    )


class FinalExperimentSelection(BaseModel):
    """
    Structured decision returned by the LLM
    when selecting the final preferred experiment.
    """

    selected_iteration: int = Field(
        description=(
            "The iteration number of the preferred experiment "
            "selected from the completed experiment history."
        )
    )

    reason: str = Field(
        description=(
            "Brief explanation of why this experiment provides "
            "the strongest overall clustering evidence compared "
            "with the other completed experiments."
        )
    )