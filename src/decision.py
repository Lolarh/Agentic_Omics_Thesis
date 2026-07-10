from typing import Literal

from pydantic import BaseModel, Field


class SuggestedParameters(BaseModel):
    """Suggested clustering parameters."""

    n_neighbors: int = Field(
        description="Suggested number of neighbors."
    )

    resolution: float = Field(
        description="Suggested Leiden clustering resolution."
    )

    n_pcs: int = Field(
        description="Suggested number of principal components."
    )


class OptimizationDecision(BaseModel):
    """Structured decision returned by the reflection LLM."""

    parameters: SuggestedParameters = Field(
        description="Suggested clustering parameters."
    )

    decision: Literal["continue", "stop"] = Field(
        description="Whether to continue optimization."
    )

    reason: str = Field(
        description="Brief explanation for the decision."
    )