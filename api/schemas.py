"""
Pydantic request and response models for all API endpoints.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    """Body for POST /ask."""

    question: str = Field(..., min_length=1, description="The MTG rules question to answer.")
    provider: Literal["anthropic", "openai"] | None = Field(
        default=None,
        description="Optional LLM provider override ('anthropic' or 'openai'). "
        "Defaults to the LLM_PROVIDER environment variable.",
    )

    @field_validator("provider", mode="before")
    @classmethod
    def normalise_provider(cls, v: object) -> object:
        """Accept any casing, e.g. 'OpenAI'."""
        return v.lower() if isinstance(v, str) else v


class AskResponse(BaseModel):
    """Response from POST /ask."""

    answer: str = Field(..., description="The model's answer to the question.")
    provider_used: str = Field(..., description="LLM provider that generated the answer.")
    chunks_retrieved: int = Field(..., description="Total number of context chunks passed to the model.")


class IngestResponse(BaseModel):
    """Response from POST /ingest."""

    message: str = Field(..., description="Human-readable status message.")
    rules_chunks: int = Field(..., description="Number of rules chunks now in the collection.")
    cards_chunks: int = Field(..., description="Number of card chunks now in the collection.")


class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: str = Field(..., description="Always 'ok' when the service is running.")
    provider: str = Field(..., description="Currently configured LLM_PROVIDER.")
    collections: dict[str, int] = Field(
        ..., description="Mapping of collection name to document count."
    )
