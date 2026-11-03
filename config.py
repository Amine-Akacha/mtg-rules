"""
Central configuration — all settings are read from environment variables.
No config values should be hardcoded elsewhere in the project.
"""

from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------------------------------------------------------------------
    # LLM provider
    # ---------------------------------------------------------------------------
    LLM_PROVIDER: Literal["anthropic", "openai"] = "anthropic"
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    OPENAI_MODEL: str = "gpt-4o"
    LLM_MAX_TOKENS: int = 2048

    # ---------------------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------------------
    CHROMA_DATA_PATH: Path = Path("./chroma_data")
    RULES_PDF_PATH: Path = Path("./mtg-rules.pdf")

    # ---------------------------------------------------------------------------
    # ChromaDB collections
    # ---------------------------------------------------------------------------
    RULES_COLLECTION: str = "mtg_rules"
    CARDS_COLLECTION: str = "mtg_cards"

    # ---------------------------------------------------------------------------
    # Embeddings
    # ---------------------------------------------------------------------------
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ---------------------------------------------------------------------------
    # HuggingFace dataset
    # ---------------------------------------------------------------------------
    CARDS_DATASET: str = "MechaCroc/magic-the-gathering"
    CARDS_SPLIT: str = "train_clean"

    # ---------------------------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------------------------
    TOP_K: int = 5

    # ---------------------------------------------------------------------------
    # Chunking
    # ---------------------------------------------------------------------------
    # Soft cap per rules chunk. all-MiniLM-L6-v2 only reads the first 256
    # word-pieces (~180 words) of each text; anything beyond is not embedded.
    RULE_CHUNK_MAX_WORDS: int = 180
    # Fixed-window fallback, used only when the PDF has no rule numbers.
    CHUNK_SIZE: int = 500   # words (whitespace-tokenised proxy for tokens)
    CHUNK_OVERLAP: int = 50

    # ---------------------------------------------------------------------------
    # Ingest
    # ---------------------------------------------------------------------------
    BATCH_SIZE: int = 100

    @field_validator("LLM_PROVIDER", mode="before")
    @classmethod
    def normalise_provider(cls, v: str) -> str:
        """Ensure provider value is lower-case before validation."""
        return v.lower()


settings = Settings()
