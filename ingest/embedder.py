"""
Shared embedding logic using sentence-transformers.
All embedding calls funnel through this module so the model is loaded once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from config import settings

if TYPE_CHECKING:
    import numpy as np
    from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return the embedding model, initialising it on first call (lazy singleton)."""
    global _model
    if _model is None:
        # Imported lazily so modules that only need the store (and tests) do not
        # pay for loading torch.
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts.

    Args:
        texts: Raw string documents to encode.

    Returns:
        List of float lists, one per input text.
    """
    model = get_model()
    embeddings: np.ndarray = model.encode(texts, show_progress_bar=False, batch_size=64)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.

    Args:
        query: The user's question.

    Returns:
        Flat float list representing the query embedding.
    """
    return embed_texts([query])[0]
