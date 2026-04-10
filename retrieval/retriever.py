"""
Top-k retrieval combining results from both MTG collections.

Rules chunks and card chunks are queried independently so that the final
context always contains representatives from both sources.
"""

from config import settings
from retrieval.chroma_store import query_collection


def retrieve(query: str) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks from rules and cards collections.

    Queries each collection separately with settings.TOP_K and concatenates
    the results, so up to 2 * TOP_K chunks are returned in total.

    Args:
        query: The user's natural-language question.

    Returns:
        Combined list of chunk dicts (each with 'text', 'metadata',
        'distance'), rules chunks first followed by card chunks.
    """
    rules_chunks = query_collection(
        settings.RULES_COLLECTION,
        query,
        n_results=settings.TOP_K,
    )

    card_chunks = query_collection(
        settings.CARDS_COLLECTION,
        query,
        n_results=settings.TOP_K,
    )

    return rules_chunks + card_chunks
