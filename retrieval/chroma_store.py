"""
ChromaDB initialisation, upsert, and query helpers.

A single PersistentClient is shared across the process lifetime.
Embeddings are always computed externally (via ingest.embedder) so that
we never rely on ChromaDB's built-in embedding functions.
"""

import chromadb
from chromadb import Collection

from config import settings
from ingest.embedder import embed_query, embed_texts

_client: chromadb.ClientAPI | None = None


def get_client() -> chromadb.ClientAPI:
    """
    Return the shared ChromaDB persistent client, creating it on first call.

    The chroma_data directory is created if it does not already exist.
    """
    global _client
    if _client is None:
        settings.CHROMA_DATA_PATH.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(settings.CHROMA_DATA_PATH))
    return _client


def get_collection(name: str) -> Collection:
    """
    Retrieve or create a ChromaDB collection by name.

    Args:
        name: Collection name (e.g. "mtg_rules" or "mtg_cards").

    Returns:
        The ChromaDB Collection object.
    """
    return get_client().get_or_create_collection(name=name)


def collection_count(name: str) -> int:
    """
    Return the number of documents stored in a collection.

    Args:
        name: Collection name.

    Returns:
        Document count (0 for empty / non-existent collections).
    """
    return get_collection(name).count()


_INGEST_COMPLETE_KEY = "ingest_complete"


def is_ingest_complete(name: str) -> bool:
    """
    Return True when a previous ingest of *name* ran to completion.

    A collection with documents but no completion marker was left behind by an
    ingest that crashed part-way through and must be rebuilt.
    """
    metadata = get_collection(name).metadata or {}
    return bool(metadata.get(_INGEST_COMPLETE_KEY))


def mark_ingest_complete(name: str) -> None:
    """Record on the collection that its ingest finished successfully."""
    collection = get_collection(name)
    # hnsw:* keys are immutable after creation; only carry over the rest.
    metadata = {
        k: v for k, v in (collection.metadata or {}).items() if not k.startswith("hnsw:")
    }
    metadata[_INGEST_COMPLETE_KEY] = True
    collection.modify(metadata=metadata)


def reset_collection(name: str) -> None:
    """Delete *name* (if it exists) so it can be rebuilt from scratch."""
    client = get_client()
    if name in {c if isinstance(c, str) else c.name for c in client.list_collections()}:
        client.delete_collection(name)


def upsert_chunks(
    collection_name: str,
    chunks: list[dict],
    batch_size: int = 100,
) -> None:
    """
    Embed and upsert chunks into a ChromaDB collection in batches.

    Using upsert (rather than add) makes repeated ingest runs idempotent —
    existing documents are overwritten with the same content rather than
    raising a duplicate-ID error.

    Args:
        collection_name: Target collection name.
        chunks: List of dicts, each with 'text' and 'metadata' keys.
        batch_size: Number of chunks to embed and write per round-trip.
    """
    collection = get_collection(collection_name)

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        texts = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]
        ids = [f"{collection_name}_{batch_start + i}" for i in range(len(batch))]
        embeddings = embed_texts(texts)

        collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )


def query_collection(
    collection_name: str,
    query: str,
    n_results: int = 5,
) -> list[dict]:
    """
    Embed *query* and return the top-n closest documents from a collection.

    Returns an empty list when the collection contains no documents.

    Args:
        collection_name: Collection to search.
        query: The user's question or search string.
        n_results: Maximum number of results to return.

    Returns:
        List of dicts with 'text', 'metadata', and 'distance' keys,
        ordered from most to least similar.
    """
    collection = get_collection(collection_name)
    count = collection.count()
    if count == 0:
        return []

    embedding = embed_query(query)
    actual_n = min(n_results, count)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=actual_n,
        include=["documents", "metadatas", "distances"],
    )

    documents: list[str] = results.get("documents", [[]])[0]
    metadatas: list[dict] = results.get("metadatas", [[]])[0]
    distances: list[float] = results.get("distances", [[]])[0]

    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]
