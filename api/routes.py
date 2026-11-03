"""
FastAPI route definitions for /ask, /ingest, and /health.

All business logic lives in the ingest/, retrieval/, and generation/ packages.
This module only wires HTTP requests to those functions.

Handlers are plain ``def`` (not ``async def``) on purpose: embedding, ChromaDB
and the LLM SDK calls are all blocking, so FastAPI must run them in its
threadpool rather than on the event loop.
"""

import threading
from collections.abc import Callable

from fastapi import APIRouter, HTTPException

from api.schemas import AskRequest, AskResponse, HealthResponse, IngestResponse
from config import settings
from generation.llm_client import LLMConfigError, LLMError, generate
from generation.prompt_builder import build_prompt
from ingest.load_cards import load_card_chunks
from ingest.load_rules import load_rules_chunks
from retrieval.chroma_store import (
    collection_count,
    is_ingest_complete,
    mark_ingest_complete,
    reset_collection,
    upsert_chunks,
)
from retrieval.retriever import retrieve

router = APIRouter()

_ingest_lock = threading.Lock()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """
    Return service status, the active LLM provider, and ChromaDB collection sizes.

    Always returns HTTP 200 when the process is running.
    """
    return HealthResponse(
        status="ok",
        provider=settings.LLM_PROVIDER,
        collections={
            settings.RULES_COLLECTION: collection_count(settings.RULES_COLLECTION),
            settings.CARDS_COLLECTION: collection_count(settings.CARDS_COLLECTION),
        },
    )


@router.post("/ask", response_model=AskResponse, tags=["rag"])
def ask(request: AskRequest) -> AskResponse:
    """
    Answer an MTG rules question using retrieval-augmented generation.

    Raises:
        HTTPException 503: When the selected provider is not configured.
        HTTPException 502: When the provider API call fails.
    """
    active_provider = request.provider or settings.LLM_PROVIDER

    chunks = retrieve(request.question)
    system_prompt, user_message = build_prompt(request.question, chunks)
    try:
        answer = generate(system_prompt, user_message, provider=active_provider)
    except LLMConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return AskResponse(
        answer=answer,
        provider_used=active_provider,
        chunks_retrieved=len(chunks),
    )


def _ingest_collection(name: str, load: Callable[[], list[dict]]) -> int:
    """
    Populate *name* unless a previous ingest already completed.

    A collection left half-populated by a crashed ingest is dropped and rebuilt.
    Returns the number of chunks ingested (0 when skipped).
    """
    if is_ingest_complete(name):
        return 0

    chunks = load()
    reset_collection(name)
    upsert_chunks(name, chunks, batch_size=settings.BATCH_SIZE)
    mark_ingest_complete(name)
    return len(chunks)


@router.post("/ingest", response_model=IngestResponse, tags=["ops"])
def ingest() -> IngestResponse:
    """
    Run the ingest pipeline to populate ChromaDB with rules and card data.

    Idempotent — a collection is only (re)built when no previous ingest of it
    ran to completion, so repeated calls do not re-embed finished data.

    Raises:
        HTTPException 404: When the rules PDF is missing from the expected path.
        HTTPException 409: When another ingest is already running.
    """
    if not _ingest_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="An ingest is already running.")

    try:
        try:
            rules_ingested = _ingest_collection(settings.RULES_COLLECTION, load_rules_chunks)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        cards_ingested = _ingest_collection(settings.CARDS_COLLECTION, load_card_chunks)
    finally:
        _ingest_lock.release()

    if rules_ingested == 0 and cards_ingested == 0:
        message = "Collections already populated — no ingestion performed."
    else:
        parts = []
        if rules_ingested:
            parts.append(f"{rules_ingested} rules chunks")
        if cards_ingested:
            parts.append(f"{cards_ingested} card chunks")
        message = f"Ingestion complete: {', '.join(parts)}."

    return IngestResponse(
        message=message,
        rules_chunks=collection_count(settings.RULES_COLLECTION),
        cards_chunks=collection_count(settings.CARDS_COLLECTION),
    )
