"""
FastAPI route definitions for /ask, /ingest, and /health.

All business logic lives in the ingest/, retrieval/, and generation/ packages.
This module only wires HTTP requests to those functions.
"""

from fastapi import APIRouter, HTTPException

from api.schemas import AskRequest, AskResponse, HealthResponse, IngestResponse
from config import settings
from generation.llm_client import generate
from generation.prompt_builder import build_prompt
from ingest.load_cards import load_card_chunks
from ingest.load_rules import load_rules_chunks
from retrieval.chroma_store import collection_count, upsert_chunks
from retrieval.retriever import retrieve

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
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
async def ask(request: AskRequest) -> AskResponse:
    """
    Answer an MTG rules question using retrieval-augmented generation.

    1. Validates the requested provider (if supplied).
    2. Retrieves the top-k most relevant chunks from both collections.
    3. Builds the prompt with retrieved context.
    4. Calls the configured LLM and returns the answer.
    """
    active_provider = (request.provider or settings.LLM_PROVIDER).lower()

    if active_provider not in ("anthropic", "openai"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid provider {active_provider!r}. Must be 'anthropic' or 'openai'.",
        )

    chunks = retrieve(request.question)
    system_prompt, user_message = build_prompt(request.question, chunks)
    answer = generate(system_prompt, user_message, provider=active_provider)

    return AskResponse(
        answer=answer,
        provider_used=active_provider,
        chunks_retrieved=len(chunks),
    )


@router.post("/ingest", response_model=IngestResponse, tags=["ops"])
async def ingest() -> IngestResponse:
    """
    Run the ingest pipeline to populate ChromaDB with rules and card data.

    Idempotent — a collection is only populated when it is empty, so repeated
    calls will not re-embed or overwrite existing data.

    Raises:
        HTTPException 404: When the rules PDF is missing from the expected path.
        HTTPException 500: On unexpected errors during dataset loading.
    """
    rules_count = collection_count(settings.RULES_COLLECTION)
    cards_count = collection_count(settings.CARDS_COLLECTION)

    rules_ingested = 0
    cards_ingested = 0

    if rules_count == 0:
        try:
            rules_chunks = load_rules_chunks()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        upsert_chunks(settings.RULES_COLLECTION, rules_chunks, batch_size=settings.BATCH_SIZE)
        rules_ingested = len(rules_chunks)

    if cards_count == 0:
        cards_chunks = load_card_chunks()
        upsert_chunks(settings.CARDS_COLLECTION, cards_chunks, batch_size=settings.BATCH_SIZE)
        cards_ingested = len(cards_chunks)

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
        rules_chunks=rules_ingested or rules_count,
        cards_chunks=cards_ingested or cards_count,
    )
