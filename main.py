"""
MTG RAG API — application entry point.

This module only instantiates the FastAPI app and mounts the router.
All routes and business logic live in api/routes.py and the supporting packages.
"""

from fastapi import FastAPI

from api.routes import router

app = FastAPI(
    title="MTG RAG API",
    description=(
        "Retrieval-Augmented Generation for Magic: The Gathering rules and "
        "card interactions. Backed by ChromaDB (local embeddings) and your "
        "choice of Anthropic or OpenAI as the LLM provider."
    ),
    version="1.0.0",
)

app.include_router(router)
