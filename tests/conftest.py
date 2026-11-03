"""
Shared fixtures. Tests never hit the network: embeddings are faked, ChromaDB
lives in a temp dir, and LLM calls are monkeypatched.
"""

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _fake_embedding(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [b / 255 for b in digest[:8]]


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A chroma_store module pointed at an empty temp dir with fake embeddings."""
    from config import settings
    from retrieval import chroma_store

    monkeypatch.setattr(settings, "CHROMA_DATA_PATH", tmp_path / "chroma")
    monkeypatch.setattr(chroma_store, "_client", None)
    monkeypatch.setattr(chroma_store, "embed_texts", lambda ts: [_fake_embedding(t) for t in ts])
    monkeypatch.setattr(chroma_store, "embed_query", _fake_embedding)
    yield chroma_store
    chroma_store._client = None
