import pytest
from fastapi.testclient import TestClient

from api import routes
from generation.llm_client import LLMConfigError, LLMError
from main import app


@pytest.fixture
def client(store):
    return TestClient(app)


def _chunks(n: int, source: str) -> list[dict]:
    return [{"text": f"{source} {i}", "metadata": {"source": source}} for i in range(n)]


def test_ask_rejects_unknown_provider(client):
    resp = client.post("/ask", json={"question": "q", "provider": "gemini"})
    assert resp.status_code == 422


def test_ask_accepts_provider_any_case(client, monkeypatch):
    monkeypatch.setattr(routes, "generate", lambda s, u, provider: f"via {provider}")
    resp = client.post("/ask", json={"question": "q", "provider": "OpenAI"})
    assert resp.status_code == 200
    assert resp.json()["provider_used"] == "openai"


@pytest.mark.parametrize("exc, status", [(LLMConfigError("no key"), 503), (LLMError("boom"), 502)])
def test_ask_maps_llm_errors(client, monkeypatch, exc, status):
    def fail(*a, **k):
        raise exc

    monkeypatch.setattr(routes, "generate", fail)
    resp = client.post("/ask", json={"question": "q"})
    assert resp.status_code == status
    assert str(exc) in resp.json()["detail"]


def test_ingest_is_idempotent(client, monkeypatch):
    monkeypatch.setattr(routes, "load_rules_chunks", lambda: _chunks(3, "rules"))
    monkeypatch.setattr(routes, "load_card_chunks", lambda: _chunks(4, "cards"))

    first = client.post("/ingest").json()
    assert (first["rules_chunks"], first["cards_chunks"]) == (3, 4)

    monkeypatch.setattr(routes, "load_rules_chunks", lambda: pytest.fail("re-ingested rules"))
    monkeypatch.setattr(routes, "load_card_chunks", lambda: pytest.fail("re-ingested cards"))
    second = client.post("/ingest").json()
    assert second["message"].startswith("Collections already populated")
    assert (second["rules_chunks"], second["cards_chunks"]) == (3, 4)


def test_ingest_rebuilds_partial_collection(client, store, monkeypatch):
    from config import settings

    # Simulate a crash mid-ingest: documents present, no completion marker.
    store.upsert_chunks(settings.CARDS_COLLECTION, _chunks(2, "cards"))
    monkeypatch.setattr(routes, "load_rules_chunks", lambda: _chunks(3, "rules"))
    monkeypatch.setattr(routes, "load_card_chunks", lambda: _chunks(5, "cards"))

    body = client.post("/ingest").json()
    assert body["cards_chunks"] == 5
    assert store.is_ingest_complete(settings.CARDS_COLLECTION)


def test_ingest_missing_pdf_returns_404_and_keeps_data(client, monkeypatch):
    def missing():
        raise FileNotFoundError("Rules PDF not found")

    monkeypatch.setattr(routes, "load_rules_chunks", missing)
    resp = client.post("/ingest")
    assert resp.status_code == 404
    # Lock must be released so a retry is possible.
    assert routes._ingest_lock.acquire(blocking=False)
    routes._ingest_lock.release()


def test_ingest_concurrent_returns_409(client):
    assert routes._ingest_lock.acquire(blocking=False)
    try:
        assert client.post("/ingest").status_code == 409
    finally:
        routes._ingest_lock.release()
