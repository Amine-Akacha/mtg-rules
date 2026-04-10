# MTG Rules RAG API

A Retrieval-Augmented Generation (RAG) service for answering Magic: The Gathering rules and card interaction questions. It embeds the official MTG Comprehensive Rules PDF and the MechaCroc/magic-the-gathering HuggingFace dataset into a local ChromaDB instance, then uses Anthropic or OpenAI to generate answers grounded in that knowledge.

---

## Prerequisites

- Docker and Docker Compose
- An Anthropic API key and/or OpenAI API key
- The MTG Comprehensive Rules PDF (named `mtg-rules.pdf`) placed in the project root

---

## Setup

1. **Copy the example env file and fill in your keys:**

   ```bash
   cp .env.example .env
   # Edit .env and set ANTHROPIC_API_KEY or OPENAI_API_KEY
   ```

2. **Place the rules PDF in the project root:**

   ```
   mtg-rules.pdf   ← must exist before building the image
   ```

3. **Build and start the container:**

   ```bash
   docker compose up --build
   ```

   The API is available at `http://localhost:8000`.

   > **Port conflict?** If port 8000 is already in use, set `HOST_PORT` in your `.env` and restart:
   > ```
   > HOST_PORT=8080
   > ```

---

## Ingest

The `/ingest` endpoint populates ChromaDB with the rules PDF chunks and card data. It is **idempotent** — collections are only filled when empty, so repeated calls are safe.

```bash
curl -X POST http://localhost:8000/ingest
```

Run this once after the container starts. Embeddings are persisted in a Docker volume (`chroma_data`) so they survive container restarts.

---

## API Endpoints

### `GET /health`

Returns service status, the active LLM provider, and the number of documents in each ChromaDB collection.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "provider": "anthropic",
  "collections": {
    "mtg_rules": 4200,
    "mtg_cards": 27000
  }
}
```

---

### `POST /ingest`

Loads the MTG rules PDF and card dataset into ChromaDB. No-ops if collections are already populated.

```bash
curl -X POST http://localhost:8000/ingest
```

```json
{
  "message": "Ingestion complete: 4200 rules chunks, 27000 card chunks.",
  "rules_chunks": 4200,
  "cards_chunks": 27000
}
```

---

### `POST /ask`

Ask a rules or card interaction question. Returns a grounded answer from the configured LLM.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Can a planeswalker be attacked directly?"}'
```

**Request body:**

| Field      | Type   | Required | Description                                      |
|------------|--------|----------|--------------------------------------------------|
| `question` | string | yes      | The MTG rules question to answer                 |
| `provider` | string | no       | Override the LLM provider: `"anthropic"` or `"openai"` |

**Response:**

```json
{
  "answer": "Yes — since 2018 (rule 506.4), opponents may attack planeswalkers directly...",
  "provider_used": "anthropic",
  "chunks_retrieved": 10
}
```

---

## Configuration

All settings are read from `.env` (see `.env.example` for the full list).

| Variable           | Default                        | Description                                    |
|--------------------|--------------------------------|------------------------------------------------|
| `LLM_PROVIDER`     | `anthropic`                    | Active LLM: `anthropic` or `openai`            |
| `ANTHROPIC_API_KEY`| —                              | Required when provider is `anthropic`          |
| `OPENAI_API_KEY`   | —                              | Required when provider is `openai`             |
| `ANTHROPIC_MODEL`  | `claude-sonnet-4-20250514`     | Anthropic model ID                             |
| `OPENAI_MODEL`     | `gpt-4o`                       | OpenAI model ID                                |
| `LLM_MAX_TOKENS`   | `2048`                         | Max tokens for LLM responses                   |
| `HOST_PORT`        | `8000`                         | Host port mapped to the container              |
| `TOP_K`            | `5`                            | Results retrieved per collection (rules + cards each) |
| `CHUNK_SIZE`       | `500`                          | Words per fallback chunk                       |
| `CHUNK_OVERLAP`    | `50`                           | Word overlap between fallback chunks           |
| `BATCH_SIZE`       | `100`                          | ChromaDB upsert batch size                     |

---

## Development (without Docker)

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Ingest and ask work the same way via `http://localhost:8000`.
