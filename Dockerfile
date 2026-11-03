FROM python:3.11-slim

WORKDIR /app

# System build dependencies needed by some Python packages (chromadb, tokenizers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch CPU-only first so sentence-transformers picks it up instead
# of pulling in the much larger CUDA-enabled wheel.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so the container starts without a network
# call. Keep this in sync with EMBEDDING_MODEL (override with --build-arg).
ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
RUN python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['EMBEDDING_MODEL'])"

# Copy application source (secrets and local state excluded via .dockerignore).
# mtg-rules.pdf must be present in the project root before running docker build.
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
