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

# Pre-download the embedding model so the container starts without a network call
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application source
COPY . .

# Copy the MTG Comprehensive Rules PDF into the image.
# mtg-rules.pdf must be present in the project root before running docker build.
COPY mtg-rules.pdf ./mtg-rules.pdf

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
