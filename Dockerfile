# syntax=docker/dockerfile:1
#
# Production image for the SEC-RAG serving app (Step 8).
#
# What goes in: the code, the prompt config, the PREBUILT vector index
# (chroma_db/), and the embedding + reranker models (baked in at build time so
# the first request never waits on a HuggingFace download).
#
# What stays out (see .dockerignore): the venv, the raw filings + intermediate
# data/ (only needed to BUILD the index, not to serve it), and .env (the API key
# is injected as an environment variable by the host, never baked into the image).
#
# Build:  docker build -t sec-rag .
# Run:    docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... sec-rag
#         open http://localhost:8000

# Matches the dev venv (3.9) so the frozen requirements.txt resolves identically.
FROM python:3.9-slim

# Faster, quieter Python; models cached in a predictable place.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /srv

# 1) Dependencies first — this layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Pre-download the models into the image (no runtime HF dependency, fast cold
#    start). Must match the names used in retrieve.py exactly.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# 3) App code + prompt config + prebuilt indexes. The raw filings + intermediate
#    data/ are intentionally absent, but BOTH engines' prebuilt stores ship:
#    Engine 1 = chroma_db/ (vectors), Engine 2 = data/tables.db (structured numbers).
COPY app/ app/
COPY retrieval/ retrieval/
COPY generation/ generation/
COPY prompts/ prompts/
COPY chroma_db/ chroma_db/
# Engine 2: the structured number store (SQLite). Serving reads this for every
# numeric question — without it, TABLE/HYBRID routes degrade to text.
COPY data/tables.db data/tables.db

EXPOSE 8000

# The host (Render/Fly/Fargate) sets $PORT; default to 8000 for local runs.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
