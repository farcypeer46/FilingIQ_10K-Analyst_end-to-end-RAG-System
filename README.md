---
title: FilingIQ — SEC 10-K Q&A
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# FilingIQ — Grounded Q&A over SEC 10-K filings

Advanced/Hybrid RAG over SEC 10-K filings: table-aware, section-aware, hybrid
retrieval + reranking, **citations enforced**, **refuses when unsupported**, and
measured on a held-out evaluation set.

Ask a question in plain English; the system retrieves real evidence from the
filings, generates an answer grounded in that evidence with **exact citations
(company · year · section)**, and says *"I can't answer that from these filings"*
when the filings don't support a claim — rather than hallucinating.

**Corpus:** AAPL, MSFT, NVDA (dev) + AMZN, GOOGL (held-out), 3 years of 10-Ks each
(15 filings), pulled from SEC EDGAR as HTML.

## Architecture

**Ingestion (offline, once):**
```
EDGAR HTML 10-K
  → strip hidden XBRL → detect Item sections (1, 1A, 7, 7A, 8)
  → extract tables as clean Markdown (table-aware; never split a table)
  → token-based chunking within sections (fits the 512-token embedder)
  → contextual LLM summaries for tables (cached)
  → embed locally (bge-small) → store: Chroma (dense) + BM25 (sparse)
```

**Online (per query):**
```
question → embed query
  → hybrid retrieval: dense (Chroma) + sparse (BM25), fused with RRF
  → cross-encoder rerank → top-k excerpts
  → LLM answers using ONLY those excerpts, citing each claim [n]
  → verify citations / refuse if unsupported
```

## Stack

| Layer | Choice |
|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` (local, free) |
| Vector store | ChromaDB (persistent) |
| Sparse retrieval | `rank-bm25` + Reciprocal Rank Fusion |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | OpenAI `gpt-4o-mini` |
| Serving | FastAPI + Uvicorn, containerized with Docker |

Only the generation LLM costs money; embeddings, reranking, vector store, and
BM25 all run locally. Table summaries are cached, so index rebuilds are $0.

## Run locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then paste your OPENAI_API_KEY

# ingest (once): download → extract → build index
python ingestion/fetch_filings.py
python ingestion/extract.py
python ingestion/build_index.py

# serve
uvicorn app.main:app --reload  # open http://localhost:8000
```

## Docker

```bash
docker build -t sec-rag .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... sec-rag
```

See [DEPLOY.md](DEPLOY.md) for deployment (Hugging Face Spaces / Render / AWS).

## API

`POST /ask` → `{question, company?, year?, top_k?}` → grounded answer JSON with
verified `citations` and a `hallucinated_citations` guard. `GET /health` for
readiness. Interactive docs at `/docs`.
