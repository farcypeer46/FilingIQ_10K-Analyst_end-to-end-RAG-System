---
title: FilingIQ
emoji: 🔍
colorFrom: gray
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# FilingIQ
### Advanced RAG system built over SEC 10-K Filings, Powered by a Dual-Engine Architecture

Trusted financial Q&A RAG system built over real SEC 10-K filings of last 3 years from 5 major
companies. Ask a question in plain English and FilingIQ pulls the *exact* figure from a
validated table store, grounds every claim in a citation, and **refuses** when the filings
don't support an answer or the evidence is uncertain, **zero hallucinated citations across
the full eval set.**

**Under the hood:** a deterministic query router over a **dual engine** exact-cell SQL
lookups for numbers, hybrid dense + BM25 retrieval with cross-encoder reranking for prose and
served on **FastAPI** and kept honest by automated **RAGAS evaluation gated in CI**.

[![Live Demo](https://img.shields.io/badge/%F0%9F%9A%80_Live_Demo-Hugging_Face_Spaces-FFD21E)](https://huggingface.co/spaces/peerfarcy46/10Kfilingiq)
![Python 3.9](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

[![FilingIQ demo](docs/demo.gif)](https://huggingface.co/spaces/peerfarcy46/10Kfilingiq)

---

## Contents

- [The Problem](#the-problem)
- [Live Demo](#live-demo)
- [Architecture](#architecture)
- [Evaluation & Results](#evaluation--results)
- [Design Decisions](#design-decisions)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Deployment & Scaling](#deployment--scaling)
- [Limitations & Roadmap](#limitations--roadmap)

---

## The Problem

**Numbers don't belong in a vector search.** Ask a vector store *"What was net income in 2025?"*
and it may confidently return **2024's figure** because the surrounding text is nearly identical, and
embeddings can't tell the difference. In finance, a near-miss is not a partial success; it's a
wrong answer with a citation attached. Worse, a plain LLM will simply **fabricate** a plausible
figure that appears in no filing at all.

The root cause: naive RAG treats two fundamentally different problems as one.
**Deterministic data** (exact figures, where there is precisely one right answer) and
**linguistic context** (risks, strategy, commentary, where meaning matters more than exactness)
fail in different ways and need different retrieval.

FilingIQ separates them. Numbers are answered by **exact-key lookup** against a validated
structured store and the LLM never generates a figure, only explains one. Prose is answered by
**semantic search** over the filings, every claim cited. A deterministic router dispatches each
question to the right engine, and when neither engine finds support, the system **refuses**
instead of guessing.

## Try it

**▶ Live demo:** ⟨YOUR_LIVE_APP_URL⟩ · Corpus: **Apple, Microsoft, NVIDIA, Amazon, Alphabet** — latest 3 years of 10-Ks each.

| Ask | FilingIQ… |
|---|---|
| *"What was Microsoft's total revenue in fiscal 2024?"* | returns **$245,122M** 📊 from the Table Engine, cited to the filing |
| *"What are NVIDIA's main supply-chain risks?"* | returns grounded prose 📄 from the Text Engine, with citations |
| *"Compare Apple's and Alphabet's net income in fiscal 2024."* | fuses both companies' exact figures 🔀 |
| *"What is Apple's current stock price?"* | **refuses** — it isn't in the filings |

## Architecture

![FilingIQ dual-engine architecture](https://github.com/user-attachments/assets/d8637a02-ee84-4033-933e-0e63149acecb)

**One parse, two stores, one router.** Each table is parsed **once** into a validated canonical
grid that feeds *both* engines — so the structured store and the searchable text can never
disagree, and **no number ever originates from an LLM**.

- **Ingestion (offline).** SEC EDGAR 10-K HTML → strip the hidden XBRL block and detect Item
  sections (BeautifulSoup/lxml) → locate tables and coarse-parse with `pandas.read_html` → the
  **Canonical Table Parser** validates each row (column dedupe, label↔value binding, unit and
  fiscal-year detection) and emits **(a)** exact records → **SQLite** and **(b)** deterministic
  captions + prose chunks → **ChromaDB**.
- **Query Router (rule-based).** Classifies each question as `TEXT`, `TABLE`, or `HYBRID` —
  free, deterministic, and unit-testable.
- **Engine 2 — Tables (SQLite).** Exact cell lookup by `(ticker, fiscal_year, line_item)`.
  Retrieved figures are pinned to the top of the context and **bypass the reranker** — they're
  ground truth, not candidates to be re-scored.
- **Engine 1 — Text (ChromaDB + BM25).** Self-query metadata filter → dense (`bge-small`) +
  sparse (BM25) retrieval → **RRF fusion** → **cross-encoder reranking** → top-5 to the LLM.
- **Grounded generation (`gpt-4o-mini`, temp 0).** Answers only from the provided evidence,
  cites every claim `[n]`, and emits an exact refusal sentence when the evidence is insufficient.

**A query, end to end:** *"What was NVIDIA's revenue in fiscal 2025?"* → router: `TABLE` →
SQL lookup `(NVDA, FY2025, TOTAL_REVENUE)` → exact cell retrieved → the LLM phrases the answer
*around* the retrieved number, with a citation — and the result is cached for next time.

## Evaluation & Results

The evaluation harness is the core of this project. A system whose performance is never
measured is a system whose claims can't be trusted, So every capability claimed here is
backed by a number. 
**Methodology firewall:** all development and tuning happened on
**Amazon & Alphabet**. **Apple, Microsoft & NVIDIA**
were held out entirely and **measured once**,
on the frozen system.


### Results by category

| Category | Metric | What it proves | Dev (AMZN/GOOGL) | **Held-out (AAPL/MSFT/NVDA)** |
|---|---|---|---:|---:|
| **Retrieval** | Context precision (RAGAS) | retrieved chunks are relevant | 0.66 | **0.70** |
| | Context recall (RAGAS) | retrieval surfaced what's needed | 0.69 | **0.73** |
| | Hit@k (lenient) | a relevant chunk is in top-k | 1.00 | **1.00** |
| | Hit@k (strict) | the *right* chunk ranks top | 0.38 | **0.58** |
| **Generation** | Faithfulness (RAGAS) | answer is grounded in the context | 0.89 | **0.90** |
| | Answer relevancy (RAGAS) | answer addresses the question | 0.84 | **0.81** |
| **Grounding** | Hallucinated citations | claims cite only provided evidence | 0.00 | **0.00** |
| **Safety** | Refusal recall | never answers the unanswerable | 1.00 | **1.00** |
| | Refusal precision | doesn't over-refuse | 0.80 | **0.75** |
| **Routing** | Routing accuracy | router picks the right engine | 1.00 | **0.97** |
| **Numeric** | Exact-match | the exact figure is correct | 0.80 | **0.67** |
| **Operational** | Latency (p50) | responsive under load | ~1.9 s | **~3.0 s** |

*Held-out is measured once, on companies and then system was never tuned against.*

**Zero hallucinated citations and zero missed refusals on filings the system was never tuned
against.** In finance, those are the two failures you can't recover from and they held.

## Design Decisions

Every architectural choice below exists for a measured reason, and knowing when *not* to add
complexity is part of the design.

- **Dual engine, not one vector store.** Exact numeric retrieval needs deterministic keys, not
  cosine similarity and year-over-year filing text is near-identical, so a single vector store
  will return the wrong year's number confidentely.
- **SQLite as the table store.** A real relational store gives exact lookup, multi-year
  stitching, and cell-level provenance so every figure is click-to-source.
- **Numbers never originate from the LLM.** Every figure traces to a real filing cell.
  Ingestion captions are *deterministic* and built from parsed values, so nothing fabricated
  can enter the index.
- **Rule-based router.** Deterministic, free, and unit-testable, it cannot stochastically
  misroute. Routing accuracy is a *measured* metric (0.97 on held-out).
- **Hybrid retrieval + RRF.** Dense retrieval catches meaning; BM25 catches exact terms
  (tickers, "Item 7A", line items). Reciprocal Rank Fusion combines both rankings.
- **Cross-encoder reranking.** The single biggest precision lever on the text side so it reads
  query and passage jointly rather than comparing compressed embeddings.
- **`bge-small-en-v1.5` embeddings.** Strong quality-for-size, runs locally at zero cost, and
  is baked into the Docker image for fast cold starts.
- **Refusal as a verbatim sentinel.** "Refuse when unsupported" is machine-checkable in the
  eval harness — not a vibe.
- **Deliberately not an agent.** FilingIQ is a grounded RAG system with rule-based routing
  *on purpose*. Restraint is a design decision, not a limitation.

### Product layer: memory & cache

- **Conversational memory.** A query rewriter resolves follow-ups (*"…how about Microsoft?"*)
  into one self-contained question. It rewrites the **question only, never the answer**, every
  grounding guarantee is preserved.
- **Semantic answer cache.** Reuses an answer for a semantically equivalent question
  (~2 s → ~20 ms, at zero cost). A **two-factor key** cosine similarity *and* exact
  company/year scope, prevents "2024 vs 2023" collisions. Server-side, shared,
  process-lifetime (empties on redeploy, so no stale answers). Hit rate exposed at `/stats`.

## Tech Stack

Ordered by the path a document takes from EDGAR to a cited answer:

| Layer | Choice |
|---|---|
| Data source | `sec-edgar-downloader` |
| HTML parsing | BeautifulSoup + lxml |
| Table parsing | `pandas.read_html` → canonical grid |
| Table store | SQLite |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Vector store | ChromaDB |
| Sparse retrieval | `rank-bm25` |
| Router | Rule-based (custom) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | OpenAI `gpt-4o-mini` (temp 0) |
| Serving | FastAPI + minimal frontend |
| Deployment | Docker → Hugging Face Spaces |
| Evaluation | RAGAS + deterministic metrics |


## Getting Started

```bash
git clone ⟨YOUR_REPO_URL⟩ && cd FilingIQ
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...   # generation + RAGAS eval only — everything else runs locally

# Ingestion (run once, in order)
python ingestion/extract.py             # section- & table-aware extraction
python ingestion/build_table_store.py   # → data/tables.db   (Table Engine)
python ingestion/build_index.py         # → chroma_db/       (Text Engine)

# Serve
uvicorn app.main:app --reload           # http://localhost:8000

# Evaluate
python evaluation/evaluate.py --golden evaluation/golden_set.yaml
```

## Project structure

```
FilingIQ/
├── ingestion/                 # OFFLINE builds both stores (run once, in order)
│   ├── fetch_filings.py       # download 3yr × 5 companies from SEC EDGAR
│   ├── extract.py             # strip hidden XBRL, detect Item sections, locate tables
│   ├── tables.py              # Canonical Table Parser 
│   ├── build_table_store.py   # validated records to SQLite (Engine 2)
│   └── build_index.py         # prose and deterministic captions to ChromaDB (Engine 1)
├── retrieval/
│   ├── router.py              # Query Router — TEXT / TABLE / HYBRID (rule-based)
│   ├── entity.py              # self-query: company/year extraction → metadata filter
│   ├── retrieve.py            # hybrid dense + BM25 → RRF fusion → cross-encoder rerank
│   ├── table_lookup.py        # exact SQL lookup against Engine 2
│   ├── rewrite.py             # conversational memory (query rewriter)
│   └── cache.py               # semantic answer cache (two-factor key)
├── generation/
│   └── generate.py            # grounded, cited, refusing answers; routes per the router
├── evaluation/
│   ├── evaluate.py            # RAGAS + deterministic + routing metrics
│   ├── golden_set.yaml        # dev set (AMZN / GOOGL)
│   ├── golden_set_test.yaml   # frozen held-out set (AAPL / MSFT / NVDA)
│   └── results_*.json         # measured results (dev + held-out)
├── prompts/
│   └── answer_v3.yaml         # versioned prompt (partial-answer rule; trust structured data)
├── app/
│   └── main.py                # FastAPI and self-contained chat UI
├── data/tables.db             # Engine 2, structured number store (SQLite)
├── chroma_db/                 # Engine 1, persisted vector store (ChromaDB)
├── Dockerfile                 # serving image (models baked in; stores shipped via Git LFS)
└── requirements.txt
```

```

## Deployment & Scaling

Containerized with Docker — embedding and reranker models are baked into the image for fast
cold starts, and the prebuilt `chroma_db/` and `data/tables.db` ship as serving artifacts via
Git LFS. Deployed on **Hugging Face Spaces**; `OPENAI_API_KEY` is injected at runtime as a
secret, never baked into the image.

```bash
docker build -t filingiq .
docker run --rm -p 8000:8000 -e OPENAI_API_KEY=sk-... filingiq
```

### Scaling path

FilingIQ runs at demo scale (15 filings, single node, ~$0), but every component was chosen so
the **architecture survives the jump to production scale** — swap the implementation, keep
the interface.

| Layer | Demo (this repo) | Production (1M+ filings) |
|---|---|---|
| Ingestion | single script, run once | distributed, incremental (Airflow/Dagster), event-driven off EDGAR |
| Text store | ChromaDB | pgvector / Qdrant / Milvus — sharded, metadata pre-filtered |
| Sparse retrieval | `rank-bm25` in-memory | Elasticsearch / OpenSearch |
| Table store | SQLite | Postgres (OLTP) + ClickHouse/BigQuery (analytics) |
| Embeddings / rerank | local CPU | batched GPU or hosted API, Redis-cached |
| Serving | single FastAPI process | K8s / ECS behind a load balancer, autoscaled |
| Cache | in-process | shared Redis across replicas |

**What stays the same at any scale:** the dual-engine split, the router, *numbers never from
an LLM*, single-source-of-truth parsing, refusal + citation verification, and the held-out
evaluation discipline.

## Limitations & Roadmap

- **Segment-line ambiguity.** When a filer uses the same label for a total and its segments
  (Amazon labels total revenue "Net sales," identical to its segment rows), the table engine
  **refuses rather than guesses**. Roadmap: segment-aware disambiguation.
- **Sub-line metric coverage.** A few operating-cash-flow and sub-segment lines fall back to
  the text engine rather than resolving to an exact cell.
- **Section detection** is regex + heuristics; some sections collapse into a generic `other`
  label, which caps *strict* retrieval Hit@k (lenient retrieval is unaffected). Roadmap: an
  ML-based section segmenter.
- **Table-parse recall deliberately favors precision.** A row enters the store only when its
  cells validate 1:1 against the detected year columns — a wrong number is worse than a
  missing one.

---
