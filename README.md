# FilingIQ
### Advanced RAG over SEC 10-K Filings, Powered by a Dual-Engine Architecture

Trusted financial Q&A system built over real SEC 10-K filings of last 3 years from 5 major
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

The evaluation harness is the core of this project — *"production-grade" and "measured" are
the same thing.* **Methodology firewall:** all development and tuning happened on
**Amazon & Alphabet** (dev set, 49-question golden set). **Apple, Microsoft & NVIDIA**
(36 answerable + 12 must-refuse questions) were held out entirely and **measured once**, on
the frozen system.

> All numbers are from `evaluation/results_*.json`. Regenerate with
> `python evaluation/evaluate.py --golden evaluation/golden_set.yaml`.

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

*Held-out is the honest number — measured once, on companies the system was never tuned against.*

**Zero hallucinated citations and zero missed refusals on filings the system was never tuned
against.** In finance, those are the two failures you can't recover from — and they held.

### Dev → held-out generalization

Numeric exact-match was **0.80 in-sample vs 0.67 held-out** — an honest gap, driven by
table-parse recall on filing formats the parser hadn't seen
(see [Limitations](#limitations--whats-next)).

Two deltas worth explaining rather than hiding:

- **Several retrieval metrics are *higher* on held-out than dev.** Dev is depressed by a known
  section-labeling bug on Amazon (Item 1/1A content collapses into a generic label — see
  Limitations), not by inflated held-out performance.
- **Held-out p50 latency (~3.0 s vs ~1.9 s).** More held-out queries take the TABLE→TEXT
  fallback path on unfamiliar table formats, adding a second retrieval pass.

### Reading the numbers honestly

- **Context precision/recall (~0.66–0.73)** are capped by the section-label collapse in
  retrieval metadata — a known, documented gap, not a mystery.
- **RAGAS metrics** carry roughly ±0.06 run-to-run judge noise; the **deterministic** metrics
  (refusal, numeric exact-match, routing, hallucinated citations) are the primary signal.
- **Every metric is computed by the same harness on both sets** — the only difference is which
  companies, so dev↔held-out deltas are apples-to-apples.


## Why each decision

- **Dual engine, not one vector store** — exact numeric retrieval needs deterministic keys, not
  cosine similarity; near-identical year-over-year text makes a single store return the wrong
  year's number.
- **SQLite table store** — a real relational store gives exact lookup, multi-year stitching, and
  cell-level provenance (click-to-source).
- **Numbers never from the LLM** — every figure traces to a real cell; ingestion captions are
  *deterministic* (built from parsed values), so nothing fabricated can enter the index.
- **Rule-based router** — deterministic, free, unit-testable, and it can't stochastically
  misroute; routing accuracy is a *measured* metric, not an assumption.
- **Hybrid retrieval + RRF** — dense catches meaning, BM25 catches exact terms (tickers, "Item 7A",
  line items); RRF fuses them.
- **Cross-encoder rerank** — the single biggest precision lever on the text side; it reads query
  and passage jointly instead of comparing compressed embeddings.
- **`bge-small-en-v1.5` embeddings** — strong quality-for-size, runs locally at $0, baked into the
  image for fast cold start.
- **Refusal as a verbatim sentinel** — makes "refuse when unsupported" machine-checkable, not a vibe.
- **Deliberate simplicity** — the routing is rule-based *on purpose*. FilingIQ is a grounded RAG
  system, **not an agent** — knowing when *not* to add complexity is part of the design.

## Product layer: memory & cache

- **Conversational memory** — a query-rewriter resolves follow-ups ("…how about Microsoft?") into
  one self-contained question. It rewrites the **question only, never the answer**, so every
  grounding guarantee is preserved. Client-side and per-user.
- **Semantic answer cache** — a cost/latency layer that reuses an answer for a semantically
  equivalent question (~2 s → ~20 ms, $0). A **two-factor key** (cosine similarity **and** exact
  company/year scope) prevents "2024 vs 2023" collisions. Server-side, shared, process-lifetime
  (empty on redeploy, so no stale answers). Hit-rate exposed at `/stats`.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Data source | `sec-edgar-downloader` | Handles EDGAR rate limits + required User-Agent |
| HTML parsing | BeautifulSoup + lxml | Robust on messy filing HTML |
| Table parsing | `pandas.read_html` → canonical grid | One validated parse feeds both engines |
| Table store (Engine 2) | SQLite | Exact lookup, multi-year stitching, provenance |
| Vector store (Engine 1) | ChromaDB | Persistent dense store |
| Sparse retrieval | `rank-bm25` | Exact-term recall |
| Embeddings | `BAAI/bge-small-en-v1.5` | Strong, local, $0 |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Precision lever |
| Router | rule-based | Deterministic, unit-testable |
| Generation | OpenAI `gpt-4o-mini` (temp 0) | Cheap; reasons over evidence, never invents numbers |
| Evaluation | RAGAS + deterministic metrics + routing accuracy | Held-out measurement |
| Serving | FastAPI + minimal frontend | Concurrency + a clean chat UI |
| Deploy | Docker → Hugging Face Spaces | $0 demo; portable image |

## Run it locally

```bash
git clone ⟨YOUR_REPO_URL⟩ && cd FilingIQ
python -m venv venv && source venv/bin/activate      # prompt shows (venv)
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...                          # generation + RAGAS only; everything else is local

# Ingestion (run once, in order):
python ingestion/extract.py            # section- & table-aware extraction
python ingestion/build_table_store.py  # -> data/tables.db  (Engine 2)
python ingestion/build_index.py        # -> chroma_db/      (Engine 1)

# Serve:
uvicorn app.main:app --reload          # http://localhost:8000

# Evaluate:
python evaluation/evaluate.py --golden evaluation/golden_set.yaml
```

## Project structure

```
FilingIQ/
├── ingestion/          # fetch → extract → tables.py (canonical parse) → build both stores
├── retrieval/          # router.py, entity.py (self-query), retrieve.py (hybrid+rerank),
│                       # table_lookup.py (SQL), rewrite.py (memory), cache.py (semantic cache)
├── generation/         # generate.py — grounded, cited, refusing answers
├── evaluation/         # golden sets + evaluate.py (RAGAS + deterministic + routing metrics)
├── prompts/            # versioned prompt configs (answer_v3.yaml)
├── app/                # FastAPI + chat UI
├── data/               # tables.db (Engine 2)
└── chroma_db/          # persisted vector store (Engine 1)
```

## Deployment

Containerized with Docker (embedding + reranker models baked in for fast cold start; the prebuilt
`chroma_db/` and `data/tables.db` ship as serving artifacts via Git LFS). Deployed on **Hugging
Face Spaces**; the `OPENAI_API_KEY` is injected at runtime as a secret, never baked into the image.

```bash
docker build -t filingiq .
docker run --rm -p 8000:8000 -e OPENAI_API_KEY=sk-... filingiq
```

## Scaling to production

This runs at demo scale (15 filings, single-node, ~$0), but every component was chosen so the
*architecture* survives a 4–5 order-of-magnitude jump — swap the implementation, keep the interface.

| Layer | Demo (here) | Production (1M+ filings) |
|---|---|---|
| Ingestion | one script, run once | distributed, incremental (Airflow/Dagster), event-driven off EDGAR |
| Text store | ChromaDB | pgvector / Qdrant / Milvus, sharded, metadata pre-filter |
| Sparse | `rank-bm25` in-memory | Elasticsearch / OpenSearch |
| Table store | SQLite | Postgres (OLTP) + ClickHouse/BigQuery (analytics) |
| Embeddings / rerank | local CPU | batched GPU or hosted API; cache in Redis |
| Serving | one FastAPI process | K8s / ECS behind a load balancer, autoscaled |
| Cache | in-process | shared Redis across replicas |

**What stays the same:** the dual-engine split, the router, "numbers never from an LLM,"
single-source-of-truth parsing, refusal + citation verification, and the held-out eval discipline.
*The hard parts don't change at scale — the stores just get bigger.*

## Limitations & what's next

- **Segment-line ambiguity.** When a filer uses the *same label* for a total and its segments
  (e.g. Amazon labels total revenue "Net sales," identical to its segment rows), the table engine
  **safely refuses** rather than guess a segment number. Next: segment-aware disambiguation.
- **Sub-line metrics.** A few operating-cash-flow / sub-segment lines degrade to the text engine.
- **Section detection** is regex + heuristics; some sections collapse to `other`, capping *strict*
  retrieval Hit@k (retrieval itself is healthy). Next: an ML section segmenter.
- **Table-parse recall** favors precision — a row is emitted only when its cells validate 1:1
  against the detected year columns. A wrong number is worse than a missing one.

---

<p align="center"><sub>Built as a study in grounded, measurable retrieval. MIT License.</sub></p>


<img width="1175" height="600" alt="image" src="https://github.com/user-attachments/assets/d8637a02-ee84-4033-933e-0e63149acecb" />
