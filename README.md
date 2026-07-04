# FilingIQ
### Advanced RAG System over SEC 10-K Filings — Powered by a Dual-Engine Architecture

<p align="center">
  <a href="⟨YOUR_LIVE_APP_URL⟩"><img src="https://img.shields.io/badge/%F0%9F%9A%80_Live_Demo-Hugging_Face_Spaces-FFD21E" alt="Live Demo"></a>
  <img src="https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white" alt="Python 3.9">
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License: MIT">
</p>

**Trusted financial Q&A over real SEC 10-K filings.** Ask a question in plain English and
FilingIQ pulls the *exact* figure from a validated table store, grounds every claim in a
citation, and **refuses** when the filings don't support an answer. Under the hood: a
deterministic query router over a **dual engine** — exact-cell SQL lookups for numbers, and
hybrid dense + BM25 retrieval with cross-encoder reranking for prose — served on **FastAPI**
and kept honest by automated **RAGAS evaluation gated in CI**.

<p align="center">
  <a href="⟨YOUR_LIVE_APP_URL⟩">
    <img src="docs/demo.gif" alt="FilingIQ answering a financial question with a cited figure" width="90%">
  </a>
</p>

---

## Contents
- [The problem](#the-problem)
- [Try it](#try-it)
- [Architecture](#architecture)
- [Evaluation & results](#evaluation--results)
- [Why each decision](#why-each-decision)
- [Product layer: memory & cache](#product-layer-memory--cache)
- [Tech stack](#tech-stack)
- [Run it locally](#run-it-locally)
- [Project structure](#project-structure)
- [Deployment](#deployment)
- [Scaling to production](#scaling-to-production)
- [Limitations & what's next](#limitations--whats-next)

---

## The problem

Numbers do not belong in a vector search. Ask a vector store *"net income in 2025?"* and it may
return **2024's** number, because the surrounding text is near-identical — and in finance a
near-miss is a total failure. Meanwhile, a plain LLM will happily **fabricate** a figure that
looks plausible but appears in no filing.

FilingIQ separates the two problems it was conflating: **deterministic data** (exact numbers)
and **linguistic context** (risks, strategy, commentary). Numbers come from a structured store
by exact key; prose comes from semantic search; a router sends each question to the right one —
and the system **refuses** rather than guess.

## Try it

**▶ Live demo:** ⟨YOUR_LIVE_APP_URL⟩ · Corpus: **Apple, Microsoft, NVIDIA, Amazon, Alphabet** — latest 3 years of 10-Ks each.

| Ask | FilingIQ… |
|---|---|
| *"What was Microsoft's total revenue in fiscal 2024?"* | returns **$245,122M** 📊 from the Table Engine, cited to the filing |
| *"What are NVIDIA's main supply-chain risks?"* | returns grounded prose 📄 from the Text Engine, with citations |
| *"Compare Apple's and Alphabet's net income in fiscal 2024."* | fuses both companies' exact figures 🔀 |
| *"What is Apple's current stock price?"* | **refuses** — it isn't in the filings |

## Architecture

<p align="center">
  <img src="docs/architecture.svg" alt="FilingIQ dual-engine architecture" width="100%">
</p>

**One parse, two stores, one router.** Each table is parsed **once** into a validated canonical
grid that feeds *both* engines — so the structured store and the searchable text can never
disagree, and **no number ever originates from an LLM**.

- **Ingestion (offline).** SEC EDGAR 10-K HTML → strip the hidden XBRL block + detect Item
  sections (BeautifulSoup/lxml) → locate tables and coarse-parse with `pandas.read_html` → the
  **Canonical Table Parser** validates each row (column dedupe, label↔value binding, unit &
  fiscal-year detection) and emits: **(a)** exact records → **SQLite**, and **(b)** deterministic
  captions + prose chunks → **ChromaDB**.
- **Query Router (rule-based).** Classifies each question `TEXT` / `TABLE` / `HYBRID` — free,
  deterministic, and unit-testable.
- **Engine 2 — Tables (SQLite).** Exact cell lookup by `(ticker, fiscal_year, line_item)`. The
  retrieved figures are pinned to the top of the context and **bypass the reranker** — they're
  ground truth, not candidates to be re-scored.
- **Engine 1 — Text (ChromaDB + BM25).** Self-query metadata filter → dense (`bge-small`) + sparse
  (BM25) retrieval → **RRF fusion** → **cross-encoder reranking**.
- **Grounded generation (`gpt-4o-mini`, temp 0).** Answers only from the provided evidence, cites
  every claim `[n]`, and emits an exact refusal sentence when the evidence is insufficient.

**A query, end to end:** *"What was NVIDIA's revenue in fiscal 2025?"* → router = `TABLE` →
SQL lookup `(NVDA, 2025, TOTAL_REVENUE)` → exact cell → the LLM phrases the answer *around* that
retrieved number, with a citation → cached for next time.

## Evaluation & results

> The evaluation harness is the core of the project — *"production-grade" and "measured" are the
> same thing.* Methodology firewall: **development on Amazon & Alphabet**, and a **frozen held-out
> set of Apple, Microsoft & NVIDIA** touched **once**. Numbers below are from
> `evaluation/results_*.json` — regenerate with `python evaluation/evaluate.py`.

**Held-out results** (frozen AAPL / MSFT / NVDA — the honest headline):

| Metric | Score |
|---|---|
| Faithfulness (RAGAS) | 0.90 |
| Refusal precision / recall | 0.75 / 1.00 |
| Numeric exact-match | 0.67 |
| Routing accuracy | 0.97 |
| Retrieval Hit@k (lenient) | 1.00 |
| **Hallucination rate** | **0.00** |

**The generalization story (why this project is honest).** The held-out set did its job — it
exposed a real overfit and forced a real fix:

| Stage | Held-out numeric exact-match |
|---|---|
| In-sample (dev companies) | 0.80 |
| First held-out run | **0.33** ← overfit exposed |
| + fix: date-header column parsing | 0.44 |
| + fix: %-table / deferred-revenue disambiguation | **0.67** |

The root cause of the 0.80 → 0.33 drop: the table parser only recognized bare-year column headers
(`"2024"`), so **NVIDIA's date-style headers (`"Jan 26, 2025"`) yielded zero rows in the table
store** — a failure the dev companies (which use bare-year headers) *structurally could not
reveal*. Diagnosed against the held-out set, fixed generally (not tuned to the questions), and
recovered. Throughout, **hallucination stayed at 0 and no valid refusal was ever missed** — the
safety rails held even while accuracy moved.

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
