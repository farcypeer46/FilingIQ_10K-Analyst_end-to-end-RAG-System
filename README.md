<h1 align="center">FilingIQ</h1>

<p align="center">
  <b>Grounded financial Q&A over SEC 10-K filings — built so the LLM can't invent a number.</b><br>
  Every figure traces to an exact cell in a real filing. Every claim is cited.
  When the filings don't support an answer, FilingIQ refuses.
</p>

<p align="center">
  <a href="⟨YOUR_LIVE_APP_URL⟩"><img src="https://img.shields.io/badge/Live_Demo-Hugging_Face_Spaces-FFD21E" alt="Live Demo"></a>
  <img src="https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white" alt="Python 3.9">
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License: MIT">
</p>

<p align="center">
  <code>Faithfulness 0.96 · Numeric exact-match 0.80 · Refusal recall 1.00 · Hallucinated citations 0</code><br>
  <sub>Dev-set results (AMZN, GOOGL) · frozen held-out test results (AAPL, MSFT, NVDA) reported below</sub>
</p>

<p align="center">
  <a href="⟨YOUR_LIVE_APP_URL⟩">
    <img src="docs/demo.gif" alt="FilingIQ returning a cited figure, then refusing an unanswerable question" width="90%">
  </a>
</p>

---

## What it does

FilingIQ answers questions across 15 real 10-K filings — **Apple, Microsoft, NVIDIA, Amazon,
and Alphabet**, three fiscal years each. The core design decision: **numbers and prose fail
differently, so they're served by different engines.** A deterministic router classifies each
question and dispatches it:

- **Numeric questions** hit a **structured SQL store** built from validated table parses —
  the answer is an exact cell with cell-level provenance, never LLM-generated.
- **Conceptual questions** hit a **hybrid text engine** (dense + BM25 retrieval, reciprocal-rank
  fusion, cross-encoder reranking) that returns cited, grounded prose.
- **"Why did X change?"** questions fuse both: the exact figures plus the filing's own explanation.
- **Unanswerable questions get refused.** In finance, "that's not in these filings" is a feature.

| Ask | FilingIQ returns |
|---|---|
| *"What was Microsoft's total revenue in fiscal 2024?"* | **$245,122M** — exact cell from the income statement, cited to the filing |
| *"What are NVIDIA's main supply-chain risks?"* | Grounded prose from Item 1A, with inline citations |
| *"What is Apple's current stock price?"* | A refusal — it isn't in the filings |

<img width="1175" height="600" alt="image" src="https://github.com/user-attachments/assets/d8637a02-ee84-4033-933e-0e63149acecb" />
