"""
router.py — the Traffic Cop (rule-based intent router).

Reads a question and decides which engine(s) should answer it:

  * TABLE  — a direct request for a specific figure ("What was Alphabet's net income
             in 2024?"). Answer from Engine 2 (SQL) — the exact cell, no guessing.
  * TEXT   — a qualitative / conceptual question ("What are the supply-chain risks?").
             Answer from Engine 1 (semantic vector search).
  * HYBRID — needs BOTH numbers and explanation ("Why did net income fall in 2025?").
             Pull the exact figures from Engine 2 AND the explanatory prose from
             Engine 1, and let the LLM reconcile them.

Rule-based on purpose (see CLAUDE.md §6): the company universe is closed, the metric
vocabulary is a small standardized map, and a deterministic classifier is free,
instant, and — crucially — **unit-testable and measured** (routing accuracy is a
reported metric). It reuses `entity.py` (company/year) and the SAME standardized
line-item map used at ingestion, so classification can't drift from the data.

The router only decides INTENT. Graceful degradation (e.g. a TABLE question with no
resolvable cell) is handled by the executor in generate.py, which falls back to TEXT.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from retrieval.entity import extract_entities
from retrieval.table_lookup import detect_metrics

TEXT, TABLE, HYBRID = "TEXT", "TABLE", "HYBRID"

# CAUSAL intent — the answer needs a NARRATIVE explanation, not just a number
# ("why did X fall", "what drove the increase"). Metric + causal => HYBRID. Note: a
# bare comparison ("compare A and B revenue", "how did revenue change 2023-2025") is
# NOT causal — it just wants the numbers across scopes, which the multi-scope TABLE
# lookup already handles. Stems have no trailing \b so "explain"/"contributed" match.
_CAUSAL = re.compile(
    r"\b(why|reason|driver|drove|because|due to|explain|contribut|"
    r"attribut|impact|affect|cause)", re.I)

# Qualitative topics that live in prose, not tables — force TEXT even if a stray
# metric word appears.
_QUALITATIVE = re.compile(
    r"\b(risk|strateg|competit|litigat|regulat|outlook|guidance|describe|discuss|"
    r"opinion|concern|worried|governance|lawsuit|acquisition|management'?s? view)\b", re.I)

# DESCRIPTIVE phrasing ("what does X say about ...", "describe", "disclose") asks for
# prose CONTENT, so a metric noun in it ("advertising revenue") is incidental, not a
# value request. These go to TEXT even when a metric matches — unless the question is
# also causal ("why did revenue change"), which still needs the numbers (HYBRID).
_DESCRIPTIVE = re.compile(
    r"\b(say[s]? about|describe|discuss|disclose|identif|characteri|"
    r"cite[s]?|view on|talk about|mention|address(es)?)\b", re.I)


@dataclass
class RouterDecision:
    route: str
    companies: list[str] = field(default_factory=list)
    years: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    needs_explanation: bool = False

    def __str__(self) -> str:
        return (f"{self.route} (companies={self.companies} years={self.years} "
                f"metrics={self.metrics} explain={self.needs_explanation})")


def classify(question: str) -> RouterDecision:
    """Route a question to TEXT / TABLE / HYBRID (deterministic)."""
    ent = extract_entities(question)
    companies = sorted(ent["companies"])
    years = sorted(ent["years"])
    metrics = detect_metrics(question)
    needs_explanation = bool(_CAUSAL.search(question))
    qualitative = bool(_QUALITATIVE.search(question))

    base = dict(companies=companies, years=years, metrics=metrics,
                needs_explanation=needs_explanation)

    # Descriptive "what does X say about ..." wants prose, not a figure -> TEXT
    # (unless it's causal, which still needs the numbers via HYBRID below).
    if _DESCRIPTIVE.search(question) and not needs_explanation:
        return RouterDecision(TEXT, **base)

    # A qualitative topic with no concrete metric is always TEXT.
    if qualitative and not metrics:
        return RouterDecision(TEXT, **base)

    if metrics:
        # numbers + a "why/how/compare" narrative -> need both engines
        if needs_explanation or qualitative:
            return RouterDecision(HYBRID, **base)
        # a plain "what was X" with a resolvable company+year -> exact SQL cell
        if companies and years:
            return RouterDecision(TABLE, **base)
        # a metric with no scope to pin a cell -> let the text engine handle it
        return RouterDecision(TEXT, **base)

    # no metric at all -> conceptual / qualitative
    return RouterDecision(TEXT, **base)


if __name__ == "__main__":
    for q in [
        "What was Alphabet's net income in fiscal 2024?",           # TABLE
        "What are the main supply-chain risks?",                    # TEXT
        "Why did Amazon's operating income change in 2024?",        # HYBRID
        "Compare Amazon and Alphabet revenue in 2024.",            # HYBRID (compare)
        "How does Alphabet describe its Other Bets segment?",       # TEXT
        "What was Amazon's total net sales in fiscal 2024?",        # TABLE
    ]:
        print(f"{classify(q)!s:<95}  <- {q}")
