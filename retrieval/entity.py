"""
entity.py — self-query entity extraction.

Reads a natural-language question and pulls out which companies and fiscal years
it refers to, then turns those into retrieval scope filters. This is the
"self-query" step: the system decides its OWN metadata filter from the question,
instead of the user picking a company/year from a dropdown.

Why rule-based (not an LLM call): the company universe is a fixed, closed set of
five tickers and the years are a small known range, so a deterministic alias map +
year regex is more reliable, free, and instant — and it's trivially unit-testable
(important for the CI gate). It also never adds latency to the request path.

Two responsibilities:
  * extract_entities(question) -> {"companies": {...}, "years": {...}}
  * build_scopes(question)     -> list of flat {company, fiscal_year} filters,
                                   one per (company, year) the question implies.

A multi-target question ("compare Amazon and Alphabet in 2024", "Google 2023 to
2025") yields MULTIPLE scopes; the retriever runs one scoped search per scope and
merges, so every target is represented and companies/years never collide. A
single-target question yields one scope; a question that names no company/year
yields NO scope (empty list) -> the caller falls back to an unfiltered search, so
this can only ADD precision, never break what already works.
"""

from __future__ import annotations

import re

# The corpus is a closed set — map every plausible surface form (name, ticker,
# unambiguous brand) to the ticker used in chunk metadata. Only strong, unambiguous
# brand tokens are included to avoid false positives (e.g. no bare "cloud").
COMPANY_ALIASES: dict[str, str] = {
    "apple": "AAPL", "aapl": "AAPL", "iphone": "AAPL",
    "microsoft": "MSFT", "msft": "MSFT", "azure": "MSFT",
    "nvidia": "NVDA", "nvda": "NVDA",
    "amazon": "AMZN", "amzn": "AMZN", "aws": "AMZN",
    "alphabet": "GOOGL", "google": "GOOGL", "googl": "GOOGL", "goog": "GOOGL",
}

# Fiscal years actually present in the index (AAPL/AMZN/GOOGL/MSFT: 2023-2025;
# NVDA: 2024-2026). A detected year outside this set is ignored.
VALID_YEARS: set[str] = {"2023", "2024", "2025", "2026"}

# Words that signal a span ("2023 to 2025") — used to fill the implied middle
# years, e.g. 2023 to 2025 -> {2023, 2024, 2025}.
_RANGE = r"(?:to|through|thru|until|-|–|—)"

# Cap the scope fan-out so a pathological question can't explode into dozens of
# filtered searches.
MAX_SCOPES = 12


def extract_entities(question: str) -> dict:
    """Return the companies and fiscal years a question refers to.

    {"companies": {"AMZN", ...}, "years": {"2024", ...}} — either may be empty.
    """
    q = question.lower()
    tokens = set(re.findall(r"[a-z0-9]+", q))

    companies = {ticker for alias, ticker in COMPANY_ALIASES.items() if alias in tokens}

    years = {y for y in re.findall(r"\b(20\d{2})\b", q) if y in VALID_YEARS}
    years |= _expand_ranges(q, years)

    return {"companies": companies, "years": years}


def _expand_ranges(q: str, years: set[str]) -> set[str]:
    """If the text spans two years ('2023 to 2025'), fill the implied middle."""
    if len(years) < 2:
        return set()
    if not re.search(rf"\b20\d{{2}}\s*{_RANGE}\s*20\d{{2}}\b", q):
        return set()
    lo, hi = min(int(y) for y in years), max(int(y) for y in years)
    return {str(y) for y in range(lo, hi + 1) if str(y) in VALID_YEARS}


def build_scopes(question: str) -> list[dict]:
    """Turn a question into retrieval scope filters (one per company x year).

    * both companies and years present -> cross product of flat
      {"company": C, "fiscal_year": Y} filters.
    * only companies -> one {"company": C} per company.
    * only years     -> one {"fiscal_year": Y} per year.
    * neither        -> [] (caller does a normal, unfiltered search).
    """
    ent = extract_entities(question)
    companies = sorted(ent["companies"])
    years = sorted(ent["years"])

    if companies and years:
        scopes = [{"company": c, "fiscal_year": y} for c in companies for y in years]
    elif companies:
        scopes = [{"company": c} for c in companies]
    elif years:
        scopes = [{"fiscal_year": y} for y in years]
    else:
        scopes = []

    return scopes[:MAX_SCOPES]


if __name__ == "__main__":
    # smoke test: show the scopes each question type produces
    for q in [
        "What was Alphabet's total revenue in fiscal 2024?",           # 1 company, 1 year
        "Compare Amazon's and Alphabet's net income in 2024.",         # 2 companies, 1 year
        "How did Google's revenue change from 2023 to 2025?",          # 1 company, year range
        "Which grew faster from 2023 to 2024 — AWS or Google Cloud?",  # 2 companies, range
        "What are the main risk factors for the business?",            # neither -> no scope
        "What were Amazon's risk factors?",                            # company only
    ]:
        print(f"Q: {q}")
        print(f"   entities: {extract_entities(q)}")
        print(f"   scopes:   {build_scopes(q)}\n")
