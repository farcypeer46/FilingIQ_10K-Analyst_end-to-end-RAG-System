"""
table_lookup.py — the read interface to Engine 2 (the SQLite Table Engine).

Given a question, resolve the company/year (via entity.py) and the metric it asks
about (via the SAME standardized line-item map used at ingestion), then fetch the
EXACT value(s) with a real SQL query against data/tables.db.

HIGH-PRECISION by design: a fact is returned ONLY when a (ticker, year, standardized
line item) resolves to exactly ONE distinct value. If several rows match (e.g.
segment-level "operating income": AWS / North America / consolidated), it stays
silent and lets the router fall back to HYBRID/TEXT — returning an ambiguous number
is the one failure a grounding system must avoid.

Because the metric synonyms come from `ingestion.tables.STANDARD_LINE_ITEMS` — the
same map used to standardize the store at ingestion — the query side and the storage
side can never drift.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path

from retrieval.entity import build_scopes

logger = logging.getLogger("table_lookup")

DB_PATH = Path("data/tables.db")

# QUERY-side synonyms: how a USER phrases a metric ("revenue", "top line", "R&D").
# Deliberately BROADER than the ingestion-side STANDARD_LINE_ITEMS map (which must
# match table row LABELS precisely to avoid picking up component rows). Both resolve
# to the same standardized codes, so query and storage stay aligned on the codes.
QUERY_METRIC_SYNONYMS: dict[str, list[str]] = {
    "TOTAL_REVENUE":       ["revenue", "revenues", "net sales", "total sales", "top line", "turnover"],
    "COST_OF_REVENUE":     ["cost of revenue", "cost of sales", "cost of goods"],
    "GROSS_PROFIT":        ["gross profit", "gross margin"],
    "RND_EXPENSE":         ["research and development", "r&d", "rnd", "r and d"],
    "OPERATING_INCOME":    ["operating income", "operating profit", "income from operations",
                            "lose money", "lost money", "losing money", "profitable", "profitability"],
    "NET_INCOME":          ["net income", "net earnings", "net profit", "bottom line"],
    "OPERATING_CASH_FLOW": ["operating cash flow", "cash flow from operations",
                            "cash from operations", "cash provided by operating"],
    "TOTAL_ASSETS":        ["total assets"],
    "TOTAL_LIABILITIES":   ["total liabilities"],
}


def detect_metrics(question: str) -> list[str]:
    """Return the standardized line-item codes a question refers to (may be empty)."""
    q = question.lower()
    codes: list[str] = []
    for code, phrases in QUERY_METRIC_SYNONYMS.items():
        if any(re.search(rf"\b{re.escape(p)}\b", q) for p in phrases):
            codes.append(code)
    return codes


class TableEngine:
    """Loads the SQLite Table Engine once and serves exact metric lookups."""

    def __init__(self, db_path: Path = DB_PATH):
        # read-only shared connection; check_same_thread=False for the served app
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True,
                                    check_same_thread=False)
        n = self.conn.execute("SELECT COUNT(*) FROM financial_metrics").fetchone()[0]
        logger.info("TableEngine ready: %d rows in %s.", n, db_path)

    def query(self, ticker: str, year: str, code: str) -> list[sqlite3.Row]:
        cur = self.conn.execute(
            "SELECT metric_value, line_item_raw, unit, source_company_fy, source_section "
            "FROM financial_metrics WHERE ticker=? AND fiscal_year=? AND line_item_standardized=?",
            (ticker, year, code),
        )
        return cur.fetchall()

    def lookup(self, question: str) -> list[dict]:
        """Return unambiguous exact-value facts for the question (may be empty).

        Each fact: {company, year, metric, row_label, value, unit, source}. Empty if
        the question names no company/year, no known metric, or the metric is
        ambiguous (multiple distinct values).
        """
        codes = detect_metrics(question)
        if not codes:
            return []

        facts: list[dict] = []
        seen: set[tuple] = set()
        for scope in build_scopes(question):
            company, year = scope.get("company"), scope.get("fiscal_year")
            if not (company and year):
                continue  # need both coordinates for an exact cell
            for code in codes:
                key = (company, year, code)
                if key in seen:
                    continue
                rows = self.query(company, year, code)
                distinct = {round(r[0], 2) for r in rows}
                if len(distinct) != 1:
                    continue  # 0 or ambiguous (segment breakdowns) -> stay silent
                seen.add(key)
                r = rows[0]
                facts.append({
                    "company": company, "year": year, "metric": code,
                    "row_label": r[1], "value": r[0], "unit": r[2],
                    "source": f"{r[3]}, {r[4]}",
                })
        return facts


def format_facts(facts: list[dict]) -> str:
    """Render facts as one context excerpt the generator can cite."""
    lines = []
    for f in facts:
        unit = f" {f['unit']}" if f.get("unit") else ""
        lines.append(f"{f['company']} FY{f['year']} — {f['row_label']}: {f['value']:,.0f}{unit}")
    return "Structured financial data (exact values from the filing tables):\n" + "\n".join(lines)
