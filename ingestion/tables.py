"""
tables.py — canonical table parsing (the single source of truth for every table).

A SEC table is parsed ONCE here into a validated canonical grid, and BOTH downstream
views are derived from it:
  * structured records  -> the SQLite Table Engine (exact numbers)
  * a deterministic caption + the clean markdown -> the Text Engine (retrieval)

Because both views come from the same parse, they can never disagree — and, crucially,
NO number is ever produced by an LLM. A wrong number in the store is worse than a
missing one, so parsing is HIGH-PRECISION: a row's cells are emitted only when its
numeric cells align 1:1 with the detected year columns; anything ambiguous is dropped.

Public API:
  canonicalize_table(md)                 -> CanonicalTable | None
  to_records(canon, company, fy, section, table_id) -> list[record dict]
  render_caption(canon, company, fy, section)       -> str
  standardize_line_item(raw_label)       -> str | None
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Plausible fiscal years in the corpus (filings cover ~2021-2026). Outside this range
# a 4-digit number is a value, not a column year.
YEAR_MIN, YEAR_MAX = 2018, 2027

# Rows that share a headline keyword but are a different KIND of number (ratios,
# per-share, share counts) — excluded from standardization so they can't create false
# ambiguity in the Table Engine.
EXCLUDE_LABEL_TERMS = ("per share", "per diluted", "diluted", "basic", "margin",
                       "percent", "%", "effective tax", " rate", "shares outstanding",
                       "weighted-average")

# --------------------------------------------------------------------------- #
# Standardized line items ("Smart Synonyms" at ingestion)
# --------------------------------------------------------------------------- #
# canonical code -> row-label keywords, MOST-SPECIFIC FIRST. A raw label maps to the
# first code whose keyword it contains (after the per-share/ratio exclusions).
STANDARD_LINE_ITEMS: list[tuple[str, list[str]]] = [
    ("TOTAL_REVENUE",       ["total revenues", "total net sales", "total revenue"]),
    ("COST_OF_REVENUE",     ["total cost of revenues", "cost of revenues", "cost of sales", "cost of revenue"]),
    ("GROSS_PROFIT",        ["gross profit"]),
    ("RND_EXPENSE",         ["research and development"]),
    ("OPERATING_INCOME",    ["total operating income", "operating income", "operating profit", "income from operations"]),
    ("NET_INCOME",          ["net income"]),
    ("OPERATING_CASH_FLOW", ["net cash provided by operating activities",
                            "cash provided by operating activities", "operating activities"]),
    ("TOTAL_ASSETS",        ["total assets"]),
    ("TOTAL_LIABILITIES",   ["total liabilities"]),
]


def standardize_line_item(raw_label: str) -> str | None:
    """Map a raw row label to a canonical code, or None if it isn't a known headline."""
    lbl = raw_label.lower()
    if any(x in lbl for x in EXCLUDE_LABEL_TERMS):
        return None
    for code, kws in STANDARD_LINE_ITEMS:
        if any(kw in lbl for kw in kws):
            return code
    return None


# --------------------------------------------------------------------------- #
# Canonical grid
# --------------------------------------------------------------------------- #

@dataclass
class CanonicalRow:
    raw_label: str
    values: dict[str, float]           # year -> value
    standardized: str | None = None


@dataclass
class CanonicalTable:
    years: list[str]                   # deduped column years, in document order
    rows: list[CanonicalRow] = field(default_factory=list)
    unit: str | None = None            # "millions" | "thousands" | "billions" | None


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def parse_markdown_rows(md: str) -> list[list[str]]:
    """Split a Markdown table into rows of trimmed cells (empties kept for alignment)."""
    rows: list[list[str]] = []
    for line in md.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and all(set(c) <= {"-", ":", " "} for c in cells):
            continue  # separator row
        rows.append(cells)
    return rows


def _as_year(cell: str) -> int | None:
    """Return the fiscal year a header cell denotes, or None.

    Handles a bare 4-digit year ("2024") AND a period-end DATE header
    ("Jan 26, 2025", "September 27, 2025", "Year Ended June 30, 2024") — for all
    corpus companies the fiscal year equals the calendar year the period ends in,
    so we take the year out of the date. Requiring a month name (letters) for the
    date case avoids misreading a stray value or footnote as a year. This is what
    lets NVIDIA's date-headed statements parse (they never use bare-year columns).
    """
    s = cell.strip()
    if re.fullmatch(r"20\d{2}", s):
        y = int(s)
    elif re.search(r"[A-Za-z]", s):          # date/period header, e.g. "Jan 26, 2025"
        m = re.search(r"\b(20\d{2})\b", s)
        if not m:
            return None
        y = int(m.group(1))
    else:
        return None
    return y if YEAR_MIN <= y <= YEAR_MAX else None


_NUM_RE = re.compile(r"^\(?-?\$?\s?[\d,]+(?:\.\d+)?\)?%?$")


def parse_number(cell: str) -> float | None:
    """Parse a financial DOLLAR cell into a float ($, commas, (parens)=negative).

    Percentage cells are rejected: the store holds dollar metrics, and a ratio row
    ("Net income 55.8%") sharing a label with the dollar row otherwise creates a
    false ambiguity that silences the exact lookup.
    """
    s = cell.strip()
    if not s or "%" in s or not _NUM_RE.match(s):
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace("$", "").replace(",", "").strip()
    if _as_year(s) is not None:        # a lone year is a header, never a value
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def _detect_year_columns(rows: list[list[str]]) -> tuple[int, list[str]] | None:
    """Find the header row with the most distinct, monotonic years (deduped)."""
    best: tuple[int, list[str]] | None = None
    for i, row in enumerate(rows):
        seq: list[int] = []
        for cell in row:
            y = _as_year(cell)
            if y is None:
                continue
            if not seq or seq[-1] != y:
                seq.append(y)          # collapse consecutive duplicate columns
        distinct = list(dict.fromkeys(seq))
        if len(distinct) >= 2 and distinct == seq:     # strictly ordered, no repeats
            if best is None or len(distinct) > len(best[1]):
                best = (i, [str(y) for y in distinct])
    return best


_UNIT_RE = re.compile(r"in\s+(thousands|millions|billions)", re.I)


def _detect_unit(md: str) -> str | None:
    m = _UNIT_RE.search(md)
    return m.group(1).lower() if m else None


def _clean_label(cell: str) -> str:
    return re.sub(r"\s+", " ", cell).strip(" .:*†()")


def canonicalize_table(md: str) -> CanonicalTable | None:
    """Parse a Markdown table into a validated canonical grid, or None if unclean.

    HIGH-PRECISION rule: a row is emitted only when its numeric cells align 1:1 with
    the detected year columns — so each value maps unambiguously to one year.
    """
    rows = parse_markdown_rows(md)
    yc = _detect_year_columns(rows)
    if not yc:
        return None
    year_row_idx, years = yc

    table = CanonicalTable(years=years, unit=_detect_unit(md))
    for row in rows[year_row_idx + 1:]:
        label = ""
        for cell in row:
            if cell and parse_number(cell) is None and _as_year(cell) is None:
                label = _clean_label(cell)
                break
        if not label or len(label) < 2:
            continue
        values = [v for v in (parse_number(c) for c in row) if v is not None]
        if len(values) != len(years):
            continue                    # ambiguous alignment -> skip rather than guess
        # Skip common-size (percentage-of-revenue) tables: their revenue/total row
        # is exactly 100.0. Such a table's cells are ratios, not dollars, and would
        # pollute the store (e.g. "Net income 55.8%" colliding with the real figure).
        if abs(values[0] - 100.0) < 0.01 and re.search(r"revenue|net sales|total", label, re.I):
            return None
        table.rows.append(CanonicalRow(
            raw_label=label,
            values={y: v for y, v in zip(years, values)},
            standardized=standardize_line_item(label),
        ))
    return table if table.rows else None


# --------------------------------------------------------------------------- #
# Derived views
# --------------------------------------------------------------------------- #

def to_records(canon: CanonicalTable, company: str, filing_fy: str,
               section: str, table_id: str) -> list[dict]:
    """Flatten the canonical grid into one record per (row, year) for the SQL engine."""
    records: list[dict] = []
    for r in canon.rows:
        for year, value in r.values.items():
            records.append({
                "ticker": company,
                "fiscal_year": year,
                "fiscal_period": "FY",
                "line_item_standardized": r.standardized,
                "line_item_raw": r.raw_label,
                "metric_value": value,
                "unit": canon.unit,
                "source_company_fy": f"{company} FY{filing_fy}",
                "source_section": section,
                "source_table_id": table_id,
            })
    return records


def render_caption(canon: CanonicalTable, company: str, filing_fy: str,
                   section: str, max_rows: int = 8) -> str:
    """Build a DETERMINISTIC, metric-rich caption from the REAL values (no LLM).

    Prefers standardized headline rows; every number is drawn straight from the
    canonical grid, so the caption can never contain a fabricated figure.
    """
    latest = canon.years[-1]
    headline = [r for r in canon.rows if r.standardized]
    others = [r for r in canon.rows if not r.standardized]
    chosen = (headline + others)[:max_rows]

    unit = f" (in {canon.unit})" if canon.unit else ""
    parts = []
    for r in chosen:
        val = r.values.get(latest)
        if val is None:
            continue
        parts.append(f"{r.raw_label} {val:,.0f} ({latest})")
    body = "; ".join(parts)
    years = ", ".join(canon.years)
    return (f"{company} FY{filing_fy} {section} table{unit}. "
            f"Years: {years}. {body}")
