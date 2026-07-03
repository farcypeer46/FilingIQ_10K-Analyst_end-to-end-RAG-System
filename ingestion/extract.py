"""
extract.py — Step 2 of the ingestion pipeline.

Turn each raw 10-K HTML filing into a flat list of tagged "pieces". A piece is
the smallest unit we later chunk + embed, and it carries the metadata that makes
citations and year-aware retrieval possible:

    {company, fiscal_year, section, type, text}

Two kinds of piece are produced per filing:
  - "prose"  : running text, grouped by SEC Item section (Business, Risk
               Factors, MD&A, ...).
  - "table"  : a financial table rendered as intact Markdown (row/col structure
               preserved), tagged with the section it sits in.

Design decisions (see CLAUDE.md §9):
  - Read EDGAR *HTML*, not PDF, so table structure survives.
  - Strip the hidden XBRL metadata block first, or get_text() returns machine junk.
  - Section detection uses strict "Item N + real title" patterns and takes the
    LAST match, which skips the table-of-contents and cross-reference noise.
  - Never split a table: each <table> becomes exactly one piece.

Output: data/extracted.json (a list of piece dicts).

Run from the repo root with the venv active:
    (venv) $ python ingestion/extract.py
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from collections import Counter
from io import StringIO
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

FILINGS_GLOB = "sec-edgar-filings/*/10-K/*/primary-document.html"
DATA_DIR = Path("data")
OUTPUT_PATH = DATA_DIR / "extracted.json"

# Strict patterns: "Item N" immediately followed by its real title. This skips
# cross-references ("see Item 1A of...") and table-of-contents page-number rows.
SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("business",             re.compile(r"Item\s+1[\.\s ]+Business", re.I)),
    ("risk_factors",         re.compile(r"Item\s+1A[\.\s ]+Risk\s+Factors", re.I)),
    ("mda",                  re.compile(r"Item\s+7[\.\s ]+Management", re.I)),
    ("market_risk",          re.compile(r"Item\s+7A[\.\s ]+Quantitative", re.I)),
    ("financial_statements", re.compile(r"Item\s+8[\.\s ]+Financial\s+Statements", re.I)),
]

# Thresholds (chars). Named so the intent is obvious and tuning is one-line.
MIN_PROSE_CHARS = 100   # drop tiny prose fragments (headers, stray labels)
MIN_TABLE_CHARS = 20    # drop trivial/empty tables
LEAD_OTHER_CHARS = 200  # only keep pre-first-section text if it's substantial
TABLE_LOCATE_PREFIX = 60  # chars of table text used to locate it in the filing

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("extract")


# --------------------------------------------------------------------------- #
# Section detection
# --------------------------------------------------------------------------- #

def get_fiscal_year(html: str) -> str:
    """Best-effort fiscal year from the first YYYYMMDD-style date in the header.

    NOTE: this is a heuristic on the leading 5 KB of the document. It is good
    enough for our large-cap corpus but is not a rigorous fiscal-period parse.
    """
    m = re.search(r"(20\d{2})\d{2}\d{2}", html[:5000])
    return m.group(1) if m else "unknown"


def find_sections(text: str) -> list[tuple[int, str]]:
    """Return sorted (start_offset, section_tag) boundaries found in `text`.

    Uses the LAST match of each pattern: the real section header appears after
    the table of contents, so the last hit is the one we want.
    """
    bounds: list[tuple[int, str]] = []
    for tag, pat in SECTION_PATTERNS:
        matches = list(pat.finditer(text))
        if matches:
            bounds.append((matches[-1].start(), tag))
    bounds.sort()
    return bounds


def section_at(pos: int, bounds: list[tuple[int, str]]) -> str:
    """Which section does character offset `pos` fall in? 'other' if before all."""
    cur = "other"
    for start, tag in bounds:
        if pos >= start:
            cur = tag
        else:
            break
    return cur


def slice_prose(text: str, bounds: list[tuple[int, str]]) -> list[tuple[str, str]]:
    """Split `text` into (section_tag, segment) spans using section boundaries."""
    if not bounds:
        return [("other", text)]
    segs: list[tuple[str, str]] = []
    if bounds[0][0] > LEAD_OTHER_CHARS:
        segs.append(("other", text[:bounds[0][0]]))
    for i, (start, tag) in enumerate(bounds):
        end = bounds[i + 1][0] if i + 1 < len(bounds) else len(text)
        segs.append((tag, text[start:end]))
    return segs


# --------------------------------------------------------------------------- #
# Table cleaning
# --------------------------------------------------------------------------- #

def clean_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn pandas' messy `read_html` output into a compact, readable table.

    SEC HTML tables are full of colspan/rowspan artifacts: all-NaN spacer rows
    and columns, the currency symbol split into its own column from the number
    it belongs to, and header cells duplicated across the columns they span
    (e.g. "2023" appearing twice). This collapses all of that so a row reads
    "Total revenue | 211915 | 198270 | 168088" — with the label<->number
    association preserved (see CLAUDE.md §9.1).
    """
    df = raw.dropna(axis=1, how="all").dropna(axis=0, how="all")
    # normalise cells to clean strings; blank out NaN; drop float suffix ("2023.0")
    df = df.astype(str).apply(lambda c: c.map(lambda x: "" if x in ("nan", "NaN", "None") else x.strip()))
    df = df.apply(lambda c: c.map(lambda x: x[:-2] if x.endswith(".0") and x[:-2].isdigit() else x))

    # drop columns that are just a bare currency/percent symbol — these are the
    # "$" cells split off from their number into an adjacent <td>.
    def _is_symbol_col(col: pd.Series) -> bool:
        vals = [v for v in col if v != ""]
        return bool(vals) and sum(v in ("$", "%") for v in vals) >= max(1, len(vals) // 2)

    df = df.loc[:, [not _is_symbol_col(df[c]) for c in df.columns]]
    df = df.loc[:, ~df.T.duplicated()]      # drop columns duplicated by colspan
    df = df.loc[:, (df != "").any()]        # drop columns emptied by the above
    df = df[(df != "").any(axis=1)]         # drop rows emptied by the above
    df.columns = [""] * df.shape[1]         # integer col names are meaningless noise
    return df


# --------------------------------------------------------------------------- #
# Per-filing extraction
# --------------------------------------------------------------------------- #

def _extract_tables(soup: BeautifulSoup, company: str, fy: str) -> list[dict]:
    """Extract every <table> as an intact Markdown piece, then remove it from
    the soup so the prose pass won't re-grab the same text.

    Section is located by finding the table's text in the FULL document text
    (tables still present) — we must do this before the tables are stripped.
    """
    full_text = soup.get_text(separator=" ", strip=True)
    tbounds = find_sections(full_text)

    pieces: list[dict] = []
    for tbl in soup.find_all("table"):
        ttext = tbl.get_text(separator=" ", strip=True)
        pos = full_text.find(ttext[:TABLE_LOCATE_PREFIX]) if len(ttext) >= 10 else -1
        try:
            df = clean_table(pd.read_html(StringIO(str(tbl)))[0])
            md = df.to_markdown(index=False)
        except Exception:
            # Not a real data table (layout table, malformed HTML) — discard.
            tbl.decompose()
            continue

        if md and len(md) > MIN_TABLE_CHARS:
            pieces.append({
                "company": company,
                "fiscal_year": fy,
                "section": section_at(pos, tbounds) if pos >= 0 else "other",
                "type": "table",
                "text": md,
            })
        tbl.decompose()  # remove so the prose pass won't re-grab it
    return pieces


def _extract_prose(soup: BeautifulSoup, company: str, fy: str) -> list[dict]:
    """Extract running text (tables already removed), grouped by section."""
    prose = soup.get_text(separator=" ", strip=True)
    pieces: list[dict] = []
    for tag, seg in slice_prose(prose, find_sections(prose)):
        seg = seg.strip()
        if len(seg) > MIN_PROSE_CHARS:
            pieces.append({
                "company": company,
                "fiscal_year": fy,
                "section": tag,
                "type": "prose",
                "text": seg,
            })
    return pieces


def extract_filing(path: Path, company: str) -> list[dict]:
    """Extract all tagged pieces (tables + prose) from a single filing."""
    html = path.read_text(encoding="utf-8")
    fy = get_fiscal_year(html)
    soup = BeautifulSoup(html, "lxml")

    # Drop hidden XBRL metadata blocks (display:none) — pure machine junk that
    # would otherwise pollute get_text(). Must happen before any extraction.
    for hidden in soup.find_all(style=lambda v: v and "display:none" in v.replace(" ", "")):
        hidden.decompose()

    # Tables first (they mutate the soup by removing themselves), then prose.
    return _extract_tables(soup, company, fy) + _extract_prose(soup, company, fy)


# --------------------------------------------------------------------------- #
# Corpus driver
# --------------------------------------------------------------------------- #

def extract_all() -> list[dict]:
    """Extract every filing under data/ and write the combined result to JSON."""
    paths = sorted(DATA_DIR.glob(FILINGS_GLOB))
    logger.info("Processing %d filing(s).", len(paths))

    all_pieces: list[dict] = []
    for path in paths:
        # data/sec-edgar-filings/<COMPANY>/10-K/<accession>/primary-document.html
        company = path.parts[2]
        pieces = extract_filing(path, company)

        secs = dict(Counter(p["section"] for p in pieces))
        fy = pieces[0]["fiscal_year"] if pieces else "?"
        n_tables = sum(1 for p in pieces if p["type"] == "table")
        n_chars = sum(len(p["text"]) for p in pieces)
        logger.info(
            "  %s FY%s: %d pieces, %d tables, %s chars | %s",
            company, fy, len(pieces), n_tables, f"{n_chars:,}", secs,
        )
        all_pieces.extend(pieces)

    OUTPUT_PATH.write_text(json.dumps(all_pieces))
    logger.info("Total: %d pieces saved to %s", len(all_pieces), OUTPUT_PATH)
    return all_pieces


if __name__ == "__main__":
    extract_all()
