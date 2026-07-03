"""
build_table_store.py — load the structured "Number Vault" (Engine 2) into SQLite.

Reads the extracted table pieces, parses each ONCE through the canonical parser
(ingestion/tables.py), and writes the resulting (row, year) records into a real
relational table `financial_metrics` in data/tables.db.

Why SQLite (not a flat file): numbers deserve a store you can query by coordinates —
`SELECT metric_value FROM financial_metrics WHERE ticker=? AND fiscal_year=? AND
line_item_standardized=?`. That gives exact lookup, multi-year "stitching", and
cell-level provenance (which filing/section/table each value came from), with zero
extra dependencies (sqlite3 ships with Python).

This is the SAME canonical parse used to build the searchable captions in
build_index.py, so the Table Engine and the Text Engine can never disagree.

Run from the repo root with the venv active:
    (venv) $ python ingestion/build_table_store.py
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from ingestion.tables import canonicalize_table, to_records

DATA_DIR = Path("data")
PIECES_PATH = DATA_DIR / "extracted.json"
DB_PATH = DATA_DIR / "tables.db"
TABLE = "financial_metrics"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("table_store")

SCHEMA = f"""
DROP TABLE IF EXISTS {TABLE};
CREATE TABLE {TABLE} (
    entry_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker                  TEXT NOT NULL,
    fiscal_year             TEXT NOT NULL,
    fiscal_period           TEXT NOT NULL,      -- 'FY' for 10-K (kept for extensibility)
    line_item_standardized  TEXT,               -- canonical code, NULL if not a headline
    line_item_raw           TEXT NOT NULL,      -- label as it appeared in the filing
    metric_value            REAL NOT NULL,
    unit                    TEXT,               -- millions | thousands | billions | NULL
    source_company_fy       TEXT NOT NULL,      -- provenance: which filing
    source_section          TEXT NOT NULL,      -- provenance: which Item section
    source_table_id         TEXT NOT NULL       -- provenance: which table
);
CREATE INDEX idx_lookup ON {TABLE} (ticker, fiscal_year, line_item_standardized);
CREATE INDEX idx_raw    ON {TABLE} (ticker, fiscal_year);
"""

_COLS = ("ticker", "fiscal_year", "fiscal_period", "line_item_standardized",
         "line_item_raw", "metric_value", "unit",
         "source_company_fy", "source_section", "source_table_id")


def build_records(pieces: list[dict]) -> list[dict]:
    """Canonical-parse every table piece into (row, year) records, de-duplicated.

    The same (ticker, year, raw_label, value) can appear in multiple filings (a 2023
    value is restated in the 2024 filing); we keep one, preferring the record whose
    provenance filing year matches the value year (the authoritative source).
    """
    best: dict[tuple, dict] = {}
    n_parsed = 0
    for i, p in enumerate(pieces):
        if p.get("type") != "table":
            continue
        canon = canonicalize_table(p["text"])
        if not canon:
            continue
        n_parsed += 1
        table_id = f"{p['company']}-{p['fiscal_year']}-{i}"
        for r in to_records(canon, p["company"], p["fiscal_year"], p["section"], table_id):
            key = (r["ticker"], r["fiscal_year"], r["line_item_raw"].lower(), r["metric_value"])
            # prefer the filing whose own year == the value's year (authoritative)
            authoritative = r["source_company_fy"].endswith(f"FY{r['fiscal_year']}")
            if key not in best or (authoritative and not best[key]["_auth"]):
                r["_auth"] = authoritative
                best[key] = r
    records = list(best.values())
    for r in records:
        r.pop("_auth", None)
    logger.info("Parsed %d tables -> %d unique records.", n_parsed, len(records))
    return records


def write_db(records: list[dict], db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.executemany(
            f"INSERT INTO {TABLE} ({','.join(_COLS)}) VALUES ({','.join('?' * len(_COLS))})",
            [tuple(r[c] for c in _COLS) for r in records],
        )
        conn.commit()
        n = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
        std = conn.execute(
            f"SELECT COUNT(*) FROM {TABLE} WHERE line_item_standardized IS NOT NULL"
        ).fetchone()[0]
        logger.info("Wrote %d rows to %s (%d with a standardized line item).", n, db_path, std)
    finally:
        conn.close()


def main() -> None:
    pieces = json.loads(PIECES_PATH.read_text())
    write_db(build_records(pieces))


if __name__ == "__main__":
    main()
