"""
fetch_filings.py — Step 1 of the ingestion pipeline.

Download the most recent 10-K filings for our fixed corpus from SEC EDGAR.

Corpus (see CLAUDE.md §5):
    dev set   : AAPL, MSFT, NVDA
    held-out  : AMZN, GOOGL   (reserved for honest evaluation)
    3 years x 5 companies = up to 15 filings, saved under ./data/

We pull EDGAR *HTML* (not PDF) because HTML preserves table row/column
structure that later extraction depends on. `download_details=True` saves
the clean `primary-document.html` alongside the raw submission.

Run from the repo root with the venv active:
    (venv) $ python ingestion/fetch_filings.py
"""

from __future__ import annotations

import logging
from pathlib import Path

# pyrefly: ignore [missing-import]
from sec_edgar_downloader import Downloader

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# SEC requires a descriptive User-Agent on every request so they can contact
# you about abusive traffic. sec-edgar-downloader builds it from these values.
COMPANY_NAME = "Salman Farcy"
EMAIL = "psalmanf6@gmail.com"

# Kept as separate lists (not one flat list) so the dev/held-out split stays
# visible and self-documenting — evaluation must never tune on held-out data.
DEV_COMPANIES = ["AAPL", "MSFT", "NVDA"]
HELD_OUT_COMPANIES = ["AMZN", "GOOGL"]
ALL_COMPANIES = DEV_COMPANIES + HELD_OUT_COMPANIES

FILING_TYPE = "10-K"
YEARS = 3  # most recent N 10-Ks per company
DATA_DIR = Path("./data")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("fetch_filings")


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #

def fetch_company(dl: Downloader, ticker: str, *, years: int = YEARS) -> int:
    """Download the `years` most recent 10-Ks for one ticker.

    Returns the number of filings downloaded. A failure for one company is
    logged and swallowed (returns 0) so a single bad ticker never aborts the
    whole corpus pull. Re-downloads are cheap: the library skips filings that
    already exist on disk, so this script is safe to re-run (idempotent).
    """
    logger.info("Fetching %d most recent %s filing(s) for %s...", years, FILING_TYPE, ticker)
    try:
        # limit=years -> most recent N filings, newest first.
        # download_details=True -> also save the clean primary-document.html.
        count = dl.get(FILING_TYPE, ticker, limit=years, download_details=True)
    except Exception:  # network / rate-limit / unknown ticker
        logger.exception("  %s: download failed", ticker)
        return 0

    logger.info("  %s: downloaded %d filing(s)", ticker, count)
    return count


def fetch_corpus(companies: list[str] = ALL_COMPANIES, *, years: int = YEARS) -> int:
    """Download the full corpus and return the total number of filings pulled."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dl = Downloader(COMPANY_NAME, EMAIL, str(DATA_DIR))

    total = sum(fetch_company(dl, ticker, years=years) for ticker in companies)

    logger.info("Done. Downloaded %d filing(s) total into %s/sec-edgar-filings/", total, DATA_DIR)
    return total


if __name__ == "__main__":
    fetch_corpus()
