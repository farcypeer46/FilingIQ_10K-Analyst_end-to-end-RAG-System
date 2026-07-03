"""
build_index.py — Step 3 of the ingestion pipeline.

Turn the tagged pieces from data/extracted.json into the searchable Text Engine:

    real tables  -> a DETERMINISTIC, metric-rich caption (built from the table's
                    OWN parsed values by ingestion/tables.py — never an LLM, so no
                    fabricated numbers) is prepended to the clean Markdown.
    prose        -> split into overlapping, token-bounded chunks.
    everything   -> embedded locally with bge-small -> stored in ChromaDB.

The embedding model (bge-small) has a hard 512-token input limit and SILENTLY
TRUNCATES anything longer. So the cardinal rule of this file is: nothing we
embed may exceed that budget. We enforce it with the model's OWN tokenizer, not
a char-count guess:
  - prose  -> RecursiveCharacterTextSplitter measured in real tokens.
  - tables -> kept whole when they fit; otherwise split by ROWS with the header
              (and the summary) repeated on every part, so each part stays a
              valid, self-describing table and no label<->number link is lost.

Other key properties:
  - Deterministic chunk IDs -> the whole build is idempotent and safe to re-run.
  - The index is a pure function of extracted.json (captions are computed from the
    tables themselves), so we rebuild the Chroma collection from scratch every run.
  - No LLM is called at all -> ingestion is $0 and needs no OPENAI_API_KEY.

Run from the repo root with the venv active:
    (venv) $ python ingestion/build_index.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from ingestion.tables import canonicalize_table, render_caption

# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DATA_DIR = Path("data")
PIECES_PATH = DATA_DIR / "extracted.json"
CHROMA_PATH = "./chroma_db"
COLLECTION = "sec_filings"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# bge-small hard limit is 512 tokens (incl. special tokens). We stay under it
# with headroom. Prose chunks target 450 tokens; a single table chunk may use up
# to 500. Overlap keeps a fact that straddles a boundary whole in one chunk.
EMBED_LIMIT = 512
PROSE_CHUNK_TOKENS = 450
PROSE_OVERLAP_TOKENS = 60
TABLE_MAX_TOKENS = 500

EMBED_BATCH = 32    # sentence-transformers encode batch
CHROMA_BATCH = 500  # Chroma limits how many items you can add at once

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("build_index")


# --------------------------------------------------------------------------- #
# Tokenizer (the source of truth for "does this fit?")
# --------------------------------------------------------------------------- #

# Load once. This is the SAME tokenizer bge-small uses to embed, so our length
# measurements match exactly what the model will see at embedding time.
_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)


def n_tokens(text: str) -> int:
    """Token count as the embedding model will actually see it (incl. specials)."""
    return len(_tokenizer.encode(text))


# --------------------------------------------------------------------------- #
# Table handling
# --------------------------------------------------------------------------- #

def is_real_table(md: str) -> bool:
    """Heuristic: a real financial table has several rows and actual numbers.

    Filters out layout/spacer tables that survived extraction but carry no data.
    """
    rows = md.count("\n")
    has_numbers = sum(c.isdigit() for c in md) > 15
    return rows >= 3 and has_numbers and len(md) > 80


def table_caption(md: str, company: str, fy: str, section: str) -> str:
    """Return a DETERMINISTIC, metric-rich caption for a table (never an LLM).

    Parses the table into its canonical grid (ingestion/tables.py) and renders a
    caption from the table's OWN values — so every number in the caption provably
    exists in the table (no fabrication). Empty string if the table can't be parsed
    cleanly, in which case the chunk is just the clean Markdown.
    """
    canon = canonicalize_table(md)
    return render_caption(canon, company, fy, section) if canon else ""


def compact_markdown(md: str) -> str:
    """Strip token-wasting padding that pandas/tabulate bakes into Markdown.

    `to_markdown` pads every cell to its column width and draws long dash runs
    in the separator row (e.g. `|:------------------:|`). Spaces are free to the
    tokenizer, but dash runs are NOT — a 21-column separator alone can cost ~900
    tokens. Collapsing the padding cuts the median table roughly in half with
    zero information loss (real cell text is single-spaced and preserved).
    """
    md = re.sub(r"-{3,}", "---", md)   # shrink separator dash runs
    md = re.sub(r" {2,}", " ", md)     # shrink cell padding
    return md


def split_table(md: str, summary: str) -> list[str]:
    """Render a table into one or more embeddable chunks under the token budget.

    The table is compacted first, then: small tables are returned whole; a table
    that would overflow bge-small's limit is split BY ROWS, with each part
    repeating (a) the table summary and (b) the Markdown header + separator rows
    — so every part is a valid, self-describing table and no number is ever
    orphaned from its column label. A pathologically wide table whose single row
    still exceeds the budget is caught by the hard fallback in build_chunks.
    """
    md = compact_markdown(md)
    prefix = f"[Table summary: {summary}]\n\n" if summary else ""
    whole = prefix + md
    if n_tokens(whole) <= TABLE_MAX_TOKENS:
        return [whole]

    lines = md.split("\n")
    if len(lines) < 3:  # no header+separator+data -> can't split sensibly
        return [whole]

    header, separator, data_rows = lines[0], lines[1], lines[2:]
    head_block = f"{prefix}{header}\n{separator}"

    parts: list[str] = []
    current: list[str] = []
    for row in data_rows:
        candidate = head_block + "\n" + "\n".join(current + [row])
        # If adding this row overflows, flush the current part first. Keep at
        # least one row per part even if a single wide row is itself over budget.
        if current and n_tokens(candidate) > TABLE_MAX_TOKENS:
            parts.append(head_block + "\n" + "\n".join(current))
            current = [row]
        else:
            current.append(row)
    if current:
        parts.append(head_block + "\n" + "\n".join(current))
    return parts


# --------------------------------------------------------------------------- #
# Chunk building
# --------------------------------------------------------------------------- #

def make_splitter() -> RecursiveCharacterTextSplitter:
    """Prose splitter whose chunk size is measured in REAL bge tokens, not chars,
    so no prose chunk can exceed the embedding model's input limit."""
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        _tokenizer,
        chunk_size=PROSE_CHUNK_TOKENS,
        chunk_overlap=PROSE_OVERLAP_TOKENS,
    )


def build_chunks(
    pieces: list[dict], splitter: RecursiveCharacterTextSplitter
) -> tuple[list[str], list[str], list[dict], int]:
    """Convert pieces into (chunks, ids, metadatas) plus a count of captioned tables.

    Tables are kept whole or row-split (table-aware) with a DETERMINISTIC caption
    prepended; prose is token-split with overlap. IDs are deterministic
    (company-fy-section-type-content_hash) so re-running yields the same index
    and true duplicates are dropped, while distinct table parts stay distinct.
    """
    chunks: list[str] = []
    ids: list[str] = []
    metadatas: list[dict] = []
    seen_ids: set[str] = set()
    n_captioned = 0

    for p in pieces:
        if p["type"] == "table":
            if not is_real_table(p["text"]):
                continue  # skip layout/junk tables
            summary = table_caption(
                p["text"], p["company"], p["fiscal_year"], p["section"]
            )
            if summary:
                n_captioned += 1
            # Structure-aware split (whole or by-rows), then a hard fallback:
            # any part a pathologically wide table left over budget is token-
            # split so NOTHING is ever silently truncated at embedding time.
            sub_chunks = []
            for part in split_table(p["text"], summary):
                if n_tokens(part) <= EMBED_LIMIT:
                    sub_chunks.append(part)
                else:
                    sub_chunks.extend(splitter.split_text(part))
        else:
            sub_chunks = splitter.split_text(p["text"])  # prose token-split

        for chunk in sub_chunks:
            # Hash the FULL chunk (not a prefix): table parts share the same
            # summary+header prefix, so a prefix hash would collide and drop them.
            cid = (
                f"{p['company']}-{p['fiscal_year']}-{p['section']}-{p['type']}-"
                + hashlib.md5(chunk.encode()).hexdigest()[:8]
            )
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            chunks.append(chunk)
            ids.append(cid)
            metadatas.append({
                "company": p["company"],
                "fiscal_year": p["fiscal_year"],
                "section": p["section"],
                "type": p["type"],
            })

    return chunks, ids, metadatas, n_captioned


# --------------------------------------------------------------------------- #
# Embedding + storage
# --------------------------------------------------------------------------- #

def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Embed all chunks locally with bge-small (free, runs on CPU)."""
    logger.info("Loading embedding model (first run downloads ~130MB)...")
    embedder = SentenceTransformer(EMBED_MODEL)
    logger.info("Embedding %d chunks (a few minutes on CPU)...", len(chunks))
    return embedder.encode(chunks, show_progress_bar=True, batch_size=EMBED_BATCH).tolist()


def store_chunks(
    ids: list[str], chunks: list[str], embeddings: list[list[float]], metadatas: list[dict]
) -> int:
    """Rebuild the Chroma collection from scratch and add all chunks.

    We drop and recreate the collection every run on purpose. The index is a
    pure function of extracted.json + cached summaries, so the build must be
    deterministic: same inputs -> same index. Appending to an existing
    collection would let chunks from an older extraction linger as stale data
    and pollute retrieval. Re-embedding is local and free (~45s).
    """
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        db.delete_collection(COLLECTION)
    except Exception:
        pass  # collection didn't exist yet (first run)
    col = db.create_collection(COLLECTION)

    for i in range(0, len(chunks), CHROMA_BATCH):
        s = slice(i, i + CHROMA_BATCH)
        col.add(ids=ids[s], documents=chunks[s], embeddings=embeddings[s], metadatas=metadatas[s])

    return col.count()


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main() -> None:
    pieces = json.loads(PIECES_PATH.read_text())
    logger.info("Loaded %d pieces.", len(pieces))

    splitter = make_splitter()
    chunks, ids, metadatas, n_captioned = build_chunks(pieces, splitter)

    logger.info("Built %d chunks. Captioned %d tables (deterministic, no LLM).",
                len(chunks), n_captioned)

    # Safety net: prove the invariant before spending time embedding.
    over = sum(1 for c in chunks if n_tokens(c) > EMBED_LIMIT)
    if over:
        logger.warning("%d chunk(s) still exceed %d tokens and will be truncated.", over, EMBED_LIMIT)
    else:
        logger.info("All %d chunks fit within the %d-token embedding limit.", len(chunks), EMBED_LIMIT)

    embeddings = embed_chunks(chunks)
    stored = store_chunks(ids, chunks, embeddings, metadatas)
    logger.info("Done. Stored %d chunks in %s", stored, CHROMA_PATH)


if __name__ == "__main__":
    main()
