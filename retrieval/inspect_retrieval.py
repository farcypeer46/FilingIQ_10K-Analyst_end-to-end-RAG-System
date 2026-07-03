"""
inspect_retrieval.py — interactive tool to MANUALLY evaluate retrieval quality.

(Named inspect_retrieval, not inspect, because a module named `inspect` shadows
the Python standard library and breaks imports.)

Type a question and see the full top-K chunks the retriever returns, each with:
  - its rerank score (the cross-encoder's relevance judgement),
  - its citation metadata [company / fiscal_year / section / type],
  - PROVENANCE: whether it came from dense (bge) [D#rank], sparse (BM25) [S#rank],
    or both — this is how you judge whether hybrid retrieval is actually earning
    its keep vs either retriever alone,
  - the full chunk text (so you can verify the answer is really in there).

Filters: append `company=NVDA` and/or `year=2025` to any question to scope it,
e.g.   research and development spend company=MSFT year=2024

Commands:  :k N  set top-K      :chars N  set preview length (0 = full text)
           :q    quit

Run from the repo root with the venv active:
    (venv) $ python retrieval/inspect_retrieval.py
    (venv) $ python retrieval/inspect_retrieval.py "supply chain risks" company=NVDA  # one-shot
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running the script directly from the repo root (put root on the path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.retrieve import HybridRetriever  # noqa: E402

CANDIDATES = 30  # how many to pull from each retriever before fusion/rerank
FILTER_KEYS = {"company": "company", "year": "fiscal_year", "fiscal_year": "fiscal_year"}


def parse_line(line: str) -> tuple[str, dict]:
    """Split a raw input line into (question, where-filter).

    Trailing `key=value` tokens become the metadata filter; everything else is
    the question. e.g. "r&d spend company=MSFT year=2024" -> ("r&d spend",
    {"company": "MSFT", "fiscal_year": "2024"}).
    """
    words, where = [], {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            if k.lower() in FILTER_KEYS:
                where[FILTER_KEYS[k.lower()]] = v.upper() if k.lower() == "company" else v
                continue
        words.append(tok)
    return " ".join(words), where


def show(r: HybridRetriever, question: str, where: dict, top_k: int, chars: int) -> None:
    """Run the full hybrid pipeline and print top-K with provenance."""
    # Reproduce search() but keep the intermediate lists so we can show where
    # each final chunk came from (dense vs sparse) and how far the reranker moved it.
    dense = r._dense(question, CANDIDATES, where or None)
    sparse = r._sparse(question, CANDIDATES, where or None)
    fused = r._rrf(dense, sparse)
    if not fused:
        print("  (no candidates — filter too narrow or no term/vector overlap)\n")
        return

    scores = r.reranker.predict([(question, r.by_id[c]["text"]) for c in fused])
    ranked = sorted(zip(fused, scores), key=lambda x: x[1], reverse=True)[:top_k]

    d_rank = {c: i + 1 for i, c in enumerate(dense)}
    s_rank = {c: i + 1 for i, c in enumerate(sparse)}

    print(f"  filter: {where or 'none'}   |   dense hits: {len(dense)}   sparse hits: {len(sparse)}\n")
    for pos, (cid, score) in enumerate(ranked, 1):
        m = r.by_id[cid]["metadata"]
        prov = []
        if cid in d_rank:
            prov.append(f"D#{d_rank[cid]}")
        if cid in s_rank:
            prov.append(f"S#{s_rank[cid]}")
        text = r.by_id[cid]["text"]
        body = text if chars == 0 else text[:chars].strip() + ("…" if len(text) > chars else "")
        print(f"  #{pos}  rerank={score:+.2f}  [{m['company']} FY{m['fiscal_year']} "
              f"{m['section']}/{m['type']}]  ({'+'.join(prov)})")
        print("      " + body.replace("\n", "\n      "))
        print()


def main() -> None:
    r = HybridRetriever()
    top_k, chars = 5, 400

    # One-shot mode: everything after the script name is a single query.
    if len(sys.argv) > 1:
        question, where = parse_line(" ".join(sys.argv[1:]))
        print(f"\nQ: {question}")
        show(r, question, where, top_k, chars)
        return

    print("\nManual retrieval inspector. Type a question. Commands: q quit  :k N  :chars N  (or Ctrl+C)")
    while True:
        try:
            line = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in (":q", "q", "quit", "exit"):
            break
        if line.startswith(":k "):
            top_k = int(line[3:]); print(f"  top_k = {top_k}"); continue
        if line.startswith(":chars "):
            chars = int(line[7:]); print(f"  preview = {chars or 'full'}"); continue

        question, where = parse_line(line)
        show(r, question, where, top_k, chars)


if __name__ == "__main__":
    main()
