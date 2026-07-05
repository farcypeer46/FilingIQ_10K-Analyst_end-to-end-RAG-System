"""
Hybrid retrieval over the SEC-filings index.

Pipeline (per query):
  dense (Chroma / bge-small)  +  sparse (BM25)   run independently
  -> Reciprocal Rank Fusion (RRF) into one candidate list
  -> cross-encoder rerank
  -> top-k chunks (with company/year/section metadata for citation)

Design notes (see CLAUDE.md):
  * The query is embedded with the SAME model used at ingestion (bge-small), and
    gets the bge retrieval instruction prefix — bge-v1.5 recommends prefixing the
    QUERY (not the passages) for short-query -> passage search.
  * BM25 is built in-memory at startup from the Chroma documents themselves, so the
    sparse and dense indexes are guaranteed to describe the SAME chunks (one source
    of truth, no drift after a rebuild). For ~2.2k short chunks this is sub-second.
  * Optional metadata filter (company / fiscal_year) is applied to BOTH sides so
    retrieval never mixes companies or years — the citation-integrity requirement.
"""
from __future__ import annotations

import logging
import re

import chromadb
from rank_bm25 import BM25Okapi
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer, CrossEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("retrieve")

# --- config (swappable in one place) ---
EMBED_MODEL = "BAAI/bge-small-en-v1.5"            # must match ingestion
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # CPU-fast; swap to
#               "BAAI/bge-reranker-base" for higher precision if latency allows
COLLECTION = "sec_filings"
CHROMA_PATH = "./chroma_db"
# bge-v1.5 query instruction (applied to the query only, not the stored passages)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RRF_K = 60                                        # standard RRF constant


def _tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens. Keeps things like 'item7a', '10k', tickers."""
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    """Loads the index + models once, then serves many queries."""

    def __init__(self, chroma_path: str = CHROMA_PATH, collection: str = COLLECTION):
        # Loading is slow (embedder + reranker + BM25 over the whole corpus), so
        # log each stage — in a served app this is the startup/readiness signal.
        logger.info("Opening Chroma collection '%s'...", collection)
        self.db = chromadb.PersistentClient(path=chroma_path)
        self.col = self.db.get_collection(collection)

        # pull every chunk once: drives BM25 and the id -> (text, metadata) lookup
        dump = self.col.get(include=["documents", "metadatas"])
        self.ids = dump["ids"]
        self.docs = dump["documents"]
        self.metas = dump["metadatas"]
        self.by_id = {
            cid: {"text": d, "metadata": m}
            for cid, d, m in zip(self.ids, self.docs, self.metas)
        }

        # sparse index, built from the same documents Chroma stores
        logger.info("Building BM25 index over %d chunks...", len(self.docs))
        self.bm25 = BM25Okapi([_tokenize(d) for d in self.docs])

        # models (loaded once; embedder must match ingestion)
        logger.info("Loading models (embedder + reranker)...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.reranker = CrossEncoder(RERANK_MODEL)
        logger.info("Retriever ready: %d chunks searchable.", len(self.ids))

    # --- the two retrievers, each returns a ranked list of chunk ids ---

    @staticmethod
    def _chroma_where(where: dict | None) -> dict | None:
        """Translate a flat {field: value} filter into Chroma's expected form.

        Chroma requires an explicit operator for multi-field filters — a bare
        multi-key dict is rejected — so we wrap 2+ conditions in `$and`.
        """
        if not where:
            return None
        if len(where) == 1:
            return where
        return {"$and": [{k: v} for k, v in where.items()]}

    def _dense(self, query: str, n: int, where: dict | None) -> list[str]:
        emb = self.embedder.encode(QUERY_PREFIX + query).tolist()
        res = self.col.query(
            query_embeddings=[emb], n_results=n,
            where=self._chroma_where(where), include=[],   # ids come back regardless
        )
        return res["ids"][0]

    def _sparse(self, query: str, n: int, where: dict | None) -> list[str]:
        scores = self.bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in order:
            if where and not self._meta_match(self.metas[i], where):
                continue
            if scores[i] <= 0:        # no term overlap -> not a real BM25 hit
                break
            out.append(self.ids[i])
            if len(out) >= n:
                break
        return out

    @staticmethod
    def _meta_match(meta: dict, where: dict) -> bool:
        """Minimal equality filter mirroring Chroma's `where` for the sparse side."""
        return all(meta.get(k) == v for k, v in where.items())

    # --- fusion ---

    @staticmethod
    def _rrf(*ranked_lists: list[str]) -> list[str]:
        """Reciprocal Rank Fusion: score = sum 1/(k + rank). Standard hybrid fuse."""
        scores = {}
        for ranked in ranked_lists:
            for rank, cid in enumerate(ranked):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        return sorted(scores, key=scores.get, reverse=True)

    # --- public API ---

    def search(
        self, query: str, top_k: int = 5, candidates: int = 30, where: dict | None = None,
        hybrid: bool = True, rerank: bool = True,
    ) -> list[dict]:
        """Return top_k reranked chunks as dicts: {id, text, metadata, rerank_score}.

        `candidates` is how many to pull from each retriever before reranking.
        `where` is an optional metadata filter, e.g. {"company": "AAPL",
        "fiscal_year": "2025"} — applied to both dense and sparse sides.
        `hybrid=False` runs dense-only (no BM25/RRF); `rerank=False` skips the
        cross-encoder — both exposed as UI toggles so the demo doubles as a
        retrieval playground (rerank_score is None when reranking is off).
        """
        dense = self._dense(query, candidates, where)
        sparse = self._sparse(query, candidates, where) if hybrid else []
        fused = self._rrf(dense, sparse) if hybrid else dense
        if not fused:
            return []

        if rerank:
            pairs = [(query, self.by_id[cid]["text"]) for cid in fused]
            scores = self.reranker.predict(pairs)
            order = sorted(zip(fused, scores), key=lambda x: x[1], reverse=True)
        else:
            order = [(cid, None) for cid in fused]  # keep fusion/dense order, no scores

        return [
            {
                "id": cid,
                "text": self.by_id[cid]["text"],
                "metadata": self.by_id[cid]["metadata"],
                "rerank_score": (float(score) if score is not None else None),
            }
            for cid, score in order[:top_k]
        ]

    def search_scoped(
        self, query: str, scopes: list[dict], top_k: int = 5, per_scope: int = 4,
        hybrid: bool = True, rerank: bool = True,
    ) -> list[dict]:
        """Run one scoped search per (company/year) filter and merge (dedup by id).

        Used by self-query filtering. Each scope is a flat metadata filter derived
        from the question (e.g. {"company": "AMZN", "fiscal_year": "2024"}). Running
        the SAME query separately under each scope guarantees balanced coverage — a
        two-company comparison keeps evidence for BOTH, instead of one company
        hogging all top_k slots as it would under a single OR'd filter.

          * no scopes  -> plain unfiltered search (this can only add precision).
          * one scope  -> a single filtered search.
          * many scopes-> per_scope hits under each filter, merged.
        """
        if not scopes:
            return self.search(query, top_k=top_k, hybrid=hybrid, rerank=rerank)
        if len(scopes) == 1:
            return self.search(query, top_k=top_k, where=scopes[0], hybrid=hybrid, rerank=rerank)

        seen: set[str] = set()
        merged: list[dict] = []
        for scope in scopes:
            for hit in self.search(query, top_k=per_scope, where=scope,
                                   hybrid=hybrid, rerank=rerank):
                if hit["id"] in seen:
                    continue
                seen.add(hit["id"])
                merged.append(hit)
        return merged


if __name__ == "__main__":
    # smoke test: prove the full hybrid path works end to end
    r = HybridRetriever()
    print(f"Loaded {len(r.ids)} chunks.\n")
    for q in [
        "What are the main risk factors related to supply chain?",
        "How much did the company spend on research and development?",
    ]:
        print(f"Q: {q}")
        for hit in r.search(q, top_k=3):
            m = hit["metadata"]
            preview = hit["text"][:120].replace("\n", " ")
            print(f"  [{m['company']} FY{m['fiscal_year']} {m['section']}] "
                  f"score={hit['rerank_score']:.2f}  {preview}...")
        print()
