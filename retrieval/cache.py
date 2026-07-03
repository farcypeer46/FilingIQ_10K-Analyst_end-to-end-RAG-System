"""
cache.py — semantic answer cache (the cost layer).

The whole pipeline is $0 except the generation LLM call. This cache skips that call
(and retrieval + rerank) when a semantically-equivalent question was already answered:
~2 s -> ~20 ms and $0 on a hit.

FINANCE SAFETY — a two-factor key. A naive cosine cache is DANGEROUS here: "net income
2024" and "net income 2023" are ~0.98 similar but have completely different answers, so
similarity alone would serve the wrong number. A hit therefore requires BOTH:
  1. cosine similarity >= threshold (conservative, 0.97), AND
  2. an EXACT match on the extracted (company, year) scope (from entity.build_scopes).
So "Apple 2023" and "Apple 2024" sit in different buckets and can never collide.

INVALIDATION is by process lifetime: the cache lives in memory and is empty on startup.
Re-ingesting the corpus means a redeploy = a fresh process = an empty cache, so a stale
(pre-reindex) answer can never be served. Simple and correct for a served app.

The cache is keyed on the REWRITTEN (standalone) question, so follow-ups like "how about
Microsoft?" are never cached by their raw, context-dependent text.
"""

from __future__ import annotations

import logging
import math

from retrieval.entity import build_scopes

logger = logging.getLogger("cache")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _scope_key(question: str) -> frozenset:
    """The (company, year) scope a question pins — the exact-match half of the key."""
    return frozenset(
        (s.get("company"), s.get("fiscal_year")) for s in build_scopes(question)
    )


class SemanticCache:
    """In-memory semantic answer cache with a (similarity + scope) two-factor key."""

    def __init__(self, threshold: float = 0.97):
        self.threshold = threshold
        self._entries: list[dict] = []   # {emb, scope, payload}
        self.hits = 0
        self.misses = 0

    def lookup(self, question: str, embedding: list[float]) -> dict | None:
        """Return a cached answer payload for an equivalent question, or None."""
        scope = _scope_key(question)
        best, best_sim = None, self.threshold
        for e in self._entries:
            if e["scope"] != scope:          # exact-scope gate first (cheap, safe)
                continue
            sim = _cosine(embedding, e["emb"])
            if sim >= best_sim:
                best, best_sim = e, sim
        if best is not None:
            self.hits += 1
            logger.info("cache HIT (sim=%.3f) for %r", best_sim, question)
            return best["payload"]
        self.misses += 1
        return None

    def store(self, question: str, embedding: list[float], payload: dict) -> None:
        self._entries.append({"emb": embedding, "scope": _scope_key(question),
                              "payload": payload})

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"entries": len(self._entries), "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0}
