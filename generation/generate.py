"""
Grounded answer generation over the retrieved excerpts.

Flow (per question):
  retrieve top-k chunks (hybrid + rerank)
  -> build a numbered, source-labeled context block
  -> LLM answers using ONLY that context, citing claims with [n]
  -> if the context doesn't support an answer, the model refuses (exact sentinel)

Design notes (see CLAUDE.md):
  * The prompt is loaded from prompts/*.yaml (versioned), never hardcoded here —
    so an eval number always maps to a known, frozen prompt.
  * Citations are enforced structurally: each excerpt is numbered [n] and mapped
    back to its company/year/section, so the caller can render real citations and
    verify the model only cited excerpts it was actually given.
  * Refusal is a verbatim sentinel string, making "refuse when unsupported"
    machine-checkable rather than a vibe.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import time

import yaml
from dotenv import load_dotenv
from openai import OpenAI

# repo root on path so this runs as `python generation/generate.py` from the root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.retrieve import HybridRetriever  # noqa: E402
from retrieval.entity import build_scopes  # noqa: E402
from retrieval.table_lookup import TableEngine, format_facts  # noqa: E402
from retrieval.router import classify, TEXT, TABLE, HYBRID  # noqa: E402,F401
from retrieval.rewrite import rewrite_followup  # noqa: E402
from retrieval.cache import SemanticCache  # noqa: E402

# Load OPENAI_API_KEY from .env for dev; override=True so .env wins over a stale
# key already exported in the shell. No-op in prod/CI (no .env there).
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("generate")

PROMPT_PATH = "prompts/answer_v2.yaml"
REQUEST_TIMEOUT = 30.0  # seconds per LLM call
MAX_RETRIES = 2         # SDK retries transient errors (429/5xx) with backoff
REQUIRED_CFG = {"model", "temperature", "max_tokens", "system", "user", "refusal_text"}

# lazy client: only created when we actually call the LLM (keeps imports keyless)
_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Return a shared OpenAI client configured for resilient serving."""
    global _client
    if _client is None:
        _client = OpenAI(timeout=REQUEST_TIMEOUT, max_retries=MAX_RETRIES)
    return _client


def _norm(text: str) -> str:
    """Normalize for refusal comparison: lowercase, drop non-alphanumerics."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _cited_numbers(text: str, n_excerpts: int) -> tuple[set[int], set[int]]:
    """Parse [n] markers the model emitted. Returns (valid, hallucinated).

    valid       = citation numbers that map to an excerpt we actually provided.
    hallucinated = numbers the model cited that were never in the context — a
                   grounding failure this system is meant to surface, not hide.
    """
    used = {int(m) for m in re.findall(r"\[(\d+)\]", text)}
    valid = {n for n in used if 1 <= n <= n_excerpts}
    return valid, used - valid


class GroundedGenerator:
    """Ties retrieval + a versioned prompt into cited, refusing answers."""

    def __init__(self, retriever: HybridRetriever | None = None, prompt_path: str = PROMPT_PATH):
        with open(prompt_path) as f:
            self.cfg = yaml.safe_load(f)

        # Fail fast at startup with a clear message if the prompt config is
        # malformed — never discover a missing key mid-request in production.
        missing = REQUIRED_CFG - self.cfg.keys()
        if missing:
            raise ValueError(f"{prompt_path} is missing required keys: {sorted(missing)}")

        self.retriever = retriever or HybridRetriever()
        self.table_engine = TableEngine()  # Engine 2: SQLite structured number store
        self.cache = SemanticCache()       # cost layer: skip LLM on repeat questions
        self._refusal_norm = _norm(self.cfg["refusal_text"])
        logger.info("Generator ready: prompt=%s model=%s", prompt_path, self.cfg["model"])

    def _format_context(self, hits: list[dict]) -> str:
        """Number each excerpt [n] and label it with its source for citation."""
        blocks = []
        for i, h in enumerate(hits, start=1):
            m = h["metadata"]
            src = f"{m['company']} FY{m['fiscal_year']}, {m['section']}"
            blocks.append(f"[{i}] ({src})\n{h['text']}")
        return "\n\n".join(blocks)

    @staticmethod
    def _facts_to_hits(facts: list[dict]) -> list[dict]:
        """Turn structured Number-Vault facts into synthetic, citable excerpts.

        Facts are grouped by (company, year) so each synthetic excerpt has coherent
        citation metadata (section = 'financial_data'). These sit at the top of the
        context; the retrieved prose still follows and provides disambiguation.
        """
        from itertools import groupby
        facts = sorted(facts, key=lambda f: (f["company"], f["year"]))
        hits: list[dict] = []
        for (co, yr), grp in groupby(facts, key=lambda f: (f["company"], f["year"])):
            grp = list(grp)
            hits.append({
                "id": f"tablestore::{co}:{yr}",
                "text": format_facts(grp),
                "metadata": {"company": co, "fiscal_year": yr, "section": "financial_data"},
                "rerank_score": float("inf"),
            })
        return hits

    @staticmethod
    def _source(i: int, h: dict) -> dict:
        """One excerpt's citation record (number -> its verifiable source)."""
        m = h["metadata"]
        return {
            "n": i,
            "company": m["company"],
            "fiscal_year": m["fiscal_year"],
            "section": m["section"],
            "chunk_id": h["id"],
        }

    def _gather(self, question: str, top_k: int) -> tuple[list[dict], str, str]:
        """Route the question and gather evidence. Returns (hits, intent, executed).

        `intent` is the Traffic Cop's classification; `executed` is what actually ran
        (a TABLE/HYBRID intent degrades to TEXT when Engine 2 resolves no cell). We
        keep them SEPARATE so routing accuracy measures the router, not Engine 2
        recall gaps.

        TABLE  -> exact SQL facts (+ a few supporting chunks); degrades to TEXT if no cell.
        TEXT   -> self-query scoped hybrid retrieval.
        HYBRID -> exact SQL facts AND scoped text retrieval (numbers + reasons).
        """
        intent = classify(question).route
        scopes = build_scopes(question)
        facts = self.table_engine.lookup(question) if intent in (TABLE, HYBRID) else []

        if intent == TABLE and facts:
            executed, k = TABLE, 3          # exact number in hand; a few chunks corroborate
        elif intent == HYBRID and facts:
            executed, k = HYBRID, top_k     # numbers + full prose
        else:
            executed, k = TEXT, top_k       # TEXT, or a TABLE/HYBRID with no resolvable cell
        text_hits = self.retriever.search_scoped(question, scopes, top_k=k)

        return self._facts_to_hits(facts) + text_hits, intent, executed

    def answer(self, question: str, top_k: int = 5, history: list[dict] | None = None) -> dict:
        """Return a grounded, cited, dual-engine answer dict.

        Keys: question, rewritten_question, answer, refused, route (TEXT/TABLE/HYBRID),
        cached (bool), citations, hallucinated_citations, contexts, scopes.

        Pipeline: conversational memory rewrites a follow-up into a standalone question
        -> semantic cache (skip everything on a repeat) -> Traffic Cop routes -> Engine 2
        (SQL exact numbers) / Engine 1 (prose) / HYBRID -> grounded, cited, refuses.
        Numbers come only from Engine 2 — never invented by the LLM. Memory only rewrites
        the QUESTION; the cache is keyed on (rewritten question + company/year scope).
        """
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")

        # 1. Conversational memory: resolve a follow-up into a self-contained question.
        q = rewrite_followup(question, history, get_client(), self.cfg["model"]) if history else question

        # 2. Semantic cache (cost layer): reuse a prior answer to an equivalent question.
        q_emb = self.retriever.embedder.encode(q).tolist()
        cached = self.cache.lookup(q, q_emb)
        if cached is not None:
            return {**cached, "question": question, "rewritten_question": q, "cached": True}

        hits, intent, executed = self._gather(q, top_k)
        scopes = build_scopes(q)

        # no evidence at all -> refuse without spending an LLM call
        if not hits:
            logger.info("No evidence (route=%s) -> refusing (no LLM call). q=%r", executed, q)
            payload = {
                "question": q, "answer": self.cfg["refusal_text"], "refused": True,
                "citations": [], "hallucinated_citations": [], "contexts": [],
                "route": executed, "routed_intent": intent, "scopes": scopes,
            }
            self.cache.store(q, q_emb, payload)
            return {**payload, "question": question, "rewritten_question": q, "cached": False}

        context = self._format_context(hits)
        messages = [
            {"role": "system", "content": self.cfg["system"]},
            {"role": "user", "content": self.cfg["user"].format(question=q, context=context)},
        ]

        # Fail loud (after SDK retries) so the caller handles it — never return a
        # silently broken answer to a financial question.
        t0 = time.perf_counter()
        try:
            resp = get_client().chat.completions.create(
                model=self.cfg["model"],
                messages=messages,
                temperature=self.cfg["temperature"],
                max_tokens=self.cfg["max_tokens"],
            )
        except Exception:
            logger.exception("LLM call failed for q=%r", q)
            raise
        latency_ms = (time.perf_counter() - t0) * 1000

        text = resp.choices[0].message.content.strip()
        refused = _norm(text) == self._refusal_norm

        # Verify citations: keep only sources the model actually referenced, and
        # surface any [n] it cited that was never in the context.
        valid, hallucinated = _cited_numbers(text, len(hits))
        citations = [] if refused else [self._source(i, hits[i - 1]) for i in sorted(valid)]
        if hallucinated:
            logger.warning("Hallucinated citation(s) %s for q=%r", sorted(hallucinated), q)

        logger.info("Answered q=%r route=%s(intent=%s) refused=%s cited=%d latency=%.0fms",
                    q, executed, intent, refused, len(citations), latency_ms)
        payload = {
            "question": q, "answer": text, "refused": refused,
            "citations": citations, "hallucinated_citations": sorted(hallucinated),
            "contexts": hits, "route": executed, "routed_intent": intent, "scopes": scopes,
        }
        self.cache.store(q, q_emb, payload)
        return {**payload, "question": question, "rewritten_question": q, "cached": False}


if __name__ == "__main__":
    # smoke test: one answerable question, one that should be refused
    gen = GroundedGenerator()
    for q in [
        "How much did Microsoft spend on research and development?",
        "What is the CEO's home address?",  # not in any filing -> must refuse
    ]:
        out = gen.answer(q, top_k=5)
        print(f"Q: {q}")
        print(f"refused={out['refused']}")
        print(out["answer"])
        if out["citations"]:
            print("cited:", ", ".join(
                f"[{c['n']}] {c['company']} FY{c['fiscal_year']} {c['section']}"
                for c in out["citations"]))
        if out["hallucinated_citations"]:
            print("!! hallucinated citations:", out["hallucinated_citations"])
        print("-" * 70)
