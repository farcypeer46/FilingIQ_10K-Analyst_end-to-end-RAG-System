"""
evaluate.py — Step 6: held-out evaluation of the RAG pipeline.

Runs the WHOLE pipeline (retrieve -> generate) over the hand-written golden set on
the HELD-OUT filings (AMZN, GOOGL) — companies the system was never tuned against —
and reports honest numbers.

Honesty rules baked in:
  * Held-out only (AMZN/GOOGL, plus BOTH for cross-company). A guardrail rejects any
    dev-set company.
  * NO metadata filter is passed — each question goes in as raw text and the system
    must find the right company/year itself (the true "user asks a question"
    scenario). Passing company/year would inflate the baseline and hide the future
    self-query upgrade's gains.

Two scoring layers:
  A. Deterministic (free, always runs): refusal precision/recall, numeric exact-match,
     retrieval Hit@k / MRR (vs labeled `relevant`), hallucinated-citation rate,
     per-category breakdown, latency.
  B. RAGAS LLM-judge (optional; needs `pip install ragas langchain-openai`):
     faithfulness, answer relevancy, context precision/recall on answerable rows.

Cost note: generation runs once per question; RAGAS adds several judge calls per
answerable question. Keep your OpenAI spend cap set.

Run from the repo root with the venv active:
    (venv) $ python evaluation/evaluate.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict

import yaml
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generation.generate import GroundedGenerator  # noqa: E402

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("evaluate")

GOLDEN_PATH = "evaluation/golden_set.yaml"
RESULTS_PATH = "evaluation/results.json"
JUDGE_MODEL = "gpt-4o-mini"
JUDGE_EMBED = "text-embedding-3-small"
TOP_K = 5


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #

# Company guardrail per golden set — enforces set purity (dev vs frozen test).
ALLOWED_COMPANIES = {
    "evaluation/golden_set.yaml": {"AMZN", "GOOGL", "BOTH"},        # dev
    "evaluation/golden_set_test.yaml": {"AAPL", "MSFT", "NVDA", "BOTH"},  # frozen TEST
}


def load_golden(path: str = GOLDEN_PATH) -> list[dict]:
    items = yaml.safe_load(open(path))["questions"]
    allowed = ALLOWED_COMPANIES.get(path, {"AMZN", "GOOGL", "BOTH"})
    bad = {q["company"] for q in items} - allowed
    if bad:
        raise ValueError(f"{path} must contain only {allowed}; found {bad}")
    return items


def run_pipeline(items: list[dict], top_k: int = TOP_K) -> list[dict]:
    """Run the full dual-engine pipeline once per question and collect outputs.

    The Traffic Cop routes each question itself (TEXT/TABLE/HYBRID) and the system
    extracts its OWN company/year — no golden-set oracle filter is ever passed in.
    """
    gen = GroundedGenerator()
    logger.info("Scoring %d questions (top_k=%d, dual-engine router)...", len(items), top_k)
    rows = []
    for i, q in enumerate(items, 1):
        t0 = time.perf_counter()
        out = gen.answer(q["question"], top_k=top_k)
        rows.append({
            "question": q["question"],
            "type": q["type"],
            "reference": q["ground_truth"],
            "should_refuse": q["ground_truth"] == "REFUSE",
            "relevant": q.get("relevant", "") or "",
            "response": out["answer"],
            "refused": out["refused"],
            "route": out["route"],                       # executed engine (post-degradation)
            "routed_intent": out.get("routed_intent"),   # the router's classification
            "expected_route": q.get("expected_route"),   # labeled ground truth (answerable only)
            "hallucinated": bool(out["hallucinated_citations"]),
            "latency_ms": round((time.perf_counter() - t0) * 1000),
            "contexts": [c["text"] for c in out["contexts"]],
            "context_meta": [c["metadata"] for c in out["contexts"]],
        })
        logger.info("  [%d/%d] %s refused=%s", i, len(items), q["type"], out["refused"])
    return rows


# --------------------------------------------------------------------------- #
# A. Deterministic scoring
# --------------------------------------------------------------------------- #

def _parse_relevant(spec: str) -> list[tuple[str, str, str]]:
    out = []
    for part in spec.split(";"):
        m = re.match(r"\s*([A-Z]+)\s+FY(\d{4})\s+(\w+)", part.strip())
        if m:
            out.append((m.group(1), m.group(2), m.group(3)))
    return out


def _primary_number(text: str) -> str | None:
    nums = [n.replace(",", "") for n in re.findall(r"\d[\d,]{3,}", text)]
    return max(nums, key=len) if nums else None


def _numeric_correct(reference: str, response: str) -> bool:
    target = _primary_number(reference)
    return bool(target) and target in response.replace(",", "")


def _retrieval(specs, context_meta) -> dict:
    def first_rank(strict: bool) -> int:
        for i, m in enumerate(context_meta, 1):
            for co, yr, sec in specs:
                if m.get("company") == co and m.get("fiscal_year") == yr:
                    if not strict or m.get("section") == sec:
                        return i
        return 0
    rs, rl = first_rank(True), first_rank(False)
    return {"hit_strict": int(bool(rs)), "hit_lenient": int(bool(rl)),
            "mrr_strict": (1.0 / rs) if rs else 0.0}


def _mean(vals) -> float:
    vals = list(vals)
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def score_deterministic(rows: list[dict]) -> dict:
    # attach per-row deterministic scores
    for r in rows:
        if r["type"] == "numeric" and not r["should_refuse"]:
            r["numeric_correct"] = _numeric_correct(r["reference"], r["response"])
        specs = _parse_relevant(r["relevant"])
        if specs and not r["should_refuse"]:
            r.update(_retrieval(specs, r["context_meta"]))

    tp = sum(1 for r in rows if r["should_refuse"] and r["refused"])
    fn = sum(1 for r in rows if r["should_refuse"] and not r["refused"])
    fp = sum(1 for r in rows if not r["should_refuse"] and r["refused"])
    tn = sum(1 for r in rows if not r["should_refuse"] and not r["refused"])

    retr = [r for r in rows if "hit_strict" in r]
    numeric = [r for r in rows if "numeric_correct" in r]
    answered = [r for r in rows if not r["should_refuse"]]

    by_cat = defaultdict(list)
    for r in retr:
        by_cat[r["type"]].append(r["hit_lenient"])

    return {
        "refusal": {
            "precision": round(tp / (tp + fp), 3) if (tp + fp) else 0.0,
            "recall": round(tp / (tp + fn), 3) if (tp + fn) else 0.0,
            "accuracy": round((tp + tn) / len(rows), 3),
            "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        },
        "retrieval": {
            "hit@k_strict": _mean(r["hit_strict"] for r in retr),
            "hit@k_lenient": _mean(r["hit_lenient"] for r in retr),
            "mrr_strict": _mean(r["mrr_strict"] for r in retr),
            "n": len(retr),
        },
        "numeric_accuracy": _mean(r["numeric_correct"] for r in numeric),
        "hallucinated_citation_rate": _mean(r["hallucinated"] for r in answered),
        "hit@k_lenient_by_type": {k: _mean(v) for k, v in by_cat.items()},
        "routing": {
            # accuracy = router INTENT vs labeled expected_route (isolates the router
            # from Engine-2 recall gaps, which only affect the EXECUTED route).
            "accuracy": _mean(
                r.get("routed_intent") == r["expected_route"]
                for r in rows if r.get("expected_route")
            ),
            "n": sum(1 for r in rows if r.get("expected_route")),
            "intent_distribution": dict(Counter(r.get("routed_intent", "?") for r in rows)),
            "executed_distribution": dict(Counter(r.get("route", "?") for r in rows)),
        },
        "latency_ms_median": sorted(r["latency_ms"] for r in rows)[len(rows) // 2],
    }


# --------------------------------------------------------------------------- #
# B. RAGAS scoring (optional)
# --------------------------------------------------------------------------- #

def score_ragas(rows: list[dict]) -> dict:
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import (
            Faithfulness, ResponseRelevancy,
            LLMContextPrecisionWithReference, LLMContextRecall,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError:
        logger.warning("RAGAS not installed — skipping LLM-judge metrics "
                       "(`pip install ragas langchain-openai` to enable).")
        return {"skipped": "ragas not installed"}

    judged = [r for r in rows if not r["should_refuse"] and not r["refused"]]
    if not judged:
        return {"note": "no answerable, non-refused rows to score"}

    dataset = EvaluationDataset.from_list([{
        "user_input": r["question"], "response": r["response"],
        "retrieved_contexts": r["contexts"], "reference": r["reference"],
    } for r in judged])
    llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=JUDGE_EMBED))
    logger.info("Running RAGAS judge on %d answerable rows...", len(judged))
    result = evaluate(dataset=dataset, metrics=[
        Faithfulness(), ResponseRelevancy(),
        LLMContextPrecisionWithReference(), LLMContextRecall(),
    ], llm=llm, embeddings=emb)
    df = result.to_pandas()
    return {c: round(float(df[c].mean()), 3) for c in df.columns if df[c].dtype.kind in "fi"}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def print_report(summary: dict, top_k: int) -> None:
    d, r = summary["deterministic"], summary["deterministic"]["refusal"]
    print("\n" + "=" * 64)
    print(f"  DUAL-ENGINE  (top_k={top_k}, held-out AMZN/GOOGL)")
    print("=" * 64)
    print(f"  Refusal    precision={r['precision']} recall={r['recall']} acc={r['accuracy']}  {r['confusion']}")
    rt = d["retrieval"]
    print(f"  Retrieval  Hit@k strict={rt['hit@k_strict']} lenient={rt['hit@k_lenient']} MRR={rt['mrr_strict']} (n={rt['n']})")
    print(f"  Numeric    exact-match={d['numeric_accuracy']}")
    print(f"  Citations  hallucinated-rate={d['hallucinated_citation_rate']}")
    rg = d["routing"]
    print(f"  Routing    intent-accuracy={rg['accuracy']} (n={rg['n']})  intent={rg['intent_distribution']}  executed={rg['executed_distribution']}")
    print(f"  Latency    median={d['latency_ms_median']}ms")
    print(f"  Hit@k lenient by type: {d['hit@k_lenient_by_type']}")
    if summary.get("ragas") and "skipped" not in summary["ragas"]:
        print(f"  RAGAS      {summary['ragas']}")
    print("=" * 64 + "\n")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=GOLDEN_PATH, help="golden set path (dev or frozen test)")
    ap.add_argument("--out", default=None, help="results file (default: results.json)")
    args = ap.parse_args()
    out_path = args.out or RESULTS_PATH

    items = load_golden(args.golden)
    print(f"Golden set: {len(items)} questions from {args.golden}.")
    rows = run_pipeline(items)

    summary = {"deterministic": score_deterministic(rows), "ragas": score_ragas(rows)}
    print_report(summary, TOP_K)

    with open(out_path, "w") as f:
        json.dump({"top_k": TOP_K, "architecture": "dual-engine-router",
                   "summary": summary, "rows": rows}, f, indent=2)
    logger.info("Wrote per-question detail to %s", out_path)


if __name__ == "__main__":
    main()
