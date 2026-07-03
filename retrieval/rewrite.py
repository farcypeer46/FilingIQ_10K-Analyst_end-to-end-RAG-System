"""
rewrite.py — conversational memory as a QUERY REWRITER (not an answer source).

A user in a chat asks follow-ups: "What was Apple's revenue in 2024?" then "how does
that compare to Microsoft?". The second question is meaningless on its own — "that"
and the implied metric/year live in the prior turn. This module resolves those
references into ONE self-contained standalone question, using the recent conversation.

CRITICAL DESIGN RULE (see CLAUDE.md §4): memory rewrites the QUESTION only; it NEVER
supplies the answer. The rewritten question then flows through the SAME dual-engine
pipeline (Traffic Cop -> SQL/text -> grounded, cited, refuses). So every guarantee is
preserved — numbers still come from Engine 2, refusal still fires, 0 hallucination —
and the LLM never answers from conversation history.

An already-standalone question is returned UNCHANGED, so single-turn use is unaffected.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("rewrite")

REWRITE_SYSTEM = """You rewrite a user's latest question into a SINGLE self-contained question for a search system over SEC 10-K filings, using the conversation so far.

Rules:
- Resolve every reference (it, that, they, their, the company, this metric, "compare to X") to EXPLICIT company names, metrics, and fiscal years drawn from the conversation.
- RECENCY: when the latest question names NO company (e.g. "who is better?", "what about their risks?"), resolve it to the company/companies from the MOST RECENT prior turn — not older turns.
- Subjective asks ("who is better/best/stronger") have no single filing answer; rewrite them into a neutral retrieval over the SAME companies (e.g. "better risk management" -> "risk management disclosures for <those companies>").
- Preserve the user's intent exactly. Do NOT answer the question. Do NOT add facts.
- If the latest question is already fully self-contained, return it UNCHANGED.
- Output ONLY the rewritten question — no preamble, no quotes.

Examples:
History:
Q: What was Apple's total revenue in fiscal 2024?
A: Apple's total net sales in fiscal 2024 were $391,035 million.
Latest: How does that compare to Microsoft?
Rewrite: Compare Apple's and Microsoft's total revenue in fiscal 2024.

History:
Q: What was NVIDIA's net income in fiscal 2025?
A: NVIDIA's net income in fiscal 2025 was $72,880 million.
Latest: And their revenue?
Rewrite: What was NVIDIA's total revenue in fiscal 2025?

History:
Q: What are Apple's main risk factors?
A: ...
Latest: What was Microsoft's operating income in fiscal 2024?
Rewrite: What was Microsoft's operating income in fiscal 2024?

History:
Q: Compare Apple's and Alphabet's net income in fiscal 2024.
A: Apple $93,736M; Alphabet $100,118M.
Q: Which company, Alphabet or Amazon, earned more from their cloud service in fiscal 2024?
A: Amazon AWS $80.1B; Alphabet Google Cloud $39.5B.
Latest: who has better risk management
Rewrite: What are the risk management disclosures for Amazon and Alphabet?"""


def _format_history(history: list[dict], max_turns: int = 3) -> str:
    """Render the last few turns as 'Q: ... / A: ...' (answers truncated)."""
    lines = []
    for turn in history[-max_turns:]:
        q = (turn.get("question") or "").strip()
        a = (turn.get("answer") or "").strip()
        if q:
            lines.append(f"Q: {q}")
            lines.append(f"A: {a[:200]}")
    return "\n".join(lines)


def rewrite_followup(question: str, history: list[dict] | None,
                     client, model: str) -> str:
    """Return a self-contained version of `question` given recent `history`.

    Returns the question UNCHANGED when there's no history or on any error (fail-safe:
    a bad rewrite must never break single-turn behaviour).
    """
    if not history:
        return question
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=120,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": f"History:\n{_format_history(history)}\n"
                                            f"Latest: {question}\nRewrite:"},
            ],
        )
        rewritten = resp.choices[0].message.content.strip().strip('"')
    except Exception as e:
        logger.warning("rewrite failed (%s); using original question", e)
        return question

    if not rewritten:
        return question
    if rewritten != question:
        logger.info("Rewrote follow-up: %r -> %r", question, rewritten)
    return rewritten
