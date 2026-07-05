"""
app/main.py — FastAPI service in front of the RAG engine (Step 7).

This is the thin INTERFACE layer. It owns no retrieval or generation logic; it
only: (1) loads the GroundedGenerator ONCE at startup, (2) accepts a question
over HTTP, (3) calls gen.answer(), (4) returns JSON / renders a page.

Endpoints:
  GET  /        -> minimal HTML page with a question box
  POST /ask     -> {question, company?, year?, top_k?} -> grounded answer JSON
  GET  /health  -> readiness probe (for deploy platforms / load balancers)

Why FastAPI + Uvicorn:
  - The heavy models (embedder, reranker, BM25) load once at startup and are
    reused across requests — the "load once, serve many" asymmetry.
  - Uvicorn is an ASGI server that handles many concurrent requests; while one
    request waits on the OpenAI API, others are served. Concurrency lives HERE,
    not in the engine.
  - Pydantic validates request bodies for free.

Run locally (from repo root, venv active):
    (venv) $ uvicorn app.main:app --reload
    open    http://localhost:8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from generation.generate import GroundedGenerator

logger = logging.getLogger("app")


# --------------------------------------------------------------------------- #
# Lifespan: build the engine once when the server starts, not per request.
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading GroundedGenerator (models + index)...")
    app.state.generator = GroundedGenerator()  # slow: ~3s; happens once
    logger.info("Ready to serve.")
    yield
    # (nothing to tear down; Chroma/models are in-process)


app = FastAPI(title="SEC-RAG", version="1.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Request/response schema
# --------------------------------------------------------------------------- #

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question")
    history: Optional[List[dict]] = Field(
        None, description="Prior turns [{question, answer}] for conversational follow-ups")
    top_k: int = Field(5, ge=1, le=20, description="How many excerpts to ground on")
    # Power-user controls (all optional; defaults reproduce the standard pipeline)
    route_override: Optional[str] = Field(None, description="Force route TEXT/TABLE/HYBRID; None = auto")
    hybrid: bool = Field(True, description="Hybrid dense+BM25 retrieval (False = dense-only)")
    rerank: bool = Field(True, description="Cross-encoder reranking on/off")
    company: Optional[str] = Field(None, description="Manual company scope filter (e.g. AAPL)")
    year: Optional[str] = Field(None, description="Manual fiscal-year scope filter (e.g. 2024)")
    use_cache: bool = Field(True, description="Semantic answer cache on/off")


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #

@app.get("/health")
def health() -> dict:
    """Readiness probe: confirms the engine loaded and reports index size."""
    gen = app.state.generator
    return {"status": "ok", "chunks": len(gen.retriever.ids)}


@app.get("/stats")
def stats() -> dict:
    """Semantic-cache stats (the cost story): hit rate, entries, hits/misses."""
    return app.state.generator.cache.stats()


def _inspector(out: dict) -> dict:
    """Behind-the-scenes view for the UI Inspector panel: the routing decision and
    the retrieved evidence with rerank scores. SQL facts (score = inf) are flagged
    as exact; rerank-off chunks have score None."""
    ctx = []
    for i, h in enumerate(out.get("contexts", []), start=1):
        m = h.get("metadata", {})
        sc = h.get("rerank_score")
        exact = (sc == float("inf"))
        ctx.append({
            "n": i,
            "source": "sql" if exact else "text",
            "company": m.get("company"),
            "fiscal_year": m.get("fiscal_year"),
            "section": m.get("section"),
            "score": (None if (sc is None or exact) else round(sc, 3)),
            "preview": (h.get("text") or "").strip()[:220],
        })
    return {
        "intent": out.get("routed_intent"),     # what the router classified
        "executed": out.get("route"),           # what actually ran (may degrade to TEXT)
        "cached": out.get("cached", False),
        "scopes": out.get("scopes", []),
        "contexts": ctx,
    }


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    """Answer a question via the dual-engine Traffic Cop, with conversational memory.

    Passing `history` lets a follow-up ("how about Microsoft?") be rewritten into a
    standalone question before the same grounded pipeline runs. The optional control
    fields (route_override, hybrid, rerank, company, year, use_cache) power the UI's
    Controls panel and Inspector.
    """
    route = req.route_override if req.route_override in ("TEXT", "TABLE", "HYBRID") else None
    try:
        out = app.state.generator.answer(
            req.question, top_k=req.top_k, history=req.history,
            route_override=route, hybrid=req.hybrid, rerank=req.rerank,
            company=(req.company or None), year=(req.year or None), use_cache=req.use_cache)
    except Exception:
        # The engine fails loud on LLM errors; translate that into a clean 502
        # instead of leaking a stack trace to the client.
        logger.exception("answer() failed")
        raise HTTPException(status_code=502, detail="Generation failed; please retry.")

    return {
        "question": req.question,
        "rewritten_question": out.get("rewritten_question", req.question),
        "answer": out["answer"],
        "refused": out["refused"],
        "route": out["route"],          # which engine answered (TEXT/TABLE/HYBRID)
        "cached": out.get("cached", False),
        "citations": out["citations"],  # each: {n, company, fiscal_year, section, chunk_id}
        "hallucinated_citations": out["hallucinated_citations"],
        "inspector": _inspector(out),   # routing decision + retrieved evidence + scores
    }


# --------------------------------------------------------------------------- #
# Minimal frontend (single self-contained page; no build step, no framework)
# --------------------------------------------------------------------------- #

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FilingIQ · 10-K analyst</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root{
      --bg:#0B0E14; --surface:#12161F; --surface2:#171C27; --raise:#1B2130;
      --line:#1E2530; --line2:#293141;
      --text:#E6EAF0; --muted:#8B93A7; --faint:#5A6274;
      --teal:#14B8A6; --teal2:#0FB5A6; --slate:#6B8AFD; --violet:#9A8CF7; --amber:#E0A93B;
      --teal-bg:rgba(20,184,166,.12); --slate-bg:rgba(107,138,253,.14); --violet-bg:rgba(154,140,247,.14);
      --amber-bg:rgba(224,169,59,.12);
      --mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
      --shadow:0 8px 40px rgba(0,0,0,.55);
    }
    *{box-sizing:border-box}
    html,body{height:100%}
    body{margin:0;background:var(--bg);color:var(--text);height:100vh;display:flex;flex-direction:column;
      font-family:'Inter',system-ui,sans-serif;line-height:1.55;-webkit-font-smoothing:antialiased;overflow:hidden}
    ::selection{background:var(--teal-bg)}
    button{font-family:inherit;cursor:pointer}
    a{color:var(--teal);text-decoration:none}

    /* top bar */
    header{flex:none;border-bottom:1px solid var(--line);background:rgba(11,14,20,.85);backdrop-filter:blur(8px)}
    .bar{max-width:1080px;margin:0 auto;padding:.7rem 1.1rem;display:flex;align-items:center;gap:.7rem}
    .logo{width:28px;height:28px;border-radius:8px;background:linear-gradient(135deg,var(--teal),#37d6c8);
      display:grid;place-items:center;color:#04302b;font-weight:800;font-size:13px;box-shadow:0 0 18px rgba(20,184,166,.35)}
    .brand{font-weight:800;letter-spacing:-.2px;font-size:1.05rem}
    .brand b{color:var(--teal)}
    .tag{font-size:.58rem;font-weight:700;letter-spacing:.7px;text-transform:uppercase;color:#79f0e3;
      background:var(--teal-bg);border:1px solid rgba(20,184,166,.25);padding:.16rem .42rem;border-radius:6px}
    .spacer{flex:1}
    .pill{display:flex;align-items:center;gap:.4rem;font-size:.74rem;color:var(--muted);font-weight:500;
      background:var(--surface);border:1px solid var(--line);border-radius:999px;padding:.3rem .6rem}
    .pill b{color:#79f0e3;font-weight:700;font-family:var(--mono)}
    .tbtn{display:flex;align-items:center;gap:.4rem;font-size:.8rem;font-weight:600;color:var(--muted);
      background:var(--surface);border:1px solid var(--line);border-radius:9px;padding:.4rem .7rem;transition:all .15s}
    .tbtn:hover{color:var(--text);border-color:var(--line2);background:var(--surface2)}
    .tbtn.on{color:#04302b;background:var(--teal);border-color:var(--teal)}
    .tbtn svg{width:15px;height:15px}

    /* chat */
    .scroll{flex:1;overflow-y:auto}
    .scroll::-webkit-scrollbar{width:10px}
    .scroll::-webkit-scrollbar-thumb{background:var(--line2);border-radius:20px;border:3px solid var(--bg)}
    .thread{max-width:820px;margin:0 auto;padding:1.6rem 1.1rem 1rem}

    .hero{padding:2.6rem 0 1rem;text-align:center}
    .hero h1{font-size:1.75rem;line-height:1.2;letter-spacing:-.5px;margin:.4rem 0 .5rem;font-weight:800}
    .hero h1 span{color:var(--teal)}
    .hero p{color:var(--muted);max-width:540px;margin:0 auto 1.6rem;font-size:.98rem}
    .egs{display:flex;flex-direction:column;gap:.5rem;max-width:560px;margin:0 auto;text-align:left}
    .eg{border:1px solid var(--line);background:var(--surface);border-radius:12px;padding:.72rem .9rem;
      cursor:pointer;font-size:.92rem;color:#c4ccdb;display:flex;gap:.6rem;align-items:center;transition:all .15s}
    .eg:hover{border-color:var(--line2);background:var(--surface2);color:var(--text)}
    .eg .k{flex:none;font-size:.58rem;font-weight:700;letter-spacing:.5px;text-transform:uppercase;color:var(--faint);
      border:1px solid var(--line2);border-radius:5px;padding:.14rem .38rem;font-family:var(--mono)}

    .msg{margin:1.1rem 0;display:flex;gap:.7rem}
    .msg.u{justify-content:flex-end}
    .bubble.u{background:var(--raise);border:1px solid var(--line2);color:var(--text);
      border-radius:14px 14px 4px 14px;padding:.65rem .95rem;max-width:76%;font-size:.97rem}
    .av{flex:none;width:29px;height:29px;border-radius:8px;display:grid;place-items:center;font-weight:800;font-size:12px;
      background:linear-gradient(135deg,var(--teal),#37d6c8);color:#04302b}
    .awrap{flex:1;min-width:0}
    .bubble.a{background:var(--surface);border:1px solid var(--line);border-radius:4px 14px 14px 14px;
      padding:.95rem 1.1rem;box-shadow:0 1px 0 rgba(255,255,255,.02)}
    .badges{display:flex;gap:.4rem;flex-wrap:wrap;align-items:center;margin-bottom:.6rem}
    .badge{font-size:.6rem;font-weight:700;letter-spacing:.5px;text-transform:uppercase;padding:.2rem .5rem;
      border-radius:6px;font-family:var(--mono)}
    .badge.table{color:var(--teal);background:var(--teal-bg);border:1px solid rgba(20,184,166,.28)}
    .badge.text{color:var(--slate);background:var(--slate-bg);border:1px solid rgba(107,138,253,.28)}
    .badge.hybrid{color:var(--violet);background:var(--violet-bg);border:1px solid rgba(154,140,247,.28)}
    .badge.cached,.badge.refused{color:var(--amber);background:var(--amber-bg);border:1px solid rgba(224,169,59,.28)}
    .inspect{margin-left:auto;font-size:.68rem;font-weight:600;color:var(--muted);background:transparent;
      border:1px solid var(--line);border-radius:6px;padding:.2rem .5rem;transition:all .15s}
    .inspect:hover{color:var(--teal);border-color:rgba(20,184,166,.4)}
    .rewrite{font-size:.8rem;color:var(--faint);font-style:italic;margin:-.1rem 0 .55rem}
    .answer{font-size:.98rem;color:#dbe2ee}
    .answer.muted{color:var(--amber)}
    .answer p{margin:.5rem 0}.answer p:first-child{margin-top:0}.answer p:last-child{margin-bottom:0}
    .answer strong{color:#fff;font-weight:700}
    .answer ol,.answer ul{margin:.5rem 0;padding-left:1.3rem}.answer li{margin:.3rem 0}
    .srch{margin-top:.9rem;border-top:1px solid var(--line);padding-top:.75rem}
    .srch .h{font-size:.6rem;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--faint);margin-bottom:.5rem;font-family:var(--mono)}
    .srcs{display:flex;flex-wrap:wrap;gap:.4rem}
    .src{display:flex;align-items:center;gap:.42rem;font-size:.78rem;color:#c4ccdb;
      border:1px solid var(--line2);border-radius:999px;padding:.26rem .6rem;background:var(--surface2)}
    .src .n{width:16px;height:16px;border-radius:5px;background:var(--raise);color:var(--teal);display:grid;place-items:center;font-size:.62rem;font-weight:700;font-family:var(--mono)}
    .src b{color:var(--text)}.src .sec{color:var(--faint)}
    .warn{margin-top:.6rem;font-size:.82rem;color:var(--amber);background:var(--amber-bg);
      border:1px solid rgba(224,169,59,.28);border-radius:9px;padding:.5rem .7rem}
    .spin{display:inline-block;width:14px;height:14px;border:2px solid var(--line2);border-top-color:var(--teal);
      border-radius:50%;animation:sp .7s linear infinite;vertical-align:-2px;margin-right:.5rem}
    @keyframes sp{to{transform:rotate(360deg)}}
    .think{color:var(--muted);font-size:.92rem}

    /* composer */
    .composer{flex:none;padding:.55rem 1.1rem 1rem;background:linear-gradient(to top,var(--bg) 62%,transparent)}
    .cwrap{max-width:820px;margin:0 auto;display:flex;align-items:flex-end;gap:.5rem;background:var(--surface);
      border:1px solid var(--line2);border-radius:14px;padding:.4rem .4rem .4rem .95rem;transition:border-color .15s,box-shadow .15s}
    .cwrap:focus-within{border-color:rgba(20,184,166,.5);box-shadow:0 0 0 3px var(--teal-bg)}
    #q{flex:1;border:0;outline:0;background:transparent;color:var(--text);font:inherit;font-size:1rem;resize:none;padding:.55rem 0;max-height:140px}
    #q::placeholder{color:var(--faint)}
    #go{flex:none;width:40px;height:40px;border:0;border-radius:10px;background:var(--teal);color:#04302b;
      display:grid;place-items:center;transition:background .15s}
    #go:hover{background:#19cbb8}#go:disabled{opacity:.45;cursor:default}
    .foot{max-width:820px;margin:.5rem auto 0;text-align:center;color:var(--faint);font-size:.72rem}

    /* drawers */
    .scrim{position:fixed;inset:0;background:rgba(0,0,0,.5);opacity:0;pointer-events:none;transition:opacity .2s;z-index:40}
    .scrim.on{opacity:1;pointer-events:auto}
    .drawer{position:fixed;top:0;right:0;height:100vh;width:360px;max-width:88vw;background:var(--surface);
      border-left:1px solid var(--line2);box-shadow:var(--shadow);transform:translateX(100%);transition:transform .24s cubic-bezier(.4,0,.2,1);
      z-index:50;display:flex;flex-direction:column}
    .drawer.on{transform:translateX(0)}
    .dhead{flex:none;display:flex;align-items:center;padding:.9rem 1.1rem;border-bottom:1px solid var(--line)}
    .dhead h3{margin:0;font-size:.95rem;font-weight:700}
    .dhead .x{margin-left:auto;color:var(--muted);background:transparent;border:0;font-size:1.2rem;line-height:1;padding:.1rem .3rem;border-radius:6px}
    .dhead .x:hover{color:var(--text);background:var(--surface2)}
    .dbody{flex:1;overflow-y:auto;padding:1.1rem}
    .grp{margin-bottom:1.35rem}
    .grp .lbl{font-size:.62rem;font-weight:700;letter-spacing:.7px;text-transform:uppercase;color:var(--faint);
      margin-bottom:.55rem;font-family:var(--mono)}
    .seg{display:flex;gap:.3rem;background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:.25rem}
    .seg button{flex:1;border:0;background:transparent;color:var(--muted);font-size:.76rem;font-weight:600;
      padding:.4rem .3rem;border-radius:7px;transition:all .12s}
    .seg button.on{background:var(--raise);color:var(--text);box-shadow:0 1px 2px rgba(0,0,0,.3)}
    .row{display:flex;align-items:center;justify-content:space-between;padding:.5rem 0}
    .row .name{font-size:.86rem;color:#c4ccdb}
    .row .desc{font-size:.72rem;color:var(--faint);margin-top:.1rem}
    /* toggle */
    .sw{position:relative;width:38px;height:22px;flex:none}
    .sw input{opacity:0;width:0;height:0}
    .sw .track{position:absolute;inset:0;background:var(--line2);border-radius:999px;transition:.15s}
    .sw .track:before{content:"";position:absolute;width:16px;height:16px;left:3px;top:3px;background:#c4ccdb;border-radius:50%;transition:.15s}
    .sw input:checked + .track{background:var(--teal)}
    .sw input:checked + .track:before{transform:translateX(16px);background:#04302b}
    .slider{width:100%;accent-color:var(--teal);margin-top:.3rem}
    select,input.txt{width:100%;background:var(--bg);border:1px solid var(--line);border-radius:9px;color:var(--text);
      font:inherit;font-size:.85rem;padding:.5rem .6rem}
    select:focus,input.txt:focus{outline:0;border-color:rgba(20,184,166,.5)}
    .two{display:flex;gap:.5rem}.two>*{flex:1}
    .kv{display:flex;align-items:center;justify-content:space-between;font-size:.8rem;padding:.35rem 0;border-bottom:1px solid var(--line)}
    .kv .k{color:var(--muted)}.kv .v{font-family:var(--mono);font-weight:600}
    .ctx{border:1px solid var(--line);border-radius:10px;padding:.6rem .7rem;margin-bottom:.55rem;background:var(--bg)}
    .ctx .top{display:flex;align-items:center;gap:.45rem;margin-bottom:.4rem;flex-wrap:wrap}
    .ctx .stag{font-size:.56rem;font-weight:700;text-transform:uppercase;letter-spacing:.5px;padding:.14rem .38rem;border-radius:5px;font-family:var(--mono)}
    .ctx .stag.sql{color:var(--teal);background:var(--teal-bg)}
    .ctx .stag.text{color:var(--slate);background:var(--slate-bg)}
    .ctx .meta{font-size:.74rem;color:#c4ccdb}.ctx .meta b{color:var(--text)}
    .ctx .score{margin-left:auto;font-family:var(--mono);font-size:.72rem;color:var(--amber)}
    .ctx .prev{font-size:.75rem;color:var(--faint);line-height:1.45;max-height:3.9em;overflow:hidden}
    .empty{color:var(--faint);font-size:.85rem;text-align:center;padding:2rem 0}
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <div class="logo">Fi</div>
      <div class="brand">Filing<b>IQ</b></div><span class="tag">10-K analyst</span>
      <div class="spacer"></div>
      <span class="pill" id="cachePill">cache <b>0%</b></span>
      <button class="tbtn" onclick="toggle('controls')" id="ctrlBtn" title="Controls">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 8 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H2a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 3.6 8a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H8a1.65 1.65 0 0 0 1-1.51V2a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V8a1.65 1.65 0 0 0 1.51 1H22a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
        Controls
      </button>
      <button class="tbtn" onclick="toggle('inspector')" id="inspBtn" title="Inspector">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m21 21 4.3 4.3" transform="scale(.8)"/></svg>
        Inspector
      </button>
      <button class="tbtn" onclick="newChat()">New chat</button>
    </div>
  </header>

  <div class="scroll" id="scroll">
    <div class="thread" id="thread">
      <div class="hero" id="hero">
        <h1>Ask the filings. <span>Get cited answers.</span></h1>
        <p>A grounded analyst over SEC 10-K filings for Apple, Microsoft, NVIDIA, Amazon and Alphabet. Numbers come from the actual filing tables, every claim is cited, and it refuses when unsupported. Open <b>Controls</b> to steer the engines, <b>Inspector</b> to see them work.</p>
        <div class="egs" id="egs">
          <div class="eg"><span class="k">Number</span>What was Microsoft's total revenue in fiscal 2024?</div>
          <div class="eg"><span class="k">Compare</span>Compare Apple's and Alphabet's net income in fiscal 2024.</div>
          <div class="eg"><span class="k">Concept</span>What are NVIDIA's main supply chain risks?</div>
          <div class="eg"><span class="k">Refusal</span>What is Apple's current stock price?</div>
        </div>
      </div>
    </div>
  </div>

  <div class="composer">
    <div class="cwrap">
      <textarea id="q" rows="1" placeholder="Ask about a filing…  (Shift+Enter for a new line)" autofocus></textarea>
      <button id="go" onclick="ask()" title="Send">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 2 11 13"/><path d="M22 2 15 22l-4-9-9-4 20-7z"/></svg>
      </button>
    </div>
    <div class="foot">Grounded in SEC EDGAR 10-K filings · numbers from a structured store · citations enforced · <a href="/docs">API</a></div>
  </div>

  <!-- Controls drawer -->
  <aside class="drawer" id="controls">
    <div class="dhead"><h3>Controls</h3><button class="x" onclick="toggle('controls')">&times;</button></div>
    <div class="dbody">
      <div class="grp">
        <div class="lbl">Route override</div>
        <div class="seg" id="segRoute">
          <button class="on" data-v="">Auto</button>
          <button data-v="TEXT">Text</button>
          <button data-v="TABLE">Table</button>
          <button data-v="HYBRID">Hybrid</button>
        </div>
      </div>
      <div class="grp">
        <div class="lbl">Retrieval</div>
        <div class="row"><div><div class="name">Hybrid search</div><div class="desc">dense + BM25 (off = dense only)</div></div>
          <label class="sw"><input type="checkbox" id="cHybrid" checked><span class="track"></span></label></div>
        <div class="row"><div><div class="name">Cross-encoder rerank</div><div class="desc">re-score the shortlist</div></div>
          <label class="sw"><input type="checkbox" id="cRerank" checked><span class="track"></span></label></div>
        <div class="row" style="display:block">
          <div class="name">Top-k chunks <span id="kVal" style="color:var(--teal);font-family:var(--mono)">5</span></div>
          <input class="slider" type="range" id="cTopk" min="1" max="12" value="5">
        </div>
      </div>
      <div class="grp">
        <div class="lbl">Scope filter</div>
        <div class="two">
          <select id="cCompany"><option value="">All companies</option><option>AAPL</option><option>MSFT</option><option>NVDA</option><option>AMZN</option><option>GOOGL</option></select>
          <input class="txt" id="cYear" placeholder="Year e.g. 2024">
        </div>
      </div>
      <div class="grp">
        <div class="lbl">Cost</div>
        <div class="row"><div><div class="name">Semantic cache</div><div class="desc">reuse answers for repeat questions</div></div>
          <label class="sw"><input type="checkbox" id="cCache" checked><span class="track"></span></label></div>
      </div>
    </div>
  </aside>

  <!-- Inspector drawer -->
  <aside class="drawer" id="inspector">
    <div class="dhead"><h3>Inspector</h3><button class="x" onclick="toggle('inspector')">&times;</button></div>
    <div class="dbody" id="inspBody">
      <div class="empty">Ask a question, then open the Inspector to see the routing decision and the retrieved evidence with rerank scores.</div>
    </div>
  </aside>
  <div class="scrim" id="scrim" onclick="closeAll()"></div>

<script>
var $=function(id){return document.getElementById(id);};
var chatHistory=[], busy=false, inspectors=[];
var q=$('q');
q.addEventListener('input',function(){q.style.height='auto';q.style.height=Math.min(q.scrollHeight,140)+'px';});
q.addEventListener('keydown',function(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();ask();}});
$('cTopk').addEventListener('input',function(){$('kVal').textContent=this.value;});
document.querySelectorAll('#egs .eg').forEach(function(el){el.addEventListener('click',function(){ask(el.textContent.replace(/^(Number|Compare|Concept|Refusal)/,'').trim());});});
document.querySelectorAll('#segRoute button').forEach(function(b){b.addEventListener('click',function(){
  document.querySelectorAll('#segRoute button').forEach(function(x){x.classList.remove('on');});b.classList.add('on');});});
refreshCache();

function toggle(id){var d=$(id),open=d.classList.contains('on');closeAll();if(!open){d.classList.add('on');$('scrim').classList.add('on');
  $(id==='controls'?'ctrlBtn':'inspBtn').classList.add('on');}}
function closeAll(){['controls','inspector'].forEach(function(i){$(i).classList.remove('on');});
  $('scrim').classList.remove('on');$('ctrlBtn').classList.remove('on');$('inspBtn').classList.remove('on');}
function newChat(){chatHistory=[];inspectors=[];location.reload();}

function controls(){
  var r=document.querySelector('#segRoute button.on').getAttribute('data-v');
  return {route_override:r||null, hybrid:$('cHybrid').checked, rerank:$('cRerank').checked,
    top_k:parseInt($('cTopk').value), company:$('cCompany').value||null, year:$('cYear').value||null,
    use_cache:$('cCache').checked};
}

function ask(preset){
  if(busy)return; if(preset)q.value=preset;
  var text=q.value.trim(); if(!text)return;
  var hero=$('hero'); if(hero)hero.remove();
  addUser(text); q.value=''; q.style.height='auto';
  var slot=addThinking(); busy=true; $('go').disabled=true;
  var body=Object.assign({question:text, history:chatHistory.slice(-3)}, controls());
  fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
    .then(function(res){if(!res.ok)throw new Error(res.status);return res.json();})
    .then(function(d){slot.innerHTML=answerBubble(d);chatHistory.push({question:text,answer:d.answer});refreshCache();})
    .catch(function(){slot.innerHTML=errBubble('Request failed. Please retry.');})
    .then(function(){busy=false;$('go').disabled=false;down();});
  down();
}

function addUser(t){var m=document.createElement('div');m.className='msg u';
  m.innerHTML="<div class='bubble u'>"+esc(t)+"</div>";$('thread').appendChild(m);down();}
function addThinking(){var m=document.createElement('div');m.className='msg a';
  m.innerHTML="<div class='av'>Fi</div><div class='awrap'><div class='bubble a'><span class='think'><span class='spin'></span>Reading the filings…</span></div></div>";
  $('thread').appendChild(m);down();return m.querySelector('.awrap');}

function answerBubble(d){
  var route=(d.route||'text').toLowerCase();
  var b="<div class='bubble a'><div class='badges'>";
  b+= d.refused ? "<span class='badge refused'>Refused</span>" : "<span class='badge "+route+"'>"+routeLabel(route)+"</span>";
  if(d.cached)b+="<span class='badge cached'>cached</span>";
  var idx=inspectors.length; inspectors.push(d.inspector||{});
  b+="<button class='inspect' onclick='showInspector("+idx+")'>Inspect</button></div>";
  if(d.rewritten_question && d.rewritten_question!==d.question)
    b+="<div class='rewrite'>Interpreted as: "+esc(d.rewritten_question)+"</div>";
  b+="<div class='answer"+(d.refused?" muted":"")+"'>"+renderMd(d.answer)+"</div>";
  if(d.citations&&d.citations.length){
    b+="<div class='srch'><div class='h'>Sources</div><div class='srcs'>";
    d.citations.forEach(function(c){b+="<span class='src'><span class='n'>"+c.n+"</span><b>"+c.company+"</b> · FY"+c.fiscal_year+" · <span class='sec'>"+prettySec(c.section)+"</span></span>";});
    b+="</div></div>";}
  if(d.hallucinated_citations&&d.hallucinated_citations.length)
    b+="<div class='warn'>Model referenced excerpts not provided: ["+d.hallucinated_citations.join("], [")+"]</div>";
  return "<div class='av'>Fi</div><div class='awrap'>"+b+"</div></div>";
}
function errBubble(m){return "<div class='av'>Fi</div><div class='awrap'><div class='bubble a'><div class='answer muted'>"+esc(m)+"</div></div></div>";}

function showInspector(i){renderInspector(inspectors[i]||{});var d=$('inspector');closeAll();d.classList.add('on');$('scrim').classList.add('on');$('inspBtn').classList.add('on');}
function renderInspector(ins){
  if(!ins||!ins.contexts){$('inspBody').innerHTML="<div class='empty'>No inspector data.</div>";return;}
  var h="<div class='grp'><div class='lbl'>Routing decision</div>";
  h+="<div class='kv'><span class='k'>Classified intent</span><span class='v' style='color:var(--slate)'>"+(ins.intent||'—')+"</span></div>";
  h+="<div class='kv'><span class='k'>Executed engine</span><span class='v' style='color:var(--teal)'>"+(ins.executed||'—')+"</span></div>";
  h+="<div class='kv'><span class='k'>Cache</span><span class='v'>"+(ins.cached?'HIT':'miss')+"</span></div>";
  var sc=(ins.scopes||[]).map(function(s){return (s.company||'*')+(s.fiscal_year?(' FY'+s.fiscal_year):'');}).join(', ');
  h+="<div class='kv' style='border:0'><span class='k'>Scope</span><span class='v'>"+(sc||'all')+"</span></div></div>";
  h+="<div class='grp'><div class='lbl'>Retrieved evidence ("+ins.contexts.length+")</div>";
  ins.contexts.forEach(function(c){
    var score = c.score===null||c.score===undefined ? (c.source==='sql'?'exact':'—') : c.score;
    h+="<div class='ctx'><div class='top'><span class='stag "+c.source+"'>"+c.source+"</span>"+
       "<span class='meta'><b>"+(c.company||'?')+"</b> · FY"+(c.fiscal_year||'?')+" · "+prettySec(c.section)+"</span>"+
       "<span class='score'>"+score+"</span></div>"+
       "<div class='prev'>"+esc(c.preview||'')+"…</div></div>";});
  h+="</div>";
  $('inspBody').innerHTML=h;
}

function routeLabel(r){return ({table:'Table engine',text:'Text engine',hybrid:'Hybrid'}[r]||r);}
function prettySec(s){return ({business:'Business',risk_factors:'Risk Factors',mda:'MD&A',market_risk:'Market Risk',
  financial_statements:'Financials',financial_data:'Financial data',other:'Filing'}[s]||s||'');}
function down(){var s=$('scroll');s.scrollTop=s.scrollHeight;}
function refreshCache(){fetch('/stats').then(function(r){return r.json();}).then(function(s){
  $('cachePill').innerHTML="cache <b>"+Math.round((s.hit_rate||0)*100)+"%</b>";}).catch(function(){});}
function esc(s){return (s||'').replace(/[&<>]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;'}[c];});}

function renderMd(src){
  var s=esc(src), parts=s.split('**'), b='';
  for(var i=0;i<parts.length;i++){b+=(i%2?'<strong>'+parts[i]+'</strong>':parts[i]);}
  var lines=b.split(String.fromCharCode(10)), out='', list=null;
  function close(){if(list){out+=(list==='ol'?'</ol>':'</ul>');list=null;}}
  for(var j=0;j<lines.length;j++){
    var line=lines[j].trim(), ol=null, ul=null, k=line.indexOf('. ');
    if(k>0 && /^[0-9]+$/.test(line.slice(0,k)))ol=line.slice(k+2);
    else if(line.slice(0,2)==='* '||line.slice(0,2)==='- '||line.slice(0,2)==='• ')ul=line.slice(2);
    if(ol!==null){if(list!=='ol'){close();out+='<ol>';list='ol';}out+='<li>'+ol+'</li>';}
    else if(ul!==null){if(list!=='ul'){close();out+='<ul>';list='ul';}out+='<li>'+ul+'</li>';}
    else{close();if(line)out+='<p>'+line+'</p>';}
  }
  close();return out;
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML

