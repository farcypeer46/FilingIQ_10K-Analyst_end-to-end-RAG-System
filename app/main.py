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


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    """Answer a question via the dual-engine Traffic Cop, with conversational memory.

    Passing `history` lets a follow-up ("how about Microsoft?") be rewritten into a
    standalone question before the same grounded pipeline runs.
    """
    try:
        out = app.state.generator.answer(req.question, top_k=req.top_k, history=req.history)
    except Exception:
        # The engine fails loud on LLM errors; translate that into a clean 502
        # instead of leaking a stack trace to the client.
        logger.exception("answer() failed")
        raise HTTPException(status_code=502, detail="Generation failed; please retry.")

    # Return only what a client needs (drop the bulky raw contexts).
    return {
        "question": req.question,
        "rewritten_question": out.get("rewritten_question", req.question),
        "answer": out["answer"],
        "refused": out["refused"],
        "route": out["route"],          # which engine answered (TEXT/TABLE/HYBRID)
        "cached": out.get("cached", False),
        "citations": out["citations"],  # each: {n, company, fiscal_year, section, chunk_id}
        "hallucinated_citations": out["hallucinated_citations"],
    }


# --------------------------------------------------------------------------- #
# Minimal frontend (single self-contained page; no build step, no framework)
# --------------------------------------------------------------------------- #

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FilingIQ — AI analyst for SEC 10-K filings</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root{
      --navy:#0A2540; --navy2:#12395f; --ink:#0B1F3A; --muted:#64748B;
      --line:#E4E9F0; --bg:#F7F9FC; --surface:#FFFFFF;
      --accent:#0FB5A6; --accent-dk:#0C8F84; --accent-weak:#E6FAF7;
      --user:#0A2540; --purple:#6D5AE6; --purple-weak:#EDEBFB;
      --warn-bg:#FFF7E6; --warn-line:#F1C878; --warn-ink:#8A5A00;
      --shadow:0 1px 2px rgba(10,37,64,.05), 0 10px 30px rgba(10,37,64,.06);
    }
    *{box-sizing:border-box}
    html,body{height:100%}
    body{margin:0;background:var(--bg);color:var(--ink);display:flex;flex-direction:column;height:100vh;
      font-family:'Inter',system-ui,-apple-system,sans-serif;line-height:1.55;-webkit-font-smoothing:antialiased}
    a{color:var(--accent-dk);text-decoration:none}

    header{background:var(--navy);color:#fff;flex:none;border-bottom:1px solid rgba(255,255,255,.06)}
    .bar{max-width:820px;margin:0 auto;padding:.8rem 1.25rem;display:flex;align-items:center;gap:.6rem}
    .logo{width:30px;height:30px;border-radius:9px;background:linear-gradient(135deg,var(--accent),#37d6c8);
      display:grid;place-items:center;color:#052f2b;font-weight:800;font-size:14px;box-shadow:0 2px 10px rgba(15,181,166,.45)}
    .brand{font-weight:800;letter-spacing:-.2px;font-size:1.1rem}
    .brand b{color:var(--accent)}
    .tag{font-size:.6rem;font-weight:700;letter-spacing:.6px;text-transform:uppercase;
      background:rgba(15,181,166,.18);color:#6ff0e3;padding:.15rem .4rem;border-radius:5px}
    .hbtn{margin-left:auto;display:flex;align-items:center;gap:.7rem}
    .cache{color:#9db4cf;font-size:.76rem;font-weight:500}
    .cache b{color:#6ff0e3;font-weight:700}
    .newchat{border:1px solid rgba(255,255,255,.18);background:transparent;color:#cfe0f2;font:inherit;
      font-size:.8rem;font-weight:600;padding:.35rem .7rem;border-radius:8px;cursor:pointer}
    .newchat:hover{background:rgba(255,255,255,.08);color:#fff}

    /* chat scroll area */
    .scroll{flex:1;overflow-y:auto}
    .thread{max-width:820px;margin:0 auto;padding:1.6rem 1.25rem 1rem}

    /* empty / hero state */
    .hero{padding:2.5rem 0 1rem;text-align:center}
    .hero h1{font-size:1.8rem;line-height:1.2;letter-spacing:-.6px;margin:.3rem 0 .5rem;font-weight:800}
    .hero h1 span{color:var(--accent-dk)}
    .hero p{color:var(--muted);margin:0 auto 1.6rem;font-size:1rem;max-width:560px}
    .egwrap{display:flex;flex-direction:column;gap:.55rem;max-width:560px;margin:0 auto;text-align:left}
    .eg{border:1px solid var(--line);background:var(--surface);border-radius:12px;padding:.75rem .95rem;
      cursor:pointer;font-size:.94rem;color:var(--navy2);display:flex;align-items:center;gap:.6rem;transition:all .15s;box-shadow:var(--shadow)}
    .eg:hover{border-color:var(--accent);background:var(--accent-weak);color:var(--accent-dk)}
    .eg .k{flex:none;font-size:.62rem;font-weight:700;letter-spacing:.4px;text-transform:uppercase;color:var(--muted);
      border:1px solid var(--line);border-radius:5px;padding:.12rem .35rem}

    /* messages */
    .msg{margin:1.1rem 0;display:flex;gap:.7rem}
    .msg.u{justify-content:flex-end}
    .bubble.u{background:var(--user);color:#fff;border-radius:14px 14px 4px 14px;padding:.7rem 1rem;max-width:78%;font-size:.98rem}
    .av{flex:none;width:30px;height:30px;border-radius:8px;display:grid;place-items:center;font-weight:800;font-size:13px}
    .av.a{background:linear-gradient(135deg,var(--accent),#37d6c8);color:#052f2b}
    .a-wrap{max-width:100%;flex:1}
    .bubble.a{background:var(--surface);border:1px solid var(--line);border-radius:4px 14px 14px 14px;
      padding:1rem 1.15rem;box-shadow:var(--shadow)}
    .badges{display:flex;gap:.4rem;flex-wrap:wrap;margin-bottom:.6rem}
    .badge{font-size:.64rem;font-weight:700;letter-spacing:.4px;text-transform:uppercase;padding:.2rem .5rem;border-radius:6px}
    .badge.table{background:var(--accent-weak);color:var(--accent-dk)}
    .badge.text{background:#EEF2F7;color:var(--navy2)}
    .badge.hybrid{background:var(--purple-weak);color:var(--purple)}
    .badge.cached{background:var(--warn-bg);color:var(--warn-ink)}
    .badge.refused{background:var(--warn-bg);color:var(--warn-ink)}
    .rewrite{font-size:.8rem;color:var(--muted);margin:-.2rem 0 .6rem;font-style:italic}
    .answer{font-size:1rem;color:var(--ink)}
    .answer.muted{color:var(--warn-ink)}
    .answer p{margin:.55rem 0} .answer p:first-child{margin-top:0} .answer p:last-child{margin-bottom:0}
    .answer strong{font-weight:700;color:var(--navy)}
    .answer ol,.answer ul{margin:.55rem 0;padding-left:1.35rem} .answer li{margin:.35rem 0}
    .srch{margin-top:1rem;border-top:1px solid var(--line);padding-top:.8rem}
    .srch .h{font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);margin-bottom:.5rem}
    .sources{display:flex;flex-wrap:wrap;gap:.45rem}
    .source{display:flex;align-items:center;gap:.45rem;font-size:.82rem;border:1px solid var(--line);
      border-radius:999px;padding:.28rem .65rem;background:#fbfdff}
    .num{flex:none;width:18px;height:18px;border-radius:5px;background:var(--navy);color:#fff;display:grid;place-items:center;font-size:.66rem;font-weight:700}
    .source b{color:var(--ink)} .source .sec{color:var(--muted)}
    .warn{margin-top:.7rem;font-size:.84rem;color:var(--warn-ink);background:var(--warn-bg);
      border:1px solid var(--warn-line);border-radius:9px;padding:.5rem .7rem}
    .spinner{display:inline-block;width:15px;height:15px;border:2px solid var(--line);
      border-top-color:var(--accent);border-radius:50%;animation:spin .7s linear infinite;vertical-align:-3px;margin-right:.5rem}
    @keyframes spin{to{transform:rotate(360deg)}}
    .thinking{color:var(--muted);font-size:.94rem}

    /* composer */
    .composer{flex:none;background:linear-gradient(to top,var(--bg) 60%,rgba(247,249,252,0));padding:.6rem 1.25rem 1rem}
    .cwrap{max-width:820px;margin:0 auto;display:flex;align-items:flex-end;gap:.5rem;background:#fff;
      border:1.5px solid var(--line);border-radius:14px;padding:.4rem .4rem .4rem .95rem;box-shadow:var(--shadow);transition:border-color .15s,box-shadow .15s}
    .cwrap:focus-within{border-color:var(--accent);box-shadow:0 0 0 4px var(--accent-weak)}
    #q{flex:1;border:0;outline:0;font:inherit;font-size:1rem;resize:none;background:transparent;color:var(--ink);padding:.55rem 0;max-height:140px}
    #go{flex:none;border:0;border-radius:10px;background:var(--accent);color:#03302c;font-weight:700;
      width:40px;height:40px;cursor:pointer;display:grid;place-items:center;transition:background .15s}
    #go:hover{background:var(--accent-dk);color:#fff} #go:disabled{opacity:.5;cursor:default}
    .foot{max-width:820px;margin:.5rem auto 0;text-align:center;color:var(--muted);font-size:.74rem}
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <div class="logo">FiQ</div>
      <div class="brand">Filing<b>IQ</b></div><span class="tag">10-K analyst</span>
      <div class="hbtn">
        <span class="cache" id="cache"></span>
        <button class="newchat" onclick="newChat()">New chat</button>
      </div>
    </div>
  </header>

  <div class="scroll" id="scroll">
    <div class="thread" id="thread">
      <div class="hero" id="hero">
        <h1>Ask the filings. <span>Get cited answers.</span></h1>
        <p>An AI analyst grounded in SEC 10-K filings for Apple, Microsoft, NVIDIA, Amazon and Alphabet. Every number comes from the actual filing tables, every claim is cited — and it refuses when the filings don't support an answer. Ask follow-ups naturally.</p>
        <div class="egwrap" id="examples">
          <div class="eg"><span class="k">Number</span>What was Microsoft's total revenue in fiscal 2024?</div>
          <div class="eg"><span class="k">Compare</span>Compare Apple's and Alphabet's net income in fiscal 2024.</div>
          <div class="eg"><span class="k">Concept</span>What are NVIDIA's main supply-chain risks?</div>
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
    <div class="foot">Grounded in SEC EDGAR 10-K filings · Numbers from a structured table store · Citations enforced · <a href="/docs">API</a></div>
  </div>

<script>
const $ = id => document.getElementById(id);
let history = [];          // [{question, answer}] for conversational follow-ups
let busy = false;

const q = $('q');
q.addEventListener('input', () => { q.style.height='auto'; q.style.height=Math.min(q.scrollHeight,140)+'px'; });
q.addEventListener('keydown', e => { if (e.key==='Enter' && !e.shiftKey){ e.preventDefault(); ask(); } });
document.querySelectorAll('#examples .eg').forEach(el =>
  el.addEventListener('click', () => ask(el.textContent.replace(/^(Number|Compare|Concept|Refusal)/,'').trim())));
refreshCache();

function newChat(){ history=[]; $('thread').innerHTML=''; location.reload(); }

async function ask(preset){
  if (busy) return;
  if (preset) q.value = preset;
  const text = q.value.trim();
  if (!text) return;
  const hero = $('hero'); if (hero) hero.remove();

  addUser(text);
  q.value=''; q.style.height='auto';
  const think = addThinking();
  busy = true; $('go').disabled = true;

  try {
    const res = await fetch('/ask', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ question:text, history: history.slice(-3) })
    });
    if (!res.ok){ think.innerHTML = errBubble("Request failed ("+res.status+"). Please retry."); return; }
    const d = await res.json();
    think.innerHTML = answerBubble(d);
    // Store the RESOLVED standalone question (not the raw text) so the next
    // follow-up's coreference chain stays clean and self-contained.
    history.push({question: d.rewritten_question || text, answer: d.answer});
    refreshCache();
  } catch(e){ think.innerHTML = errBubble("Network error."); }
  finally { busy=false; $('go').disabled=false; scrollDown(); }
  scrollDown();
}

function addUser(t){
  const m=document.createElement('div'); m.className='msg u';
  m.innerHTML="<div class='bubble u'>"+escapeHtml(t)+"</div>";
  $('thread').appendChild(m); scrollDown();
}
function addThinking(){
  const m=document.createElement('div'); m.className='msg a';
  m.innerHTML="<div class='av a'>Fi</div><div class='a-wrap'><div class='bubble a'><span class='thinking'><span class='spinner'></span>Reading the filings…</span></div></div>";
  $('thread').appendChild(m); scrollDown();
  return m.querySelector('.a-wrap');
}
function answerBubble(d){
  let b = "<div class='bubble a'><div class='badges'>";
  const route=(d.route||'text').toLowerCase();
  if (d.refused) b += "<span class='badge refused'>Refused</span>";
  else b += "<span class='badge "+route+"'>"+routeLabel(route)+"</span>";
  if (d.cached) b += "<span class='badge cached'>⚡ cached</span>";
  b += "</div>";
  if (d.rewritten_question && d.rewritten_question !== d.question)
    b += "<div class='rewrite'>Interpreted as: “"+escapeHtml(d.rewritten_question)+"”</div>";
  b += "<div class='answer"+(d.refused?" muted":"")+"'>"+renderMd(d.answer)+"</div>";
  if (d.citations && d.citations.length){
    b += "<div class='srch'><div class='h'>Sources</div><div class='sources'>";
    for (const c of d.citations)
      b += "<span class='source'><span class='num'>"+c.n+"</span><b>"+c.company+"</b> · FY"+c.fiscal_year+" · <span class='sec'>"+prettySec(c.section)+"</span></span>";
    b += "</div></div>";
  }
  if (d.hallucinated_citations && d.hallucinated_citations.length)
    b += "<div class='warn'>⚠ Model referenced excerpts not provided: ["+d.hallucinated_citations.join("], [")+"]</div>";
  return "<div class='av a'>Fi</div><div class='a-wrap'>"+b+"</div></div>";
}
function errBubble(msg){ return "<div class='av a'>Fi</div><div class='a-wrap'><div class='bubble a'><div class='answer muted'>"+escapeHtml(msg)+"</div></div></div>"; }

function routeLabel(r){ return ({table:'📊 Table engine',text:'📄 Text engine',hybrid:'🔀 Hybrid'}[r]||r); }
function prettySec(s){ return ({business:'Business',risk_factors:'Risk Factors',mda:'MD&A',
  market_risk:'Market Risk',financial_statements:'Financials',financial_data:'Financial data',other:'Filing'}[s]||s); }
function scrollDown(){ const s=$('scroll'); s.scrollTop=s.scrollHeight; }
async function refreshCache(){
  try{ const s = await (await fetch('/stats')).json();
    $('cache').innerHTML = "cache <b>"+Math.round((s.hit_rate||0)*100)+"%</b> hit"; }catch(e){}
}
function escapeHtml(s){ return (s||'').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

// Minimal, safe Markdown -> HTML.
function renderMd(src){
  let s = escapeHtml(src).replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
  const flush = t => (t==='ol'?'</ol>':t==='ul'?'</ul>':'');
  let out='', list=null;
  for (const raw of s.split(/\\n/)){
    const line = raw.trim();
    const ol = line.match(/^\\d+\\.\\s+(.*)$/);
    const ul = line.match(/^[-*]\\s+(.*)$/);
    if (ol){ if(list!=='ol'){ out+=flush(list)+'<ol>'; list='ol'; } out+='<li>'+ol[1]+'</li>'; }
    else if (ul){ if(list!=='ul'){ out+=flush(list)+'<ul>'; list='ul'; } out+='<li>'+ul[1]+'</li>'; }
    else { out+=flush(list); list=null; if(line) out+='<p>'+line+'</p>'; }
  }
  return out + flush(list);
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML
