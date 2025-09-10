#!/usr/bin/env python3
"""
Working Football Assistant - Functional Application
Simplified version that works without all advanced dependencies
"""

import os
import re
import uuid
import json
from typing import Optional, Dict, Any, List

try:
    from fastapi import FastAPI, Request, Query
    from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Try to import existing components
try:
    from api.football_data import get_standings, resolve_competition, list_competitions
    FOOTBALL_DATA_AVAILABLE = True
except ImportError:
    FOOTBALL_DATA_AVAILABLE = False

try:
    from api.livescore_api import get_live_scores_lsa
    LIVESCORE_AVAILABLE = True
except ImportError:
    LIVESCORE_AVAILABLE = False

try:
    from llm.openrouter_llm import query_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Try to import advanced components
try:
    from llm.llm_intelligence_engine import create_intelligence_engine
    INTELLIGENCE_ENGINE_AVAILABLE = True
except ImportError:
    INTELLIGENCE_ENGINE_AVAILABLE = False

try:
    from llm.football_knowledge_engine import create_football_knowledge_engine
    KNOWLEDGE_ENGINE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_ENGINE_AVAILABLE = False

try:
    from llm.intelligent_entity_resolver import create_entity_resolver, resolve_entities_from_query
    ENTITY_RESOLVER_AVAILABLE = True
except ImportError:
    ENTITY_RESOLVER_AVAILABLE = False

if not FASTAPI_AVAILABLE:
    print("❌ FastAPI not available. Please install: pip install fastapi uvicorn")
    print("🔧 Running in basic mode without web interface...")
    
    # Basic mode - just show status and exit gracefully
    print("\n📊 Component Status:")
    print(f"  FastAPI: {'✅' if FASTAPI_AVAILABLE else '❌'}")
    print(f"  Football Data: {'✅' if FOOTBALL_DATA_AVAILABLE else '❌'}")
    print(f"  LiveScore: {'✅' if LIVESCORE_AVAILABLE else '❌'}")
    print(f"  LLM: {'✅' if LLM_AVAILABLE else '❌'}")
    print(f"  Intelligence Engine: {'✅' if INTELLIGENCE_ENGINE_AVAILABLE else '❌'}")
    print(f"  Knowledge Engine: {'✅' if KNOWLEDGE_ENGINE_AVAILABLE else '❌'}")
    print(f"  Entity Resolver: {'✅' if ENTITY_RESOLVER_AVAILABLE else '❌'}")
    
    print("\n💡 To use the web interface, please ensure FastAPI is properly installed:")
    print("   pip install fastapi uvicorn")
    exit(1)

# Initialize FastAPI app
app = FastAPI(
    title="Working Football Assistant",
    description="Football assistant with graceful component handling",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
intelligence_coordinator = None
knowledge_engine = None
entity_resolver = None

# In-memory conversation state
CONV_STORE: Dict[str, List[Dict]] = {}
CONV_CTX: Dict[str, Dict] = {}
MAX_TURNS = 20

# ----------------------
# Live score formatting helpers (shared by endpoint and chat)
# ----------------------

def _ls_minute(m: dict) -> str:
    """Extract minute/time as 65'"""
    try:
        for k in ("minute", "time", "timer", "status"):
            v = (m.get(k) if isinstance(m, dict) else None) or ""
            if isinstance(v, (int, float)):
                return f"{int(v)}'"
            if isinstance(v, str) and v.strip():
                import re as _re
                s = v.strip()
                mm = _re.search(r"(\d{1,3})", s)
                if mm:
                    return f"{mm.group(1)}'"
                return s
    except Exception:
        pass
    return ""

def _ls_team_name(match: dict, side: str, idx: int) -> str:
    side_l = side.lower()
    # direct
    v = match.get(f"{side_l}_team") or match.get(f"{side_l}_name")
    if isinstance(v, str) and v.strip():
        return v.strip()
    # nested objects {home:{name:..}}
    obj = match.get(side_l) or {}
    if isinstance(obj, dict):
        nm = obj.get("name") or obj.get("team")
        if isinstance(nm, str) and nm.strip():
            return nm.strip()
    # teams container
    teams = match.get("teams")
    if isinstance(teams, dict):
        t = teams.get(side_l)
        if isinstance(t, str) and t.strip():
            return t.strip()
        if isinstance(t, dict):
            nm = t.get("name") or t.get("team") or t.get("title")
            if isinstance(nm, str) and nm.strip():
                return nm.strip()
    if isinstance(teams, list) and len(teams) >= 2:
        t = teams[idx]
        if isinstance(t, str) and t.strip():
            return t.strip()
        if isinstance(t, dict):
            nm = t.get("name") or t.get("team") or t.get("title")
            if isinstance(nm, str) and nm.strip():
                return nm.strip()
    return "Home" if side_l == "home" else "Away"

def _ls_scores(match: dict) -> tuple[str, str]:
    """Extract (home, away) scores preferring string score fields.
    Primary: scores.score or score (string "H-A"). Fallbacks: list/tuple, dict, teams[] embedded scores,
    then numeric fields. Left number is home, right number is away.
    """
    try:
        import re as _re
        # 1) Nested string: match['scores']['score']
        scd = match.get('scores')
        if isinstance(scd, dict):
            s2 = scd.get('score')
            if isinstance(s2, str) and s2.strip():
                m2 = _re.search(r"^\s*(\d{1,3})\D+(\d{1,3})\s*$", s2.strip())
                if m2:
                    return m2.group(1), m2.group(2)
        # 2) Root string: match['score']
        sc = match.get('score')
        if isinstance(sc, str) and sc.strip():
            m = _re.search(r"^\s*(\d{1,3})\D+(\d{1,3})\s*$", sc.strip())
            if m:
                return m.group(1), m.group(2)
        # 3) List/Tuple
        if isinstance(sc, (list, tuple)) and len(sc) >= 2:
            return str(sc[0]), str(sc[1])
        # 4) Dict
        if isinstance(sc, dict):
            home_keys = ("home","h",0,"left"); away_keys = ("away","a",1,"right")
            hs = next((sc[k] for k in home_keys if k in sc), None)
            as_ = next((sc[k] for k in away_keys if k in sc), None)
            if hs is not None and as_ is not None:
                return str(hs), str(as_)
        # 5) Teams list embedded
        teams = match.get('teams')
        if isinstance(teams, list) and len(teams) >= 2:
            def _sv(x):
                if isinstance(x, dict):
                    for k in ('score','goals','points'):
                        if k in x: return x[k]
                return None
            htry = _sv(teams[0]); atry = _sv(teams[1])
            if htry is not None and atry is not None:
                return str(htry), str(atry)
        # 6) Numeric fields
        hs = match.get('home_score') or (match.get('home', {}) or {}).get('goals')
        as_ = match.get('away_score') or (match.get('away', {}) or {}).get('goals')
        return str(hs or 0), str(as_ or 0)
    except Exception:
        return '0','0'

def format_live_scores_payload(raw: Any, filter_kw: Optional[str] = None) -> str:
    """Return lines of live scores filtered by league/competition/country keyword."""
    matches: List[Dict[str, Any]] = []
    if isinstance(raw, dict):
        matches = (
            raw.get("matches")
            or raw.get("data", {}).get("match")
            or raw.get("response")
            or raw.get("fixtures")
            or raw.get("events")
            or []
        )
    elif isinstance(raw, list):
        matches = raw
    lines: List[str] = []
    kw = (filter_kw or "").lower().strip()
    for m in matches[:100]:
        ltxt = str(m.get("league", m.get("competition", ""))).lower()
        ctxt = str((m.get("country") or {}).get("name") or "").lower()
        if kw and (kw not in ltxt and kw not in ctxt):
            continue
        home = _ls_team_name(m, "home", 0)
        away = _ls_team_name(m, "away", 1)
        hs, as_ = _ls_scores(m)
        minute = _ls_minute(m)
        spacer = "  "
        lines.append(f"{home} {hs}{spacer}{minute} {as_} {away}".strip())
    if not lines:
        return "No live matches currently available."
    return "\n".join(lines)

# ----------------------
# Follow-up merge helper: ensure only one follow-up is appended
# ----------------------

def _merge_follow_up(answer: str, question: str, context: Dict) -> str:
    try:
        fu = _generate_follow_up(question, answer, context)
    except Exception:
        fu = ""
    if not fu or not fu.strip():
        return answer
    # Avoid duplicates if a similar follow-up already exists in the answer
    norm_ans = (answer or "").strip()
    norm_fu = fu.strip()
    if norm_fu.lower() in norm_ans.lower():
        return answer
    # Also skip if the answer already ends with a typical follow-up line
    tail = norm_ans.splitlines()[-1].strip().lower() if norm_ans.splitlines() else ""
    if any(kw in tail for kw in ("would you like", "shall i", "want ")) and tail.endswith("?"):
        return answer
    return f"{answer}\n\n{fu}"

@app.on_event("startup")
async def startup_event():
    """Initialize available components on startup"""
    global intelligence_coordinator, knowledge_engine, entity_resolver
    
    print("🚀 Initializing Working Football Assistant...")
    
    # Try to initialize advanced components
    if INTELLIGENCE_ENGINE_AVAILABLE:
        try:
            intelligence_coordinator = create_intelligence_engine()
            print("✅ Intelligence Engine initialized")
        except Exception as e:
            print(f"⚠️ Intelligence Engine failed: {e}")
    
    if KNOWLEDGE_ENGINE_AVAILABLE:
        try:
            knowledge_engine = create_football_knowledge_engine()
            print("✅ Knowledge Engine initialized")
        except Exception as e:
            print(f"⚠️ Knowledge Engine failed: {e}")
    
    if ENTITY_RESOLVER_AVAILABLE:
        try:
            entity_resolver = create_entity_resolver()
            print("✅ Entity Resolver initialized")
        except Exception as e:
            print(f"⚠️ Entity Resolver failed: {e}")
    
    print("🎯 System ready!")

@app.get("/")
def root():
    return {
        "ok": True,
        "msg": "Working Football Assistant API",
        "version": "1.0.0",
        "available_components": {
            "football_data": FOOTBALL_DATA_AVAILABLE,
            "livescore": LIVESCORE_AVAILABLE,
            "llm": LLM_AVAILABLE,
            "intelligence_engine": intelligence_coordinator is not None,
            "knowledge_engine": knowledge_engine is not None,
            "entity_resolver": entity_resolver is not None,
        },
        "ui": "/chat",
        "docs": "/docs"
    }

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return PlainTextResponse("", status_code=204)

@app.get("/health")
def health_check():
    """Health check with component status"""
    components = {
        "football_data": FOOTBALL_DATA_AVAILABLE,
        "livescore": LIVESCORE_AVAILABLE,
        "llm": LLM_AVAILABLE,
        "intelligence_engine": intelligence_coordinator is not None,
        "knowledge_engine": knowledge_engine is not None,
        "entity_resolver": entity_resolver is not None,
    }
    
    advanced_count = sum([
        intelligence_coordinator is not None,
        knowledge_engine is not None,
        entity_resolver is not None
    ])
    
    if advanced_count >= 2:
        mode = "advanced"
    elif advanced_count >= 1:
        mode = "partial"
    else:
        mode = "basic"
    
    return {
        "status": "healthy",
        "mode": mode,
        "components": components,
        "advanced_components": f"{advanced_count}/3"
    }

@app.get("/chat", response_class=HTMLResponse)
def chat_ui():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>smart Football Assistant</title>
  <style>
    :root{
      font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      --pitch-dark:#FFFFFF;       /* deep pitch green */
      --pitch-light:#FFFFFF;      /* lighter pitch green */
      --pitch-line:rgba(255,255,255,0.05);
      --surface:#101834;          /* card surface */
      --surface-border:#213055;   /* card border */
      --primary:#22c55e;          /* vibrant grass green */
      --primary-700:#16a34a;
      --accent:#f59e0b;           /* football accent (amber) */
      --text:#eaf2ff;
      --muted:#97a7c7;
    }
    body{
      margin:0;color:var(--text);
      background-image:
        radial-gradient(1200px 600px at 12% -10%, rgba(255,255,255,0.08), transparent 60%),
        repeating-linear-gradient(90deg, var(--pitch-line) 0 2px, transparent 2px 78px),
        linear-gradient(160deg, var(--pitch-light), var(--pitch-dark));
      background-attachment: fixed;
    }
   .wrap{max-width:900px;margin:24px auto;padding:0 18px}
.card{background:rgba(255, 255, 255,0.85);backdrop-filter:blur(6px);border:1px solid #E5E7EB;border-radius:18px;padding:22px;margin-bottom:16px}
h1{margin:0 0 10px;text-align:center;color:#2563EB}
.subtitle{text-align:center;color:#6B7280;margin-bottom:20px}
.row{display:flex;gap:10px;margin-top:10px;justify-content:center;}
textarea{flex:1;min-height:88px;border-radius:14px;padding:14px;border:1px solid #E5E7EB;background:#F5F6F7;color:#2C2C2C;font-size:16px;line-height:1.5;letter-spacing:.2px;font-family:inherit}
textarea::placeholder{color:#9CA3AF;opacity:.9;font-style:italic}
button{padding:12px 16px;border-radius:14px;border:0;background:#2563EB;color:#FFFFFF;cursor:pointer;font-weight:650;position:relative;overflow:hidden;box-shadow:0 4px 10px rgba(37,99,235,0.2)}
button:hover{background:#1E40AF;box-shadow:0 6px 16px rgba(37,99,235,0.3)}
button:disabled{opacity:.6;cursor:not-allowed}
.msg{padding:14px 16px;border-radius:12px;margin:12px 0}
.user{background:#F3F4F6;border:1px solid #E5E7EB;color:#2C2C2C}
.assistant{background:#F9FAFB;border:1px solid #E5E7EB;color:#2C2C2C}
.system{background:#E5E7EB;border:1px solid #D1D5DB;font-size:13px;color:#374151}
.error{background:#FEE2E2;border:1px solid #DC2626;color:#DC2626}
.small{color:#6B7280;font-size:12px}
.suggestion{background:#F9FAFB;border:1px dashed #9CA3AF;color:#6B7280;pointer-events:none}
.buttons-row{position:relative;z-index:2}
pre{white-space:pre-wrap;word-wrap:break-word;margin:0}
.status{padding:8px 12px;border-radius:8px;font-size:12px;margin:4px 0}
.status.advanced{background:#ECFDF5;border:1px solid #16A34A;color:#15803D}
.status.partial{background:#FFFBEB;border:1px solid #FBBF24;color:#B45309}
.status.basic{background:#FEF2F2;border:1px solid #DC2626;color:#B91C1C}
/* Loading UI for Ask button */
#askBtn.loading{opacity:0.9;cursor:progress}
#askBtn.loading .loader{
  position:absolute;left:8px;right:8px;bottom:8px;height:3px;border-radius:2px;background:rgba(229,231,235,0.6);overflow:hidden
}
#askBtn.loading .loader::after{
  content:"";display:block;height:100%;width:35%;border-radius:2px;background:#2563EB;animation:loadBar 1.1s linear infinite
}

    @keyframes loadBar{0%{margin-left:-35%}100%{margin-left:100%}}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>⚽  Football Assistant</h1>
      
      
      <div id="status-container"></div>
      <div class="small" id="cid" style="opacity:.75;margin:6px 0 8px" hidden="true"></div>
      <div id="log"></div>
      
      <div class="row">
        <textarea id="q" placeholder="Ask about football - teams, standings, scores, or any football question..."></textarea>
        <button id="askBtn">Ask</button>
      </div>
      
      <div class="row buttons-row" style="justify-content:center;">
        <button id="scoresBtn">Live Scores</button>
        <button id="healthBtn">System Status</button>
        <button id="newBtn" title="Start a new conversation">New Chat</button>
      </div>
      
      
    </div>
  </div>

<script>
const log = document.getElementById('log');
const statusContainer = document.getElementById('status-container');
const askBtn = document.getElementById('askBtn');
const qEl = document.getElementById('q');

function addMsg(role, text) {
  const sanitize = (s) => String(s).replace(/[&<>]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;'}[c]))
  const d = document.createElement('div')
  d.className = 'msg ' + (role === 'user' ? 'user' : role === 'system' ? 'system' : role === 'error' ? 'error' : 'assistant')
  d.innerHTML = '<pre>' + sanitize(text) + '</pre>'
  log.appendChild(d)
  d.scrollIntoView({behavior: 'smooth', block: 'end'})
}

function showStatus(data) {
  const mode = data.mode || 'unknown';
  const statusClass = mode;
  let statusText = '';
  
  if (mode === 'advanced') {
    statusText = '✅ Advanced Mode: All AI components active';
  } else if (mode === 'partial') {
    statusText = '⚠️ Partial Mode: Some AI components active (' + data.advanced_components + ')';
  } else {
    statusText = '🔧 Basic Mode: Core functionality only';
  }
  
  statusContainer.innerHTML = `<div class="status ${statusClass}" hidden= "true">${statusText}</div>`;
}

function ensureConvId() {
  let id = localStorage.getItem('convId');
  if (!id) {
    id = (crypto.randomUUID && crypto.randomUUID()) || (Date.now().toString(36) + Math.random().toString(36).slice(2));
    localStorage.setItem('convId', id);
  }
  document.getElementById('cid').textContent = "Conversation: " + id;
  return id;
}

let convId = ensureConvId();

// Check system status on load
fetch('/health')
  .then(r => r.json())
  .then(showStatus)
  .catch(() => showStatus({mode: 'basic'}));

document.getElementById('newBtn').onclick = () => {
  localStorage.removeItem('convId');
  convId = ensureConvId();
  log.innerHTML = '';
  addMsg('system', 'Started a new conversation. Ask me about football!');
};

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  const ct = r.headers.get('content-type') || '';
  return ct.includes('application/json') ? r.json() : r.text();
}

document.getElementById('askBtn').onclick = async () => {
  const q = qEl.value.trim();
  if (!q) return;
  
  addMsg('user', q);
  qEl.value = '';
  // Enter loading state
  const prevPlaceholder = qEl.getAttribute('placeholder') || '';
  qEl.setAttribute('placeholder', 'gathering information for your answer');
  askBtn.disabled = true;
  const prevHTML = askBtn.innerHTML;
  askBtn.classList.add('loading');
  askBtn.innerHTML = 'Thinking…<div class="loader"></div>';
  
  try {
    const data = await postJSON('/ask', {question: q, conversation_id: convId});
    
    if (typeof data === 'string') {
      addMsg('assistant', data);
    } else if (data.error) {
      addMsg('error', 'Error: ' + data.error);
    } else {
      let ans = data.answer || data.message || ''
      if (typeof ans !== 'string') ans = JSON.stringify(data, null, 2)
      addMsg('assistant', ans)
    }
  } catch (e) {
    addMsg('error', 'Connection error: ' + e.message);
  } finally {
    // Exit loading state
    askBtn.disabled = false;
    askBtn.classList.remove('loading');
    askBtn.innerHTML = prevHTML;
    qEl.setAttribute('placeholder', prevPlaceholder || 'Ask about football - teams, standings, scores, or any football question...');
  }
};

function addSuggestion(text) {
  if (!text) return;
  const d = document.createElement('div')
  d.className = 'msg suggestion'
  d.innerHTML = '<pre>' + String(text).replace(/[&<>]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;'}[c])) + '</pre>'
  log.appendChild(d)
  d.scrollIntoView({behavior: 'smooth', block: 'end'})
}

document.getElementById('scoresBtn').onclick = async () => {
  try {
    const r = await fetch('/live-scores');
    const data = await r.json();
    
    if (data.error) {
      addMsg('error', 'Live scores error: ' + data.error);
    } else {
      if (data.formatted_scores) {
        addMsg('assistant', data.formatted_scores);
      } else {
        addMsg('assistant', 'Live scores:\\n' + JSON.stringify(data, null, 2));
      }
    }
  } catch (e) {
    addMsg('error', 'Error fetching live scores: ' + e.message);
  }
};


document.getElementById('healthBtn').onclick = async () => {
  try {
    const r = await fetch('/health');
    const data = await r.json();
    showStatus(data);
    addMsg('system', 'System Status:\\n' + JSON.stringify(data, null, 2));
  } catch (e) {
    addMsg('error', 'Error checking system status: ' + e.message);
  }
};

// Auto-focus on textarea
document.getElementById('q').focus();

// Enter to submit (Shift+Enter for new line)
document.getElementById('q').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    document.getElementById('askBtn').click();
  }
});

// Initial welcome message
addMsg('system', 'Welcome to the Working Football Assistant! Ask me about football teams, standings, or any football question.');
</script>
</body>
</html>
    """

@app.get("/live-scores")
def live_scores(league: Optional[str] = Query(None, description="Optional league/competition keyword to filter")):
    """Get live scores with clean formatting"""
    if LIVESCORE_AVAILABLE:
        try:
            raw_scores = get_live_scores_lsa()
            # Use unified formatter (handles names, correct home/away scores, filters)
            try:
                formatted = format_live_scores_payload(raw_scores, league)
                return {"formatted_scores": formatted}
            except Exception:
                pass
            # Format into: "{home} {homeScore}  {minute} {awayScore} {away}"
            def _minute(m: dict) -> str:
                for k in ("minute", "time", "timer", "status"):
                    v = (m.get(k) if isinstance(m, dict) else None) or ""
                    if isinstance(v, (int, float)):
                        return f"{int(v)}'"
                    if isinstance(v, str) and v.strip():
                        # normalize examples: "65'", "65", "LIVE 65'"
                        s = v.strip()
                        # pick last number chunk
                        import re as _re
                        m2 = _re.search(r"(\d{1,3})", s)
                        if m2:
                            return f"{m2.group(1)}'"
                        return s
                return ""

            def _score_parts(m: dict) -> tuple[str, str]:
                hs = str(m.get("home_score") or m.get("homeGoals") or m.get("home", {}).get("goals") or "")
                as_ = str(m.get("away_score") or m.get("awayGoals") or m.get("away", {}).get("goals") or "")
                sc = m.get("score") or ""
                if (not hs or not as_) and isinstance(sc, str) and "-" in sc:
                    try:
                        h, a = sc.replace(" ", "").split("-", 1)
                        hs = hs or h
                        as_ = as_ or a
                    except Exception:
                        pass
                return (hs or "0", as_ or "0")

            if isinstance(raw_scores, dict) and "matches" in raw_scores:
                # Helper to extract team names from various shapes
                def _team_name(match: dict, side: str) -> str:
                    side_l = side.lower()
                    # direct fields like home_team / away_team
                    v = match.get(f"{side_l}_team") or match.get(f"{side_l}_name")
                    if isinstance(v, str) and v.strip():
                        return v.strip()
                    # nested objects e.g., {home: {name: ..}}
                    obj = match.get(side_l) or {}
                    if isinstance(obj, dict):
                        n = obj.get("name") or obj.get("team")
                        if isinstance(n, str) and n.strip():
                            return n.strip()
                    # alternate containers: teams can be dict or list
                    teams = match.get("teams")
                    # dict form {home: .., away: ..}
                    if isinstance(teams, dict):
                        n = teams.get(side_l)
                        if isinstance(n, str) and n.strip():
                            return n.strip()
                        if isinstance(n, dict):
                            nm = n.get("name") or n.get("team")
                            if isinstance(nm, str) and nm.strip():
                                return nm.strip()
                    # list form [home, away]
                    if isinstance(teams, list) and len(teams) >= 2:
                        idx = 0 if side_l == "home" else 1
                        t = teams[idx]
                        if isinstance(t, str) and t.strip():
                            return t.strip()
                        if isinstance(t, dict):
                            nm = t.get("name") or t.get("team") or t.get("title")
                            if isinstance(nm, str) and nm.strip():
                                return nm.strip()
                    return "Home" if side_l == "home" else "Away"
                european_leagues = [
                    "premier league", "la liga", "serie a", "bundesliga", "ligue 1",
                    "champions league", "europa league", "eredivisie", "primeira liga", "conference league"
                ]
                lines = []
                for match in raw_scores.get("matches", [])[:40]:
                    ltxt = str(match.get("league", match.get("competition", ""))).lower()
                    if league:
                        if league.lower() not in ltxt:
                            continue
                    else:
                        if european_leagues and not any(x in ltxt for x in european_leagues):
                            continue
                    home = _team_name(match, "home")
                    away = _team_name(match, "away")
                    hs, as_ = _ls_scores(match)
                    # list scores: scores/goals/result = [home, away] or inside teams list
                    if (hs == "0" and as_ == "0"):
                        scores = match.get("scores") or match.get("goals") or match.get("result")
                        if isinstance(scores, list) and len(scores) >= 2:
                            try:
                                hs = str(scores[0])
                                as_ = str(scores[1])
                            except Exception:
                                pass
                        teams = match.get("teams")
                        if isinstance(teams, list) and len(teams) >= 2:
                            def _from_team(obj):
                                if isinstance(obj, dict):
                                    for key in ("score", "goals", "points"):
                                        v = obj.get(key)
                                        if isinstance(v, (int, float, str)):
                                            return str(v)
                                return None
                            htry = _from_team(teams[0])
                            atry = _from_team(teams[1])
                            hs = htry or hs
                            as_ = atry or as_
                    minute = _minute(match) or ""
                    spacer = "  "  # double-space between home score and minute
                    line = f"{home} {hs}{spacer}{minute} {as_} {away}".strip()
                    lines.append(line)
                if not lines:
                    lines.append("No European matches currently live.")
                return {"formatted_scores": "\n".join(lines)}
            # If the API returned some other structure, keep it simple
            if isinstance(raw_scores, dict):
                # Try a generic best-effort mapping
                matches = (
                    raw_scores.get("data", {}).get("match")
                    or raw_scores.get("response")
                    or raw_scores.get("fixtures")
                    or raw_scores.get("events")
                    or []
                )
                out = []
                for m in matches[:60]:
                    # team names from several forms
                    home = (
                        (m.get("home", {}) or {}).get("name")
                        or m.get("home_name")
                        or m.get("home_team")
                    )
                    away = (
                        (m.get("away", {}) or {}).get("name")
                        or m.get("away_name")
                        or m.get("away_team")
                    )
                    # teams as list
                    if not home or not away:
                        teams = m.get("teams")
                        if isinstance(teams, list) and len(teams) >= 2:
                            h, a = teams[0], teams[1]
                            if isinstance(h, dict):
                                home = home or h.get("name") or h.get("team") or h.get("title")
                            elif isinstance(h, str):
                                home = home or h
                            if isinstance(a, dict):
                                away = away or a.get("name") or a.get("team") or a.get("title")
                            elif isinstance(a, str):
                                away = away or a
                    home = (home or "Home").strip()
                    away = (away or "Away").strip()

                    # scores
                    hs = str(m.get("home_score") or 0)
                    as_ = str(m.get("away_score") or 0)
                    scores = m.get("scores") or m.get("goals") or m.get("result")
                    if isinstance(scores, list) and len(scores) >= 2:
                        try:
                            hs = str(scores[0]); as_ = str(scores[1])
                        except Exception:
                            pass
                    if (hs == "0" and as_ == "0") and isinstance(m.get("teams"), list) and len(m.get("teams")) >= 2:
                        def _sv(x):
                            if isinstance(x, dict):
                                for k in ("score","goals","points"):
                                    if k in x: return str(x[k])
                            return None
                        hs = _sv(m["teams"][0]) or hs
                        as_ = _sv(m["teams"][1]) or as_

                    # minute
                    minute = str(m.get("time") or m.get("minute") or "").strip()
                    if minute and not minute.endswith("'"):
                        minute = f"{minute}'"

                    # optional filter by league/country keyword
                    ltxt = str(m.get("league", m.get("competition", ""))).lower()
                    ctxt = str((m.get("country") or {}).get("name") or "").lower()
                    if league:
                        lk = league.lower()
                        if lk not in ltxt and lk not in ctxt:
                            continue

                    out.append(f"{home} {hs}  {minute} {as_} {away}".strip())
                if out:
                    return {"formatted_scores": "\n".join(out)}
            return {"formatted_scores": "No live matches available."}
        except Exception as e:
            return {"error": f"Live scores service error: {e}"}
    else:
        return {"error": "Live scores service not available", "message": "Please configure LiveScore API"}

@app.get("/standings")
def standings(league: str = Query("PL", description="League code or name")):
    """Get league standings with fallback"""
    if FOOTBALL_DATA_AVAILABLE:
        try:
            from api.football_data import resolve_competition
            code = resolve_competition(league)
            return get_standings(code)
        except Exception as e:
            return {"error": f"Standings service error: {e}"}
    else:
        return {"error": "Football data service not available", "message": "Please configure Football-Data API"}

@app.get("/standings/multiple")
def multiple_standings(leagues: str = Query("PL,PD,SA,BL1,FL1", description="Comma-separated league codes")):
    """Get standings for multiple leagues"""
    if FOOTBALL_DATA_AVAILABLE:
        try:
            from api.football_data import get_many_standings
            league_list = [l.strip() for l in leagues.split(",")]
            return get_many_standings(league_list)
        except Exception as e:
            return {"error": f"Multiple standings service error: {e}"}
    else:
        return {"error": "Football data service not available", "message": "Please configure Football-Data API"}

@app.get("/leagues")
def available_leagues():
    """List available leagues and their codes"""
    if FOOTBALL_DATA_AVAILABLE:
        try:
            from api.football_data import list_competitions, competition_map
            competitions = list_competitions()
            mapping = competition_map()
            
            # Popular leagues with their codes
            popular_leagues = {
                "PL": "Premier League (England)",
                "PD": "La Liga (Spain)",
                "SA": "Serie A (Italy)",
                "BL1": "Bundesliga (Germany)",
                "FL1": "Ligue 1 (France)",
                "DED": "Eredivisie (Netherlands)",
                "PPL": "Primeira Liga (Portugal)",
                "CL": "Champions League"
            }
            
            return {
                "popular_leagues": popular_leagues,
                "all_competitions": competitions[:20],  # Limit to first 20
                "aliases": {k: v for k, v in list(mapping.items())[:20]}  # Sample of aliases
            }
        except Exception as e:
            return {"error": f"Leagues service error: {e}"}
    else:
        return {"error": "Football data service not available"}

@app.get("/team/{team_name}")
def get_team_info(team_name: str):
    """Get specific team information including goal difference"""
    if FOOTBALL_DATA_AVAILABLE:
        try:
            from api.football_data import get_many_standings, resolve_competition, list_competitions
            
            # Search in ALL available leagues, not just major ones
            try:
                all_competitions = list_competitions()
                all_league_codes = [comp.get("code") for comp in all_competitions if comp.get("code")]
                # Prioritize major leagues but include all
                priority_leagues = ["PL", "PD", "SA", "BL1", "FL1", "DED", "PPL", "CL", "ELC"]
                leagues = [league for league in priority_leagues if league in all_league_codes]
                leagues.extend([code for code in all_league_codes if code not in leagues][:15])  # Add more leagues
            except:
                # Fallback to expanded set if dynamic fetch fails
                leagues = ["PL", "PD", "SA", "BL1", "FL1", "DED", "PPL", "CL", "ELC"]
            
            team_info = None
            
            for league_code in leagues:
                try:
                    data = get_standings(league_code)
                    table = data['standings'][0]['table']
                    
                    # Search for team by name (flexible matching)
                    for team_data in table:
                        team_names = [
                            team_data['team']['name'].lower(),
                            team_data['team']['shortName'].lower(),
                            team_data['team']['tla'].lower()
                        ]
                        
                        if any(team_name.lower() in name or name in team_name.lower() for name in team_names):
                            team_info = {
                                "team": team_data['team']['name'],
                                "league": data['competition']['name'],
                                "position": team_data['position'],
                                "points": team_data['points'],
                                "played": team_data['playedGames'],
                                "won": team_data['won'],
                                "draw": team_data['draw'],
                                "lost": team_data['lost'],
                                "goals_for": team_data['goalsFor'],
                                "goals_against": team_data['goalsAgainst'],
                                "goal_difference": team_data['goalDifference']
                            }
                            return team_info
                except:
                    continue
            
            if not team_info:
                return {"error": f"Team '{team_name}' not found in available leagues"}
            
        except Exception as e:
            return {"error": f"Team lookup error: {e}"}
    else:
        return {"error": "Football data service not available"}

@app.get("/competitions")
def competitions():
    """List competitions with fallback"""
    if FOOTBALL_DATA_AVAILABLE:
        try:
            return {"competitions": list_competitions()}
        except Exception as e:
            return {"error": f"Competitions service error: {e}"}
    else:
        return {"error": "Football data service not available"}

@app.get("/table")
def table(league: Optional[str] = None):
    """Return a formatted league table from Football-Data (quick test endpoint)."""
    if not FOOTBALL_DATA_AVAILABLE:
        return {"error": "Football data service not available"}
    try:
        code = resolve_competition(league or 'PL')
        std = get_standings(code)
        from llm.openrouter_llm import format_league_table
        return {"table": format_league_table(std, code), "code": code}
    except Exception as e:
        return {"error": f"Failed to get table: {e}"}

@app.get("/api-football/leagues")
def api_football_leagues():
    """List leagues from API-Football with pagination handling (best-effort)."""
    try:
        from api.api_football import get_all_leagues
        leagues = get_all_leagues()
        return {"count": len(leagues), "leagues": leagues[:20]}
    except Exception as e:
        return {"error": f"API-Football leagues error: {e}"}

@app.get("/api-football/teams")
def api_football_teams(league: Optional[int] = None, season: Optional[int] = None):
    """List teams for a league+season using API-Football, paginated under the hood."""
    try:
        from api.api_football import get_all_teams_for_league, find_league_id_by_code_or_name
        if league is None and season is not None:
            return {"error": "Provide league id or code/name with season"}
        league_id = league
        if league_id is None:
            return {"error": "Missing 'league' query param (id or code/name)"}
        if isinstance(league_id, str):
            # Try resolve code/name to id
            league_id = find_league_id_by_code_or_name(league_id, season=season)
        if not league_id:
            return {"error": "Could not resolve league"}
        if season is None:
            return {"error": "Missing 'season' query param (e.g., 2024)"}
        teams = get_all_teams_for_league(int(league_id), int(season))
        return {"count": len(teams), "teams": teams[:50]}
    except Exception as e:
        return {"error": f"API-Football teams error: {e}"}

@app.post("/ask")
async def ask_question(request: Request):
    """Enhanced ask endpoint with graceful degradation"""
    try:
        data = await request.json()
        question = (data.get("question") or "").strip()
        conv_id = data.get("conversation_id") or str(uuid.uuid4())
        
        if not question:
            return JSONResponse({"error": "Question is required"}, status_code=400)
        
        # Get conversation history
        history = CONV_STORE.get(conv_id, [])
        context = CONV_CTX.get(conv_id, {})
        
        # Try advanced processing if available
        if intelligence_coordinator and knowledge_engine and entity_resolver:
            try:
                # Advanced processing
                answer = await process_advanced_query(question, conv_id, history, context)
                mode = "advanced"
            except Exception as e:
                # Fall back to basic processing
                answer = await process_basic_query(question, history, context)
                answer += f"\n\n_Advanced processing failed, using basic mode: {e}_"
                mode = "fallback"
        else:
            # Basic processing
            answer = await process_basic_query(question, history, context)
            mode = "basic"
        
        # Update conversation history
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        
        if len(history) > 2 * MAX_TURNS:
            history = history[-2 * MAX_TURNS:]
        
        CONV_STORE[conv_id] = history
        # Persist updated context if modified by processing
        CONV_CTX[conv_id] = context

        
        
        return JSONResponse({
            "answer": answer,
            "conversation_id": conv_id,
            "mode": mode
        })
        
    except Exception as e:
        return JSONResponse({"error": f"Failed to process question: {e}"}, status_code=500)

async def process_advanced_query(question: str, conv_id: str, history: List[Dict], context: Dict) -> str:
    """Process with advanced AI components"""
    try:
        # Live scores (deterministic, avoids LLM)
        def _detect_live(q: str) -> bool:
            ql = (q or "").lower()
            return ("live" in ql) and ("score" in ql or "scores" in ql)

        if _detect_live(question) and LIVESCORE_AVAILABLE:
            try:
                raw = get_live_scores_lsa()
                # Prefer a competition keyword from question
                try:
                    from llm.openrouter_llm import guess_league_code_from_text, _league_name_for_code
                    code = guess_league_code_from_text(question)
                    kw = _league_name_for_code(code) if code else None
                except Exception:
                    kw = None
                # If no mapped league, use last words as a free keyword (country/league name)
                if not kw:
                    import re as _re
                    tokens = [w for w in _re.findall(r"[A-Za-z0-9]+", question.lower()) if w not in {"show","me","live","scores","score","from","in","for","of","please","pls"}]
                    kw = " ".join(tokens[-3:]) if tokens else None
                txt = format_live_scores_payload(raw, kw)
                return _merge_follow_up(txt, question, context)
            except Exception:
                pass

        # Use intelligence engine for processing
        result = await intelligence_coordinator.process_intelligent_query(
            query=question,
            conversation_id=conv_id,
            user_preferences=context.get('preferences', {})
        )
        
        intent = result.get('intent', {})
        answer = result.get('answer', '')
        
        # If no answer from advanced processing, fall back to basic
        if not answer or len(answer.strip()) < 10:
            return await process_basic_query(question, history, context)
        
        # Return conversational answer with a contextual follow-up (single)
        return _merge_follow_up(answer, question, context)
    
    except Exception as e:
        # Fall back to basic processing if advanced fails
        basic_response = await process_basic_query(question, history, context)
        return f"{basic_response}\n\n_Advanced processing failed, using basic mode: {str(e)[:100]}_"

def _clean_team_text(s: str) -> str:
    t = re.sub(r"[?!.]", " ", s.lower())
    t = re.sub(r"\b(and|what about|about|pls|please|team)\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _code_to_name(code: str) -> str:
    try:
        from llm.openrouter_llm import _league_name_for_code
        return _league_name_for_code(code) or code
    except Exception:
        return code

def _enrich_question_with_focus(question: str, focus_code: Optional[str]) -> str:
    """If the question uses deictic phrases like 'this league', inline the resolved league name."""
    if not focus_code:
        return question
    try:
        name = _code_to_name(focus_code)
        q = question
        lowers = q.lower()
        replacements = [
            ('this league', name),
            ('this competition', name),
            ('this table', name),
            ('this season', name),
            ('the league', name),
        ]
        for pat, rep in replacements:
            # replace case-insensitively while preserving original casing length
            q = re.sub(rf"(?i)\b{re.escape(pat)}\b", rep, q)
        # as a fallback, annotate explicitly for the LLM
        if q == question:
            q = f"{question} (league: {name})"
        return q
    except Exception:
        return question

def _extract_team_from_stat_answer(answer: str, label: Optional[str] = None) -> Optional[str]:
    try:
        txt = (answer or "").strip()
        if not txt:
            return None
        import re
        if label:
            m = re.match(rf"^(.*?)\s+{re.escape(label)}\s*:\s*", txt, flags=re.I)
            if m:
                return m.group(1).strip()
        # generic fallback: take text before the last colon and strip common labels
        if ':' in txt:
            left = txt.split(':', 1)[0]
            # remove known labels if present
            left = re.sub(r"\b(GD|points|position|wins|draws|losses|played|goals scored|goals conceded|form)\b","", left, flags=re.I).strip()
            return left or None
    except Exception:
        return None
    return None

def _generate_follow_up(question: str, answer: str, context: Dict) -> str:
    """Create a short, contextual follow‑up suggestion."""
    try:
        from llm.openrouter_llm import (
            guess_league_code_from_text,
            _league_name_for_code,
            _detect_metric,
            _detect_table_request,
            _strip_league_and_stat_words,
        )
    except Exception:
        return "\n\nWould you like a prediction or another stat?"

    code = guess_league_code_from_text(question) or context.get('last_league_code')
    league = _league_name_for_code(code) if code else None

    # Table → suggest prediction or deeper insights
    if _detect_table_request(question):
        if league:
            return f"\n\nWould you like a quick prediction for the {league} title race or a specific team's outlook?"
        return "\n\nWant a quick prediction for the title race or a specific team?"

    # Team stat → suggest related metrics or next match prediction
    metric = _detect_metric(question)
    team = _extract_team_from_stat_answer(answer, context.get('last_metric_label'))
    if not team:
        try:
            team_guess = _strip_league_and_stat_words(question).strip()
            if len(team_guess) > 1:
                team = team_guess
        except Exception:
            pass

    if metric and team:
        base = f"Want {team}'s recent form or a prediction for their next match"
        if league:
            base += f" in {league}"
        return f"\n\n{base}?"

    # Leader/position styled queries
    ql = question.lower()
    if any(k in ql for k in ["who is 1st", "who is first", "who's first", "top of", "leader "]):
        if league:
            return f"\n\nWould you like a short prediction for the {league} leader or a comparison with the runner‑up?"
        return "\n\nWant a short prediction for the leader or a comparison with the runner‑up?"

    if league:
        return f"\n\nWould you like another stat or a prediction in {league}?"
    return "\n\nWould you like another stat, comparison, or a prediction?"

def _try_followup_stat(question: str, history: List[Dict], context: Dict) -> Optional[str]:
    try:
        from llm.openrouter_llm import (
            _detect_metric,
            guess_league_code_from_text,
            try_answer_stat_from_football_data_with_data,
            try_answer_stat_from_football_data,
            _strip_league_and_stat_words,
        )
    except Exception:
        return None

    q = question.strip().lower()
    # If explicit team text exists in this question, do NOT treat as follow-up
    try:
        explicit_team = _strip_league_and_stat_words(question).strip()
    except Exception:
        explicit_team = ""
    if explicit_team:
        return None
    # True follow-up detection (elliptical/pronouns)
    is_followup = (
        q.startswith('and ') or q.startswith("what about") or q.startswith('about ')
        or ' they ' in f" {q} " or ' them ' in f" {q} " or ' their ' in f" {q} "
        or len(q.split()) <= 3
    )
    if not is_followup:
        return None

    # Get team guess from current message; if missing, use last team from context/history
    team_guess = _clean_team_text(question)
    if not team_guess:
        team_guess = context.get('last_team_name')
    if not team_guess and history:
        # parse last assistant stat answer
        for m in reversed(history):
            if m.get('role') == 'assistant':
                team = _extract_team_from_stat_answer(m.get('content',''), context.get('last_metric_label'))
                if team:
                    team_guess = team
                    break
    if not team_guess:
        return None

    # League: prefer current question, else context, else last user message
    league_code = guess_league_code_from_text(question) or context.get('last_league_code')
    if not league_code:
        for m in reversed(history):
            if m.get('role') == 'user':
                league_code = guess_league_code_from_text(m.get('content','') or '')
                if league_code:
                    break

    # Metric: prefer current question, else context, else last user metric
    metric = _detect_metric(question)
    metric_key = None
    metric_label = None
    if metric:
        metric_key, metric_label = metric
    else:
        metric_key = context.get('last_metric_key')
        metric_label = context.get('last_metric_label')
    if not metric_key or not metric_label:
        for m in reversed(history):
            mk = _detect_metric(m.get('content','') or '')
            if mk:
                metric_key, metric_label = mk
                break

    if not league_code or not metric_key:
        return None

    # Build synthetic question and answer via existing stat path
    synthetic = f"{team_guess} {metric_label} in {_code_to_name(league_code)}"
    try:
        return try_answer_stat_from_football_data(synthetic)
    except Exception:
        return None


async def process_basic_query(question: str, history: List[Dict], context: Dict) -> str:
    """Process with basic components"""
    if LLM_AVAILABLE:
        try:
            # Live scores intent (LLM-free fast path)
            def _detect_live(q: str) -> bool:
                ql = (q or "").lower()
                return ("live" in ql) and ("score" in ql or "scores" in ql)

            if _detect_live(question) and LIVESCORE_AVAILABLE:
                try:
                    from llm.openrouter_llm import guess_league_code_from_text, _league_name_for_code
                    code = guess_league_code_from_text(question)
                    league_kw = _league_name_for_code(code) if code else None
                except Exception:
                    league_kw = None
                # Fallback: extract a free-text competition keyword from the question
                def _live_kw_from_text(q: str) -> str:
                    import re as _re
                    bad = {
                        'show','me','the','please','pls','live','score','scores','from','in','for','of','today','now','match','matches'
                    }
                    words = [_w for _w in _re.findall(r"[A-Za-z0-9]+", q.lower()) if _w not in bad]
                    return " ".join(words[-3:])  # last few words often contain the cup/league name
                free_kw = _live_kw_from_text(question) if not league_kw else None
                try:
                    raw = get_live_scores_lsa()
                    # Reuse the endpoint formatter via local functions for consistency
                    def _minute(m: dict) -> str:
                        for k in ("minute","time","timer","status"):
                            v = (m.get(k) if isinstance(m, dict) else None) or ""
                            if isinstance(v, (int,float)): return f"{int(v)}'"
                            if isinstance(v, str) and v.strip():
                                import re as _re
                                s = v.strip()
                                mm = _re.search(r"(\d{1,3})", s)
                                if mm: return f"{mm.group(1)}'"
                                return s
                        return ""
                    def _team_name(m: dict, idx: int, side: str) -> str:
                        side_l = side.lower()
                        v = m.get(f"{side_l}_team") or m.get(f"{side_l}_name")
                        if isinstance(v, str) and v.strip(): return v.strip()
                        obj = m.get(side_l) or {}
                        if isinstance(obj, dict):
                            n = obj.get("name") or obj.get("team")
                            if isinstance(n, str) and n.strip(): return n.strip()
                        teams = m.get("teams")
                        if isinstance(teams, dict):
                            t = teams.get(side_l)
                            if isinstance(t, str) and t.strip(): return t.strip()
                            if isinstance(t, dict):
                                nm = t.get("name") or t.get("team")
                                if isinstance(nm, str) and nm.strip(): return nm.strip()
                        if isinstance(teams, list) and len(teams) >= 2:
                            t = teams[idx]
                            if isinstance(t, str) and t.strip(): return t.strip()
                            if isinstance(t, dict):
                                nm = t.get("name") or t.get("team") or t.get("title")
                                if isinstance(nm, str) and nm.strip(): return nm.strip()
                        return "Home" if side_l=="home" else "Away"
                    def _score_parts(m: dict) -> tuple[str,str]:
                        hs = str(m.get("home_score") or m.get("homeGoals") or m.get("home", {}).get("goals") or "")
                        as_ = str(m.get("away_score") or m.get("awayGoals") or m.get("away", {}).get("goals") or "")
                        sc = m.get("score") or ""
                        if (not hs or not as_) and isinstance(sc,str) and "-" in sc:
                            try:
                                h,a = sc.replace(" ","").split("-",1)
                                hs = hs or h; as_ = as_ or a
                            except Exception:
                                pass
                        return (hs or "0", as_ or "0")
                    lines = []
                    matches = []
                    if isinstance(raw, dict):
                        matches = raw.get("matches") or raw.get("data", {}).get("match") or raw.get("response") or raw.get("fixtures") or raw.get("events") or []
                    elif isinstance(raw, list):
                        matches = raw
                    for m in matches[:60]:
                        ltxt = str(m.get("league", m.get("competition",""))).lower()
                        if league_kw and league_kw.lower() not in ltxt:
                            continue
                        if (not league_kw) and free_kw and free_kw not in ltxt:
                            continue
                        home = _team_name(m,0,"home"); away = _team_name(m,1,"away")
                        hs,as_ = _score_parts(m)
                        # list scores fallback
                        if (hs=="0" and as_=="0") and isinstance(m.get("teams"), list) and len(m.get("teams"))>=2:
                            t0,t1 = m["teams"][0], m["teams"][1]
                            def _sv(x):
                                if isinstance(x,dict):
                                    for k in ("score","goals","points"): 
                                        if k in x: return str(x[k])
                                return None
                            hs = _sv(t0) or hs; as_ = _sv(t1) or as_
                        minute = _minute(m)
                        line = f"{home} {hs}  {minute} {as_} {away}".strip()
                        lines.append(line)
                    if not lines:
                        lines.append("No live matches currently available.")
                    ans = "\n".join(lines)
                    return _merge_follow_up(ans, question, context)
                except Exception:
                    pass

            # Lightweight follow-up handler (e.g., "and juventus?")
            follow = _try_followup_stat(question, history, context)
            if follow:
                # update conversational context heuristically using last message
                try:
                    from llm.openrouter_llm import _detect_metric, _fd_league_code_from_text
                    mk = _detect_metric(history[-1]['content']) if history else None
                    if mk:
                        context['last_metric_key'], context['last_metric_label'] = mk
                    lc = _fd_league_code_from_text(history[-1]['content']) if history else None
                    if lc:
                        context['last_league_code'] = lc
                except Exception:
                    pass
                return follow

            # If user asked for a league table, short-circuit and print it directly
            try:
                from llm.openrouter_llm import _detect_table_request, guess_league_code_from_text, format_league_table
                if _detect_table_request(question):
                    ql = question.lower()
                    code = guess_league_code_from_text(question)
                    if not code:
                        return "Please specify which league you want the table for (e.g., La Liga, Serie A)."
                    try:
                        std = get_standings(code)
                        # Persist context
                        context['last_league_code'] = code
                        table_txt = format_league_table(std, code)
                        return table_txt + _generate_follow_up(question, table_txt, context)
                    except Exception:
                        # Fallback via API-Football if configured
                        try:
                            from llm.openrouter_llm import _table_via_api_football
                            tbl = _table_via_api_football(code)
                            if tbl:
                                return tbl + _generate_follow_up(question, tbl, context)
                        except Exception:
                            pass
            except Exception:
                pass

            # Try to get data for multiple leagues
            standings = None
            scores = None
            
            if FOOTBALL_DATA_AVAILABLE:
                try:
                    from api.football_data import get_many_standings, list_competitions
                    
                    # Get ALL available leagues dynamically instead of hardcoded subset
                    try:
                        all_competitions = list_competitions()
                        # Extract all league codes from competitions
                        all_league_codes = [comp.get("code") for comp in all_competitions if comp.get("code")]
                        # Filter to get major leagues plus other important ones
                        priority_leagues = ["PL", "PD", "SA", "BL1", "FL1", "DED", "PPL", "CL", "ELC", "CLI"]
                        
                        # Use all available leagues, prioritizing the major ones
                        leagues_to_fetch = []
                        for league in priority_leagues:
                            if league in all_league_codes:
                                leagues_to_fetch.append(league)
                        
                        # Add other available leagues (limit to reasonable number for performance)
                        remaining_leagues = [code for code in all_league_codes
                                           if code not in leagues_to_fetch and code not in ["WC", "EC"]]
                        leagues_to_fetch.extend(remaining_leagues[:10])  # Add up to 10 more leagues
                        
                        print(f"🏆 Fetching standings for {len(leagues_to_fetch)} leagues: {leagues_to_fetch}")
                        standings = get_many_standings(leagues_to_fetch)
                    except Exception as e:
                        print(f"⚠️ Failed to get all competitions, falling back to major leagues: {e}")
                        # Fallback to expanded major leagues if dynamic fetch fails
                        standings = get_many_standings(["PL", "PD", "SA", "BL1", "FL1", "DED", "PPL", "CL"])
                    
                    # Also add individual league access for better data extraction
                    if not standings:
                        standings = {}
                        for league_code in ["PL", "PD", "SA", "BL1", "FL1"]:
                            try:
                                standings[league_code] = get_standings(league_code)
                            except:
                                continue
                                
                except Exception as e:
                    try:
                        # Fallback to expanded major leagues
                        standings = get_many_standings(["PL", "PD", "SA", "BL1", "FL1", "DED", "PPL", "CL"])
                    except:
                        pass
            
            if LIVESCORE_AVAILABLE:
                try:
                    scores = get_live_scores_lsa()
                except:
                    pass
            
            # Check if this is a specific stat query (like goal difference) and try direct answer
            from llm.openrouter_llm import (
                try_answer_stat_from_football_data_with_data,
                try_answer_stat_from_football_data,
                try_answer_league_leader,
            )
            
            # First try with cached data
            stat_answer = None
            if standings:
                stat_answer = try_answer_stat_from_football_data_with_data(question, standings)
            
            # If no result from cached data, try direct API call
            if not stat_answer:
                try:
                    stat_answer = try_answer_stat_from_football_data(question)
                except Exception as e:
                    print(f"Direct API call failed: {e}")
            
            if stat_answer:
                # update conversational context for follow-ups
                try:
                    from llm.openrouter_llm import _detect_metric, guess_league_code_from_text
                    mk = _detect_metric(question)
                    if mk:
                        context['last_metric_key'], context['last_metric_label'] = mk
                    lc = guess_league_code_from_text(question)
                    if lc:
                        context['last_league_code'] = lc
                    # try to capture team name from answer
                    team = _extract_team_from_stat_answer(stat_answer, context.get('last_metric_label'))
                    if team:
                        context['last_team_name'] = team
                except Exception:
                    pass
                return stat_answer + _generate_follow_up(question, stat_answer, context)

            # Leader/top-of-table queries (e.g., "who is 1st in la liga")
            leader = try_answer_league_leader(question)
            if leader:
                return leader + _generate_follow_up(question, leader, context)
            
            # Use LLM with targeted league context when possible
            try:
                from llm.openrouter_llm import guess_league_code_from_text
                focus_code = guess_league_code_from_text(question) or context.get('last_league_code')
            except Exception:
                focus_code = None

            llm_standings = standings or {"message": "Standings unavailable"}
            if focus_code and isinstance(standings, dict) and focus_code in standings:
                llm_standings = {focus_code: standings[focus_code]}
            elif focus_code and FOOTBALL_DATA_AVAILABLE:
                try:
                    # Fetch just the requested league if not in the cached set
                    llm_standings = {focus_code: get_standings(focus_code)}
                except Exception:
                    pass

            # Enrich question with inferred focus when deictic
            q_for_llm = _enrich_question_with_focus(question, focus_code)
            answer = query_llm(
                question=q_for_llm,
                scores=scores or {"message": "Live scores unavailable"},
                standings=llm_standings,
                history=history,
                focus=focus_code or ""
            )

            # Guard: if LLM failed (e.g., invalid model -> 404), fall back to deterministic paths
            if isinstance(answer, str) and answer.strip().lower().startswith('(llm error)'):
                # Try leader fast-path
                leader_fb = try_answer_league_leader(question)
                if leader_fb:
                    return leader_fb + _generate_follow_up(question, leader_fb, context)
                # Try stat with cached data first
                try:
                    stat_fb = None
                    if standings:
                        from llm.openrouter_llm import try_answer_stat_from_football_data_with_data
                        stat_fb = try_answer_stat_from_football_data_with_data(question, standings)
                    if not stat_fb:
                        from llm.openrouter_llm import try_answer_stat_from_football_data
                        stat_fb = try_answer_stat_from_football_data(question)
                    if stat_fb:
                        return stat_fb + _generate_follow_up(question, stat_fb, context)
                except Exception:
                    pass
                # As a last resort, provide a concise notice
                fallback_msg = "I couldn’t reach the AI model right now. I can still help with tables and stats—try asking for a specific league table or team stat."
                return fallback_msg + _generate_follow_up(question, fallback_msg, context)
            
            # If the user asked for a table/standings and LLM didn't return one, force-generate it
            try:
                from llm.openrouter_llm import _detect_table_request, guess_league_code_from_text, format_league_table
                if _detect_table_request(question):
                    # detect league code robustly (typos + FD index)
                    code = guess_league_code_from_text(question)
                    if not code:
                        return "Please specify which league you want the table for (e.g., La Liga, Serie A)."
                    try:
                        std = get_standings(code)
                        forced = format_league_table(std, code)
                        # If no clear table lines in answer, replace with forced table
                        if 'Pos' not in answer or '-'*10 not in answer:
                            return forced + _generate_follow_up(question, forced, context)
                        # Else append
                        answer = f"{answer}\n\n{forced}"
                    except Exception:
                        pass
            except Exception:
                pass

            # Return clean conversational answer with a contextual follow-up
            return answer + _generate_follow_up(question, answer, context)
            
        except Exception as e:
            return f"I understand you're asking about football, but I'm having trouble accessing the AI services right now. Error: {e}"
    else:
        # Fallback to simple responses
        return generate_simple_response(question)

def generate_simple_response(question: str) -> str:
    """Generate simple responses when LLM unavailable"""
    q_lower = question.lower()
    
    if any(word in q_lower for word in ['standings', 'table', 'position']):
        return "You can check standings using the 'Standings' button or visit football-data.org for current league tables."
    
    elif any(word in q_lower for word in ['scores', 'live', 'results']):
        return "You can check live scores using the 'Live Scores' button or visit your favorite sports website."
    
    elif any(word in q_lower for word in ['formation', 'tactic', 'strategy']):
        return "For tactical analysis, I'd recommend checking team-specific sports analysis websites or football forums."
    
    elif any(word in q_lower for word in ['prediction', 'predict', 'who will win']):
        return "Match predictions require complex analysis. Try sports betting sites or football analysis platforms for predictions."
    
    else:
        return f"I received your question about: '{question}'. Unfortunately, advanced AI processing is currently unavailable. Please use the buttons above for live scores and standings, or try rephrasing your question."

if __name__ == "__main__":
    import uvicorn
    print(" Starting Working Football Assistant...")
    print(" Access the web interface at: http://localhost:8000/chat")
    uvicorn.run(app, host="0.0.0.0", port=8000)
