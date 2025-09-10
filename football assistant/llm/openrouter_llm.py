import os,re,time,requests
from typing import Tuple
from typing import Any, List, Dict, Optional
from api.football_data import get_standings

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.output_parsers import StrOutputParser
    _LC_OK = True
except Exception:
    _LC_OK = False

OPENROUTER_KEY   = os.getenv("OPENROUTER_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "")
BASE_URL         = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
SITE_URL         = os.getenv("OPENROUTER_SITE_URL")
APP_NAME         = os.getenv("OPENROUTER_APP_NAME", "Football Assistant")

_SYSTEM_MSG = """You are a concise soccer assistant.

Guidelines:
- Use chat history for continuity but prioritize the current question.
- If the user asks for standings/table without a league, ask which league first.
- For tactics/formation/rules: answer directly and briefly, with an example if helpful.
- Prefer provided factual data (standings/live) over assumptions; never invent numbers.
- Keep answers focused and 1–2 sentences unless the user requests a table.
"""


_chain = None

def _hist_to_messages(history: List[Dict[str, str]], max_turns: int = 6) -> List[Any]:
    msgs: List[Any] = []
    # take last N user/assistant pairs
    h = history[-(max_turns*2):] if history else []
    for m in h:
        role = (m.get("role") or "").lower()
        content = m.get("content", "")
        if not content:
            continue
        if role == "user":
            msgs.append(HumanMessage(content=content))
        else:
            msgs.append(AIMessage(content=content))
    return msgs

def _compact_one(std: dict, max_rows: int = 6) -> str:
    try:
        comp = (std.get("competition") or {}).get("name", "")
        season = ((std.get("season") or {}).get("startDate") or "")[:4]
        standings = std.get("standings") or []
        total = next((t for t in standings if t.get("type") == "TOTAL"), standings[0] if standings else {})
        rows = total.get("table") or []
        lines = [f"{r.get('position')}. {r.get('team',{}).get('name')} - {r.get('points')} pts"
                 for r in rows[:max_rows]]
        header = f"{comp} {season}".strip()
        return (header + "\n" if header else "") + "\n".join(lines)
    except Exception:
        return str(std)[:800]

def _compact_standings_multi(s: Any) -> str:
    """Accept a single standings JSON OR {code: standings} dict."""
    if isinstance(s, dict) and "standings" not in s and "competition" not in s:
        parts = []
        for code, std in s.items():
            parts.append(f"[{code}]\n{_compact_one(std)}")
        return "\n\n".join(parts)[:3500]
    return _compact_one(s)

def _build_chain():
    if not _LC_OK or not OPENROUTER_KEY or not OPENROUTER_MODEL:
        class _Stub:
            def invoke(self, _ctx):
                if not _LC_OK:        return "(LLM not configured) Install langchain-openai & langchain-community."
                if not OPENROUTER_KEY:return "(LLM not configured) Set OPENROUTER_KEY."
                return "(LLM not configured) Set OPENROUTER_MODEL to a valid model id."
        return _Stub()
    llm = ChatOpenAI(
        api_key=OPENROUTER_KEY, base_url=BASE_URL, model=OPENROUTER_MODEL,
        temperature=0.2, max_tokens=300,
        default_headers={**({"HTTP-Referer": SITE_URL} if SITE_URL else {}), "X-Title": APP_NAME},
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_MSG),
        MessagesPlaceholder("history"),
        ("human", "Question: {question}\n\nFocus league: {focus}\n\nStandings (compact):\n{standings}\n\nLive Scores:\n{live_scores}")
    ])
    return prompt | llm | StrOutputParser()

def query_llm(question: str, scores: Any, standings: Any,
              history: Optional[List[Dict[str, str]]] = None, focus: Optional[str] = None) -> str:
    global _chain
    if _chain is None:
        _chain = _build_chain()
    # Build role-aware history and compact standings text
    msgs = _hist_to_messages(history or []) if _LC_OK else []
    std_text = _compact_standings_multi(standings)
    try:
        out = _chain.invoke({
            "history": msgs, "live_scores": scores, "standings": std_text,
            "question": question, "focus": focus or ""
        })
        return (out or "").strip() or "(no answer produced)"
    except Exception as e:
        return f"(LLM error) {e}"
# ===== Football-Data competitions index (cached) =====
_FD_INDEX: List[Dict] = []
_FD_INDEX_TS: float = 0.0

def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _fd_build_index(force: bool = False) -> List[Dict]:
    """
    Cache all competitions from Football-Data so we can resolve any league from text.
    """
    global _FD_INDEX, _FD_INDEX_TS
    if _FD_INDEX and not force and (time.time() - _FD_INDEX_TS < 6 * 3600):
        print(f"DEBUG INDEX DEBUG: Using cached index with {len(_FD_INDEX)} competitions")
        return _FD_INDEX

    tok = os.getenv("FOOTBALL_DATA_KEY")
    if not tok:
        print(f"DEBUG INDEX DEBUG: No FOOTBALL_DATA_KEY found")
        _FD_INDEX, _FD_INDEX_TS = [], time.time()
        return _FD_INDEX

    try:
        url = "https://api.football-data.org/v4/competitions"
        print(f"DEBUG INDEX DEBUG: Fetching competitions from {url}")
        r = requests.get(url, headers={"X-Auth-Token": tok}, timeout=20)
        r.raise_for_status()
        comps = r.json().get("competitions", []) or []
        print(f"DEBUG INDEX DEBUG: Retrieved {len(comps)} competitions from API")
        _FD_INDEX = [{
            "code": c.get("code"),
            "name": c.get("name"),
            "area": (c.get("area") or {}).get("name"),
            "type": c.get("type"),
        } for c in comps if c.get("code")]
        _FD_INDEX_TS = time.time()
        print(f"DEBUG INDEX DEBUG: Built index with {len(_FD_INDEX)} valid competitions")
        return _FD_INDEX
    except Exception as e:
        print(f"DEBUG INDEX DEBUG: Failed to build index: {e}")
        _FD_INDEX, _FD_INDEX_TS = [], time.time()
        return _FD_INDEX

def _fd_league_code_from_text(question: str) -> Optional[str]:
    """
    Resolve a league code by fuzzy-matching any competition name in the user's text.
    Works for 'la liga', 'laliga ea sports', 'bundesliga', etc.
    """
    txt = _norm(question)
    print(f"DEBUG LEAGUE DEBUG: Normalized question: '{txt}'")
    idx = _fd_build_index()
    print(f"DEBUG LEAGUE DEBUG: Index has {len(idx)} competitions")
    if not idx:
        return None

    # score best match by overlap of tokens
    best = (0, None)
    top_matches = []
    for it in idx:
        name = _norm(it["name"])
        area = _norm(it.get("area") or "")
        typ  = (it.get("type") or "").lower()
        score = 0
        # direct contain / overlap
        if name in txt or txt in name:
            score += 60
        # country/area hints e.g. "spain", "england"
        if area and (area in txt or txt in area):
            score += 25
        if typ == "league":
            score += 5
        if score > 0:
            top_matches.append((score, it["code"], name, area))
        if score > best[0]:
            best = (score, it["code"])
    
    # Show top 5 matches for debugging
    top_matches.sort(reverse=True)
    print(f"DEBUG LEAGUE DEBUG: Top 5 matches:")
    for i, (score, code, name, area) in enumerate(top_matches[:5]):
        print(f"  {i+1}. {code}: '{name}' (area: '{area}') - score: {score}")
    
    print(f"DEBUG LEAGUE DEBUG: Best match: {best}")
    return best[1]

def _strip_league_and_stat_words(question: str) -> str:
    """
    Remove common league tokens & stat words so the remainder is likely the team text.
    """
    stop = {
        # generic
        "what","whats","what's","is","are","the","a","an","of","for","in","on","please","pls","tell","give","me","team","how","many",
        # leagueish words
        "league","table","standings","division","liga","bundesliga","premier","preimer","premire","premeir","serie","seria","ligue","championship","la","la liga","laliga","eredivisie","primeira",
        # statish words
        "gd","goal","goals","difference","goal-difference","goal difference","points","pts","rank","position","place","wins","win","draws","losses","lost","played","matches","games","gf","ga","scored","conceded","against"
    }
    # remove multiword first
    out = question
    for phrase in ["la liga","goal difference","goal-difference","serie a","seria a","premier league","preimer league","premire league","premeir league","ligue 1","primeira liga","goals scored","goals conceded","how many goals"]:
        out = re.sub(r"(?i)\b" + re.escape(phrase) + r"\b", " ", out)
    
    # Clean up possessives: "real madrid's" -> "real madrid"
    out = re.sub(r"'s\b", "", out)
    
    # then tokens
    words = [w for w in re.findall(r"[A-Za-z0-9']+", out) if _norm(w) not in stop]
    result = " ".join(words).strip()
    return result

def _fd_find_team_row(standings: Dict, team_text: str) -> Optional[Dict]:
    """
    Find team row by exact → contains match against table team names.
    """
    want = _norm(team_text)
    total = next((s for s in standings.get("standings", []) if s.get("type") == "TOTAL"), None)
    rows = total.get("table", []) if total else []

    # exact
    for r in rows:
        t = (r.get("team") or {})
        nm = _norm(t.get("name", ""))
        sn = _norm(t.get("shortName", ""))
        tla = _norm(t.get("tla", ""))
        if nm == want:
            return r
    # contains (both ways)
    for r in rows:
        t = (r.get("team") or {})
        nm = _norm(t.get("name", ""))
        sn = _norm(t.get("shortName", ""))
        tla = _norm(t.get("tla", ""))
        if want and (want in nm or nm in want or want in sn or want in tla):
            return r
    return None

_METRIC_ALIASES = {
    "goalDifference": {"gd","goal diff","goal-difference","goal difference"},
    "points": {"points","pts"},
    "position": {"position","rank","place"},
    "won": {"won","wins","win"},
    "draw": {"draw","draws","ties"},
    "lost": {"lost","loss","losses","defeats"},
    "playedGames": {"played","played games","matches","games","apps","appearances","p"},
    "goalsFor": {"gf","goals for","for","scored","goals scored","how many goals","total goals"},
    "goalsAgainst": {"ga","goals against","against","conceded","goals conceded","goals allowed"},
    "form": {"form","recent form","current form"},
}

def _detect_metric(question: str) -> Optional[Tuple[str, str]]:
    """
    Return (canonical_key, pretty_label) from text, or None.
    """
    q = _norm(question)
    q_original = question.lower()

    # Abbreviation-first: handle exact tokens to avoid ambiguity
    tokens = set(q.split())
    if 'gd' in tokens:
        return 'goalDifference', 'GD'
    if 'ga' in tokens or 'against' in tokens or 'conceded' in tokens:
        return 'goalsAgainst', 'goals conceded'
    if 'gf' in tokens or 'scored' in q_original or 'goals for' in q_original:
        return 'goalsFor', 'goals scored'
    
    # Special handling for goals - check context to distinguish scored vs conceded
    if any(word in q_original for word in ["goals", "goal"]):
        if any(word in q_original for word in ["scored", "for", "how many goals"]) and "against" not in q_original and "conceded" not in q_original:
            return "goalsFor", "goals scored"
        elif any(word in q_original for word in ["conceded", "against"]):
            return "goalsAgainst", "goals conceded"
        elif "goals" in q_original and ("scored" not in q_original and "conceded" not in q_original and "against" not in q_original):
            # Default to goals scored if no clear direction
            return "goalsFor", "goals scored"
    
    # Regular detection for other metrics
    for canon, syns in _METRIC_ALIASES.items():
        # Skip goals since we handled them above
        if canon in ["goalsFor", "goalsAgainst"]:
            continue
            
        for s in sorted(syns, key=len, reverse=True):
            if _norm(s) in q.split() or s in q_original:
                label = {"goalDifference":"GD","points":"points","position":"position","won":"wins",
                         "draw":"draws","lost":"losses","playedGames":"played","goalsFor":"goals scored","goalsAgainst":"goals conceded","form":"form"}[canon]
                return canon, label
    return None

def _metric_value(row: Dict, canon: str) -> Optional[int]:
    if canon == "goalDifference":
        gd = row.get("goalDifference")
        if gd is not None: return gd
        return int(row.get("goalsFor", 0)) - int(row.get("goalsAgainst", 0))
    elif canon == "form":
        # Form is a string, return as-is
        return row.get("form", "N/A")
    return row.get(canon)

def guess_league_code_from_text(question: str) -> Optional[str]:
    """Combine typo-tolerant pattern detection with FD index fallback."""
    q_lower = (question or "").lower()
    league_patterns = {
        'PD': ['la liga', 'laliga', 'primera division', 'spain', 'spanish league'],
        'PL': ['premier league', 'preimer league', 'premire league', 'premeir league', 'epl', 'england', 'english premier', 'premier'],
        'BL1': ['bundesliga', 'germany', 'german league', 'deutsche'],
        'SA': ['serie a', 'seria a', 'italy', 'italian league', 'italian serie'],
        'FL1': ['ligue 1', 'france', 'french league', 'ligue'],
        'DED': ['eredivisie', 'netherlands', 'dutch league', 'holland'],
        'PPL': ['primeira liga', 'portugal', 'portuguese league', 'liga portugal'],
        'CL': ['champions league', 'ucl', 'uefa champions', 'champions'],
        'ELC': ['championship', 'english championship', 'efl championship'],
    }
    for league_code, patterns in league_patterns.items():
        if any(p in q_lower for p in patterns):
            return league_code
    # FD index fallback
    code = _fd_league_code_from_text(question)
    if code:
        return code
    # Last resort: try resolve_competition on tokens
    try:
        from api.football_data import resolve_competition
        for w in question.split():
            r = resolve_competition(w)
            if r and len(r) >= 2:
                return r
    except Exception:
        pass
    return None

def format_league_table(standings_data: Dict, league_code: str) -> str:
    """
    Format a complete league table for display
    """
    try:
        competition_name = standings_data.get("competition", {}).get("name", league_code)
        season = standings_data.get("season", {}).get("startDate", "")[:4]
        
        # Get the table data
        standings = standings_data.get("standings", [])
        total_standings = next((s for s in standings if s.get("type") == "TOTAL"), standings[0] if standings else {})
        table = total_standings.get("table", [])
        
        if not table:
            return f"No table data available for {league_code}"
        
        # Create formatted table
        header = f"\n{competition_name} {season} - League Table\n"
        header += "=" * (len(header) - 2) + "\n"
        
        # Table headers
        table_str = f"{'Pos':<3} {'Team':<25} {'P':<3} {'W':<3} {'D':<3} {'L':<3} {'GF':<3} {'GA':<3} {'GD':<4} {'Pts':<3}\n"
        table_str += "-" * 75 + "\n"
        
        # Table rows
        for team_data in table:
            pos = team_data.get('position', 0)
            name = team_data.get('team', {}).get('name', 'Unknown')[:23]  # Truncate long names
            played = team_data.get('playedGames', 0)
            won = team_data.get('won', 0)
            draw = team_data.get('draw', 0)
            lost = team_data.get('lost', 0)
            gf = team_data.get('goalsFor', 0)
            ga = team_data.get('goalsAgainst', 0)
            gd = team_data.get('goalDifference', 0)
            points = team_data.get('points', 0)
            
            table_str += f"{pos:<3} {name:<25} {played:<3} {won:<3} {draw:<3} {lost:<3} {gf:<3} {ga:<3} {gd:<+4} {points:<3}\n"
        
        return header + table_str
        
    except Exception as e:
        return f"Error formatting table for {league_code}: {e}"

def _detect_table_request(question: str) -> bool:
    """
    Detect if user wants to see a complete league table
    """
    q_lower = question.lower()
    table_keywords = [
        "table", "standings", "league table", "show table", "print table",
        "full table", "complete table", "entire table", "all teams"
    ]
    return any(keyword in q_lower for keyword in table_keywords)

def try_answer_stat_from_football_data(question: str) -> Optional[str]:
    """
    STAT-FIRST path (no API-Football needed):
    1) detect league from text (Football-Data competitions)
    2) fetch that league's standings
    3) find team in that table
    4) return the requested stat as one-liner
    """
    print(f"DEBUG STAT DEBUG: Processing question: '{question}'")
    
    # Check if this is a table request first
    if _detect_table_request(question):
        print(f"DEBUG STAT DEBUG: Table request detected")
        # Detect league and return formatted table
        from api.football_data import resolve_competition
        q_lower = question.lower()
        code = None
        
        # Use the same enhanced league detection
        league_patterns = {
            'PD': ['la liga', 'laliga', 'primera division', 'spain', 'spanish league'],
            'PL': ['premier league', 'preimer league', 'premire league', 'premeir league', 'epl', 'england', 'english premier', 'premier'],
            'BL1': ['bundesliga', 'germany', 'german league', 'deutsche'],
            'SA': ['serie a', 'seria a', 'italy', 'italian league', 'italian serie'],
            'FL1': ['ligue 1', 'france', 'french league', 'ligue'],
            'DED': ['eredivisie', 'netherlands', 'dutch league', 'holland'],
            'PPL': ['primeira liga', 'portugal', 'portuguese league', 'liga portugal'],
            'CL': ['champions league', 'ucl', 'uefa champions', 'champions'],
            'ELC': ['championship', 'english championship', 'efl championship'],
        }
        
        for league_code, patterns in league_patterns.items():
            if any(pattern in q_lower for pattern in patterns):
                code = league_code
                break
        
        if not code:
            code = _fd_league_code_from_text(question)
            if not code:
                return "Please specify the league (e.g., La Liga, Premier League)."
        
        print(f"DEBUG STAT DEBUG: Table request for league: {code}")
        
        try:
            std = get_standings(code)
            formatted_table = format_league_table(std, code)
            print(f"DEBUG STAT DEBUG: Generated table for {code}")
            return formatted_table
        except Exception as e:
            print(f"DEBUG STAT DEBUG: Failed to generate table: {e}")
            # Fallback: try API-Football standings if configured
            try:
                tbl = _table_via_api_football(code)
                if tbl:
                    return tbl
            except Exception as _:
                pass
            return f"Unable to retrieve table for {code}: {e}"
    
    metric = _detect_metric(question)
    print(f"DEBUG STAT DEBUG: Detected metric: {metric}")
    if not metric:
        return None
    canon, label = metric

    # Enhanced league detection supporting ALL available leagues
    from api.football_data import resolve_competition
    q_lower = question.lower()
    code = None
    
    # Comprehensive league detection with all major leagues
    league_patterns = {
        'PD': ['la liga', 'laliga', 'primera division', 'spain', 'spanish league'],
        'PL': ['premier league', 'preimer league', 'premire league', 'premeir league', 'epl', 'england', 'english premier', 'premier'],
        'BL1': ['bundesliga', 'germany', 'german league', 'deutsche'],
        'SA': ['serie a', 'seria a', 'italy', 'italian league', 'italian serie'],
        'FL1': ['ligue 1', 'france', 'french league', 'ligue'],
        'DED': ['eredivisie', 'netherlands', 'dutch league', 'holland'],
        'PPL': ['primeira liga', 'portugal', 'portuguese league', 'liga portugal'],
        'CL': ['champions league', 'ucl', 'uefa champions', 'champions'],
        'ELC': ['championship', 'english championship', 'efl championship'],
        'CLI': ['copa libertadores', 'libertadores'],
        'BSA': ['serie a brazil', 'campeonato brasileiro', 'brazilian league'],
        'WC': ['world cup', 'fifa world cup', 'mundial'],
        'EC': ['euro', 'european championship', 'euros', 'uefa euro'],
        'EL': ['europa league', 'uefa europa'],
        'ECL': ['conference league', 'uefa conference'],
        'PD2': ['segunda division', 'la liga 2', 'spanish second'],
        'EL1': ['league one', 'english league one'],
        'EL2': ['league two', 'english league two']
    }
    
    # Check all league patterns
    for league_code, patterns in league_patterns.items():
        if any(pattern in q_lower for pattern in patterns):
            code = league_code
            print(f"DEBUG STAT DEBUG: Pattern detection found {patterns[0]}, using {code}")
            break
    
    if not code:
        # Fallback to the FD index method for other leagues
        code = _fd_league_code_from_text(question)
        print(f"DEBUG STAT DEBUG: FD index fallback detected: {code}")
        
        # If still no good match, try resolve_competition with expanded validation
        if not code:
            for word in question.split():
                resolved = resolve_competition(word)
                # Accept any valid competition code, not just the original 5
                if resolved and len(resolved) >= 2:
                    code = resolved
                    print(f"DEBUG STAT DEBUG: resolve_competition found: {code}")
                    break
    
    print(f"DEBUG STAT DEBUG: Final league code: {code}")
    if not code:
        return None  # can't identify the league

    # fetch table
    try:
        std = get_standings(code)  # your existing football_data.get_standings
        print(f"DEBUG STAT DEBUG: Successfully fetched standings for {code}")
    except Exception as e:
        print(f"DEBUG STAT DEBUG: Failed to fetch standings for {code}: {e}")
        return None

    # extract team
    team_guess = _strip_league_and_stat_words(question)
    print(f"DEBUG STAT DEBUG: Team guess after stripping: '{team_guess}'")
    if not team_guess:
        return None
    row = _fd_find_team_row(std, team_guess)
    print(f"DEBUG STAT DEBUG: Found team row: {bool(row)}")
    if row:
        team_name = (row.get("team") or {}).get("name", "Unknown")
        print(f"DEBUG STAT DEBUG: Team found: {team_name}")
    if not row:
        return None

    val = _metric_value(row, canon)
    print(f"DEBUG STAT DEBUG: Metric value: {val}")
    if val is None:
        return None

    team_name = (row.get("team") or {}).get("name", team_guess)
    result = f"{team_name} {label}: {val}"
    print(f"DEBUG STAT DEBUG: Final result: '{result}'")
    return result

def _league_name_for_code(code: str) -> Optional[str]:
    mapping = {
        'PD': 'La Liga',
        'PL': 'Premier League',
        'SA': 'Serie A',
        'BL1': 'Bundesliga',
        'FL1': 'Ligue 1',
        'DED': 'Eredivisie',
        'PPL': 'Primeira Liga',
        'CL': 'UEFA Champions League',
        'ELC': 'Championship',
    }
    return mapping.get(code)

def _table_via_api_football(code: str) -> Optional[str]:
    """Fallback: fetch standings from API-Football and format similarly."""
    try:
        from api.api_football import find_league_id_by_code_or_name
        from api.api_football import _get as _af_get  # reuse headers/session
    except Exception:
        return None
    name = _league_name_for_code(code) or code
    league_id = find_league_id_by_code_or_name(name)
    if not league_id:
        return None
    # try current season heuristically
    import datetime
    yr = datetime.datetime.now().year
    # API-Football standings endpoint
    try:
        js = _af_get("/standings", {"league": league_id, "season": yr})
        resp = js.get('response') or []
        if not resp:
            return None
        league = (resp[0] or {}).get('league') or {}
        season = str(league.get('season', yr))
        groups = (league.get('standings') or [[]])
        table = groups[0] if groups else []
        # Convert to Football-Data-like minimal structure for reuse
        fd_like = {
            "competition": {"name": league.get('name')},
            "season": {"startDate": f"{season}-08-01"},
            "standings": [
                {"type": "TOTAL", "table": [
                    {
                        "position": row.get('rank'),
                        "team": {"name": (row.get('team') or {}).get('name', '')},
                        "playedGames": row.get('all', {}).get('played'),
                        "won": row.get('all', {}).get('win'),
                        "draw": row.get('all', {}).get('draw'),
                        "lost": row.get('all', {}).get('lose'),
                        "goalsFor": row.get('all', {}).get('goals', {}).get('for'),
                        "goalsAgainst": row.get('all', {}).get('goals', {}).get('against'),
                        "goalDifference": (row.get('goalsDiff') if row.get('goalsDiff') is not None else (
                            (row.get('all', {}).get('goals', {}).get('for') or 0) - (row.get('all', {}).get('goals', {}).get('against') or 0)
                        )),
                        "points": row.get('points'),
                    }
                    for row in table
                ]}
            ]
        }
        return format_league_table(fd_like, code)
    except Exception:
        return None

def try_answer_league_leader(question: str) -> Optional[str]:
    """Answer queries like 'who is 1st in la liga' by fetching top of table."""
    q = question.lower()
    leader_phrases = ["who is 1st", "who is first", "who's first", "who is top", "top of", "who leads", "leader of"]
    if not any(p in q for p in leader_phrases):
        return None

    # Detect league code
    code = None
    league_patterns = {
        'PD': ['la liga', 'laliga', 'primera division', 'spain', 'spanish league'],
        'PL': ['premier league', 'preimer league', 'premire league', 'premeir league', 'epl', 'england', 'english premier', 'premier'],
        'BL1': ['bundesliga', 'germany', 'german league', 'deutsche'],
        'SA': ['serie a', 'seria a', 'italy', 'italian league', 'italian serie'],
        'FL1': ['ligue 1', 'france', 'french league', 'ligue'],
    }
    for league_code, patterns in league_patterns.items():
        if any(p in q for p in patterns):
            code = league_code
            break
    if not code:
        code = _fd_league_code_from_text(question) or 'PL'

    # Try Football-Data first
    try:
        std = get_standings(code)
        total = next((s for s in (std.get('standings') or []) if s.get('type') == 'TOTAL'), None)
        rows = total.get('table') if total else []
        if rows:
            top = rows[0]
            nm = (top.get('team') or {}).get('name')
            pts = top.get('points')
            return f"{nm} are 1st in {code} with {pts} points"
    except Exception:
        pass

    # Fallback via API-Football
    tbl = _table_via_api_football(code)
    if tbl:
        # Extract first line after header and underline
        lines = [l for l in tbl.splitlines() if l.strip()]
        # Find first row like '1  Team ...'
        for l in lines:
            if l.strip().startswith('1'):
                team = l.split(maxsplit=2)[1]
                return f"{team} are 1st in {code}"
    return None
def try_answer_stat_from_football_data_with_data(question: str, standings_data: Dict) -> Optional[str]:
    """
    Enhanced version that uses already-fetched multi-league data instead of making new API calls.
    """
    print(f"DEBUG STAT WITH DATA DEBUG: Processing question: '{question}'")
    
    # Check if this is a table request first
    if _detect_table_request(question):
        print(f"DEBUG STAT WITH DATA DEBUG: Table request detected")
        
        # Detect league
        from api.football_data import resolve_competition
        q_lower = question.lower()
        code = None
        
        league_patterns = {
            'PD': ['la liga', 'laliga', 'primera division', 'spain', 'spanish league'],
            'PL': ['premier league', 'epl', 'england', 'english premier', 'premier'],
            'BL1': ['bundesliga', 'germany', 'german league', 'deutsche'],
            'SA': ['serie a', 'seria a', 'italy', 'italian league', 'italian serie'],
            'FL1': ['ligue 1', 'france', 'french league', 'ligue'],
            'DED': ['eredivisie', 'netherlands', 'dutch league', 'holland'],
            'PPL': ['primeira liga', 'portugal', 'portuguese league', 'liga portugal'],
            'CL': ['champions league', 'ucl', 'uefa champions', 'champions'],
            'ELC': ['championship', 'english championship', 'efl championship'],
        }
        
        for league_code, patterns in league_patterns.items():
            if any(pattern in q_lower for pattern in patterns):
                code = league_code
                break
        
        if not code:
            code = _fd_league_code_from_text(question)
        
        print(f"DEBUG STAT WITH DATA DEBUG: Table request for league: {code}")
        
        if code and code in standings_data:
            try:
                formatted_table = format_league_table(standings_data[code], code)
                print(f"DEBUG STAT WITH DATA DEBUG: Generated table for {code}")
                return formatted_table
            except Exception as e:
                print(f"DEBUG STAT WITH DATA DEBUG: Failed to generate table: {e}")
                # fallback below
        # Fallback: fetch live and/or API-Football standings
        try:
            std = get_standings(code)
            return format_league_table(std, code)
        except Exception as e2:
            print(f"DEBUG STAT WITH DATA DEBUG: Football-Data standings fetch failed: {e2}")
            try:
                tbl = _table_via_api_football(code)
                if tbl:
                    return tbl
            except Exception as _:
                pass
        return f"Table data not available for {code or 'requested league'}"
    
    metric = _detect_metric(question)
    print(f"DEBUG STAT WITH DATA DEBUG: Detected metric: {metric}")
    if not metric:
        return None
    canon, label = metric

    # Enhanced league detection supporting ALL available leagues
    from api.football_data import resolve_competition
    q_lower = question.lower()
    code = None
    
    # Comprehensive league detection with all major leagues
    league_patterns = {
        'PD': ['la liga', 'laliga', 'primera division', 'spain', 'spanish league'],
        'PL': ['premier league', 'epl', 'england', 'english premier', 'premier'],
        'BL1': ['bundesliga', 'germany', 'german league', 'deutsche'],
        'SA': ['serie a', 'seria a', 'italy', 'italian league', 'italian serie'],
        'FL1': ['ligue 1', 'france', 'french league', 'ligue'],
        'DED': ['eredivisie', 'netherlands', 'dutch league', 'holland'],
        'PPL': ['primeira liga', 'portugal', 'portuguese league', 'liga portugal'],
        'CL': ['champions league', 'ucl', 'uefa champions', 'champions'],
        'ELC': ['championship', 'english championship', 'efl championship'],
        'CLI': ['copa libertadores', 'libertadores'],
        'BSA': ['serie a brazil', 'campeonato brasileiro', 'brazilian league'],
        'WC': ['world cup', 'fifa world cup', 'mundial'],
        'EC': ['euro', 'european championship', 'euros', 'uefa euro'],
        'EL': ['europa league', 'uefa europa'],
        'ECL': ['conference league', 'uefa conference'],
        'PD2': ['segunda division', 'la liga 2', 'spanish second'],
        'EL1': ['league one', 'english league one'],
        'EL2': ['league two', 'english league two']
    }
    
    # Check all league patterns
    for league_code, patterns in league_patterns.items():
        if any(pattern in q_lower for pattern in patterns):
            code = league_code
            print(f"DEBUG STAT WITH DATA DEBUG: Pattern detection found {patterns[0]}, using {code}")
            break
    
    if not code:
        # Try FD index method and resolve_competition as fallbacks
        code = _fd_league_code_from_text(question)
        if not code:
            for word in question.split():
                resolved = resolve_competition(word)
                if resolved and len(resolved) >= 2:
                    code = resolved
                    print(f"DEBUG STAT WITH DATA DEBUG: resolve_competition found: {code}")
                    break
    
    print(f"DEBUG STAT WITH DATA DEBUG: Final league code: {code}")
    if not code:
        print(f"DEBUG STAT WITH DATA DEBUG: No league detected, returning None")
        return None

    # Use the already-fetched standings data
    if not standings_data or code not in standings_data:
        print(f"DEBUG STAT WITH DATA DEBUG: No data for {code} in standings_data")
        return None
    
    std = standings_data[code]
    print(f"DEBUG STAT WITH DATA DEBUG: Using cached standings for {code}")

    # extract team
    team_guess = _strip_league_and_stat_words(question)
    print(f"DEBUG STAT WITH DATA DEBUG: Team guess after stripping: '{team_guess}'")
    if not team_guess:
        return None
    
    row = _fd_find_team_row(std, team_guess)
    print(f"DEBUG STAT WITH DATA DEBUG: Found team row: {bool(row)}")
    if row:
        team_name = (row.get("team") or {}).get("name", "Unknown")
        print(f"DEBUG STAT WITH DATA DEBUG: Team found: {team_name}")
    if not row:
        return None

    val = _metric_value(row, canon)
    print(f"DEBUG STAT WITH DATA DEBUG: Metric value: {val}")
    if val is None:
        return None

    team_name = (row.get("team") or {}).get("name", team_guess)
    result = f"{team_name} {label}: {val}"
    print(f"DEBUG STAT WITH DATA DEBUG: Final result: '{result}'")
    return result
    
