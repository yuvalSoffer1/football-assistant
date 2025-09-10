# football_data.py
import os
from functools import lru_cache
from typing import Dict, List, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass  

API_KEY = os.getenv("FOOTBALL_DATA_KEY")  
BASE = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY} if API_KEY else {}

def _get(path: str, **params) -> Dict[str, Any]:
    """
    Small helper around HTTP GET. We import 'requests' lazily here so that
    just importing this module never fails if 'requests' is missing.
    """
    try:
        import requests  
    except Exception as e:
        raise ImportError(
            "The 'requests' package is required. Run: pip install requests"
        ) from e

    resp = requests.get(f"{BASE}{path}", headers=HEADERS, params=params or None, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------- Competitions / mapping ----------

@lru_cache(maxsize=4)
def list_competitions(plan: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Return competitions from the API.
    plan examples: TIER_ONE, TIER_TWO, TIER_THREE, TIER_FOUR
    """
    params = {"plan": plan} if plan else {}
    data = _get("/competitions", **params)
    return data.get("competitions", [])

@lru_cache(maxsize=4)
def competition_map(plan: Optional[str] = None) -> Dict[str, str]:
    """
    Map friendly names/aliases -> API codes (e.g., 'la liga' -> 'PD').
    Uses API data + handy aliases.
    """
    m: Dict[str, str] = {}
    for c in list_competitions(plan):
        code = c.get("code")
        name = c.get("name", "")
        area = (c.get("area") or {}).get("name", "")
        for k in {code, name, f"{area} {name}", name.lower()}:
            if k:
                m[str(k).lower()] = code

    # Helpful aliases
    m.update({
        "premier league": "PL", "epl": "PL",
        "la liga": "PD", "laliga": "PD", "primera division": "PD",
        "bundesliga": "BL1",
        "serie a": "SA",
        "ligue 1": "FL1",
        "eredivisie": "DED",
        "primeira liga": "PPL",
        "champions league": "CL", "ucl": "CL",
    })
    return m

def resolve_competition(q: Optional[str], plan: Optional[str] = None) -> str:
    """Accept code/name/alias and return a valid competition code (default PL)."""
    if not q:
        return "PL"
    ql = q.lower()
    m = competition_map(plan)
    if ql in m:
        return m[ql]
    for alias, code in m.items():
        if ql in alias:
            return code
    return q.upper()  # allow direct codes like PD/SA/BL1/FL1/CL

# ---------- Standings / Teams / Matches ----------

@lru_cache(maxsize=128)
def get_standings(code: str = "PL", season: Optional[int] = None) -> Dict[str, Any]:
    params = {} if season is None else {"season": season}
    return _get(f"/competitions/{code}/standings", **params)

def get_many_standings(leagues: Optional[List[str]] = None,
                       season: Optional[int] = None) -> Dict[str, Any]:
    """
    Return {code: standings_json} for many leagues.
    If leagues is None, default to top-5.
    """
    codes = [resolve_competition(x) for x in leagues] if leagues else ["PL", "PD", "SA", "BL1", "FL1"]
    out: Dict[str, Any] = {}
    for code in codes:
        try:
            out[code] = get_standings(code, season)
        except Exception as e:
            out[code] = {"error": str(e)}
    return out

def get_teams(code: str) -> Dict[str, Any]:
    """All teams in a competition."""
    return _get(f"/competitions/{code}/teams")

def get_team(team_id: int | str) -> Dict[str, Any]:
    return _get(f"/teams/{team_id}")

def get_team_matches(team_id: int | str,
                     status: Optional[str] = None,
                     limit: Optional[int] = None) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if status: params["status"] = status  # e.g. "SCHEDULED,IN_PLAY,FINISHED"
    if limit:  params["limit"]  = limit
    return _get(f"/teams/{team_id}/matches", **params)
