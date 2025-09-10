import os
import time
from typing import Any, Dict, List, Optional

import requests


BASE_URL = "https://v3.football.api-sports.io"
API_KEY = os.getenv("API_FOOTBALL_KEY")


def _headers() -> Dict[str, str]:
    key = API_KEY or os.getenv("API_FOOTBALL_KEY")
    return {"x-apisports-key": key} if key else {}


def _get(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 25) -> Dict[str, Any]:
    if not _headers():
        return {"error": "Set API_FOOTBALL_KEY in your environment"}
    r = requests.get(f"{BASE_URL}{path}", headers=_headers(), params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_live_scores() -> Dict[str, Any]:
    """Return all live fixtures across all leagues."""
    return _get("/fixtures", {"live": "all"})


def list_leagues(page: int = 1) -> Dict[str, Any]:
    """List leagues (single page). See get_all_leagues() for pagination."""
    return _get("/leagues", {"page": page})


def get_all_leagues(max_pages: int = 50, sleep_sec: float = 0.2) -> List[Dict[str, Any]]:
    """
    Retrieve all leagues across pages (best-effort, respects basic rate-limits).
    Returns a flat list of league objects from the API under response[].
    """
    out: List[Dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        try:
            data = list_leagues(page)
            items = data.get("response") or []
            if not items:
                break
            out.extend(items)
            time.sleep(sleep_sec)
        except Exception:
            break
    return out


def list_teams(league_id: Optional[int] = None, season: Optional[int] = None, page: int = 1) -> Dict[str, Any]:
    """
    List teams (single page). If league_id/season provided, filters to a competition season.
    """
    params: Dict[str, Any] = {"page": page}
    if league_id is not None:
        params["league"] = league_id
    if season is not None:
        params["season"] = season
    return _get("/teams", params)


def get_all_teams_for_league(league_id: int, season: int, max_pages: int = 50, sleep_sec: float = 0.2) -> List[Dict[str, Any]]:
    """
    Retrieve all teams for a given league and season across pages.
    Returns a flat list of team objects under response[].
    """
    out: List[Dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        try:
            data = list_teams(league_id=league_id, season=season, page=page)
            items = data.get("response") or []
            if not items:
                break
            out.extend(items)
            time.sleep(sleep_sec)
        except Exception:
            break
    return out


def find_league_id_by_code_or_name(code_or_name: str, season: Optional[int] = None) -> Optional[int]:
    """
    Best-effort helper to map a short code or name to an API-Football league id.
    This scans get_all_leagues() and tries to match by code/name/country.
    """
    want = (code_or_name or "").strip().lower()
    if not want:
        return None
    leagues = get_all_leagues()
    for item in leagues:
        league = (item or {}).get("league") or {}
        name = (league.get("name") or "").lower()
        type_ = (league.get("type") or "").lower()
        code = ((league.get("code") or "").lower())
        country = ((item.get("country") or {}).get("name") or "").lower()
        if want in {code, name} or want in country or want in f"{country} {name}".strip():
            # Optional: verify season availability
            if season is not None:
                seasons = (league.get("seasons") or [])
                if not any(s.get("year") == season for s in seasons):
                    continue
            return league.get("id")
    return None
