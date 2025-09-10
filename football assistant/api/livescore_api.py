# livescore_api.py
import os
import requests

BASE = "https://livescore-api.com/api-client"
KEY = os.getenv("LIVESCORE_API_KEY")
SECRET = os.getenv("LIVESCORE_API_SECRET")

def _get(path: str, **params):
    """
    Call LiveScore API with ?key=f"{KEY}&secret={SECRET}"
    Docs: https://live-score-api.com/football-api
    """
    if not KEY or not SECRET:
        return {"error": "Set LIVESCORE_API_KEY and LIVESCORE_API_SECRET in your env"}
    p = {"key": KEY, "secret": SECRET}
    p.update(params)
    r = requests.get(f"{BASE}/{path}.json", params=p, timeout=20)
    r.raise_for_status()
    return r.json()

def get_live_scores_lsa():
    """
    Try the standard live endpoints. Some plans expose matches/live, others scores/live.
    """
    last_err = None
    for ep in ("matches/live", "scores/live"):
        try:
            return _get(ep)
        except Exception as e:
            last_err = e
    # Return a soft error payload instead of raising
    return {"error": f"LiveScore API request failed: {last_err}"}
