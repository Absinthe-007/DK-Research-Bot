"""
Injury reports from ESPN's public injuries JSON (keyed by ESPN athlete id, so no name
matching), with a BeautifulSoup fallback on the ESPN injuries HTML page.

Statuses normalized to: Out | Doubtful | Questionable | Probable | Day-To-Day | IR | PUP | Suspended
"""
from __future__ import annotations

import logging
import re

from bs4 import BeautifulSoup

from app.db.base import session_scope
from app.db.models import InjuryReport
from app.db.reference import ESPN_PATH
from app.ingestion.common import player_ids_by_espn, player_index, resolve_player, tracked, ts
from app.ingestion.http import http

log = logging.getLogger(__name__)
HTML = {"NFL": "https://www.espn.com/nfl/injuries", "NBA": "https://www.espn.com/nba/injuries"}
_ID = re.compile(r"/id/(\d+)")
_STATUS = {
    "out": "Out", "doubtful": "Doubtful", "questionable": "Questionable", "probable": "Probable",
    "day-to-day": "Day-To-Day", "injured reserve": "IR", "ir": "IR", "physically unable to perform": "PUP",
    "pup": "PUP", "suspension": "Suspended", "suspended": "Suspended",
}


def normalize_status(s: str | None) -> str:
    return _STATUS.get((s or "").strip().lower(), (s or "Unknown").strip().title())


async def _from_api(sport: str) -> list[dict]:
    body, _ = await http().get(f"https://site.api.espn.com/apis/site/v2/sports/{ESPN_PATH[sport]}/injuries", cache_ttl=300)
    out = []
    for team in body.get("injuries", []):
        for inj in team.get("injuries", []):
            ath = inj.get("athlete") or {}
            links = " ".join(l.get("href", "") for l in ath.get("links", []))
            m = _ID.search(links)
            det = inj.get("details") or {}
            out.append(dict(
                espn_id=m.group(1) if m else None, name=ath.get("displayName"),
                team=(ath.get("team") or {}).get("abbreviation"),
                status=normalize_status(inj.get("status")),
                injury=" ".join(x for x in (det.get("side"), det.get("type"), det.get("detail")) if x and x != "Not Specified") or None,
                comment=inj.get("shortComment"), return_date=det.get("returnDate"), reported_at=ts(inj.get("date")),
            ))
    return [o for o in out if o["name"]]


async def _from_html(sport: str) -> list[dict]:
    html, _ = await http().get(HTML[sport], as_text=True, cache_ttl=300)
    soup = BeautifulSoup(html, "lxml")
    out = []
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True).upper() for th in table.find_all("th")]
        if "NAME" not in headers or "STATUS" not in headers:
            continue
        ix = {h: i for i, h in enumerate(headers)}
        for tr in (table.find("tbody") or table).find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < len(headers):
                continue
            a = tds[ix["NAME"]].find("a", href=True)
            m = _ID.search(a["href"]) if a else None
            out.append(dict(espn_id=m.group(1) if m else None, name=tds[ix["NAME"]].get_text(" ", strip=True), team=None,
                            status=normalize_status(tds[ix["STATUS"]].get_text(strip=True)), injury=None,
                            comment=tds[ix["COMMENT"]].get_text(" ", strip=True) if "COMMENT" in ix else None,
                            return_date=None, reported_at=None))
    return out


@tracked("injuries")
async def sync_injuries(sport: str) -> int:
    source = "espn_api"
    try:
        items = await _from_api(sport)
    except Exception as e:
        log.warning("ESPN injury API failed (%s); falling back to HTML", e)
        items, source = await _from_html(sport), "espn_html"
    async with session_scope() as s:
        by_id = await player_ids_by_espn(s, sport)
        by_name = await player_index(s, sport)
        s.add_all([InjuryReport(sport=sport, player_id=by_id.get(i["espn_id"]) or resolve_player(i["name"], by_name),
                                player_name_raw=i["name"], team_abbr=i["team"], status=i["status"], injury=i["injury"],
                                comment=i["comment"], return_date=i["return_date"], reported_at=i["reported_at"],
                                source=source) for i in items])
    return len(items)
