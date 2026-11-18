"""
NBA feed via nba_api: player game logs, TeamPlayerOnOffDetails and live scoreboard lineups.

* Every request goes through `RateLimiter` (1.5 s minimum gap) and carries browser-style headers, which
  stats.nba.com expects. nba_api is synchronous, so calls run in a worker thread.
* Caveat: stats.nba.com (game logs, on/off) frequently refuses datacenter IPs, including GitHub runners; the
  README says the same. Each fetch is therefore best-effort with bounded retries, and a failure is logged
  and skipped rather than raised. The live endpoints (cdn.nba.com) are usually reachable but can 403 too.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable
from datetime import date, datetime, timezone

from app.db.base import session_scope, upsert
from app.db.models_ext import NbaApiGamelog, NbaLiveLineup, NbaOnOff
from app.ingestion.common import tracked
from app.utils.names import name_key

log = logging.getLogger(__name__)

RATE_LIMIT_SECONDS = 1.5
REQUEST_TIMEOUT = 30
MAX_ATTEMPTS = 3

NBA_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "Connection": "keep-alive",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
}
# the live (cdn.nba.com) endpoints reject a Host header pointing at stats.nba.com
LIVE_HEADERS = {k: v for k, v in NBA_HEADERS.items() if k != "Host"}


class RateLimiter:
    """Blocks so consecutive calls are at least `delay` seconds apart. Clock and sleep are injectable for tests."""

    def __init__(self, delay: float = RATE_LIMIT_SECONDS, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.delay, self._clock, self._sleep = delay, clock, sleep
        self._last: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            gap = self.delay - (now - self._last)
            if gap > 0:
                self._sleep(gap)
        self._last = self._clock()


_limiter = RateLimiter()


def call_endpoint(factory: Callable[..., object], *, limiter: RateLimiter | None = None, attempts: int = MAX_ATTEMPTS,
                  sleep: Callable[[float], None] = time.sleep, **kwargs):
    """Rate-limited, retried call of an nba_api endpoint class. Returns the endpoint object."""
    lim = limiter or _limiter
    last: Exception | None = None
    for i in range(1, attempts + 1):
        lim.wait()
        try:
            return factory(timeout=REQUEST_TIMEOUT, **kwargs)
        except Exception as e:  # nba_api raises requests / JSON errors; all are retryable here
            last = e
            log.warning("%s attempt %d/%d failed: %s", getattr(factory, "__name__", factory), i, attempts, e)
            if i < attempts:
                sleep(2.0 * i)
    raise RuntimeError(f"{getattr(factory, '__name__', factory)} failed after {attempts} attempts: {last}")


# ------------------------------------------------------------------ pure parsers
def current_season_str(today: date | None = None) -> str:
    t = today or date.today()
    y = t.year if t.month >= 10 else t.year - 1
    return f"{y}-{str(y + 1)[2:]}"


def _f(v):
    try:
        return None if v is None or v != v else float(v)
    except (TypeError, ValueError):
        return None


def _i(v):
    x = _f(v)
    return None if x is None else int(x)


def parse_gamelog_rows(records: list[dict], season: str) -> list[dict]:
    """PlayerGameLogs rows (upper-case NBA column names) -> nba_api_gamelog rows."""
    out = []
    for r in records:
        if r.get("PLAYER_ID") is None or not r.get("GAME_ID"):
            continue
        out.append(dict(
            nba_person_id=int(r["PLAYER_ID"]), nba_game_id=str(r["GAME_ID"]), name_key=name_key(r.get("PLAYER_NAME", "")),
            player_name=r.get("PLAYER_NAME", ""), team_abbr=r.get("TEAM_ABBREVIATION", ""), matchup=r.get("MATCHUP"),
            game_date=datetime.fromisoformat(str(r["GAME_DATE"])[:10]).date(), season=season, wl=r.get("WL"),
            minutes=_f(r.get("MIN")), pts=_i(r.get("PTS")), reb=_i(r.get("REB")), ast=_i(r.get("AST")),
            fg3m=_i(r.get("FG3M")), stl=_i(r.get("STL")), blk=_i(r.get("BLK")), tov=_i(r.get("TOV")),
            plus_minus=_f(r.get("PLUS_MINUS"))))
    return out


def parse_onoff_rows(on_records: list[dict], off_records: list[dict], season: str) -> list[dict]:
    """On/off court datasets -> nba_onoff rows. `COURT_STATUS` is 'On' / 'Off'; rating columns exist when the
    request used measure_type_detailed_defense='Advanced'."""
    out = []
    for recs in (on_records, off_records):
        for r in recs:
            if r.get("VS_PLAYER_ID") is None or r.get("TEAM_ID") is None:
                continue
            status = str(r.get("COURT_STATUS") or "").strip().title()
            if status not in ("On", "Off"):
                continue
            out.append(dict(
                season=season, team_id=int(r["TEAM_ID"]), nba_person_id=int(r["VS_PLAYER_ID"]), court_status=status,
                team_abbr=r.get("TEAM_ABBREVIATION"), player_name=r.get("VS_PLAYER_NAME"),
                name_key=name_key(r.get("VS_PLAYER_NAME", "")), gp=_i(r.get("GP")), minutes=_f(r.get("MIN")),
                plus_minus=_f(r.get("PLUS_MINUS")), off_rating=_f(r.get("OFF_RATING")),
                def_rating=_f(r.get("DEF_RATING")), net_rating=_f(r.get("NET_RATING")), pace=_f(r.get("PACE"))))
    return out


_ISO_DUR = re.compile(r"PT(?:(\d+)M)?(?:([\d.]+)S)?")


def parse_live_minutes(s: str | None) -> float | None:
    """'PT24M31.00S' -> 24.52"""
    m = _ISO_DUR.fullmatch(s or "")
    if not m:
        return None
    return round(int(m.group(1) or 0) + float(m.group(2) or 0) / 60.0, 2)


def parse_live_lineup(game: dict) -> list[dict]:
    """live BoxScore `game` dict -> nba_live_lineup rows (both teams)."""
    rows = []
    gid, gstatus = str(game.get("gameId")), game.get("gameStatus")
    for side in ("homeTeam", "awayTeam"):
        team = game.get(side) or {}
        for p in team.get("players", []):
            stats = p.get("statistics") or {}
            rows.append(dict(
                nba_game_id=gid, nba_person_id=int(p["personId"]), team_abbr=team.get("teamTricode"),
                player_name=p.get("name"), name_key=name_key(p.get("name", "")), position=p.get("position") or None,
                starter=str(p.get("starter", "0")) == "1", status=p.get("status"),
                played=str(p.get("played", "0")) == "1", minutes=parse_live_minutes(stats.get("minutes")),
                not_playing_reason=p.get("notPlayingReason"), game_status=gstatus))
    return rows


# ------------------------------------------------------------------ fetchers (network, blocking)
def fetch_gamelogs(season: str, date_from: date | None = None, limiter: RateLimiter | None = None) -> list[dict]:
    from nba_api.stats.endpoints import PlayerGameLogs

    kw = dict(season_nullable=season, season_type_nullable="Regular Season", headers=NBA_HEADERS)
    if date_from:
        kw["date_from_nullable"] = date_from.strftime("%m/%d/%Y")
    ep = call_endpoint(PlayerGameLogs, limiter=limiter, **kw)
    return parse_gamelog_rows(ep.get_data_frames()[0].to_dict("records"), season)


def fetch_team_onoff(team_id: int, season: str, limiter: RateLimiter | None = None) -> list[dict]:
    from nba_api.stats.endpoints import TeamPlayerOnOffDetails

    ep = call_endpoint(TeamPlayerOnOffDetails, limiter=limiter, team_id=team_id, season=season,
                       measure_type_detailed_defense="Advanced", headers=NBA_HEADERS)
    d = ep.get_normalized_dict()
    return parse_onoff_rows(d.get("PlayersOnCourtTeamPlayerOnOffDetails", []),
                            d.get("PlayersOffCourtTeamPlayerOnOffDetails", []), season)


def fetch_live_lineups(limiter: RateLimiter | None = None) -> list[dict]:
    """Today's scoreboard -> one live boxscore per game -> starters / active / inactive."""
    from nba_api.live.nba.endpoints import boxscore, scoreboard

    games = call_endpoint(scoreboard.ScoreBoard, limiter=limiter, headers=LIVE_HEADERS).get_dict()["scoreboard"]["games"]
    rows: list[dict] = []
    for g in games:
        try:
            box = call_endpoint(boxscore.BoxScore, limiter=limiter, attempts=2, game_id=g["gameId"],
                                headers=LIVE_HEADERS).get_dict()["game"]
            box.setdefault("gameStatus", g.get("gameStatus"))
            rows += parse_live_lineup(box)
        except Exception as e:     # boxscore isn't published until close to tip-off
            log.info("no live boxscore yet for %s: %s", g.get("gameId"), e)
    return rows


# ------------------------------------------------------------------ persistence
@tracked("nba_feed_gamelogs")
async def sync_gamelogs(season: str | None = None, date_from: date | None = None) -> int:
    rows = await asyncio.to_thread(fetch_gamelogs, season or current_season_str(), date_from)
    async with session_scope() as s:
        return await upsert(s, NbaApiGamelog, rows, ["nba_person_id", "nba_game_id"])


@tracked("nba_feed_onoff")
async def sync_onoff(season: str | None = None) -> int:
    from nba_api.stats.static import teams

    season = season or current_season_str()
    rows: list[dict] = []
    for t in teams.get_teams():
        try:
            rows += await asyncio.to_thread(fetch_team_onoff, t["id"], season)
        except Exception as e:
            log.warning("on/off failed for %s: %s", t["abbreviation"], e)
    async with session_scope() as s:
        return await upsert(s, NbaOnOff, rows, ["season", "team_id", "nba_person_id", "court_status"])


@tracked("nba_feed_lineups")
async def sync_live_lineups() -> int:
    rows = await asyncio.to_thread(fetch_live_lineups)
    for r in rows:
        r["updated_at"] = datetime.now(timezone.utc)
    async with session_scope() as s:
        return await upsert(s, NbaLiveLineup, rows, ["nba_game_id", "nba_person_id"])


async def sync_all(season: str | None = None, date_from: date | None = None) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, fn, kw in (("gamelogs", sync_gamelogs, dict(season=season, date_from=date_from)),
                         ("onoff", sync_onoff, dict(season=season)), ("lineups", sync_live_lineups, {})):
        try:
            out[name] = await fn(**kw)
        except Exception as e:
            log.warning("nba_feed %s failed: %s", name, e)
            out[name] = -1
    return out
