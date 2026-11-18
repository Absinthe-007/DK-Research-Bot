"""
Free NBA lineups, inactives and game logs from ESPN's public summary endpoint.

Why: stats.nba.com and cdn.nba.com refuse GitHub-hosted runners, so `nba_feed` cannot fill `nba_live_lineup` or
`nba_api_gamelog` there. ESPN is already used (and reachable) by the rest of the bot, needs no key and covers every
game on the board.

Per game (`/basketball/nba/summary?event=<id>`):
* pre-game   - `injuries` per team with a status. "Out" -> status INACTIVE (read by the adjustment layer); Day-To-Day,
               Questionable etc. are left out so the stored injury report keeps deciding those.
* in / final - `boxscore.players`: starters, minutes and `didNotPlay` + reason for every dressed player.
* final      - also written to `nba_api_gamelog` (minutes, pts, reb, ast, 3pm, stl, blk, tov, +/-), which the grader
               and the adjustment layer's participation checks read.

IDs are ESPN's (athlete id in `nba_person_id`, event id in `nba_game_id`). Everything downstream joins on `name_key`.
Final rows get `updated_at = tip-off + 3h` so they sit inside the grader's evidence window whenever the job runs.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import text

from app.db.base import session_scope, upsert
from app.db.models_ext import NbaApiGamelog, NbaLiveLineup
from app.ingestion.common import tracked
from app.ingestion.http import http
from app.utils.names import name_key

log = logging.getLogger(__name__)

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
GAME_STATUS = {"pre": 1, "in": 2, "post": 3}
FINAL_ROW_OFFSET = timedelta(hours=3)


# ------------------------------------------------------------------ pure parsing
def _num(v) -> float | None:
    try:
        s = str(v).strip().replace("+", "")
        return None if s in ("", "-", "--") else float(s)
    except (TypeError, ValueError):
        return None


def _made(v) -> int | None:
    """'2-5' -> 2"""
    try:
        return int(str(v).split("-")[0])
    except (TypeError, ValueError):
        return None


def _minutes(v) -> float | None:
    s = str(v).strip() if v is not None else ""
    if ":" in s:                                   # '36:12'
        try:
            m, sec = s.split(":")[:2]
            return int(m) + int(sec) / 60
        except ValueError:
            return None
    return _num(s)


def parse_start(s) -> datetime | None:
    if not s:
        return None
    t = str(s).strip()
    if t.endswith("Z"):
        t = t[:-1]
        if len(t) == 16:                           # ESPN sends '2026-11-18T00:30Z' (no seconds)
            t += ":00"
        t += "+00:00"
    try:
        d = datetime.fromisoformat(t)
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def season_for(d: date) -> str:
    y = d.year if d.month >= 10 else d.year - 1
    return f"{y}-{str(y + 1)[2:]}"


def game_meta(body: dict) -> dict | None:
    """-> {gid, state, start, home, away, winner}"""
    try:
        comp = body["header"]["competitions"][0]
        gid = str(body["header"]["id"])
    except (KeyError, IndexError, TypeError):
        return None
    sides = {c.get("homeAway"): c for c in comp.get("competitors", [])}
    abbr = lambda side: (sides.get(side) or {}).get("team", {}).get("abbreviation")
    winner = next((c["team"]["abbreviation"] for c in comp.get("competitors", []) if c.get("winner")), None)
    return dict(gid=gid, state=(comp.get("status", {}).get("type", {}) or {}).get("state"),
                start=parse_start(comp.get("date")), home=abbr("home"), away=abbr("away"), winner=winner)


def parse_injury_rows(body: dict, meta: dict) -> list[dict]:
    rows = []
    gs = GAME_STATUS.get(meta["state"])
    for team in body.get("injuries") or []:
        abbr = (team.get("team") or {}).get("abbreviation")
        for inj in team.get("injuries") or []:
            a = inj.get("athlete") or {}
            if str(inj.get("status", "")).strip().lower() not in ("out", "out for season"):
                continue
            try:
                pid = int(a["id"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(dict(nba_game_id=meta["gid"], nba_person_id=pid, team_abbr=abbr, player_name=a.get("displayName"),
                             name_key=name_key(a.get("displayName", "")), position=((a.get("position") or {}).get("abbreviation")),
                             starter=False, status="INACTIVE", played=False, minutes=None,
                             not_playing_reason=((inj.get("details") or {}).get("type") or inj.get("status")),
                             game_status=gs))
    return rows


def _col(labels: list[str]) -> dict[str, int]:
    return {l: i for i, l in enumerate(labels)}


def parse_box(body: dict, meta: dict) -> tuple[list[dict], list[dict]]:
    """Box score -> (lineup rows, gamelog rows). Gamelog rows only once the game is final."""
    lineup, logs = [], []
    gs = GAME_STATUS.get(meta["state"])
    final = meta["state"] == "post"
    start = meta["start"] or datetime.now(timezone.utc)
    gdate = (start - timedelta(hours=6)).date()
    for team in (body.get("boxscore") or {}).get("players") or []:
        abbr = (team.get("team") or {}).get("abbreviation")
        for grp in team.get("statistics") or []:
            col = _col(grp.get("labels") or [])
            for p in grp.get("athletes") or []:
                a = p.get("athlete") or {}
                try:
                    pid = int(a["id"])
                except (KeyError, TypeError, ValueError):
                    continue
                st = p.get("stats") or []
                get = lambda k: st[col[k]] if k in col and col[k] < len(st) else None
                mins = _minutes(get("MIN"))
                dnp = bool(p.get("didNotPlay")) or not st
                played = (mins or 0) > 0 and not dnp
                nm = a.get("displayName")
                lineup.append(dict(
                    nba_game_id=meta["gid"], nba_person_id=pid, team_abbr=abbr, player_name=nm, name_key=name_key(nm or ""),
                    position=(a.get("position") or {}).get("abbreviation"), starter=bool(p.get("starter")),
                    status="ACTIVE", played=played, minutes=(mins if played else 0.0 if final else mins),
                    not_playing_reason=(p.get("reason") if dnp else None), game_status=gs))
                if final:
                    home = abbr == meta["home"]
                    other = meta["away"] if home else meta["home"]
                    logs.append(dict(
                        nba_person_id=pid, nba_game_id=meta["gid"], name_key=name_key(nm or ""), player_name=nm or "",
                        team_abbr=abbr or "", matchup=(f"{abbr} vs. {other}" if home else f"{abbr} @ {other}"),
                        game_date=gdate, season=season_for(gdate),
                        wl=(None if meta["winner"] is None else ("W" if meta["winner"] == abbr else "L")),
                        minutes=(mins if played else 0.0), pts=_i(get("PTS")), reb=_i(get("REB")), ast=_i(get("AST")),
                        fg3m=_made(get("3PT")), stl=_i(get("STL")), blk=_i(get("BLK")), tov=_i(get("TO")),
                        plus_minus=_num(get("+/-"))))
    return lineup, logs


def _i(v) -> int | None:
    n = _num(v)
    return None if n is None else int(n)


def parse_summary(body: dict) -> dict | None:
    """-> {meta, lineup, gamelog}; box-score rows win over injury rows for the same player."""
    meta = game_meta(body)
    if not meta or meta["state"] not in GAME_STATUS:
        return None
    inj = parse_injury_rows(body, meta)
    box, logs = parse_box(body, meta)
    merged = {r["nba_person_id"]: r for r in inj}
    merged.update({r["nba_person_id"]: r for r in box})
    return dict(meta=meta, lineup=list(merged.values()), gamelog=logs)


# ------------------------------------------------------------------ network
async def scoreboard_events(day: date) -> list[dict]:
    body, _ = await http().get(f"{BASE}/scoreboard", params=dict(dates=day.strftime("%Y%m%d"), limit=1000), cache_ttl=300)
    out = []
    for e in body.get("events", []):
        st = ((e.get("status") or {}).get("type") or {}).get("state")
        out.append(dict(id=str(e["id"]), state=st, date=e.get("date")))
    return out


async def fetch_summary(event_id: str) -> dict | None:
    body, _ = await http().get(f"{BASE}/summary", params=dict(event=event_id), cache_ttl=120)
    return parse_summary(body)


async def _summaries(days: list[date], states: set[str], skip_ids: set[str] = frozenset()) -> list[dict]:
    ids: list[str] = []
    for d in days:
        try:
            ids += [e["id"] for e in await scoreboard_events(d) if e["state"] in states and e["id"] not in skip_ids]
        except Exception as e:  # one bad day must not sink the rest
            log.warning("espn scoreboard %s failed: %s", d, e)
    out: list[dict] = []
    sem = asyncio.Semaphore(4)

    async def one(i: str):
        async with sem:
            try:
                r = await fetch_summary(i)
                if r:
                    out.append(r)
            except Exception as e:
                log.warning("espn summary %s failed: %s", i, e)

    await asyncio.gather(*(one(i) for i in dict.fromkeys(ids)))
    return out


def _today() -> date:
    return datetime.now(timezone.utc).date()


# ------------------------------------------------------------------ jobs
@tracked("espn_nba_lineups")
async def sync_lineups(days_back: int = 1, days_ahead: int = 1) -> int:
    """Pre-game Out list for upcoming games, live/final box score state for today's and yesterday's games."""
    today = _today()
    days = [today + timedelta(days=i) for i in range(-days_back, days_ahead + 1)]
    results = await _summaries(days, {"pre", "in", "post"})
    now = datetime.now(timezone.utc)
    rows: list[dict] = []
    finals: list[tuple[str, datetime]] = []
    for r in results:
        meta = r["meta"]
        stamp = now
        if meta["state"] == "post":
            stamp = (meta["start"] or now) + FINAL_ROW_OFFSET
            finals.append((meta["gid"], stamp))
        rows += [dict(x, updated_at=stamp) for x in r["lineup"]]
    async with session_scope() as s:
        n = await upsert(s, NbaLiveLineup, rows, ["nba_game_id", "nba_person_id"])
        for gid, stamp in finals:           # pre-game Out rows of games that have since finished become final evidence
            await s.execute(text("""UPDATE nba_live_lineup SET game_status = 3, updated_at = :t
                                    WHERE nba_game_id = :g AND status = 'INACTIVE' AND game_status <> 3"""), {"g": gid, "t": stamp})
    log.info("espn lineups: %d rows from %d games", n, len(results))
    return n


@tracked("espn_nba_gamelogs")
async def sync_gamelogs(days_back: int = 10) -> int:
    """Final box scores of the last `days_back` days -> nba_api_gamelog (games already stored are skipped)."""
    today = _today()
    async with session_scope() as s:
        have = {r[0] for r in (await s.execute(text(
            "SELECT DISTINCT nba_game_id FROM nba_api_gamelog WHERE game_date >= :d"), {"d": today - timedelta(days=days_back + 2)})).all()}
    results = await _summaries([today - timedelta(days=i) for i in range(0, days_back + 1)], {"post"}, skip_ids=have)
    rows = [x for r in results for x in r["gamelog"]]
    async with session_scope() as s:
        n = await upsert(s, NbaApiGamelog, rows, ["nba_person_id", "nba_game_id"])
    log.info("espn gamelogs: %d rows from %d new games", n, len(results))
    return n
