"""
ESPN scrapers - teams, rosters, schedule + DraftKings game lines, DraftKings player props,
and box scores. All free, keyless, and reachable from GitHub Actions runners
(unlike stats.nba.com, which blocks cloud IPs).

Endpoints
  site  : https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/...
            teams, teams/{id}/roster, scoreboard?dates=YYYYMMDD-YYYYMMDD, summary?event={id}
  core  : https://sports.core.api.espn.com/v2/sports/{sport}/leagues/{league}/events/{id}/
            competitions/{id}/odds/100/propBets        (provider 100 = DraftKings)
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select, text

from app.db.base import session_scope, upsert
from app.db.models import Game, NbaPlayerGamelog, NflPlayerGame, OddsSnapshot, Player, PropSnapshot, Team, Venue
from app.db.reference import ESPN_PATH, PROP_TYPES
from app.ingestion.common import parse_american, parse_float, player_ids_by_espn, tracked, ts
from app.ingestion.http import http
from app.utils.names import name_key

log = logging.getLogger(__name__)
SITE = "https://site.api.espn.com/apis/site/v2/sports"
CORE = "https://sports.core.api.espn.com/v2/sports"
DK = "100"
_ATH = re.compile(r"/athletes/(\d+)")
ASSUMED_PRICE = -115


def _core_league(sport: str) -> str:
    sp, lg = ESPN_PATH[sport].split("/")
    return f"{CORE}/{sp}/leagues/{lg}"


def season_label(sport: str, year: int) -> str:
    """ESPN NBA season 'year' is the END year (2027 -> '2026-27'); NFL is the start year."""
    return str(year) if sport == "NFL" else f"{year - 1}-{str(year)[2:]}"


# ---------------------------------------------------------------------------
# Teams + rosters
# ---------------------------------------------------------------------------
@tracked("espn_teams")
async def sync_teams(sport: str) -> int:
    body, _ = await http().get(f"{SITE}/{ESPN_PATH[sport]}/teams", cache_ttl=86400)
    teams = body["sports"][0]["leagues"][0]["teams"]
    rows = [dict(sport=sport, espn_id=t["team"]["id"], abbr=t["team"]["abbreviation"], name=t["team"]["displayName"])
            for t in teams]
    async with session_scope() as s:
        return await upsert(s, Team, rows, ["sport", "espn_id"], ["abbr", "name"])


def _roster_athletes(body: dict) -> list[dict]:
    out = []
    for a in body.get("athletes", []):
        out.extend(a["items"] if isinstance(a, dict) and "items" in a else [a])   # NFL groups by unit
    return out


@tracked("espn_rosters")
async def sync_rosters(sport: str) -> int:
    async with session_scope() as s:
        teams = (await s.execute(select(Team).where(Team.sport == sport))).scalars().all()
    rows = []
    for t in teams:
        try:
            body, _ = await http().get(f"{SITE}/{ESPN_PATH[sport]}/teams/{t.espn_id}/roster", cache_ttl=3600)
        except Exception as e:
            log.warning("roster %s failed: %s", t.abbr, e)
            continue
        for a in _roster_athletes(body):
            rows.append(dict(sport=sport, espn_id=str(a["id"]), full_name=a["displayName"],
                             name_key=name_key(a["displayName"]), team_abbr=t.abbr,
                             position=(a.get("position") or {}).get("abbreviation"),
                             headshot=(a.get("headshot") or {}).get("href")))
    async with session_scope() as s:
        return await upsert(s, Player, rows, ["sport", "espn_id"], ["full_name", "name_key", "team_abbr", "position", "headshot"])


async def _ensure_athlete(s, sport: str, espn_id: str, cache: dict[str, int], ref_url: str | None = None) -> int | None:
    """Look up a player by ESPN id; fetch from the core API if we've never seen him."""
    if espn_id in cache:
        return cache[espn_id]
    url = ref_url or f"{_core_league(sport)}/athletes/{espn_id}"
    try:
        a, _ = await http().get(url.replace("http://", "https://"), cache_ttl=86400)
    except Exception as e:
        log.warning("athlete %s lookup failed: %s", espn_id, e)
        return None
    team_abbr = None
    tref = (a.get("team") or {}).get("$ref")
    if tref:
        tid = re.search(r"/teams/(\d+)", tref)
        if tid:
            team_abbr = (await s.execute(text("SELECT abbr FROM teams WHERE sport=:s AND espn_id=:t"),
                                         {"s": sport, "t": tid.group(1)})).scalar()
    await upsert(s, Player, [dict(sport=sport, espn_id=espn_id, full_name=a.get("displayName", espn_id),
                                  name_key=name_key(a.get("displayName", "")), team_abbr=team_abbr,
                                  position=(a.get("position") or {}).get("abbreviation"),
                                  headshot=(a.get("headshot") or {}).get("href"))],
                 ["sport", "espn_id"], ["full_name", "name_key", "position"])
    pid = (await s.execute(select(Player.id).where(Player.sport == sport, Player.espn_id == espn_id))).scalar()
    cache[espn_id] = pid
    return pid


# ---------------------------------------------------------------------------
# Schedule + DraftKings game lines
# ---------------------------------------------------------------------------
def _dk_odds(comp: dict) -> dict | None:
    for o in comp.get("odds") or []:
        if str((o.get("provider") or {}).get("id")) == DK:
            return o
    return None


def _line_rows(game_id: int, o: dict) -> list[dict]:
    rows = []
    ml, sp, tot = o.get("moneyline") or {}, o.get("pointSpread") or {}, o.get("total") or {}
    for side in ("home", "away"):
        m = ml.get(side) or {}
        rows.append(dict(game_id=game_id, market_type="h2h", side=side, point=None,
                         price=parse_american((m.get("close") or {}).get("odds")),
                         open_price=parse_american((m.get("open") or {}).get("odds")), open_point=None))
        p = sp.get(side) or {}
        rows.append(dict(game_id=game_id, market_type="spreads", side=side,
                         price=parse_american((p.get("close") or {}).get("odds")),
                         point=parse_float((p.get("close") or {}).get("line")),
                         open_price=parse_american((p.get("open") or {}).get("odds")),
                         open_point=parse_float((p.get("open") or {}).get("line"))))
    for side in ("over", "under"):
        t = tot.get(side) or {}
        rows.append(dict(game_id=game_id, market_type="totals", side=side,
                         price=parse_american((t.get("close") or {}).get("odds")),
                         point=parse_float((t.get("close") or {}).get("line")) or o.get("overUnder"),
                         open_price=parse_american((t.get("open") or {}).get("odds")),
                         open_point=parse_float((t.get("open") or {}).get("line"))))
    return [r for r in rows if r["price"] is not None or r["point"] is not None]


@tracked("espn_schedule")
async def sync_schedule(sport: str, start: date, end: date) -> int:
    """Upsert games/venues and snapshot DK game lines for every event in [start, end].
    (ESPN's scoreboard rejects date ranges for these leagues, so we request one day at a time.)"""
    events, d = [], start
    while d <= end:
        body, _ = await http().get(f"{SITE}/{ESPN_PATH[sport]}/scoreboard",
                                   params={"dates": f"{d:%Y%m%d}", "limit": 1000}, cache_ttl=60)
        events += body.get("events", [])
        d += timedelta(days=1)
    events = list({e["id"]: e for e in events}.values())
    n = 0
    async with session_scope() as s:
        teams = {r.espn_id: r.id for r in (await s.execute(select(Team).where(Team.sport == sport))).scalars()}
        for e in events:
            comp = e["competitions"][0]
            home = next(c for c in comp["competitors"] if c["homeAway"] == "home")
            away = next(c for c in comp["competitors"] if c["homeAway"] == "away")
            if home["team"]["id"] not in teams or away["team"]["id"] not in teams:
                continue   # all-star / exhibition opponents
            v = comp.get("venue") or {}
            venue_id = None
            if v.get("id"):
                await upsert(s, Venue, [dict(espn_id=str(v["id"]), name=v.get("fullName", ""),
                                             city=(v.get("address") or {}).get("city"),
                                             state=(v.get("address") or {}).get("state"),
                                             country=(v.get("address") or {}).get("country"),
                                             indoor=bool(v.get("indoor")))],
                             ["espn_id"], ["name", "indoor"])
                venue_id = (await s.execute(select(Venue.id).where(Venue.espn_id == str(v["id"])))).scalar()
            st = comp["status"]["type"]
            row = dict(sport=sport, espn_event_id=e["id"], season=season_label(sport, e["season"]["year"]),
                       season_type=e["season"].get("type", 2), week=(e.get("week") or {}).get("number"),
                       start_time=ts(e["date"]), home_team_id=teams[home["team"]["id"]],
                       away_team_id=teams[away["team"]["id"]], venue_id=venue_id,
                       neutral_site=bool(comp.get("neutralSite")), status=st["name"], completed=bool(st["completed"]),
                       home_score=int(home["score"]) if st["completed"] and home.get("score") else None,
                       away_score=int(away["score"]) if st["completed"] and away.get("score") else None)
            await upsert(s, Game, [row], ["espn_event_id"],
                         [k for k in row if k not in ("espn_event_id", "sport")])
            n += 1
            o = _dk_odds(comp)
            if o and not st["completed"] and st["name"] == "STATUS_SCHEDULED":
                gid = (await s.execute(select(Game.id).where(Game.espn_event_id == e["id"]))).scalar()
                await _insert_changed_lines(s, gid, _line_rows(gid, o))
    return n


async def _insert_changed_lines(s, game_id: int, rows: list[dict]) -> None:
    latest = {(r.market_type, r.side): (r.price, r.point) for r in (await s.execute(text("""
        SELECT DISTINCT ON (market_type, side) market_type, side, price, point FROM odds_snapshots
        WHERE game_id=:g ORDER BY market_type, side, captured_at DESC"""), {"g": game_id})).all()}
    new = [OddsSnapshot(**r) for r in rows if latest.get((r["market_type"], r["side"])) != (r["price"], r["point"])]
    s.add_all(new)


# ---------------------------------------------------------------------------
# DraftKings player props
# ---------------------------------------------------------------------------
def _implied(a: int) -> float:
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def parse_prop_items(items: list[dict]) -> list[dict]:
    """ESPN propBets items -> normalized rows (without player resolution).

    * Two-way 'Total X' markets arrive as consecutive (Over, Under) pairs per athlete+type+line.
    * 'X Milestones' ladders: ESPN files several DK sub-markets (full game, 1st half, ...) under
      the SAME type name, one ladder after another. We split on threshold resets and keep only
      the ladder that agrees with the full-game two-way line (its N+ price at ceil(line) must be
      within 5 pts of the Over price). Ladders that can't be verified are dropped.
    """
    out: list[dict] = []
    groups: dict[tuple, list[dict]] = defaultdict(list)
    ladders: dict[tuple, list[dict]] = defaultdict(list)
    for it in items:
        tname = (it.get("type") or {}).get("name")
        if tname not in PROP_TYPES:
            continue
        m = _ATH.search((it.get("athlete") or {}).get("$ref", ""))
        if not m:
            continue
        odds = it.get("odds") or {}
        price = parse_american((odds.get("american") or {}).get("value"))
        mkey, kind = PROP_TYPES[tname]
        est = False
        if price is None:
            # ESPN sometimes republishes DK prop LINES without prices (seen on NFL). For two-way
            # markets assume DK's standard -115/-115 hold and label it; one-sided ladders need a price.
            if kind != "ou":
                continue
            price, est = ASSUMED_PRICE, True
        base = dict(espn_athlete=m.group(1), athlete_ref=it["athlete"]["$ref"], market_key=mkey, price=price,
                    price_estimated=est, open_price=parse_american((odds.get("american") or {}).get("open")),
                    book_updated_at=ts(it.get("lastUpdated")))
        total = (odds.get("total") or {})
        if kind == "ou":
            line = parse_float(total.get("value"))
            if line is None:
                line = parse_float(((it.get("current") or {}).get("target") or {}).get("value"))
            groups[(m.group(1), mkey, line)].append(dict(base, kind="ou", line=line,
                                                         open_line=parse_float(total.get("open"))))
        elif kind == "alt":
            thr = parse_float(str(total.get("value", "")).rstrip("+"))
            if thr is not None:
                ladders[(m.group(1), mkey)].append(dict(base, kind="alt", side="Over", line=thr - 0.5,
                                                        open_line=None, thr=thr))
        else:  # "yes0.5" / "yes1.5": anytime TD markets are single, unambiguous
            out.append(dict(base, kind="alt", side="Over", line=float(kind[3:]), open_line=None))

    main_over: dict[tuple, tuple[float, int]] = {}
    for (ath, mkey, line), rows in groups.items():
        if line is None:
            continue
        for i in range(0, len(rows) - 1, 2):     # Over first, then Under (validated vs milestone ladder)
            out.append(dict(rows[i], side="Over"))
            out.append(dict(rows[i + 1], side="Under"))
            if not rows[i]["price_estimated"]:
                main_over[(ath, mkey)] = (line, rows[i]["price"])

    for key, rungs in ladders.items():
        segs, cur = [], []
        for r in rungs:                           # split where the threshold resets
            if cur and r["thr"] <= cur[-1]["thr"]:
                segs.append(cur)
                cur = []
            cur.append(r)
        if cur:
            segs.append(cur)
        ref = main_over.get(key)
        if not ref:
            continue                              # unverifiable ladder -> drop
        need = ref[0] + 0.5                       # e.g. Over 11.5 <-> "12+"
        best, gap = None, 1.0
        for seg in segs:
            hit = next((r for r in seg if r["thr"] == need), None)
            if hit:
                g = abs(_implied(hit["price"]) - _implied(ref[1]))
                if g < gap:
                    best, gap = seg, g
        if best and gap <= 0.05:
            out.extend({k: v for k, v in r.items() if k != "thr"} for r in best)
    return out


async def fetch_event_props(sport: str, event_id: str) -> list[dict]:
    url = f"{_core_league(sport)}/events/{event_id}/competitions/{event_id}/odds/{DK}/propBets"
    items, page = [], 1
    while True:
        body, _ = await http().get(url, params={"lang": "en", "region": "us", "limit": 1000, "page": page}, cache_ttl=60)
        items += body.get("items", [])
        if page >= int(body.get("pageCount") or 1):
            break
        page += 1
    return parse_prop_items(items)


@tracked("espn_props")
async def sync_props(sport: str, hours_ahead: int = 30) -> int:
    now = datetime.now(timezone.utc)
    async with session_scope() as s:
        games = (await s.execute(select(Game).where(
            Game.sport == sport, Game.completed.is_(False), Game.start_time > now - timedelta(minutes=5),
            Game.start_time < now + timedelta(hours=hours_ahead)))).scalars().all()
    total = 0
    for g in games:
        try:
            rows = await fetch_event_props(sport, g.espn_event_id)
        except Exception as e:
            log.warning("props failed for %s: %s", g.espn_event_id, e)
            continue
        async with session_scope() as s:
            cache = await player_ids_by_espn(s, sport)
            latest = {(r.player_id, r.market_key, r.kind, r.side, r.line): r.price for r in (await s.execute(text("""
                SELECT DISTINCT ON (player_id, market_key, kind, side, line) player_id, market_key, kind, side, line, price
                FROM prop_snapshots WHERE game_id=:g ORDER BY player_id, market_key, kind, side, line, captured_at DESC"""),
                {"g": g.id})).all()}
            new = []
            for r in rows:
                pid = await _ensure_athlete(s, sport, r["espn_athlete"], cache, r["athlete_ref"])
                if pid is None:
                    continue
                key = (pid, r["market_key"], r["kind"], r["side"], r["line"])
                if latest.get(key) == r["price"]:
                    continue        # unchanged -> no new snapshot (keeps the DB small)
                latest[key] = r["price"]
                new.append(PropSnapshot(game_id=g.id, player_id=pid, market_key=r["market_key"], kind=r["kind"],
                                        side=r["side"], line=r["line"], price=r["price"], price_estimated=r["price_estimated"],
                                        open_price=r["open_price"],
                                        open_line=r["open_line"], book_updated_at=r["book_updated_at"]))
            s.add_all(new)
            total += len(new)
    return total


# ---------------------------------------------------------------------------
# Box scores
# ---------------------------------------------------------------------------
def _num(v) -> float:
    try:
        return float(str(v).replace("+", ""))
    except (TypeError, ValueError):
        return 0.0


def _split(v, i) -> float:
    parts = re.split(r"[-/]", str(v))
    return _num(parts[i]) if len(parts) > i else 0.0


def parse_nba_box(summary: dict) -> list[dict]:
    rows = []
    for team in summary["boxscore"]["players"]:
        abbr = team["team"]["abbreviation"]
        for st in team["statistics"]:
            keys = st.get("keys") or []
            for a in st.get("athletes", []):
                if a.get("didNotPlay") or not a.get("stats"):
                    continue
                v = dict(zip(keys, a["stats"]))
                mins = _num(v.get("minutes"))
                if mins <= 0:
                    continue
                rows.append(dict(espn_id=a["athlete"]["id"], name=a["athlete"]["displayName"],
                                 position=(a["athlete"].get("position") or {}).get("abbreviation"),
                                 team_abbr=abbr, starter=bool(a.get("starter")), minutes=mins,
                                 pts=int(_num(v.get("points"))), reb=int(_num(v.get("rebounds"))),
                                 oreb=int(_num(v.get("offensiveRebounds"))), ast=int(_num(v.get("assists"))),
                                 fg3m=int(_split(v.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted"), 0)),
                                 stl=int(_num(v.get("steals"))), blk=int(_num(v.get("blocks"))),
                                 tov=int(_num(v.get("turnovers"))),
                                 fga=int(_split(v.get("fieldGoalsMade-fieldGoalsAttempted"), 1)),
                                 fta=int(_split(v.get("freeThrowsMade-freeThrowsAttempted"), 1)),
                                 plus_minus=_num(v.get("plusMinus"))))
    return rows


NFL_FIELDS = {
    "passing": {"completions/passingAttempts": ("completions", "pass_attempts"), "passingYards": "pass_yds",
                "passingTouchdowns": "pass_tds", "interceptions": "interceptions", "sacks-sackYardsLost": ("sacks_taken", None)},
    "rushing": {"rushingAttempts": "carries", "rushingYards": "rush_yds", "rushingTouchdowns": "rush_tds"},
    "receiving": {"receptions": "receptions", "receivingYards": "rec_yds", "receivingTouchdowns": "rec_tds",
                  "receivingTargets": "targets"},
    "defensive": {"totalTackles": "tackles", "sacks": "sacks", "tacklesForLoss": "tfl", "passesDefended": "passes_defended",
                  "QBHits": "qb_hits"},
    "interceptions": {"interceptions": "def_int"},
    "kicking": {"fieldGoalsMade/fieldGoalAttempts": ("fgm", "fga"), "extraPointsMade/extraPointAttempts": ("xpm", None)},
}
NFL_STAT_COLS = ("completions", "pass_attempts", "pass_yds", "pass_tds", "interceptions", "sacks_taken", "carries",
                 "rush_yds", "rush_tds", "targets", "receptions", "rec_yds", "rec_tds", "tackles", "sacks", "tfl",
                 "qb_hits", "passes_defended", "def_int", "fgm", "fga", "xpm")


def parse_nfl_box(summary: dict) -> list[dict]:
    players: dict[str, dict] = {}
    for team in summary["boxscore"]["players"]:
        abbr = team["team"]["abbreviation"]
        for st in team["statistics"]:
            fields = NFL_FIELDS.get(st.get("name"))
            if not fields:
                continue
            keys = st.get("keys") or []
            for a in st.get("athletes", []):
                aid = a["athlete"]["id"]
                p = players.setdefault(aid, dict(espn_id=aid, name=a["athlete"]["displayName"], team_abbr=abbr,
                                                 position=(a["athlete"].get("position") or {}).get("abbreviation")))
                for k, val in zip(keys, a.get("stats", [])):
                    tgt = fields.get(k)
                    if tgt is None:
                        continue
                    if isinstance(tgt, tuple):
                        if tgt[0]:
                            p[tgt[0]] = _split(val, 0)
                        if tgt[1]:
                            p[tgt[1]] = _split(val, 1)
                    else:
                        p[tgt] = _num(val)
    return list(players.values())


@tracked("espn_boxscores")
async def sync_boxscores(sport: str, limit: int = 400) -> int:
    async with session_scope() as s:
        games = (await s.execute(text("""
            SELECT g.id, g.espn_event_id, g.season, g.week, g.start_time, h.abbr home, a.abbr away
            FROM games g JOIN teams h ON h.id=g.home_team_id JOIN teams a ON a.id=g.away_team_id
            WHERE g.sport=:sp AND g.completed AND NOT g.boxscore_loaded AND g.season_type IN (2,3)
            ORDER BY g.start_time LIMIT :lim"""), {"sp": sport, "lim": limit})).all()
    n = 0
    for g in games:
        try:
            summ, _ = await http().get(f"{SITE}/{ESPN_PATH[sport]}/summary", params={"event": g.espn_event_id})
            rows = parse_nba_box(summ) if sport == "NBA" else parse_nfl_box(summ)
        except Exception as e:
            log.warning("box %s failed: %s", g.espn_event_id, e)
            if "404" in str(e):   # ESPN has no box for this event; don't retry forever
                async with session_scope() as s:
                    await s.execute(text("UPDATE games SET boxscore_loaded=true WHERE id=:g"), {"g": g.id})
            continue
        gdate = (g.start_time - timedelta(hours=5)).date()   # US local game date
        async with session_scope() as s:
            await upsert(s, Player, [dict(sport=sport, espn_id=r["espn_id"], full_name=r["name"],
                                          name_key=name_key(r["name"]), position=r.get("position"),
                                          team_abbr=r["team_abbr"]) for r in rows],
                         ["sport", "espn_id"], ["full_name", "name_key"])   # never overwrite current team from history
            ids = await player_ids_by_espn(s, sport)
            out = []
            for r in rows:
                opp = g.away if r["team_abbr"] == g.home else g.home
                base = dict(player_id=ids[r["espn_id"]], game_id=g.id, game_date=gdate, team_abbr=r["team_abbr"],
                            opp_abbr=opp)
                if sport == "NBA":
                    out.append(dict(base, season=g.season, is_home=r["team_abbr"] == g.home,
                                    **{k: r[k] for k in ("starter", "minutes", "pts", "reb", "oreb", "ast", "fg3m",
                                                         "stl", "blk", "tov", "fga", "fta", "plus_minus")}))
                else:
                    stats = {k: r.get(k, 0.0) for k in NFL_STAT_COLS}
                    out.append(dict(base, season=int(g.season), week=g.week or 0, position=r.get("position"), **stats))
            model = NbaPlayerGamelog if sport == "NBA" else NflPlayerGame
            n += await upsert(s, model, out, ["player_id", "game_id"])
            if sport == "NFL" and out:   # target share from team totals
                await s.execute(text("""UPDATE nfl_player_games p SET target_share = p.targets / NULLIF(t.tt,0)
                    FROM (SELECT team_abbr, SUM(targets) tt FROM nfl_player_games WHERE game_id=:g GROUP BY 1) t
                    WHERE p.game_id=:g AND p.team_abbr=t.team_abbr"""), {"g": g.id})
            await s.execute(text("UPDATE games SET boxscore_loaded=true WHERE id=:g"), {"g": g.id})
    return n
