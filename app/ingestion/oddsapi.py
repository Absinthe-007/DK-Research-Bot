"""
The Odds API (https://the-odds-api.com) - OPTIONAL second source for DraftKings prices.

ESPN stays the primary feed. This module only runs when ODDS_API_KEY is set, and it spends credits
sparingly (the free tier is ~500 credits/month):

  * Player props: for upcoming games where ESPN published a prop LINE without a PRICE (the bot
    would otherwise assume -115/-115 and refuse to flag it), fetch just those markets from
    DraftKings and store the real prices as newer snapshots at the same line.
  * Game lines: for upcoming games where ESPN has no DraftKings spread/total/moneyline, fetch
    h2h/spreads/totals for the whole sport in one call.

Cost model (per the Odds API docs): /odds costs 1 credit per market per region; /events is free;
/events/{id}/odds costs 1 credit per market per region. We use regions=us + bookmakers=draftkings,
so a request costs (#markets requested). Safety rails:
  * ODDS_API_MAX_CREDITS_PER_RUN caps one refresh.
  * ODDS_API_MIN_REMAINING stops all calls once the account's remaining credits (from the
    x-requests-remaining response header) fall below the reserve.
  * A game is re-fetched at most once per ODDS_API_MIN_REFETCH_MINUTES.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from app.config import get_settings
from app.db.base import session_scope
from app.db.models import IngestRun, PropSnapshot
from app.ingestion.common import parse_american, parse_float, player_index, resolve_player, tracked, ts
from app.ingestion.espn import _insert_changed_lines
from app.ingestion.http import http

log = logging.getLogger(__name__)
BASE = "https://api.the-odds-api.com/v4/sports"
SPORT_KEY = {"NBA": "basketball_nba", "NFL": "americanfootball_nfl"}
BOOK = "draftkings"

# Odds API player-prop market key -> our market_key (same names for the markets we model).
PROP_MARKETS = {
    "NBA": {
        "player_points": "player_points", "player_rebounds": "player_rebounds", "player_assists": "player_assists",
        "player_threes": "player_threes", "player_blocks": "player_blocks", "player_steals": "player_steals",
        "player_points_rebounds_assists": "player_points_rebounds_assists",
        "player_points_rebounds": "player_points_rebounds", "player_points_assists": "player_points_assists",
        "player_rebounds_assists": "player_rebounds_assists",
    },
    "NFL": {
        "player_pass_yds": "player_pass_yds", "player_pass_tds": "player_pass_tds",
        "player_pass_completions": "player_pass_completions", "player_pass_attempts": "player_pass_attempts",
        "player_pass_interceptions": "player_pass_interceptions", "player_rush_yds": "player_rush_yds",
        "player_rush_attempts": "player_rush_attempts", "player_receptions": "player_receptions",
        "player_reception_yds": "player_reception_yds", "player_rush_reception_yds": "player_rush_reception_yds",
    },
}


class Budget:
    """Tracks credits for one run and the account-wide remaining balance."""

    def __init__(self, max_run: int, reserve: int):
        self.max_run, self.reserve = max_run, reserve
        self.spent = 0
        self.remaining: int | None = None

    def can_spend(self, n: int) -> bool:
        if self.spent + n > self.max_run:
            return False
        return self.remaining is None or self.remaining - n >= self.reserve

    def record(self, hdrs: dict, n: int) -> None:
        self.spent += n
        h = {k.lower(): v for k, v in hdrs.items()}
        try:
            self.remaining = int(h["x-requests-remaining"])
        except (KeyError, ValueError):
            pass


def enabled() -> bool:
    return bool(get_settings().odds_api_key)


async def _get(path: str, params: dict, budget: Budget, cost: int):
    if not budget.can_spend(cost):
        raise RuntimeError(f"Odds API budget stop (spent {budget.spent}, remaining {budget.remaining})")
    st = get_settings()
    body, hdrs = await http().get(f"{BASE}/{path}", params={**params, "apiKey": st.odds_api_key}, cache_ttl=60)
    budget.record(hdrs, cost)
    return body, hdrs


def _nick(name: str) -> str:
    """'Los Angeles Clippers' / 'LA Clippers' -> 'clippers' (ESPN and the Odds API disagree on city names)."""
    parts = name.lower().replace(".", "").split()
    return " ".join(parts[-2:]) if len(parts) > 2 and parts[-2] == "trail" else (parts[-1] if parts else "")


def _same_team(a: str, b: str) -> bool:
    return _nick(a) == _nick(b)


async def _event_map(sport: str, budget: Budget, games: list) -> dict[int, str]:
    """Our game.id -> Odds API event id, matched on start time (+-20 min) and both team nicknames. /events is free."""
    body, hdrs = await http().get(f"{BASE}/{SPORT_KEY[sport]}/events",
                                  params={"apiKey": get_settings().odds_api_key}, cache_ttl=300)
    budget.record(hdrs, 0)
    out: dict[int, str] = {}
    for g in games:
        for ev in body:
            t = ts(ev.get("commence_time"))
            if t and abs(t - g.start_time) <= timedelta(minutes=20) \
                    and _same_team(ev["home_team"], g.home_name) and _same_team(ev["away_team"], g.away_name):
                out[g.id] = ev["id"]
                break
    return out


def parse_event_props(event: dict, sport: str) -> list[dict]:
    """Odds API event odds -> [{market_key, player_name, side, line, price}] for DraftKings only."""
    rows = []
    for bk in event.get("bookmakers", []):
        if bk.get("key") != BOOK:
            continue
        for mk in bk.get("markets", []):
            mkey = PROP_MARKETS[sport].get(mk.get("key"))
            if not mkey:
                continue
            for o in mk.get("outcomes", []):
                price, line = parse_american(o.get("price")), parse_float(o.get("point"))
                if price is None or line is None or o.get("name") not in ("Over", "Under"):
                    continue
                rows.append(dict(market_key=mkey, player_name=o.get("description") or "", side=o["name"],
                                 line=line, price=price, updated=ts(mk.get("last_update"))))
    return rows


async def _games_needing_prop_prices(sport: str, hours_ahead: int, refetch_min: int) -> list:
    """Upcoming games with ESPN prop lines whose latest snapshot has an assumed (estimated) price."""
    now = datetime.now(timezone.utc)
    async with session_scope() as s:
        rows = (await s.execute(text("""
            SELECT g.id, g.start_time, h.name AS home_name, a.name AS away_name,
                   array_agg(DISTINCT l.market_key) AS markets
            FROM games g JOIN teams h ON h.id=g.home_team_id JOIN teams a ON a.id=g.away_team_id
            JOIN (SELECT DISTINCT ON (game_id, player_id, market_key, kind, side, line)
                         game_id, market_key, price_estimated
                  FROM prop_snapshots WHERE kind='ou'
                  ORDER BY game_id, player_id, market_key, kind, side, line, captured_at DESC) l ON l.game_id=g.id
            WHERE g.sport=:sp AND NOT g.completed AND g.status='STATUS_SCHEDULED'
              AND g.start_time > :now AND g.start_time < :end AND l.price_estimated
              AND NOT EXISTS (SELECT 1 FROM ingest_runs r WHERE r.job = 'oddsapi_props:' || g.id
                              AND r.started_at > :cut)
            GROUP BY g.id, g.start_time, h.name, a.name ORDER BY g.start_time"""),
            {"sp": sport, "now": now, "end": now + timedelta(hours=hours_ahead),
             "cut": now - timedelta(minutes=refetch_min)})).all()
    return list(rows)


async def _mark_fetched(game_id: int) -> None:
    async with session_scope() as s:
        s.add(IngestRun(job=f"oddsapi_props:{game_id}", finished_at=datetime.now(timezone.utc), ok=True, rows=0))


@tracked("oddsapi_props")
async def sync_props(sport: str, hours_ahead: int = 80, budget: Budget | None = None) -> int:
    """Replace ESPN's assumed -115 prices with real DraftKings prices from the Odds API."""
    if not enabled():
        return 0
    st = get_settings()
    budget = budget or Budget(st.odds_api_max_credits_per_run, st.odds_api_min_remaining)
    games = await _games_needing_prop_prices(sport, hours_ahead, st.odds_api_min_refetch_minutes)
    if not games:
        return 0
    emap = await _event_map(sport, budget, games)
    total = 0
    for g in games:
        ev_id = emap.get(g.id)
        if not ev_id:
            log.info("oddsapi: no event match for game %s (%s @ %s)", g.id, g.away_name, g.home_name)
            continue
        wanted = [m for m in g.markets if m in PROP_MARKETS[sport]]
        if not wanted:
            continue
        try:
            ev, _ = await _get(f"{SPORT_KEY[sport]}/events/{ev_id}/odds",
                               {"regions": "us", "bookmakers": BOOK, "markets": ",".join(wanted),
                                "oddsFormat": "american"}, budget, cost=len(wanted))
        except RuntimeError as e:
            log.warning("%s", e)
            break
        except Exception as e:
            log.warning("oddsapi props failed for game %s: %s", g.id, e)
            continue
        await _mark_fetched(g.id)
        rows = parse_event_props(ev, sport)
        async with session_scope() as s:
            index = await player_index(s, sport)
            latest = {(r.player_id, r.market_key, r.side, r.line): (r.price, r.price_estimated) for r in (await s.execute(text("""
                SELECT DISTINCT ON (player_id, market_key, kind, side, line) player_id, market_key, side, line, price, price_estimated
                FROM prop_snapshots WHERE game_id=:g AND kind='ou'
                ORDER BY player_id, market_key, kind, side, line, captured_at DESC"""), {"g": g.id})).all()}
            new = []
            for r in rows:
                pid = resolve_player(r["player_name"], index)
                key = (pid, r["market_key"], r["side"], r["line"])
                cur = latest.get(key)
                # Only fill gaps: the line must already exist from ESPN with an assumed price.
                if pid is None or cur is None or not cur[1]:
                    continue
                new.append(PropSnapshot(game_id=g.id, player_id=pid, market_key=r["market_key"], kind="ou",
                                        side=r["side"], line=r["line"], price=r["price"], price_estimated=False,
                                        book_updated_at=r["updated"]))
            s.add_all(new)
            total += len(new)
    log.info("oddsapi props %s: %d real prices filled, %d credits spent, %s remaining",
             sport, total, budget.spent, budget.remaining)
    return total


@tracked("oddsapi_lines")
async def sync_game_lines(sport: str, hours_ahead: int = 24 * 8, budget: Budget | None = None) -> int:
    """Fallback for games where ESPN has no DraftKings game lines. One call (3 credits) covers the sport."""
    if not enabled():
        return 0
    st = get_settings()
    budget = budget or Budget(st.odds_api_max_credits_per_run, st.odds_api_min_remaining)
    now = datetime.now(timezone.utc)
    async with session_scope() as s:
        games = (await s.execute(text("""
            SELECT g.id, g.start_time, h.name AS home_name, a.name AS away_name
            FROM games g JOIN teams h ON h.id=g.home_team_id JOIN teams a ON a.id=g.away_team_id
            WHERE g.sport=:sp AND NOT g.completed AND g.status='STATUS_SCHEDULED'
              AND g.start_time > :now AND g.start_time < :end
              AND NOT EXISTS (SELECT 1 FROM odds_snapshots o WHERE o.game_id=g.id AND o.market_type='spreads')"""),
            {"sp": sport, "now": now, "end": now + timedelta(hours=hours_ahead)})).all()
    if not games:
        return 0
    try:
        events, _ = await _get(SPORT_KEY[sport] + "/odds",
                               {"regions": "us", "bookmakers": BOOK, "markets": "h2h,spreads,totals",
                                "oddsFormat": "american"}, budget, cost=3)
    except RuntimeError as e:
        log.warning("%s", e)
        return 0
    n = 0
    async with session_scope() as s:
        for g in games:
            ev = next((e for e in events if (t := ts(e.get("commence_time"))) and abs(t - g.start_time) <= timedelta(minutes=20)
                       and _same_team(e["home_team"], g.home_name) and _same_team(e["away_team"], g.away_name)), None)
            if not ev:
                continue
            rows = _line_rows(g.id, ev)
            await _insert_changed_lines(s, g.id, rows)
            n += len(rows)
    return n


def _line_rows(game_id: int, ev: dict) -> list[dict]:
    """Odds API h2h/spreads/totals (DraftKings) -> odds_snapshots rows, same shape ESPN produces."""
    rows = []
    bk = next((b for b in ev.get("bookmakers", []) if b.get("key") == BOOK), None)
    if not bk:
        return rows
    for mk in bk.get("markets", []):
        for o in mk.get("outcomes", []):
            price, pt = parse_american(o.get("price")), parse_float(o.get("point"))
            if mk["key"] in ("h2h", "spreads"):
                if o.get("name") == ev["home_team"]:
                    side = "home"
                elif o.get("name") == ev["away_team"]:
                    side = "away"
                else:
                    continue
                rows.append(dict(game_id=game_id, market_type=mk["key"], side=side, price=price,
                                 point=None if mk["key"] == "h2h" else pt, open_price=None, open_point=None))
            elif mk["key"] == "totals" and o.get("name") in ("Over", "Under"):
                rows.append(dict(game_id=game_id, market_type="totals", side=o["name"].lower(), price=price,
                                 point=pt, open_price=None, open_point=None))
    return [r for r in rows if r["price"] is not None or r["point"] is not None]
