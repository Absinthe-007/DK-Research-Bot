"""
Market feed: timestamped odds from The Odds API (`ODDS_API_KEY`, DraftKings) plus the public Kalshi and
Polymarket prediction-market quotes, stored as OPEN vs CURRENT snapshots in `market_lines`.

* `open_*` is written once, on first sight of a (source, event, market, selection, line).
* `cur_*` is overwritten on every poll **until the event starts**; after that the row is frozen, so `cur_*`
  is the closing line used for Closing Line Value.
* Fair (vig-free) probabilities are attached to two-way pairs: Shin for American prices, plain
  normalisation for prediction-market probabilities (they carry no bookmaker vig, only a bid/ask spread).
* The Odds API costs credits per market per call. All spending goes through the existing
  `app.ingestion.oddsapi.Budget` (per-run cap + account reserve). Kalshi and Polymarket are free.

Kalshi start times are approximate: its payload has the expected end of the event, not the tip-off, so
tip-off is estimated as that time minus the typical game length. Polymarket and The Odds API give exact start times.
"""
from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import or_
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.config import get_settings
from app.db.base import session_scope
from app.db.models_ext import MarketLine
from app.engine.devig import fair_from_probs, two_way_fair
from app.engine.odds_math import implied_prob
from app.ingestion.common import tracked
from app.ingestion.http import http
from app.ingestion.oddsapi import BASE as ODDS_BASE
from app.ingestion.oddsapi import PROP_MARKETS, SPORT_KEY, Budget

log = logging.getLogger(__name__)

BOOK = "draftkings"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
POLY_GAMMA = "https://gamma-api.polymarket.com"
POLY_CLOB = "https://clob.polymarket.com"
KALSHI_SERIES = {  # series_ticker -> (sport, our market_key); override with env KALSHI_SERIES="SPORT:SERIES:market,..."
    "KXNBAGAME": ("NBA", "h2h"), "KXNBATOTAL": ("NBA", "totals"), "KXNBASPREAD": ("NBA", "spreads"),
    "KXNFLGAME": ("NFL", "h2h"), "KXNFLTOTAL": ("NFL", "totals"), "KXNFLSPREAD": ("NFL", "spreads"),
}
POLY_SERIES = {"NBA": "10345", "NFL": "12185"}
POLY_TYPES = {"moneyline": "h2h", "spreads": "spreads", "totals": "totals"}
GAME_HOURS = {"NBA": 2.5, "NFL": 3.25}


@dataclass
class Quote:
    source: str
    sport: str
    event_key: str
    market_key: str
    selection: str
    line: float = 0.0
    price: int | None = None           # American, for sportsbook quotes
    prob: float | None = None          # raw implied (book) or mid (prediction market)
    fair: float | None = None          # vig-free probability when a complete pair is known
    pair_key: str | None = None
    home_team: str | None = None
    away_team: str | None = None
    commence_time: datetime | None = None
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> tuple:
        return (self.source, self.event_key, self.market_key, self.selection, float(self.line))


def _dt(s) -> datetime | None:
    if not s:
        return None
    try:
        txt = str(s).strip().replace("Z", "+00:00").replace(" ", "T", 1)
        # Python 3.10 rejects short offsets like "+00" / "-05"; pad to "+00:00".
        txt = re.sub(r"([+-]\d{2})$", r"\1:00", txt) if "T" in txt else txt
        d = datetime.fromisoformat(txt)
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _f(v) -> float | None:
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ fair-probability pairing
def attach_fair(quotes: list[Quote]) -> list[Quote]:
    groups: dict[str, list[Quote]] = defaultdict(list)
    for q in quotes:
        if q.pair_key:
            groups[q.pair_key].append(q)
    for g in groups.values():
        if len(g) != 2:
            continue
        a, b = g
        if a.price is not None and b.price is not None:
            a.fair, b.fair = two_way_fair(a.price, b.price, "shin")
        elif a.prob is not None and b.prob is not None:
            a.fair, b.fair = fair_from_probs([a.prob, b.prob])
    for q in quotes:                                     # one-sided prediction-market quotes: the mid is the fair price
        if q.fair is None and q.price is None and q.prob is not None and not q.pair_key:
            q.fair = q.prob
    return quotes


# ------------------------------------------------------------------ The Odds API
def parse_odds_event(ev: dict, sport: str, bookmaker: str = BOOK) -> list[Quote]:
    """One Odds API event (h2h / spreads / totals / player props) -> quotes from `bookmaker` only."""
    home, away, eid = ev.get("home_team"), ev.get("away_team"), ev.get("id")
    commence = _dt(ev.get("commence_time"))
    src = f"oddsapi:{bookmaker}"
    out: list[Quote] = []
    for bk in ev.get("bookmakers", []):
        if bk.get("key") != bookmaker:
            continue
        for mk in bk.get("markets", []):
            key, outs = mk.get("key"), mk.get("outcomes", [])
            for o in outs:
                price = o.get("price")
                if price is None:
                    continue
                price = int(price)
                name, point, player = o.get("name"), o.get("point"), o.get("description")
                base = dict(source=src, sport=sport, event_key=str(eid), market_key=key, price=price,
                            prob=implied_prob(price), home_team=home, away_team=away, commence_time=commence)
                if key == "h2h":
                    sel = "home" if name == home else "away" if name == away else str(name).lower()
                    out.append(Quote(selection=sel, line=0.0, pair_key=f"{eid}|h2h", **base))
                elif key == "spreads" and point is not None:
                    sel = "home" if name == home else "away"
                    out.append(Quote(selection=sel, line=float(point), pair_key=f"{eid}|spreads|{abs(float(point))}", **base))
                elif key == "totals" and point is not None:
                    out.append(Quote(selection=str(name), line=float(point), pair_key=f"{eid}|totals|{float(point)}", **base))
                elif str(key).startswith("player_") and player and name in ("Over", "Under") and point is not None:
                    out.append(Quote(selection=f"{name}|{player}", line=float(point),
                                     pair_key=f"{eid}|{key}|{player}|{float(point)}", **base))
                # one-sided props (Yes / anytime TD) carry no pair, so no fair price
    return out


def odds_api_key() -> str:
    return os.environ.get("ODDS_API_KEY") or get_settings().odds_api_key


async def fetch_odds_api(sport: str, budget: Budget, *, game_markets=("h2h", "spreads", "totals"), prop_markets=(),
                         prop_events_hours: float = 6.0, max_prop_events: int = 8) -> list[Quote]:
    key = odds_api_key()
    if not key:
        return []
    quotes: list[Quote] = []
    n = len(game_markets)
    if budget.can_spend(n):
        body, hdrs = await http().get(f"{ODDS_BASE}/{SPORT_KEY[sport]}/odds", params=dict(
            apiKey=key, regions="us", markets=",".join(game_markets), oddsFormat="american", bookmakers=BOOK, dateFormat="iso"))
        budget.record(hdrs, n)
        events = body if isinstance(body, list) else []
        for ev in events:
            quotes += parse_odds_event(ev, sport)
        if prop_markets:
            now = datetime.now(timezone.utc)
            soon = [e for e in events if (c := _dt(e.get("commence_time"))) and now < c < now + timedelta(hours=prop_events_hours)]
            allowed = [m for m in prop_markets if m in PROP_MARKETS[sport]]
            for ev in soon[:max_prop_events]:
                if not allowed or not budget.can_spend(len(allowed)):
                    break
                pb, ph = await http().get(f"{ODDS_BASE}/{SPORT_KEY[sport]}/events/{ev['id']}/odds", params=dict(
                    apiKey=key, regions="us", markets=",".join(allowed), oddsFormat="american", bookmakers=BOOK, dateFormat="iso"))
                budget.record(ph, len(allowed))
                quotes += parse_odds_event(pb, sport)
    else:
        log.info("odds api: credit budget exhausted (spent=%s remaining=%s)", budget.spent, budget.remaining)
    return quotes


# ------------------------------------------------------------------ Kalshi
def best_quotes_from_orderbook(ob: dict) -> tuple[float | None, float | None]:
    """(yes_bid, yes_ask) in probability units from a Kalshi orderbook payload.
    Handles `orderbook_fp` (dollar strings, ascending) and the legacy `orderbook` (cents). yes ask = 1 - best no bid."""
    fp = ob.get("orderbook_fp")
    if fp:
        yes = [_f(p) for p, _ in fp.get("yes_dollars", [])]
        no = [_f(p) for p, _ in fp.get("no_dollars", [])]
    else:
        legacy = ob.get("orderbook") or {}
        yes = [None if p is None else p / 100.0 for p, _ in (legacy.get("yes") or [])]
        no = [None if p is None else p / 100.0 for p, _ in (legacy.get("no") or [])]
    yes, no = [p for p in yes if p is not None], [p for p in no if p is not None]
    bid = max(yes) if yes else None
    ask = 1.0 - max(no) if no else None
    return bid, ask


def mid(bid: float | None, ask: float | None) -> float | None:
    if bid is not None and ask is not None and ask >= bid:
        return (bid + ask) / 2.0
    return bid if bid is not None else ask


def parse_kalshi_market(m: dict, sport: str, market_key: str) -> Quote | None:
    bid, ask = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
    bid = None if not bid else bid                  # 0.00 = nobody is bidding
    ask = None if ask is None or ask >= 1.0 else ask  # 1.00 = nobody is offering
    p = mid(bid, ask)
    if p is None:
        return None
    end = _dt(m.get("occurrence_datetime") or m.get("expected_expiration_time"))
    start = end - timedelta(hours=GAME_HOURS.get(sport, 3.0)) if end else None
    event = m.get("event_ticker") or m.get("ticker")
    return Quote(source="kalshi", sport=sport, event_key=event, market_key=market_key,
                 selection=f"yes|{m.get('yes_sub_title') or m.get('ticker')}",
                 line=_f(m.get("floor_strike")) or 0.0, prob=p, pair_key=event if market_key == "h2h" else None,
                 commence_time=start, extra={"ticker": m.get("ticker"), "yes_bid": bid, "yes_ask": ask})


async def fetch_kalshi(sport: str, series: dict | None = None, max_pages: int = 3) -> list[Quote]:
    quotes: list[Quote] = []
    for ticker, (sp, mkey) in (series or KALSHI_SERIES).items():
        if sp != sport:
            continue
        cursor = None
        for _ in range(max_pages):
            params = {"status": "open", "series_ticker": ticker, "limit": 200, **({"cursor": cursor} if cursor else {})}
            try:
                body, _ = await http().get(f"{KALSHI}/markets", params=params, cache_ttl=60)
            except Exception as e:
                log.warning("kalshi %s failed: %s", ticker, e)
                break
            for m in body.get("markets", []):
                q = parse_kalshi_market(m, sport, mkey)
                if q is None and m.get("ticker"):           # no top-of-book in the list payload: ask the order book
                    try:
                        ob, _ = await http().get(f"{KALSHI}/markets/{m['ticker']}/orderbook")
                        bid, ask = best_quotes_from_orderbook(ob)
                        m = {**m, "yes_bid_dollars": bid, "yes_ask_dollars": ask}
                        q = parse_kalshi_market(m, sport, mkey)
                    except Exception as e:
                        log.debug("kalshi orderbook %s: %s", m.get("ticker"), e)
                if q:
                    quotes.append(q)
            cursor = body.get("cursor")
            if not cursor:
                break
    return quotes


# ------------------------------------------------------------------ Polymarket
def _json_list(v) -> list:
    import json

    if isinstance(v, list):
        return v
    try:
        return json.loads(v) if v else []
    except (TypeError, ValueError):
        return []


def parse_polymarket_event(ev: dict, sport: str) -> list[Quote]:
    out: list[Quote] = []
    for m in ev.get("markets") or []:
        mkey = POLY_TYPES.get(m.get("sportsMarketType") or "")
        if not mkey:
            continue
        outcomes, prices = _json_list(m.get("outcomes")), [_f(x) for x in _json_list(m.get("outcomePrices"))]
        if len(outcomes) != 2:
            continue
        p0 = mid(_f(m.get("bestBid")), _f(m.get("bestAsk")))
        if p0 is None and prices and prices[0] is not None:
            p0 = prices[0]
        if p0 is None:
            continue
        start = _dt(m.get("gameStartTime") or ev.get("startTime"))
        eid = str(ev.get("slug") or m.get("slug"))
        for i, name in enumerate(outcomes):
            out.append(Quote(source="polymarket", sport=sport, event_key=eid, market_key=mkey, selection=str(name),
                             line=_f(m.get("line")) or 0.0, prob=p0 if i == 0 else 1.0 - p0,
                             pair_key=f"{eid}|{m.get('slug')}", commence_time=start,
                             extra={"token": (_json_list(m.get("clobTokenIds")) or [None, None])[i]}))
    return out


async def fetch_polymarket(sport: str, limit: int = 100) -> list[Quote]:
    try:
        body, _ = await http().get(f"{POLY_GAMMA}/events", cache_ttl=60, params=dict(
            series_id=POLY_SERIES[sport], active="true", closed="false", limit=limit, order="startDate", ascending="true"))
    except Exception as e:
        log.warning("polymarket %s failed: %s", sport, e)
        return []
    quotes: list[Quote] = []
    for ev in body if isinstance(body, list) else []:
        quotes += parse_polymarket_event(ev, sport)
    return quotes


# ------------------------------------------------------------------ persistence
def _row(q: Quote, now: datetime) -> dict:
    return dict(source=q.source, sport=q.sport, event_key=q.event_key, market_key=q.market_key, selection=q.selection,
                line=float(q.line), pair_key=q.pair_key, home_team=q.home_team, away_team=q.away_team,
                commence_time=q.commence_time, open_price=q.price, open_prob=q.prob, open_fair=q.fair, open_at=now,
                cur_price=q.price, cur_prob=q.prob, cur_fair=q.fair, cur_at=now, n_updates=1)


async def apply_snapshot(quotes: list[Quote], now: datetime | None = None, chunk: int = 1500) -> int:
    """Upsert quotes. Open columns are inserted once; current columns update only before the event starts."""
    now = now or datetime.now(timezone.utc)
    dedup = {q.key: q for q in attach_fair(quotes)}
    rows = [_row(q, now) for q in dedup.values() if q.commence_time is None or q.commence_time > now]
    if not rows:
        return 0
    async with session_scope() as s:
        for i in range(0, len(rows), chunk):
            stmt = pg_insert(MarketLine).values(rows[i:i + chunk])
            ex = stmt.excluded
            stmt = stmt.on_conflict_do_update(
                constraint="uq_market_line",
                set_=dict(cur_price=ex.cur_price, cur_prob=ex.cur_prob, cur_fair=ex.cur_fair, cur_at=ex.cur_at,
                          commence_time=ex.commence_time, n_updates=MarketLine.n_updates + 1),
                where=or_(MarketLine.commence_time.is_(None), MarketLine.commence_time > now))
            await s.execute(stmt)
    return len(rows)


@tracked("market_feed")
async def snapshot(sport: str, *, use_odds_api: bool = True, prop_markets=(), budget: Budget | None = None) -> int:
    """One polling pass for `sport` across all sources; each source fails independently."""
    cfg = get_settings()
    budget = budget or Budget(cfg.odds_api_max_credits_per_run, cfg.odds_api_min_remaining)
    quotes: list[Quote] = []
    if use_odds_api and odds_api_key():
        try:
            quotes += await fetch_odds_api(sport, budget, prop_markets=prop_markets)
        except Exception as e:
            log.warning("odds api failed: %s", e)
    quotes += await fetch_kalshi(sport)
    quotes += await fetch_polymarket(sport)
    return await apply_snapshot(quotes)
