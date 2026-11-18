"""
Post-game grader for settled predictions.

Differences from the original `app/engine/grade.py` (which is left untouched and keeps running):

* **Active-but-zero is a miss, not a DNP.** ESPN box scores omit players with no stats (NBA rows need
  minutes > 0, NFL rows need a stat line), so "no row" is ambiguous. We resolve it with participation evidence:
  NBA `nba_api_gamelog` / `nba_live_lineup`, NFL `nfl_usage_weekly` (snap counts + PBP). A player who was
  active but recorded nothing grades against a 0 (an Over loses). Only a player with positive evidence that he
  did not play is voided (`result='void'`, stake returned). With no evidence either way the pick stays pending
  rather than being guessed.
* **CLV is logged** per graded pick in `clv_log`: closing breakeven probability minus the breakeven probability
  at the price we flagged, plus the vig-free close when `market_lines` captured both sides.

Only picks with `result IS NULL` are touched, so it is safe to run alongside the original grader.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from app.db.base import session_scope
from app.db.reference import MARKET_INDEX
from app.engine.devig import clv as clv_prob_points
from app.engine.grade import settle
from app.engine.odds_math import implied_prob
from app.engine.projector import NBA_COLS, NFL_COLS, stat_sql
from app.ingestion.common import tracked
from app.utils.names import name_key

log = logging.getLogger(__name__)
MIN_TEAM_WEEK_ROWS = 15          # an NFL team-week with fewer usage rows than this is treated as "no data"


# ------------------------------------------------------------------ pure logic
def classify_nba(box_minutes: float | None, gamelog_minutes: float | None = None, live: dict | None = None) -> str:
    """'played' | 'dnp' | 'unknown' for an NBA player on a finished game."""
    for m in (box_minutes, gamelog_minutes):
        if m is not None and m > 0:
            return "played"
    if live:
        if live.get("played") or (live.get("minutes") or 0) > 0:
            return "played"
        if str(live.get("status") or "").upper() == "INACTIVE" or (live.get("game_status") == 3 and not live.get("played")):
            return "dnp"
    if gamelog_minutes is not None:      # a log row with zero minutes: on the roster, did not play
        return "dnp"
    return "unknown"


def classify_nfl(has_box_row: bool, usage_row_present: bool, team_week_rows: int) -> str:
    """Box-score row or any snap/PBP usage row means he was active. Absent while the team's week is
    otherwise well covered means inactive (DNP); otherwise we cannot tell."""
    if has_box_row or usage_row_present:
        return "played"
    return "dnp" if team_week_rows >= MIN_TEAM_WEEK_ROWS else "unknown"


def grade_pick(side: str, line: float | None, price: int, status: str, actual: float | None):
    """-> (result, actual, profit_units) or None when it must stay pending."""
    if status == "dnp":
        return "void", None, 0.0
    if status != "played":
        return None
    act = 0.0 if actual is None else float(actual)    # active with no recorded stats: a real zero
    res, profit = settle(side, line, act, price)
    return res, act, profit


def clv_record(placed_price: int | None, close_price: int | None, close_fair: float | None = None) -> dict | None:
    """Closing line value in probability points (positive = we beat the close)."""
    if placed_price is None or close_price is None:
        return None
    placed, close = implied_prob(int(placed_price)), implied_prob(int(close_price))
    pts = clv_prob_points(int(placed_price), close)
    return {"placed_prob": placed, "close_prob": close, "close_fair": close_fair, "clv_prob": pts, "beat_close": pts > 0}


# ------------------------------------------------------------------ evidence queries
async def _nba_evidence(s, nk: str, start) -> tuple[float | None, dict | None]:
    glog = (await s.execute(text("""
        SELECT minutes FROM nba_api_gamelog WHERE name_key = :nk
          AND game_date BETWEEN :d0 AND :d1 ORDER BY game_date LIMIT 1"""),
        {"nk": nk, "d0": (start - timedelta(days=1)).date(), "d1": (start + timedelta(days=1)).date()})).first()
    live = (await s.execute(text("""
        SELECT status, played, minutes, game_status FROM nba_live_lineup WHERE name_key = :nk
          AND game_status = 3 AND updated_at BETWEEN :a AND :b ORDER BY updated_at DESC LIMIT 1"""),
        {"nk": nk, "a": start, "b": start + timedelta(hours=12)})).mappings().first()
    return (None if glog is None else (glog.minutes if glog.minutes is not None else 0.0)), (dict(live) if live else None)


async def _nfl_evidence(s, nk: str, season, week, team) -> tuple[bool, int]:
    if season is None or week is None:
        return False, 0
    try:
        season_i = int(str(season)[:4])
    except ValueError:
        return False, 0
    present = (await s.execute(text("""
        SELECT 1 FROM nfl_usage_weekly WHERE season = :se AND week = :wk AND name_key = :nk LIMIT 1"""),
        {"se": season_i, "wk": week, "nk": nk})).first() is not None
    n = 0
    if team:
        n = (await s.execute(text("""
            SELECT COUNT(*) FROM nfl_usage_weekly WHERE season = :se AND week = :wk AND team_abbr = :tm"""),
            {"se": season_i, "wk": week, "tm": team})).scalar() or 0
    return present, int(n)


async def _closing(s, pk: dict, nk: str) -> tuple[int | None, float | None, str | None]:
    """(close_price, close_fair, source). Prefers the market feed's frozen close, then the last pre-game snapshot."""
    cands = (await s.execute(text("""
        SELECT selection, cur_price, cur_fair FROM market_lines
        WHERE source LIKE 'oddsapi:%' AND market_key = :m AND line IS NOT DISTINCT FROM :ln
          AND selection LIKE :side AND commence_time BETWEEN :a AND :b"""),
        {"m": pk["market_key"], "ln": pk["line"], "side": f"{pk['side']}|%",
         "a": pk["start_time"] - timedelta(hours=3), "b": pk["start_time"] + timedelta(hours=3)})).all()
    for sel, cur_price, cur_fair in cands:
        if name_key(sel.split("|", 1)[1]) == nk and cur_price is not None:
            return int(cur_price), cur_fair, "market_lines"
    if pk.get("close_price") is not None:
        return int(pk["close_price"]), None, "picks"
    row = (await s.execute(text("""
        SELECT price FROM prop_snapshots
        WHERE game_id = :g AND player_id = :p AND market_key = :m AND kind = :k AND side = :sd AND line IS NOT DISTINCT FROM :ln
          AND captured_at <= :st AND NOT COALESCE(price_estimated, false)
        ORDER BY captured_at DESC LIMIT 1"""),
        {"g": pk["game_id"], "p": pk["player_id"], "m": pk["market_key"], "k": pk["kind"], "sd": pk["side"],
         "ln": pk["line"], "st": pk["start_time"]})).first()
    return (int(row.price), None, "prop_snapshots") if row else (None, None, None)


# ------------------------------------------------------------------ jobs
@tracked("grade_v2")
async def grade_settled(sport: str, void_unknown_after_hours: float | None = None) -> int:
    """Grade still-open picks on finished games. Returns the number settled (wins, losses, pushes and voids)."""
    table = "nba_player_gamelogs" if sport == "NBA" else "nfl_player_games"
    allowed = NBA_COLS if sport == "NBA" else NFL_COLS
    settled = pending = 0
    async with session_scope() as s:
        picks = (await s.execute(text("""
            SELECT pk.id, pk.game_id, pk.player_id, pk.market_key, pk.kind, pk.side, pk.line, pk.price, pk.close_price,
                   g.start_time, g.season, g.week, p.name_key, p.team_abbr
            FROM picks pk JOIN games g ON g.id = pk.game_id JOIN players p ON p.id = pk.player_id
            WHERE pk.sport = :sp AND pk.result IS NULL AND g.completed = true
              AND g.start_time < now() - interval '4 hours'"""), {"sp": sport})).mappings().all()
        for pk in picks:
            info = MARKET_INDEX.get(pk["market_key"])
            if not info:
                continue
            col = stat_sql(info[3], allowed, "l")
            extra = ", l.minutes AS minutes" if sport == "NBA" else ""
            box = (await s.execute(text(f"""
                SELECT {col} AS actual{extra} FROM {table} l WHERE l.game_id = :g AND l.player_id = :p LIMIT 1"""),
                {"g": pk["game_id"], "p": pk["player_id"]})).first()
            actual = None if box is None or box.actual is None else float(box.actual)
            if sport == "NBA":
                glog, live = (None, None) if box is not None else await _nba_evidence(s, pk["name_key"], pk["start_time"])
                status = classify_nba(None if box is None else float(box.minutes or 0), glog, live)
            else:
                present, n = (False, 0) if box is not None else await _nfl_evidence(
                    s, pk["name_key"], pk["season"], pk["week"], pk["team_abbr"])
                status = classify_nfl(box is not None, present, n)
            graded = grade_pick(pk["side"], pk["line"], int(pk["price"]), status, actual)
            if graded is None:
                age_h = (datetime.now(timezone.utc) - pk["start_time"]).total_seconds() / 3600.0
                if void_unknown_after_hours is not None and age_h and age_h > void_unknown_after_hours:
                    graded = ("void", None, 0.0)
                else:
                    pending += 1
                    continue
            result, act, profit = graded
            close_price, _fair, _src = await _closing(s, pk, pk["name_key"])
            res = await s.execute(text("""
                UPDATE picks SET actual = :act, result = :res, profit_units = :pu,
                       close_price = COALESCE(:cp, close_price), graded_at = now()
                WHERE id = :id AND result IS NULL"""),
                {"act": act, "res": result, "pu": profit, "cp": close_price, "id": pk["id"]})
            settled += res.rowcount or 0
    log.info("%s grade_v2: %d settled, %d pending (no participation evidence yet)", sport, settled, pending)
    return settled


@tracked("clv_log")
async def log_clv(sport: str) -> int:
    """Write a clv_log row for every graded (win/loss/push) pick that doesn't have one yet."""
    n = 0
    async with session_scope() as s:
        picks = (await s.execute(text("""
            SELECT pk.id, pk.game_id, pk.player_id, pk.market_key, pk.kind, pk.side, pk.line, pk.price, pk.close_price,
                   g.start_time, p.name_key
            FROM picks pk JOIN games g ON g.id = pk.game_id JOIN players p ON p.id = pk.player_id
            WHERE pk.sport = :sp AND pk.result IN ('win', 'loss', 'push')
              AND NOT EXISTS (SELECT 1 FROM clv_log c WHERE c.pick_id = pk.id)"""), {"sp": sport})).mappings().all()
        for pk in picks:
            close_price, fair, src = await _closing(s, pk, pk["name_key"])
            rec = clv_record(int(pk["price"]), close_price, fair)
            if rec is None:
                continue
            await s.execute(text("""
                INSERT INTO clv_log (pick_id, placed_price, placed_prob, close_price, close_prob, close_fair,
                                     clv_prob, beat_close, source)
                VALUES (:id, :pp, :ppr, :cp, :cpr, :cf, :clv, :beat, :src)
                ON CONFLICT (pick_id) DO NOTHING"""),
                {"id": pk["id"], "pp": int(pk["price"]), "ppr": rec["placed_prob"], "cp": close_price,
                 "cpr": rec["close_prob"], "cf": fair, "clv": rec["clv_prob"], "beat": rec["beat_close"], "src": src})
            n += 1
    return n


async def run(sport: str, void_unknown_after_hours: float | None = None) -> dict[str, int]:
    return {"settled": await grade_settled(sport, void_unknown_after_hours), "clv_logged": await log_clv(sport)}
