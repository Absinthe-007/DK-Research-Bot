"""
grade.py: Settles active picks in the database against completed box scores,
captures closing line value (close_price), and computes profit_units.
Supports both the live pipeline (`grade`) and standalone unit tests (`grade_bet`).
"""
from __future__ import annotations

import logging

from sqlalchemy import text

from app.db.base import session_scope
from app.db.reference import MARKET_INDEX
from app.engine.odds_math import american_to_implied, profit_per_unit
from app.engine.projector import NBA_COLS, NFL_COLS, stat_sql
from app.ingestion.common import tracked

log = logging.getLogger(__name__)


def settle(side: str, line: float | None, actual: float, price: int) -> tuple[str, float]:
    """Returns (result, profit_units) for a 1-unit flat wager."""
    thr = 0.5 if line is None else float(line)
    if actual == thr:
        return "push", 0.0
    won = (actual > thr) if side == "Over" else (actual < thr)
    if won:
        return "win", round(profit_per_unit(price), 4)
    return "loss", -1.0


def grade_bet(bet_result: str, stake: float, odds: int, closing_odds: int) -> dict:
    """Standalone bet grading and CLV evaluator for unit tests."""
    profit = 0.0
    if bet_result == "win":
        profit = stake * profit_per_unit(odds)
    elif bet_result == "loss":
        profit = -stake
    elif bet_result == "push":
        profit = 0.0
    else:
        raise ValueError("Invalid bet_result. Use 'win', 'loss', or 'push'.")

    placed_imp = american_to_implied(odds)
    closing_imp = american_to_implied(closing_odds)
    clv_pct = (closing_imp - placed_imp) * 100.0

    return {
        "result": bet_result,
        "stake": round(stake, 2),
        "profit_loss": round(profit, 2),
        "roi_percentage": round((profit / stake) * 100.0, 2) if stake > 0 else 0.0,
        "placed_odds": odds,
        "closing_odds": closing_odds,
        "clv_percentage": round(clv_pct, 2),
        "beat_closing_line": closing_imp > placed_imp,
    }


@tracked("grade_picks")
async def grade(sport: str) -> int:
    """Grades unsettled picks for completed games in `sport` and records closing prices."""
    async with session_scope() as s:
        unsettled = (
            await s.execute(
                text("""
                SELECT pk.id, pk.game_id, pk.player_id, pk.market_key, pk.kind,
                       pk.side, pk.line, pk.price, g.start_time
                FROM picks pk
                JOIN games g ON g.id = pk.game_id
                WHERE pk.sport = :sp AND pk.result IS NULL AND g.completed = true
                """),
                {"sp": sport},
            )
        ).mappings().all()

        if not unsettled:
            return 0

        graded_count = 0
        table = "nba_player_gamelogs" if sport == "NBA" else "nfl_player_games"
        allowed = NBA_COLS if sport == "NBA" else NFL_COLS

        for pk in unsettled:
            m_info = MARKET_INDEX.get(pk["market_key"])
            if not m_info:
                continue
            _key, _sp, _lbl, expr, _fam = m_info
            col_sql = stat_sql(expr, allowed, "l")

            row = (
                await s.execute(
                    text(f"""
                    SELECT {col_sql} AS actual
                    FROM {table} l
                    WHERE l.game_id = :g AND l.player_id = :p
                    LIMIT 1
                    """),
                    {"g": pk["game_id"], "p": pk["player_id"]},
                )
            ).first()

            if row is None or row.actual is None:
                continue

            actual_val = float(row.actual)
            res_str, prof_u = settle(pk["side"], pk["line"], actual_val, int(pk["price"]))

            # Look up the last pre-game price captured for that exact (game, player, market, kind, side, line)
            close_row = (
                await s.execute(
                    text("""
                    SELECT price FROM prop_snapshots
                    WHERE game_id = :g AND player_id = :p AND market_key = :m
                      AND kind = :k AND side = :sd
                      AND (line = :ln OR (line IS NULL AND :ln IS NULL))
                      AND captured_at <= :st
                      AND NOT COALESCE(price_estimated, false)
                    ORDER BY captured_at DESC LIMIT 1
                    """),
                    {
                        "g": pk["game_id"],
                        "p": pk["player_id"],
                        "m": pk["market_key"],
                        "k": pk["kind"],
                        "sd": pk["side"],
                        "ln": pk["line"],
                        "st": pk["start_time"],
                    },
                )
            ).first()
            close_price = int(close_row.price) if close_row and close_row.price is not None else None

            await s.execute(
                text("""
                UPDATE picks
                SET actual = :act, result = :res, profit_units = :pu,
                    close_price = COALESCE(:cp, close_price), graded_at = now()
                WHERE id = :id
                """),
                {
                    "act": actual_val,
                    "res": res_str,
                    "pu": prof_u,
                    "cp": close_price,
                    "id": pk["id"],
                },
            )
            graded_count += 1

        log.info("%s grade: %d/%d picks settled", sport, graded_count, len(unsettled))
        return graded_count