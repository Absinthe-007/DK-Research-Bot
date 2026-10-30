"""
Grade picks once box scores land: actual stat vs line -> win/loss/push (DNP = void, as DK
voids props for players who don't play), units won at the pick price, and closing price
(last DK snapshot before kickoff) for closing-line value.
"""
from __future__ import annotations

from sqlalchemy import text

from app.db.base import session_scope
from app.db.reference import MARKET_INDEX
from app.engine.odds_math import profit_per_unit
from app.engine.projector import NBA_COLS, NFL_COLS, stat_sql
from app.ingestion.common import tracked


@tracked("grade")
async def grade(sport: str) -> int:
    table = "nba_player_gamelogs" if sport == "NBA" else "nfl_player_games"
    cols = NBA_COLS if sport == "NBA" else NFL_COLS
    n = 0
    async with session_scope() as s:
        picks = (await s.execute(text("""
            SELECT pk.* FROM picks pk JOIN games g ON g.id=pk.game_id
            WHERE pk.sport=:sp AND pk.result IS NULL AND g.completed AND g.boxscore_loaded"""), {"sp": sport})).mappings().all()
        for pk in picks:
            if not pk["active"]:
                # result is varchar(6): the old value 'dropped' (7 chars) crashed the whole nightly grade
                await s.execute(text("UPDATE picks SET result='pulled' WHERE id=:i"), {"i": pk["id"]})
                continue
            expr = stat_sql(MARKET_INDEX[pk["market_key"]][3], cols)
            actual = (await s.execute(text(f"SELECT {expr} FROM {table} WHERE player_id=:p AND game_id=:g"),
                                      {"p": pk["player_id"], "g": pk["game_id"]})).scalar()
            close = (await s.execute(text("""
                SELECT ps.price FROM prop_snapshots ps JOIN games g ON g.id=ps.game_id
                WHERE ps.game_id=:g AND ps.player_id=:p AND ps.market_key=:m AND ps.kind=:k AND ps.side=:sd
                  AND ps.line=:l AND ps.captured_at < g.start_time ORDER BY ps.captured_at DESC LIMIT 1"""),
                dict(g=pk["game_id"], p=pk["player_id"], m=pk["market_key"], k=pk["kind"], sd=pk["side"],
                     l=pk["line"]))).scalar()
            if actual is None:
                result, units = "void", 0.0
            else:
                actual = float(actual)
                if actual == pk["line"]:
                    result, units = "push", 0.0
                else:
                    won = actual > pk["line"] if pk["side"] == "Over" else actual < pk["line"]
                    result = "win" if won else "loss"
                    units = profit_per_unit(pk["price"]) if won else -1.0
            await s.execute(text("""UPDATE picks SET actual=:a, result=:r, profit_units=:u, close_price=:c WHERE id=:i"""),
                            dict(a=actual, r=result, u=units, c=close, i=pk["id"]))
            n += 1
    return n
