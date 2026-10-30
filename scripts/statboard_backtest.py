"""
Walk-forward backtest of the stat board: for each day, rewind the clock to just before that day's
first game, project every player with only the data that existed then, and score against the box
scores. Players who were projected but never got in the game count as a miss (actual = 0) and are
reported as the DNP rate - the live accuracy panel can't see those.

    python -m scripts.statboard_backtest --sport NBA --start 2026-10-24 --end 2026-10-29
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import text

from app.db.base import session_scope


async def run(sport: str, start: date, end: date, module: str = "app.engine.statboard") -> pd.DataFrame:
    sb = importlib.import_module(module)
    frames = []
    async with session_scope() as s:
        days = (await s.execute(text("""
            SELECT DISTINCT date_trunc('day', start_time - interval '10 hours') d, MIN(start_time) first
            FROM games WHERE sport=:sp AND completed AND boxscore_loaded AND season_type IN (2,3)
              AND start_time >= :a AND start_time < :b GROUP BY 1 ORDER BY 1"""),
            {"sp": sport, "a": datetime.combine(start, datetime.min.time(), timezone.utc),
             "b": datetime.combine(end + timedelta(days=1), datetime.min.time(), timezone.utc)})).all()
        for _, first in days:
            as_of = first - timedelta(minutes=30)
            rows = await (sb._nba(s, as_of) if sport == "NBA" else sb._nfl(s, as_of))
            df = pd.DataFrame(rows)
            if df.empty:
                continue
            # keep this slate only (NFL looks 8 days ahead)
            gids = (await s.execute(text("""SELECT id FROM games WHERE id = ANY(:g) AND start_time < :hi"""),
                                    {"g": df.game_id.unique().tolist(), "hi": as_of + timedelta(hours=20)})).scalars().all()
            frames.append(df[df.game_id.isin(gids)])
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames)
        table = "nba_player_gamelogs" if sport == "NBA" else "nfl_player_games"
        cols = sorted(set(df.stat) - {"pra", "td_prob"})
        box = pd.DataFrame((await s.execute(text(
            f"SELECT game_id, player_id, {', '.join(cols)} FROM {table} WHERE game_id = ANY(:g)"),
            {"g": df.game_id.unique().tolist()})).mappings().all())
    if sport == "NBA":
        box["pra"] = box.pts + box.reb + box.ast
    else:
        box["td_prob"] = ((box.rush_tds + box.rec_tds) > 0).astype(float)
    long = box.melt(["game_id", "player_id"], var_name="stat", value_name="actual")
    out = df.merge(long, on=["game_id", "player_id", "stat"], how="left")
    out["played"] = out.actual.notna()
    out["actual"] = out.actual.fillna(0.0)
    return out


def summarize(out: pd.DataFrame) -> pd.DataFrame:
    played = out[out.played]
    g = out.groupby("stat")
    res = pd.DataFrame({
        "n": g.size(),
        "dnp_pct": 100 * (1 - g.played.mean()),
        "mae_all": g.apply(lambda d: (d["mean"] - d.actual).abs().mean()),
        "mae_played": played.groupby("stat").apply(lambda d: (d["mean"] - d.actual).abs().mean()),
        "bias_played": played.groupby("stat").apply(lambda d: (d["mean"] - d.actual).mean()),
        "in_range": played.groupby("stat").apply(lambda d: d.actual.between(d.lo, d.hi).mean()),
    })
    return res.round(3)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", choices=["NBA", "NFL"], required=True)
    ap.add_argument("--start", type=date.fromisoformat, required=True)
    ap.add_argument("--end", type=date.fromisoformat, default=date.today())
    ap.add_argument("--module", default="app.engine.statboard")
    a = ap.parse_args()
    out = await run(a.sport, a.start, a.end, a.module)
    if out.empty:
        print("nothing to score"); return
    print(f"{a.sport}: {out[['game_id','player_id']].drop_duplicates().shape[0]} player-games, "
          f"{out.game_id.nunique()} games")
    print(summarize(out).to_string())


if __name__ == "__main__":
    asyncio.run(main())
