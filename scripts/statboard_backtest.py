"""
Walk-forward backtest of the stat board: for each slate day, rewind the clock to just before
that day's first game, project every player with only the data that existed then, and score
against the completed box scores.

Players who were projected but never got in the game count as a miss in `mae_all` (actual = 0)
and are reported as `dnp_pct` - the live accuracy panel can't see those.

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
        days = (
            await s.execute(
                text("""
                SELECT date_trunc('day', start_time - interval '10 hours') AS d,
                       MIN(start_time) AS first
                FROM games
                WHERE sport = :sp AND completed AND boxscore_loaded AND season_type IN (2, 3)
                  AND start_time >= :a AND start_time < :b
                GROUP BY 1 ORDER BY 1
                """),
                {
                    "sp": sport,
                    "a": datetime.combine(start, datetime.min.time(), timezone.utc),
                    "b": datetime.combine(end + timedelta(days=1), datetime.min.time(), timezone.utc),
                },
            )
        ).all()

        for slate_day, first in days:
            as_of = first - timedelta(minutes=30)
            rows = await (sb._nba(s, as_of) if sport == "NBA" else sb._nfl(s, as_of))
            df = pd.DataFrame(rows)
            if df.empty:
                continue
            # Keep only games belonging to this exact slate day so multi-day lookaheads never double-count
            gids = (
                await s.execute(
                    text("""
                    SELECT id FROM games
                    WHERE id = ANY(:g)
                      AND date_trunc('day', start_time - interval '10 hours') = :d
                    """),
                    {"g": df.game_id.unique().tolist(), "d": slate_day},
                )
            ).scalars().all()
            if gids:
                frames.append(df[df.game_id.isin(gids)])

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["game_id", "player_id", "stat"], keep="last")

        table = "nba_player_gamelogs" if sport == "NBA" else "nfl_player_games"
        min_filter = "AND minutes > 0" if sport == "NBA" else ""
        cols = sorted(set(df.stat) - {"pra", "td_prob"})
        box = pd.DataFrame(
            (
                await s.execute(
                    text(
                        f"SELECT game_id, player_id, {', '.join(cols)} "
                        f"FROM {table} WHERE game_id = ANY(:g) {min_filter}"
                    ),
                    {"g": df.game_id.unique().tolist()},
                )
            )
            .mappings()
            .all()
        )
        if box.empty:
            return pd.DataFrame()

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
    df = out.copy()
    df["err"] = df["mean"] - df["actual"]
    df["abs_err"] = df["err"].abs()
    df["in_rng"] = (df["actual"] >= df["lo"]) & (df["actual"] <= df["hi"])

    played = df[df["played"]]
    g_all = df.groupby("stat")
    g_pl = played.groupby("stat")

    res = pd.DataFrame(
        {
            "n": g_all.size(),
            "dnp_pct": 100.0 * (1.0 - g_all["played"].mean()),
            "mae_all": g_all["abs_err"].mean(),
            "mae_played": g_pl["abs_err"].mean(),
            "bias_played": g_pl["err"].mean(),
            "in_range": g_pl["in_rng"].mean(),
        }
    )
    return res.round(3)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", choices=["NBA", "NFL"], required=True)
    ap.add_argument("--start", type=date.fromisoformat, required=True)
    ap.add_argument("--end", type=date.fromisoformat, default=date.today())
    ap.add_argument("--module", default="app.engine.statboard")
    a = ap.parse_args()
    out = await run(a.sport, a.start, a.end, a.module)
    if out.empty:
        print("nothing to score")
        return
    n_pg = out[["game_id", "player_id"]].drop_duplicates().shape[0]
    print(f"{a.sport}: {n_pg} player-games, {out.game_id.nunique()} games")
    print(summarize(out).to_string())


if __name__ == "__main__":
    asyncio.run(main())