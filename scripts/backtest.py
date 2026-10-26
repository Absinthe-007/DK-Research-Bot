"""
Walk-forward CALIBRATION check for the distribution layer (NBA gamelogs).

For every player-game with >= MIN prior games, project using ONLY prior games (no leakage):
EWMA mean/sd -> fitted family -> P(Over) at a synthetic line = prior 10-game median - 0.5.
Then report:
  * Brier score and log loss of P(Over)
  * reliability table: predicted-probability buckets vs realized hit rate
  * 80% interval coverage (should be ~0.80; lower = tails too thin)
Opponent/pace/weather factors are excluded on purpose - this isolates whether the
recency weighting + distribution family are calibrated. Tune `halflife` and families here.

    python -m scripts.backtest --market player_rebounds --halflife 4
"""
import argparse

import numpy as np
import pandas as pd
import asyncio

from sqlalchemy import text

from app.db.base import session_scope
from app.db.reference import MARKET_INDEX
from app.engine.baselines import ewm_mean_std
from app.engine.distributions import fit, prob_over_under

MIN = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="player_points")
    ap.add_argument("--halflife", type=float, default=4.0)
    ap.add_argument("--family", default=None, help="override family (poisson/negbin/normal/gamma)")
    a = ap.parse_args()
    _, sport, _, expr, family = MARKET_INDEX[a.market]
    assert sport == "NBA", "this script backtests NBA gamelogs"
    family = a.family or family
    async def load():
        async with session_scope() as s:
            res = await s.execute(text("SELECT player_id, game_date, minutes, pts, reb, ast, fg3m, stl, blk, tov "
                                       "FROM nba_player_gamelogs WHERE minutes > 10 ORDER BY player_id, game_date"))
            return pd.DataFrame(res.mappings().all())
    df = asyncio.run(load())
    df["y"] = df.eval(expr)

    recs = []
    for _, g in df.groupby("player_id"):
        y = g["y"].to_numpy(float)
        for i in range(MIN, len(y)):
            prior = y[max(0, i - 25):i]
            mu, sd = ewm_mean_std(prior, a.halflife)
            sd = max(sd, 0.6 * mu**0.5) if mu > 0 else 1.0
            line = np.median(prior[-10:]) - 0.5 if np.median(prior[-10:]) % 1 == 0 else np.median(prior[-10:])
            try:
                d = fit(family, max(mu, 0.1), sd)
            except ValueError:
                continue
            p_over, _, _ = prob_over_under(d, line)
            lo, hi = d._frozen().ppf(0.10), d._frozen().ppf(0.90)
            recs.append((p_over, float(y[i] > line), lo <= y[i] <= hi))
    r = pd.DataFrame(recs, columns=["p", "hit", "in80"])
    p = r["p"].clip(1e-4, 1 - 1e-4)
    print(f"market={a.market} family={family} halflife={a.halflife} n={len(r)}")
    print(f"Brier={np.mean((p - r.hit) ** 2):.4f}  LogLoss={-np.mean(r.hit*np.log(p)+(1-r.hit)*np.log(1-p)):.4f}"
          f"  80%-coverage={r.in80.mean():.3f}")
    r["bucket"] = pd.cut(r.p, np.linspace(0, 1, 11))
    print(r.groupby("bucket", observed=True).agg(n=("hit", "size"), predicted=("p", "mean"), realized=("hit", "mean")))


if __name__ == "__main__":
    main()
