"""
Walk-forward CALIBRATION check for the distribution layer (NBA & NFL gamelogs).

For every player-game with >= MIN prior games, project using ONLY prior games (no leakage):
    NBA: EWMA(per-minute rate) x EWMA(minutes) + Law of Total Variance + League CV shrinkage
    NFL: EWMA(per-game stat) + League CV shrinkage
    -> fitted family -> P(Over) at a centered half-point synthetic line (floor(median) + 0.5).

Reports:
  * Brier score and log loss of P(Over) (both raw parametric and empirical-blended)
  * Reliability table: predicted-probability buckets vs realized hit rate
  * 80% interval coverage (10th to 90th integer-lattice percentile; target ~0.80)

    python -m scripts.backtest --market player_rebounds --halflife 4
    python -m scripts.backtest --market player_reception_yds --halflife 3
"""
from __future__ import annotations

import argparse
import asyncio
import math

import numpy as np
import pandas as pd
from sqlalchemy import text

from app.db.base import session_scope
from app.db.reference import MARKET_INDEX
from app.engine.baselines import ewm_mean_std, ewm_weights
from app.engine.distributions import (
    blend_with_history,
    empirical_hit_rate,
    fit,
    prob_over_under,
)
from app.engine.projector import shrink_sd

MIN_NBA = 8
MIN_NFL = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="player_points")
    ap.add_argument("--halflife", type=float, default=None, help="override EWMA halflife (default: 4.0 NBA, 3.0 NFL)")
    ap.add_argument("--family", default=None, help="override family (poisson/negbin/normal/gamma)")
    a = ap.parse_args()

    if a.market not in MARKET_INDEX:
        raise SystemExit(f"Unknown market {a.market!r}. Choose from: {', '.join(sorted(MARKET_INDEX))}")

    _, sport, _, expr, default_family = MARKET_INDEX[a.market]
    family = a.family or default_family
    hl = a.halflife if a.halflife is not None else (4.0 if sport == "NBA" else 3.0)
    min_games = MIN_NBA if sport == "NBA" else MIN_NFL

    async def load() -> pd.DataFrame:
        async with session_scope() as s:
            if sport == "NBA":
                res = await s.execute(
                    text(
                        "SELECT player_id, game_date, minutes, pts, reb, ast, fg3m, stl, blk, tov "
                        "FROM nba_player_gamelogs WHERE minutes >= 10 ORDER BY player_id, game_date"
                    )
                )
            else:
                res = await s.execute(
                    text(
                        "SELECT player_id, game_date, completions, pass_attempts, pass_yds, pass_tds, "
                        "interceptions, carries, rush_yds, rush_tds, targets, receptions, rec_yds, rec_tds "
                        "FROM nfl_player_games ORDER BY player_id, game_date"
                    )
                )
            return pd.DataFrame(res.mappings().all())

    df = asyncio.run(load())
    if df.empty:
        print(f"No gamelogs found for sport={sport}.")
        return

    df["y"] = df.eval(expr).astype(float)

    # Compute league-typical CV across players with sufficient sample
    p_stats = df.groupby("player_id")["y"].agg(["mean", "std", "count"])
    valid_cv = p_stats[(p_stats["count"] >= min_games) & (p_stats["mean"] >= 1.0)]
    league_cv = float((valid_cv["std"] / valid_cv["mean"]).mean()) if len(valid_cv) else 0.5

    recs = []
    lookback = 25 if sport == "NBA" else 17

    for _, g in df.groupby("player_id"):
        y = g["y"].to_numpy(float)
        mins = g["minutes"].to_numpy(float) if sport == "NBA" else None

        for i in range(min_games, len(y)):
            start_idx = max(0, i - lookback)
            prior_y = y[start_idx:i]

            if sport == "NBA" and mins is not None:
                prior_m = mins[start_idx:i]
                rate_mu, sd_rate = ewm_mean_std(prior_y / prior_m, hl)
                min_mu, sd_min_raw = ewm_mean_std(prior_m, hl)
                mu = max(0.01, rate_mu * min_mu)
                sd_min = shrink_sd(0.0 if np.isnan(sd_min_raw) else sd_min_raw, len(prior_m), min_mu, 0.18, k=10.0)
                sd_rate_clean = 0.0 if np.isnan(sd_rate) else sd_rate
                sd_player = ((min_mu**2) * (sd_rate_clean**2) + (rate_mu**2) * (sd_min**2)) ** 0.5
                sd = max(shrink_sd(sd_player, len(prior_y), mu, league_cv), 0.6 * mu**0.5, 0.3)
            else:
                mu, sd_raw = ewm_mean_std(prior_y, hl)
                mu = max(0.01, mu)
                sd = max(shrink_sd(0.0 if np.isnan(sd_raw) else sd_raw, len(prior_y), mu, league_cv), 0.5 * mu**0.5, 0.25)

            # Symmetric half-point synthetic line centered near recent median/mean without a downward -0.5 bias
            med10 = float(np.median(prior_y[-10:]))
            line = math.floor(med10) + 0.5 if (med10 % 1.0) != 0.5 else med10
            if line < 0.5:
                line = 0.5

            try:
                d = fit(family, max(mu, 0.1), sd)
            except ValueError:
                continue

            p_over, _, _ = prob_over_under(d, line)
            w = ewm_weights(len(prior_y), hl)
            hr, n_eff = empirical_hit_rate(prior_y, line, w)
            p_blend = blend_with_history(p_over, hr, n_eff)

            f_dist = d._frozen()
            lo = math.floor(float(f_dist.ppf(0.10)))
            hi = math.ceil(float(f_dist.ppf(0.90)))
            recs.append((p_over, p_blend, float(y[i] > line), lo <= y[i] <= hi))

    r = pd.DataFrame(recs, columns=["p_raw", "p_blend", "hit", "in80"])
    if r.empty:
        print("Not enough player-game history to evaluate.")
        return

    p_raw = r["p_raw"].clip(1e-4, 1 - 1e-4)
    p_bl = r["p_blend"].clip(1e-4, 1 - 1e-4)
    brier_raw = float(np.mean((p_raw - r.hit) ** 2))
    brier_bl = float(np.mean((p_bl - r.hit) ** 2))
    ll_raw = float(-np.mean(r.hit * np.log(p_raw) + (1 - r.hit) * np.log(1 - p_raw)))
    ll_bl = float(-np.mean(r.hit * np.log(p_bl) + (1 - r.hit) * np.log(1 - p_bl)))

    print(f"market={a.market} ({sport}) family={family} halflife={hl} league_cv={league_cv:.3f} n={len(r)}")
    print(f"Raw Parametric : Brier={brier_raw:.4f}  LogLoss={ll_raw:.4f}  80%-coverage={r.in80.mean():.3f}")
    print(f"Hist-Blended   : Brier={brier_bl:.4f}  LogLoss={ll_bl:.4f}")
    r["bucket"] = pd.cut(r.p_blend, np.linspace(0, 1, 11))
    print(
        r.groupby("bucket", observed=True).agg(
            n=("hit", "size"),
            pred_raw=("p_raw", "mean"),
            pred_blend=("p_blend", "mean"),
            realized=("hit", "mean"),
        )
    )


if __name__ == "__main__":
    main()