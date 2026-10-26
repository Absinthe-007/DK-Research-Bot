"""
Dynamic baselines + contextual adjustment factors.

Everything here is a pure function so it can be unit-tested and back-tested in isolation.
The CONTEXT coefficients (weather, rest) are reasonable public-research priors, not fitted
values - treat them as starting points and calibrate with scripts/backtest.py on your data.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Recency weighting
# ---------------------------------------------------------------------------
def ewm_weights(n: int, halflife: float = 4.0) -> np.ndarray:
    """Weights for a series ordered OLDEST -> NEWEST. Newest game weight 1, a game `halflife`
    games ago weight 0.5. halflife=4 puts ~50% of total weight on the last ~4 games for a
    20-game sample - matching 'favor the last 3-5 games' while keeping season context."""
    ages = np.arange(n - 1, -1, -1, dtype=float)
    return 0.5 ** (ages / halflife)


def ewm_mean_std(values, halflife: float = 4.0) -> tuple[float, float]:
    """Exponentially weighted mean and (bias-corrected) weighted std."""
    x = np.asarray(values, float)
    if len(x) == 0:
        return float("nan"), float("nan")
    w = ewm_weights(len(x), halflife)
    mu = float(np.sum(w * x) / np.sum(w))
    if len(x) < 2:
        return mu, float("nan")
    v1, v2 = np.sum(w), np.sum(w**2)
    var = float(np.sum(w * (x - mu) ** 2) / (v1 - v2 / v1))   # reliability-weights correction
    return mu, var**0.5


def shrink(sample: float, n: float, prior: float, k: float) -> float:
    """Empirical-Bayes shrinkage: k = sample size at which sample and prior get equal weight."""
    return (n * sample + k * prior) / (n + k)


# ---------------------------------------------------------------------------
# Opponent / pace
# ---------------------------------------------------------------------------
def opponent_factor(opp_allowed_pg: float, league_allowed_pg: float, games: int, k: float = 8.0) -> float:
    """Defense-vs-position multiplier, shrunk toward 1.0.
    e.g. opponent allows 28 PRA/g to guards vs league 25 -> raw 1.12; with 6 games and k=8
    -> 1 + 0.12 * 6/14 = 1.051. Early-season defensive splits are mostly noise, hence k."""
    if not league_allowed_pg or not games:
        return 1.0
    raw = opp_allowed_pg / league_allowed_pg
    return shrink(raw, games, 1.0, k)


def nba_pace_factor(team_pace: float, opp_pace: float, league_pace: float) -> float:
    """Expected game possessions ~ team_pace * opp_pace / league_pace (log5-style).
    Relative to the player's own team pace (which his baseline already reflects) that is
    opp_pace / league_pace."""
    if not (team_pace and opp_pace and league_pace):
        return 1.0
    return opp_pace / league_pace


# ---------------------------------------------------------------------------
# NFL weather
# ---------------------------------------------------------------------------
PASS_MARKETS = {"player_pass_yds", "player_pass_completions", "player_pass_tds", "player_reception_yds",
                "player_receptions", "player_pass_attempts"}
RUSH_MARKETS = {"player_rush_yds", "player_rush_attempts"}


def weather_factor(market_key: str, wind_mph: float | None, precip_prob: float | None,
                   temp_f: float | None, roof: str = "open") -> tuple[float, list[str]]:
    """Multiplicative adjustment + human-readable flags.
    Priors (public research on NFL pbp, 2000-2023):
      * passing efficiency is ~flat below ~10 mph sustained wind, then degrades ~1-1.5%/mph;
        >20 mph games see ~15-20% fewer pass yards.
      * heavy precipitation: ~4-6% fewer pass yds, slight shift to the run.
      * extreme cold (<20F): small (~3%) passing decline, mostly via ball handling.
    """
    if roof == "dome":
        return 1.0, []
    f, flags = 1.0, []
    if wind_mph is not None and wind_mph > 10:
        flags.append(f"WIND {wind_mph:.0f} mph")
        if market_key in PASS_MARKETS:
            f *= max(0.80, 1 - 0.0125 * (wind_mph - 10))
        elif market_key in RUSH_MARKETS:
            f *= min(1.06, 1 + 0.004 * (wind_mph - 10))
    if precip_prob is not None and precip_prob >= 0.6:
        flags.append(f"PRECIP {precip_prob:.0%}")
        if market_key in PASS_MARKETS:
            f *= 0.95
        elif market_key in RUSH_MARKETS:
            f *= 1.02
    if temp_f is not None and temp_f < 20:
        flags.append(f"COLD {temp_f:.0f}F")
        if market_key in PASS_MARKETS:
            f *= 0.97
    if roof == "retractable" and flags:
        flags.append("RETRACTABLE ROOF - confirm open/closed before trusting weather adj")
    return f, flags


# ---------------------------------------------------------------------------
# NBA rest / minutes
# ---------------------------------------------------------------------------
def nba_minutes_factor(is_b2b: bool, is_3in4: bool, age: int | None = None) -> tuple[float, list[str]]:
    """Minutes multiplier. B2B second legs average ~2-4% fewer minutes for rotation players
    (and higher DNP-rest risk for stars, which is FLAGGED rather than baked into the mean)."""
    f, flags = 1.0, []
    if is_b2b:
        f *= 0.97
        flags.append("BACK-TO-BACK (2nd leg) - check load-management news")
        if age and age >= 32:
            f *= 0.98
    if is_3in4:
        f *= 0.99
        flags.append("3-IN-4 NIGHTS")
    return f, flags


def with_without_factor(rate_with: float, n_with: int, rate_without: float, n_without: int,
                        k: float = 6.0) -> float:
    """Usage-redistribution multiplier when a teammate is OUT, from the player's own per-minute
    production in games that teammate missed. Shrunk toward 1.0 because n_without is usually
    tiny (and those games are confounded by other absences)."""
    if not n_without or not rate_with:
        return 1.0
    return shrink(rate_without / rate_with, n_without, 1.0, k)
