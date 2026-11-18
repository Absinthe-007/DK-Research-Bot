"""
odds_math.py: Precision odds conversion and vig-removal algorithms (Multiplicative, Power, and Shin).
Supports both the live scanner/record pipeline and standalone test suite.
"""
from __future__ import annotations

import math


def implied_prob(american_odds: int) -> float:
    """Converts American odds to raw implied (breakeven) probability."""
    if american_odds == 0:
        raise ValueError("American odds cannot be 0")
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    return abs(american_odds) / (abs(american_odds) + 100.0)


# Alias for test_engine.py compatibility
american_to_implied = implied_prob


def implied_to_american(prob: float) -> int:
    """Converts a probability (0 < p < 1) back to American odds."""
    if prob <= 0.0 or prob >= 1.0:
        raise ValueError("Probability must be strictly between 0 and 1.")
    if prob >= 0.5:
        return int(round(-100.0 * prob / (1.0 - prob)))
    return int(round(100.0 * (1.0 - prob) / prob))


def american_to_decimal(american_odds: int) -> float:
    """Converts American odds to decimal payout odds (including stake)."""
    if american_odds > 0:
        return 1.0 + (american_odds / 100.0)
    return 1.0 + (100.0 / abs(american_odds))


def profit_per_unit(american_odds: int) -> float:
    """Net profit won on a 1-unit stake if the bet wins (b in Kelly formula)."""
    if american_odds > 0:
        return american_odds / 100.0
    return 100.0 / abs(american_odds)


def devig_multiplicative(odds_list: list[int]) -> list[float]:
    """Removes vig proportionally across all outcomes."""
    raw = [implied_prob(o) for o in odds_list]
    tot = sum(raw)
    if tot <= 0:
        return [1.0 / len(odds_list)] * len(odds_list)
    return [p / tot for p in raw]


remove_vig_multiplicative = devig_multiplicative


def devig_power(odds_list: list[int], tol: float = 1e-7, max_iter: int = 100) -> list[float]:
    """Removes vig using the Power Method: solves for exponent k such that sum(q_i^k) = 1."""
    raw = [implied_prob(o) for o in odds_list]
    if abs(sum(raw) - 1.0) < tol:
        return raw
    low, high = 0.01, 10.0
    for _ in range(max_iter):
        k = 0.5 * (low + high)
        val = sum(p ** k for p in raw)
        if abs(val - 1.0) < tol:
            break
        if val > 1.0:
            low = k
        else:
            high = k
    res = [p ** k for p in raw]
    s = sum(res)
    return [p / s for p in res]


remove_vig_power = devig_power


def devig_shin(odds_list: list[int], tol: float = 1e-7, max_iter: int = 100) -> list[float]:
    """
    Removes vig using Shin's (1991, 1993) insider-trading model.
    For a 2-way market, solves analytically in closed form; for n >= 3, bisects on z in [0, 1).
    """
    raw = [implied_prob(o) for o in odds_list]
    s = sum(raw)
    if s <= 1.0 + tol:
        return [p / s for p in raw]

    # Exact closed form for 2-outcome markets (Over/Under, moneyline). With d = q1 - q2, Shin's insider
    # proportion is z = (s - 1)(d^2 - s) / (s (d^2 - 1)); plugging z into the general Shin probability
    # formula gives the fair probabilities. (An earlier shortcut here was only right near 50/50.)
    if len(raw) == 2:
        d = raw[0] - raw[1]
        if abs(d * d - 1.0) < 1e-12:
            return [raw[0] / s, raw[1] / s]
        z = (s - 1.0) * (d * d - s) / (s * (d * d - 1.0))
        z = min(max(z, 0.0), 0.99)
        probs = [(math.sqrt(z * z + 4.0 * (1.0 - z) * q * q / s) - z) / (2.0 * (1.0 - z)) for q in raw]
        tot = sum(probs)
        return [p / tot for p in probs]

    def shin_probs(z: float) -> list[float]:
        return [
            (math.sqrt(z**2 + 4.0 * (1.0 - z) * (q**2) / s) - z) / (2.0 * (1.0 - z))
            for q in raw
        ]

    low, high = 0.0, 0.99
    for _ in range(max_iter):
        z = 0.5 * (low + high)
        probs = shin_probs(z)
        tot = sum(probs)
        if abs(tot - 1.0) < tol:
            return [p / tot for p in probs]
        if tot > 1.0:
            low = z
        else:
            high = z
    probs = shin_probs(0.5 * (low + high))
    tot = sum(probs)
    return [p / tot for p in probs]


remove_vig_shin = devig_shin