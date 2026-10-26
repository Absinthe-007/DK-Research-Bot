"""
Odds conversions and de-vigging.

American odds A:
    A > 0  -> decimal = 1 + A/100      e.g. +150 -> 2.50
    A < 0  -> decimal = 1 + 100/|A|    e.g. -110 -> 1.909
Implied probability (INCLUDING vig) = 1 / decimal. This is the BREAKEVEN win rate for a bet
at that price - the number a model must beat.

De-vig: a two-way market's implied probs sum to >1 (the overround). Removing it gives the
market's "fair" probability - useful as a sanity anchor (big model-vs-fair gaps usually mean
the model is missing information, e.g. an injury).
  * multiplicative: p_i / sum(p)                 - simple, standard
  * power:          find k with sum(p_i^k) = 1   - shifts more vig onto longshots, which
                                                   matches the favourite-longshot bias better
"""
from __future__ import annotations

from collections.abc import Sequence

from scipy.optimize import brentq


def american_to_decimal(a: float) -> float:
    if a == 0 or -100 < a < 100:
        raise ValueError(f"invalid American odds: {a}")
    return 1 + (a / 100 if a > 0 else 100 / abs(a))


def decimal_to_american(d: float) -> int:
    if d <= 1:
        raise ValueError("decimal odds must be > 1")
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))


def implied_prob(a: float) -> float:
    """Breakeven probability at American price `a` (vig included)."""
    return 1 / american_to_decimal(a)


def profit_per_unit(a: float) -> float:
    """Net profit on a 1-unit winning stake (the Kelly 'b')."""
    return american_to_decimal(a) - 1


def prob_to_american(p: float) -> int:
    """Fair American price for probability p."""
    return decimal_to_american(1 / p)


def overround(prices: Sequence[float]) -> float:
    return sum(implied_prob(a) for a in prices) - 1


def devig_multiplicative(prices: Sequence[float]) -> list[float]:
    raw = [implied_prob(a) for a in prices]
    s = sum(raw)
    return [p / s for p in raw]


def devig_power(prices: Sequence[float]) -> list[float]:
    raw = [implied_prob(a) for a in prices]
    if abs(sum(raw) - 1) < 1e-9:
        return raw
    k = brentq(lambda k: sum(p**k for p in raw) - 1, 0.5, 5.0)
    return [p**k for p in raw]
