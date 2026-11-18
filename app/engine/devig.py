"""
De-vigged implied-probability helpers: a convenience layer over `odds_math` (which keeps the actual
multiplicative / power / Shin solvers). Used by the market feed (fair open/close probabilities) and the
grader (closing line value measured against the fair, vig-free close).
"""
from __future__ import annotations

from collections.abc import Sequence

from app.engine.odds_math import (
    devig_multiplicative,
    devig_power,
    devig_shin,
    implied_prob,
    implied_to_american,
)

METHODS = {"multiplicative": devig_multiplicative, "power": devig_power, "shin": devig_shin}


def overround(prices: Sequence[int]) -> float:
    """Bookmaker margin: sum of raw implied probabilities minus 1 (0.045 = 4.5% hold)."""
    return sum(implied_prob(p) for p in prices) - 1.0


def fair_probs(prices: Sequence[int], method: str = "shin") -> list[float]:
    """Vig-free probabilities for a complete set of mutually exclusive outcomes (American prices)."""
    if len(prices) < 2:
        raise ValueError("need at least two outcomes to remove vig")
    try:
        return METHODS[method](list(prices))
    except KeyError:
        raise ValueError(f"unknown de-vig method {method!r}; use one of {sorted(METHODS)}") from None


def two_way_fair(price_a: int, price_b: int, method: str = "shin") -> tuple[float, float]:
    a, b = fair_probs([price_a, price_b], method)
    return a, b


def fair_from_probs(probs: Sequence[float], method: str = "multiplicative") -> list[float]:
    """Normalise quoted probabilities (prediction markets, ask-side quotes) so they sum to 1.
    `power` solves for k with sum(p_i ** k) = 1; Shin needs odds, so it falls back to `power` here."""
    p = [min(max(float(x), 1e-6), 1 - 1e-6) for x in probs]
    s = sum(p)
    if method == "multiplicative" or abs(s - 1.0) < 1e-9:
        return [x / s for x in p]
    lo, hi = 0.01, 10.0
    for _ in range(100):
        k = 0.5 * (lo + hi)
        tot = sum(x ** k for x in p)
        if abs(tot - 1.0) < 1e-9:
            break
        lo, hi = (k, hi) if tot > 1.0 else (lo, k)
    out = [x ** k for x in p]
    t = sum(out)
    return [x / t for x in out]


def fair_price(prob: float) -> int:
    """Fair American price for a probability (clamped away from 0/1)."""
    return implied_to_american(min(max(prob, 0.001), 0.999))


def edge_vs_fair(model_prob: float, price: int, fair_prob: float | None = None) -> dict:
    """Model edge against the raw breakeven and (when known) the vig-free probability."""
    be = implied_prob(price)
    out = {"breakeven": be, "edge_vs_breakeven": model_prob - be}
    if fair_prob is not None:
        out["fair"] = fair_prob
        out["edge_vs_fair"] = model_prob - fair_prob
    return out


def clv(placed_price: int, close_prob: float) -> float:
    """Closing line value in probability points: close - placed breakeven.
    Positive means the market later priced our side as MORE likely than the price we took."""
    return close_prob - implied_prob(placed_price)
