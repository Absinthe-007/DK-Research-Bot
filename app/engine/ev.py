"""
ev.py: Expected Value (EV), mathematical edge screening, and dataclass results.
Supports both scanner.py (`screen`) and test_engine.py (`calculate_ev`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.engine.odds_math import implied_prob, profit_per_unit


@dataclass
class ScreenResult:
    side: str
    line: float | None
    price: int
    model_prob: float
    push_prob: float
    breakeven_prob: float
    fair_prob: float | None
    edge: float
    ev_per_unit: float
    flagged: bool
    flags: list[str] = field(default_factory=list)


def screen(
    side: str,
    line: float | None,
    price: int,
    model_prob: float,
    push_prob: float = 0.0,
    fair_prob: float | None = None,
    edge_threshold: float = 0.08,
    flags: list[str] | None = None,
) -> ScreenResult:
    """
    Screens a single prop side for +EV and probability edge.
    - `model_prob` is unconditional P(win).
    - `push_prob` is P(push).
    - Actioned (non-push) win probability is model_prob / (1 - push_prob).
    """
    b = profit_per_unit(price)
    be_cond = implied_prob(price)
    act = max(1e-9, 1.0 - push_prob)
    be_uncond = be_cond * act

    p_loss = max(0.0, 1.0 - model_prob - push_prob)
    ev = (model_prob * b) - p_loss

    ref_prob = fair_prob if fair_prob is not None else be_uncond
    edge = model_prob - ref_prob
    flagged = bool(edge >= edge_threshold and ev > 0.0)

    return ScreenResult(
        side=side,
        line=line,
        price=int(price),
        model_prob=float(model_prob),
        push_prob=float(push_prob),
        breakeven_prob=float(be_uncond),
        fair_prob=None if fair_prob is None else float(fair_prob),
        edge=float(edge),
        ev_per_unit=float(ev),
        flagged=flagged,
        flags=list(flags or []),
    )


def calculate_ev(true_prob: float, odds: int) -> dict:
    """Standalone helper for unit tests and direct EV queries."""
    b = profit_per_unit(odds)
    decimal_odds = b + 1.0
    ev_pct = (true_prob * b) - (1.0 - true_prob)
    mkt_imp = implied_prob(odds)
    edge = true_prob - mkt_imp
    return {
        "true_prob": round(true_prob, 4),
        "market_implied_prob": round(mkt_imp, 4),
        "decimal_odds": round(decimal_odds, 4),
        "ev_percentage": round(ev_pct * 100.0, 2),
        "edge_percentage": round(edge * 100.0, 2),
        "is_positive_ev": ev_pct > 0.0,
    }