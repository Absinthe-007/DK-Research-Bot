"""
Expected value and edge screening.

Definitions (per 1 unit staked):
    b        = net decimal profit on a win  (= decimal_odds - 1)
    p_w, p_l, p_push = model probabilities of win / loss / push   (sum to 1)
    EV       = p_w * b - p_l * 1                     (push returns the stake: contributes 0)

    This is exactly the requested  EV = P(win)*Profit - P(lose)*Stake.

    edge     = P(win | action) - breakeven
             = p_w / (1 - p_push)  -  implied_prob(price)

`breakeven` is the VIGGED implied probability - i.e. the actual price you would pay - so a
positive edge means positive EV, and EV/unit = edge * (b + 1) * (1 - p_push) exactly.
The default flag threshold is edge >= 3.5 percentage points.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from app.engine.odds_math import implied_prob, profit_per_unit


@dataclass
class EdgeResult:
    side: str
    line: float | None
    price: int
    model_prob: float          # P(win) unconditional
    push_prob: float
    breakeven_prob: float
    fair_prob: float | None    # de-vigged market prob for this side
    edge: float
    ev_per_unit: float
    flagged: bool
    flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def expected_value(p_win: float, price: int, p_push: float = 0.0) -> float:
    b = profit_per_unit(price)
    p_loss = max(0.0, 1 - p_win - p_push)
    return p_win * b - p_loss


def edge(p_win: float, price: int, p_push: float = 0.0) -> float:
    action = 1 - p_push
    if action <= 0:
        return 0.0
    return p_win / action - implied_prob(price)


def screen(side: str, line: float | None, price: int, p_win: float, p_push: float = 0.0,
           fair_prob: float | None = None, threshold: float = 0.035,
           context_flags: list[str] | None = None, max_model_market_gap: float = 0.15) -> EdgeResult:
    e = edge(p_win, price, p_push)
    flags = list(context_flags or [])
    flagged = e >= threshold
    # Sanity guard: a model that disagrees with a liquid market by >15 pts is far more often
    # missing information (injury, role change) than finding free money.
    if fair_prob is not None and (p_win / max(1e-9, 1 - p_push)) - fair_prob > max_model_market_gap:
        flags.append("MODEL/MARKET GAP >15pts - verify news before acting")
    return EdgeResult(side, line, price, p_win, p_push, implied_prob(price), fair_prob, e,
                      expected_value(p_win, price, p_push), flagged, flags)
