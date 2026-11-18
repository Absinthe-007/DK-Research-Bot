"""
kelly.py: Fractional Kelly bankroll allocation with per-bet and total slate exposure caps.
Supports both scanner.py (`BetCandidate`, `allocate`) and test_engine.py (`calculate_kelly_stake`).
"""
from __future__ import annotations

from dataclasses import dataclass

from app.engine.odds_math import profit_per_unit


@dataclass
class BetCandidate:
    id: str
    game_id: int
    win_prob: float
    price: int
    push_prob: float = 0.0


def kelly_fraction_single(win_prob: float, price: int, push_prob: float = 0.0) -> float:
    """Full Kelly fraction accounting for push probability: f* = (b*p_win - p_loss) / b."""
    b = profit_per_unit(price)
    loss_prob = max(0.0, 1.0 - win_prob - push_prob)
    if b <= 0:
        return 0.0
    f = (b * win_prob - loss_prob) / b
    return max(0.0, float(f))


def allocate(
    candidates: list[BetCandidate],
    bankroll: float,
    kelly_fraction: float = 0.25,
    max_stake_pct: float = 0.03,
    max_total_exposure_pct: float = 0.25,
) -> dict[str, dict]:
    """
    Sizes a slate of simultaneous bets using fractional Kelly, capped per bet (`max_stake_pct`)
    and scaled down proportionally if total slate action exceeds `max_total_exposure_pct`.
    """
    raw_alloc: dict[str, dict] = {}
    for c in candidates:
        full_k = kelly_fraction_single(c.win_prob, c.price, c.push_prob)
        frac_k = min(full_k * kelly_fraction, max_stake_pct)
        raw_alloc[c.id] = {
            "kelly_full": round(full_k, 5),
            "stake_pct": frac_k,
            "stake_amount": round(bankroll * frac_k, 2),
        }

    total_pct = sum(v["stake_pct"] for v in raw_alloc.values())
    if total_pct > max_total_exposure_pct and total_pct > 0:
        scale = max_total_exposure_pct / total_pct
        for v in raw_alloc.values():
            v["stake_pct"] = round(v["stake_pct"] * scale, 5)
            v["stake_amount"] = round(bankroll * v["stake_pct"], 2)
    else:
        for v in raw_alloc.values():
            v["stake_pct"] = round(v["stake_pct"], 5)

    return raw_alloc


def calculate_kelly_stake(
    true_prob: float,
    odds: int,
    bankroll: float,
    fraction: float = 0.25,
    max_bankroll_pct: float = 0.05,
) -> dict:
    """Standalone helper for unit tests and single-bet sizing."""
    b = profit_per_unit(odds)
    q = 1.0 - true_prob
    kelly_full = ((b * true_prob) - q) / b
    if kelly_full <= 0:
        return {
            "recommended_stake": 0.0,
            "bankroll_percentage": 0.0,
            "raw_kelly_pct": round(kelly_full * 100.0, 2),
            "reason": "Negative or zero edge; no bet recommended.",
        }
    adj = min(kelly_full * fraction, max_bankroll_pct)
    return {
        "recommended_stake": round(bankroll * adj, 2),
        "bankroll_percentage": round(adj * 100.0, 2),
        "raw_kelly_pct": round(kelly_full * 100.0, 2),
        "fraction_used": fraction,
    }