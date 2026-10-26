"""
Fractional Kelly staking.

Full Kelly for a binary bet:   f* = (b*p - q) / b       (p = win prob given action, q = 1-p)
It maximizes long-run log growth IF p is known exactly. Our p is an estimate, and over-betting
an over-estimated edge is far more damaging than under-betting (growth falls off a cliff past
2x Kelly). Hence:
  * fractional Kelly (default 1/4) - roughly halves volatility vs 1/2-Kelly for modest growth cost
  * per-bet cap (default 2% of bankroll)
  * total-exposure cap across a slate (default 10%) with proportional scale-down
  * correlation haircut: multiple bets on the SAME game share outcome drivers (pace, script,
    weather), so their stakes are divided by sqrt(#bets in that game).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from app.engine.odds_math import profit_per_unit


def kelly_full(p_win: float, price: int, p_push: float = 0.0) -> float:
    action = 1 - p_push
    if action <= 0:
        return 0.0
    p = p_win / action
    b = profit_per_unit(price)
    return max(0.0, (b * p - (1 - p)) / b)


@dataclass
class BetCandidate:
    key: str            # unique id (e.g. edge id)
    game_id: int
    p_win: float
    price: int
    p_push: float = 0.0


def allocate(bets: list[BetCandidate], bankroll: float, fraction: float = 0.25,
             max_stake_pct: float = 0.02, max_total_pct: float = 0.10) -> dict[str, dict]:
    per_game = Counter(b.game_id for b in bets)
    out: dict[str, dict] = {}
    for b in bets:
        fk = kelly_full(b.p_win, b.price, b.p_push)
        pct = min(fk * fraction / math.sqrt(per_game[b.game_id]), max_stake_pct)
        out[b.key] = {"kelly_full": fk, "stake_pct": pct}
    total = sum(v["stake_pct"] for v in out.values())
    scale = min(1.0, max_total_pct / total) if total > 0 else 1.0
    for v in out.values():
        v["stake_pct"] *= scale
        v["stake"] = round(v["stake_pct"] * bankroll, 2)
    return out
