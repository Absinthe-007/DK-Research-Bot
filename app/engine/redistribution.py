"""
Dynamic injury redistribution of minutes (NBA) or usage shares (NFL targets / carries).

Pure functions, no I/O. Nothing here changes the original projections on its own; callers opt in by
passing a projection's baseline through `redistribute` and using the adjusted number.

Model: every player has a baseline share assuming a fully healthy roster. A player with play probability
p (from injury status) keeps `base * p` in expectation and vacates `base * (1 - p)`. The vacated amount is
handed to teammates in proportion to their weight, boosted for teammates in the same group (position /
role) as the absent player, then capped (e.g. 40 min, or a max target share) with excess re-spread over
teammates who still have room.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

# Probability the player suits up, by normalised report status. Conservative and easy to override.
STATUS_PLAY_PROB: dict[str, float] = {
    "out": 0.0, "ir": 0.0, "pup": 0.0, "suspended": 0.0, "inactive": 0.0, "doubtful": 0.10,
    "questionable": 0.50, "day-to-day": 0.60, "gtd": 0.50, "probable": 0.90, "active": 1.0, "healthy": 1.0,
}


def play_probability(status: str | None, overrides: Mapping[str, float] | None = None, default: float = 1.0) -> float:
    key = (status or "").strip().lower()
    table = {**STATUS_PLAY_PROB, **{k.lower(): v for k, v in (overrides or {}).items()}}
    return float(min(1.0, max(0.0, table.get(key, default))))


@dataclass
class Redistribution:
    values: dict[str, float]
    vacated: float                       # expected share freed by absences
    unallocated: float = 0.0             # share that could not be placed because every teammate hit its cap
    boosts: dict[str, float] = field(default_factory=dict)   # per-player change vs baseline


def redistribute(
    base: Mapping[str, float],
    play_prob: Mapping[str, float],
    *,
    weights: Mapping[str, float] | None = None,
    groups: Mapping[str, str] | None = None,
    same_group_boost: float = 2.0,
    caps: Mapping[str, float] | float | None = None,
) -> Redistribution:
    """`base` and the returned `values` are in the same units (minutes, or a share of team volume).
    `play_prob` defaults to 1.0 for players it doesn't mention."""
    names = list(base)
    p = {n: min(1.0, max(0.0, float(play_prob.get(n, 1.0)))) for n in names}
    w0 = {n: max(0.0, float((weights or {}).get(n, base[n]))) for n in names}
    cap = {n: (float(caps) if isinstance(caps, (int, float)) else float((caps or {}).get(n, float("inf")))) for n in names}

    value = {n: float(base[n]) * p[n] for n in names}               # absent players keep only their expected part
    vacated_by = {n: float(base[n]) * (1.0 - p[n]) for n in names}
    total_vacated = sum(vacated_by.values())
    if total_vacated <= 0:
        return Redistribution(values=value, vacated=0.0)

    received = {n: 0.0 for n in names}
    for absent, amount in vacated_by.items():
        if amount <= 0:
            continue
        w = {}
        for j in names:
            if j == absent or p[j] <= 0:
                continue
            boost = same_group_boost if groups and groups.get(j) is not None and groups.get(j) == groups.get(absent) else 1.0
            w[j] = w0[j] * p[j] * boost
        tw = sum(w.values())
        if tw <= 0:
            continue
        for j, wj in w.items():
            received[j] += amount * wj / tw

    # apply caps; re-spread what a capped player can't take over teammates that still have room
    final = {n: value[n] + received[n] for n in names}
    pool = 0.0
    for _ in range(12):
        over = {n: final[n] - cap[n] for n in names if final[n] > cap[n] + 1e-12}
        for n, ex in over.items():
            final[n] = cap[n]
            pool += ex
        if pool <= 1e-12:
            break
        room = {n: cap[n] - final[n] for n in names if p[n] > 0 and cap[n] - final[n] > 1e-12}
        tw = sum(w0[n] * p[n] for n in room)
        if not room or tw <= 0:
            break
        spread = pool
        pool = 0.0
        for n in room:
            final[n] += spread * w0[n] * p[n] / tw
    # a final pass may leave small overages after the last spread; clip and count them as unallocated
    for n in names:
        if final[n] > cap[n]:
            pool += final[n] - cap[n]
            final[n] = cap[n]
    boosts = {n: final[n] - float(base[n]) for n in names}
    return Redistribution(values=final, vacated=total_vacated, unallocated=max(0.0, pool), boosts=boosts)


def redistribute_minutes(
    minutes: Mapping[str, float],
    status: Mapping[str, str | None],
    *,
    cap: float = 40.0,
    positions: Mapping[str, str] | None = None,
    overrides: Mapping[str, float] | None = None,
) -> Redistribution:
    """NBA: baseline minutes -> expected minutes given injury statuses (`status` is name -> report status)."""
    pp = {n: play_probability(status.get(n), overrides) for n in minutes}
    return redistribute(minutes, pp, groups=positions, caps=cap)


def redistribute_usage(
    shares: Mapping[str, float],
    status: Mapping[str, str | None],
    *,
    cap: float = 0.45,
    positions: Mapping[str, str] | None = None,
    overrides: Mapping[str, float] | None = None,
) -> Redistribution:
    """NFL: baseline target (or carry) share -> expected share given injury statuses."""
    pp = {n: play_probability(status.get(n), overrides) for n in shares}
    return redistribute(shares, pp, groups=positions, caps=cap)


def usage_multiplier(base: float, adjusted: float) -> float:
    """Ratio to scale a rate-based projection (e.g. points ~ minutes) by the redistribution."""
    return adjusted / base if base > 0 else 1.0
