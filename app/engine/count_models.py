"""
Count-stat probabilities straight from `scipy.stats` (nbinom / poisson), with an explicit
"did he play?" mixture. Complements `app.engine.distributions` (which discretises normal/gamma too and is
what the live scanner uses); this module is the small, dependency-light entry point for the upgrades.

Lines follow sportsbook rules: a half-point line has no push; a whole-number line pushes on equality.
If the player is a DNP the prop is voided (stake returned), which is reported as `void`.
"""
from __future__ import annotations

import math

from scipy import stats

OVERDISPERSION_THRESHOLD = 1.05    # var/mean above this -> negative binomial, otherwise Poisson


def choose_family(mean: float, var: float | None) -> str:
    if var is None or mean <= 0:
        return "poisson"
    return "nbinom" if var / mean > OVERDISPERSION_THRESHOLD else "poisson"


def frozen(mean: float, var: float | None = None, family: str = "auto"):
    """scipy frozen distribution with the requested mean (and variance for nbinom)."""
    mean = max(float(mean), 1e-6)
    fam = choose_family(mean, var) if family == "auto" else family
    if fam == "poisson":
        return stats.poisson(mean)
    if fam in ("nbinom", "negbin"):
        v = max(float(var if var is not None else mean * 1.25), mean * 1.0001)
        n = mean * mean / (v - mean)             # size / dispersion
        return stats.nbinom(n, n / (n + mean))   # scipy parameterisation: success prob p = n / (n + mean)
    raise ValueError(f"unsupported family {family!r}")


def over_under(
    mean: float, line: float, var: float | None = None, family: str = "auto", p_active: float = 1.0,
) -> dict[str, float]:
    """Unconditional P(over / under / push / void) for a count stat. Probabilities sum to 1."""
    d = frozen(mean, var, family)
    if float(line).is_integer():
        k = int(line)
        under = float(d.cdf(k - 1)) if k >= 1 else 0.0
        push = float(d.pmf(k))
        over = max(0.0, 1.0 - under - push)
    else:
        under = float(d.cdf(math.floor(line)))
        push, over = 0.0, max(0.0, 1.0 - under)
    pa = min(1.0, max(0.0, float(p_active)))
    return {"over": over * pa, "under": under * pa, "push": push * pa, "void": 1.0 - pa}


def fair_probability(result: dict[str, float], side: str) -> float:
    """Win probability for a side conditional on the bet being live (pushes and voids refunded)."""
    live = result["over"] + result["under"]
    return result[side.lower()] / live if live > 0 else 0.5


def prob_table(mean: float, lines: list[float], var: float | None = None, family: str = "auto") -> list[dict]:
    """P(over) for a ladder of lines, e.g. milestone markets (N+ means over N - 0.5)."""
    return [{"line": ln, **over_under(mean, ln, var, family)} for ln in lines]


def as_prop_distribution(mean: float, var: float | None = None):
    """Bridge to the scanner's `PropDistribution` so a result here is interchangeable with the original path."""
    from app.engine.distributions import fit

    fam = choose_family(mean, var)
    return fit("poisson" if fam == "poisson" else "negbin", mean, None if var is None else math.sqrt(var))
