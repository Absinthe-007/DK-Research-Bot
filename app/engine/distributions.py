"""
Outcome distributions for player props.

Every family is DISCRETIZED onto integers before computing Over/Under, because box-score
stats are integers (yards included). That makes half-point and whole-number lines exact:

    line = 24.5 -> P(Over) = P(X >= 25), P(Under) = P(X <= 24), P(Push) = 0
    line = 25   -> P(Over) = P(X >= 26), P(Under) = P(X <= 24), P(Push) = P(X = 25)

Continuous families (normal, gamma) are discretized with a continuity correction:
    P(X = k) = F(k + 0.5) - F(k - 0.5)

Family choice (statistical assumptions):
  * Poisson(lambda): events arrive independently at a constant rate -> var = mean. Good for rare
    counts (blocks, steals, TDs, INTs).
  * Negative binomial: Poisson whose rate itself varies game to game (minutes, game script)
    -> var > mean. Most "count" props (rebounds, assists, receptions) are over-dispersed;
    if the sample shows var <= mean we fall back to Poisson rather than fit an invalid NB.
  * Normal: sums of many small contributions (points, PRA, pass yds) - CLT.
  * Gamma: non-negative, right-skewed (a WR's 120-yd game is more likely than a -20-yd one).
    Rushing yards can be slightly negative; the gamma's 0 floor is an accepted approximation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class PropDistribution:
    family: str
    params: dict
    support_max: int = field(default=0)

    def _frozen(self):
        p = self.params
        if self.family == "poisson":
            return stats.poisson(p["mu"])
        if self.family == "negbin":
            return stats.nbinom(p["n"], p["p"])
        if self.family == "normal":
            return stats.norm(p["mu"], p["sigma"])
        if self.family == "gamma":
            return stats.gamma(p["shape"], scale=p["scale"])
        raise ValueError(self.family)

    @property
    def discrete(self) -> bool:
        return self.family in ("poisson", "negbin")

    def cdf_int(self, k: int) -> float:
        """P(X <= k) on the integer lattice."""
        d = self._frozen()
        if k < 0 and self.family != "normal":
            return 0.0
        return float(d.cdf(k) if self.discrete else d.cdf(k + 0.5))

    def pmf_int(self, k: int) -> float:
        d = self._frozen()
        return float(d.pmf(k) if self.discrete else d.cdf(k + 0.5) - d.cdf(k - 0.5))

    @property
    def mean(self) -> float:
        return float(self._frozen().mean())

    @property
    def std(self) -> float:
        return float(self._frozen().std())


def fit(family: str, mean: float, sd: float | None = None) -> PropDistribution:
    """Method-of-moments fit from a projected mean and standard deviation."""
    mean = max(mean, 1e-6)
    if family == "poisson" or (family == "negbin" and (sd is None or sd**2 <= mean * 1.02)):
        return PropDistribution("poisson", {"mu": mean})
    if family == "negbin":
        var = sd**2
        n = mean**2 / (var - mean)      # "size" / dispersion
        p = n / (n + mean)
        return PropDistribution("negbin", {"n": n, "p": p})
    if sd is None or sd <= 0:
        raise ValueError(f"{family} needs a positive sd")
    if family == "normal":
        return PropDistribution("normal", {"mu": mean, "sigma": sd})
    if family == "gamma":
        shape = (mean / sd) ** 2
        return PropDistribution("gamma", {"shape": shape, "scale": sd**2 / mean})
    raise ValueError(family)


def prob_over_under(dist: PropDistribution, line: float | None) -> tuple[float, float, float]:
    """Returns (P_over, P_under, P_push). line=None means a yes/no 'at least one' market."""
    if line is None:            # e.g. anytime TD == Over 0.5
        line = 0.5
    if float(line).is_integer():
        k = int(line)
        under = dist.cdf_int(k - 1)
        push = dist.pmf_int(k)
        return max(0.0, 1 - under - push), under, push
    k = math.floor(line)
    under = dist.cdf_int(k)
    return 1 - under, under, 0.0


def empirical_hit_rate(values: np.ndarray, line: float | None, weights: np.ndarray | None = None) -> tuple[float, float]:
    """Weighted historical frequency of Over (excluding pushes). Returns (rate, effective_n).
    effective_n = (sum w)^2 / sum w^2  (Kish) so recency weighting doesn't overstate certainty."""
    line = 0.5 if line is None else line
    values = np.asarray(values, float)
    w = np.ones_like(values) if weights is None else np.asarray(weights, float)
    mask = values != line
    v, w = values[mask], w[mask]
    if len(v) == 0:
        return 0.5, 0.0
    rate = float(np.sum(w * (v > line)) / np.sum(w))
    n_eff = float(np.sum(w) ** 2 / np.sum(w**2))
    return rate, n_eff


def blend_with_history(model_p: float, hist_p: float, n_eff: float, prior_strength: float = 20.0) -> float:
    """Beta-binomial shrinkage: treat the model probability as a prior worth `prior_strength`
    pseudo-games and update with the observed hit rate. Small samples barely move the model;
    a long, consistent history pulls it toward reality."""
    return (model_p * prior_strength + hist_p * n_eff) / (prior_strength + n_eff)


# ---------------------------------------------------------------------------
# Market anchoring
# ---------------------------------------------------------------------------
def refit_at_mean(dist: PropDistribution, new_mean: float) -> PropDistribution:
    """Same SHAPE, different centre: keep the coefficient of variation (normal/gamma) or the
    variance-to-mean ratio (negbin); Poisson is fully determined by its mean."""
    new_mean = max(new_mean, 1e-3)
    m, sd = dist.mean, dist.std
    if dist.family == "poisson":
        return fit("poisson", new_mean)
    if dist.family == "negbin":
        ratio = sd**2 / m
        return fit("negbin", new_mean, (ratio * new_mean) ** 0.5)
    return fit(dist.family, new_mean, sd / m * new_mean)


def market_implied_mean(dist: PropDistribution, line: float, fair_over: float) -> float | None:
    """Solve for the mean at which this distribution shape gives P(Over line | action) = fair_over.
    That is the market's projection expressed in the model's own distribution family."""
    from scipy.optimize import brentq

    def f(mu):
        o, u, p = prob_over_under(refit_at_mean(dist, mu), line)
        return o / max(1e-9, o + u) - fair_over
    lo, hi = 1e-3, max(5.0, line * 4 + 10)
    try:
        if f(lo) * f(hi) > 0:
            return None
        return brentq(f, lo, hi, xtol=1e-3)
    except (ValueError, RuntimeError):
        return None
