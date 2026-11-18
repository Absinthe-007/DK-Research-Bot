"""
Outcome distributions for player props.

Every family is DISCRETIZED onto non-negative integers before computing Over/Under,
because box-score stats are integers (yards included). That makes half-point and
whole-number lines exact:

    line = 24.5 -> P(Over) = P(X >= 25), P(Under) = P(X <= 24), P(Push) = 0
    line = 25   -> P(Over) = P(X >= 26), P(Under) = P(X <= 24), P(Push) = P(X = 25)

Continuous families (normal, gamma) are discretized with a continuity correction
and zero-floored at k = 0 so P(X < 0) = 0:
    P(X = 0) = F(0.5)
    P(X = k) = F(k + 0.5) - F(k - 0.5)   for k >= 1
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
        """P(X <= k) on the non-negative integer lattice (k >= 0)."""
        if k < 0:
            return 0.0
        d = self._frozen()
        val = float(d.cdf(k) if self.discrete else d.cdf(k + 0.5))
        return min(1.0, max(0.0, val))

    def pmf_int(self, k: int) -> float:
        """P(X = k) on the non-negative integer lattice."""
        if k < 0:
            return 0.0
        d = self._frozen()
        if self.discrete:
            return float(d.pmf(k))
        if k == 0:
            return float(d.cdf(0.5))
        return max(0.0, float(d.cdf(k + 0.5) - d.cdf(k - 0.5)))

    @property
    def mean(self) -> float:
        p = self.params
        if self.family == "poisson":
            return float(p["mu"])
        if self.family == "negbin":
            return float(p["n"] * (1.0 - p["p"]) / p["p"])
        if self.family == "normal":
            return float(p["mu"])
        if self.family == "gamma":
            return float(p["shape"] * p["scale"])
        return float(self._frozen().mean())

    @property
    def std(self) -> float:
        p = self.params
        if self.family == "poisson":
            return float(math.sqrt(max(0.0, p["mu"])))
        if self.family == "negbin":
            return float(math.sqrt(max(0.0, p["n"] * (1.0 - p["p"]))) / p["p"])
        if self.family == "normal":
            return float(p["sigma"])
        if self.family == "gamma":
            return float(math.sqrt(max(0.0, p["shape"])) * p["scale"])
        return float(self._frozen().std())


def fit(family: str, mean: float, sd: float | None = None) -> PropDistribution:
    """Method-of-moments fit from a projected mean and standard deviation."""
    mean = max(float(mean), 1e-6)
    if family == "poisson" or (family == "negbin" and (sd is None or sd**2 <= mean * 1.02)):
        return PropDistribution("poisson", {"mu": mean})
    if family == "negbin":
        var = float(sd)**2
        n = mean**2 / (var - mean)      # dispersion ("size") parameter
        p = n / (n + mean)
        return PropDistribution("negbin", {"n": n, "p": p})
    if sd is None or sd <= 0:
        raise ValueError(f"{family} needs a positive sd")
    sd = float(sd)
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
        over = max(0.0, 1.0 - under - push)
        return over, under, push
    k = math.floor(line)
    under = dist.cdf_int(k)
    return max(0.0, 1.0 - under), under, 0.0


def empirical_hit_rate(
    values: np.ndarray, line: float | None, weights: np.ndarray | None = None
) -> tuple[float, float]:
    """Weighted historical frequency of Over (excluding pushes). Returns (rate, effective_n).
    effective_n = (sum w)^2 / sum w^2  (Kish) so recency weighting doesn't overstate certainty."""
    line = 0.5 if line is None else line
    values = np.asarray(values, float)
    w = np.ones_like(values) if weights is None else np.asarray(weights, float)
    mask = values != line
    v, w = values[mask], w[mask]
    if len(v) == 0 or np.sum(w) <= 0:
        return 0.5, 0.0
    rate = float(np.sum(w * (v > line)) / np.sum(w))
    n_eff = float(np.sum(w) ** 2 / np.sum(w**2))
    return rate, n_eff


def blend_with_history(
    model_p: float, hist_p: float, n_eff: float, prior_strength: float = 20.0
) -> float:
    """Beta-binomial shrinkage: treat the model probability as a prior worth `prior_strength`
    pseudo-games and update with the observed hit rate."""
    return (model_p * prior_strength + hist_p * n_eff) / (prior_strength + n_eff)


# ---------------------------------------------------------------------------
# Market anchoring
# ---------------------------------------------------------------------------
def refit_at_mean(dist: PropDistribution, new_mean: float) -> PropDistribution:
    """Same SHAPE, different centre: keep the coefficient of variation (normal/gamma)
    or the dispersion parameter `n` / variance-to-mean ratio (negbin); Poisson is fully
    determined by its mean."""
    new_mean = max(float(new_mean), 1e-3)
    m, sd = dist.mean, dist.std
    if dist.family == "poisson":
        return fit("poisson", new_mean)
    if dist.family == "negbin":
        # Preserve the overdispersion ratio above 1.02 so refitting at a smaller mean
        # stays smoothly in the NegBin family during root-finding.
        ratio = max(1.05, sd**2 / max(m, 1e-6))
        return fit("negbin", new_mean, (ratio * new_mean) ** 0.5)
    return fit(dist.family, new_mean, (sd / max(m, 1e-6)) * new_mean)


def market_implied_mean(dist: PropDistribution, line: float, fair_over: float) -> float | None:
    """Solve for the mean at which this distribution shape gives P(Over line | action) = fair_over.
    Note: `fair_over` is the conditional non-push probability P(Over | Over or Under)."""
    from scipy.optimize import brentq

    if not (0.01 <= fair_over <= 0.99):
        return None

    def f(mu):
        o, u, _p = prob_over_under(refit_at_mean(dist, mu), line)
        tot = o + u
        if tot <= 1e-9:
            return -fair_over if mu < line else 1.0 - fair_over
        return (o / tot) - fair_over

    lo, hi = 1e-3, max(5.0, float(line) * 4.0 + 15.0)
    try:
        flo, fhi = f(lo), f(hi)
        if flo * fhi > 0:
            return None
        return float(brentq(f, lo, hi, xtol=1e-3))
    except (ValueError, RuntimeError):
        return None