"""
Probability calibration (scikit-learn fits, numpy predictions): learn how stated model probabilities map
to observed hit rates from graded picks, then apply that map. Opt-in: the original model probabilities are
never overwritten; callers pass them through `ProbCalibrator.predict` when they want the calibrated number.

A fitted calibrator is plain JSON (`to_dict` / `from_dict`), so it can live in the repo or the data release.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

EPS = 1e-4


def _clip(p):
    return np.clip(np.asarray(p, float), EPS, 1 - EPS)


def brier(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    p, y = _clip(probs), np.asarray(outcomes, float)
    return float(np.mean((p - y) ** 2))


def log_loss(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    p, y = _clip(probs), np.asarray(outcomes, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_table(probs: Sequence[float], outcomes: Sequence[int], bins: int = 10) -> list[dict]:
    """Predicted vs observed rate per probability bin."""
    p, y = np.asarray(probs, float), np.asarray(outcomes, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1.0 else (p <= hi))
        if m.any():
            rows.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                         "avg_pred": float(p[m].mean()), "hit_rate": float(y[m].mean())})
    return rows


class ProbCalibrator:
    """Identity until fitted with at least `min_n` settled outcomes."""

    def __init__(self, method: str = "isotonic", min_n: int = 50):
        if method not in ("isotonic", "platt"):
            raise ValueError("method must be 'isotonic' or 'platt'")
        self.method, self.min_n = method, min_n
        self.x: list[float] = []
        self.y: list[float] = []
        self.coef: tuple[float, float] | None = None       # Platt: a, b on logit(p)
        self.n_fit = 0

    @property
    def fitted(self) -> bool:
        return bool(self.x) if self.method == "isotonic" else self.coef is not None

    def fit(self, probs: Sequence[float], outcomes: Sequence[int]) -> "ProbCalibrator":
        p, y = _clip(probs), np.asarray(outcomes, int)
        self.n_fit = len(p)
        if len(p) < self.min_n or len(set(y.tolist())) < 2:
            return self                                    # not enough signal: stay identity
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip").fit(p, y)
            self.x, self.y = iso.X_thresholds_.tolist(), iso.y_thresholds_.tolist()
        else:
            from sklearn.linear_model import LogisticRegression

            z = np.log(p / (1 - p)).reshape(-1, 1)
            lr = LogisticRegression(C=1e6).fit(z, y)
            self.coef = (float(lr.coef_[0][0]), float(lr.intercept_[0]))
        return self

    def predict(self, probs):
        scalar = np.isscalar(probs)
        p = _clip(probs)
        if not self.fitted:
            out = p
        elif self.method == "isotonic":
            out = np.interp(p, self.x, self.y)
        else:
            a, b = self.coef  # type: ignore[misc]
            out = 1.0 / (1.0 + np.exp(-(a * np.log(p / (1 - p)) + b)))
        out = _clip(out)
        return float(out) if scalar or out.ndim == 0 else out

    def to_dict(self) -> dict:
        return {"method": self.method, "min_n": self.min_n, "x": self.x, "y": self.y,
                "coef": list(self.coef) if self.coef else None, "n_fit": self.n_fit}

    @classmethod
    def from_dict(cls, d: dict) -> "ProbCalibrator":
        c = cls(d.get("method", "isotonic"), d.get("min_n", 50))
        c.x, c.y, c.n_fit = list(d.get("x", [])), list(d.get("y", [])), int(d.get("n_fit", 0))
        c.coef = tuple(d["coef"]) if d.get("coef") else None  # type: ignore[assignment]
        return c


def shrink_to_market(model_p: float, market_p: float, weight: float) -> float:
    """Convex blend; `weight` is the share given to the model (the repo's anchoring idea, on probabilities)."""
    w = min(1.0, max(0.0, weight))
    return w * model_p + (1 - w) * market_p


def logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))
