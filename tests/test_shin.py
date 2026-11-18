"""2-way Shin must match the general (bisection) Shin solution, including lopsided prices."""
import math

import pytest

from app.engine.odds_math import devig_shin, implied_prob


def _shin_bisect(odds):
    q = [implied_prob(o) for o in odds]
    s = sum(q)
    f = lambda z: [(math.sqrt(z * z + 4 * (1 - z) * x * x / s) - z) / (2 * (1 - z)) for x in q]
    lo, hi = 0.0, 0.99
    for _ in range(200):
        z = (lo + hi) / 2
        lo, hi = (z, hi) if sum(f(z)) > 1 else (lo, z)
    p = f(z)
    t = sum(p)
    return [x / t for x in p]


@pytest.mark.parametrize("odds", [[-115, -115], [-130, 100], [-200, 160], [-400, 300], [-800, 500], [-115, -105], [500, -800]])
def test_two_way_matches_bisection(odds):
    got, want = devig_shin(odds), _shin_bisect(odds)
    assert got == pytest.approx(want, abs=1e-5)
    assert sum(got) == pytest.approx(1.0, abs=1e-9)


def test_favorite_shaded_less_than_old_shortcut():
    # -400/+300: true Shin favorite ~77.5% (old shortcut gave 78.9%)
    assert devig_shin([-400, 300])[0] == pytest.approx(0.775, abs=0.002)
