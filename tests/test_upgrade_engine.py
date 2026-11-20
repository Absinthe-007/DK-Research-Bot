"""Pure-math tests for the upgrade helpers: de-vig, injury redistribution, scipy count models, calibration."""
import random

import pytest

from app.engine import calibration as cal
from app.engine import count_models as cm
from app.engine import devig
from app.engine import redistribution as rd
from app.engine.odds_math import implied_prob


def test_devig_two_way_sums_to_one_and_orders_correctly():
    for method in ("multiplicative", "power", "shin"):
        a, b = devig.two_way_fair(-130, 110, method)
        assert a + b == pytest.approx(1.0)
        assert a > b
        assert a < implied_prob(-130)           # vig removed -> below the raw breakeven
    assert devig.overround([-110, -110]) == pytest.approx(0.047619, abs=1e-5)
    with pytest.raises(ValueError):
        devig.fair_probs([-110, -110], "nope")


def test_fair_from_probs_normalises_prediction_market_quotes():
    out = devig.fair_from_probs([0.55, 0.50])
    assert sum(out) == pytest.approx(1.0) and out[0] == pytest.approx(0.55 / 1.05)
    assert sum(devig.fair_from_probs([0.55, 0.50], "power")) == pytest.approx(1.0)


def test_clv_is_positive_when_close_is_shorter_than_placed():
    assert devig.clv(+110, implied_prob(-105)) > 0       # we took plus money, it closed as a favourite
    assert devig.clv(-130, implied_prob(+100)) < 0


def test_play_probability_statuses():
    assert rd.play_probability("Out") == 0.0
    assert rd.play_probability("Questionable") == 0.5
    assert rd.play_probability(None) == 1.0
    assert rd.play_probability("questionable", {"Questionable": 0.3}) == 0.3


def test_redistribution_conserves_minutes_and_respects_cap():
    base = {"A": 34.0, "B": 30.0, "C": 26.0, "D": 20.0, "E": 10.0}
    res = rd.redistribute_minutes(base, {"A": "Out", "B": "Questionable"}, cap=40.0)
    assert res.values["A"] == 0.0
    assert res.values["B"] < base["B"] + res.boosts["B"] + 1e-9 and res.values["B"] > 0
    assert max(res.values.values()) <= 40.0 + 1e-9
    assert sum(res.values.values()) == pytest.approx(sum(base.values()))
    assert res.unallocated == pytest.approx(0.0)
    assert res.vacated == pytest.approx(34.0 + 15.0)


def test_redistribution_same_position_group_gets_more():
    shares = {"WR1": 0.30, "WR2": 0.22, "TE": 0.22, "RB": 0.12, "WR3": 0.14}
    pos = {"WR1": "WR", "WR2": "WR", "TE": "TE", "RB": "RB", "WR3": "WR"}
    plain = rd.redistribute_usage(shares, {"WR1": "Out"})
    grouped = rd.redistribute_usage(shares, {"WR1": "Out"}, positions=pos)
    assert grouped.boosts["WR2"] > plain.boosts["WR2"]
    assert sum(grouped.values.values()) == pytest.approx(1.0)


def test_redistribution_all_capped_reports_unallocated_and_noop_when_healthy():
    res = rd.redistribute({"A": 30, "B": 30}, {"A": 0.0}, caps=30)
    assert res.values["B"] == 30 and res.unallocated == pytest.approx(30.0)
    healthy = rd.redistribute({"A": 30, "B": 30}, {})
    assert healthy.values == {"A": 30.0, "B": 30.0} and healthy.vacated == 0.0
    assert rd.usage_multiplier(30, 36) == pytest.approx(1.2)


def test_count_model_lines_push_and_void():
    half = cm.over_under(5.2, 4.5, var=8.0, p_active=0.95)
    assert half["push"] == 0.0 and half["void"] == pytest.approx(0.05)
    assert sum(half.values()) == pytest.approx(1.0)
    whole = cm.over_under(5.0, 5, var=None)
    assert whole["push"] > 0.1 and sum(whole.values()) == pytest.approx(1.0)
    assert cm.fair_probability(whole, "over") + cm.fair_probability(whole, "under") == pytest.approx(1.0)


def test_count_model_family_choice_and_overdispersion_widens_tails():
    assert cm.choose_family(4.0, None) == "poisson"
    assert cm.choose_family(4.0, 4.1) == "poisson"
    assert cm.choose_family(4.0, 7.0) == "nbinom"
    tight = cm.over_under(4.0, 9.5, var=4.0)["over"]
    wide = cm.over_under(4.0, 9.5, var=9.0)["over"]
    assert wide > tight
    table = cm.prob_table(4.0, [2.5, 3.5, 4.5], var=6.0)
    assert [r["over"] for r in table] == sorted((r["over"] for r in table), reverse=True)
    assert cm.as_prop_distribution(4.0, 6.0).family == "negbin"


def test_calibrator_identity_when_underfit_and_improves_brier_when_fitted():
    c = cal.ProbCalibrator("isotonic").fit([0.6, 0.7], [1, 0])
    assert not c.fitted and c.predict(0.7) == pytest.approx(0.7)
    rng = random.Random(7)
    probs = [rng.random() for _ in range(600)]
    outcomes = [1 if rng.random() < p * 0.75 else 0 for p in probs]      # overconfident model
    for method in ("isotonic", "platt"):
        fitted = cal.ProbCalibrator(method).fit(probs, outcomes)
        assert fitted.fitted
        adj = [fitted.predict(p) for p in probs]
        assert cal.brier(adj, outcomes) < cal.brier(probs, outcomes)
        clone = cal.ProbCalibrator.from_dict(fitted.to_dict())
        assert clone.predict(0.8) == pytest.approx(fitted.predict(0.8))
    rel = cal.reliability_table(probs, outcomes, bins=5)
    assert sum(r["n"] for r in rel) == 600
    assert cal.shrink_to_market(0.6, 0.5, 0.35) == pytest.approx(0.535)


def test_fit_shrink_and_apply():
    import numpy as np

    from app.engine import calibration as cal

    rng = np.random.default_rng(0)
    ref = rng.uniform(0.45, 0.6, 2000)
    model = np.clip(ref + rng.normal(0, 0.1, 2000), 0.05, 0.95)
    true_p = ref + 0.3 * (model - ref)                      # only 30% of the claimed gap is real
    y = (rng.random(2000) < true_p).astype(int)
    k = cal.fit_shrink(model, ref, y)
    assert 0.15 <= k <= 0.5
    assert cal.fit_shrink(model[:10], ref[:10], y[:10]) == cal.DEFAULT_SHRINK       # too few picks -> default
    assert cal.apply_shrink(0.7, 0.5, 0.5) == pytest.approx(0.6)
    assert cal.apply_shrink(0.5, 0.5, 0.5) == pytest.approx(0.5)
    assert 0 < cal.apply_shrink(1.0, 0.99, 1.0) < 1
