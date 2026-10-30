import math

import numpy as np
import pytest

from app.engine import baselines as B
from app.engine.distributions import blend_with_history, empirical_hit_rate, fit, prob_over_under
from app.engine.ev import edge, expected_value, screen
from app.engine.kelly import BetCandidate, allocate, kelly_full
from app.engine.odds_math import (
    american_to_decimal, decimal_to_american, devig_multiplicative, devig_power, implied_prob, prob_to_american,
)


# ---------------- odds math ----------------
def test_american_decimal_roundtrip():
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert american_to_decimal(-110) == pytest.approx(1.909090, 1e-5)
    for a in (-250, -110, 100, 145, 600):
        assert decimal_to_american(american_to_decimal(a)) == a


def test_implied_and_devig():
    assert implied_prob(-110) == pytest.approx(0.5238, 1e-3)
    fair = devig_multiplicative([-110, -110])
    assert fair == pytest.approx([0.5, 0.5])
    pw = devig_power([-200, 170])
    assert sum(pw) == pytest.approx(1.0)
    assert pw[0] > devig_multiplicative([-200, 170])[0] - 0.02   # power shifts vig onto the dog
    assert prob_to_american(0.5) == 100


def test_invalid_odds():
    with pytest.raises(ValueError):
        american_to_decimal(50)


# ---------------- distributions ----------------
def test_poisson_half_point():
    d = fit("poisson", 2.0)
    over, under, push = prob_over_under(d, 1.5)
    # P(X>=2) for Poisson(2) = 1 - e^-2(1+2)
    assert over == pytest.approx(1 - math.exp(-2) * 3, 1e-9)
    assert push == 0 and over + under == pytest.approx(1)


def test_integer_line_push():
    d = fit("poisson", 3.0)
    over, under, push = prob_over_under(d, 3)
    assert push == pytest.approx(math.exp(-3) * 27 / 6)
    assert over + under + push == pytest.approx(1)


def test_negbin_overdispersion_and_fallback():
    nb = fit("negbin", 8.0, 4.0)       # var 16 > mean 8
    assert nb.family == "negbin" and nb.mean == pytest.approx(8.0) and nb.std == pytest.approx(4.0)
    assert fit("negbin", 8.0, 2.0).family == "poisson"   # var < mean -> Poisson


def test_normal_continuity():
    d = fit("normal", 25.0, 6.0)
    over, under, _ = prob_over_under(d, 24.5)
    from scipy.stats import norm
    assert over == pytest.approx(1 - norm.cdf(-0.5 / 6))   # P(X>=25) = 1 - F(24.5) after continuity correction
    assert over + under == pytest.approx(1)


def test_gamma_moments():
    d = fit("gamma", 60.0, 25.0)
    assert d.mean == pytest.approx(60.0) and d.std == pytest.approx(25.0)


def test_anytime_td_none_line():
    d = fit("poisson", 0.5)
    over, _, _ = prob_over_under(d, None)
    assert over == pytest.approx(1 - math.exp(-0.5))


def test_empirical_and_blend():
    rate, n = empirical_hit_rate(np.array([10, 12, 8, 15, 11]), 10.5)
    assert rate == pytest.approx(3 / 5) and n == pytest.approx(5)
    assert blend_with_history(0.5, 1.0, 0.0) == 0.5
    assert 0.5 < blend_with_history(0.5, 1.0, 20) < 1.0


# ---------------- baselines ----------------
def test_ewm_favors_recent():
    mu, _ = B.ewm_mean_std([10] * 10 + [20] * 3)
    assert mu > 13.5            # simple mean would be 12.3


def test_opponent_factor_shrinks():
    assert B.opponent_factor(28, 25, 0) == 1.0
    f = B.opponent_factor(28, 25, 6, k=8)
    assert 1.0 < f < 28 / 25


def test_weather():
    f, flags = B.weather_factor("player_pass_yds", 22, 0.1, 50)
    assert f == pytest.approx(0.85) and flags
    assert B.weather_factor("player_pass_yds", 30, 0.9, 10, "dome") == (1.0, [])


# ---------------- EV / Kelly ----------------
def test_ev_formula():
    # 55% at -110: EV = .55*(100/110) - .45
    assert expected_value(0.55, -110) == pytest.approx(0.55 * 100 / 110 - 0.45)
    assert edge(0.55, -110) == pytest.approx(0.55 - 110 / 210)
    # push removes action but not value sign
    assert expected_value(0.5, 100, p_push=0.2) == pytest.approx(0.5 - 0.3)


def test_edge_ev_identity():
    p, a, push = 0.57, -115, 0.05
    e = edge(p, a, push)
    assert expected_value(p, a, push) == pytest.approx(e * american_to_decimal(a) * (1 - push))


def test_screen_threshold():
    assert screen("Over", 24.5, -110, 0.57).flagged          # edge ~4.6 pts
    assert not screen("Over", 24.5, -110, 0.55).flagged      # ~2.6 pts
    r = screen("Over", 24.5, -110, 0.75, fair_prob=0.5)
    assert any("GAP" in f for f in r.flags)


def test_kelly():
    assert kelly_full(0.55, 100) == pytest.approx(0.10)
    assert kelly_full(0.45, 100) == 0.0
    out = allocate([BetCandidate("a", 1, 0.6, 100), BetCandidate("b", 1, 0.6, 100), BetCandidate("c", 2, 0.58, -110)],
                   bankroll=1000, fraction=0.25, max_stake_pct=0.02, max_total_pct=0.04)
    assert sum(v["stake_pct"] for v in out.values()) <= 0.04 + 1e-9
    assert all(v["stake_pct"] <= 0.02 for v in out.values())


# ---------------- scrapers / planner ----------------
def test_planner_checkpoints():
    from datetime import datetime, timedelta, timezone
    from app.planner import CHECKPOINTS, due
    now = datetime(2026, 10, 26, 22, 37, tzinfo=timezone.utc)
    tip = datetime(2026, 10, 27, 0, 0, tzinfo=timezone.utc)          # 8pm ET tip
    # T-90m = 22:30 falls between last run (20:00) and now -> due
    assert due([tip], CHECKPOINTS["NBA"], now - timedelta(hours=2, minutes=37), now)
    # already ran at 22:31 -> nothing new until T-30m
    assert not due([tip], CHECKPOINTS["NBA"], datetime(2026, 10, 26, 22, 31, tzinfo=timezone.utc), now)
    # no games -> never due
    assert not due([], CHECKPOINTS["NFL"], None, now)


def test_parse_prop_items_pairs_and_milestones():
    from app.ingestion.espn import parse_prop_items
    ath = {"$ref": "http://x/athletes/123?lang=en"}
    items = [
        {"athlete": ath, "type": {"name": "Total Points"}, "odds": {"american": {"value": "-114"}, "total": {"value": "21.5"}}},
        {"athlete": ath, "type": {"name": "Total Points"}, "odds": {"american": {"value": "-112"}, "total": {"value": "21.5"}}},
        {"athlete": ath, "type": {"name": "Points Milestones"}, "odds": {"american": {"value": "-113"}, "total": {"value": "22+"}}},
        {"athlete": ath, "type": {"name": "Points Milestones"}, "odds": {"american": {"value": "+150"}, "total": {"value": "25+"}}},
        # second ladder (a 1st-half sub-market) filed under the same type name -> must be dropped
        {"athlete": ath, "type": {"name": "Points Milestones"}, "odds": {"american": {"value": "-300"}, "total": {"value": "8+"}}},
        {"athlete": ath, "type": {"name": "Points Milestones"}, "odds": {"american": {"value": "+200"}, "total": {"value": "12+"}}},
        {"athlete": ath, "type": {"name": "Anytime Touchdown Scorer"}, "odds": {"american": {"value": "+120"}, "total": {}}},
        {"athlete": ath, "type": {"name": "First Basket"}, "odds": {"american": {"value": "+900"}}},
    ]
    rows = parse_prop_items(items)
    ou = [r for r in rows if r["kind"] == "ou"]
    assert [(r["side"], r["price"], r["line"]) for r in ou] == [("Over", -114, 21.5), ("Under", -112, 21.5)]
    pts_alt = sorted(r["line"] for r in rows if r["kind"] == "alt" and r["market_key"] == "player_points")
    assert pts_alt == [21.5, 24.5]          # only the ladder consistent with the 21.5 main line
    assert [r["line"] for r in rows if r["market_key"] == "player_anytime_td"] == [0.5]
    assert len(rows) == 5


def test_parse_american():
    from app.ingestion.common import parse_american, parse_float
    assert parse_american("+170") == 170 and parse_american("EVEN") == 100 and parse_american("OFF") is None
    assert parse_float("o43.5") == 43.5 and parse_float("+4.5") == 4.5


def test_infer_nfl_group_from_stats():
    import pandas as pd
    from app.engine.statboard import _infer_group
    row = dict(pass_attempts=0, carries=0, targets=0, fgm=0, xpm=0, tackles=0)
    mk = lambda **kw: pd.DataFrame([row | kw] * 3)
    assert _infer_group(mk(pass_attempts=30)) == "QB"
    assert _infer_group(mk(carries=15, targets=2)) == "RB"
    assert _infer_group(mk(targets=6, carries=1)) == "REC"
    assert _infer_group(mk(fgm=2, xpm=3)) == "K"
    assert _infer_group(mk(tackles=5)) == "DEF"


def test_pick_result_fits_column():
    from app.db.models import Pick
    width = Pick.__table__.c.result.type.length
    for r in ("win", "loss", "push", "void", "pulled"):
        assert len(r) <= width
