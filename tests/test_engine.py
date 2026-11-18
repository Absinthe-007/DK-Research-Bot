"""
test_engine.py: Unit tests for DK-Research-Bot precision math and risk engine.
"""
import pytest
from app.engine.odds_math import (
    american_to_implied,
    implied_to_american,
    remove_vig_multiplicative,
    remove_vig_power,
    remove_vig_shin,
)
from app.engine.ev import calculate_ev
from app.engine.kelly import calculate_kelly_stake
from app.engine.grade import grade_bet


def test_american_to_implied_conversion():
    # Standard underdog (+) and favorite (-)
    assert pytest.approx(american_to_implied(100), 0.001) == 0.50
    assert pytest.approx(american_to_implied(150), 0.001) == 0.40
    assert pytest.approx(american_to_implied(-110), 0.001) == 0.5238


def test_implied_to_american_roundtrip():
    prob = 0.60
    odds = implied_to_american(prob)
    converted_back = american_to_implied(odds)
    assert pytest.approx(prob, 0.02) == converted_back


def test_vig_removal_methods():
    # Classic -110 / -110 market
    odds_list = [-110, -110]
    mult_res = remove_vig_multiplicative(odds_list)
    power_res = remove_vig_power(odds_list)
    
    # Probabilities should sum to 1.0
    assert pytest.approx(sum(mult_res), 1e-5) == 1.0
    assert pytest.approx(sum(power_res), 1e-5) == 1.0


def test_calculate_ev_positive():
    # 55% true probability at +100 odds (Even money)
    res = calculate_ev(true_prob=0.55, odds=100)
    assert res["is_positive_ev"] is True
    assert res["ev_percentage"] > 0.0
    assert res["edge_percentage"] > 0.0


def test_calculate_kelly_stake_zero_edge():
    # Negative or zero edge should yield zero stake
    res = calculate_kelly_stake(true_prob=0.45, odds=-110, bankroll=1000.0)
    assert res["recommended_stake"] == 0.0
    assert res["bankroll_percentage"] == 0.0


def test_calculate_kelly_stake_positive():
    # Positive edge should calculate a fractional stake within bankroll cap
    res = calculate_kelly_stake(true_prob=0.58, odds=100, bankroll=1000.0, fraction=0.25)
    assert res["recommended_stake"] > 0.0
    assert res["recommended_stake"] <= 250.0 # Hard cap or fraction check


def test_grade_bet_win_and_clv():
    # Win bet at +100, market closes at -120 (price shortened toward our side = good CLV)
    res = grade_bet(bet_result="win", stake=100.0, odds=100, closing_odds=-120)
    assert res["profit_loss"] == 100.0
    assert res["beat_closing_line"] is True
    assert res["clv_percentage"] > 0.0


def test_grade_bet_negative_clv():
    # Bet at +100, market drifts to +120 (we got a worse price than the close)
    res = grade_bet(bet_result="loss", stake=100.0, odds=100, closing_odds=120)
    assert res["profit_loss"] == -100.0
    assert res["beat_closing_line"] is False
    assert res["clv_percentage"] < 0.0


def test_grade_bet_favorite_win_and_push():
    win = grade_bet(bet_result="win", stake=110.0, odds=-110, closing_odds=-110)
    assert win["profit_loss"] == 100.0
    push = grade_bet(bet_result="push", stake=50.0, odds=-110, closing_odds=-110)
    assert push["profit_loss"] == 0.0


def test_grade_bet_invalid_result():
    with pytest.raises(ValueError):
        grade_bet(bet_result="void", stake=10.0, odds=100, closing_odds=100)


def test_ev_negative_odds():
    res = calculate_ev(true_prob=0.60, odds=-110)
    assert res["decimal_odds"] == pytest.approx(1.9091, abs=1e-4)
    assert res["is_positive_ev"] is True


def test_implied_to_american_underdog_and_invalid():
    assert implied_to_american(0.40) == 150
    with pytest.raises(ValueError):
        implied_to_american(1.0)


def test_shin_sums_to_one_and_symmetric():
    res = remove_vig_shin([-110, -110])
    assert sum(res) == pytest.approx(1.0, abs=1e-9)
    assert res[0] == pytest.approx(0.5, abs=1e-6)


def test_shin_lopsided_market_shrinks_longshot_more_than_multiplicative():
    odds = [-400, 300]
    shin = remove_vig_shin(odds)
    mult = remove_vig_multiplicative(odds)
    assert sum(shin) == pytest.approx(1.0, abs=1e-9)
    assert all(0.0 < p < 1.0 for p in shin)
    # Favorite-longshot bias: Shin gives the favorite more, the longshot less
    assert shin[0] > mult[0]
    assert shin[1] < mult[1]


def test_shin_three_way_market():
    res = remove_vig_shin([-150, 280, 320])
    assert sum(res) == pytest.approx(1.0, abs=1e-9)
    assert all(p > 0 for p in res)


def test_shin_fair_market_passthrough():
    res = remove_vig_shin([100, 100])
    assert res == pytest.approx([0.5, 0.5])
