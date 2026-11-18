"""Opt-in adjustment layer: pure maths and the feature flag (database behaviour is in the integration test)."""
import pytest

from app.engine import adjust as adj


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("ADJUSTED_PROJECTIONS", raising=False)
    assert not adj.adjusted_enabled()
    for v in ("1", "true", "YES", "on"):
        monkeypatch.setenv("ADJUSTED_PROJECTIONS", v)
        assert adj.adjusted_enabled()
    monkeypatch.setenv("ADJUSTED_PROJECTIONS", "0")
    assert not adj.adjusted_enabled()


def test_healthy_roster_is_a_noop():
    out = adj.conditional_multipliers({1: 30.0, 2: 25.0}, {})
    assert all(s.factor == 1.0 and s.p_active == 1.0 for s in out.values())


def test_star_out_lifts_teammates_and_out_player_is_zeroed():
    base = {1: 36.0, 2: 30.0, 3: 28.0, 4: 22.0, 5: 14.0, 6: 8.0}
    out = adj.conditional_multipliers(base, {1: "Out"}, cap=40.0)
    assert out[1].factor == 0.0 and out[1].p_active == 0.0
    assert all(out[k].factor > 1.0 for k in range(2, 7))
    assert out[2].adj <= 40.0
    assert out[2].factor == pytest.approx(out[2].adj / 30.0)            # inside the clip range


def test_questionable_player_keeps_his_own_projection_but_teammates_get_the_expected_bump():
    base = {1: 34.0, 2: 30.0, 3: 26.0, 4: 20.0, 5: 12.0}
    out = adj.conditional_multipliers(base, {1: "Questionable"})
    assert out[1].factor == pytest.approx(1.0) and out[1].p_active == 0.5   # conditional on playing: unchanged
    assert out[2].factor > 1.0
    full_out = adj.conditional_multipliers(base, {1: "Out"})
    assert 1.0 < out[2].factor < full_out[2].factor                          # expected bump is smaller than a certain absence


def test_elasticity_and_clip():
    base = {1: 36.0, 2: 6.0, 3: 30.0, 4: 28.0, 5: 20.0}
    full = adj.conditional_multipliers(base, {1: "Out"}, hi=1.6)
    damp = adj.conditional_multipliers(base, {1: "Out"}, hi=1.6, elasticity=0.5)
    assert damp[3].factor - 1.0 == pytest.approx((full[3].factor - 1.0) / 2) or damp[3].factor < full[3].factor
    assert max(s.factor for s in full.values()) <= 1.6


def test_build_rows_only_scales_requested_stats_and_skips_unchanged():
    shifts = {1: adj.Shift(34.0, 1.2, 1.0, 20.0), 2: adj.Shift(30.0, 1.0, 1.0, 20.0), 3: adj.Shift(0.0, 0.0, 0.0, 0.0)}
    stat_rows = [dict(player_id=1, stat="pts", mean=20.0, lo=12.0, hi=28.0), dict(player_id=1, stat="foo", mean=5.0, lo=None, hi=None),
                 dict(player_id=2, stat="pts", mean=10.0, lo=5.0, hi=15.0), dict(player_id=3, stat="pts", mean=9.0, lo=1.0, hi=2.0)]
    rows = adj.build_rows(7, "NBA", shifts, stat_rows, adj.NBA_SCALED, {1: ("Active", "live_lineup")}, {1: 28.0})
    assert len(rows) == 1
    r = rows[0]
    assert (r["player_id"], r["stat"], r["original_mean"], r["adjusted_mean"], r["factor"]) == (1, "pts", 20.0, 24.0, 1.2)
    assert (r["lo"], r["hi"], r["source"], r["status"]) == (14.4, 33.6, "live_lineup", "Active")


def test_cli_adjust_is_skipped_without_flag(monkeypatch):
    import asyncio

    from app import pipeline_v2

    monkeypatch.delenv("ADJUSTED_PROJECTIONS", raising=False)
    asyncio.run(pipeline_v2.adjust(("NBA",)))      # must return without touching the database
