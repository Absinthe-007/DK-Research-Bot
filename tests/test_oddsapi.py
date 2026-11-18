"""Unit tests for the optional Odds API connector (no network, no DB)."""
from app.ingestion.oddsapi import Budget, _line_rows, _same_team, parse_event_props


def test_team_matching_handles_city_name_differences():
    assert _same_team("Los Angeles Clippers", "LA Clippers")
    assert _same_team("Portland Trail Blazers", "Portland Trail Blazers")
    assert not _same_team("Los Angeles Lakers", "LA Clippers")


def test_parse_event_props_keeps_only_draftkings_two_way():
    ev = {"bookmakers": [
        {"key": "fanduel", "markets": [{"key": "player_points", "outcomes": [
            {"name": "Over", "description": "A B", "price": -110, "point": 20.5}]}]},
        {"key": "draftkings", "markets": [{"key": "player_points", "last_update": "2026-11-18T12:00:00Z", "outcomes": [
            {"name": "Over", "description": "A B", "price": -112, "point": 20.5},
            {"name": "Under", "description": "A B", "price": -108, "point": 20.5},
            {"name": "Yes", "description": "A B", "price": 100, "point": 0.5}]}]}]}
    rows = parse_event_props(ev, "NBA")
    assert [(r["side"], r["price"], r["line"]) for r in rows] == [("Over", -112, 20.5), ("Under", -108, 20.5)]


def test_line_rows_maps_home_away_over_under():
    ev = {"home_team": "H", "away_team": "A", "bookmakers": [{"key": "draftkings", "markets": [
        {"key": "h2h", "outcomes": [{"name": "H", "price": -150}, {"name": "A", "price": 130}]},
        {"key": "spreads", "outcomes": [{"name": "H", "price": -110, "point": -3.5}, {"name": "A", "price": -110, "point": 3.5}]},
        {"key": "totals", "outcomes": [{"name": "Over", "price": -110, "point": 221.5}, {"name": "Under", "price": -110, "point": 221.5}]}]}]}
    rows = _line_rows(1, ev)
    assert len(rows) == 6
    assert {(r["market_type"], r["side"]): r["point"] for r in rows}[("spreads", "home")] == -3.5


def test_budget_respects_run_cap_and_reserve():
    b = Budget(max_run=10, reserve=50)
    assert b.can_spend(10) and not b.can_spend(11)
    b.record({"x-requests-remaining": "55"}, 4)
    assert b.can_spend(5) and not b.can_spend(6)
