"""Ingest modules: they import, and their parsers / throttling / filters behave on realistic payloads (no network, no DB)."""
import importlib
import json
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

MODULES = [
    "app.ingest.nfl_feed", "app.ingest.nba_feed", "app.ingest.weather_feed", "app.ingest.market_feed",
    "app.engine.devig", "app.engine.redistribution", "app.engine.count_models", "app.engine.calibration",
    "app.eval.grader", "app.pipeline_v2", "app.db.models_ext",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    assert importlib.import_module(name)


def test_new_tables_registered_and_original_tables_untouched():
    import app.db.models as M
    import app.db.models_ext  # noqa: F401
    from app.db.base import Base

    for t in ("nfl_usage_weekly", "nfl_depth_current", "nfl_injury_feed", "nba_api_gamelog", "nba_onoff",
              "nba_live_lineup", "kickoff_weather", "market_lines", "clv_log"):
        assert t in Base.metadata.tables
    assert [c.name for c in M.Projection.__table__.columns][:4] == ["id", "game_id", "player_id", "market_key"]


# ---------------------------------------------------------------- NFL
def test_nfl_usage_merges_snaps_with_pbp_targets_and_carries():
    from app.ingest import nfl_feed as nf

    snaps = pl.DataFrame({"season": [2026, 2026], "week": [10, 10], "game_type": ["REG", "REG"],
                          "player": ["Ja'Marr Chase", "Joe Mixon"], "team": ["CIN", "HOU"], "position": ["WR", "RB"],
                          "offense_snaps": [60, 41], "offense_pct": [0.95, 0.66]})
    pbp = pl.DataFrame({"season": [2026] * 5, "week": [10] * 5, "season_type": ["REG"] * 5, "posteam": ["CIN", "CIN", "CIN", "HOU", "HOU"],
                        "pass_attempt": [1, 1, 0, 0, 0], "rush_attempt": [0, 0, 1, 1, 1],
                        "receiver_player_id": ["g1", "g1", None, None, None],
                        "rusher_player_id": [None, None, "g1", "g2", "g2"]})
    usage = nf.build_pbp_usage(pbp, {"g1": "Ja'Marr Chase", "g2": "Joe Mixon"})
    assert usage[(2026, 10, "CIN", "g1")] == {"targets": 2, "carries": 1, "name": "Ja'Marr Chase"}
    rows = {r["name_key"]: r for r in nf.build_usage_rows(snaps, usage)}
    chase = rows["jamarr chase"]
    assert (chase["offense_snaps"], chase["targets"], chase["carries"]) == (60, 2, 1)
    assert (rows["joe mixon"]["carries"], rows["joe mixon"]["targets"]) == (2, 0)


def test_nfl_injury_and_depth_builders_normalise_teams_and_pick_latest_snapshot():
    from app.ingest import nfl_feed as nf

    inj = pl.DataFrame({"season": [2026], "week": [10], "team": ["LA"], "full_name": ["Puka Nacua"], "position": ["WR"],
                        "gsis_id": ["00-1"], "report_status": ["Questionable"], "report_primary_injury": ["Knee"],
                        "practice_status": ["Limited Participation in Practice"], "practice_primary_injury": ["Knee"]})
    row = nf.build_injury_rows(inj)[0]
    assert row["team_abbr"] == "LAR" and row["report_status"] == "Questionable"

    depth = pl.DataFrame({"dt": ["2026-11-10", "2026-11-17", "2026-11-17"], "team": ["WAS", "WAS", "WAS"],
                          "player_name": ["Old Guy", "Jayden Daniels", "Marcus Mariota"], "espn_id": ["1", "2", "3"],
                          "gsis_id": ["a", "b", "c"], "pos_abb": ["QB", "QB", "QB"], "pos_rank": [1, 1, 2]})
    rows = nf.build_depth_rows(depth)
    assert {r["name_key"] for r in rows} == {"jayden daniels", "marcus mariota"}
    assert all(r["team_abbr"] == "WSH" for r in rows)
    assert nf.current_season(datetime(2026, 11, 18)) == 2026 and nf.current_season(datetime(2027, 1, 20)) == 2026


# ---------------------------------------------------------------- NBA
def test_rate_limiter_enforces_minimum_gap_and_headers_are_browser_like():
    from app.ingest import nba_feed as nb

    t = [0.0]
    slept: list[float] = []

    def sleep(s):
        slept.append(s)
        t[0] += s

    lim = nb.RateLimiter(1.5, clock=lambda: t[0], sleep=sleep)
    lim.wait()                                   # first call: no delay
    t[0] += 0.4
    lim.wait()                                   # 0.4s later -> must wait the remaining 1.1s
    assert slept == [pytest.approx(1.1)]
    t[0] += 5
    lim.wait()
    assert len(slept) == 1
    assert nb.RATE_LIMIT_SECONDS == 1.5
    assert "Chrome" in nb.NBA_HEADERS["User-Agent"] and nb.NBA_HEADERS["Referer"] == "https://www.nba.com/"
    assert nb.NBA_HEADERS["x-nba-stats-origin"] == "stats" and "Host" not in nb.LIVE_HEADERS


def test_call_endpoint_retries_then_succeeds_and_passes_headers():
    from app.ingest import nba_feed as nb

    calls = []

    def factory(**kw):
        calls.append(kw)
        if len(calls) < 3:
            raise ConnectionError("blocked")
        return "ok"

    lim = nb.RateLimiter(1.5, clock=lambda: 0.0, sleep=lambda s: None)
    assert nb.call_endpoint(factory, limiter=lim, sleep=lambda s: None, headers=nb.NBA_HEADERS, team_id=1) == "ok"
    assert len(calls) == 3 and calls[0]["headers"] is nb.NBA_HEADERS and calls[0]["timeout"] == nb.REQUEST_TIMEOUT

    def always_fails(**kw):
        raise ConnectionError("403")

    with pytest.raises(RuntimeError):
        nb.call_endpoint(always_fails, limiter=lim, sleep=lambda s: None, attempts=2)


def test_nba_parsers_gamelog_onoff_and_live_lineup():
    from app.ingest import nba_feed as nb

    logs = nb.parse_gamelog_rows([{"PLAYER_ID": 203999, "PLAYER_NAME": "Nikola Jokić", "GAME_ID": "0022600123",
                                   "GAME_DATE": "2026-11-16T00:00:00", "TEAM_ABBREVIATION": "DEN", "MATCHUP": "DEN vs. UTA",
                                   "WL": "W", "MIN": 34.5, "PTS": 28, "REB": 12, "AST": 9, "FG3M": 2, "STL": 1, "BLK": 0,
                                   "TOV": 3, "PLUS_MINUS": 11.0}], "2026-27")
    assert logs[0]["name_key"] == "nikola jokic" and logs[0]["game_date"].isoformat() == "2026-11-16" and logs[0]["pts"] == 28

    on = [{"TEAM_ID": 1, "TEAM_ABBREVIATION": "DEN", "VS_PLAYER_ID": 203999, "VS_PLAYER_NAME": "Jokić, Nikola",
           "COURT_STATUS": "On", "GP": 12, "MIN": 400.0, "PLUS_MINUS": 80, "OFF_RATING": 121.0, "DEF_RATING": 110.0, "NET_RATING": 11.0}]
    off = [{**on[0], "COURT_STATUS": "Off", "OFF_RATING": 108.0, "DEF_RATING": 115.0, "NET_RATING": -7.0}]
    rows = nb.parse_onoff_rows(on, off, "2026-27")
    assert {r["court_status"] for r in rows} == {"On", "Off"} and rows[0]["net_rating"] == 11.0

    assert nb.parse_live_minutes("PT24M31.00S") == pytest.approx(24.52, abs=0.01) and nb.parse_live_minutes(None) is None
    game = {"gameId": "0022600124", "gameStatus": 2, "homeTeam": {"teamTricode": "LAL", "players": [
        {"personId": 2544, "name": "LeBron James", "position": "F", "starter": "1", "status": "ACTIVE", "played": "1",
         "statistics": {"minutes": "PT10M00.00S"}},
        {"personId": 1, "name": "Bench Guy", "starter": "0", "status": "INACTIVE", "played": "0", "notPlayingReason": "INACTIVE_INJURY",
         "statistics": {}}]}, "awayTeam": {"teamTricode": "UTA", "players": []}}
    lineup = nb.parse_live_lineup(game)
    assert lineup[0]["starter"] is True and lineup[0]["minutes"] == 10.0 and lineup[1]["status"] == "INACTIVE"
    assert nb.current_season_str(datetime(2026, 11, 18).date()) == "2026-27" and nb.current_season_str(datetime(2027, 3, 1).date()) == "2026-27"


# ---------------------------------------------------------------- weather
def test_stadiums_json_is_complete_and_indoor_stadiums_are_skipped():
    from app.ingest import weather_feed as wf

    st = wf.load_stadiums()
    assert len(st) == 32 and all(-130 < v["lon"] < -60 and 24 < v["lat"] < 50 for v in st.values())
    assert wf.outdoor_stadium("GB", st)["stadium"] == "Lambeau Field"
    for dome in ("DET", "MIN", "NO", "LV", "LAR", "LAC"):
        assert wf.outdoor_stadium(dome, st) is None
    for retractable in ("DAL", "ATL", "HOU", "IND", "ARI"):
        assert wf.is_indoor(st[retractable]) and wf.outdoor_stadium(retractable, st) is None
    assert wf.outdoor_stadium("XXX", st) is None


def test_parse_hourly_reads_kickoff_hour_and_three_hour_window():
    from app.ingest import weather_feed as wf

    base = datetime(2026, 11, 22, 17, 0)
    hourly = {"time": [(base + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(8)],
              "wind_speed_10m": [5, 8, 12, 18, 25, 9, 7, 6], "wind_gusts_10m": [9, 14, 20, 30, 38, 15, 12, 10],
              "precipitation": [0, 0, 0.02, 0.1, 0.0, 0, 0, 0], "precipitation_probability": [10, 20, 40, 80, 30, 5, 0, 0],
              "temperature_2m": [40, 41, 42, 43, 44, 44, 43, 42]}
    out = wf.parse_hourly(hourly, datetime(2026, 11, 22, 18, 25, tzinfo=timezone.utc))   # 18:25Z -> the 18:00 hour
    assert out["wind_mph"] == 8 and out["gust_mph"] == 14 and out["precip_prob"] == pytest.approx(0.2)
    assert out["window_wind_max_mph"] == 25 and out["window_precip_in"] == pytest.approx(0.12)
    assert wf.parse_hourly(hourly, datetime(2026, 12, 1, 18, 0, tzinfo=timezone.utc)) is None


# ---------------------------------------------------------------- markets
ODDS_EVENT = {"id": "ev1", "home_team": "Boston Celtics", "away_team": "Miami Heat", "commence_time": "2026-11-19T00:30:00Z",
              "bookmakers": [
    {"key": "fanduel", "markets": [{"key": "h2h", "outcomes": [{"name": "Boston Celtics", "price": -999}]}]},
    {"key": "draftkings", "markets": [
        {"key": "h2h", "outcomes": [{"name": "Boston Celtics", "price": -200}, {"name": "Miami Heat", "price": 170}]},
        {"key": "spreads", "outcomes": [{"name": "Boston Celtics", "price": -110, "point": -5.5}, {"name": "Miami Heat", "price": -110, "point": 5.5}]},
        {"key": "totals", "outcomes": [{"name": "Over", "price": -110, "point": 220.5}, {"name": "Under", "price": -110, "point": 220.5}]},
        {"key": "player_points", "outcomes": [
            {"name": "Over", "description": "Jayson Tatum", "price": -115, "point": 27.5},
            {"name": "Under", "description": "Jayson Tatum", "price": -105, "point": 27.5}]},
        {"key": "player_anytime_td", "outcomes": [{"name": "Yes", "description": "X", "price": 150}]}]}]}


def test_odds_api_event_parses_only_draftkings_and_pairs_get_fair_probs():
    from app.ingest import market_feed as mf

    quotes = mf.attach_fair(mf.parse_odds_event(ODDS_EVENT, "NBA"))
    assert {q.source for q in quotes} == {"oddsapi:draftkings"} and len(quotes) == 8
    by = {(q.market_key, q.selection): q for q in quotes}
    home, away = by[("h2h", "home")], by[("h2h", "away")]
    assert home.fair + away.fair == pytest.approx(1.0) and home.fair < home.prob
    assert by[("spreads", "home")].line == -5.5 and by[("spreads", "away")].line == 5.5
    assert by[("spreads", "home")].pair_key == by[("spreads", "away")].pair_key
    over = by[("player_points", "Over|Jayson Tatum")]
    assert over.line == 27.5 and over.fair is not None and over.fair > by[("player_points", "Under|Jayson Tatum")].fair
    assert mf.Quote(source="x", sport="NBA", event_key="e", market_key="m", selection="s").key[-1] == 0.0


def test_kalshi_market_and_orderbook_parsing():
    from app.ingest import market_feed as mf

    m = {"ticker": "KXNBAGAME-26NOV22BOSGSW-GSW", "event_ticker": "KXNBAGAME-26NOV22BOSGSW", "yes_bid_dollars": "0.2800",
         "yes_ask_dollars": "0.3900", "yes_sub_title": "Golden State", "occurrence_datetime": "2026-11-23T04:30:00Z"}
    q = mf.parse_kalshi_market(m, "NBA", "h2h")
    assert q.prob == pytest.approx(0.335) and q.selection == "yes|Golden State" and q.pair_key == "KXNBAGAME-26NOV22BOSGSW"
    assert q.commence_time == datetime(2026, 11, 23, 2, 0, tzinfo=timezone.utc)        # end time minus 2.5h game length
    assert mf.parse_kalshi_market({**m, "yes_bid_dollars": "0.0000", "yes_ask_dollars": "1.0000"}, "NBA", "h2h") is None
    total = mf.parse_kalshi_market({**m, "floor_strike": 259.5, "event_ticker": "T"}, "NBA", "totals")
    assert total.line == 259.5 and total.pair_key is None and mf.attach_fair([total])[0].fair == pytest.approx(0.335)

    ob = {"orderbook_fp": {"yes_dollars": [["0.2700", "10"], ["0.2800", "5"]], "no_dollars": [["0.5900", "3"], ["0.6100", "2"]]}}
    assert mf.best_quotes_from_orderbook(ob) == (0.28, pytest.approx(0.39))
    legacy = {"orderbook": {"yes": [[27, 10], [28, 5]], "no": [[59, 3], [61, 2]]}}
    assert mf.best_quotes_from_orderbook(legacy) == (0.28, pytest.approx(0.39))


def test_polymarket_event_parsing_keeps_only_game_markets():
    from app.ingest import market_feed as mf

    ev = {"slug": "nba-phx-sac-2026-11-18", "startTime": "2026-11-19T03:00:00Z", "markets": [
        {"slug": "nba-phx-sac-2026-11-18", "sportsMarketType": "moneyline", "outcomes": json.dumps(["Suns", "Kings"]),
         "outcomePrices": json.dumps(["0.40", "0.60"]), "bestBid": 0.39, "bestAsk": 0.41, "gameStartTime": "2026-11-19 03:00:00+00",
         "clobTokenIds": json.dumps(["t0", "t1"])},
        {"slug": "x-exact", "sportsMarketType": "exact_margin", "outcomes": json.dumps(["Yes", "No"]), "outcomePrices": json.dumps(["0.5", "0.5"])}]}
    qs = mf.attach_fair(mf.parse_polymarket_event(ev, "NBA"))
    assert [q.selection for q in qs] == ["Suns", "Kings"] and qs[0].prob == pytest.approx(0.40)
    assert qs[0].fair + qs[1].fair == pytest.approx(1.0) and qs[0].commence_time == datetime(2026, 11, 19, 3, 0, tzinfo=timezone.utc)
