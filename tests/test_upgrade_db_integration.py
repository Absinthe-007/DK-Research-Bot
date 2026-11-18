"""
Opt-in integration test against a real (throwaway) Postgres. Skipped unless both are set:

    DKBOT_TEST_DB=1 DATABASE_URL=postgresql+asyncpg://dkbot:dkbot@localhost:5432/dkbot_test pytest tests/test_upgrade_db_integration.py

The database name must contain "test": the test drops and recreates every table.
"""
import asyncio
import os
from datetime import datetime, timedelta, timezone

import pytest

URL = os.environ.get("DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not (os.environ.get("DKBOT_TEST_DB") and "test" in URL.rsplit("/", 1)[-1]),
                                reason="set DKBOT_TEST_DB=1 and a DATABASE_URL whose database name contains 'test'")


async def _scenario():
    from sqlalchemy import select, text

    from app.db import models as M
    from app.db import models_ext as X
    from app.db.base import engine, session_scope, upsert
    from app.eval import grader
    from app.ingest import market_feed as mf
    from app.ingest import nfl_feed
    from scripts import init_db

    async with engine.begin() as conn:                           # clean slate (also removes the current_props view)
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
    await init_db.main()                                         # creates original + additive tables, seeds markets

    now = datetime.now(timezone.utc)
    # ---------------- feeds: idempotent upserts, wholesale depth-chart replace
    usage = [dict(season=2026, week=10, team_abbr="CIN", name_key=f"p{i}", player_name=f"P{i}", position="WR", gsis_id=None,
                  offense_snaps=50.0, offense_pct=0.8, targets=i, carries=0) for i in range(16)]
    async with session_scope() as s:
        await upsert(s, X.NflUsageWeekly, usage, ["season", "week", "team_abbr", "name_key"])
        await upsert(s, X.NflUsageWeekly, usage, ["season", "week", "team_abbr", "name_key"])
        n_usage = (await s.execute(text("SELECT COUNT(*) FROM nfl_usage_weekly"))).scalar()
    assert n_usage == 16
    depth = nfl_feed.build_depth_rows(__import__("polars").DataFrame(
        {"dt": ["2026-11-17"], "team": ["CIN"], "player_name": ["Joe Burrow"], "espn_id": ["1"], "gsis_id": ["g"], "pos_abb": ["QB"], "pos_rank": [1]}))
    async with session_scope() as s:
        await s.execute(X.NflDepthCurrent.__table__.delete())
        await upsert(s, X.NflDepthCurrent, depth, ["team_abbr", "pos_abb", "pos_rank", "name_key"])

    # ---------------- market snapshots: open is written once, current moves, frozen after the start
    start = now + timedelta(hours=2)
    def q(price, fair=None):
        return mf.Quote(source="oddsapi:draftkings", sport="NBA", event_key="ev1", market_key="player_points",
                        selection="Over|Test Player", line=24.5, price=price, prob=0.5, fair=fair, commence_time=start)
    assert await mf.apply_snapshot([q(-110)], now) == 1
    assert await mf.apply_snapshot([q(-125)], now + timedelta(minutes=30)) == 1
    async with session_scope() as s:
        row = (await s.execute(select(X.MarketLine))).scalar_one()
    assert (row.open_price, row.cur_price, row.n_updates) == (-110, -125, 2)
    async with session_scope() as s:                              # the game has now started in the DB's eyes
        await s.execute(text("UPDATE market_lines SET commence_time = :t"), {"t": now - timedelta(hours=1)})
    await mf.apply_snapshot([q(-150)], now)                       # incoming looks future-dated, stored row is past -> ignored
    async with session_scope() as s:
        row = (await s.execute(select(X.MarketLine))).scalar_one()
    assert row.cur_price == -125 and row.n_updates == 2           # closing line is frozen
    assert await mf.apply_snapshot([q(-130), q(-130)], now) == 1  # duplicates in one batch are collapsed

    # ---------------- grader: seed a finished NBA game and NFL game
    game_start = now - timedelta(hours=10)
    async with session_scope() as s:
        home, away = M.Team(sport="NBA", espn_id="1", abbr="AAA", name="A"), M.Team(sport="NBA", espn_id="2", abbr="BBB", name="B")
        s.add_all([home, away])
        await s.flush()
        nba_game = M.Game(sport="NBA", espn_event_id="g-nba", season="2027", start_time=game_start, home_team_id=home.id,
                          away_team_id=away.id, completed=True, boxscore_loaded=True)
        nfl_game = M.Game(sport="NFL", espn_event_id="g-nfl", season="2026", week=10, start_time=game_start, home_team_id=home.id,
                          away_team_id=away.id, completed=True, boxscore_loaded=True)
        s.add_all([nba_game, nfl_game])
        names = ["Box Row", "Active Zero", "Inactive", "No Evidence"]
        nba_players = [M.Player(sport="NBA", espn_id=f"n{i}", full_name=n, name_key=n.lower(), team_abbr="AAA") for i, n in enumerate(names)]
        nfl_players = [M.Player(sport="NFL", espn_id="f0", full_name="Snap Only", name_key="p3", team_abbr="CIN"),
                       M.Player(sport="NFL", espn_id="f1", full_name="Absent Guy", name_key="absent guy", team_abbr="CIN")]
        s.add_all(nba_players + nfl_players)
        await s.flush()
        s.add(M.NbaPlayerGamelog(player_id=nba_players[0].id, game_id=nba_game.id, season="2027", game_date=game_start.date(),
                                 team_abbr="AAA", opp_abbr="BBB", is_home=True, minutes=30, pts=20, reb=5, oreb=1, ast=2, fg3m=1,
                                 stl=0, blk=0, tov=1, fga=10, fta=2))
        await s.execute(text("""INSERT INTO nba_api_gamelog (nba_person_id, nba_game_id, name_key, player_name, team_abbr, game_date,
                                season, minutes, pts, reb) VALUES (2, 'x', 'active zero', 'Active Zero', 'AAA', :d, '2026-27', 12.0, 0, 0)"""),
                        {"d": game_start.date()})
        await s.execute(text("""INSERT INTO nba_live_lineup (nba_game_id, nba_person_id, name_key, status, played, game_status, updated_at)
                                VALUES ('x', 3, 'inactive', 'INACTIVE', false, 3, :u)"""), {"u": game_start + timedelta(hours=3)})
        def pick(game, player, mkt, side, line, price, close=None):
            return M.Pick(sport=game.sport, game_id=game.id, player_id=player.id, market_key=mkt, kind="ou", side=side, line=line,
                          price=price, first_price=price, model_prob=0.55, edge=0.1, stake_pct=0.01, proj_mean=1.0, close_price=close)
        s.add_all([pick(nba_game, nba_players[0], "player_points", "Over", 24.5, -110),      # 20 pts -> loss
                   pick(nba_game, nba_players[1], "player_rebounds", "Over", 0.5, -120),     # active, 0 reb -> loss (not DNP)
                   pick(nba_game, nba_players[2], "player_points", "Over", 10.5, -110),      # confirmed inactive -> void
                   pick(nba_game, nba_players[3], "player_points", "Over", 10.5, -110),      # no evidence -> pending
                   pick(nfl_game, nfl_players[0], "player_receptions", "Over", 0.5, -140),   # snaps, no stat line -> loss
                   pick(nfl_game, nfl_players[1], "player_receptions", "Over", 2.5, -110)])  # absent while team-week covered -> void
        s.add(X.MarketLine(source="oddsapi:draftkings", sport="NBA", event_key="ev9", market_key="player_points",
                           selection="Over|Box Row", line=24.5, commence_time=game_start, open_price=-110, cur_price=-130, cur_fair=0.55))
    out_nba = await grader.run("NBA")
    out_nfl = await grader.run("NFL")
    async with session_scope() as s:
        rows = (await s.execute(text("""SELECT p.full_name, pk.result, pk.actual, pk.profit_units FROM picks pk
                                        JOIN players p ON p.id = pk.player_id ORDER BY pk.id"""))).all()
        clv = (await s.execute(text("SELECT pick_id, close_price, close_fair, clv_prob, beat_close, source FROM clv_log"))).all()
    return rows, clv, out_nba, out_nfl


def test_end_to_end_against_postgres():
    rows, clv, out_nba, out_nfl = asyncio.run(_scenario())
    res = {r[0]: (r[1], r[2], r[3]) for r in rows}
    assert res["Box Row"] == ("loss", 20.0, -1.0)
    assert res["Active Zero"] == ("loss", 0.0, -1.0)           # active with 0 stats is a miss, not a DNP
    assert res["Inactive"] == ("void", None, 0.0)
    assert res["No Evidence"][0] is None                       # left pending, not guessed
    assert res["Snap Only"][0] == "loss" and res["Absent Guy"][0] == "void"
    assert out_nba["settled"] == 3 and out_nfl["settled"] == 2
    # CLV: Box Row flagged at -110 and closed -130 with fair 0.55 from market_lines -> beat the close
    box_clv = [c for c in clv if c[5] == "market_lines"]
    assert len(box_clv) == 1 and box_clv[0][1] == -130 and box_clv[0][2] == pytest.approx(0.55) and box_clv[0][4] is True
    assert out_nba["clv_logged"] >= 1
