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
    await engine.dispose()
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


async def _adjust_scenario():
    from sqlalchemy import text

    from app.db import models as M
    from app.db.base import engine, session_scope
    from app.engine import adjust
    from scripts import init_db

    async with engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
    await init_db.main()
    now = datetime.now(timezone.utc)
    async with session_scope() as s:
        aaa, bbb, cin = (M.Team(sport=sp, espn_id=i, abbr=a, name=a) for sp, i, a in (("NBA", "1", "AAA"), ("NBA", "2", "BBB"), ("NFL", "3", "CIN")))
        s.add_all([aaa, bbb, cin])
        await s.flush()
        mk = lambda sp, eid, **kw: M.Game(sport=sp, espn_event_id=eid, season="2026", home_team_id=aaa.id, away_team_id=bbb.id, **kw)  # noqa: E731
        nba_game = mk("NBA", "up-nba", start_time=now + timedelta(hours=5))
        nfl_game = M.Game(sport="NFL", espn_event_id="up-nfl", season="2026", week=10, start_time=now + timedelta(hours=40),
                          home_team_id=cin.id, away_team_id=bbb.id)
        past = [mk("NBA", f"past{i}", start_time=now - timedelta(days=i + 1), completed=True) for i in range(10)]
        s.add_all([nba_game, nfl_game, *past])
        names = ["star", "p1", "p2", "p3", "p4", "p5", "p6", "p7"]
        nba = {n: M.Player(sport="NBA", espn_id=n, full_name=n, name_key=n, team_abbr="AAA") for n in names}
        nfl = {n: M.Player(sport="NFL", espn_id=f"f-{n}", full_name=n, name_key=n, team_abbr="CIN", position=pos)
               for n, pos in (("wr1", "WR"), ("wr2", "WR"), ("te1", "TE"), ("rb1", "RB"))}
        s.add_all([*nba.values(), *nfl.values()])
        await s.flush()
        for i, g in enumerate(past):      # the star's usual minutes (the stat board drops him while Out)
            s.add(M.NbaPlayerGamelog(player_id=nba["star"].id, game_id=g.id, season="2026", game_date=(now - timedelta(days=i + 1)).date(),
                                     team_abbr="AAA", opp_abbr="BBB", is_home=True, minutes=34, pts=25, reb=5, oreb=1, ast=5, fg3m=2,
                                     stl=1, blk=0, tov=2, fga=18, fta=5))
        base_min = {"p1": 30, "p2": 28, "p3": 24, "p4": 20, "p5": 14, "p6": 10, "p7": 8}
        for n, m in base_min.items():
            for stat, mean in (("minutes", m), ("pts", m * 0.5)):
                s.add(M.StatProjection(sport="NBA", game_id=nba_game.id, player_id=nba[n].id, stat=stat, mean=mean, lo=mean * 0.6,
                                       hi=mean * 1.4, model_mean=mean, sample=10))
        usage = {"wr1": (40, 0, "WR"), "wr2": (30, 0, "WR"), "te1": (20, 0, "TE"), "rb1": (10, 60, "RB")}
        for wk in range(6, 10):
            for n, (t, c, pos) in usage.items():
                await s.execute(text("""INSERT INTO nfl_usage_weekly (season, week, team_abbr, name_key, position, targets, carries)
                                        VALUES (2026, :w, 'CIN', :n, :p, :t, :c)"""), {"w": wk, "n": n, "p": pos, "t": t, "c": c})
        for n in ("wr2", "te1", "rb1"):
            for stat in ("targets", "rec_yds", "carries"):
                s.add(M.StatProjection(sport="NFL", game_id=nfl_game.id, player_id=nfl[n].id, stat=stat, mean=10.0, lo=5, hi=15, model_mean=10.0, sample=8))
        for pl, st in ((nba["star"], "Out"), (nba["p2"], "Questionable"), (nfl["wr1"], "Out")):
            s.add(M.InjuryReport(sport=pl.sport, player_id=pl.id, player_name_raw=pl.full_name, status=st, source="test"))
        await s.execute(text("""INSERT INTO nba_live_lineup (nba_game_id, nba_person_id, name_key, status, game_status, updated_at)
                                VALUES ('x', 1, 'p2', 'ACTIVE', 1, :u), ('x', 2, 'p6', 'INACTIVE', 1, :u)"""), {"u": now})
    async with session_scope() as s:
        before = (await s.execute(text("SELECT game_id, player_id, stat, mean, lo, hi FROM stat_projections ORDER BY 1, 2, 3"))).all()
    n_nba, n_nfl = await adjust.build_adjusted("NBA"), await adjust.build_adjusted("NFL")
    async with session_scope() as s:
        after = (await s.execute(text("SELECT game_id, player_id, stat, mean, lo, hi FROM stat_projections ORDER BY 1, 2, 3"))).all()
        adjusted = (await s.execute(text("""SELECT p.full_name, a.sport, a.stat, a.original_mean, a.adjusted_mean, a.factor, a.p_active, a.status,
                                            a.source FROM adjusted_stat_projections a JOIN players p ON p.id = a.player_id"""))).all()
        await s.execute(text("UPDATE stat_projections SET actual = 14.0 WHERE stat = 'pts' AND player_id = :p"), {"p": nba["p1"].id})
    report = await adjust.compare_report("NBA")
    await engine.dispose()
    return before, after, adjusted, n_nba, n_nfl, report


def test_adjustment_layer_against_postgres():
    before, after, adjusted, n_nba, n_nfl, report = asyncio.run(_adjust_scenario())
    assert before == after                                                     # original projections untouched
    nba = {(r[0], r[2]): r for r in adjusted if r[1] == "NBA"}
    nfl = {(r[0], r[2]): r for r in adjusted if r[1] == "NFL"}
    assert nba[("p1", "minutes")][5] > 1.0 and nba[("p1", "pts")][3] == 15.0
    assert nba[("p1", "pts")][4] == pytest.approx(15.0 * nba[("p1", "pts")][5], abs=1e-2)
    assert nba[("p1", "minutes")][8] == "injury_report"
    assert nba[("p2", "pts")][8] == "live_lineup" and nba[("p2", "pts")][5] > 1.0   # confirmed active overrides Questionable
    assert not any(n == "p6" for n, _ in nba)                                  # live INACTIVE: out, no adjusted row
    assert not any(n == "star" for n, _ in nba)
    assert ("wr2", "targets") in nfl and ("te1", "rec_yds") in nfl and nfl[("wr2", "targets")][5] > 1.0
    assert all(1.0 < r[5] <= 1.4 for r in nfl.values())                        # conservative NFL clip
    assert all(r[0] in ("wr2", "te1") or r[2] != "targets" or r[5] > 1.0 for r in nfl.values())
    assert not any(n == "rb1" and s == "carries" for n, s in nfl)              # nobody's carries changed
    assert n_nba == len(nba) and n_nfl == len(nfl)
    rep = {r["stat"]: r for r in report}
    assert rep["pts"]["n"] == 1 and rep["pts"]["mae_original"] == pytest.approx(1.0)       # |15 - 14|
    assert rep["pts"]["mae_adjusted"] == pytest.approx(abs(nba[("p1", "pts")][4] - 14.0))
