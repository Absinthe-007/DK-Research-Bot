"""
Export the GitHub Pages data (docs/data/board.json): expected stat lines for every player in
upcoming games, game context (weather, rest, injuries) and projection accuracy. No prices.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import text

from app.db.base import session_scope
from app.engine.record import breakdown
from app.engine.scanner import jsonable
from app.engine.statboard import NBA_STATS, NFL_STATS

log = logging.getLogger(__name__)
OUT = Path(__file__).resolve().parent.parent / "docs" / "data"

GAMES_SQL = """
SELECT g.id, g.start_time, g.week, ht.abbr home, at.abbr away, ht.name home_name, at.name away_name,
       v.name venue, v.indoor, v.city,
       (SELECT row_to_json(w) FROM (SELECT temp_f, wind_mph, wind_gust_mph, precip_prob, short_forecast, source, captured_at
               FROM weather_forecasts wf WHERE wf.game_id=g.id ORDER BY captured_at DESC LIMIT 1) w) weather,
       (SELECT json_agg(r) FROM (SELECT t.abbr, tr.days_rest, tr.is_b2b, tr.is_3in4, tr.off_bye, tr.short_week
               FROM team_rest tr JOIN teams t ON t.id=tr.team_id WHERE tr.game_id=g.id) r) rest
FROM games g JOIN teams ht ON ht.id=g.home_team_id JOIN teams at ON at.id=g.away_team_id
LEFT JOIN venues v ON v.id=g.venue_id
WHERE g.sport=:sp AND g.start_time > now() - interval '4 hours' AND g.start_time < now() + interval '8 days'
  AND g.season_type IN (2,3)
ORDER BY g.start_time
"""

PROJ_SQL = """
SELECT sp.game_id, sp.player_id, sp.stat, sp.mean, sp.lo, sp.hi, sp.anchored, sp.last5, sp.season_avg, sp.sample,
       p.full_name, p.position, p.team_abbr, p.headshot
FROM stat_projections sp JOIN players p ON p.id=sp.player_id JOIN games g ON g.id=sp.game_id
WHERE sp.sport=:sp AND g.start_time > now() - interval '4 hours' AND NOT g.completed
"""

INJ_SQL = """
SELECT DISTINCT ON (player_name_raw) player_id, player_name_raw player, team_abbr team, status, injury, comment, return_date
FROM injury_reports WHERE sport=:sp AND captured_at > now() - interval '36 hours'
  AND status IN ('Out','Doubtful','Questionable','Day-To-Day','IR','PUP','Suspended')
ORDER BY player_name_raw, captured_at DESC
"""

ACC_SQL = """
SELECT stat, COUNT(actual) n, COUNT(*) listed, AVG(ABS(actual-mean)) mae, AVG(actual-mean) bias,
       AVG(CASE WHEN actual IS NULL THEN NULL WHEN actual BETWEEN lo AND hi THEN 1.0 ELSE 0.0 END) in_range,
       AVG(mean) FILTER (WHERE actual IS NOT NULL) avg_proj, AVG(actual) avg_actual,
       COUNT(actual)::float / NULLIF(COUNT(*), 0) played_pct   -- listed players who actually got in the game
FROM stat_projections sp JOIN games g ON g.id=sp.game_id
WHERE sp.sport=:sp AND g.completed AND g.boxscore_loaded AND g.start_time > now() - make_interval(days => :days)
GROUP BY stat
"""

# Coverage: of everyone who actually got on the floor/field in the graded games, how many had a projection.
COVER_SQL = """
WITH gg AS (SELECT g.id FROM games g WHERE g.sport=:sp AND g.completed AND g.boxscore_loaded
              AND g.start_time > now() - make_interval(days => :days)
              AND EXISTS (SELECT 1 FROM stat_projections sp WHERE sp.game_id=g.id)),
     played AS (SELECT game_id, player_id FROM {box} WHERE game_id IN (SELECT id FROM gg) {played_filter})
SELECT (SELECT COUNT(*) FROM gg) games, COUNT(*) played,
       COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM stat_projections sp
                         WHERE sp.game_id=played.game_id AND sp.player_id=played.player_id)) projected
FROM played
"""

RECENT_SQL = """
SELECT g.start_time, ht.abbr home, at.abbr away, p.full_name player, p.team_abbr team, sp.stat, sp.mean, sp.lo, sp.hi, sp.actual
FROM stat_projections sp JOIN players p ON p.id=sp.player_id JOIN games g ON g.id=sp.game_id
JOIN teams ht ON ht.id=g.home_team_id JOIN teams at ON at.id=g.away_team_id
WHERE sp.sport=:sp AND sp.actual IS NOT NULL AND sp.stat = ANY(:stats)
ORDER BY g.start_time DESC, sp.mean DESC LIMIT 300
"""


# DraftKings main two-way line for every projected player/market, with the model's probability for each side.
LINES_SQL = """
SELECT pr.game_id, pr.player_id, pr.market_key, e.side, e.line, e.price, e.model_prob, e.edge, e.flagged, ps.price_estimated
FROM edges e JOIN projections pr ON pr.id=e.projection_id JOIN prop_snapshots ps ON ps.id=e.prop_snapshot_id
JOIN games g ON g.id=pr.game_id
WHERE e.sport=:sp AND e.kind='ou' AND g.start_time > now() - interval '4 hours' AND NOT g.completed
"""

# Latest DraftKings game lines (spread / total / moneyline) per upcoming game.
GAME_LINES_SQL = """
SELECT DISTINCT ON (o.game_id, o.market_type, o.side) o.game_id, o.market_type, o.side, o.price, o.point
FROM odds_snapshots o JOIN games g ON g.id=o.game_id
WHERE g.sport=:sp AND g.start_time > now() - interval '4 hours' AND NOT g.completed
ORDER BY o.game_id, o.market_type, o.side, o.captured_at DESC
"""

# Graded record of every play the model flagged (frozen at its last pre-game price).
PICKS_SUMMARY_SQL = """
SELECT sport, COUNT(*) FILTER (WHERE result IN ('win','loss')) n, COUNT(*) FILTER (WHERE result='win') wins,
       COUNT(*) FILTER (WHERE result='loss') losses, COUNT(*) FILTER (WHERE result IN ('push','void')) voids,
       COALESCE(SUM(profit_units) FILTER (WHERE result IN ('win','loss')), 0) units,
       AVG(model_prob) FILTER (WHERE result IN ('win','loss')) avg_model_prob
FROM picks GROUP BY sport
"""

PICKS_RECENT_SQL = """
SELECT pk.sport, g.start_time, p.full_name player, p.team_abbr team, pk.market_key, pk.side, pk.line, pk.price,
       pk.model_prob, pk.actual, pk.result, pk.profit_units
FROM picks pk JOIN players p ON p.id=pk.player_id JOIN games g ON g.id=pk.game_id
WHERE pk.result IN ('win','loss') ORDER BY g.start_time DESC, pk.id DESC LIMIT 40
"""

# Every graded win/loss play with the fields the breakdowns need (calibration, market type, CLV).
PICKS_GRADED_SQL = """
SELECT pk.sport, pk.game_id, pk.player_id, pk.market_key, pk.kind, pk.side, pk.price, pk.first_price, pk.close_price,
       pk.model_prob, pk.edge, pk.result, pk.profit_units
FROM picks pk JOIN games g ON g.id=pk.game_id
WHERE pk.result IN ('win','loss') AND g.start_time > now() - interval '30 days'
"""

# Plays currently flagged for upcoming games (the model's biggest disagreements with DraftKings).
LEANS_SQL = """
SELECT pk.sport, g.id game_id, g.start_time, p.full_name player, p.team_abbr team, pk.market_key, pk.kind, pk.side, pk.line,
       pk.price, pk.model_prob, pk.edge, pk.proj_mean
FROM picks pk JOIN players p ON p.id=pk.player_id JOIN games g ON g.id=pk.game_id
WHERE pk.active AND pk.result IS NULL AND g.start_time > now() AND NOT g.completed
ORDER BY pk.edge DESC LIMIT 60
"""


def _implied(american) -> float | None:
    if american is None:
        return None
    return 100 / (american + 100) if american > 0 else -american / (-american + 100)


def win_info(L: dict | None) -> dict | None:
    """Projected winner from DraftKings' moneyline (margin removed) and projected score from spread + total."""
    if not L:
        return None
    ph, pa = _implied(L.get("ml_home")), _implied(L.get("ml_away"))
    if ph is None or pa is None:
        return None
    home = ph / (ph + pa)
    out = {"home": round(home, 3), "away": round(1 - home, 3), "pick": "home" if home >= 0.5 else "away"}
    if L.get("total") is not None and L.get("spread") is not None:
        out["score_home"] = round((L["total"] - L["spread"]) / 2, 1)
        out["score_away"] = round((L["total"] + L["spread"]) / 2, 1)
    return out


WINNERS_SQL = """
WITH ml AS (SELECT DISTINCT ON (o.game_id, o.side) o.game_id, o.side, o.price
            FROM odds_snapshots o JOIN games g ON g.id=o.game_id
            WHERE o.market_type='h2h' AND o.captured_at <= g.start_time AND g.completed AND g.sport=:sp
            ORDER BY o.game_id, o.side, o.captured_at DESC)
SELECT g.id, g.start_time, g.home_score, g.away_score, ht.abbr home, at.abbr away,
       (SELECT price FROM ml WHERE game_id=g.id AND side='home') ml_home, (SELECT price FROM ml WHERE game_id=g.id AND side='away') ml_away
FROM games g JOIN teams ht ON ht.id=g.home_team_id JOIN teams at ON at.id=g.away_team_id
WHERE g.sport=:sp AND g.completed AND g.home_score IS NOT NULL AND g.season_type IN (2,3) AND g.start_time > now() - interval '60 days'
ORDER BY g.start_time DESC
"""


def _market_to_stat(defs: dict) -> dict:
    return {v[2]: k for k, v in defs.items() if v[2]}


def _game_lines(rows) -> dict:
    out: dict = {}
    for r in rows:
        d = out.setdefault(r["game_id"], {})
        if r["market_type"] == "totals" and r["side"] == "over":
            d["total"] = r["point"]
        elif r["market_type"] == "spreads" and r["side"] == "home":
            d["spread"] = r["point"]            # home spread: negative = home favored
        elif r["market_type"] == "h2h":
            d["ml_" + r["side"]] = r["price"]
    return out


async def export() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    board: dict = {"generated_at": datetime.now(timezone.utc).isoformat(), "sports": {}}
    async with session_scope() as s:
        for sp, defs in (("NBA", NBA_STATS), ("NFL", NFL_STATS)):
            games = [dict(r) for r in (await s.execute(text(GAMES_SQL), {"sp": sp})).mappings().all()]
            inj = [dict(r) for r in (await s.execute(text(INJ_SQL), {"sp": sp})).mappings().all()]
            status = {i["player_id"]: i for i in inj if i["player_id"]}
            players: dict[tuple, dict] = {}
            try:
                gl = _game_lines((await s.execute(text(GAME_LINES_SQL), {"sp": sp})).mappings().all())
                for g in games:
                    g["lines"] = gl.get(g["id"])
                    g["win"] = win_info(g["lines"])
            except Exception as e:                     # odds are a bonus: never block the board
                log.warning("game lines skipped: %s", e)
            lines: dict[tuple, dict] = {}
            try:
                m2s = _market_to_stat(defs)
                for r in (await s.execute(text(LINES_SQL), {"sp": sp})).mappings().all():
                    st_ = m2s.get(r["market_key"])
                    if not st_:
                        continue
                    slot = lines.setdefault((r["game_id"], r["player_id"]), {}).setdefault(st_, {"l": r["line"]})
                    if r["line"] != slot["l"]:
                        continue
                    k = "o" if r["side"] == "Over" else "u"
                    slot[k] = r["price"]
                    slot["p" + k] = round(float(r["model_prob"]), 3)
                    slot["est"] = int(bool(r["price_estimated"])) or slot.get("est", 0)
            except Exception as e:
                log.warning("prop lines skipped: %s", e)
            for r in (await s.execute(text(PROJ_SQL), {"sp": sp})).mappings().all():
                key = (r["game_id"], r["player_id"])
                if key not in players:
                    st = status.get(r["player_id"])
                    players[key] = dict(g=r["game_id"], id=r["player_id"], name=r["full_name"], pos=r["position"],
                                        team=r["team_abbr"], img=r["headshot"], n=r["sample"],
                                        status=st and st["status"], injury=st and st["injury"], s={})
                players[key]["s"][r["stat"]] = [r["mean"], r["lo"], r["hi"], int(bool(r["anchored"])),
                                                r["last5"], r["season_avg"]]
            for key, ln in lines.items():
                if key in players:
                    players[key]["ln"] = ln
            acc, cover = {}, {}
            # NFL box scores also list linemen/long snappers; only count positions we project.
            box, pf = (("nba_player_gamelogs", "AND minutes > 0") if sp == "NBA" else
                       ("nfl_player_games", "AND player_id IN (SELECT id FROM players WHERE sport='NFL' AND "
                        "position IN ('QB','RB','FB','WR','TE','PK','K','LB','CB','S','DT','DE','DB','NT','OLB','ILB','MLB','SS','FS'))"))
            for label, days in (("7d", 7), ("30d", 30), ("season", 400)):
                acc[label] = {r["stat"]: dict(r) for r in (await s.execute(text(ACC_SQL), {"sp": sp, "days": days})).mappings().all()}
                cover[label] = dict((await s.execute(text(COVER_SQL.format(box=box, played_filter=pf)),
                                                      {"sp": sp, "days": days})).mappings().one())
            key_stats = ["pts", "reb", "ast", "pra"] if sp == "NBA" else ["pass_yds", "rush_yds", "rec_yds", "receptions", "tackles", "sacks"]
            recent = [dict(r) for r in (await s.execute(text(RECENT_SQL), {"sp": sp, "stats": key_stats})).mappings().all()]
            board["sports"][sp] = jsonable({
                "labels": {k: v[0] for k, v in defs.items()} | ({"td_prob": "TD %"} if sp == "NFL" else {}),
                "games": games, "players": list(players.values()), "injuries": inj, "accuracy": acc, "coverage": cover, "recent": recent,
            })
        try:
            summ = [dict(r) for r in (await s.execute(text(PICKS_SUMMARY_SQL))).mappings().all()]
            leans = [dict(r) for r in (await s.execute(text(LEANS_SQL))).mappings().all()]
            board["record"] = jsonable({
                "summary": summ,
                "recent": [dict(r) for r in (await s.execute(text(PICKS_RECENT_SQL))).mappings().all()],
                "leans": leans,
                "breakdown": breakdown([dict(r) for r in (await s.execute(text(PICKS_GRADED_SQL))).mappings().all()]),
            })
        except Exception as e:
            log.warning("picks record skipped: %s", e)
        try:
            for sp in ("NBA", "NFL"):
                rows = [dict(r) for r in (await s.execute(text(WINNERS_SQL), {"sp": sp})).mappings().all()]
                graded = []
                for r in rows:
                    w = win_info({"ml_home": r["ml_home"], "ml_away": r["ml_away"]})
                    if not w or r["home_score"] == r["away_score"]:
                        continue
                    won = "home" if r["home_score"] > r["away_score"] else "away"
                    graded.append({"date": r["start_time"], "home": r["home"], "away": r["away"], "hs": r["home_score"], "as": r["away_score"],
                                   "pick": r[w["pick"]], "prob": w[w["pick"]], "ok": w["pick"] == won})
                board["sports"][sp]["winners"] = jsonable({
                    "n": len(graded), "correct": sum(g["ok"] for g in graded),
                    "avg_prob": (sum(g["prob"] for g in graded) / len(graded)) if graded else None, "recent": graded[:25]})
        except Exception as e:
            log.warning("winner record skipped: %s", e)
        board["last_runs"] = jsonable([dict(r) for r in (await s.execute(text("""
            SELECT DISTINCT ON (job) job, finished_at, ok, rows FROM ingest_runs ORDER BY job, started_at DESC"""
        ))).mappings().all()])
    path = OUT / "board.json"
    path.write_text(json.dumps(board, separators=(",", ":"), default=str))
    return path


