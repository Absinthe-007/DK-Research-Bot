"""
Export the GitHub Pages data (docs/data/board.json): expected stat lines for every player in
upcoming games, game context (weather, rest, injuries) and projection accuracy. No prices.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import text

from app.db.base import session_scope
from app.engine.scanner import jsonable
from app.engine.statboard import NBA_STATS, NFL_STATS

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
SELECT stat, COUNT(*) n, AVG(ABS(actual-mean)) mae, AVG(actual-mean) bias,
       AVG(CASE WHEN actual BETWEEN lo AND hi THEN 1.0 ELSE 0.0 END) in_range,
       AVG(mean) avg_proj, AVG(actual) avg_actual
FROM stat_projections sp JOIN games g ON g.id=sp.game_id
WHERE sp.sport=:sp AND sp.actual IS NOT NULL AND g.start_time > now() - make_interval(days => :days)
GROUP BY stat
"""

RECENT_SQL = """
SELECT g.start_time, ht.abbr home, at.abbr away, p.full_name player, p.team_abbr team, sp.stat, sp.mean, sp.lo, sp.hi, sp.actual
FROM stat_projections sp JOIN players p ON p.id=sp.player_id JOIN games g ON g.id=sp.game_id
JOIN teams ht ON ht.id=g.home_team_id JOIN teams at ON at.id=g.away_team_id
WHERE sp.sport=:sp AND sp.actual IS NOT NULL AND sp.stat = ANY(:stats)
ORDER BY g.start_time DESC, sp.mean DESC LIMIT 300
"""


async def export() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    board: dict = {"generated_at": datetime.now(timezone.utc).isoformat(), "sports": {}}
    async with session_scope() as s:
        for sp, defs in (("NBA", NBA_STATS), ("NFL", NFL_STATS)):
            games = [dict(r) for r in (await s.execute(text(GAMES_SQL), {"sp": sp})).mappings().all()]
            inj = [dict(r) for r in (await s.execute(text(INJ_SQL), {"sp": sp})).mappings().all()]
            status = {i["player_id"]: i for i in inj if i["player_id"]}
            players: dict[tuple, dict] = {}
            for r in (await s.execute(text(PROJ_SQL), {"sp": sp})).mappings().all():
                key = (r["game_id"], r["player_id"])
                if key not in players:
                    st = status.get(r["player_id"])
                    players[key] = dict(g=r["game_id"], id=r["player_id"], name=r["full_name"], pos=r["position"],
                                        team=r["team_abbr"], img=r["headshot"], n=r["sample"],
                                        status=st and st["status"], injury=st and st["injury"], s={})
                players[key]["s"][r["stat"]] = [r["mean"], r["lo"], r["hi"], int(bool(r["anchored"])),
                                                r["last5"], r["season_avg"]]
            acc = {}
            for label, days in (("7d", 7), ("30d", 30), ("season", 400)):
                acc[label] = {r["stat"]: dict(r) for r in (await s.execute(text(ACC_SQL), {"sp": sp, "days": days})).mappings().all()}
            key_stats = ["pts", "reb", "ast", "pra"] if sp == "NBA" else ["pass_yds", "rush_yds", "rec_yds", "receptions", "tackles", "sacks"]
            recent = [dict(r) for r in (await s.execute(text(RECENT_SQL), {"sp": sp, "stats": key_stats})).mappings().all()]
            board["sports"][sp] = jsonable({
                "labels": {k: v[0] for k, v in defs.items()} | ({"td_prob": "TD %"} if sp == "NFL" else {}),
                "games": games, "players": list(players.values()), "injuries": inj, "accuracy": acc, "recent": recent,
            })
        board["last_runs"] = jsonable([dict(r) for r in (await s.execute(text("""
            SELECT DISTINCT ON (job) job, finished_at, ok, rows FROM ingest_runs ORDER BY job, started_at DESC"""
        ))).mappings().all()])
    path = OUT / "board.json"
    path.write_text(json.dumps(board, separators=(",", ":"), default=str))
    return path


