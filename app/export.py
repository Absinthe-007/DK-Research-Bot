"""
Export the GitHub Pages data (docs/data/board.json): expected stat lines for every
player in upcoming games, game context (weather, rest, injuries) and projection accuracy.
No prices on the main statboard; full pick ledger, CLV, and steam deltas in `record`.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import text

from app.db.base import session_scope
from app.engine.odds_math import devig_shin, implied_prob
from app.engine.record import breakdown
from app.engine.scanner import jsonable
from app.engine.statboard import NBA_STATS, NFL_STATS

log = logging.getLogger(__name__)
OUT = Path(__file__).resolve().parent.parent / "docs" / "data"

GAMES_SQL = """
SELECT g.id, g.start_time, g.week, ht.abbr home, at.abbr away,
       ht.name home_name, at.name away_name, v.name venue, v.indoor, v.city,
       (SELECT row_to_json(w) FROM (SELECT temp_f, wind_mph, wind_gust_mph, precip_prob,
            short_forecast, source, captured_at FROM weather_forecasts wf
            WHERE wf.game_id=g.id ORDER BY captured_at DESC LIMIT 1) w) weather,
       (SELECT json_agg(r) FROM (SELECT t.abbr, tr.days_rest, tr.is_b2b, tr.is_3in4,
            tr.off_bye, tr.short_week FROM team_rest tr JOIN teams t ON t.id=tr.team_id
            WHERE tr.game_id=g.id) r) rest
FROM games g
JOIN teams ht ON ht.id=g.home_team_id
JOIN teams at ON at.id=g.away_team_id
LEFT JOIN venues v ON v.id=g.venue_id
WHERE g.sport=:sp AND g.start_time > now() - interval '4 hours'
  AND g.start_time < now() + interval '8 days'
  AND g.season_type IN (2,3)
ORDER BY g.start_time
"""

PROJ_SQL = """
SELECT sp.game_id, sp.player_id, sp.stat, sp.mean, sp.lo, sp.hi,
       sp.anchored, sp.last5, sp.season_avg, sp.sample,
       p.full_name, p.position, p.team_abbr, p.headshot
FROM stat_projections sp
JOIN players p ON p.id=sp.player_id
JOIN games g ON g.id=sp.game_id
WHERE sp.sport=:sp AND g.start_time > now() - interval '4 hours' AND NOT g.completed
"""

INJ_SQL = """
SELECT DISTINCT ON (COALESCE(player_id::text, player_name_raw))
       player_id, player_name_raw player, team_abbr team,
       status, injury, comment, return_date
FROM injury_reports
WHERE sport=:sp AND captured_at > now() - interval '36 hours'
  AND status IN ('Out','Doubtful','Questionable','Day-To-Day','IR','PUP','Suspended')
ORDER BY COALESCE(player_id::text, player_name_raw), captured_at DESC
"""

ACC_SQL = """
SELECT stat, COUNT(actual) n, COUNT(*) listed,
       AVG(ABS(actual-mean)) mae,
       AVG(actual-mean) bias,
       AVG(CASE WHEN actual IS NULL THEN NULL WHEN actual BETWEEN lo AND hi THEN 1.0 ELSE 0.0 END) in_range,
       AVG(mean) FILTER (WHERE actual IS NOT NULL) avg_proj,
       AVG(actual) avg_actual,
       COUNT(actual)::float / NULLIF(COUNT(*), 0) played_pct
FROM stat_projections sp JOIN games g ON g.id=sp.game_id
WHERE sp.sport=:sp AND g.completed AND g.boxscore_loaded
  AND g.start_time > now() - make_interval(days => :days)
GROUP BY stat
"""

# Coverage: of everyone who actually got on the floor/field in the graded games, how many had a projection.
COVER_SQL = """
WITH gg AS (SELECT g.id FROM games g
            WHERE g.sport=:sp AND g.completed AND g.boxscore_loaded
              AND g.start_time > now() - make_interval(days => :days)
              AND EXISTS (SELECT 1 FROM stat_projections sp WHERE sp.game_id=g.id)),
     played AS (SELECT game_id, player_id FROM {box}
                WHERE game_id IN (SELECT id FROM gg) {played_filter})
SELECT (SELECT COUNT(*) FROM gg) games,
       COUNT(*) played,
       COUNT(*) FILTER (WHERE EXISTS (
           SELECT 1 FROM stat_projections sp
           WHERE sp.game_id=played.game_id AND sp.player_id=played.player_id)) projected
FROM played
"""

RECENT_SQL = """
SELECT g.start_time, ht.abbr home, at.abbr away, p.full_name player, p.team_abbr team,
       sp.stat, sp.mean, sp.lo, sp.hi, sp.actual
FROM stat_projections sp
JOIN players p ON p.id=sp.player_id
JOIN games g ON g.id=sp.game_id
JOIN teams ht ON ht.id=g.home_team_id
JOIN teams at ON at