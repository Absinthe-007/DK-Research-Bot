"""
Rest metrics for upcoming games, computed from the games table (full season schedules come
from ESPN, so each team's previous game is always known).

  days_rest = calendar days between games minus 1  (0 = NBA back-to-back)
  NBA 3-in-4: this game is the 3rd in a 4-night window
  NFL off_bye: >= 13 days; short_week: <= 4 days (Thursday games)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from app.db.base import session_scope, upsert
from app.db.models import TeamRest
from app.ingestion.common import tracked

SQL = """
WITH tg AS (
  SELECT g.id game_id, g.sport, t.team_id, (g.start_time - interval '5 hours')::date d, g.start_time
  FROM games g CROSS JOIN LATERAL (VALUES (g.home_team_id), (g.away_team_id)) t(team_id)
  WHERE g.season_type IN (2,3)
), w AS (
  SELECT tg.*, LAG(d) OVER (PARTITION BY team_id ORDER BY start_time) prev_d,
         LAG(d, 2) OVER (PARTITION BY team_id ORDER BY start_time) prev2_d
  FROM tg
)
SELECT game_id, team_id, sport, (d - prev_d) - 1 AS days_rest, (d - prev2_d) <= 3 AS three_in_four
FROM w WHERE start_time BETWEEN :lo AND :hi
"""


@tracked("rest")
async def sync_rest(days_ahead: int = 8) -> int:
    now = datetime.now(timezone.utc)
    async with session_scope() as s:
        rows = (await s.execute(text(SQL), {"lo": now - timedelta(hours=6), "hi": now + timedelta(days=days_ahead)})).all()
        out = []
        for r in rows:
            dr = r.days_rest
            out.append(dict(game_id=r.game_id, team_id=r.team_id, days_rest=dr,
                            is_b2b=r.sport == "NBA" and dr == 0, is_3in4=bool(r.sport == "NBA" and r.three_in_four),
                            off_bye=bool(r.sport == "NFL" and dr is not None and dr >= 12),
                            short_week=bool(r.sport == "NFL" and dr is not None and dr <= 3)))
        return await upsert(s, TeamRest, out, ["game_id", "team_id"])
