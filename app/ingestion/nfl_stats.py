"""
OPTIONAL NFL enrichment from nflverse (free GitHub-hosted datasets): snap %, air yards and
WOPR, joined onto ESPN box-score rows by (name, team, week). If nflverse is unavailable the
pipeline simply runs without these columns.
"""
from __future__ import annotations

import asyncio
import logging

import pandas as pd
from sqlalchemy import text

from app.db.base import session_scope
from app.ingestion.common import tracked
from app.utils.names import name_key

log = logging.getLogger(__name__)
NFLVERSE_TO_ESPN = {"LA": "LAR", "WAS": "WSH"}


async def _load(fn, **kw) -> pd.DataFrame:
    df = await asyncio.to_thread(fn, **kw)
    return df.to_pandas() if hasattr(df, "to_pandas") else df


@tracked("nfl_enrich")
async def enrich(season: int) -> int:
    import nflreadpy as nfl
    ps = await _load(nfl.load_player_stats, seasons=[season], summary_level="week")
    snaps = await _load(nfl.load_snap_counts, seasons=[season])
    ps = ps[ps["player_id"].notna()].copy()
    ps["nk"] = ps["player_display_name"].map(name_key)
    ps["tm"] = ps["team"].map(lambda t: NFLVERSE_TO_ESPN.get(t, t))
    snaps["nk"] = snaps["player"].map(name_key)
    snaps["tm"] = snaps["team"].map(lambda t: NFLVERSE_TO_ESPN.get(t, t))
    m = ps.merge(snaps[["nk", "tm", "week", "offense_pct"]], on=["nk", "tm", "week"], how="left")
    rows = [dict(nk=r.nk, tm=r.tm, wk=int(r.week),
                 snap=None if pd.isna(r.offense_pct) else float(r.offense_pct),
                 ay=None if pd.isna(r.receiving_air_yards) else float(r.receiving_air_yards),
                 wopr=None if pd.isna(r.wopr) else float(r.wopr))
            for r in m.itertuples(index=False)]
    async with session_scope() as s:
        await s.execute(text("CREATE TEMP TABLE enr (nk text, tm text, wk int, snap float, ay float, wopr float) ON COMMIT DROP"))
        for i in range(0, len(rows), 3000):
            await s.execute(text("INSERT INTO enr VALUES (:nk,:tm,:wk,:snap,:ay,:wopr)"), rows[i:i + 3000])
        res = await s.execute(text("""
            UPDATE nfl_player_games g SET offense_snap_pct=e.snap, air_yards=e.ay, wopr=e.wopr
            FROM enr e JOIN players p ON p.name_key=e.nk AND p.sport='NFL'
            WHERE g.player_id=p.id AND g.team_abbr=e.tm AND g.week=e.wk AND g.season=:season"""), {"season": season})
        return res.rowcount or 0
