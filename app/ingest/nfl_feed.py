"""
NFL feed via nflreadpy (nflverse): snap counts, play-by-play targets / carries, weekly injury reports and
depth charts. Loaders return polars frames; the `build_*` functions are pure (frame in, row dicts out) so they
can be unit-tested with tiny hand-made frames, and `sync_*` persist into the additive tables in
`app/db/models_ext.py`. nflverse is a GitHub-hosted dataset, so it works from Actions runners.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import delete

from app.db.base import session_scope, upsert
from app.db.models_ext import NflDepthCurrent, NflInjuryFeed, NflUsageWeekly
from app.ingestion.common import tracked
from app.ingestion.nfl_stats import NFLVERSE_TO_ESPN
from app.utils.names import name_key

log = logging.getLogger(__name__)


def current_season(today: datetime | None = None) -> int:
    t = today or datetime.now(timezone.utc)
    return t.year if t.month >= 3 else t.year - 1


def _team(t) -> str | None:
    return None if t is None else NFLVERSE_TO_ESPN.get(str(t), str(t))


def _rows(df) -> list[dict]:
    """polars or pandas frame -> list of dicts (None for nulls)."""
    if hasattr(df, "to_dicts"):
        return df.to_dicts()
    return [{k: (None if v != v else v) for k, v in r.items()} for r in df.to_dict("records")]


def _num(v):
    return None if v is None or v != v else v


# ------------------------------------------------------------------ pure builders
def build_pbp_usage(pbp, players) -> dict[tuple, dict]:
    """Targets and carries per (season, week, team, gsis_id) from play-by-play.
    `players` maps gsis_id -> display name (from nfl.load_players)."""
    out: dict[tuple, dict] = {}
    for r in _rows(pbp):
        if r.get("season_type") not in (None, "REG") or not r.get("week"):
            continue
        team = _team(r.get("posteam"))
        if not team:
            continue
        if r.get("pass_attempt") == 1 and r.get("receiver_player_id"):
            gid, field = r["receiver_player_id"], "targets"
        elif r.get("rush_attempt") == 1 and r.get("rusher_player_id"):
            gid, field = r["rusher_player_id"], "carries"
        else:
            continue
        key = (int(r["season"]), int(r["week"]), team, gid)
        rec = out.setdefault(key, {"targets": 0, "carries": 0, "name": players.get(gid)})
        rec[field] += 1
    return out


def build_usage_rows(snaps, pbp_usage: dict[tuple, dict]) -> list[dict]:
    """Merge snap counts (keyed by name) with PBP usage (keyed by gsis_id -> name) on (season, week, team, name_key)."""
    rows: dict[tuple, dict] = {}
    for r in _rows(snaps):
        if r.get("game_type") not in (None, "REG") or r.get("week") is None:
            continue
        nk = name_key(r.get("player"))
        team = _team(r.get("team"))
        if not nk or not team:
            continue
        key = (int(r["season"]), int(r["week"]), team, nk)
        rows[key] = dict(season=key[0], week=key[1], team_abbr=team, name_key=nk, player_name=r.get("player"),
                         position=r.get("position"), gsis_id=None, offense_snaps=_num(r.get("offense_snaps")),
                         offense_pct=_num(r.get("offense_pct")), targets=None, carries=None)
    for (season, week, team, gsis), u in pbp_usage.items():
        nk = name_key(u.get("name") or "")
        if not nk:
            continue
        rec = rows.setdefault((season, week, team, nk), dict(
            season=season, week=week, team_abbr=team, name_key=nk, player_name=u.get("name"), position=None,
            gsis_id=None, offense_snaps=None, offense_pct=None, targets=None, carries=None))
        rec["gsis_id"] = gsis
        rec["targets"] = (rec["targets"] or 0) + u["targets"]
        rec["carries"] = (rec["carries"] or 0) + u["carries"]
    return list(rows.values())


def build_injury_rows(df) -> list[dict]:
    rows: dict[tuple, dict] = {}
    for r in _rows(df):
        nk = name_key(r.get("full_name"))
        team = _team(r.get("team"))
        if not nk or not team or r.get("week") is None:
            continue
        key = (int(r["season"]), int(r["week"]), team, nk)
        rows[key] = dict(season=key[0], week=key[1], team_abbr=team, name_key=nk, player_name=r.get("full_name"),
                         position=r.get("position"), gsis_id=r.get("gsis_id"),
                         report_status=r.get("report_status"), report_injury=r.get("report_primary_injury"),
                         practice_status=r.get("practice_status"), practice_injury=r.get("practice_primary_injury"))
    return list(rows.values())


def build_depth_rows(df) -> list[dict]:
    """Latest snapshot only. Handles both the pre-2025 and the 2025+ nflverse depth chart schemas."""
    data = _rows(df)
    if not data:
        return []
    ts_key = "dt" if "dt" in data[0] else None
    latest = max((r[ts_key] for r in data if r.get(ts_key) is not None), default=None) if ts_key else None
    rows: dict[tuple, dict] = {}
    for r in data:
        if ts_key and latest is not None and r.get(ts_key) != latest:
            continue
        name = r.get("player_name") or r.get("full_name")
        pos = r.get("pos_abb") or r.get("position")
        rank = r.get("pos_rank") or r.get("depth_team")
        team = _team(r.get("team") or r.get("club_code"))
        nk = name_key(name)
        if not (nk and pos and team and rank is not None):
            continue
        rows[(team, pos, int(rank), nk)] = dict(
            team_abbr=team, pos_abb=pos, pos_rank=int(rank), name_key=nk, player_name=name,
            espn_id=None if r.get("espn_id") is None else str(r.get("espn_id")), gsis_id=r.get("gsis_id"),
            snapshot_at=_as_dt(latest))
    return list(rows.values())


def _as_dt(v):
    if v is None:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


# ------------------------------------------------------------------ loaders (network)
async def _load(name: str, **kw):
    import nflreadpy as nfl

    return await asyncio.to_thread(getattr(nfl, name), **kw)


async def load_players_map() -> dict[str, str]:
    df = await _load("load_players")
    return {r["gsis_id"]: r["display_name"] for r in _rows(df) if r.get("gsis_id") and r.get("display_name")}


# ------------------------------------------------------------------ persistence
@tracked("nfl_feed_usage")
async def sync_usage(season: int | None = None) -> int:
    season = season or current_season()
    snaps = await _load("load_snap_counts", seasons=[season])
    pbp = await _load("load_pbp", seasons=[season])
    rows = build_usage_rows(snaps, build_pbp_usage(pbp, await load_players_map()))
    async with session_scope() as s:
        return await upsert(s, NflUsageWeekly, rows, ["season", "week", "team_abbr", "name_key"])


@tracked("nfl_feed_injuries")
async def sync_injuries(season: int | None = None) -> int:
    rows = build_injury_rows(await _load("load_injuries", seasons=[season or current_season()]))
    async with session_scope() as s:
        return await upsert(s, NflInjuryFeed, rows, ["season", "week", "team_abbr", "name_key"])


@tracked("nfl_feed_depth")
async def sync_depth_charts(season: int | None = None) -> int:
    rows = build_depth_rows(await _load("load_depth_charts", seasons=[season or current_season()]))
    if not rows:
        return 0
    async with session_scope() as s:                  # "current" table: replace wholesale, atomically
        await s.execute(delete(NflDepthCurrent))
        return await upsert(s, NflDepthCurrent, rows, ["team_abbr", "pos_abb", "pos_rank", "name_key"])


async def sync_all(season: int | None = None) -> dict[str, int]:
    """Each feed fails independently so one nflverse hiccup doesn't drop the others."""
    out: dict[str, int] = {}
    for name, fn in (("usage", sync_usage), ("injuries", sync_injuries), ("depth", sync_depth_charts)):
        try:
            out[name] = await fn(season)
        except Exception as e:
            log.warning("nfl_feed %s failed: %s", name, e)
            out[name] = -1
    return out
