"""
Command-line entry point used by GitHub Actions (and locally).

    python -m app.pipeline plan                       # which sports are due? (no DB) -> JSON + \$GITHUB_OUTPUT
    python -m app.pipeline refresh --sport NBA        # lines, props, injuries, rest, weather -> scan -> export
    python -m app.pipeline nightly                    # finals, box scores, grading, rosters, prune -> export
    python -m app.pipeline backfill --sport NBA --start 2025-10-21 --end 2026-04-12
    python -m app.pipeline export
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import text

from app import planner
from app.db.base import session_scope
from app.engine import statboard
from app.engine.grade import grade
from app.engine.scanner import scan
from app.export import export
from app.ingestion import espn, injuries, rest, weather

log = logging.getLogger("pipeline")


async def _ensure_teams(sport: str) -> None:
    async with session_scope() as s:
        n = (await s.execute(text("SELECT COUNT(*) FROM teams WHERE sport=:s"), {"s": sport})).scalar()
    if not n:
        await espn.sync_teams(sport)
        await espn.sync_rosters(sport)


async def _safe(coro_fn, *a, **kw):
    """One failing source must not sink the whole run."""
    try:
        return await coro_fn(*a, **kw)
    except Exception as e:
        log.warning("%s failed: %s", getattr(coro_fn, "__name__", coro_fn), e)
        return None


async def refresh(sport: str) -> None:
    today = datetime.now(timezone.utc).date()
    await _ensure_teams(sport)
    await _safe(espn.sync_schedule, sport, today - timedelta(days=1), today + timedelta(days=7))
    # Catch any late-finishing games from last night so box scores & grades are fresh before scanning
    await _safe(espn.sync_boxscores, sport, limit=25)
    await _safe(grade, sport)
    await _safe(statboard.grade, sport)

    await _safe(injuries.sync_injuries, sport)
    await _safe(rest.sync_rest)
    if sport == "NFL":
        await _safe(weather.sync_weather)
    await _safe(espn.sync_props, sport, hours_ahead=30 if sport == "NBA" else 80)
    await _safe(scan, sport)          # market anchors (prices are used internally only)
    await _safe(statboard.build, sport)  # expected box-score line for every player

    state = planner.load_state()
    state[f"last_refresh_{sport}"] = datetime.now(timezone.utc).isoformat()
    planner.save_state(state)


async def nightly() -> None:
    today = datetime.now(timezone.utc).date()
    for sport in ("NBA", "NFL"):
        await _ensure_teams(sport)
        await _safe(espn.sync_schedule, sport, today - timedelta(days=4), today + timedelta(days=10))
        await _safe(espn.sync_boxscores, sport)
        await _safe(grade, sport)
        await _safe(statboard.grade, sport)
        await _safe(espn.sync_rosters, sport)   # every night: trades, signings, releases
    if today.weekday() in (1, 4):               # NFL enrichment after MNF / TNF
        from app.ingestion.nfl_stats import enrich
        await _safe(enrich, today.year if today.month >= 3 else today.year - 1)
    await _safe(prune)


async def prune() -> None:
    """Keep the database small enough to round-trip through a GitHub release asset."""
    async with session_scope() as s:
        # 1. Remove old edges first so foreign keys on projections / prop_snapshots never block pruning
        await s.execute(text("""
            DELETE FROM edges e USING projections p, games g
            WHERE e.projection_id = p.id AND p.game_id = g.id
              AND g.start_time < now() - interval '2 days'
        """))
        # 2. Remove old projections
        await s.execute(text("""
            DELETE FROM projections p USING games g
            WHERE p.game_id = g.id AND g.start_time < now() - interval '2 days'
        """))
        # 3. Prune intermediate prop_snapshots older than 3 days, keeping the closing pre-game snapshot
        await s.execute(text("""
            DELETE FROM prop_snapshots ps USING games g
            WHERE ps.game_id = g.id
              AND g.start_time < now() - interval '3 days'
              AND ps.id NOT IN (
                  SELECT DISTINCT ON (c.game_id, c.player_id, c.market_key, c.kind, c.side, c.line) c.id
                  FROM prop_snapshots c
                  JOIN games gg ON gg