"""
Command-line entry point used by GitHub Actions (and locally).

    python -m app.pipeline plan                       # which sports are due? (no DB) -> JSON + $GITHUB_OUTPUT
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
    await espn.sync_schedule(sport, today - timedelta(days=1), today + timedelta(days=7))
    await _safe(injuries.sync_injuries, sport)
    await _safe(rest.sync_rest)
    if sport == "NFL":
        await _safe(weather.sync_weather)
    await espn.sync_props(sport, hours_ahead=30 if sport == "NBA" else 80)
    await scan(sport)
    state = planner.load_state()
    state[f"last_refresh_{sport}"] = datetime.now(timezone.utc).isoformat()
    planner.save_state(state)


async def nightly() -> None:
    today = datetime.now(timezone.utc).date()
    for sport in ("NBA", "NFL"):
        await _ensure_teams(sport)
        await espn.sync_schedule(sport, today - timedelta(days=4), today + timedelta(days=10))
        await espn.sync_boxscores(sport)
        await grade(sport)
        if today.weekday() == 1:          # Tuesday: refresh rosters (trades, signings)
            await _safe(espn.sync_rosters, sport)
    if today.weekday() in (1, 4):          # NFL enrichment after MNF / TNF
        from app.ingestion.nfl_stats import enrich
        await _safe(enrich, today.year if today.month >= 3 else today.year - 1)
    await prune()


async def prune() -> None:
    """Keep the database small enough to round-trip through a GitHub release asset."""
    async with session_scope() as s:
        await s.execute(text("""
            DELETE FROM prop_snapshots ps USING games g
            WHERE ps.game_id=g.id AND g.start_time < now() - interval '3 days'
              AND ps.id NOT IN (SELECT DISTINCT ON (c.game_id, c.player_id, c.market_key, c.kind, c.side, c.line) c.id
                                FROM prop_snapshots c JOIN games gg ON gg.id=c.game_id
                                WHERE c.captured_at < gg.start_time AND gg.start_time < now() - interval '3 days'
                                ORDER BY c.game_id, c.player_id, c.market_key, c.kind, c.side, c.line, c.captured_at DESC)
              AND ps.id NOT IN (SELECT prop_snapshot_id FROM edges)"""))
        await s.execute(text("""DELETE FROM projections p USING games g WHERE p.game_id=g.id
                                AND g.start_time < now() - interval '2 days'"""))
        await s.execute(text("DELETE FROM injury_reports WHERE captured_at < now() - interval '10 days'"))
        await s.execute(text("DELETE FROM weather_forecasts WHERE captured_at < now() - interval '14 days'"))
        await s.execute(text("DELETE FROM ingest_runs WHERE started_at < now() - interval '14 days'"))


async def backfill(sport: str, start: date, end: date) -> None:
    await _ensure_teams(sport)
    d = start
    while d <= end:   # commit in 2-week chunks so a crash doesn't lose everything
        chunk_end = min(end, d + timedelta(days=13))
        await espn.sync_schedule(sport, d, chunk_end)
        await espn.sync_boxscores(sport, limit=400)
        d = chunk_end + timedelta(days=1)
    while await espn.sync_boxscores(sport, limit=200):
        pass


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "refresh", "nightly", "backfill", "export", "scan", "grade"])
    ap.add_argument("--sport", choices=["NBA", "NFL"])
    ap.add_argument("--start", type=date.fromisoformat)
    ap.add_argument("--end", type=date.fromisoformat)
    ap.add_argument("--force", action="store_true", help="plan: mark both sports due")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if a.cmd == "plan":
        sys.argv = ["planner"] + (["--force"] if a.force else [])
        planner.main()
        return
    if a.cmd == "refresh":
        for sp in [a.sport] if a.sport else ["NBA", "NFL"]:
            await refresh(sp)
    elif a.cmd == "nightly":
        await nightly()
    elif a.cmd == "backfill":
        await backfill(a.sport, a.start, a.end)
    elif a.cmd == "scan":
        await scan(a.sport)
    elif a.cmd == "grade":
        await grade(a.sport)
    print(await export())


if __name__ == "__main__":
    asyncio.run(main())
