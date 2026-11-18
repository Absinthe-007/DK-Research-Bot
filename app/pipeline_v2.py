"""
Entry point for the prediction-upgrade jobs (GitHub Actions workflows 01-03). Kept separate from
`app/pipeline.py` so the original refresh / nightly flow, and the projections it produces, are untouched.

    python -m app.pipeline_v2 ingest [--sport NBA|NFL]   # nflverse + nba_api feeds, kickoff weather, opening odds
    python -m app.pipeline_v2 odds   [--sport ...]       # odds + prediction-market snapshot (open vs current)
    python -m app.pipeline_v2 gate   [--sport ...] [--hours 6]   # is any game about to start? (no DB) -> $GITHUB_OUTPUT
    python -m app.pipeline_v2 grade  [--sport ...]       # box scores -> grade picks -> log CLV
    python -m app.pipeline_v2 adjust [--sport ...]       # OPT-IN (ADJUSTED_PROJECTIONS=1): injury/inactives adjustment
    python -m app.pipeline_v2 adjust-report [--sport ...]  # original vs adjusted accuracy on graded rows
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

from app.config import get_settings

log = logging.getLogger("pipeline_v2")
SPORTS = ("NBA", "NFL")


STEP_TIMEOUT = 600.0   # seconds; a rate-limited API can ask the shared HTTP client to sleep for hours


async def _safe(fn, *a, **kw):
    try:
        return await asyncio.wait_for(fn(*a, **kw), STEP_TIMEOUT)
    except asyncio.TimeoutError:
        log.warning("%s timed out after %.0fs; moving on", getattr(fn, "__name__", fn), STEP_TIMEOUT)
        return None
    except Exception as e:
        log.warning("%s failed: %s", getattr(fn, "__name__", fn), e)
        return None


def parse_scoreboard_starts(body: dict, now: datetime) -> list[datetime]:
    """Start times of games that have not started yet, from an ESPN scoreboard payload."""
    out = []
    for e in body.get("events", []):
        comps = e.get("competitions") or []
        st = (comps[0].get("status", {}).get("type", {}) if comps else {})
        try:
            t = datetime.fromisoformat(e["date"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        if not st.get("completed") and st.get("state", "pre") == "pre" and t > now:
            out.append(t)
    return out


async def upcoming_starts_daily(sport: str, now: datetime) -> list[datetime]:
    """One scoreboard request per calendar day. ESPN has been answering 400 to multi-day `dates=A-B` ranges
    (which `planner.upcoming_starts` uses), while single dates work, so the gate avoids ranges."""
    from app.db.reference import ESPN_PATH
    from app.ingestion.http import http

    starts: list[datetime] = []
    for offset in (-1, 0, 1):
        day = (now + timedelta(days=offset)).strftime("%Y%m%d")
        body, _ = await http().get(f"https://site.api.espn.com/apis/site/v2/sports/{ESPN_PATH[sport]}/scoreboard",
                                   params={"dates": day, "limit": 300})
        starts += parse_scoreboard_starts(body, now)
    return sorted(set(starts))


async def games_soon(sport: str, hours: float, now: datetime | None = None) -> bool:
    """True when a game of `sport` starts within `hours` (ESPN scoreboard; no database needed)."""
    now = now or datetime.now(timezone.utc)
    starts = await upcoming_starts_daily(sport, now)
    return any(now < t <= now + timedelta(hours=hours) for t in starts)


async def ingest(sports=SPORTS) -> None:
    from app.ingest import espn_nba, market_feed, nba_feed, nfl_feed, weather_feed
    from app.ingestion import espn

    today = datetime.now(timezone.utc).date()
    for sp in sports:
        await _safe(espn.sync_schedule, sp, today - timedelta(days=1), today + timedelta(days=8))
    if "NFL" in sports:
        log.info("nfl feed: %s", await _safe(nfl_feed.sync_all))
        await _safe(weather_feed.sync_kickoff_weather)
    if "NBA" in sports:
        log.info("nba feed: %s", await _safe(nba_feed.sync_all, None, today - timedelta(days=10)))
        # free ESPN source: works from GitHub runners (stats.nba.com does not)
        log.info("espn nba lineups: %s", await _safe(espn_nba.sync_lineups, 1, 2))
        log.info("espn nba gamelogs: %s", await _safe(espn_nba.sync_gamelogs, 10))
    from app.ingestion.oddsapi import Budget

    cfg = get_settings()
    budget = Budget(cfg.odds_api_max_credits_per_run, cfg.odds_api_min_remaining)
    for sp in sports:                                   # morning snapshot = the "open" for most lines
        await _safe(market_feed.snapshot, sp, budget=budget)


async def odds(sports=SPORTS, hours: float = 6.0, gate: bool = True) -> None:
    from app.ingest import espn_nba, market_feed, nba_feed
    from app.ingestion.oddsapi import Budget

    cfg = get_settings()
    budget = Budget(cfg.odds_api_max_credits_per_run, cfg.odds_api_min_remaining)
    for sp in sports:
        if gate and not await _safe(games_soon, sp, hours):
            log.info("%s: no game starting within %.0fh, skipping", sp, hours)
            continue
        log.info("%s: %s quotes written", sp, await _safe(market_feed.snapshot, sp, budget=budget))
        if sp == "NBA":                                 # starters / inactives firm up close to tip-off
            await _safe(espn_nba.sync_lineups, 0, 1)    # free ESPN source (works on GitHub runners)
            await _safe(nba_feed.sync_live_lineups)     # official feed: best effort, usually blocked there


async def grade(sports=SPORTS) -> None:
    from app.engine import statboard
    from app.engine.grade import grade as original_grade
    from app.eval import grader
    from app.ingestion import espn

    for sp in sports:
        await _safe(espn.sync_boxscores, sp)
        await _safe(original_grade, sp)                 # existing grader first: unchanged behaviour
        await _safe(statboard.grade, sp)
        if sp == "NBA":
            await _safe(sync_recent_nba_logs)
        else:
            from app.ingest import nfl_feed
            await _safe(nfl_feed.sync_usage)            # snap counts = participation evidence for the new grader
        log.info("%s: %s", sp, await _safe(grader.run, sp))


async def sync_recent_nba_logs() -> None:
    from app.ingest import espn_nba, nba_feed

    await _safe(espn_nba.sync_gamelogs, 4)              # free ESPN final box scores first
    await _safe(espn_nba.sync_lineups, 2, 0)            # final lineup state: played / DNP evidence
    await nba_feed.sync_gamelogs(date_from=datetime.now(timezone.utc).date() - timedelta(days=3))


async def adjust(sports=SPORTS, force: bool = False) -> None:
    """Flag-gated. Reads `stat_projections`, writes `adjusted_stat_projections`; the original rows are never modified."""
    from app.engine import adjust as adj
    from app.ingestion import injuries

    if not (force or adj.adjusted_enabled()):
        log.info("ADJUSTED_PROJECTIONS is not enabled; skipping (original projections are unaffected either way)")
        return
    for sp in sports:
        await _safe(injuries.sync_injuries, sp)         # fresh statuses; ESPN marks inactives "Out" near kickoff
        log.info("%s: %s adjusted rows", sp, await _safe(adj.build_adjusted, sp))


async def adjust_report(sports=SPORTS) -> None:
    from app.engine import adjust as adj

    for sp in sports:
        rows = await _safe(adj.compare_report, sp) or []
        if not rows:
            log.info("%s: no graded adjusted rows yet", sp)
        for r in rows:
            print(f"{sp} {r['stat']:<14} n={r['n']:<5} MAE original={r['mae_original']:.3f} adjusted={r['mae_adjusted']:.3f} "
                  f"improved={r['share_improved']:.0%}")


def write_output(key: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")
    print(f"{key}={value}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ingest", "odds", "gate", "grade", "adjust", "adjust-report"])
    ap.add_argument("--sport", choices=list(SPORTS))
    ap.add_argument("--hours", type=float, default=6.0, help="gate/odds: only poll when a game starts within this many hours")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--force", action="store_true", help="adjust: run even when ADJUSTED_PROJECTIONS is not set")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sports = (a.sport,) if a.sport else SPORTS
    if a.cmd == "gate":
        flags = {sp: bool(await _safe(games_soon, sp, a.hours)) for sp in sports}
        write_output("any", str(any(flags.values())).lower())
        write_output("sports", ",".join(sp for sp, v in flags.items() if v))
    elif a.cmd == "ingest":
        await ingest(sports)
    elif a.cmd == "odds":
        await odds(sports, a.hours, gate=not a.no_gate)
    elif a.cmd == "grade":
        await grade(sports)
    elif a.cmd == "adjust":
        await adjust(sports, a.force)
    elif a.cmd == "adjust-report":
        await adjust_report(sports)


if __name__ == "__main__":
    asyncio.run(main())
