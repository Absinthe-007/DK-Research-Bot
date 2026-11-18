"""
Game-aware scheduler for GitHub Actions.

A cheap cron tick (every 15 min) runs `python -m app.pipeline plan`. It fetches the ESPN
scoreboards (2 requests, no database) and decides whether a full refresh is due, based on
CHECKPOINTS measured backwards from each game's actual kickoff/tip time:

  NBA  -10h  props usually posted by morning ET
        -4h  after the afternoon injury-report update
       -90m  pre-game injury report window
       -30m  final: starting lineups / late scratches
  NFL  -72h  midweek (Thu/Fri practice reports, props for Sunday)
       -26h  day before (final injury designations issued Friday/Saturday)
        -5h  morning of: weather forecast is now reliable
       -75m  after inactives (posted 90 min before kickoff)
       -25m  final

A refresh is due when any checkpoint of a not-yet-started game falls inside
(last_run, now + SLACK]. Because checkpoints follow real start times, international
morning games, Saturday NFL games, holiday NBA slates etc. are handled automatically, and
nothing runs on days without games. last_run is stored in state/state.json, which the
workflow keeps in the `data` GitHub release next to the database dump.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.db.reference import ESPN_PATH
from app.ingestion.http import http

STATE = Path(os.environ.get("STATE_PATH") or Path(__file__).resolve().parent.parent / "state" / "state.json")
SLACK = timedelta(minutes=8)   # a tick slightly early still counts (cron jitter)
CHECKPOINTS = {
    "NBA": [timedelta(hours=10), timedelta(hours=4), timedelta(minutes=90), timedelta(minutes=30)],
    "NFL": [timedelta(hours=72), timedelta(hours=26), timedelta(hours=5), timedelta(minutes=75), timedelta(minutes=25)],
}


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1, sort_keys=True))


async def upcoming_starts(sport: str, now: datetime) -> list[datetime]:
    events = []
    for i in range(-1, 4):      # ESPN scoreboard takes single dates; UTC day -1 covers late US games
        d = now + timedelta(days=i)
        body, _ = await http().get(f"https://site.api.espn.com/apis/site/v2/sports/{ESPN_PATH[sport]}/scoreboard",
                                   params={"dates": f"{d:%Y%m%d}", "limit": 500})
        events += body.get("events", [])
    out = []
    for e in {e["id"]: e for e in events}.values():
        st = e["competitions"][0]["status"]["type"]
        t = datetime.fromisoformat(e["date"].replace("Z", "+00:00"))
        if not st["completed"] and t > now:
            out.append(t)
    return out


def due(starts: list[datetime], checkpoints: list[timedelta], last_run: datetime | None, now: datetime) -> list[str]:
    """Return the checkpoint labels that fired since last_run (pure function, unit-tested)."""
    lo = last_run or (now - timedelta(hours=1))
    hits = []
    for t in starts:
        for cp in checkpoints:
            when = t - cp
            if lo < when <= now + SLACK:
                hits.append(f"{t:%m-%d %H:%MZ} T-{int(cp.total_seconds() // 60)}m")
    return hits


async def plan(now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    state = load_state()
    result = {}
    for sport in ("NBA", "NFL"):
        last = state.get(f"last_refresh_{sport}")
        last_dt = datetime.fromisoformat(last) if last else None
        try:
            starts = await upcoming_starts(sport, now)
        except Exception as e:      # ESPN hiccup: don't run blind, try next tick
            result[sport] = {"due": False, "error": str(e)}
            continue
        hits = due(starts, CHECKPOINTS[sport], last_dt, now)
        result[sport] = {"due": bool(hits), "checkpoints": hits, "games_next_4d": len(starts)}
    return result


def main() -> None:
    """`python -m app.planner [--force]` - prints the plan and writes GitHub Actions outputs."""
    import asyncio
    import sys
    res = asyncio.run(plan())
    if "--force" in sys.argv:
        for v in res.values():
            v["due"] = True
    sports = [s for s, v in res.items() if v.get("due")]
    print(json.dumps(res, indent=1, default=str))
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"sports={json.dumps(sports)}\nany={'true' if sports else 'false'}\n")


if __name__ == "__main__":
    main()
