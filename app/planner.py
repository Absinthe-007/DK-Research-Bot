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

A refresh is due when any checkpoint of a not-yet-started game falls inside (last_run, now].
Because checkpoints follow real start times, international morning games, Saturday NFL games,
holiday NBA slates etc. are handled automatically, and nothing runs on days without games.

last_run is stored in state/state.json, which the workflow keeps in the `data` GitHub release
next to the database dump.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.db.reference import ESPN_PATH
from app.ingestion.http import http

STATE = Path(
    os.environ.get("STATE_PATH")
    or Path(__file__).resolve().parent.parent / "state" / "state.json"
)
SLACK = timedelta(minutes=8)  # kept for backwards compatibility with existing unit tests

CHECKPOINTS = {
    "NBA": [
        timedelta(hours=10),
        timedelta(hours=4),
        timedelta(minutes=90),
        timedelta(minutes=30),
    ],
    "NFL": [
        timedelta(hours=72),
        timedelta(hours=26),
        timedelta(hours=5),
        timedelta(minutes=75),
        timedelta(minutes=25),
    ],
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
    d_start = now - timedelta(days=1)
    d_end = now + timedelta(days=3)
    url = f"https://site.api.espn.com/apis/site/v2/sports/{ESPN_PATH[sport]}/scoreboard"
    body, _ = await http().get(
        url,
        params={"dates": f"{d_start:%Y%m%d}-{d_end:%Y%m%d}", "limit": 500},
    )
    events = body.get("events", [])
    out = []
    for e in {e["id"]: e for e in events if "id" in e}.values():
        comps = e.get("competitions") or []
        if not comps:
            continue
        st = comps[0].get("status", {}).get("type", {})
        state_str = st.get("state", "pre")
        t = datetime.fromisoformat(e["date"].replace("Z", "+00:00"))
        if not st.get("completed") and state_str == "pre" and t > now:
            out.append(t)
    return sorted(out)


def due(
    starts: list[datetime],
    checkpoints: list[timedelta],
    last_run: datetime | None,
    now: datetime,
) -> list[str]:
    """Return the checkpoint labels that fired since last_run (pure function, unit-tested).
    Uses `lo < when <= now` when `last_run` is present so a checkpoint never fires twice
    across consecutive cron ticks."""
    lo = last_run or (now - timedelta(hours=12))
    if lo.tzinfo is None:
        lo = lo.replace(tzinfo=timezone.utc)
    hi = now if last_run is not None else (now + SLACK)
    hits = []
    for t in starts:
        for cp in checkpoints:
            when = t - cp
            if lo < when <= hi:
                hits.append(f"{t:%m-%d %H:%MZ} T-{int(cp.total_seconds() // 60)}m")
    return hits


async def plan(now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    state = load_state()
    result = {}
    for sport in ("NBA", "NFL"):
        last = state.get(f"last_refresh_{sport}")
        last_dt = datetime.fromisoformat(last.replace("Z", "+00:00")) if last else None
        if last_dt and last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        try:
            starts = await upcoming_starts(sport, now)
        except Exception as e:      # ESPN hiccup: don't run blind, try next tick
            result[sport] = {"due": False, "error": str(e)}
            continue
        hits = due(starts, CHECKPOINTS[sport], last_dt, now)
        result[sport] = {
            "due": bool(hits),
            "checkpoints": hits,
            "games_next_4d": len(starts),
        }
    return result


def