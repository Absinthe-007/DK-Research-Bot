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
STALE_AFTER = timedelta(hours=6)  # planner errors force a refresh once data is this old
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


# Rebuild the board once games are over so finished games drop off and the next slate is current.
# Must exceed export.py's "started within the last 4 hours" window so the finished games fall out.
POST_GAME = {"NBA": timedelta(hours=4, minutes=15), "NFL": timedelta(hours=4, minutes=30)}


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1, sort_keys=True))


async def upcoming_starts(sport: str, now: datetime, recent: list | None = None) -> list[datetime]:
    """Future, not-yet-started game times. One scoreboard request per calendar day: ESPN answers
    HTTP 400 to multi-day `dates=A-B` ranges (single dates work), which silently stopped every
    scheduled refresh in Nov 2026. A day that fails is skipped; only if every day fails do we raise."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/{ESPN_PATH[sport]}/scoreboard"
    events: dict[str, dict] = {}
    errors: list[Exception] = []
    days = 5
    for offset in range(-1, days - 1):          # yesterday .. +3 days (UTC-ish, covers late US games)
        day = now + timedelta(days=offset)
        try:
            body, _ = await http().get(url, params={"dates": f"{day:%Y%m%d}", "limit": 300})
        except Exception as e:                  # noqa: BLE001 - one bad day shouldn't hide the rest
            errors.append(e)
            continue
        for e in body.get("events", []):
            if "id" in e:
                events[e["id"]] = e
    if errors and len(errors) == days:
        raise errors[0]
    out = []
    for e in events.values():
        comps = e.get("competitions") or []
        if not comps:
            continue
        st = comps[0].get("status", {}).get("type", {})
        state_str = st.get("state", "pre")
        t = datetime.fromisoformat(e["date"].replace("Z", "+00:00"))
        if not st.get("completed") and state_str == "pre" and t > now:
            out.append(t)
        elif recent is not None and t <= now:
            recent.append(t)            # already started/finished: used for the post-game refresh
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
        recent: list[datetime] = []
        try:
            starts = await upcoming_starts(sport, now, recent)
        except Exception as e:      # ESPN down: normally retry next tick, but never go stale for long
            stale = last_dt is None or (now - last_dt) > STALE_AFTER
            result[sport] = {"due": stale, "error": str(e), "stale_fallback": stale}
            continue
        hits = due(starts, CHECKPOINTS[sport], last_dt, now)
        # post-game: negative checkpoint = start + POST_GAME
        hits += [h.replace("T--", "T+") for h in due(recent, [-POST_GAME[sport]], last_dt, now)]
        result[sport] = {
            "due": bool(hits),
            "checkpoints": hits,
            "games_next_4d": len(starts),
        }
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
