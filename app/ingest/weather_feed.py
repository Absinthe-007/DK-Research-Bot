"""
Kickoff weather for outdoor NFL games, from Open-Meteo, using the static `data/stadiums.json`
(home team abbreviation -> stadium coordinates + roof type). Dome and retractable-roof stadiums are
skipped: no forecast is fetched and no row is written.

Stores the kickoff-hour wind / gust / precipitation / temperature plus the kickoff -> +3h window max wind
and total precipitation in `kickoff_weather`. The original weather ingestion (`app/ingestion/weather.py`,
NWS-first) is untouched and keeps feeding the existing projector.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import text

from app.db.base import session_scope, upsert
from app.db.models_ext import KickoffWeather
from app.ingestion.common import tracked
from app.ingestion.http import http

log = logging.getLogger(__name__)
STADIUMS_PATH = Path(__file__).resolve().parents[2] / "data" / "stadiums.json"
INDOOR_ROOFS = {"dome", "retractable"}
WINDOW_HOURS = 3
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
HOURLY = "temperature_2m,precipitation_probability,precipitation,wind_speed_10m,wind_gusts_10m"


def load_stadiums(path: Path | str = STADIUMS_PATH) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["stadiums"]


def is_indoor(stadium: dict | None) -> bool:
    return bool(stadium) and stadium.get("roof", "open") in INDOOR_ROOFS


def outdoor_stadium(home_abbr: str, stadiums: dict[str, dict] | None = None) -> dict | None:
    """The stadium record if the game is played outdoors, else None (unknown teams are also None)."""
    st = (stadiums if stadiums is not None else load_stadiums()).get(home_abbr)
    return None if st is None or is_indoor(st) else st


def parse_hourly(hourly: dict, kickoff: datetime, window_hours: int = WINDOW_HOURS) -> dict | None:
    """Pick the kickoff hour out of an Open-Meteo `hourly` block (times requested in UTC)."""
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    kickoff = kickoff.astimezone(timezone.utc)
    hour0 = kickoff.replace(minute=0, second=0, microsecond=0)
    times = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in hourly.get("time", [])]
    if hour0 not in times:
        return None
    i0 = times.index(hour0)
    idx = [i for i, t in enumerate(times) if hour0 <= t <= hour0 + timedelta(hours=window_hours)]

    def at(key, i):
        v = (hourly.get(key) or [None] * len(times))[i]
        return None if v is None else float(v)

    winds = [w for w in (at("wind_speed_10m", i) for i in idx) if w is not None]
    precs = [p for p in (at("precipitation", i) for i in idx) if p is not None]
    return {
        "wind_mph": at("wind_speed_10m", i0), "gust_mph": at("wind_gusts_10m", i0),
        "precip_in": at("precipitation", i0), "precip_prob": _pct(at("precipitation_probability", i0)),
        "temp_f": at("temperature_2m", i0),
        "window_wind_max_mph": max(winds) if winds else None,
        "window_precip_in": round(sum(precs), 3) if precs else None,
    }


def _pct(v: float | None) -> float | None:
    return None if v is None else v / 100.0


async def fetch_kickoff_weather(lat: float, lon: float, kickoff: datetime) -> dict | None:
    """One Open-Meteo call (cached 15 min by the shared HTTP client) -> kickoff-hour values."""
    body, _ = await http().get(OPEN_METEO, cache_ttl=900, params=dict(
        latitude=lat, longitude=lon, timezone="UTC", wind_speed_unit="mph", temperature_unit="fahrenheit",
        precipitation_unit="inch", forecast_days=16, hourly=HOURLY))
    return parse_hourly(body.get("hourly") or {}, kickoff)


@tracked("weather_feed")
async def sync_kickoff_weather(days_ahead: int = 8) -> int:
    """Fetch weather for every upcoming outdoor NFL game in the DB; returns rows written."""
    stadiums = load_stadiums()
    now = datetime.now(timezone.utc)
    async with session_scope() as s:
        games = (await s.execute(text("""
            SELECT g.id, g.start_time, t.abbr FROM games g JOIN teams t ON t.id = g.home_team_id
            WHERE g.sport = 'NFL' AND g.completed = false AND g.neutral_site = false
              AND g.start_time > :now AND g.start_time < :end"""),
            {"now": now, "end": now + timedelta(days=days_ahead)})).all()
    rows = []
    for gid, start, home in games:
        st = outdoor_stadium(home, stadiums)
        if st is None:
            continue                                   # dome / retractable / unknown: skip
        try:
            wx = await fetch_kickoff_weather(st["lat"], st["lon"], start)
        except Exception as e:  # one bad forecast must not sink the rest
            log.warning("open-meteo failed for game %s (%s): %s", gid, home, e)
            continue
        if wx:
            rows.append(dict(game_id=gid, home_abbr=home, stadium=st["stadium"], kickoff=start,
                             fetched_at=datetime.now(timezone.utc), **wx))
    async with session_scope() as s:
        await upsert(s, KickoffWeather, rows, ["game_id"])
    return len(rows)
