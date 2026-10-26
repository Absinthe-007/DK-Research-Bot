"""
Game-window weather for outdoor NFL games.

Venue coordinates: ESPN gives each venue an address + `indoor` flag. We geocode the city once
with Open-Meteo's free geocoder (cached in `venues`), with a small table for international
sites. A few km of geocoding error is irrelevant for a forecast.

Forecast: National Weather Service hourly (US, free, requires a UA with contact) with
Open-Meteo as global fallback. We store the AVERAGE over kickoff -> +3.25h plus the max gust.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.db.base import session_scope
from app.db.models import Game, Venue, WeatherForecast
from app.db.reference import KNOWN_VENUES
from app.ingestion.common import tracked
from app.ingestion.http import http

log = logging.getLogger(__name__)
GAME_HOURS = 3.25
_MPH = re.compile(r"(\d+)")
COUNTRY_CODES = {"USA": "US", "England": "GB", "United Kingdom": "GB", "Germany": "DE", "Mexico": "MX",
                 "Spain": "ES", "France": "FR", "Ireland": "IE", "Brazil": "BR", "Australia": "AU", "Canada": "CA"}

US_STATES = dict(AL="Alabama", AK="Alaska", AZ="Arizona", AR="Arkansas", CA="California", CO="Colorado",
                 CT="Connecticut", DE="Delaware", DC="District of Columbia", FL="Florida", GA="Georgia", HI="Hawaii",
                 ID="Idaho", IL="Illinois", IN="Indiana", IA="Iowa", KS="Kansas", KY="Kentucky", LA="Louisiana",
                 ME="Maine", MD="Maryland", MA="Massachusetts", MI="Michigan", MN="Minnesota", MS="Mississippi",
                 MO="Missouri", MT="Montana", NE="Nebraska", NV="Nevada", NH="New Hampshire", NJ="New Jersey",
                 NM="New Mexico", NY="New York", NC="North Carolina", ND="North Dakota", OH="Ohio", OK="Oklahoma",
                 OR="Oregon", PA="Pennsylvania", RI="Rhode Island", SC="South Carolina", SD="South Dakota",
                 TN="Tennessee", TX="Texas", UT="Utah", VT="Vermont", VA="Virginia", WA="Washington",
                 WV="West Virginia", WI="Wisconsin", WY="Wyoming")


async def geocode(v: Venue) -> tuple[float, float, int] | None:
    if v.name in KNOWN_VENUES:
        return KNOWN_VENUES[v.name]
    if not v.city:
        return None
    params = {"name": v.city, "count": 10, "language": "en"}
    cc = COUNTRY_CODES.get(v.country or "USA")
    if cc:
        params["countryCode"] = cc
    body, _ = await http().get("https://geocoding-api.open-meteo.com/v1/search", params=params, cache_ttl=86400 * 30)
    res = body.get("results") or []
    if cc == "US" and v.state:
        want = US_STATES.get(v.state.upper(), v.state).lower()
        res = [r for r in res if str(r.get("admin1", "")).lower() == want] or res
    res.sort(key=lambda r: -(r.get("population") or 0))
    if not res:
        return None
    r = res[0]
    return r["latitude"], r["longitude"], int((r.get("elevation") or 0) * 3.281)


def _parse_wind(s: str | None) -> float | None:
    nums = [int(x) for x in _MPH.findall(s or "")]
    return float(max(nums)) if nums else None


async def _nws(lat: float, lon: float, start: datetime) -> dict | None:
    meta, _ = await http().get(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}", cache_ttl=86400)
    fc, _ = await http().get(meta["properties"]["forecastHourly"], cache_ttl=900)
    end = start + timedelta(hours=GAME_HOURS)
    periods = [p for p in fc["properties"]["periods"]
               if start - timedelta(hours=1) <= datetime.fromisoformat(p["startTime"]) <= end]
    if not periods:
        return None
    winds = [w for w in (_parse_wind(p.get("windSpeed")) for p in periods) if w is not None]
    pops = [(p.get("probabilityOfPrecipitation") or {}).get("value") or 0 for p in periods]
    return dict(temp_f=sum(p["temperature"] for p in periods) / len(periods),
                wind_mph=sum(winds) / len(winds) if winds else None, wind_gust_mph=max(winds) if winds else None,
                precip_prob=max(pops) / 100.0, precip_in=None, short_forecast=periods[0].get("shortForecast"), source="nws")


async def _open_meteo(lat: float, lon: float, start: datetime) -> dict | None:
    body, _ = await http().get("https://api.open-meteo.com/v1/forecast", cache_ttl=900, params=dict(
        latitude=lat, longitude=lon, timezone="UTC", wind_speed_unit="mph", temperature_unit="fahrenheit",
        precipitation_unit="inch", forecast_days=16,
        hourly="temperature_2m,precipitation_probability,precipitation,wind_speed_10m,wind_gusts_10m"))
    h = body["hourly"]
    end = start + timedelta(hours=GAME_HOURS)
    idx = [i for i, t in enumerate(h["time"])
           if start - timedelta(hours=1) <= datetime.fromisoformat(t).replace(tzinfo=timezone.utc) <= end]
    if not idx:
        return None
    avg = lambda k: sum(h[k][i] or 0 for i in idx) / len(idx)  # noqa: E731
    return dict(temp_f=avg("temperature_2m"), wind_mph=avg("wind_speed_10m"),
                wind_gust_mph=max(h["wind_gusts_10m"][i] or 0 for i in idx),
                precip_prob=max(h["precipitation_probability"][i] or 0 for i in idx) / 100.0,
                precip_in=sum(h["precipitation"][i] or 0 for i in idx), short_forecast=None, source="open-meteo")


@tracked("weather")
async def sync_weather(days_ahead: int = 6) -> int:
    now = datetime.now(timezone.utc)
    async with session_scope() as s:
        rows = (await s.execute(select(Game, Venue).join(Venue, Game.venue_id == Venue.id).where(
            Game.sport == "NFL", Game.completed.is_(False), Game.start_time > now,
            Game.start_time < now + timedelta(days=days_ahead), Venue.indoor.is_(False)))).all()
        for _g, v in rows:   # geocode once, cache on the venue row
            if v.lat is None:
                try:
                    geo = await geocode(v)
                except Exception as e:
                    log.warning("geocode %s failed: %s", v.name, e)
                    geo = None
                if geo:
                    v.lat, v.lon, v.altitude_ft = geo
        pairs = [(g.id, g.start_time, v.lat, v.lon, v.country) for g, v in rows if v.lat is not None]
    out = []
    for gid, start, lat, lon, country in pairs:
        wx = None
        if (country or "USA") == "USA":
            try:
                wx = await _nws(lat, lon, start)
            except Exception as e:
                log.info("NWS failed (%s); using Open-Meteo", e)
        if wx is None:
            try:
                wx = await _open_meteo(lat, lon, start)
            except Exception as e:
                log.warning("weather unavailable for game %s: %s", gid, e)
        if wx:
            out.append(WeatherForecast(game_id=gid, **wx))
    async with session_scope() as s:
        s.add_all(out)
    return len(out)
