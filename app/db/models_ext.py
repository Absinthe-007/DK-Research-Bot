"""
Additive tables for the NFL/NBA prediction upgrades. Nothing here touches the original
projection tables (`projections`, `stat_projections`, `edges`, `picks` keep their schema), so the
original projections are unchanged. `scripts/init_db.py` imports this module so `create_all`
builds these tables on the next run.

Feed tables are keyed by (name_key, team, ...) rather than ESPN ids because nflverse and
stats.nba.com use their own ids; join to `players` on `players.name_key` + `players.sport`.
"""
from datetime import date, datetime

from sqlalchemy import BigInteger, Boolean, Date, DateTime, Float, Integer, String, Text, UniqueConstraint, func
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

FALSE = sql_text("false")


# ---------------------------------------------------------------- NFL (nflreadpy)
class NflUsageWeekly(Base):
    """Snap counts + play-by-play targets/carries per player-week."""
    __tablename__ = "nfl_usage_weekly"
    season: Mapped[int] = mapped_column(Integer, primary_key=True)
    week: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_abbr: Mapped[str] = mapped_column(String(6), primary_key=True)
    name_key: Mapped[str] = mapped_column(String(96), primary_key=True)
    player_name: Mapped[str | None] = mapped_column(String(96))
    position: Mapped[str | None] = mapped_column(String(8))
    gsis_id: Mapped[str | None] = mapped_column(String(16))
    offense_snaps: Mapped[float | None] = mapped_column(Float)
    offense_pct: Mapped[float | None] = mapped_column(Float)
    targets: Mapped[int | None] = mapped_column(Integer)
    carries: Mapped[int | None] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class NflDepthCurrent(Base):
    """Latest depth chart snapshot (replaced wholesale on each sync)."""
    __tablename__ = "nfl_depth_current"
    team_abbr: Mapped[str] = mapped_column(String(6), primary_key=True)
    pos_abb: Mapped[str] = mapped_column(String(8), primary_key=True)
    pos_rank: Mapped[int] = mapped_column(Integer, primary_key=True)
    name_key: Mapped[str] = mapped_column(String(96), primary_key=True)
    player_name: Mapped[str | None] = mapped_column(String(96))
    espn_id: Mapped[str | None] = mapped_column(String(16))
    gsis_id: Mapped[str | None] = mapped_column(String(16))
    snapshot_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class NflInjuryFeed(Base):
    """Official weekly injury report (report + practice status) from nflverse."""
    __tablename__ = "nfl_injury_feed"
    season: Mapped[int] = mapped_column(Integer, primary_key=True)
    week: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_abbr: Mapped[str] = mapped_column(String(6), primary_key=True)
    name_key: Mapped[str] = mapped_column(String(96), primary_key=True)
    player_name: Mapped[str | None] = mapped_column(String(96))
    position: Mapped[str | None] = mapped_column(String(8))
    gsis_id: Mapped[str | None] = mapped_column(String(16))
    report_status: Mapped[str | None] = mapped_column(String(24))
    report_injury: Mapped[str | None] = mapped_column(String(96))
    practice_status: Mapped[str | None] = mapped_column(String(48))
    practice_injury: Mapped[str | None] = mapped_column(String(96))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ---------------------------------------------------------------- NBA (nba_api)
class NbaApiGamelog(Base):
    __tablename__ = "nba_api_gamelog"
    nba_person_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    nba_game_id: Mapped[str] = mapped_column(String(12), primary_key=True)
    name_key: Mapped[str] = mapped_column(String(96), index=True)
    player_name: Mapped[str] = mapped_column(String(96))
    team_abbr: Mapped[str] = mapped_column(String(6))
    matchup: Mapped[str | None] = mapped_column(String(16))
    game_date: Mapped[date] = mapped_column(Date, index=True)
    season: Mapped[str] = mapped_column(String(8))
    wl: Mapped[str | None] = mapped_column(String(2))
    minutes: Mapped[float | None] = mapped_column(Float)
    pts: Mapped[int | None] = mapped_column(Integer)
    reb: Mapped[int | None] = mapped_column(Integer)
    ast: Mapped[int | None] = mapped_column(Integer)
    fg3m: Mapped[int | None] = mapped_column(Integer)
    stl: Mapped[int | None] = mapped_column(Integer)
    blk: Mapped[int | None] = mapped_column(Integer)
    tov: Mapped[int | None] = mapped_column(Integer)
    plus_minus: Mapped[float | None] = mapped_column(Float)


class NbaOnOff(Base):
    """TeamPlayerOnOffDetails (Advanced): team ratings with a player on vs off the court."""
    __tablename__ = "nba_onoff"
    season: Mapped[str] = mapped_column(String(8), primary_key=True)
    team_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    nba_person_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    court_status: Mapped[str] = mapped_column(String(4), primary_key=True)    # On | Off
    team_abbr: Mapped[str | None] = mapped_column(String(6))
    player_name: Mapped[str | None] = mapped_column(String(96))
    name_key: Mapped[str | None] = mapped_column(String(96), index=True)
    gp: Mapped[int | None] = mapped_column(Integer)
    minutes: Mapped[float | None] = mapped_column(Float)
    plus_minus: Mapped[float | None] = mapped_column(Float)
    off_rating: Mapped[float | None] = mapped_column(Float)
    def_rating: Mapped[float | None] = mapped_column(Float)
    net_rating: Mapped[float | None] = mapped_column(Float)
    pace: Mapped[float | None] = mapped_column(Float)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class NbaLiveLineup(Base):
    """Latest live-boxscore roster state per game (starter / active / inactive / played)."""
    __tablename__ = "nba_live_lineup"
    nba_game_id: Mapped[str] = mapped_column(String(12), primary_key=True)
    nba_person_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    team_abbr: Mapped[str | None] = mapped_column(String(6))
    player_name: Mapped[str | None] = mapped_column(String(96))
    name_key: Mapped[str | None] = mapped_column(String(96), index=True)
    position: Mapped[str | None] = mapped_column(String(8))
    starter: Mapped[bool] = mapped_column(Boolean, server_default=FALSE)
    status: Mapped[str | None] = mapped_column(String(12))                   # ACTIVE | INACTIVE
    played: Mapped[bool] = mapped_column(Boolean, server_default=FALSE)
    minutes: Mapped[float | None] = mapped_column(Float)
    not_playing_reason: Mapped[str | None] = mapped_column(String(96))
    game_status: Mapped[int | None] = mapped_column(Integer)                 # 1 pre, 2 live, 3 final
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ---------------------------------------------------------------- weather
class KickoffWeather(Base):
    __tablename__ = "kickoff_weather"
    game_id: Mapped[int] = mapped_column(Integer, primary_key=True)          # games.id
    home_abbr: Mapped[str] = mapped_column(String(6))
    stadium: Mapped[str | None] = mapped_column(String(64))
    kickoff: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    wind_mph: Mapped[float | None] = mapped_column(Float)                    # kickoff hour
    gust_mph: Mapped[float | None] = mapped_column(Float)
    precip_in: Mapped[float | None] = mapped_column(Float)
    precip_prob: Mapped[float | None] = mapped_column(Float)
    temp_f: Mapped[float | None] = mapped_column(Float)
    window_wind_max_mph: Mapped[float | None] = mapped_column(Float)         # kickoff -> +3h
    window_precip_in: Mapped[float | None] = mapped_column(Float)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ---------------------------------------------------------------- markets
class MarketLine(Base):
    """One row per (source, event, market, selection, line): the OPEN snapshot is written once,
    the CURRENT snapshot is overwritten on every poll until the event starts, so after kickoff
    `cur_*` is the closing line."""
    __tablename__ = "market_lines"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(24))          # oddsapi:draftkings | kalshi | polymarket
    sport: Mapped[str] = mapped_column(String(3))
    event_key: Mapped[str] = mapped_column(String(96))
    market_key: Mapped[str] = mapped_column(String(48))
    selection: Mapped[str] = mapped_column(String(96))        # home | away | Over|<player> | yes ...
    line: Mapped[float] = mapped_column(Float, server_default="0")
    pair_key: Mapped[str | None] = mapped_column(String(160))
    home_team: Mapped[str | None] = mapped_column(String(48))
    away_team: Mapped[str | None] = mapped_column(String(48))
    commence_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    open_price: Mapped[int | None] = mapped_column(Integer)
    open_prob: Mapped[float | None] = mapped_column(Float)
    open_fair: Mapped[float | None] = mapped_column(Float)
    open_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    cur_price: Mapped[int | None] = mapped_column(Integer)
    cur_prob: Mapped[float | None] = mapped_column(Float)
    cur_fair: Mapped[float | None] = mapped_column(Float)
    cur_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    n_updates: Mapped[int] = mapped_column(Integer, server_default="1")
    __table_args__ = (UniqueConstraint("source", "event_key", "market_key", "selection", "line", name="uq_market_line"),)


class ClvLog(Base):
    """Closing Line Value per graded pick (positive = we beat the closing price)."""
    __tablename__ = "clv_log"
    pick_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    placed_price: Mapped[int | None] = mapped_column(Integer)
    placed_prob: Mapped[float | None] = mapped_column(Float)      # breakeven prob at the placed price
    close_price: Mapped[int | None] = mapped_column(Integer)
    close_prob: Mapped[float | None] = mapped_column(Float)      # breakeven prob at the closing price (same vig regime)
    close_fair: Mapped[float | None] = mapped_column(Float)      # vig-free closing prob when a two-way pair was captured
    clv_prob: Mapped[float | None] = mapped_column(Float)        # close_prob - placed_prob
    beat_close: Mapped[bool | None] = mapped_column(Boolean)
    source: Mapped[str | None] = mapped_column(String(24))
    logged_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    note: Mapped[str | None] = mapped_column(Text)
