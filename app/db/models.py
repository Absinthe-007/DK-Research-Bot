"""
Relational schema (PostgreSQL). Every entity is keyed by ESPN ids, so props, box scores,
rosters and injuries join without fuzzy name matching.

* Shared dimensions (teams, players, venues, games) carry `sport`.
* Box scores are decoupled by sport (nba_player_gamelogs / nfl_player_games).
* Odds are standardized by canonical `market_key` in `prop_snapshots`; snapshots are
  append-only but only written when a price/line CHANGES, which keeps the DB small enough
  to round-trip through a GitHub release asset.
* `picks` keeps the last pre-game version of every flagged edge so it can be graded.
"""
from datetime import date, datetime

from sqlalchemy import (
    BigInteger, Boolean, Date, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func,
)
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

TRUE, FALSE = sql_text("true"), sql_text("false")


class Team(Base):
    __tablename__ = "teams"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sport: Mapped[str] = mapped_column(String(3))
    espn_id: Mapped[str] = mapped_column(String(8))
    abbr: Mapped[str] = mapped_column(String(6))
    name: Mapped[str] = mapped_column(String(64))
    __table_args__ = (UniqueConstraint("sport", "espn_id"),)


class Player(Base):
    __tablename__ = "players"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sport: Mapped[str] = mapped_column(String(3), index=True)
    espn_id: Mapped[str] = mapped_column(String(16))
    full_name: Mapped[str] = mapped_column(String(96))
    name_key: Mapped[str] = mapped_column(String(96), index=True)
    position: Mapped[str | None] = mapped_column(String(8))
    team_abbr: Mapped[str | None] = mapped_column(String(6), index=True)
    headshot: Mapped[str | None] = mapped_column(String(256))
    __table_args__ = (UniqueConstraint("sport", "espn_id"),)


class Venue(Base):
    __tablename__ = "venues"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    espn_id: Mapped[str] = mapped_column(String(12), unique=True)
    name: Mapped[str] = mapped_column(String(128))
    city: Mapped[str | None] = mapped_column(String(64))
    state: Mapped[str | None] = mapped_column(String(32))
    country: Mapped[str | None] = mapped_column(String(48))
    indoor: Mapped[bool] = mapped_column(Boolean, server_default=FALSE)
    lat: Mapped[float | None] = mapped_column(Float)
    lon: Mapped[float | None] = mapped_column(Float)
    altitude_ft: Mapped[int | None] = mapped_column(Integer)


class Game(Base):
    __tablename__ = "games"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sport: Mapped[str] = mapped_column(String(3), index=True)
    espn_event_id: Mapped[str] = mapped_column(String(16), unique=True)
    season: Mapped[str] = mapped_column(String(8))
    season_type: Mapped[int] = mapped_column(Integer, server_default="2")   # 1 pre, 2 regular, 3 post
    week: Mapped[int | None] = mapped_column(Integer)
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    venue_id: Mapped[int | None] = mapped_column(ForeignKey("venues.id"))
    neutral_site: Mapped[bool] = mapped_column(Boolean, server_default=FALSE)
    status: Mapped[str] = mapped_column(String(24), server_default="STATUS_SCHEDULED")
    completed: Mapped[bool] = mapped_column(Boolean, server_default=FALSE)
    home_score: Mapped[int | None] = mapped_column(Integer)
    away_score: Mapped[int | None] = mapped_column(Integer)
    boxscore_loaded: Mapped[bool] = mapped_column(Boolean, server_default=FALSE)


class NbaPlayerGamelog(Base):
    __tablename__ = "nba_player_gamelogs"
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), primary_key=True)
    season: Mapped[str] = mapped_column(String(8), index=True)
    game_date: Mapped[date] = mapped_column(Date, index=True)
    team_abbr: Mapped[str] = mapped_column(String(6), index=True)
    opp_abbr: Mapped[str] = mapped_column(String(6), index=True)
    is_home: Mapped[bool] = mapped_column(Boolean)
    starter: Mapped[bool] = mapped_column(Boolean, server_default=FALSE)
    minutes: Mapped[float] = mapped_column(Float)
    pts: Mapped[int] = mapped_column(Integer)
    reb: Mapped[int] = mapped_column(Integer)
    oreb: Mapped[int] = mapped_column(Integer)
    ast: Mapped[int] = mapped_column(Integer)
    fg3m: Mapped[int] = mapped_column(Integer)
    stl: Mapped[int] = mapped_column(Integer)
    blk: Mapped[int] = mapped_column(Integer)
    tov: Mapped[int] = mapped_column(Integer)
    fga: Mapped[int] = mapped_column(Integer)
    fta: Mapped[int] = mapped_column(Integer)
    plus_minus: Mapped[float | None] = mapped_column(Float)


class NflPlayerGame(Base):
    __tablename__ = "nfl_player_games"
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), primary_key=True)
    season: Mapped[int] = mapped_column(Integer, index=True)
    week: Mapped[int] = mapped_column(Integer)
    game_date: Mapped[date] = mapped_column(Date, index=True)
    team_abbr: Mapped[str] = mapped_column(String(6))
    opp_abbr: Mapped[str] = mapped_column(String(6), index=True)
    position: Mapped[str | None] = mapped_column(String(8))
    completions: Mapped[float] = mapped_column(Float, server_default="0")
    pass_attempts: Mapped[float] = mapped_column(Float, server_default="0")
    pass_yds: Mapped[float] = mapped_column(Float, server_default="0")
    pass_tds: Mapped[float] = mapped_column(Float, server_default="0")
    interceptions: Mapped[float] = mapped_column(Float, server_default="0")
    sacks_taken: Mapped[float] = mapped_column(Float, server_default="0")
    carries: Mapped[float] = mapped_column(Float, server_default="0")
    rush_yds: Mapped[float] = mapped_column(Float, server_default="0")
    rush_tds: Mapped[float] = mapped_column(Float, server_default="0")
    targets: Mapped[float] = mapped_column(Float, server_default="0")
    receptions: Mapped[float] = mapped_column(Float, server_default="0")
    rec_yds: Mapped[float] = mapped_column(Float, server_default="0")
    rec_tds: Mapped[float] = mapped_column(Float, server_default="0")
    # defense / kicking
    tackles: Mapped[float] = mapped_column(Float, server_default="0")
    sacks: Mapped[float] = mapped_column(Float, server_default="0")
    tfl: Mapped[float] = mapped_column(Float, server_default="0")
    qb_hits: Mapped[float] = mapped_column(Float, server_default="0")
    passes_defended: Mapped[float] = mapped_column(Float, server_default="0")
    def_int: Mapped[float] = mapped_column(Float, server_default="0")
    fgm: Mapped[float] = mapped_column(Float, server_default="0")
    fga: Mapped[float] = mapped_column(Float, server_default="0")
    xpm: Mapped[float] = mapped_column(Float, server_default="0")
    # optional nflverse enrichment
    offense_snap_pct: Mapped[float | None] = mapped_column(Float)
    air_yards: Mapped[float | None] = mapped_column(Float)
    target_share: Mapped[float | None] = mapped_column(Float)
    wopr: Mapped[float | None] = mapped_column(Float)


class Market(Base):
    __tablename__ = "markets"
    key: Mapped[str] = mapped_column(String(48), primary_key=True)
    sport: Mapped[str] = mapped_column(String(3))
    label: Mapped[str] = mapped_column(String(64))
    stat_expr: Mapped[str] = mapped_column(String(128))
    dist_family: Mapped[str] = mapped_column(String(16))


class OddsSnapshot(Base):
    """DK game lines: moneyline / spread / total."""
    __tablename__ = "odds_snapshots"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"), index=True)
    market_type: Mapped[str] = mapped_column(String(12))    # h2h | spreads | totals
    side: Mapped[str] = mapped_column(String(16))           # home | away | over | under
    price: Mapped[int | None] = mapped_column(Integer)
    point: Mapped[float | None] = mapped_column(Float)
    open_price: Mapped[int | None] = mapped_column(Integer)
    open_point: Mapped[float | None] = mapped_column(Float)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class PropSnapshot(Base):
    __tablename__ = "prop_snapshots"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"), index=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    market_key: Mapped[str] = mapped_column(ForeignKey("markets.key"))
    kind: Mapped[str] = mapped_column(String(4))             # ou | alt
    side: Mapped[str] = mapped_column(String(8))             # Over | Under
    line: Mapped[float] = mapped_column(Float)
    price: Mapped[int] = mapped_column(Integer)
    price_estimated: Mapped[bool] = mapped_column(Boolean, server_default=FALSE)   # ESPN showed the line but no price
    open_price: Mapped[int | None] = mapped_column(Integer)
    open_line: Mapped[float | None] = mapped_column(Float)
    book_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    __table_args__ = (Index("ix_props_latest", "game_id", "player_id", "market_key", "kind", "side", "line", "captured_at"),)


class InjuryReport(Base):
    __tablename__ = "injury_reports"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    sport: Mapped[str] = mapped_column(String(3), index=True)
    player_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"), index=True)
    player_name_raw: Mapped[str] = mapped_column(String(96))
    team_abbr: Mapped[str | None] = mapped_column(String(6))
    status: Mapped[str] = mapped_column(String(24))
    injury: Mapped[str | None] = mapped_column(String(64))
    comment: Mapped[str | None] = mapped_column(Text)
    return_date: Mapped[str | None] = mapped_column(String(16))
    reported_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source: Mapped[str] = mapped_column(String(16))
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class WeatherForecast(Base):
    __tablename__ = "weather_forecasts"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"), index=True)
    temp_f: Mapped[float | None] = mapped_column(Float)
    wind_mph: Mapped[float | None] = mapped_column(Float)
    wind_gust_mph: Mapped[float | None] = mapped_column(Float)
    precip_prob: Mapped[float | None] = mapped_column(Float)
    precip_in: Mapped[float | None] = mapped_column(Float)
    short_forecast: Mapped[str | None] = mapped_column(String(128))
    source: Mapped[str] = mapped_column(String(16))
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class TeamRest(Base):
    __tablename__ = "team_rest"
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"), primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), primary_key=True)
    days_rest: Mapped[int | None] = mapped_column(Integer)
    is_b2b: Mapped[bool] = mapped_column(Boolean, server_default=FALSE)
    is_3in4: Mapped[bool] = mapped_column(Boolean, server_default=FALSE)
    off_bye: Mapped[bool] = mapped_column(Boolean, server_default=FALSE)
    short_week: Mapped[bool] = mapped_column(Boolean, server_default=FALSE)


class Projection(Base):
    __tablename__ = "projections"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"), index=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    market_key: Mapped[str] = mapped_column(ForeignKey("markets.key"))
    model_version: Mapped[str] = mapped_column(String(16))
    mean: Mapped[float] = mapped_column(Float)
    std: Mapped[float] = mapped_column(Float)
    dist_family: Mapped[str] = mapped_column(String(16))
    dist_params: Mapped[dict] = mapped_column(JSONB)
    sample_size: Mapped[int] = mapped_column(Integer)
    inputs: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (UniqueConstraint("game_id", "player_id", "market_key", "model_version"),)


class Edge(Base):
    """Every evaluated prop side from the LATEST scan (replaced each scan)."""
    __tablename__ = "edges"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    projection_id: Mapped[int] = mapped_column(ForeignKey("projections.id", ondelete="CASCADE"))
    prop_snapshot_id: Mapped[int] = mapped_column(ForeignKey("prop_snapshots.id", ondelete="CASCADE"))
    sport: Mapped[str] = mapped_column(String(3), index=True)
    kind: Mapped[str] = mapped_column(String(4))
    side: Mapped[str] = mapped_column(String(8))
    line: Mapped[float] = mapped_column(Float)
    price: Mapped[int] = mapped_column(Integer)
    model_prob: Mapped[float] = mapped_column(Float)
    push_prob: Mapped[float] = mapped_column(Float, server_default="0")
    breakeven_prob: Mapped[float] = mapped_column(Float)
    fair_prob: Mapped[float | None] = mapped_column(Float)
    edge: Mapped[float] = mapped_column(Float, index=True)
    ev_per_unit: Mapped[float] = mapped_column(Float)
    kelly_full: Mapped[float] = mapped_column(Float, server_default="0")
    stake_pct: Mapped[float] = mapped_column(Float, server_default="0")
    flagged: Mapped[bool] = mapped_column(Boolean, server_default=FALSE)
    flags: Mapped[list] = mapped_column(JSONB, server_default=sql_text("'[]'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Pick(Base):
    """Flagged edges, frozen at their last pre-game state, then graded after the box score lands."""
    __tablename__ = "picks"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    sport: Mapped[str] = mapped_column(String(3), index=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"), index=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"))
    market_key: Mapped[str] = mapped_column(String(48))
    kind: Mapped[str] = mapped_column(String(4))
    side: Mapped[str] = mapped_column(String(8))
    line: Mapped[float] = mapped_column(Float)
    price: Mapped[int] = mapped_column(Integer)
    first_price: Mapped[int] = mapped_column(Integer)
    model_prob: Mapped[float] = mapped_column(Float)
    edge: Mapped[float] = mapped_column(Float)
    stake_pct: Mapped[float] = mapped_column(Float)
    proj_mean: Mapped[float] = mapped_column(Float)
    active: Mapped[bool] = mapped_column(Boolean, server_default=TRUE)      # still flagged at last pre-game scan
    first_flagged_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    close_price: Mapped[int | None] = mapped_column(Integer)
    actual: Mapped[float | None] = mapped_column(Float)
    result: Mapped[str | None] = mapped_column(String(6))                   # win | loss | push | void
    profit_units: Mapped[float | None] = mapped_column(Float)
    __table_args__ = (UniqueConstraint("game_id", "player_id", "market_key", "kind", "side", "line"),)


class StatProjection(Base):
    """Expected box-score line for every rostered player in upcoming games (no prices involved).
    Frozen at its last pre-game refresh, then `actual` is filled in for accuracy tracking."""
    __tablename__ = "stat_projections"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    sport: Mapped[str] = mapped_column(String(3), index=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"), index=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    stat: Mapped[str] = mapped_column(String(24))
    mean: Mapped[float] = mapped_column(Float)
    lo: Mapped[float] = mapped_column(Float)
    hi: Mapped[float] = mapped_column(Float)
    model_mean: Mapped[float] = mapped_column(Float)
    anchored: Mapped[bool] = mapped_column(Boolean, server_default=FALSE)
    last5: Mapped[float | None] = mapped_column(Float)
    season_avg: Mapped[float | None] = mapped_column(Float)
    sample: Mapped[int] = mapped_column(Integer)
    actual: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (UniqueConstraint("game_id", "player_id", "stat"),)


class IngestRun(Base):
    __tablename__ = "ingest_runs"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    job: Mapped[str] = mapped_column(String(48), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    rows: Mapped[int] = mapped_column(Integer, server_default="0")
    ok: Mapped[bool] = mapped_column(Boolean, server_default=FALSE)
    error: Mapped[str | None] = mapped_column(Text)
