"""Central configuration, loaded from environment / .env (GitHub Actions sets these as env vars)."""
from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MODEL_WEIGHT_BY_MARKET: dict[str, float] = {
    # NBA high-liquidity scoring markets (lean heavily on DraftKings anchor)
    "player_points": 0.20,
    "player_points_rebounds_assists": 0.25,
    "player_points_rebounds": 0.25,
    "player_points_assists": 0.25,
    "player_rebounds_assists": 0.45,
    # NBA counting & peripheral markets (model rate x minutes adds strong signal)
    "player_assists": 0.55,
    "player_rebounds": 0.50,
    "player_threes": 0.55,
    "player_steals": 0.55,
    "player_blocks": 0.50,
    "player_steals_blocks": 0.50,
    "player_turnovers": 0.45,
    # NFL markets
    "player_pass_yds": 0.25,
    "player_rush_yds": 0.30,
    "player_reception_yds": 0.55,
    "player_receptions": 0.50,
    "player_pass_completions": 0.40,
    "player_pass_attempts": 0.35,
    "player_rush_attempts": 0.35,
    "player_pass_tds": 0.45,
    "player_pass_interceptions": 0.45,
    "player_rush_reception_yds": 0.35,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://dkbot:dkbot@localhost:5432/dkbot"
    contact_email: str = "dk-research-bot@users.noreply.github.com"  # NWS asks for a contact in the UA

    # Raised from 0.035: 3-5 pt gaps sit inside the model's own error, and the graded record (51% hit vs 59% expected)
    # shows the small-gap plays are not beating DraftKings' margin. The by-gap table on the dashboard shows whether
    # bigger gaps pay more; lower this again only if they do not.
    edge_threshold: float = 0.08

    # Final projection = MODEL_WEIGHT * model mean + (1 - MODEL_WEIGHT) * market-implied mean.
    # Sports-betting markets are sharp; an un-anchored box-score model disagrees with DK by
    # ~12 pts of probability on average (measured on a real slate), almost all of it noise.
    # Raise this only after the graded record shows the model adds information.
    model_weight: float = 0.35

    # One-sided "milestone" alt lines (e.g. 25+ points) are only flagged inside this price band:
    # deep tails are where any model's error is largest relative to the price.
    max_model_market_gap: float = 0.30
    alt_edge_threshold: float = 0.10
    alt_min_price: int = -250
    alt_max_price: int = 250

    kelly_fraction: float = 0.25
    max_stake_pct: float = 0.02
    max_total_exposure_pct: float = 0.10
    bankroll: float = 1000.0

    @field_validator("database_url", mode="before")
    @classmethod
    def _ensure_asyncpg(cls, v: str) -> str:
        if isinstance(v, str):
            if v.startswith("postgres://"):
                return "postgresql+asyncpg://" + v[len("postgres://") :]
            if v.startswith("postgresql://"):
                return "postgresql+asyncpg://" + v[len("postgresql://") :]
        return v

    def model_weight_for(self, market_key: str) -> float:
        """Per-market model weight. Backtested on ~8k graded NBA projections: the market alone
        is best for points/PRA (high-volume, heavily-bet), while counting stats gain from leaning on the model more."""
        w = MODEL_WEIGHT_BY_MARKET.get(market_key, self.model_weight)
        return float(min(1.0, max(0.0, w)))

    @property
    def user_agent(self) -> str:
        return f"dk-research-bot/2.1 (+{self.contact_email})"


@lru_cache
def get_settings() -> Settings:
    return Settings()