"""Central configuration, loaded from environment / .env (GitHub Actions sets these as env vars)."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://dkbot:dkbot@localhost:5432/dkbot"
    contact_email: str = "dk-research-bot@users.noreply.github.com"   # NWS asks for a contact in the UA

    edge_threshold: float = 0.035
    # Final projection = MODEL_WEIGHT * model mean + (1 - MODEL_WEIGHT) * market-implied mean.
    # Sports-betting markets are sharp; an un-anchored box-score model disagrees with DK by
    # ~12 pts of probability on average (measured on a real slate), almost all of it noise.
    # Raise this only after the graded record shows the model adds information.
    model_weight: float = 0.35
    # One-sided "milestone" alt lines (e.g. 25+ points) are only flagged inside this price band:
    # deep tails are where any model's error is largest relative to the price.
    max_model_market_gap: float = 0.30
    alt_edge_threshold: float = 0.06
    alt_min_price: int = -250
    alt_max_price: int = 250
    kelly_fraction: float = 0.25
    max_stake_pct: float = 0.02
    max_total_exposure_pct: float = 0.10
    bankroll: float = 1000.0

    @property
    def user_agent(self) -> str:
        return f"dk-research-bot/2.0 (+{self.contact_email})"


@lru_cache
def get_settings() -> Settings:
    return Settings()
