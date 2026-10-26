"""
Create tables + views and seed the market dictionary. Idempotent: safe to run on every
GitHub Actions job after restoring the database (new tables from upgrades get created).

    python -m scripts.init_db
"""
import asyncio

from sqlalchemy import text

from app.db import models as M
from app.db.base import Base, engine, session_scope, upsert
from app.db.reference import MARKETS

# additive column migrations (create_all never alters existing tables)
MIGRATIONS = [
    "ALTER TABLE prop_snapshots ADD COLUMN IF NOT EXISTS price_estimated boolean DEFAULT false NOT NULL",
]

VIEWS = [
    """
    DROP VIEW IF EXISTS current_props;
    CREATE VIEW current_props AS
    SELECT DISTINCT ON (ps.game_id, ps.player_id, ps.market_key, ps.kind, ps.side, ps.line)
           ps.*, g.sport, g.start_time
    FROM prop_snapshots ps JOIN games g ON g.id = ps.game_id
    WHERE ps.captured_at > now() - interval '30 hours'
    ORDER BY ps.game_id, ps.player_id, ps.market_key, ps.kind, ps.side, ps.line, ps.captured_at DESC
    """,
]


async def main():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        for m in MIGRATIONS:
            await conn.execute(text(m))
        for v in VIEWS:
            for stmt in filter(str.strip, v.split(";")):
                await conn.execute(text(stmt))
    async with session_scope() as s:
        await upsert(s, M.Market, [dict(key=k, sport=sp, label=lb, stat_expr=e, dist_family=f)
                                   for k, sp, lb, e, f in MARKETS], ["key"])
    print("database ready")


if __name__ == "__main__":
    asyncio.run(main())
