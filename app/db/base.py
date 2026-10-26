"""Engine / session factories + a generic Postgres upsert helper."""
from collections.abc import Iterable, Sequence
from contextlib import asynccontextmanager

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings


class Base(DeclarativeBase):
    pass


_settings = get_settings()
engine = create_async_engine(_settings.database_url, pool_size=10, max_overflow=20, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


@asynccontextmanager
async def session_scope():
    async with SessionLocal() as s:
        try:
            yield s
            await s.commit()
        except Exception:
            await s.rollback()
            raise


async def upsert(
    session: AsyncSession,
    model,
    rows: Sequence[dict],
    conflict_cols: Iterable[str],
    update_cols: Iterable[str] | None = None,
    chunk: int = 2000,
) -> int:
    """INSERT ... ON CONFLICT DO UPDATE in chunks (asyncpg caps bind params at 32767)."""
    if not rows:
        return 0
    conflict_cols = list(conflict_cols)
    if update_cols is None:
        update_cols = [c for c in rows[0] if c not in conflict_cols]
    update_cols = list(update_cols)
    chunk = max(1, min(chunk, 32000 // max(1, len(rows[0]))))   # stay under the bind-param cap
    n = 0
    for i in range(0, len(rows), chunk):
        batch = rows[i : i + chunk]
        stmt = pg_insert(model).values(batch)
        if update_cols:
            stmt = stmt.on_conflict_do_update(
                index_elements=conflict_cols,
                set_={c: getattr(stmt.excluded, c) for c in update_cols},
            )
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=conflict_cols)
        await session.execute(stmt)
        n += len(batch)
    return n
