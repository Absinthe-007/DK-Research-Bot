"""Shared helpers for ingestion modules: job bookkeeping, id lookups, parsing."""
from __future__ import annotations

import functools
import logging
import re
import traceback
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import session_scope
from app.db.models import IngestRun, Player
from app.utils.names import best_match

log = logging.getLogger(__name__)
_ODDS = re.compile(r"^[+-]?\d+$")


def parse_american(v) -> int | None:
    """'+170' -> 170, '-112' -> -112, 'EVEN' -> 100, 'OFF'/'' -> None."""
    if v is None:
        return None
    s = str(v).strip().upper()
    if s in ("EVEN", "EV", "PK"):
        return 100
    if not _ODDS.match(s):
        return None
    n = int(s)
    return n if abs(n) >= 100 else None


def parse_float(v) -> float | None:
    if v is None:
        return None
    s = re.sub(r"[^\d.+-]", "", str(v))   # 'o43.5' -> 43.5, '+4.5' -> 4.5
    try:
        return float(s)
    except ValueError:
        return None


def ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


async def player_ids_by_espn(s: AsyncSession, sport: str) -> dict[str, int]:
    rows = (await s.execute(select(Player.espn_id, Player.id).where(Player.sport == sport))).all()
    return dict(rows)


async def player_index(s: AsyncSession, sport: str) -> dict[str, int]:
    rows = (await s.execute(select(Player.name_key, Player.id).where(Player.sport == sport))).all()
    return dict(rows)


def resolve_player(name: str, index: dict[str, int]) -> int | None:
    return best_match(name, index)


def tracked(job: str):
    """Records each job execution in ingest_runs; the wrapped coroutine returns a row count."""
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*a, **kw):
            async with session_scope() as s:
                run = IngestRun(job=job)
                s.add(run)
                await s.flush()
                run_id = run.id
            ok, rows, err = False, 0, None
            try:
                rows = await fn(*a, **kw) or 0
                ok = True
                return rows
            except Exception:
                err = traceback.format_exc()
                log.exception("job %s failed", job)
                raise
            finally:
                async with session_scope() as s:
                    r = await s.get(IngestRun, run_id)
                    r.finished_at, r.rows, r.ok, r.error = datetime.now(timezone.utc), rows, ok, err
        return wrapper
    return deco
