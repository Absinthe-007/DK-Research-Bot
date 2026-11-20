"""Fit the honesty-shrink factor from the graded picks ledger (same 30-day window the dashboard uses)."""
from __future__ import annotations

import logging

from sqlalchemy import text

from app.engine.calibration import DEFAULT_SHRINK, fit_shrink

log = logging.getLogger(__name__)

FIT_SQL = """
SELECT pk.model_prob, pk.edge, pk.result
FROM picks pk JOIN games g ON g.id=pk.game_id
WHERE pk.result IN ('win','loss') AND pk.edge IS NOT NULL
  AND g.start_time > now() - interval '30 days'
"""


async def current_shrink(session) -> float:
    try:
        rows = (await session.execute(text(FIT_SQL))).mappings().all()
        return fit_shrink(
            [float(r["model_prob"]) for r in rows],
            [float(r["model_prob"]) - float(r["edge"]) for r in rows],
            [1 if r["result"] == "win" else 0 for r in rows],
        )
    except Exception as e:                     # never block a scan
        log.warning("shrink fit skipped, using default: %s", e)
        return DEFAULT_SHRINK
