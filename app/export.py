"""
Export the dashboard data (docs/data/board.json) that GitHub Pages serves. Plain JSON, no
server, no LLM - the static page in docs/ renders it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import text

from app.config import get_settings
from app.db.base import session_scope
from app.db.reference import MARKET_INDEX
from app.engine.scanner import jsonable

OUT = Path(__file__).resolve().parent.parent / "docs" / "data"

EDGES_SQL = """
SELECT e.id, p.full_name player, p.team_abbr team, p.position pos, p.headshot, pj.market_key, e.kind, e.side, e.line,
       e.price, e.model_prob, e.push_prob, e.breakeven_prob, e.fair_prob, e.edge, e.ev_per_unit, e.kelly_full,
       e.stake_pct, e.flagged, e.flags, pj.mean proj_mean, pj.std proj_std, pj.dist_family, pj.sample_size,
       pj.inputs, g.id game_id, g.start_time, ht.abbr home, at.abbr away, ps.open_price, ps.open_line, ps.captured_at
FROM edges e JOIN projections pj ON pj.id=e.projection_id JOIN prop_snapshots ps ON ps.id=e.prop_snapshot_id
JOIN players p ON p.id=pj.player_id JOIN games g ON g.id=pj.game_id
JOIN teams ht ON ht.id=g.home_team_id JOIN teams at ON at.id=g.away_team_id
WHERE e.sport=:sp AND g.start_time > now() AND (e.flagged OR e.kind='ou')
ORDER BY e.flagged DESC, e.edge DESC
"""

GAMES_SQL = """
SELECT g.id, g.start_time, g.week, ht.abbr home, at.abbr away, ht.name home_name, at.name away_name,
       v.name venue, v.indoor, v.city,
       (SELECT json_agg(x) FROM (SELECT DISTINCT ON (market_type, side) market_type, side, price, point, open_price, open_point
               FROM odds_snapshots o WHERE o.game_id=g.id ORDER BY market_type, side, captured_at DESC) x) lines,
       (SELECT row_to_json(w) FROM (SELECT temp_f, wind_mph, wind_gust_mph, precip_prob, short_forecast, source, captured_at
               FROM weather_forecasts wf WHERE wf.game_id=g.id ORDER BY captured_at DESC LIMIT 1) w) weather,
       (SELECT json_agg(r) FROM (SELECT t.abbr, tr.days_rest, tr.is_b2b, tr.is_3in4, tr.off_bye, tr.short_week
               FROM team_rest tr JOIN teams t ON t.id=tr.team_id WHERE tr.game_id=g.id) r) rest
FROM games g JOIN teams ht ON ht.id=g.home_team_id JOIN teams at ON at.id=g.away_team_id
LEFT JOIN venues v ON v.id=g.venue_id
WHERE g.sport=:sp AND g.start_time > now() - interval '4 hours' AND g.start_time < now() + interval '7 days'
ORDER BY g.start_time
"""

INJ_SQL = """
SELECT DISTINCT ON (player_name_raw) player_name_raw player, team_abbr team, status, injury, comment, return_date, captured_at
FROM injury_reports WHERE sport=:sp AND captured_at > now() - interval '36 hours'
  AND status IN ('Out','Doubtful','Questionable','Day-To-Day')
ORDER BY player_name_raw, captured_at DESC
"""

RECORD_SQL = """
SELECT COUNT(*) FILTER (WHERE result='win') w, COUNT(*) FILTER (WHERE result='loss') l,
       COUNT(*) FILTER (WHERE result='push') p, COALESCE(SUM(profit_units),0) units,
       COUNT(*) FILTER (WHERE result IN ('win','loss','push')) n,
       AVG(CASE WHEN close_price IS NOT NULL THEN
           (CASE WHEN close_price<0 THEN -close_price/(100.0-close_price) ELSE 100.0/(close_price+100) END)
         - (CASE WHEN price<0 THEN -price/(100.0-price) ELSE 100.0/(price+100) END) END) clv
FROM picks WHERE sport=:sp AND result IN ('win','loss','push') AND (CAST(:days AS int) IS NULL
  OR first_flagged_at > now() - make_interval(days => CAST(:days AS int)))
"""

RESULTS_SQL = """
SELECT pk.market_key, pk.kind, pk.side, pk.line, pk.price, pk.close_price, pk.model_prob, pk.edge, pk.proj_mean,
       pk.actual, pk.result, pk.profit_units, p.full_name player, p.team_abbr team, g.start_time
FROM picks pk JOIN players p ON p.id=pk.player_id JOIN games g ON g.id=pk.game_id
WHERE pk.sport=:sp AND pk.result IN ('win','loss','push','void') ORDER BY g.start_time DESC, pk.edge DESC LIMIT 150
"""


def _slim(e: dict) -> dict:
    inp = e.pop("inputs") or {}
    e["recent"] = inp.pop("recent", [])
    e["adjust"] = {k: v for k, v in inp.items() if k not in ("flags",)}
    e["label"] = MARKET_INDEX[e["market_key"]][2]
    return e


async def export() -> Path:
    st = get_settings()
    OUT.mkdir(parents=True, exist_ok=True)
    board: dict = {"generated_at": datetime.now(timezone.utc).isoformat(), "edge_threshold": st.edge_threshold,
                   "bankroll": st.bankroll, "kelly_fraction": st.kelly_fraction, "sports": {}}
    async with session_scope() as s:
        for sp in ("NBA", "NFL"):
            edges = [_slim(dict(r)) for r in (await s.execute(text(EDGES_SQL), {"sp": sp})).mappings().all()]
            picks = [e for e in edges if e["flagged"]]
            lines = [{k: e[k] for k in ("player", "team", "market_key", "label", "side", "line", "price", "model_prob",
                                        "push_prob", "fair_prob", "edge", "proj_mean", "game_id", "kind")}
                     for e in edges if not e["flagged"] and e["kind"] == "ou" and e["side"] == "Over"][:400]
            rec = {}
            for label, days in (("all", None), ("30d", 30), ("7d", 7)):
                r = (await s.execute(text(RECORD_SQL), {"sp": sp, "days": days})).mappings().first()
                rec[label] = dict(r)
            board["sports"][sp] = jsonable({
                "games": [dict(r) for r in (await s.execute(text(GAMES_SQL), {"sp": sp})).mappings().all()],
                "picks": picks, "lines": lines,
                "injuries": [dict(r) for r in (await s.execute(text(INJ_SQL), {"sp": sp})).mappings().all()],
                "record": rec,
                "results": [dict(r) for r in (await s.execute(text(RESULTS_SQL), {"sp": sp})).mappings().all()],
            })
        board["last_runs"] = jsonable([dict(r) for r in (await s.execute(text("""
            SELECT DISTINCT ON (job) job, finished_at, ok, rows FROM ingest_runs ORDER BY job, started_at DESC"""
        ))).mappings().all()])
    path = OUT / "board.json"
    path.write_text(json.dumps(board, separators=(",", ":"), default=str))
    return path
