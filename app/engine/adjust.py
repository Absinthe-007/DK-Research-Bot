"""
Opt-in, flag-gated injury / inactives adjustment layer.

NBA is minutes-first: when a rotation player is out, his minutes go to teammates and their counting stats scale with
the extra minutes. NFL redistributes target share and carry share. Both use `redistribution.redistribute`.

Rules that keep the original projections intact:
  * `stat_projections` is only read, never written. Results go to `adjusted_stat_projections`, which stores the
    original mean next to the adjusted one, so `compare_report` can show which is more accurate after grading.
  * A player's own adjustment is conditional on him playing: a Questionable starter keeps his projection (props on a
    DNP are voided); only his teammates absorb the expected absence.
  * Only rows whose factor differs from 1 by >= 1% are stored.

What the original stat board already does (so this doesn't double count): it drops Out players and applies a
per-minute usage shift for the single biggest absence. It does not give teammates extra minutes and it does not
shift NFL target/carry share, which is what this layer adds. Gate: set ADJUSTED_PROJECTIONS=1 (CLI skips otherwise).
"""
from __future__ import annotations

import logging
import os
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from app.db.base import session_scope, upsert
from app.db.models_ext import AdjustedStatProjection
from app.engine.redistribution import play_probability, redistribute
from app.engine.statboard import OUT_STATUSES
from app.ingestion.common import tracked

log = logging.getLogger(__name__)

NBA_SCALED = ("minutes", "pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "pra")
NFL_TARGET_STATS = ("targets", "receptions", "rec_yds", "rec_tds")
NFL_CARRY_STATS = ("carries", "rush_yds", "rush_tds")
NBA_MIN_CAP = 40.0
NFL_TARGET_CAP, NFL_CARRY_CAP = 0.45, 0.70
# NFL defaults are deliberately conservative and UNTUNED: backups below MIN_SHARE absorb some vacated volume that this
# model can't see, so only 70% of the extra share is passed through and the factor is capped at 1.4. Use compare_report
# on graded games to decide whether to loosen them.
NFL_ELASTICITY, NFL_MAX_FACTOR = 0.7, 1.4
MIN_SHARE = 0.02            # ignore NFL players with less than 2% of the team's targets / carries
MIN_FACTOR_CHANGE = 0.01
LIVE_STATUS = {"INACTIVE": "Out", "ACTIVE": "Active"}


def adjusted_enabled() -> bool:
    return os.environ.get("ADJUSTED_PROJECTIONS", "").strip().lower() in ("1", "true", "yes", "on")


# ------------------------------------------------------------------ pure core
@dataclass
class Shift:
    adj: float          # usage (minutes or share) after redistribution, conditional on playing
    factor: float       # multiplier for his stats, clipped to [lo, hi]
    p_active: float
    vacated: float      # expected usage freed on his team by everyone else's absences


def conditional_multipliers(
    base: Mapping[Hashable, float], status: Mapping[Hashable, str | None], *, groups: Mapping | None = None,
    cap: float = float("inf"), same_group_boost: float = 2.0, lo: float = 0.6, hi: float = 1.6,
    elasticity: float = 1.0, overrides: Mapping[str, float] | None = None,
) -> dict[Hashable, Shift]:
    """Per-player stat multiplier given the team's baseline usage and each player's report status.
    `elasticity` < 1 dampens the pass-through of extra usage into stats (extra minutes are lower quality)."""
    pp = {k: play_probability(status.get(k), overrides) for k in base}
    out: dict[Hashable, Shift] = {}
    if all(p >= 1.0 for p in pp.values()):
        return {k: Shift(float(v), 1.0, 1.0, 0.0) for k, v in base.items() if v > 0}
    for k, b in base.items():
        if b <= 0:
            continue
        if pp[k] <= 0:
            out[k] = Shift(0.0, 0.0, 0.0, 0.0)
            continue
        res = redistribute(base, {**pp, k: 1.0}, groups=groups, same_group_boost=same_group_boost, caps=cap)
        factor = min(hi, max(lo, 1.0 + elasticity * (res.values[k] / b - 1.0)))
        out[k] = Shift(res.values[k], factor, pp[k], res.vacated)
    return out


def build_rows(game_id: int, sport: str, shifts: Mapping[int, Shift], stat_rows: list[dict], stats: tuple[str, ...],
               statuses: Mapping[int, tuple[str | None, str]], base: Mapping[int, float]) -> list[dict]:
    rows = []
    for r in stat_rows:
        sh = shifts.get(r["player_id"])
        if sh is None or sh.p_active <= 0 or r["stat"] not in stats or abs(sh.factor - 1.0) < MIN_FACTOR_CHANGE:
            continue
        st, src = statuses.get(r["player_id"], (None, "injury_report"))
        rows.append(dict(
            game_id=game_id, player_id=r["player_id"], stat=r["stat"], sport=sport, original_mean=float(r["mean"]),
            adjusted_mean=round(float(r["mean"]) * sh.factor, 3),
            lo=None if r.get("lo") is None else round(float(r["lo"]) * sh.factor, 3),
            hi=None if r.get("hi") is None else round(float(r["hi"]) * sh.factor, 3),
            factor=round(sh.factor, 4), base_usage=base.get(r["player_id"]), adj_usage=round(sh.adj, 4),
            p_active=sh.p_active, status=st, source=src))
    return rows


# ------------------------------------------------------------------ database layer
async def _upcoming(s, sport: str, hours: float, now: datetime):
    return (await s.execute(text("""
        SELECT g.id game_id, g.start_time, g.season, g.week, ht.abbr home, at.abbr away
        FROM games g JOIN teams ht ON ht.id = g.home_team_id JOIN teams at ON at.id = g.away_team_id
        WHERE g.sport = :sp AND NOT g.completed AND g.start_time > :now AND g.start_time < :hi AND g.season_type IN (2, 3)
        ORDER BY g.start_time"""), {"sp": sport, "now": now, "hi": now + timedelta(hours=hours)})).mappings().all()


async def _injury_status(s, sport: str, now: datetime) -> dict[int, str]:
    rows = (await s.execute(text("""
        SELECT DISTINCT ON (player_id) player_id, status FROM injury_reports
        WHERE sport = :sp AND player_id IS NOT NULL AND captured_at > CAST(:now AS timestamptz) - interval '36 hours'
          AND captured_at <= :now ORDER BY player_id, captured_at DESC"""), {"sp": sport, "now": now})).all()
    return {pid: st for pid, st in rows}


async def _live_status(s, now: datetime) -> dict[str, str]:
    """name_key -> Out | Active from the NBA live boxscore (pre-game inactives), only while the game hasn't finished."""
    rows = (await s.execute(text("""
        SELECT DISTINCT ON (name_key) name_key, status FROM nba_live_lineup
        WHERE updated_at > CAST(:now AS timestamptz) - interval '5 hours' AND game_status IN (1, 2) AND name_key IS NOT NULL
        ORDER BY name_key, updated_at DESC"""), {"now": now})).all()
    return {nk: LIVE_STATUS[st] for nk, st in rows if st in LIVE_STATUS}


async def _stat_rows(s, sport: str, game_id: int) -> list[dict]:
    return [dict(r) for r in (await s.execute(text(
        "SELECT player_id, stat, mean, lo, hi FROM stat_projections WHERE sport = :sp AND game_id = :g"),
        {"sp": sport, "g": game_id})).mappings().all()]


async def _build_nba(s, now: datetime, hours: float) -> list[dict]:
    games = await _upcoming(s, "NBA", hours, now)
    if not games:
        return []
    teams = sorted({g["home"] for g in games} | {g["away"] for g in games})
    players = (await s.execute(text(
        "SELECT id, name_key, team_abbr FROM players WHERE sport = 'NBA' AND team_abbr = ANY(:t)"), {"t": teams})).all()
    inj, live = await _injury_status(s, "NBA", now), await _live_status(s, now)
    out: list[dict] = []
    for g in games:
        stat_rows = await _stat_rows(s, "NBA", g["game_id"])
        mins = {r["player_id"]: float(r["mean"]) for r in stat_rows if r["stat"] == "minutes"}
        for team in (g["home"], g["away"]):
            roster = [(pid, nk) for pid, nk, tm in players if tm == team]
            statuses: dict[int, tuple[str | None, str]] = {}
            for pid, nk in roster:
                if nk in live:
                    statuses[pid] = (live[nk], "live_lineup")
                elif pid in inj:
                    statuses[pid] = (inj[pid], "injury_report")
            absent = [pid for pid, _ in roster if pid not in mins and statuses.get(pid, ("",))[0] in OUT_STATUSES + ("Doubtful", "Questionable")]
            base = {pid: mins[pid] for pid, _ in roster if pid in mins}
            if absent:   # the stat board drops Out players, so recover their usual minutes from recent games
                rec = (await s.execute(text("""
                    SELECT player_id, AVG(minutes) FROM (
                      SELECT player_id, minutes, ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY game_date DESC) rn
                      FROM nba_player_gamelogs WHERE player_id = ANY(:ids) AND minutes > 0) t
                    WHERE rn <= 10 GROUP BY player_id"""), {"ids": absent})).all()
                base.update({pid: float(m) for pid, m in rec if m is not None and float(m) >= 8.0})
            if len(base) < 5:
                continue
            shifts = conditional_multipliers(base, {pid: st[0] for pid, st in statuses.items()}, cap=NBA_MIN_CAP)
            team_rows = [r for r in stat_rows if r["player_id"] in base]
            out += build_rows(g["game_id"], "NBA", shifts, team_rows, NBA_SCALED, statuses, base)
    return out


async def _build_nfl(s, now: datetime, hours: float) -> list[dict]:
    games = await _upcoming(s, "NFL", hours, now)
    if not games:
        return []
    inj = await _injury_status(s, "NFL", now)
    out: list[dict] = []
    for g in games:
        try:
            season, week = int(str(g["season"])[:4]), int(g["week"])
        except (TypeError, ValueError):
            continue
        usage = (await s.execute(text("""
            SELECT u.team_abbr, u.name_key, MAX(u.position) pos, COALESCE(SUM(u.targets), 0) t, COALESCE(SUM(u.carries), 0) c
            FROM nfl_usage_weekly u WHERE u.season = :se AND u.week BETWEEN :w0 AND :w1 AND u.team_abbr IN (:a, :b)
            GROUP BY u.team_abbr, u.name_key"""), {"se": season, "w0": week - 4, "w1": week - 1, "a": g["home"], "b": g["away"]})).all()
        if not usage:
            continue
        ids = {(tm, nk): pid for pid, nk, tm in (await s.execute(text(
            "SELECT id, name_key, team_abbr FROM players WHERE sport = 'NFL' AND team_abbr IN (:a, :b)"),
            {"a": g["home"], "b": g["away"]})).all()}
        stat_rows = await _stat_rows(s, "NFL", g["game_id"])
        for team in (g["home"], g["away"]):
            mine = [(nk, pos, int(t), int(c)) for tm, nk, pos, t, c in usage if tm == team and (team, nk) in ids]
            statuses = {ids[(team, nk)]: (inj[ids[(team, nk)]], "injury_report") for nk, *_ in mine if ids[(team, nk)] in inj}
            for idx, stats, cap in ((2, NFL_TARGET_STATS, NFL_TARGET_CAP), (3, NFL_CARRY_STATS, NFL_CARRY_CAP)):
                total = sum(m[idx] for m in mine)
                if total < 20:
                    continue
                base = {ids[(team, m[0])]: m[idx] / total for m in mine if m[idx] / total >= MIN_SHARE}
                groups = {ids[(team, m[0])]: m[1] for m in mine if ids[(team, m[0])] in base}
                shifts = conditional_multipliers(base, {p: st[0] for p, st in statuses.items()}, groups=groups, cap=cap,
                                                 elasticity=NFL_ELASTICITY, hi=NFL_MAX_FACTOR)
                out += build_rows(g["game_id"], "NFL", shifts, [r for r in stat_rows if r["player_id"] in base], stats, statuses, base)
    return out


@tracked("adjust_projections")
async def build_adjusted(sport: str, hours: float | None = None) -> int:
    """Recompute adjustments for upcoming games. Replaces earlier adjusted rows for those games only."""
    now = datetime.now(timezone.utc)
    hours = hours or (36 if sport == "NBA" else 96)
    async with session_scope() as s:
        rows = await (_build_nba(s, now, hours) if sport == "NBA" else _build_nfl(s, now, hours))
        games = [g["game_id"] for g in await _upcoming(s, sport, hours, now)]
        if games:
            await s.execute(text("DELETE FROM adjusted_stat_projections WHERE sport = :sp AND game_id = ANY(:g)"),
                            {"sp": sport, "g": games})
        return await upsert(s, AdjustedStatProjection, rows, ["game_id", "player_id", "stat"])


async def compare_report(sport: str) -> list[dict]:
    """Original vs adjusted mean absolute error on graded rows, per stat. Only rows the adjustment changed appear."""
    async with session_scope() as s:
        rows = (await s.execute(text("""
            SELECT a.stat, COUNT(*) n, AVG(ABS(a.original_mean - sp.actual)) mae_original,
                   AVG(ABS(a.adjusted_mean - sp.actual)) mae_adjusted,
                   AVG(CASE WHEN ABS(a.adjusted_mean - sp.actual) < ABS(a.original_mean - sp.actual) THEN 1.0 ELSE 0.0 END) share_improved
            FROM adjusted_stat_projections a
            JOIN stat_projections sp ON sp.game_id = a.game_id AND sp.player_id = a.player_id AND sp.stat = a.stat
            WHERE a.sport = :sp AND sp.actual IS NOT NULL GROUP BY a.stat ORDER BY a.stat"""), {"sp": sport})).mappings().all()
    return [dict(r) for r in rows]
