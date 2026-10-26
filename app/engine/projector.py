"""
Projection pipeline: (game, player, market) -> fitted outcome distribution + audit trail.

    NBA:  mean = EWMA(per-minute rate) x projected minutes x opp factor x pace factor x usage shift
    NFL:  mean = EWMA(per-game stat)   x opp (defense-vs-position) factor x weather factor
    sd   : recency-weighted sample sd, rescaled to the adjusted mean (coefficient of variation
           held constant), floored to avoid over-confident tails on small samples.

Every factor lands in `inputs`, so each probability shown on the dashboard is explainable.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.reference import MARKET_INDEX
from app.engine import baselines as B
from app.engine.distributions import PropDistribution, blend_with_history, empirical_hit_rate, fit, prob_over_under

log = logging.getLogger(__name__)
MODEL_VERSION = "v2.0"
LOOKBACK = 25
MIN_GAMES = 5

NBA_COLS = {"pts", "reb", "ast", "fg3m", "stl", "blk", "tov"}
NFL_COLS = {"completions", "pass_attempts", "pass_yds", "pass_tds", "interceptions", "carries", "rush_yds",
            "rush_tds", "targets", "receptions", "rec_yds", "rec_tds"}


def stat_sql(expr: str, allowed: set[str], alias: str = "") -> str:
    """'pts+reb+ast' -> '(COALESCE(g.pts,0)+...)'. Whitelisted - never interpolates user input."""
    parts = expr.split("+")
    if not all(re.fullmatch(r"[a-z0-9_]+", p) and p in allowed for p in parts):
        raise ValueError(f"bad stat expr {expr}")
    pre = f"{alias}." if alias else ""
    return "(" + "+".join(f"COALESCE({pre}{p},0)" for p in parts) + ")"


def _f(v):
    return None if v is None else float(v)


_CV_CACHE: dict[tuple, float] = {}


async def league_cv(s: AsyncSession, sport: str, col: str, position: str | None = None) -> float:
    """League-typical coefficient of variation for this stat (players with >= 10 games).
    Individual sample sds from ~15 games are noisy and frequently too small, which makes tail
    probabilities (alt lines!) wildly over-confident; we shrink each player's sd toward this."""
    key = (sport, col, position)
    if key not in _CV_CACHE:
        if sport == "NBA":
            sql = f"""SELECT AVG(sd/mu) FROM (SELECT STDDEV({col}) sd, AVG({col}) mu FROM nba_player_gamelogs g
                      WHERE g.minutes >= 15 GROUP BY player_id HAVING COUNT(*) >= 10 AND AVG({col}) >= 1) x"""
            params = {}
        else:
            sql = f"""SELECT AVG(sd/mu) FROM (SELECT STDDEV({col}) sd, AVG({col}) mu FROM nfl_player_games w
                      WHERE w.position = :pos GROUP BY player_id HAVING COUNT(*) >= 6 AND AVG({col}) >= 1) x"""
            params = {"pos": position}
        v = (await s.execute(text(sql), params)).scalar()
        _CV_CACHE[key] = float(v) if v else 0.5
    return _CV_CACHE[key]


def shrink_sd(sd_player: float, n: int, mean: float, cv_league: float, k: float = 10.0) -> float:
    """Pool variances: n games of the player's own spread vs k pseudo-games at the league CV."""
    sd_player = 0.0 if sd_player != sd_player else sd_player   # NaN-safe
    return ((n * sd_player**2 + k * (cv_league * mean) ** 2) / (n + k)) ** 0.5


@dataclass
class Projection:
    player_id: int
    game_id: int
    market_key: str
    dist: PropDistribution
    mean: float
    std: float
    sample_size: int
    history: np.ndarray
    weights: np.ndarray
    inputs: dict
    flags: list[str] = field(default_factory=list)

    def probs(self, line: float | None, blend: bool = True) -> dict:
        over, under, push = prob_over_under(self.dist, line)
        out = {"model_over": over, "model_under": under, "push": push}
        if blend and len(self.history) >= MIN_GAMES:
            hr, n_eff = empirical_hit_rate(self.history, line, self.weights)
            act = 1 - push
            over_c = blend_with_history(over / act if act else 0.5, hr, n_eff)
            out.update(hist_over_rate=hr, hist_n_eff=n_eff, over=over_c * act, under=(1 - over_c) * act)
        else:
            out.update(over=over, under=under)
        return out


async def _game_ctx(s: AsyncSession, game_id: int, player_id: int) -> dict:
    row = (await s.execute(text("""
        SELECT g.id, g.sport, g.season, g.start_time, ht.abbr AS home, at.abbr AS away,
               p.team_abbr, p.position, p.full_name, COALESCE(v.indoor, false) AS indoor, v.altitude_ft
        FROM games g JOIN teams ht ON ht.id=g.home_team_id JOIN teams at ON at.id=g.away_team_id
        JOIN players p ON p.id=:pid LEFT JOIN venues v ON v.id=g.venue_id
        WHERE g.id=:gid"""), {"gid": game_id, "pid": player_id})).mappings().first()
    if not row:
        raise LookupError("game/player not found")
    ctx = dict(row)
    if ctx["team_abbr"] not in (ctx["home"], ctx["away"]):
        raise LookupError(f"{ctx['full_name']} ({ctx['team_abbr']}) not on either team")
    ctx["opp"] = ctx["away"] if ctx["team_abbr"] == ctx["home"] else ctx["home"]
    return ctx


async def latest_weather(s: AsyncSession, game_id: int) -> dict | None:
    r = (await s.execute(text("""SELECT temp_f, wind_mph, wind_gust_mph, precip_prob, precip_in, short_forecast,
        source, captured_at FROM weather_forecasts WHERE game_id=:g ORDER BY captured_at DESC LIMIT 1"""),
        {"g": game_id})).mappings().first()
    return dict(r) if r else None


async def latest_injury(s: AsyncSession, player_id: int) -> dict | None:
    r = (await s.execute(text("""SELECT status, injury, comment, captured_at FROM injury_reports
        WHERE player_id=:p AND captured_at > now() - interval '36 hours' ORDER BY captured_at DESC LIMIT 1"""),
        {"p": player_id})).mappings().first()
    return dict(r) if r else None


async def team_rest(s: AsyncSession, game_id: int, team_abbr: str) -> dict | None:
    r = (await s.execute(text("""SELECT tr.* FROM team_rest tr JOIN teams t ON t.id=tr.team_id
        WHERE tr.game_id=:g AND t.abbr=:a"""), {"g": game_id, "a": team_abbr})).mappings().first()
    return dict(r) if r else None


# ---------------------------------------------------------------------------
# NBA
# ---------------------------------------------------------------------------
async def _nba(s: AsyncSession, ctx: dict, player_id: int, expr: str, family: str) -> Projection:
    col = stat_sql(expr, NBA_COLS, "g")
    hist = (await s.execute(text(f"""
        SELECT g.game_id, g.game_date, g.minutes, {col} AS v FROM nba_player_gamelogs g
        WHERE g.player_id=:p AND g.minutes > 0 AND g.game_date < :d ORDER BY g.game_date DESC LIMIT :n"""),
        {"p": player_id, "n": LOOKBACK, "d": ctx["start_time"].date()})).all()[::-1]     # oldest -> newest
    if len(hist) < MIN_GAMES:
        raise ValueError(f"only {len(hist)} games of history")
    mins = np.array([h.minutes for h in hist], float)
    vals = np.array([h.v for h in hist], float)
    flags: list[str] = []

    # per-minute rate x minutes decouples role (minutes) from efficiency (rate)
    rate_mu, _ = B.ewm_mean_std(vals / mins)
    min_mu, _ = B.ewm_mean_std(mins)

    rest = await team_rest(s, ctx["id"], ctx["team_abbr"]) or {}
    m_fac, f = B.nba_minutes_factor(rest.get("is_b2b", False), rest.get("is_3in4", False))
    flags += f

    # opponent: stat allowed per team-game vs league average (current season, with prior-season fallback in k)
    opp = (await s.execute(text(f"""
        WITH per_game AS (SELECT g.opp_abbr, g.game_id, SUM({col}) AS v FROM nba_player_gamelogs g
                          WHERE g.season=:season GROUP BY 1,2)
        SELECT AVG(v) FILTER (WHERE opp_abbr=:opp) AS opp_pg, COUNT(*) FILTER (WHERE opp_abbr=:opp) AS n,
               AVG(v) AS lg FROM per_game"""), {"season": ctx["season"], "opp": ctx["opp"]})).first()
    o_fac = B.opponent_factor(float(opp.opp_pg or 0), float(opp.lg or 0), int(opp.n or 0))

    # pace: possessions ~ FGA + 0.44*FTA - OREB + TOV, from our own box scores (no stats.nba.com needed)
    pace = (await s.execute(text("""
        WITH tp AS (SELECT team_abbr, game_id, SUM(fga + 0.44*fta - oreb + tov) poss FROM nba_player_gamelogs
                    WHERE season=:season GROUP BY 1,2)
        SELECT AVG(poss) FILTER (WHERE team_abbr=:tm) tp, AVG(poss) FILTER (WHERE team_abbr=:opp) op, AVG(poss) lg
        FROM tp"""), {"season": ctx["season"], "tm": ctx["team_abbr"], "opp": ctx["opp"]})).first()
    # pace is partly captured by the opponent factor already -> apply half the deviation
    p_fac = 1 + 0.5 * (B.nba_pace_factor(_f(pace.tp), _f(pace.op), _f(pace.lg)) - 1)

    wwf, usage_note, mate_games = await _nba_usage_shift(s, ctx, player_id, col)
    u_fac = 1.0
    if usage_note:
        # Avoid double counting: if recent games were ALREADY played without the teammate, the
        # EWMA baseline has partly absorbed the shift. With s = EWMA weight share of 'without'
        # games: factor = wwf / ((1-s) + s*wwf)   (-> wwf when s=0, -> 1 when s=1).
        w_hist = B.ewm_weights(len(hist))
        absent = np.array([h.game_id not in mate_games for h in hist], float)
        share = float(np.sum(w_hist * absent) / np.sum(w_hist))
        u_fac = wwf / ((1 - share) + share * wwf)
        flags.append(f"{usage_note}; baseline already {share:.0%} without him -> applied {u_fac:.3f}")

    mean = rate_mu * min_mu * m_fac * o_fac * p_fac * u_fac
    w = B.ewm_weights(len(vals))
    _, raw_sd = B.ewm_mean_std(vals)
    raw_mu = float(np.average(vals, weights=w))
    sd_player = raw_sd * mean / raw_mu if raw_mu else raw_sd
    cv = await league_cv(s, "NBA", col)
    sd = max(shrink_sd(sd_player, len(vals), mean, cv), 0.6 * mean**0.5, 0.3)
    return Projection(player_id, ctx["id"], "", fit(family, mean, sd), mean, sd, len(vals), vals, w, dict(
        rate_per_min=rate_mu, proj_minutes=min_mu * m_fac, minutes_factor=m_fac, opp_factor=o_fac,
        opp_allowed_pg=_f(opp.opp_pg), league_allowed_pg=_f(opp.lg), pace_factor=p_fac, usage_factor=u_fac,
        sd_player=sd_player, league_cv=cv,
        rest=rest and {k: rest[k] for k in ("days_rest", "is_b2b", "is_3in4")}), flags)


async def _nba_usage_shift(s: AsyncSession, ctx: dict, player_id: int, col: str) -> tuple[float, str | None, set]:
    """If a 24+ mpg teammate is OUT/DOUBTFUL tonight, compare this player's per-minute production
    in games that teammate missed vs played. Uses the single biggest absence (no compounding)."""
    out = (await s.execute(text("""
        WITH inj AS (SELECT DISTINCT ON (player_id) player_id, status FROM injury_reports
                     WHERE sport='NBA' AND player_id IS NOT NULL AND captured_at > now() - interval '36 hours'
                     ORDER BY player_id, captured_at DESC)
        SELECT p.id, p.full_name, AVG(g.minutes) AS mpg FROM inj JOIN players p ON p.id=inj.player_id
        JOIN nba_player_gamelogs g ON g.player_id=p.id AND g.season=:season
        WHERE inj.status IN ('Out','Doubtful') AND p.team_abbr=:tm AND p.id<>:pid
        GROUP BY p.id, p.full_name HAVING AVG(g.minutes) >= 24 ORDER BY mpg DESC LIMIT 1"""),
        {"season": ctx["season"], "tm": ctx["team_abbr"], "pid": player_id})).first()
    if not out:
        return 1.0, None, set()
    r = (await s.execute(text(f"""
        WITH mine AS (SELECT g.game_id, g.minutes, {col} AS v FROM nba_player_gamelogs g
                      WHERE g.player_id=:pid AND g.season=:season AND g.minutes>0),
             his AS (SELECT game_id FROM nba_player_gamelogs WHERE player_id=:tid AND season=:season AND minutes>0)
        SELECT SUM(v) FILTER (WHERE h.game_id IS NOT NULL) / NULLIF(SUM(minutes) FILTER (WHERE h.game_id IS NOT NULL),0) AS rw,
               COUNT(*) FILTER (WHERE h.game_id IS NOT NULL) AS nw,
               SUM(v) FILTER (WHERE h.game_id IS NULL) / NULLIF(SUM(minutes) FILTER (WHERE h.game_id IS NULL),0) AS rwo,
               COUNT(*) FILTER (WHERE h.game_id IS NULL) AS nwo
        FROM mine m LEFT JOIN his h USING (game_id)"""),
        {"pid": player_id, "tid": out.id, "season": ctx["season"]})).first()
    fac = B.with_without_factor(float(r.rw or 0), int(r.nw or 0), float(r.rwo or 0), int(r.nwo or 0))
    mate_games = set((await s.execute(text("SELECT game_id FROM nba_player_gamelogs WHERE player_id=:t AND minutes>0"),
                                      {"t": out.id})).scalars())
    return fac, (f"TEAMMATE OUT: {out.full_name} ({out.mpg:.0f} mpg) -> with/without factor {fac:.3f} "
                 f"(n_without={r.nwo})"), mate_games


# ---------------------------------------------------------------------------
# NFL
# ---------------------------------------------------------------------------
async def _nfl(s: AsyncSession, ctx: dict, player_id: int, expr: str, family: str, market_key: str) -> Projection:
    col = stat_sql(expr, NFL_COLS, "w")
    hist = (await s.execute(text(f"""
        SELECT w.game_date, w.offense_snap_pct, w.target_share, {col} AS v FROM nfl_player_games w
        WHERE w.player_id=:p AND w.game_date < :d ORDER BY w.game_date DESC LIMIT 17"""),
        {"p": player_id, "d": ctx["start_time"].date()})).all()[::-1]
    if len(hist) < MIN_GAMES:
        raise ValueError(f"only {len(hist)} games of history")
    vals = np.array([h.v for h in hist], float)
    mu, sd = B.ewm_mean_std(vals, halflife=3.0)     # NFL roles change fast and samples are short
    flags: list[str] = []

    pos = ctx["position"] or "WR"
    opp = (await s.execute(text(f"""
        WITH per_game AS (SELECT w.opp_abbr, w.game_id, SUM({col}) AS v FROM nfl_player_games w
                          WHERE w.season=:season AND w.position=:pos GROUP BY 1,2)
        SELECT AVG(v) FILTER (WHERE opp_abbr=:opp) opp_pg, COUNT(*) FILTER (WHERE opp_abbr=:opp) n, AVG(v) lg
        FROM per_game"""), {"season": int(ctx["season"]), "pos": pos, "opp": ctx["opp"]})).first()
    o_fac = B.opponent_factor(float(opp.opp_pg or 0), float(opp.lg or 0), int(opp.n or 0), k=5.0)

    wx = await latest_weather(s, ctx["id"])
    w_fac, wflags = B.weather_factor(market_key, wx and wx["wind_mph"], wx and wx["precip_prob"],
                                     wx and wx["temp_f"], "dome" if ctx["indoor"] else "open")
    flags += wflags

    rest = await team_rest(s, ctx["id"], ctx["team_abbr"]) or {}
    if rest.get("short_week"):
        flags.append("SHORT WEEK (TNF)")
    if rest.get("off_bye"):
        flags.append("OFF BYE")

    role = [h.offense_snap_pct for h in hist if h.offense_snap_pct is not None]
    label = "snap%"
    if len(role) < 6:
        role, label = [h.target_share for h in hist if h.target_share is not None], "target share"
    if len(role) >= 6 and abs(np.mean(role[-3:]) - np.mean(role[:-3])) > 0.10:
        flags.append(f"ROLE CHANGE: {label} {np.mean(role[:-3]):.0%} -> {np.mean(role[-3:]):.0%} (last 3)")

    mean = mu * o_fac * w_fac
    cv = await league_cv(s, "NFL", col, pos)
    sd = max(shrink_sd(sd * mean / mu if mu else sd, len(vals), mean, cv), 0.5 * mean**0.5, 0.25)
    w = B.ewm_weights(len(vals), 3.0)
    return Projection(player_id, ctx["id"], "", fit(family, mean, sd), mean, sd, len(vals), vals, w, dict(
        ewm_mean=mu, opp_factor=o_fac, opp_allowed_pg=_f(opp.opp_pg), league_allowed_pg=_f(opp.lg), position=pos,
        weather_factor=w_fac, weather=wx and {k: wx[k] for k in ("temp_f", "wind_mph", "precip_prob", "source")},
        rest=rest and {k: rest[k] for k in ("days_rest", "off_bye", "short_week")}), flags)


# ---------------------------------------------------------------------------
async def project(s: AsyncSession, game_id: int, player_id: int, market_key: str) -> Projection:
    _key, sport, _label, expr, family = MARKET_INDEX[market_key]
    ctx = await _game_ctx(s, game_id, player_id)
    proj = await (_nba(s, ctx, player_id, expr, family) if sport == "NBA"
                  else _nfl(s, ctx, player_id, expr, family, market_key))
    proj.market_key = market_key
    inj = await latest_injury(s, player_id)
    if inj and inj["status"] not in ("Active",):
        proj.flags.insert(0, f"PLAYER STATUS: {inj['status']} ({inj.get('injury') or 'n/a'}) - "
                              "DK voids props if he doesn't play, but minutes may be limited")
        proj.inputs["injury"] = {k: (str(v) if v is not None else None) for k, v in inj.items()}
    return proj
