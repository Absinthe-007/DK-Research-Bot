"""
Stat board: the expected box-score line for EVERY rostered player in upcoming games.

No prices are shown or needed. For each player and stat:

  NBA   mean = EWMA(per-minute rate) x projected minutes x opponent factor x pace factor x usage shift
  NFL   mean = EWMA(per-game value)  x opponent (defense-vs-position) factor x weather factor

  range = 20th-80th percentile of the fitted distribution (Poisson / NegBin / Normal / Gamma), with
          the player's own spread shrunk toward the league-typical spread for that stat.

When DraftKings has a line on that player/stat, the scan has already computed a market-anchored
mean (model blended with the line); we use it because it is measurably more accurate, but the
price itself never appears anywhere.

Everything is computed in bulk with pandas (a handful of SQL queries per sport, not per player).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sqlalchemy import text

from app.db.base import session_scope
from app.engine import baselines as B
from app.engine.distributions import fit
from app.ingestion.common import tracked

log = logging.getLogger(__name__)

NBA_STATS = {   # stat -> (label, distribution family, DK market key used for anchoring)
    "minutes": ("MIN", "normal", None),
    "pts": ("PTS", "normal", "player_points"), "reb": ("REB", "negbin", "player_rebounds"),
    "ast": ("AST", "negbin", "player_assists"), "fg3m": ("3PM", "negbin", "player_threes"),
    "stl": ("STL", "poisson", "player_steals"), "blk": ("BLK", "poisson", "player_blocks"),
    "tov": ("TOV", "poisson", None), "pra": ("PRA", "normal", "player_points_rebounds_assists"),
}
NFL_STATS = {
    "completions": ("CMP", "normal", "player_pass_completions"), "pass_attempts": ("ATT", "normal", "player_pass_attempts"),
    "pass_yds": ("PASS YDS", "normal", "player_pass_yds"), "pass_tds": ("PASS TD", "poisson", "player_pass_tds"),
    "interceptions": ("INT", "poisson", "player_pass_interceptions"),
    "carries": ("CAR", "negbin", "player_rush_attempts"), "rush_yds": ("RUSH YDS", "gamma", "player_rush_yds"),
    "rush_tds": ("RUSH TD", "poisson", None),
    "targets": ("TGT", "negbin", None), "receptions": ("REC", "negbin", "player_receptions"),
    "rec_yds": ("REC YDS", "gamma", "player_reception_yds"), "rec_tds": ("REC TD", "poisson", None),
    "tackles": ("TKL", "negbin", None), "sacks": ("SACKS", "poisson", None), "qb_hits": ("QB HITS", "poisson", None),
    "passes_defended": ("PD", "poisson", None), "def_int": ("DEF INT", "poisson", None),
    "fgm": ("FGM", "poisson", None), "xpm": ("XPM", "poisson", None),
}
NFL_GROUPS = {
    "QB": ["completions", "pass_attempts", "pass_yds", "pass_tds", "interceptions", "carries", "rush_yds", "rush_tds"],
    "RB": ["carries", "rush_yds", "rush_tds", "targets", "receptions", "rec_yds", "rec_tds"],
    "REC": ["targets", "receptions", "rec_yds", "rec_tds", "carries", "rush_yds"],
    "DEF": ["tackles", "sacks", "qb_hits", "passes_defended", "def_int"],
    "K": ["fgm", "xpm"],
}
POS_GROUP = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "REC", "TE": "REC", "K": "K", "PK": "K"}
OUT_STATUSES = ("Out", "IR", "PUP", "Suspended")
NBA_LOOKBACK, NFL_LOOKBACK = 25, 17
NBA_MIN_TEAM_GP = 3         # once a team has played this many games, a player must have a role THIS season
NBA_ROLE_WINDOW = 5         # ...judged over the team's last N games
NBA_MIN_PARTICIPATION = 0.4  # skip players who got into fewer than this share of them (unless on the injury report)
NBA_PRIOR_GAMES = 3.0       # current-season minutes outweigh the long window after ~3 games
NFL_MIN_TEAM_GP, NFL_ROLE_WINDOW = 2, 3   # NFL: must show up in one of the team's last 3 box scores


def _group(pos: str | None) -> str:
    return POS_GROUP.get(pos or "", "DEF")


def _range(family: str, mean: float, sd: float) -> tuple[float, float]:
    try:
        d = fit(family, max(mean, 0.01), max(sd, 0.05))
        f = d._frozen()
        lo, hi = float(f.ppf(0.2)), float(f.ppf(0.8))
        return max(0.0, lo), max(0.0, hi)
    except Exception:
        return max(0.0, mean - sd), mean + sd


def _league_cv(df: pd.DataFrame, stat: str, min_games: int) -> float:
    g = df.groupby("player_id")[stat].agg(["mean", "std", "count"])
    g = g[(g["count"] >= min_games) & (g["mean"] >= 0.5)]
    return float((g["std"] / g["mean"]).mean()) if len(g) else 0.6


def _opp_factors(df: pd.DataFrame, stats: list[str], by_pos: bool, k: float) -> dict:
    """{(opp, pos, stat): factor} from 'allowed per team-game vs league average', shrunk to 1."""
    keys = ["opp_abbr", "game_id"] + (["pgroup"] if by_pos else [])
    per = df.groupby(keys)[stats].sum().reset_index()
    out = {}
    grp_cols = ["pgroup"] if by_pos else []
    lg = per.groupby(grp_cols)[stats].mean() if grp_cols else per[stats].mean()
    for keyvals, sub in per.groupby(["opp_abbr"] + grp_cols):
        opp = keyvals[0] if isinstance(keyvals, tuple) else keyvals
        pos = keyvals[1] if by_pos else None
        for st in stats:
            league = (lg.loc[pos, st] if by_pos else lg[st])
            out[(opp, pos, st)] = B.opponent_factor(float(sub[st].mean()), float(league or 0), len(sub), k)
    return out


async def _context(s, sport: str, hours: int, as_of: datetime | None = None):
    """Upcoming games + who's on those teams. `as_of` (backtests only) rewinds the clock: games that
    started after it are projected using only data that existed before it."""
    now = as_of or datetime.now(timezone.utc)
    games = pd.DataFrame((await s.execute(text("""
        SELECT g.id game_id, g.start_time, g.season, ht.abbr home, at.abbr away, COALESCE(v.indoor,false) indoor
        FROM games g JOIN teams ht ON ht.id=g.home_team_id JOIN teams at ON at.id=g.away_team_id
        LEFT JOIN venues v ON v.id=g.venue_id
        WHERE g.sport=:sp AND (:bt OR NOT g.completed) AND g.start_time > :now AND g.start_time < :hi AND g.season_type IN (2,3)
        ORDER BY g.start_time"""), {"sp": sport, "now": now, "hi": now + timedelta(hours=hours), "bt": as_of is not None})).mappings().all())
    if games.empty:
        return games, None, None, None, None
    teams = sorted(set(games.home) | set(games.away))
    players = pd.DataFrame((await s.execute(text("""
        SELECT id player_id, full_name, position, team_abbr, headshot FROM players
        WHERE sport=:sp AND team_abbr = ANY(:t)"""), {"sp": sport, "t": teams})).mappings().all())
    inj = pd.DataFrame((await s.execute(text("""
        SELECT DISTINCT ON (player_id) player_id, status, injury FROM injury_reports
        WHERE sport=:sp AND player_id IS NOT NULL AND captured_at > CAST(:now AS timestamptz) - interval '36 hours' AND captured_at <= :now
        ORDER BY player_id, captured_at DESC"""), {"sp": sport, "now": now})).mappings().all())
    rest = pd.DataFrame((await s.execute(text("""
        SELECT tr.game_id, t.abbr, tr.is_b2b, tr.is_3in4, tr.short_week, tr.off_bye FROM team_rest tr
        JOIN teams t ON t.id=tr.team_id WHERE tr.game_id = ANY(:g)"""), {"g": games.game_id.tolist()})).mappings().all())
    anchors = pd.DataFrame((await s.execute(text("""
        SELECT game_id, player_id, market_key, (inputs->>'blended_mean')::float blended FROM projections
        WHERE game_id = ANY(:g) AND inputs ? 'blended_mean' AND NOT (inputs ? 'suppressed')"""),
        {"g": games.game_id.tolist()})).mappings().all())
    if as_of is not None:           # backtest: score the raw model, not the market-blended mean
        anchors = anchors.iloc[0:0]
    return games, players, inj, rest, anchors


def _anchor_map(anchors: pd.DataFrame) -> dict:
    if anchors is None or anchors.empty:
        return {}
    return {(r.game_id, r.player_id, r.market_key): r.blended for r in anchors.itertuples() if r.blended is not None}


def _status_map(inj: pd.DataFrame) -> dict:
    return {} if inj is None or inj.empty else {r.player_id: (r.status, r.injury) for r in inj.itertuples()}


def _rows_for(sport, game, p, stat_means, sds, families, last5, season_avg, n, anchors, labels_mk):
    out = []
    for st, mean in stat_means.items():
        mk = labels_mk.get(st)
        anchored = False
        model_mean = mean
        if mk and (game, p, mk) in anchors:
            mean, anchored = float(anchors[(game, p, mk)]), True
        lo, hi = _range(families[st], mean, sds[st] * (mean / model_mean if model_mean > 0 else 1))
        out.append(dict(sport=sport, game_id=int(game), player_id=int(p), stat=st, mean=round(mean, 2), lo=round(lo, 1),
                        hi=round(hi, 1), model_mean=round(model_mean, 2), anchored=anchored,
                        last5=None if last5.get(st) is None else round(last5[st], 2),
                        season_avg=None if season_avg.get(st) is None else round(season_avg[st], 2), sample=int(n)))
    return out


# ---------------------------------------------------------------------------
async def _nba(s, as_of: datetime | None = None) -> list[dict]:
    games, players, inj, rest, anchors = await _context(s, "NBA", 36, as_of)
    if games.empty or players.empty:
        return []
    ids = players.player_id.tolist()
    logs = pd.DataFrame((await s.execute(text("""
        SELECT player_id, game_id, game_date, season, team_abbr, opp_abbr, minutes, pts, reb, ast, fg3m, stl, blk, tov,
               fga, fta, oreb FROM nba_player_gamelogs
        WHERE (player_id = ANY(:p) OR game_date > CAST(:now AS date) - 120) AND game_date < CAST(:now AS date)
        ORDER BY game_date"""), {"p": ids, "now": as_of or datetime.now(timezone.utc)})).mappings().all())
    if logs.empty:
        return []
    logs["pra"] = logs.pts + logs.reb + logs.ast
    stats = [k for k in NBA_STATS if k != "minutes"]
    season = games.season.iloc[0]
    cur = logs[logs.season == season]
    opp_src = cur if cur.game_id.nunique() >= 60 else logs[logs.game_date >= logs.game_date.max() - pd.Timedelta(days=150)]
    oppf = _opp_factors(opp_src, stats, False, 8.0)
    poss = opp_src.assign(poss=opp_src.fga + 0.44 * opp_src.fta - opp_src.oreb + opp_src.tov) \
        .groupby(["team_abbr", "game_id"]).poss.sum().groupby("team_abbr").mean()
    lg_pace = float(poss.mean()) if len(poss) else None
    cvs = {st: _league_cv(logs, st, 10) for st in stats + ["minutes"]}
    status = _status_map(inj)
    anchor = _anchor_map(anchors)
    mk_of = {k: v[2] for k, v in NBA_STATS.items()}
    fam = {k: v[1] for k, v in NBA_STATS.items()}
    rest_of = {(r.game_id, r.abbr): r for r in rest.itertuples()} if not rest.empty else {}

    # biggest absence per team (24+ mpg, Out/Doubtful) for the with/without usage shift
    mpg = logs[logs.season.isin([season, logs.season.max()])].groupby("player_id").minutes.mean()
    absent = {}
    for p in players.itertuples():
        st_ = status.get(p.player_id, ("", ""))[0]
        if st_ in OUT_STATUSES + ("Doubtful",) and mpg.get(p.player_id, 0) >= 24:
            if p.team_abbr not in absent or mpg[p.player_id] > mpg[absent[p.team_abbr]]:
                absent[p.team_abbr] = p.player_id
    games_of = logs.groupby("player_id").game_id.apply(set).to_dict()
    # current-season role: which of his team's games this season he actually got into
    team_games = cur.groupby("team_abbr").apply(lambda d: list(dict.fromkeys(d.sort_values("game_date").game_id)))
    cur_games = cur[cur.minutes > 0].groupby("player_id").game_id.apply(set).to_dict()

    rows, by_player = [], {pid: g for pid, g in logs[logs.player_id.isin(ids)].groupby("player_id")}
    for g in games.itertuples():
        for side, opp in ((g.home, g.away), (g.away, g.home)):
            for p in players[players.team_abbr == side].itertuples():
                if status.get(p.player_id, ("",))[0] in OUT_STATUSES or p.player_id not in by_player:
                    continue
                h = by_player[p.player_id].tail(NBA_LOOKBACK)
                h = h[h.minutes > 0]
                if len(h) < 3 or (pd.Timestamp(g.start_time).date() - h.game_date.max()).days > 240:
                    continue
                mins = h.minutes.to_numpy(float)
                w = B.ewm_weights(len(h))
                min_mu, _ = B.ewm_mean_std(mins)
                # The 25-game window reaches back into last season (incl. April tank/rest games),
                # so early in the year it says nothing about TODAY's rotation. Lean on this season.
                tg = team_games.get(side, [])
                mine = cur_games.get(p.player_id, set())
                if len(tg) >= NBA_MIN_TEAM_GP:
                    recent = tg[-NBA_ROLE_WINDOW:]
                    played_recent = sum(gid in mine for gid in recent) / len(recent)
                    if not mine or (played_recent < NBA_MIN_PARTICIPATION and status.get(p.player_id) is None):
                        continue          # not in the rotation this season (and not just back from injury)
                if mine:
                    hc = h[h.game_id.isin(mine)]
                    cur_mu, _ = B.ewm_mean_std(hc.minutes.to_numpy(float))
                    min_mu = B.shrink(cur_mu, len(hc), min_mu, NBA_PRIOR_GAMES)
                r = rest_of.get((g.game_id, side))
                m_fac, _ = B.nba_minutes_factor(bool(r and r.is_b2b), bool(r and r.is_3in4))
                pace_f = 1.0
                if lg_pace and side in poss and opp in poss:
                    pace_f = 1 + 0.5 * (B.nba_pace_factor(poss[side], poss[opp], lg_pace) - 1)
                mate = absent.get(side)
                share = 0.0
                if mate and mate != p.player_id:
                    mg = games_of.get(mate, set())
                    share = float(np.sum(w * np.array([gid not in mg for gid in h.game_id])) / np.sum(w))
                if min_mu * m_fac < 8:      # deep bench: not worth projecting
                    continue
                means, sds = {"minutes": min_mu * m_fac}, {}
                _, sd_min = B.ewm_mean_std(mins)
                sds["minutes"] = B.shrink(sd_min if sd_min == sd_min else 0, len(h), cvs["minutes"] * min_mu, 10) if min_mu else 1
                for st in stats:
                    vals = h[st].to_numpy(float)
                    rate, _ = B.ewm_mean_std(vals / mins)
                    u = 1.0
                    if mate and mate != p.player_id:
                        mg = games_of.get(mate, set())
                        with_m = h[h.game_id.isin(mg)]
                        wo = h[~h.game_id.isin(mg)]
                        rw = with_m[st].sum() / with_m.minutes.sum() if len(with_m) and with_m.minutes.sum() else 0
                        rwo = wo[st].sum() / wo.minutes.sum() if len(wo) and wo.minutes.sum() else 0
                        wwf = B.with_without_factor(rw, len(with_m), rwo, len(wo))
                        u = wwf / ((1 - share) + share * wwf)
                    mean = max(0.0, rate * min_mu * m_fac * oppf.get((opp, None, st), 1.0) * pace_f * u)
                    _, sd_raw = B.ewm_mean_std(vals)
                    raw_mu = float(np.average(vals, weights=w)) or 1e-6
                    sds[st] = max(B_shrink_sd(sd_raw * mean / raw_mu, len(h), mean, cvs[st]), 0.5 * mean ** 0.5)
                    means[st] = mean
                last5 = {st: float(h[st].tail(5).mean()) for st in means}
                seas = h[h.season == season]
                savg = {st: (float(seas[st].mean()) if len(seas) else None) for st in means}
                rows += _rows_for("NBA", g.game_id, p.player_id, means, sds, fam, last5, savg, len(h), anchor, mk_of)
    return rows


def _nfl_relevant(grp: str, m: dict) -> bool:
    """Skip deep reserves / special-teamers so the board only lists players with a real role."""
    g = m.get
    if grp == "QB":
        return g("pass_attempts", 0) >= 8
    if grp == "RB":
        return g("carries", 0) >= 2 or g("targets", 0) >= 1.5
    if grp == "REC":
        return g("targets", 0) >= 1.5
    if grp == "DEF":
        return g("tackles", 0) >= 1.5 or g("sacks", 0) >= 0.15
    return True


def _infer_group(h: pd.DataFrame) -> str:
    """Position group from what a player actually does, for the ~10% with no position on file."""
    m = h[["pass_attempts", "carries", "targets", "fgm", "xpm", "tackles"]].mean()
    if m.pass_attempts >= 5:
        return "QB"
    if m.fgm + m.xpm >= 0.5:
        return "K"
    if m.carries >= 2 and m.carries >= m.targets:
        return "RB"
    if m.targets >= 1:
        return "REC"
    return "DEF"


def _starting_qbs(logs: pd.DataFrame, cur: pd.DataFrame, players: pd.DataFrame, status: dict) -> dict:
    """team -> {player_ids} of QBs who may start: the one with the most pass attempts over the team's
    last two games (on the roster, not ruled out), plus a regular starter coming back from injury
    (on the report as Questionable/Probable/Active and averaging 20+ attempts this season)."""
    out: dict[str, set] = {}
    roster = players.set_index("player_id").team_abbr.to_dict()
    src = cur if not cur.empty else logs
    ok = lambda pid: status.get(pid, ("",))[0] not in OUT_STATUSES + ("Doubtful",)
    for team, d in src[src.pgroup == "QB"].groupby("team_abbr"):
        recent = list(dict.fromkeys(src[src.team_abbr == team].sort_values("game_date").game_id))[-2:]
        att = d[d.game_id.isin(recent)].groupby("player_id").pass_attempts.sum().sort_values(ascending=False)
        picks = set()
        for pid in att.index:
            if roster.get(pid) == team and ok(pid):
                picks.add(int(pid))
                break
        for pid, mu in d.groupby("player_id").pass_attempts.mean().items():
            if roster.get(pid) == team and pid in status and ok(pid) and mu >= 20:
                picks.add(int(pid))
        out[team] = picks
    return out


def B_shrink_sd(sd_player: float, n: int, mean: float, cv: float, k: float = 10.0) -> float:
    sd_player = 0.0 if sd_player != sd_player else sd_player
    return ((n * sd_player ** 2 + k * (cv * mean) ** 2) / (n + k)) ** 0.5


# ---------------------------------------------------------------------------
async def _nfl(s, as_of: datetime | None = None) -> list[dict]:
    games, players, inj, rest, anchors = await _context(s, "NFL", 24 * 8, as_of)
    if games.empty or players.empty:
        return []
    from app.engine.projector import latest_weather
    stats = list(NFL_STATS)
    ids = players.player_id.tolist()
    logs = pd.DataFrame((await s.execute(text(f"""
        SELECT player_id, game_id, game_date, season, team_abbr, opp_abbr, position, {", ".join(stats)}
        FROM nfl_player_games
        WHERE (player_id = ANY(:p) OR game_date > CAST(:now AS date) - 400) AND game_date < CAST(:now AS date)
        ORDER BY game_date"""), {"p": ids, "now": as_of or datetime.now(timezone.utc)})).mappings().all())
    if logs.empty:
        return []
    # ESPN box scores carry no position, so take it from the player table (all players, not just this
    # slate's - the defense-vs-position factors need every opponent's games). Without this every row
    # fell into "DEF" and the opponent adjustment silently became 1.0 for every QB/RB/WR/TE.
    pos_of = dict((await s.execute(text(
        "SELECT id, position FROM players WHERE sport='NFL' AND position IS NOT NULL AND position <> ''"))).all())
    logs["position"] = logs.player_id.map(pos_of).fillna(logs.position)
    inferred = logs.groupby("player_id").apply(_infer_group).to_dict()
    logs["pgroup"] = [(_group(ps) if ps else inferred.get(pid, "DEF")) for pid, ps in zip(logs.player_id, logs.position)]
    season = int(games.season.iloc[0])
    cur = logs[logs.season == season]
    oppf = _opp_factors(cur if cur.game_id.nunique() >= 40 else logs, stats, True, 5.0)
    cvs = {st: _league_cv(logs, st, 6) for st in stats}
    status = _status_map(inj)
    anchor = _anchor_map(anchors)
    mk_of = {k: v[2] for k, v in NFL_STATS.items()}
    fam = {k: v[1] for k, v in NFL_STATS.items()}
    weather_mk = {"pass_yds": "player_pass_yds", "completions": "player_pass_completions", "pass_attempts": "player_pass_attempts",
                  "pass_tds": "player_pass_tds", "rec_yds": "player_reception_yds", "receptions": "player_receptions",
                  "rush_yds": "player_rush_yds", "carries": "player_rush_attempts"}
    by_player = {pid: g for pid, g in logs[logs.player_id.isin(ids)].groupby("player_id")}
    team_games = cur.groupby("team_abbr").apply(lambda d: list(dict.fromkeys(d.sort_values("game_date").game_id)))
    cur_games = cur.groupby("player_id").game_id.apply(set).to_dict()
    starter_qb = _starting_qbs(logs, cur, players, status)

    rows = []
    for g in games.itertuples():
        wx = await latest_weather(s, int(g.game_id))
        for side, opp in ((g.home, g.away), (g.away, g.home)):
            for p in players[players.team_abbr == side].itertuples():
                if status.get(p.player_id, ("",))[0] in OUT_STATUSES or p.player_id not in by_player:
                    continue
                h = by_player[p.player_id].tail(NFL_LOOKBACK)
                if len(h) < 2 or (pd.Timestamp(g.start_time).date() - h.game_date.max()).days > 400:
                    continue
                grp = _group(p.position) if p.position else h.pgroup.iloc[-1]
                # must have a role THIS season: appeared in one of the team's last few games (box scores
                # only list players who recorded a stat), unless he's on the injury report (returning)
                tg = team_games.get(side, [])
                mine = cur_games.get(p.player_id, set())
                if len(tg) >= NFL_MIN_TEAM_GP and not (set(tg[-NFL_ROLE_WINDOW:]) & mine) \
                        and status.get(p.player_id) is None:
                    continue
                if grp == "QB" and side in starter_qb and p.player_id not in starter_qb[side]:
                    continue              # backups only get a line if the starter is out
                wanted = [st for st in NFL_GROUPS[grp] if h[st].sum() > 0 or st in NFL_GROUPS[grp][:3]]
                if grp == "REC":
                    wanted = [st for st in wanted if st not in ("carries", "rush_yds") or h.carries.mean() >= 0.5]
                means, sds = {}, {}
                for st in wanted:
                    vals = h[st].to_numpy(float)
                    mu, sd = B.ewm_mean_std(vals, halflife=3.0)
                    wf, _ = B.weather_factor(weather_mk.get(st, ""), wx and wx["wind_mph"], wx and wx["precip_prob"],
                                             wx and wx["temp_f"], "dome" if g.indoor else "open")
                    mean = max(0.0, mu * oppf.get((opp, grp, st), 1.0) * wf)
                    sds[st] = max(B_shrink_sd(sd * mean / mu if mu else 0, len(h), mean, cvs[st], 8.0), 0.4 * mean ** 0.5, 0.05)
                    means[st] = mean
                if not means or not _nfl_relevant(grp, means):
                    continue
                last5 = {st: float(h[st].tail(5).mean()) for st in means}
                seas = h[h.season == season]
                savg = {st: (float(seas[st].mean()) if len(seas) else None) for st in means}
                rows += _rows_for("NFL", g.game_id, p.player_id, means, sds, fam, last5, savg, len(h), anchor, mk_of)
                # anytime-TD probability as its own "stat" (0-1)
                td_mu = means.get("rush_tds", 0) + means.get("rec_tds", 0)
                if td_mu > 0:
                    pr = 1 - float(np.exp(-td_mu))
                    rows.append(dict(sport="NFL", game_id=int(g.game_id), player_id=int(p.player_id), stat="td_prob",
                                     mean=round(pr, 3), lo=0, hi=1, model_mean=round(pr, 3), anchored=False,
                                     last5=round(float((h.rush_tds + h.rec_tds).tail(5).gt(0).mean()), 2), season_avg=None,
                                     sample=len(h)))
    return rows


@tracked("statboard")
async def build(sport: str) -> int:
    async with session_scope() as s:
        rows = await (_nba(s) if sport == "NBA" else _nfl(s))
        if not rows:
            return 0
        gids = sorted({r["game_id"] for r in rows})
        await s.execute(text("DELETE FROM stat_projections WHERE game_id = ANY(:g) AND actual IS NULL"), {"g": gids})
        from app.db.base import upsert
        from app.db.models import StatProjection
        await upsert(s, StatProjection, rows, ["game_id", "player_id", "stat"])
        return len(rows)


@tracked("statboard_grade")
async def grade(sport: str) -> int:
    """Fill `actual` for finished games (players with no box-score line are left ungraded)."""
    async with session_scope() as s:
        if sport == "NBA":
            sql = """UPDATE stat_projections sp SET actual = CASE sp.stat WHEN 'pra' THEN l.pts+l.reb+l.ast
                       WHEN 'minutes' THEN l.minutes WHEN 'pts' THEN l.pts WHEN 'reb' THEN l.reb WHEN 'ast' THEN l.ast
                       WHEN 'fg3m' THEN l.fg3m WHEN 'stl' THEN l.stl WHEN 'blk' THEN l.blk WHEN 'tov' THEN l.tov END
                     FROM nba_player_gamelogs l JOIN games g ON g.id=l.game_id
                     WHERE sp.sport='NBA' AND sp.actual IS NULL AND l.game_id=sp.game_id AND l.player_id=sp.player_id"""
        else:
            cases = " ".join(f"WHEN '{c}' THEN l.{c}" for c in NFL_STATS)
            sql = f"""UPDATE stat_projections sp SET actual = CASE sp.stat {cases}
                        WHEN 'td_prob' THEN CASE WHEN l.rush_tds + l.rec_tds > 0 THEN 1 ELSE 0 END END
                      FROM nfl_player_games l
                      WHERE sp.sport='NFL' AND sp.actual IS NULL AND l.game_id=sp.game_id AND l.player_id=sp.player_id"""
        res = await s.execute(text(sql))
        return res.rowcount or 0
