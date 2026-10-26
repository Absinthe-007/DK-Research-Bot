"""
+EV scanner: latest DraftKings prop prices x model projections -> edges, Kelly sizing, and
the `picks` ledger (flagged edges frozen at their last pre-game state for grading).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import delete, text

from app.config import get_settings
from app.db.base import session_scope
from app.db.models import Edge
from app.db.models import Projection as ProjRow
from app.engine.distributions import market_implied_mean, prob_over_under, refit_at_mean
from app.engine.ev import screen
from app.engine.kelly import BetCandidate, allocate
from app.engine.odds_math import devig_power
from app.engine.projector import MODEL_VERSION, project
from app.ingestion.common import tracked

log = logging.getLogger(__name__)

CURRENT_PROPS_SQL = """
SELECT * FROM current_props WHERE sport=:sport AND start_time > now()
ORDER BY game_id, player_id, market_key, kind, line
"""


def jsonable(d):
    from decimal import Decimal

    import numpy as np
    if isinstance(d, Decimal):
        return float(d)
    if isinstance(d, dict):
        return {k: jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple, set)):
        return [jsonable(v) for v in d]
    if isinstance(d, (np.floating, np.integer)):
        return d.item()
    if isinstance(d, np.ndarray):
        return d.tolist()
    if hasattr(d, "isoformat"):
        return d.isoformat()
    return d


@tracked("ev_scan")
async def scan(sport: str) -> int:
    st = get_settings()
    async with session_scope() as s:
        props = (await s.execute(text(CURRENT_PROPS_SQL), {"sport": sport})).mappings().all()
        markets: dict[tuple, dict] = defaultdict(dict)
        for p in props:
            markets[(p["game_id"], p["player_id"], p["market_key"], p["kind"], p["line"])][p["side"]] = p
        by_player: dict[tuple, list] = defaultdict(list)
        for (gid, pid, mkey, kind, line), sides in markets.items():
            by_player[(gid, pid, mkey)].append((kind, line, sides))

        # ---- pass 1: model projection + market-implied mean for every player/market
        proj_cache: dict[tuple, object] = {}
        anchors: dict[tuple, tuple] = {}
        for ck, entries in by_player.items():
            try:
                proj = await project(s, *ck)
            except Exception as e:
                log.debug("skip %s: %s", ck, e)
                proj_cache[ck] = None
                continue
            proj_cache[ck] = proj
            # main two-way line = the pair priced closest to 50/50 (estimated prices count: the LINE is still DK's)
            pairs = [(line, sides) for kind, line, sides in entries if kind == "ou" and "Over" in sides and "Under" in sides]
            if pairs:
                line0, sides0 = min(pairs, key=lambda x: abs(devig_power([x[1]["Over"]["price"], x[1]["Under"]["price"]])[0] - 0.5))
                fair_over = devig_power([sides0["Over"]["price"], sides0["Under"]["price"]])[0]
                m_mkt = market_implied_mean(proj.dist, line0, fair_over)
                if m_mkt is not None:
                    anchors[ck] = (m_mkt, line0)

        # ---- slate calibration: remove the model's SYSTEMATIC bias per market (e.g. a stale
        # early-season baseline that runs 4% low for everyone) so only player-specific
        # disagreement with the market survives. Median of market/model means, n >= 8.
        ratios: dict[str, list] = defaultdict(list)
        for ck, (m_mkt, _l) in anchors.items():
            if proj_cache[ck].mean > 0:
                ratios[ck[2]].append(m_mkt / proj_cache[ck].mean)
        calib = {m: float(np.clip(np.median(r), 0.8, 1.25)) for m, r in ratios.items() if len(r) >= 8}
        pooled = [x for r in ratios.values() for x in r]
        if len(pooled) >= 8:          # small slates (one MNF game): pool all markets for this sport
            for m in ratios:
                calib.setdefault(m, float(np.clip(np.median(pooled), 0.8, 1.25)))

        # ---- pass 2: blend, evaluate every line
        results = []
        for ck, entries in by_player.items():
            proj = proj_cache.get(ck)
            if proj is None:
                continue
            anchored = None
            if ck in anchors:
                m_mkt, line0 = anchors[ck]
                c = calib.get(ck[2], 1.0)
                blended = st.model_weight * proj.mean * c + (1 - st.model_weight) * m_mkt
                anchored = refit_at_mean(proj.dist, blended)
                proj.inputs.update(model_mean=proj.mean, slate_calibration=c, market_mean=m_mkt,
                                   blended_mean=blended, model_weight=st.model_weight, anchor_line=line0)
                gap = abs(proj.mean * c - m_mkt) / max(m_mkt, 0.5)
                if gap > st.max_model_market_gap:
                    # A 30%+ disagreement is almost always news/role the box scores can't see
                    # (minutes cap, new team, starter returning) - show it, never auto-flag it.
                    proj.flags.append(f"MODEL FAR FROM MARKET ({proj.mean * c:.1f} vs {m_mkt:.1f}) - likely role/news change")
                    proj.inputs["suppressed"] = True
            else:
                proj.flags.append("NO TWO-WAY MARKET TO ANCHOR - informational only")
            dist = anchored or proj.dist
            for kind, line, sides in entries:
                over_p, under_p, push = prob_over_under(dist, line)
                over, under = sides.get("Over"), sides.get("Under")
                fair = devig_power([over["price"], under["price"]]) if (over and under) else [None, None]
                for row, p_win, f in ((over, over_p, fair[0]), (under, under_p, fair[1])):
                    if row is None:
                        continue
                    flags = proj.flags + (["DK PRICE NOT PUBLISHED - assumed -115, check the app"]
                                          if row.get("price_estimated") else [])
                    res = screen(row["side"], line, row["price"], p_win, push, f, st.edge_threshold, flags)
                    if kind == "alt":
                        # one-sided ladders carry a bigger, unknowable hold -> stricter bar + price band
                        res.flagged = (res.edge >= st.alt_edge_threshold
                                       and st.alt_min_price <= row["price"] <= st.alt_max_price)
                    if anchored is None or proj.inputs.get("suppressed"):
                        res.flagged = False   # no market anchor / implausible gap: informational only
                    results.append((proj, row, kind, res))
            if anchored is not None:     # store the distribution actually used
                proj.dist, proj.mean, proj.std = anchored, anchored.mean, anchored.std

        # ---- persist projections + edges (latest scan only)
        await s.execute(delete(Edge).where(Edge.sport == sport))
        await s.execute(text("""DELETE FROM projections p USING games g WHERE p.game_id=g.id AND g.sport=:sp
                                AND p.model_version=:v AND g.start_time > now()"""), {"sp": sport, "v": MODEL_VERSION})
        proj_ids: dict[tuple, int] = {}
        for proj in filter(None, proj_cache.values()):
            row = ProjRow(game_id=proj.game_id, player_id=proj.player_id, market_key=proj.market_key,
                          model_version=MODEL_VERSION, mean=proj.mean, std=proj.std, dist_family=proj.dist.family,
                          dist_params=jsonable(proj.dist.params), sample_size=proj.sample_size,
                          inputs=jsonable({**proj.inputs, "flags": proj.flags, "recent": proj.history[-10:]}))
            s.add(row)
            await s.flush()
            proj_ids[(proj.game_id, proj.player_id, proj.market_key)] = row.id

        # One play per player+market: a ladder of correlated alt lines is ONE opinion, not five bets.
        best: dict[tuple, tuple] = {}
        for item in results:
            p, _row, _k, r = item
            key = (p.game_id, p.player_id, p.market_key)
            if r.flagged and (key not in best or r.ev_per_unit > best[key][3].ev_per_unit):
                best[key] = item
        keep = {id(v) for v in best.values()}
        for item in results:
            if item[3].flagged and id(item) not in keep:
                item[3].flagged = False
                item[3].flags = item[3].flags + ["SUPERSEDED by a better line for this player/market"]
        flagged = [r for r in results if r[3].flagged]
        sizing = allocate([BetCandidate(str(i), p.game_id, r.model_prob, r.price, r.push_prob)
                           for i, (p, _row, _k, r) in enumerate(flagged)],
                          st.bankroll, st.kelly_fraction, st.max_stake_pct, st.max_total_exposure_pct)
        flagged_ids = {id(x): str(i) for i, x in enumerate(flagged)}
        for item in results:
            p, row, kind, r = item
            k = sizing.get(flagged_ids.get(id(item), ""), {"kelly_full": 0.0, "stake_pct": 0.0})
            s.add(Edge(projection_id=proj_ids[(p.game_id, p.player_id, p.market_key)], prop_snapshot_id=row["id"],
                       sport=sport, kind=kind, side=r.side, line=r.line, price=r.price, model_prob=r.model_prob,
                       push_prob=r.push_prob, breakeven_prob=r.breakeven_prob, fair_prob=r.fair_prob, edge=r.edge,
                       ev_per_unit=r.ev_per_unit, kelly_full=k["kelly_full"], stake_pct=k["stake_pct"],
                       flagged=r.flagged, flags=r.flags))

        # ---- picks ledger
        now = datetime.now(timezone.utc)
        scanned_games = list({k[0] for k in markets})
        if scanned_games:
            await s.execute(text("""UPDATE picks SET active=false WHERE game_id = ANY(:g) AND result IS NULL"""),
                            {"g": scanned_games})
        for i, (p, row, kind, r) in enumerate(flagged):
            await s.execute(text("""
                INSERT INTO picks (sport, game_id, player_id, market_key, kind, side, line, price, first_price,
                                   model_prob, edge, stake_pct, proj_mean, active, first_flagged_at, last_seen_at)
                VALUES (:sp,:g,:p,:m,:k,:sd,:l,:pr,:pr,:mp,:e,:st,:pm,true,:now,:now)
                ON CONFLICT (game_id, player_id, market_key, kind, side, line) DO UPDATE SET
                  price=EXCLUDED.price, model_prob=EXCLUDED.model_prob, edge=EXCLUDED.edge, stake_pct=EXCLUDED.stake_pct,
                  proj_mean=EXCLUDED.proj_mean, active=true, last_seen_at=EXCLUDED.last_seen_at"""),
                dict(sp=sport, g=p.game_id, p=p.player_id, m=p.market_key, k=kind, sd=r.side, l=r.line, pr=r.price,
                     mp=r.model_prob, e=r.edge, st=sizing[str(i)]["stake_pct"], pm=p.mean, now=now))
        log.info("%s scan: %d markets, %d flagged", sport, len(markets), len(flagged))
        return len(flagged)
