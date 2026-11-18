"""
Breakdowns of the graded pick ledger. Overall ROI hides whether the model is calibrated, which markets
carry the edge (if any), and whether the sample is padded by correlated bets on the same player.

Pure functions over plain dicts, so they are unit-testable without a database. Input rows need:
sport, game_id, player_id, market_key, kind, price, first_price, close_price, model_prob, edge,
result ('win'|'loss'), profit_units.
"""
from __future__ import annotations

from collections import defaultdict

from app.engine.odds_math import implied_prob

# Model-chance buckets: lower bound inclusive, upper exclusive.
CALIBRATION_BUCKETS = [(0.0, 0.55, "<55%"), (0.55, 0.60, "55–60%"), (0.60, 0.65, "60–65%"), (0.65, 1.01, "65%+")]
# Gap (edge) buckets, to test whether bigger claimed gaps actually pay more.
EDGE_BUCKETS = [(0.0, 0.05, "<5%"), (0.05, 0.08, "5–8%"), (0.08, 0.12, "8–12%"), (0.12, 9.0, "12%+")]


def market_class(market_key: str, kind: str) -> str:
    """Alt (milestone) ladders, combo stats (Pts+Reb, PRA, ...) and single stats behave differently."""
    if kind == "alt":
        return "Alt lines"
    parts = market_key.removeprefix("player_")
    if parts in ("steals_blocks", "points_rebounds", "points_assists", "rebounds_assists", "points_rebounds_assists",
                 "rush_reception_yds", "pass_rush_yds"):
        return "Combo stats"
    return "Single stats"


def _summary(rows: list[dict]) -> dict:
    n = len(rows)
    wins = sum(r["result"] == "win" for r in rows)
    units = sum(r["profit_units"] or 0.0 for r in rows)
    return {
        "n": n, "wins": wins, "losses": n - wins, "units": units,
        "hit": wins / n if n else None,
        "roi": units / n if n else None,
        "avg_model": sum(r["model_prob"] for r in rows) / n if n else None,
        "avg_breakeven": sum(implied_prob(r["price"]) for r in rows) / n if n else None,
    }


def _bucketed(rows: list[dict], key, buckets) -> list[dict]:
    out = []
    for lo, hi, label in buckets:
        sel = [r for r in rows if lo <= key(r) < hi]
        if sel:
            out.append({"label": label, **_summary(sel)})
    return out


def one_per_player_game(rows: list[dict]) -> list[dict]:
    """Keep the highest-edge play per (game, player). Several overlapping markets on one player (Pts, PRA,
    Pts+Ast, ...) are one opinion, so counting each inflates the sample and makes streaks look meaningful."""
    best: dict[tuple, dict] = {}
    for r in rows:
        k = (r["game_id"], r["player_id"])
        if k not in best or r["edge"] > best[k]["edge"]:
            best[k] = r
    return list(best.values())


def clv(rows: list[dict]) -> dict | None:
    """Closing-line value: how the price moved between flag time and kickoff, as breakeven-probability points.
    Positive = the market moved toward our side (we got a better price than the close). Prices are single-sided
    (not de-vigged), which is fine for a *difference* between two prices of the same side."""
    vals, vals_last = [], []
    for r in rows:
        c = r.get("close_price")
        if c is None:
            continue
        if r.get("first_price") is not None:
            vals.append(implied_prob(c) - implied_prob(r["first_price"]))
        vals_last.append(implied_prob(c) - implied_prob(r["price"]))
    if not vals and not vals_last:
        return None
    out = {}
    for name, v in (("vs_first", vals), ("vs_last", vals_last)):
        out[name] = {"n": len(v), "avg": sum(v) / len(v), "beat": sum(x > 0 for x in v) / len(v)} if v else None
    return out


def breakdown(rows: list[dict]) -> dict:
    """{sport: {overall, deduped, calibration, by_edge, by_class, by_market, clv}} for graded win/loss plays."""
    by_sport: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["result"] in ("win", "loss"):
            by_sport[r["sport"]].append(r)
    out = {}
    for sp, rs in by_sport.items():
        classes = defaultdict(list)
        markets = defaultdict(list)
        for r in rs:
            classes[market_class(r["market_key"], r["kind"])].append(r)
            markets[(r["market_key"], r["kind"])].append(r)
        out[sp] = {
            "overall": _summary(rs),
            "deduped": _summary(one_per_player_game(rs)),
            "calibration": _bucketed(rs, lambda r: r["model_prob"], CALIBRATION_BUCKETS),
            "by_edge": _bucketed(rs, lambda r: r["edge"], EDGE_BUCKETS),
            "by_class": [{"label": k, **_summary(v)} for k, v in sorted(classes.items())],
            "by_market": sorted(({"market_key": k[0], "kind": k[1], **_summary(v)} for k, v in markets.items()),
                                key=lambda x: -x["n"]),
            "clv": clv(rs),
        }
    return out
