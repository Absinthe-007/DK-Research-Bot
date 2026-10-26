# DK Research Bot (GitHub edition)

NFL and NBA DraftKings prop research that runs **entirely on free GitHub Actions**. It needs no paid odds API, no LLM calls and no server. The results publish to a GitHub Pages dashboard.

**Dashboard:** `https://<your-user>.github.io/<repo>/`

## What it does

| Step | Source | Notes |
|---|---|---|
| DraftKings game lines and player props | ESPN's public odds feed (provider 100 = DraftKings) | O/U props plus "N+" milestone ladders and anytime TDs |
| Box scores (stat history) | ESPN summary API | Used for both sports. stats.nba.com blocks GitHub's IPs; ESPN doesn't |
| Injuries | ESPN injuries JSON, with an HTML fallback | Keyed by ESPN athlete id, so no name matching is needed |
| Weather | NWS hourly forecast, with Open-Meteo as fallback | Outdoor NFL venues only, averaged over the game window |
| Snap %, air yards, WOPR | nflverse (optional) | Added twice a week |

The model projects each player from a recency-weighted average, adjusted for opponent, pace, rest, usage shifts and weather. It then **anchors that projection to the DraftKings line**: `final = 0.35·model + 0.65·market`, with a per-slate bias correction. On a real slate, the raw model disagreed with DraftKings by about 12 probability points on average, and almost all of that was noise. Anchoring brings the gap down to about 4 points, so only player-specific disagreement survives.

A play is flagged when model P(win) minus DraftKings' breakeven is at least **3.5%**, or at least 6% for alt lines.
- Each player and market gets at most one play.
- A play is never flagged when the model is more than 30% off the market. A gap that big usually means news the box scores can't see.

Every flagged play is frozen at its last pre-game price. It's graded after the box score comes in, and the dashboard shows the hit rate, units, ROI and closing-line value.

## When it runs

A planner job ticks every 15 minutes (`7,22,37,52 * * * *`). It makes 10 ESPN requests and needs no database. It launches a full refresh only at checkpoints measured back from each game's **actual start time**:

| | Checkpoints before start | Why |
|---|---|---|
| NBA | 10h, 4h, 90m, 30m | Props are posted by morning; the afternoon injury report; the pre-game report; lineups and late scratches |
| NFL | 72h, 26h, 5h, 75m, 25m | Practice reports; final designations; the reliable weather window; inactives (posted at T-90m); final check |

Days without games cost nothing. Saturday, international and holiday games are picked up automatically. A **nightly job** runs at 09:41 UTC (05:41 ET). It loads box scores, grades picks, prunes old odds and makes a one-line heartbeat commit. That commit stops GitHub from disabling the schedule after 60 days without repo activity.

## Setup (one time)

1. **Settings → Pages → Source: GitHub Actions.**
2. **Actions → pipeline → Run workflow → mode = backfill**, sport = both, start = about a season back (e.g. `2025-10-01`). This takes about 15 minutes.
3. That's it. The schedule takes over. To refresh right now, run the workflow with `mode = refresh`.

Where the state lives:
- **Database:** Postgres runs inside each job. Between runs, it's saved as a `pg_dump` in the `data` release (a few MB).
- **Planner state:** `state.json` is stored in the same release.
- **Dashboard:** deployed to Pages. Data files are never committed, so the repo doesn't grow.

## Run locally

```bash
docker run -d -p 5432:5432 -e POSTGRES_USER=dkbot -e POSTGRES_PASSWORD=dkbot -e POSTGRES_DB=dkbot postgres:16
pip install -r requirements.txt
python -m scripts.init_db
python -m app.pipeline backfill --sport NBA --start 2025-10-21 --end 2026-04-15
python -m app.pipeline refresh --sport NBA         # writes docs/data/board.json
cd docs && python -m http.server                    # open http://localhost:8000
pytest -q
```

## Tuning (environment variables / `app/config.py`)

| Variable | Default | Meaning |
|---|---|---|
| `EDGE_THRESHOLD` | 0.035 | Minimum edge to flag an O/U play |
| `ALT_EDGE_THRESHOLD` | 0.06 | Minimum edge to flag a milestone (alt) line |
| `MODEL_WEIGHT` | 0.35 | Weight on the model vs the market. Raise it only if the graded record and CLV show the model adds information |
| `MAX_MODEL_MARKET_GAP` | 0.30 | Plays where the model is further than this from the market are suppressed |
| `KELLY_FRACTION`, `MAX_STAKE_PCT`, `MAX_TOTAL_EXPOSURE_PCT`, `BANKROLL` | 0.25, 0.02, 0.10, 1000 | Stake sizing |

## Caveats

* ESPN's feeds are public but undocumented, so field names can change. Parsing is defensive, and each source fails on its own without stopping the rest of the run.
* ESPN sometimes shows NFL prop **lines without prices**. In that case the bot assumes -115 on both sides and labels the play "DK PRICE NOT PUBLISHED". Check the DraftKings app before acting on it.
* These are probabilistic estimates, not guarantees. Bet only what you can afford to lose. If gambling stops being fun: 1-800-GAMBLER.
