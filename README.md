# DK Research Bot (GitHub edition)

NFL and NBA DraftKings prop research that runs **entirely on free GitHub Actions**. It needs no paid odds API, no LLM calls and no server. An optional [The Odds API](https://the-odds-api.com) key adds real DraftKings prices where ESPN omits them (see below). The results publish to a GitHub Pages dashboard.

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

A play is flagged when model P(win) minus DraftKings' breakeven is at least **8%**, or at least 10% for alt lines (raised from 3.5% / 6%: small gaps sit inside the model's error).
- Each player and market gets at most one play.
- A play is never flagged when the model is more than 30% off the market. A gap that big usually means news the box scores can't see.

Every flagged play is frozen at its last pre-game price. It's graded after the box score comes in, and the dashboard shows the hit rate, units, ROI and closing-line value, plus breakdowns by model confidence, gap size and market type, and an ROI view with one play per player per game.

## Dashboard

* Scoreboard tiles for every upcoming game with DraftKings spread, total and a **projected winner** (moneyline with the margin removed; graded against finished games).
* Every player's projected line. Under each number you can show the **DraftKings line** (▲/▼ = projection above/below it) or the likely range. Click a player for line prices and the model's over/under chance.
* Scorecard: share of stat lines inside the projected range, average miss, winner-pick record and the graded record of flagged plays.

Model notes (backtested on 105 NBA games, Nov 2026): longer half-lives for noisy stats, DK game total as a scoring-environment factor and spread-based blowout minutes were all tested and moved points error by < 0.02, so they ship **off** (`STATBOARD_TUNE`, see `app/engine/statboard.py`). Per-market blend weights (`MODEL_WEIGHT_BY_MARKET`) are on: the market alone is best for points/PRA, the model adds more for counting stats. Flagged plays have not beaten DraftKings' margin so far (see the track record card), so treat them as research leads.

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
| `EDGE_THRESHOLD` | 0.08 | Minimum edge to flag an O/U play |
| `ALT_EDGE_THRESHOLD` | 0.10 | Minimum edge to flag a milestone (alt) line |
| `MODEL_WEIGHT` | 0.35 | Weight on the model vs the market. Raise it only if the graded record and CLV show the model adds information |
| `MAX_MODEL_MARKET_GAP` | 0.30 | Plays where the model is further than this from the market are suppressed |
| `KELLY_FRACTION`, `MAX_STAKE_PCT`, `MAX_TOTAL_EXPOSURE_PCT`, `BANKROLL` | 0.25, 0.02, 0.10, 1000 | Stake sizing |

## Caveats

* ESPN's feeds are public but undocumented, so field names can change. Parsing is defensive, and each source fails on its own without stopping the rest of the run.
* ESPN sometimes shows NFL prop **lines without prices**. In that case the bot assumes -115 on both sides and labels the play "DK PRICE NOT PUBLISHED". Check the DraftKings app before acting on it.
* These are probabilistic estimates, not guarantees. Bet only what you can afford to lose. If gambling stops being fun: 1-800-GAMBLER.


## Optional: The Odds API

ESPN remains the primary source. If you add a free key from the-odds-api.com, the pipeline also:

* replaces ESPN's assumed -115 prop prices with real DraftKings prices (those plays then become flaggable instead of "DK PRICE NOT PUBLISHED"), and
* fills DraftKings game lines for games where ESPN has none.

Setup: repo **Settings -> Secrets and variables -> Actions -> New repository secret**, name `ODDS_API_KEY`. Remove the secret to turn it off.
Credit use is capped (`odds_api_max_credits_per_run`, a reserve floor `odds_api_min_remaining`, and a per-game re-fetch gap) so the free tier (~500 credits/month) is not exhausted; tune them in `app/config.py`.


## Prediction upgrades (additive)

These modules add new data feeds, helpers and a second grader **without changing the original projections**: `projector.py`,
`statboard.py`, `scanner.py`, `pipeline.py` and the existing tables are untouched. New data lands in new tables
(`app/db/models_ext.py`) and the helpers are opt-in, so nothing feeds back into a projection until you wire it in.

| Piece | Where | Notes |
|---|---|---|
| NFL feed | `app/ingest/nfl_feed.py` | nflreadpy: snap counts, PBP targets/carries, weekly injuries, depth charts |
| NBA feed | `app/ingest/nba_feed.py` | nba_api: game logs, `TeamPlayerOnOffDetails`, live scoreboard lineups; browser headers, 1.5 s between calls. stats.nba.com often blocks GitHub IPs, so each fetch is best-effort |
| Weather | `app/ingest/weather_feed.py` + `data/stadiums.json` | Open-Meteo kickoff-hour wind/precip; dome and retractable stadiums are skipped |
| Markets | `app/ingest/market_feed.py` | The Odds API (`ODDS_API_KEY`), Kalshi and Polymarket public quotes; `market_lines` keeps OPEN vs CURRENT and freezes at the start (= closing line) |
| Engine | `app/engine/{devig,redistribution,count_models,calibration}.py` | de-vig helpers, injury minutes/usage redistribution, `scipy.stats` nbinom/poisson with a DNP-void mixture, isotonic/Platt calibration |
| Grader | `app/eval/grader.py` | active player with 0 stats = miss, confirmed DNP = void, no evidence = stays pending; logs CLV to `clv_log` |

Workflows (all UTC cron; GitHub ignores daylight saving, so they shift an hour in summer):

* `01_daily_ingest.yml`: 13:00 UTC (8:00 AM EST)
* `02_odds_snapshot.yml`: every 30 min, 15:00-04:59 UTC, but only when ESPN shows a game starting within 6 h
* `03_postgame_grade.yml`: 09:00 UTC (4:00 AM EST)

The database is Postgres (not SQLite), so persistence is the same `pg_dump` in the `data` release that `pipeline.yml` uses (`scripts/ci_db.sh`), not a commit.
All DB-writing workflows share the `pipeline` concurrency group so two jobs never overwrite each other's dump. The Odds API free tier (~500 credits/month)
will not last at a 30-minute cadence; the run stops at `ODDS_API_MIN_REMAINING`, and Kalshi/Polymarket are free.

Local run: `python -m app.pipeline_v2 ingest|odds|grade|gate`. DB integration test:
`DKBOT_TEST_DB=1 DATABASE_URL=postgresql+asyncpg://.../dkbot_test pytest tests/test_upgrade_db_integration.py` (the DB name must contain "test").
