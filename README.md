# DK Research Bot

NFL and NBA player-prop research that runs **entirely on free GitHub Actions**. No server, no paid odds API required, no LLM calls. Results publish to a GitHub Pages dashboard that shows every upcoming game, each player's expected stat line, projected scores and winners, and a graded track record.

**Dashboard:** `https://<your-user>.github.io/<repo>/`

> Research tool, not betting advice. The flagged plays have not beaten DraftKings' margin so far (see the track record card on the dashboard). Treat them as leads to investigate.

## What you get

* **Today's slate** for NBA and NFL: scoreboard tiles with the DraftKings total, a **projected score** and a **projected winner** per game. Winners are graded against finished games.
* **Every player's projected stat line** (points, rebounds, assists, 3s, passing/rushing/receiving yards, receptions, sacks, TDs and more). Under each number, switch between the **DraftKings line** (▲/▼ = projection above/below it) and the likely range. Click a player for line prices and the model's over/under chance.
* **Search and filters** by player, team, position and "only players with a DraftKings line".
* **Where the model disagrees with DraftKings**: today's biggest gaps.
* **Track record**: hit rate, units, ROI per $100, closing-line value (CLV) and push-aware results, with breakdowns by model confidence, gap size and market type, plus a one-play-per-player-per-game ROI view.
* **Accuracy by stat**: average miss and share of lines landing inside the projected range, with an overall accuracy figure.
* A plain-English "How to read the page" guide for people who don't bet.

## Data sources

| Data | Source | Notes |
|---|---|---|
| DraftKings game lines and player props | ESPN public odds feed (provider 100 = DraftKings) | O/U props, "N+" milestone ladders, anytime TDs |
| Real DraftKings prop prices and missing game lines | [The Odds API](https://the-odds-api.com) (optional) | Needs the `ODDS_API_KEY` secret; ESPN stays primary |
| Box scores (stat history) | ESPN summary API | Both sports. stats.nba.com blocks GitHub's IPs; ESPN doesn't |
| NBA lineups, inactives, game logs | ESPN summary API (`app/ingest/espn_nba.py`) | Free and reachable from GitHub runners. Pre-game "Out" statuses, starters, minutes, DNP reasons |
| Injuries | ESPN injuries JSON, HTML fallback | Keyed by ESPN athlete id, no name matching needed |
| Weather | NWS hourly forecast, Open-Meteo fallback | Outdoor NFL venues only, averaged over the game window; `data/stadiums.json` marks domes |
| NFL snap %, air yards, WOPR, PBP usage, depth charts | nflverse via `nflreadpy` (optional) | Twice-weekly enrichment |
| NBA game logs and on/off splits | `nba_api` (best effort) | stats.nba.com often blocks GitHub IPs, so it fails fast and the ESPN feed covers the gap |
| Prediction-market quotes | Kalshi and Polymarket public quotes | Free, no key; stored in `market_lines` |

## How the model works

1. **Projection.** Each player starts from a recency-weighted average of recent box scores, adjusted for opponent, pace, rest, usage shifts, teammates out and weather. Over/under chances come from discretized outcome distributions, so half-point and whole-number lines (including pushes) are exact. Opt-in helpers for negative-binomial / Poisson count models with a did-not-play mixture, injury redistribution and isotonic/Platt calibration live in `app/engine/` but are not wired into the live projections yet.
2. **Market anchoring.** The projection is blended with the DraftKings line: `final = w·model + (1−w)·market`, with a per-slate bias correction. The default weight is 0.35, and **per-market weights** (`MODEL_WEIGHT_BY_MARKET` in `app/config.py`) lean on the market for heavily bet scoring props (points ≈ 0.20) and on the model for peripheral counting stats (assists, rebounds, 3s, steals ≈ 0.50–0.55). On a real slate the raw model disagreed with DraftKings by about 12 probability points on average, almost all noise; anchoring cuts that to about 4, so only player-specific disagreement survives.
3. **Vig removal.** Implied probabilities are de-vigged with the Shin method (power-method fallback), so edges are measured against fair odds, not raw prices.
4. **Flagging.** A play is flagged when model P(win) minus DraftKings' breakeven is at least **8%** (10% for alt lines). Each player and market gets at most one play, milestone lines are only flagged inside a −250/+250 price band, and a play is suppressed when the model is more than 30% away from the market (usually news the box scores can't see).
5. **Staking.** Fractional Kelly (25%), capped at 2% of bankroll per play and 10% total exposure.
6. **Grading.** Flagged plays are frozen at their last pre-game price. After the box score arrives they are graded: an active player with 0 stats is a miss, a confirmed DNP is void, no evidence leaves the play pending. Pushes are handled correctly, and first vs closing prices feed CLV.

Backtest notes (105 NBA games, Nov 2026): longer half-lives for noisy stats, the DK game total as a scoring-environment factor, and spread-based blowout minutes each moved points error by less than 0.02, so they ship **off** (`STATBOARD_TUNE` in `app/engine/statboard.py`). Per-market blend weights are **on**.

### Optional injury / inactives adjustment (off by default)

`app/engine/adjust.py` redistributes NBA **minutes** (teammates gain minutes when a rotation player is out) and NFL **target / carry share**, then scales the matching stats. It writes `adjusted_stat_projections` next to the originals, which are never modified. A player's own adjustment is conditional on him playing, so a Questionable starter keeps his number and only teammates absorb the expected absence. Pre-game NBA lineup inactives override injury-report statuses.

* Enable: set the repo **variable** `ADJUSTED_PROJECTIONS` to `1` (Settings → Secrets and variables → Actions → Variables). `02_odds_snapshot.yml` then runs it after each snapshot.
* Judge it: `python -m app.pipeline_v2 adjust-report` (also run in `03_postgame_grade.yml`) compares original vs adjusted mean absolute error on graded rows. Don't switch the dashboard or picks to the adjusted numbers until it shows an improvement. The NFL settings are conservative and untuned.

## When it runs

All crons are UTC. GitHub ignores daylight saving, so times shift by an hour in summer.

| Workflow | Schedule | What it does |
|---|---|---|
| `pipeline.yml`, planner | every 15 min (`7,22,37,52 * * * *`) | 10 ESPN requests, no database. Launches a full refresh only at checkpoints measured back from each game's **actual start time** |
| `pipeline.yml`, nightly | 09:41 UTC (05:41 ET) | Loads box scores, grades picks, prunes old odds, makes a one-line heartbeat commit so GitHub doesn't disable the schedule after 60 quiet days |
| `01_daily_ingest.yml` | 13:00 UTC (8:00 AM EST) | nflverse feeds, NBA feeds, kickoff weather, first odds snapshot (the "open" for most lines) |
| `02_odds_snapshot.yml` | every 2 h at 15, 17, 19, 21, 23, 01, 03 UTC | Odds and prediction-market snapshot. A cheap gate job runs first and skips everything unless ESPN shows a game starting within 6 h |
| `03_postgame_grade.yml` | 09:00 UTC (4:00 AM EST) | Box scores and participation evidence, grades open picks, logs CLV, runs the adjustment report |
| `ci.yaml` | push and PR to `main` | `pytest` with coverage on `app/engine` |

Refresh checkpoints before each game's start:

| | Checkpoints | Why |
|---|---|---|
| NBA | 10h, 4h, 90m, 30m | Props are up by morning; afternoon injury report; pre-game report; lineups and late scratches |
| NFL | 72h, 26h, 5h, 75m, 25m | Practice reports; final designations; the reliable weather window; inactives (posted at T-90m); final check |

Days without games cost nothing, and Saturday, international and holiday games are picked up automatically. Lines freeze at tip-off / kickoff, so the last snapshot before the start is the closing line used for CLV.

## Setup (one time)

1. **Settings → Pages → Source: GitHub Actions.**
2. **Actions → pipeline → Run workflow → mode = `backfill`**, sport = both, start = about a season back (for example `2025-10-01`). Takes about 15 minutes.
3. The schedule takes over. To refresh immediately, run the workflow with `mode = refresh`.
4. *(Optional)* Add the `ODDS_API_KEY` secret (below).

Where state lives:

* **Database:** Postgres runs inside each job and is saved between runs as a `pg_dump` in the `data` release (a few MB), via `scripts/ci_db.sh`.
* **Planner state:** `state.json` in the same release.
* **Dashboard:** deployed to Pages. Data files are never committed, so the repo doesn't grow.
* **Concurrency:** every DB-writing workflow shares the `pipeline` concurrency group, so two jobs never overwrite each other's dump.

## Optional: The Odds API

ESPN remains the primary source. With a free key from the-odds-api.com the pipeline also replaces ESPN's assumed −115 prop prices with real DraftKings prices (those plays become flaggable instead of "DK PRICE NOT PUBLISHED") and fills in DraftKings game lines where ESPN has none.

Add it under **Settings → Secrets and variables → Actions → New repository secret**, name `ODDS_API_KEY`. Remove the secret to turn it off.

Credit use is capped so the free tier (~500 credits/month) lasts: `odds_api_max_credits_per_run` (40), a reserve floor `odds_api_min_remaining` (60, env `ODDS_API_MIN_REMAINING`), and `odds_api_min_refetch_minutes` (120) per game. The snapshot runs every 2 hours rather than every 30 minutes for the same reason (about 21 credits per sport on a game day). Kalshi and Polymarket are free.

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

Upgrade jobs: `python -m app.pipeline_v2 ingest|odds|grade|adjust|adjust-report|gate`.

DB integration test: `DKBOT_TEST_DB=1 DATABASE_URL=postgresql+asyncpg://.../dkbot_test pytest tests/test_upgrade_db_integration.py` (the DB name must contain "test").

## Tuning

Environment variables (see `app/config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `EDGE_THRESHOLD` | 0.08 | Minimum edge to flag an O/U play |
| `ALT_EDGE_THRESHOLD` | 0.10 | Minimum edge to flag a milestone (alt) line |
| `MODEL_WEIGHT` | 0.35 | Default weight on the model vs the market, used when a market has no entry in `MODEL_WEIGHT_BY_MARKET`. Raise it only if the graded record and CLV show the model adds information |
| `MAX_MODEL_MARKET_GAP` | 0.30 | Plays where the model is further than this from the market are suppressed |
| `ALT_MIN_PRICE`, `ALT_MAX_PRICE` | −250, 250 | Price band for flagging milestone lines |
| `KELLY_FRACTION`, `MAX_STAKE_PCT`, `MAX_TOTAL_EXPOSURE_PCT`, `BANKROLL` | 0.25, 0.02, 0.10, 1000 | Stake sizing |
| `ODDS_API_KEY` | unset | Turns on The Odds API layer |
| `ADJUSTED_PROJECTIONS` | unset | Repo variable; `1` enables the injury/inactives adjustment |

## Project layout

```
app/engine/      projector, statboard, scanner, EV, Kelly, Shin/power devig, grading, calibration, adjustment
app/ingestion/   ESPN, injuries, weather, The Odds API, nflverse stats
app/ingest/      upgrade feeds: ESPN NBA lineups, nflverse, nba_api, Open-Meteo, Kalshi/Polymarket
app/eval/        post-game grader with CLV logging
app/pipeline.py  original refresh / nightly flow      app/pipeline_v2.py  ingest / odds / grade / adjust jobs
app/planner.py   kickoff-relative checkpoint planner  app/export.py       builds docs/data/board.json
docs/            the GitHub Pages dashboard (single static page)
tests/           pytest suite with ESPN fixtures
```

## Caveats

* ESPN's feeds are public but undocumented, so field names can change. Parsing is defensive and each source fails on its own without stopping the rest of the run.
* Without an Odds API key, ESPN sometimes shows NFL prop **lines without prices**. The bot then assumes −115 on both sides and labels the play "DK PRICE NOT PUBLISHED". Check the DraftKings app before acting on it.
* stats.nba.com and cdn.nba.com refuse GitHub-hosted runners; the `nba_api` feed is best effort and the ESPN NBA feed is what keeps lineups and inactives flowing.
* These are probabilistic estimates, not guarantees. Bet only what you can afford to lose. If gambling stops being fun: 1-800-GAMBLER.
