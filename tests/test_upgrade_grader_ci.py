"""Grader rules (active-with-zero is a miss, DNP is void, CLV sign) plus workflow / requirements sanity checks."""
import re
from pathlib import Path

import pytest

from app.eval import grader as g

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- grader
def test_active_player_with_zero_stats_is_a_miss_not_a_dnp():
    # Over 0.5 rebounds, player was active, recorded nothing (no box-score row): real loss, not void
    assert g.grade_pick("Over", 0.5, -120, "played", None) == ("loss", 0.0, -1.0)
    assert g.grade_pick("Under", 0.5, -120, "played", 0.0)[0] == "win"
    res, act, profit = g.grade_pick("Over", 24.5, +100, "played", 31)
    assert (res, act, profit) == ("win", 31.0, 1.0)
    assert g.grade_pick("Over", 5.0, -110, "played", 5)[0] == "push"


def test_confirmed_dnp_is_void_and_unknown_stays_pending():
    assert g.grade_pick("Over", 20.5, -110, "dnp", None) == ("void", None, 0.0)
    assert g.grade_pick("Over", 20.5, -110, "unknown", None) is None


def test_nba_participation_classification():
    assert g.classify_nba(22.0) == "played"
    assert g.classify_nba(None, gamelog_minutes=9.5) == "played"
    assert g.classify_nba(None, live={"played": True, "minutes": 0.0, "status": "ACTIVE", "game_status": 3}) == "played"
    assert g.classify_nba(None, live={"played": False, "minutes": None, "status": "INACTIVE", "game_status": 3}) == "dnp"
    assert g.classify_nba(None, live={"played": False, "minutes": None, "status": "ACTIVE", "game_status": 3}) == "dnp"   # on roster, no minutes
    assert g.classify_nba(None, live={"played": False, "status": "ACTIVE", "game_status": 2}) == "unknown"              # still live
    assert g.classify_nba(None) == "unknown"
    assert g.classify_nba(None, gamelog_minutes=0.0) == "dnp"


def test_nfl_participation_classification():
    assert g.classify_nfl(True, False, 0) == "played"
    assert g.classify_nfl(False, True, 0) == "played"                                # snaps but no stat line
    assert g.classify_nfl(False, False, g.MIN_TEAM_WEEK_ROWS + 10) == "dnp"          # team-week covered, he's absent
    assert g.classify_nfl(False, False, 3) == "unknown"                              # no snap data for that team-week yet


def test_clv_record_sign_and_missing_close():
    better = g.clv_record(+110, -105, close_fair=0.5)
    assert better["beat_close"] and better["clv_prob"] > 0 and better["close_fair"] == 0.5
    worse = g.clv_record(-130, +100)
    assert not worse["beat_close"] and worse["clv_prob"] < 0
    assert g.clv_record(-110, None) is None and g.clv_record(None, -110) is None
    assert g.clv_record(-110, -110)["clv_prob"] == pytest.approx(0.0)


# ---------------------------------------------------------------- workflows / requirements
WF = ROOT / ".github" / "workflows"


@pytest.mark.parametrize("name,cron", [("01_daily_ingest.yml", "0 13 * * *"), ("03_postgame_grade.yml", "0 6 * * *"),
                                        ("02_odds_snapshot.yml", "0 15,17,19,21,23,1,3 * * *")])
def test_workflow_schedule_and_concurrency(name, cron):
    text = (WF / name).read_text()
    assert f'cron: "{cron}"' in text
    assert re.search(r"concurrency:\s*\n\s+group: pipeline\s*\n\s+cancel-in-progress: false", text)   # shares pipeline.yml's DB lock
    assert "scripts/ci_db.sh restore" in text and "scripts/ci_db.sh save" in text and "workflow_dispatch" in text
    assert "pipeline_v2" in text


def test_odds_snapshot_gates_before_touching_the_database():
    text = (WF / "02_odds_snapshot.yml").read_text()
    assert "needs.gate.outputs.any == 'true'" in text and "pipeline_v2 gate" in text


def test_requirements_include_upgrade_dependencies_once():
    lines = [l.split("#")[0].strip().lower() for l in (ROOT / "requirements.txt").read_text().splitlines()]
    names = [re.split(r"[<>=\[]", l)[0] for l in lines if l]
    for dep in ("nflreadpy", "nba_api", "scipy", "scikit-learn", "requests"):
        assert names.count(dep) == 1, dep


def test_ci_script_and_data_files_exist():
    assert (ROOT / "scripts" / "ci_db.sh").stat().st_mode & 0o111
    assert (ROOT / "data" / "stadiums.json").exists()


def test_gate_parses_only_future_unstarted_games():
    from datetime import datetime, timedelta, timezone

    from app.pipeline_v2 import parse_scoreboard_starts

    now = datetime(2026, 11, 18, 19, 0, tzinfo=timezone.utc)
    f = lambda t: (now + timedelta(hours=t)).strftime("%Y-%m-%dT%H:%MZ")  # noqa: E731
    body = {"events": [
        {"date": f(5), "competitions": [{"status": {"type": {"state": "pre", "completed": False}}}]},
        {"date": f(-1), "competitions": [{"status": {"type": {"state": "in", "completed": False}}}]},
        {"date": f(-9), "competitions": [{"status": {"type": {"state": "post", "completed": True}}}]},
        {"date": "garbage", "competitions": []}]}
    assert parse_scoreboard_starts(body, now) == [now + timedelta(hours=5)]
