import asyncio
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

FIX = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIX / name).read_text())


def test_parse_start_handles_espn_format_without_seconds():
    from app.ingest.espn_nba import parse_start

    assert parse_start("2026-11-18T00:30Z") == datetime(2026, 11, 18, 0, 30, tzinfo=timezone.utc)
    assert parse_start("2026-11-18T00:30:00Z") == datetime(2026, 11, 18, 0, 30, tzinfo=timezone.utc)
    assert parse_start(None) is None and parse_start("nope") is None


def test_pre_game_injuries_become_inactive_rows_only_for_out():
    from app.ingest import espn_nba as e

    r = e.parse_summary(load("espn_nba_pre.json"))
    assert r["meta"]["state"] == "pre" and r["gamelog"] == []
    names = {x["name_key"]: x for x in r["lineup"]}
    assert all(x["status"] == "INACTIVE" and x["game_status"] == 1 and not x["played"] for x in r["lineup"])
    assert "sam merrill" in names and "jaylon tyson" not in names     # Out kept, Day-To-Day left to the injury report


def test_final_box_gives_lineup_and_gamelog_rows():
    from app.ingest import espn_nba as e

    r = e.parse_summary(load("espn_nba_final.json"))
    m = r["meta"]
    assert m["state"] == "post" and m["home"] == "TOR" and m["away"] == "NY"
    by = {x["name_key"]: x for x in r["lineup"]}
    anunoby = by["og anunoby"]
    assert anunoby["starter"] and anunoby["played"] and anunoby["minutes"] == 36 and anunoby["game_status"] == 3
    dnp = [x for x in r["lineup"] if x["not_playing_reason"] and not x["played"]]
    assert dnp and all(x["minutes"] == 0.0 for x in dnp)
    logs = {x["name_key"]: x for x in r["gamelog"]}
    jb = logs["og anunoby"]
    assert jb["minutes"] == 36 and jb["matchup"].startswith("NY @ TOR") and jb["game_date"] == date(2026, 11, 17)
    assert jb["season"] == "2026-27" and jb["wl"] in ("W", "L") and isinstance(jb["pts"], int)
    assert all(x["minutes"] == 0.0 for x in r["gamelog"] if x["nba_person_id"] in {d["nba_person_id"] for d in dnp})


def test_box_rows_override_injury_rows_for_same_player():
    from app.ingest import espn_nba as e

    body = load("espn_nba_final.json")
    a = body["boxscore"]["players"][0]["statistics"][0]["athletes"][0]["athlete"]
    body["injuries"] = [{"team": {"abbreviation": "NY"}, "injuries": [{"status": "Out", "athlete": {"id": a["id"], "displayName": a["displayName"]}}]}]
    r = e.parse_summary(body)
    row = next(x for x in r["lineup"] if x["nba_person_id"] == int(a["id"]))
    assert row["status"] == "ACTIVE" and row["played"]


def test_unknown_state_and_bad_payload_are_ignored():
    from app.ingest import espn_nba as e

    assert e.parse_summary({}) is None
    b = load("espn_nba_pre.json"); b["header"]["competitions"][0]["status"]["type"] = {"state": "weird"}
    assert e.parse_summary(b) is None


def test_downstream_readers_agree_with_new_rows():
    """The adjustment layer reads status via LIVE_STATUS and the grader classifies played/dnp from final rows."""
    from app.engine.adjust import LIVE_STATUS
    from app.eval.grader import classify_nba
    from app.ingest import espn_nba as e

    pre = e.parse_summary(load("espn_nba_pre.json"))["lineup"][0]
    assert LIVE_STATUS[pre["status"]] == "Out"
    fin = e.parse_summary(load("espn_nba_final.json"))["lineup"]
    played = next(x for x in fin if x["played"]); dnp = next(x for x in fin if not x["played"])
    assert classify_nba(None, None, played) == "played"
    assert classify_nba(None, None, dnp) == "dnp"


def test_sync_lineups_stamps_final_rows_inside_grader_window(monkeypatch):
    from app.ingest import espn_nba as e

    final = e.parse_summary(load("espn_nba_final.json")); pre = e.parse_summary(load("espn_nba_pre.json"))
    seen = {}

    async def fake_summaries(days, states, skip_ids=frozenset()):
        return [final, pre]

    class _S:
        async def execute(self, *a, **k): seen.setdefault("sql", []).append(a[1] if len(a) > 1 else k)

    class _Ctx:
        async def __aenter__(self): return _S()
        async def __aexit__(self, *a): return False

    async def fake_upsert(s, model, rows, keys):
        seen["rows"] = rows
        return len(rows)

    monkeypatch.setattr(e, "_summaries", fake_summaries)
    monkeypatch.setattr(e, "session_scope", lambda: _Ctx())
    monkeypatch.setattr(e, "upsert", fake_upsert)
    fn = e.sync_lineups.__wrapped__ if hasattr(e.sync_lineups, "__wrapped__") else e.sync_lineups
    n = asyncio.run(fn(1, 1))
    start = final["meta"]["start"]
    fin_rows = [r for r in seen["rows"] if r["game_status"] == 3]
    assert n == len(seen["rows"]) and fin_rows
    assert all(0 <= (r["updated_at"] - start).total_seconds() <= 12 * 3600 for r in fin_rows)   # grader's evidence window
    assert seen["sql"] and seen["sql"][0]["g"] == final["meta"]["gid"]                          # pre-game Out rows finalised
