"""Regression: ESPN 400s on multi-day `dates=A-B`; the planner must use single-day requests."""
import asyncio
from datetime import datetime, timezone

from app import planner


class FakeHttp:
    def __init__(self):
        self.dates = []

    async def get(self, url, params=None, **kw):
        d = params["dates"]
        self.dates.append(d)
        if "-" in d:
            raise RuntimeError("400 Bad Request")
        return {"events": [{"id": d, "date": "2026-11-20T01:15:00Z",
                            "competitions": [{"status": {"type": {"state": "pre", "completed": False}}}]}]}, None


def test_single_day_requests(monkeypatch):
    fake = FakeHttp()
    monkeypatch.setattr(planner, "http", lambda: fake)
    now = datetime(2026, 11, 19, 9, 40, tzinfo=timezone.utc)
    starts = asyncio.run(planner.upcoming_starts("NFL", now))
    assert starts and all("-" not in d for d in fake.dates) and len(fake.dates) == 5


def test_error_falls_back_when_stale(monkeypatch):
    async def boom(sport, now, recent=None):
        raise RuntimeError("down")
    monkeypatch.setattr(planner, "upcoming_starts", boom)
    monkeypatch.setattr(planner, "load_state", lambda: {"last_refresh_NBA": "2026-11-18T00:00:00Z"})
    res = asyncio.run(planner.plan(datetime(2026, 11, 19, 9, 40, tzinfo=timezone.utc)))
    assert res["NBA"]["due"] is True


def test_post_game_refresh_fires_once(monkeypatch):
    """A finished game triggers one rebuild ~4h15m after tip-off so it drops off the board."""
    tip = datetime(2026, 11, 24, 3, 0, tzinfo=timezone.utc)

    async def fake(sport, now, recent=None):
        if recent is not None and sport == "NBA":
            recent.append(tip)
        return []
    monkeypatch.setattr(planner, "upcoming_starts", fake)
    monkeypatch.setattr(planner, "load_state", lambda: {"last_refresh_NBA": "2026-11-24T06:30:00Z",
                                                        "last_refresh_NFL": "2026-11-24T06:30:00Z"})
    before = asyncio.run(planner.plan(datetime(2026, 11, 24, 7, 10, tzinfo=timezone.utc)))
    after = asyncio.run(planner.plan(datetime(2026, 11, 24, 7, 20, tzinfo=timezone.utc)))
    assert before["NBA"]["due"] is False
    assert after["NBA"]["due"] is True and after["NBA"]["checkpoints"][0].startswith("11-24 03:00Z T+")
