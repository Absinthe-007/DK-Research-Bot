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
    async def boom(sport, now):
        raise RuntimeError("down")
    monkeypatch.setattr(planner, "upcoming_starts", boom)
    monkeypatch.setattr(planner, "load_state", lambda: {"last_refresh_NBA": "2026-11-18T00:00:00Z"})
    res = asyncio.run(planner.plan(datetime(2026, 11, 19, 9, 40, tzinfo=timezone.utc)))
    assert res["NBA"]["due"] is True
