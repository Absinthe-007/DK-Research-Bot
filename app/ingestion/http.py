"""
Resilient async HTTP layer shared by every connector.

Staying unblocked is done the *durable* way - by being a well-behaved client - rather than by
disguising traffic:
  * per-host token-bucket rate limits (configurable, conservative defaults)
  * exponential backoff + full jitter on 429/5xx/timeouts, honoring Retry-After
  * TTL response cache so repeated tool calls don't re-hit upstream
  * one honest, identifiable User-Agent (NWS *requires* this; others appreciate it)
  * a global concurrency cap

Disguised scraping (UA rotation, fingerprint spoofing) mostly buys a short reprieve before an
IP ban and can breach site terms; every source in this project has a public JSON endpoint or
an open dataset, so it isn't needed.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from app.config import get_settings

log = logging.getLogger(__name__)

# requests/second per host (conservative; ESPN publishes no limit, so stay polite).
HOST_RATES: dict[str, float] = {
    "site.api.espn.com": 4.0,
    "sports.core.api.espn.com": 4.0,
    "api.weather.gov": 2.0,
    "api.open-meteo.com": 2.0,
    "geocoding-api.open-meteo.com": 1.0,
    "www.espn.com": 0.5,
    "default": 1.0,
}


class TokenBucket:
    def __init__(self, rate: float, burst: int = 1):
        self.rate, self.capacity = rate, burst
        self.tokens, self.updated = float(burst), time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self.lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                await asyncio.sleep((1 - self.tokens) / self.rate)


class HttpClient:
    def __init__(self, max_concurrency: int = 8, max_retries: int = 5, timeout: float = 20.0):
        s = get_settings()
        self._client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": s.user_agent, "Accept": "application/json, text/html;q=0.9"},
        )
        self._sem = asyncio.Semaphore(max_concurrency)
        self._buckets: dict[str, TokenBucket] = {}
        self._cache: dict[str, tuple[float, Any, dict]] = {}
        self.max_retries = max_retries

    def _bucket(self, host: str) -> TokenBucket:
        if host not in self._buckets:
            self._buckets[host] = TokenBucket(HOST_RATES.get(host, HOST_RATES["default"]))
        return self._buckets[host]

    @staticmethod
    def _key(url: str, params: dict | None) -> str:
        return hashlib.sha1((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()

    async def get(
        self,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
        cache_ttl: float = 0,
        as_text: bool = False,
    ) -> tuple[Any, dict]:
        """Returns (body, response_headers). Body is parsed JSON unless as_text."""
        key = self._key(url, params)
        if cache_ttl and key in self._cache:
            exp, body, hdrs = self._cache[key]
            if exp > time.time():
                return body, hdrs

        host = urlparse(url).netloc
        attempt = 0
        while True:
            await self._bucket(host).acquire()
            async with self._sem:
                try:
                    r = await self._client.get(url, params=params, headers=headers)
                except (httpx.TimeoutException, httpx.TransportError) as e:
                    r, err = None, e
                else:
                    err = None
            retryable = r is None or r.status_code in (429, 500, 502, 503, 504)
            if not retryable:
                r.raise_for_status()
                body = r.text if as_text else r.json()
                hdrs = dict(r.headers)
                if cache_ttl:
                    self._cache[key] = (time.time() + cache_ttl, body, hdrs)
                return body, hdrs
            attempt += 1
            if attempt > self.max_retries:
                if r is not None:
                    r.raise_for_status()
                raise err  # type: ignore[misc]
            retry_after = r.headers.get("Retry-After") if r is not None else None
            # Full-jitter exponential backoff (AWS architecture blog): sleep U(0, base*2^n)
            delay = float(retry_after) if retry_after and retry_after.isdigit() else random.uniform(0, min(60, 1.5 * 2**attempt))
            log.warning("GET %s -> %s; retry %d in %.1fs", url, r.status_code if r else err, attempt, delay)
            await asyncio.sleep(delay)

    async def aclose(self) -> None:
        await self._client.aclose()


_client: HttpClient | None = None


def http() -> HttpClient:
    global _client
    if _client is None:
        _client = HttpClient()
    return _client
