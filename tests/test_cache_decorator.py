import asyncio
import fakeredis
import pytest

import cache

@pytest.mark.asyncio
async def test_cache_result(monkeypatch):
    fake = fakeredis.FakeRedis()
    monkeypatch.setattr(cache, "redis_client", fake)
    calls = {"count": 0}

    @cache.cache_result(ttl=1)
    async def add(a: int, b: int) -> int:
        calls["count"] += 1
        return a + b

    assert await add(1, 2) == 3
    assert await add(1, 2) == 3
    assert calls["count"] == 1
