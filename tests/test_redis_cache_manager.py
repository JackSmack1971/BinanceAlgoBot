import asyncio
import numpy as np
import fakeredis
import pytest

from src.cache.redis_cache_manager import IntelligentRedisCache, _default_cache
from src.indicators.numba_indicators import NumbaIndicatorEngine


@pytest.fixture(autouse=True)
def fake_redis(monkeypatch):
    fake = fakeredis.aioredis.FakeRedis()
    monkeypatch.setattr(_default_cache, "client", fake)
    yield


@pytest.mark.asyncio
async def test_cache_set_get_numpy():
    cache = IntelligentRedisCache()
    cache.client = fakeredis.aioredis.FakeRedis()
    arr = np.arange(1000, dtype=np.float64)
    await cache.set("arr", {"n": 1}, arr, 10)
    res = await cache.get("arr", {"n": 1})
    assert np.array_equal(arr, res)


@pytest.mark.asyncio
async def test_invalidate_and_ratio():
    cache = IntelligentRedisCache()
    cache.client = fakeredis.aioredis.FakeRedis()
    await cache.set("x", {"a": 1}, 5, 10)
    assert await cache.get("x", {"a": 1}) == 5
    assert cache.hit_ratio() == 1.0
    await cache.invalidate("x")
    assert await cache.get("x", {"a": 1}) is None


@pytest.mark.asyncio
async def test_concurrent_indicator_cache(monkeypatch):
    fake = fakeredis.aioredis.FakeRedis()
    monkeypatch.setattr(_default_cache, "client", fake)
    engine = NumbaIndicatorEngine()
    data = np.random.rand(20).astype(np.float64)

    await engine.sma(data, 5)
    _default_cache.hits = 0
    _default_cache.misses = 0

    async def run():
        await engine.sma(data, 5)

    tasks = [run() for _ in range(1000)]
    await asyncio.gather(*tasks)
    assert _default_cache.hit_ratio() > 0.9

