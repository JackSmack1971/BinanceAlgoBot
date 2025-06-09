import fakeredis
import pytest

from data_feed import DataFeed
import cache

class DummyClient:
    def get_klines(self, symbol: str, interval: str):
        # Return 50 rows of dummy klines
        row = [1]*12
        return [row for _ in range(50)]

@pytest.mark.asyncio
async def test_data_feed_max_rows(monkeypatch):
    fake = fakeredis.FakeRedis()
    monkeypatch.setattr(cache, "redis_client", fake)
    client = DummyClient()
    feed = DataFeed(client, "BTCUSDT", "1m")
    feed.max_rows = 10

    async def fake_insert(self, data):
        return None

    from database.market_data_repository import MarketDataRepository
    monkeypatch.setattr(MarketDataRepository, "insert_market_data", fake_insert)
    data = await feed.get_data()
    assert len(data) == 10

