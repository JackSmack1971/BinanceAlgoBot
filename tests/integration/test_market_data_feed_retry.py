import asyncio
import os

import pytest

from src.data.market_data_feed import MarketDataFeed
from binance.client import Client

class DummyClient:
    def __init__(self):
        self.calls = 0

    def get_klines(self, symbol: str, interval: str):
        self.calls += 1
        if self.calls < 3:
            raise RuntimeError("fail")
        return [[1,2,3,4,5,6,7,8,9,10,11,12]]

@pytest.mark.asyncio
async def test_fetch_klines_retries(monkeypatch):
    client = DummyClient()
    feed = MarketDataFeed(client, symbol="BTCUSDT", interval="1m")
    monkeypatch.setenv("API_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("API_RETRY_DELAY", "0")
    monkeypatch.setenv("API_TIMEOUT", "1")
    data = await feed._fetch_klines()
    assert client.calls == 3
    assert data[0][0] == 1
