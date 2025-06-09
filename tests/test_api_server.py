import hmac
from hashlib import sha256

import pytest
from fastapi.testclient import TestClient

from src.api.server import app


def sign(path: str, secret: str) -> str:
    return hmac.new(secret.encode(), path.encode(), sha256).hexdigest()


@pytest.mark.asyncio
async def test_market_data_endpoint(monkeypatch):
    secret = "abc"
    monkeypatch.setenv("API_SECRET", secret)
    client = TestClient(app)

    async def dummy_fetch(symbol):
        return [{"symbol": symbol}]

    monkeypatch.setattr(
        "src.api.server.SecureMarketDataRepository.fetch_by_symbol", dummy_fetch
    )

    sig = sign("/market_data", secret)
    resp = client.get("/market_data", params={"symbol": "BTCUSDT"}, headers={"X-Signature": sig})
    assert resp.status_code == 200
    assert resp.json()["data"] == [{"symbol": "BTCUSDT"}]
