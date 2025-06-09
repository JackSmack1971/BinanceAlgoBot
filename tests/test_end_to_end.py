import asyncio
import pandas as pd
import pytest
from unittest.mock import AsyncMock, patch

from scripts.health_check import validate_system
from strategies.base_strategy import BaseStrategy
from PerformanceAnalyzer import PerformanceAnalyzer


class DummyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(object(), "BTCUSDT", "15m", 1000.0, 0.01)

    def calculate_indicators(self):
        pass

    def run(self):
        index = pd.date_range("2020-01-01", periods=2, freq="D")
        return pd.DataFrame({"close": [100, 110], "signal": [0, 1], "position": [0, 1]}, index=index)

    def open_position(self, side: str, price: float, size: float):
        self.position_manager.open_position(self.symbol, side, price, size)

    def close_position(self, price: float):
        self.position_manager.close_position(self.symbol, price)


@pytest.mark.asyncio
async def test_full_workflow(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_SECRET_KEY", "s")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")

    with patch("database.database_connection.DatabaseConnection.connect", AsyncMock()), \
         patch("database.database_connection.DatabaseConnection.disconnect", AsyncMock()), \
         patch("binance.client.Client.ping", return_value={}):
        assert await validate_system()

    strategy = DummyStrategy()
    analyzer = PerformanceAnalyzer()
    from service.service_locator import ServiceLocator
    class DummyMetricsService:
        async def insert_performance_metrics(self, **kwargs):
            return None

    ServiceLocator().register("PerformanceMetricsService", DummyMetricsService())
    results = await analyzer.calculate_performance(strategy.run())
    assert "metrics" in results
