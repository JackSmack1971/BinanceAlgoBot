import pytest
from strategy_factory import StrategyFactory
from strategies.base_strategy import BaseStrategy
from unittest.mock import MagicMock
from binance.client import Client


class MockStrategy(BaseStrategy):
    def __init__(self, client: Client, symbol: str, interval: str, initial_balance: float = 10000, risk_per_trade: float = 0.01):
        super().__init__(client, symbol, interval, initial_balance, risk_per_trade)

    def calculate_indicators(self):
        return None

    def run(self):
        return None

    def open_position(self, side: str, price: float, size: float):
        pass

    def close_position(self, price: float):
        pass


def test_strategy_factory_creation():
    factory = StrategyFactory()
    assert factory is not None


def test_strategy_factory_create_strategy():
    StrategyFactory.register_strategy("MOCK", MockStrategy)
    factory = StrategyFactory()
    mock_client = MagicMock()
    strategy = factory.create_strategy("MOCK", mock_client, "BTC", "1m")
    assert strategy is not None
    assert strategy.symbol == "BTC"
    assert strategy.interval == "1m"


def test_strategy_factory_invalid_symbol():
    factory = StrategyFactory()
    mock_client = MagicMock()
    with pytest.raises(ValueError):
        factory.create_strategy("MOCK", mock_client, None, "1m")


def test_strategy_factory_invalid_timeframe():
    factory = StrategyFactory()
    mock_client = MagicMock()
    with pytest.raises(ValueError):
        factory.create_strategy("MOCK", mock_client, "BTC", None)
