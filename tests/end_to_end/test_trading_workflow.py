import pandas as pd
import pytest
from unittest.mock import Mock

from execution_engine import ExecutionEngine
from exchange_interface import ExchangeInterface
from strategies.base_strategy import BaseStrategy
from position_manager import PositionManager


class DummyExchange(ExchangeInterface):
    async def place_order(self, *_, **__):
        return True

    def fetch_market_data(self, symbol):
        return []

    def get_account_balance(self):
        return {}

    def get_order_status(self, order_id):
        return "filled"

    def get_current_price(self, symbol):  # pragma: no cover - unused in test
        return 100.0


class SimpleStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(Mock(), "BTCUSDT", "15m", 1000, 0.1)

    def calculate_indicators(self):
        return None

    def run(self):
        return pd.DataFrame({"signal": [0, 1], "position": [0, 1]})

    def open_position(self, side: str, price: float, size: float):
        self.position_manager.open_position(self.symbol, side, price, size)

    def close_position(self, price: float):
        self.position_manager.close_position(self.symbol, price)


@pytest.mark.asyncio
async def test_execute_trades_e2e(monkeypatch):
    strategy = SimpleStrategy()
    engine = ExecutionEngine(DummyExchange(), strategy, 1, strategy.position_manager)

    calls = {"buy": 0}

    async def fake_buy(self):
        calls["buy"] += 1

    async def fake_sell(self):
        calls["sell"] = 1

    monkeypatch.setattr(engine, "_execute_buy", fake_buy.__get__(engine))
    monkeypatch.setattr(engine, "_execute_sell", fake_sell.__get__(engine))

    signals = pd.DataFrame({"signal": [0, 1], "position": [0, 1]})
    await engine.execute_trades(signals)

    assert calls["buy"] == 1
