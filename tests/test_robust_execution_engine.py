import asyncio
from decimal import Decimal
from typing import Any

import pytest

from src.execution import RobustExecutionEngine, OrderType


class DummyExchange:
    def __init__(self) -> None:
        self.orders = []
        self.calls = 0

    async def place_order(self, **kwargs: Any) -> dict:
        self.calls += 1
        if self.calls == 1 and kwargs.get("retry", False):
            raise Exception("network")
        self.orders.append(kwargs)
        return {"orderId": "1"}

    async def get_order(self, *_: Any, **__: Any) -> dict:
        return {
            "status": "FILLED",
            "fills": [
                {
                    "price": "1",
                    "qty": "1",
                    "commission": "0",
                    "tradeId": "t1",
                }
            ],
        }

    async def cancel_order(self, *_: Any, **__: Any) -> None:
        return None


class DummyDB:
    class Txn:
        async def __aenter__(self) -> "DummyDB.Txn":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    def transaction(self) -> "DummyDB.Txn":
        return DummyDB.Txn()

    async def save_order(self, _order: Any) -> None:
        return None


class DummyRisk:
    async def validate_order(self, _order: Any) -> Any:
        class Result:
            approved = True
            reason = ""

        return Result()


class DummyPM:
    def __init__(self) -> None:
        self.updated = False
        self.db = DummyDB()
        self.risk_manager = DummyRisk()

    async def update_position_from_fill(self, **_: Any) -> None:
        self.updated = True


@pytest.mark.asyncio
async def test_execute_order_success() -> None:
    engine = RobustExecutionEngine(DummyExchange(), DummyPM(), DummyRisk(), DummyDB())
    order = await engine.execute_order({"symbol": "BTCUSDT", "side": "BUY", "quantity": 1})
    assert order.is_filled


@pytest.mark.asyncio
async def test_retry_logic() -> None:
    exchange = DummyExchange()
    original = exchange.place_order

    async def place_order_fail_once(**kwargs: Any) -> dict:
        if exchange.calls == 0:
            exchange.calls += 1
            raise Exception("network")
        return await original(**kwargs)

    exchange.place_order = place_order_fail_once  # type: ignore[assignment]
    engine = RobustExecutionEngine(exchange, DummyPM(), DummyRisk(), DummyDB())
    order = await engine.execute_order({"symbol": "BTCUSDT", "side": "BUY", "quantity": 1})
    assert order.exchange_order_id == "1"
    assert exchange.calls >= 2
