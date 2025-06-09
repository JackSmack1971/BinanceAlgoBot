import pytest

from src.execution.smart_execution.backtesting import run_backtest
from src.execution.smart_execution.market_impact_model import MarketImpactModel
from src.execution.smart_execution.execution_algorithms import TWAPExecutionAlgorithm
from src.execution.smart_execution.smart_execution_engine import SmartExecutionEngine
from src.execution.smart_execution.venue_router import Venue, VenueRouter
from src.execution.smart_execution.performance_tracker import PerformanceTracker


@pytest.mark.asyncio
async def test_run_backtest(monkeypatch):
    router = VenueRouter([Venue("A", 0.001, 0.1, 0.9)])
    tracker = PerformanceTracker()
    algo = TWAPExecutionAlgorithm(MarketImpactModel())
    engine = SmartExecutionEngine(router, {"limit": algo}, tracker)

    async def dummy_place_order(self, order):
        return {"order_id": 1, "symbol": order["symbol"], "price": order["price"]}

    monkeypatch.setattr(engine, "_place_order", dummy_place_order)

    orders = [
        {
            "symbol": "BTCUSDT",
            "side": "buy",
            "quantity": 2.0,
            "order_type": "limit",
            "price": 10.0,
            "time_in_force": "GTC",
        }
    ]
    result = await run_backtest(engine, orders)
    assert result.slippage == tracker.average_slippage()
