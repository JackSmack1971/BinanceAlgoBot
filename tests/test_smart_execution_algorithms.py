import pytest

from src.execution.smart_execution.market_impact_model import MarketImpactModel
from src.execution.smart_execution.execution_algorithms import (
    TWAPExecutionAlgorithm,
    AdaptiveExecutionAlgorithm,
    VWAPExecutionAlgorithm,
    IcebergExecutionAlgorithm,
)


@pytest.mark.asyncio
async def test_twap_generate_orders():
    model = MarketImpactModel()
    algo = TWAPExecutionAlgorithm(model, slices=3)
    request = {
        "symbol": "BTCUSDT",
        "side": "buy",
        "quantity": 9.0,
        "order_type": "limit",
        "price": 10.0,
        "time_in_force": "GTC",
    }
    orders = await algo.generate_orders(request)
    assert len(orders) == 3
    assert all(o["quantity"] == 3 for o in orders)


@pytest.mark.asyncio
async def test_adaptive_generate_orders():
    model = MarketImpactModel(base_impact=0.1)
    algo = AdaptiveExecutionAlgorithm(model, base_slices=2)
    request = {
        "symbol": "BTCUSDT",
        "side": "buy",
        "quantity": 10.0,
        "order_type": "limit",
        "price": 10.0,
        "time_in_force": "GTC",
    }
    orders = await algo.generate_orders(request)
    assert len(orders) >= 2

@pytest.mark.asyncio
async def test_vwap_generate_orders():
    model = MarketImpactModel()
    algo = VWAPExecutionAlgorithm(model, volume_profile=[1, 2])
    request = {
        "symbol": "BTCUSDT",
        "side": "buy",
        "quantity": 3.0,
        "order_type": "limit",
        "price": 10.0,
        "time_in_force": "GTC",
    }
    orders = await algo.generate_orders(request)
    assert len(orders) == 2
    assert orders[0]["quantity"] == pytest.approx(1.0)
    assert orders[1]["quantity"] == pytest.approx(2.0)

@pytest.mark.asyncio
async def test_iceberg_generate_orders():
    model = MarketImpactModel()
    algo = IcebergExecutionAlgorithm(model, visible_size=2.0)
    request = {
        "symbol": "BTCUSDT",
        "side": "buy",
        "quantity": 5.0,
        "order_type": "limit",
        "price": 10.0,
        "time_in_force": "GTC",
    }
    orders = await algo.generate_orders(request)
    assert len(orders) == 3
    assert orders[0]["quantity"] == 2.0
    assert orders[-1]["quantity"] == 1.0


def test_market_impact_estimate():
    model = MarketImpactModel(base_impact=0.1)
    assert model.estimate(10) == 1.0

