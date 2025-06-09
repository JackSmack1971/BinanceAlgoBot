import asyncio

import aiohttp
import pytest
from prometheus_client import REGISTRY

from src.monitoring.health_checks import HealthServer
from src.monitoring.observability import (
    ORDERS_TOTAL,
    SLIPPAGE_HIST,
    PORTFOLIO_VALUE,
    PNL_GAUGE,
)
from src.monitoring.performance_tracking import PerformanceTracker


@pytest.mark.asyncio
async def test_performance_metrics_update() -> None:
    await PerformanceTracker.record_order("s1")
    await PerformanceTracker.record_slippage("s1", 0.1)
    await PerformanceTracker.set_portfolio_value("acc", 100.0)
    await PerformanceTracker.record_pnl("acc", 5.0)

    assert ORDERS_TOTAL.labels(strategy="s1")._value.get() == 1
    count = REGISTRY.get_sample_value(
        "slippage_histogram_count", {"strategy": "s1"}
    )
    assert count == 1
    assert PORTFOLIO_VALUE.labels(account="acc")._value.get() == 100.0
    assert PNL_GAUGE.labels(account="acc")._value.get() == 5.0


@pytest.mark.asyncio
async def test_health_server() -> None:
    server = HealthServer(port=8081)
    await server.start()

    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8081/healthz") as resp:
            assert resp.status == 200
            assert await resp.text() == "ok"
    await server.stop()
