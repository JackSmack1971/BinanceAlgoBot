from __future__ import annotations

from .observability import (
    ORDERS_TOTAL,
    SLIPPAGE_HIST,
    PORTFOLIO_VALUE,
    PNL_GAUGE,
    RISK_EXPOSURE,
    MARKET_DATA_LATENCY,
)


class PerformanceTracker:
    """Track trading performance metrics."""

    @staticmethod
    async def record_order(strategy: str) -> None:
        ORDERS_TOTAL.labels(strategy=strategy).inc()

    @staticmethod
    async def record_slippage(strategy: str, value: float) -> None:
        SLIPPAGE_HIST.labels(strategy=strategy).observe(value)

    @staticmethod
    async def set_portfolio_value(account: str, value: float) -> None:
        PORTFOLIO_VALUE.labels(account=account).set(value)

    @staticmethod
    async def record_pnl(account: str, value: float) -> None:
        PNL_GAUGE.labels(account=account).set(value)

    @staticmethod
    async def set_risk(account: str, value: float) -> None:
        RISK_EXPOSURE.labels(account=account).set(value)

    @staticmethod
    async def record_market_latency(value: float) -> None:
        MARKET_DATA_LATENCY.observe(value)
