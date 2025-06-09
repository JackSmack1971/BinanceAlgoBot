from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .smart_execution_engine import SmartExecutionEngine


@dataclass
class BacktestResult:
    slippage: float


async def run_backtest(engine: SmartExecutionEngine, orders: Iterable[dict]) -> BacktestResult:
    for req in orders:
        await engine.execute_order(req)
    return BacktestResult(slippage=engine.tracker.average_slippage())
