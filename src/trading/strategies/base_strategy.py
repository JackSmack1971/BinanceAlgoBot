from __future__ import annotations

from typing import Protocol, runtime_checkable
from pandas import DataFrame

from ...trading.types import Price, Quantity, Symbol


@runtime_checkable
class StrategyProtocol(Protocol):
    """Protocol for trading strategies."""

    symbol: Symbol
    interval: str

    def calculate_indicators(self) -> DataFrame | None:
        ...

    def run(self) -> DataFrame | None:
        ...

    def open_position(self, side: str, price: Price, size: Quantity) -> None:
        ...

    def close_position(self, price: Price) -> None:
        ...
