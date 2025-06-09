from __future__ import annotations

import logging
from typing import Iterable

from pydantic import BaseModel, Field, PositiveFloat


class Trade(BaseModel):
    symbol: str
    quantity: PositiveFloat
    side: str = Field(..., pattern="^(buy|sell)$")


class ComplianceViolation(Exception):
    pass


class ComplianceMonitor:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        fmt = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s"
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S"))
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.propagate = False

    def check_trades(self, trades: Iterable[Trade]) -> None:
        for trade in trades:
            if trade.quantity > 1e6:
                self.logger.error("Trade size exceeds limit: %s", trade)
                raise ComplianceViolation("Trade size exceeds limit")
        self.logger.info("Compliance check passed for %d trades", len(list(trades)))
