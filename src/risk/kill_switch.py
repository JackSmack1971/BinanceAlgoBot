from __future__ import annotations

import asyncio
import logging
from typing import Iterable, TYPE_CHECKING

from exceptions import BaseTradingException

if TYPE_CHECKING:  # pragma: no cover
    from execution_engine import ExecutionEngine


class KillSwitchError(BaseTradingException):
    """Raised when kill switch operations fail."""


class KillSwitch:
    def __init__(self, engines: Iterable["ExecutionEngine"]) -> None:
        self.engines = list(engines)
        self.logger = logging.getLogger(__name__)
        self._configure_logger()

    def _configure_logger(self) -> None:
        fmt = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s"
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S"))
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.propagate = False

    async def activate(self) -> None:
        try:
            await asyncio.gather(*(engine.cancel_all_orders() for engine in self.engines))
            await asyncio.gather(*(engine.close_positions(order_by_loss=True) for engine in self.engines))
            self.logger.warning("Kill switch activated")
        except Exception as exc:
            self.logger.error("Kill switch failure: %s", exc, exc_info=True)
            raise KillSwitchError("Kill switch activation failed") from exc
