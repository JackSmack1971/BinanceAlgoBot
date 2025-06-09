from __future__ import annotations

import asyncio
import os
from typing import Dict

from src.execution.order_types import OrderRequest, OrderResponse
from src.execution.order_validation import validate_order

from .execution_algorithms import BaseExecutionAlgorithm
from .venue_router import VenueRouter
from .performance_tracker import PerformanceTracker

API_TIMEOUT = float(os.environ.get("API_TIMEOUT", "5"))
RETRY_ATTEMPTS = int(os.environ.get("RETRY_ATTEMPTS", "3"))


class SmartExecutionEngine:
    def __init__(self, router: VenueRouter, algorithms: Dict[str, BaseExecutionAlgorithm],
                 tracker: PerformanceTracker) -> None:
        self.router = router
        self.algorithms = algorithms
        self.tracker = tracker

    async def _place_order(self, sub_order: OrderRequest) -> OrderResponse:
        for _ in range(RETRY_ATTEMPTS):
            try:
                await asyncio.sleep(0)  # simulate API call
                return {"order_id": 1, "symbol": sub_order["symbol"], "price": sub_order.get("price", 0)}
            except asyncio.TimeoutError:  # pragma: no cover
                await asyncio.sleep(0)
        raise TimeoutError("Order placement failed")

    async def execute_order(self, request: OrderRequest) -> None:
        await validate_order(request["symbol"], request["quantity"], request.get("price", 0))
        venue = await self.router.select_best_venue()
        algo = self.algorithms[request["order_type"]]
        for sub_order in await algo.generate_orders(request):
            response = await asyncio.wait_for(self._place_order(sub_order), API_TIMEOUT)
            self.tracker.add_record(sub_order, response)
