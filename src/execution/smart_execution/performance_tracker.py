from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from src.execution.order_types import OrderRequest, OrderResponse


@dataclass
class ExecutionRecord:
    request: OrderRequest
    response: OrderResponse
    slippage: float


@dataclass
class PerformanceTracker:
    records: List[ExecutionRecord] = field(default_factory=list)

    def add_record(self, request: OrderRequest, response: OrderResponse) -> None:
        filled_price = response.get("price", 0)
        expected = request.get("price", 0)
        slip = filled_price - expected
        self.records.append(ExecutionRecord(request, response, slip))

    def average_slippage(self) -> float:
        if not self.records:
            return 0.0
        total = sum(r.slippage for r in self.records)
        return total / len(self.records)
