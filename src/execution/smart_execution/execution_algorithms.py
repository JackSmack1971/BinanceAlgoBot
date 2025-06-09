from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import List

from src.execution.order_types import OrderRequest
from .market_impact_model import MarketImpactModel


class ExecutionAlgoError(Exception):
    """Custom exception for execution algorithm errors."""


class BaseExecutionAlgorithm(ABC):
    def __init__(self, impact_model: MarketImpactModel) -> None:
        self.impact_model = impact_model

    @abstractmethod
    async def generate_orders(self, request: OrderRequest) -> List[OrderRequest]:
        """Split the request into smaller orders."""
        raise NotImplementedError


class TWAPExecutionAlgorithm(BaseExecutionAlgorithm):
    def __init__(self, impact_model: MarketImpactModel, slices: int = 4) -> None:
        super().__init__(impact_model)
        self.slices = slices

    async def generate_orders(self, request: OrderRequest) -> List[OrderRequest]:
        try:
            slice_qty = request["quantity"] / self.slices
            orders = []
            for _ in range(self.slices):
                sub_req = dict(request)
                sub_req["quantity"] = slice_qty
                orders.append(sub_req)
                await asyncio.sleep(0)
            return orders
        except Exception as exc:  # pragma: no cover
            raise ExecutionAlgoError("TWAP split failed") from exc


class VWAPExecutionAlgorithm(BaseExecutionAlgorithm):
    def __init__(self, impact_model: MarketImpactModel, volume_profile: List[float]) -> None:
        super().__init__(impact_model)
        self.volume_profile = volume_profile

    async def generate_orders(self, request: OrderRequest) -> List[OrderRequest]:
        try:
            total = sum(self.volume_profile)
            orders = []
            for weight in self.volume_profile:
                sub_req = dict(request)
                sub_req["quantity"] = request["quantity"] * weight / total
                orders.append(sub_req)
                await asyncio.sleep(0)
            return orders
        except Exception as exc:  # pragma: no cover
            raise ExecutionAlgoError("VWAP split failed") from exc


class IcebergExecutionAlgorithm(BaseExecutionAlgorithm):
    def __init__(self, impact_model: MarketImpactModel, visible_size: float) -> None:
        super().__init__(impact_model)
        self.visible_size = visible_size

    async def generate_orders(self, request: OrderRequest) -> List[OrderRequest]:
        try:
            qty = request["quantity"]
            orders = []
            while qty > 0:
                sub_req = dict(request)
                sub_req["quantity"] = min(self.visible_size, qty)
                orders.append(sub_req)
                qty -= sub_req["quantity"]
                await asyncio.sleep(0)
            return orders
        except Exception as exc:  # pragma: no cover
            raise ExecutionAlgoError("Iceberg split failed") from exc


class AdaptiveExecutionAlgorithm(BaseExecutionAlgorithm):
    def __init__(self, impact_model: MarketImpactModel, base_slices: int = 4) -> None:
        super().__init__(impact_model)
        self.base_slices = base_slices

    async def generate_orders(self, request: OrderRequest) -> List[OrderRequest]:
        impact = self.impact_model.estimate(request["quantity"])
        slices = max(1, int(self.base_slices * (1 + impact)))
        algo = TWAPExecutionAlgorithm(self.impact_model, slices)
        return await algo.generate_orders(request)
