from __future__ import annotations

from pydantic import BaseModel, Field, PositiveFloat

from ..trading.types import Price, Quantity, Symbol


class RiskParameters(BaseModel):
    """Configuration for risk management."""

    max_risk: PositiveFloat = Field(..., lt=1)
    max_position_size: Quantity
    stop_loss: Price


class PositionRisk(BaseModel):
    symbol: Symbol
    quantity: Quantity
    entry_price: Price
    stop_loss: Price

    def potential_loss(self) -> Price:
        return Price((self.entry_price - self.stop_loss) * self.quantity)
