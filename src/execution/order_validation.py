from __future__ import annotations

from pydantic import ValidationError
from src.validation.input_validator import TradingOrderValidator


async def validate_order(symbol: str, quantity: float, price: float) -> TradingOrderValidator:
    """Validate order parameters using TradingOrderValidator."""
    try:
        return TradingOrderValidator(symbol=symbol, quantity=quantity, price=price)
    except ValidationError as exc:
        raise

