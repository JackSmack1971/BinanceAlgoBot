from __future__ import annotations

from hashlib import sha256
import hmac
import os
from typing import ClassVar

from pydantic import BaseModel, ValidationError, field_validator, ConfigDict

ALLOWED_SYMBOLS: set[str] = {
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
}


def verify_signature(message: str, signature: str, secret: str) -> bool:
    computed = hmac.new(secret.encode(), message.encode(), sha256).hexdigest()
    return hmac.compare_digest(computed, signature)


class TradingSymbolValidator(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str

    @field_validator("symbol")
    @classmethod
    def symbol_whitelist(cls, v: str) -> str:
        if v.upper() not in ALLOWED_SYMBOLS:
            raise ValueError(f"Symbol {v} not allowed")
        return v.upper()


class TradingOrderValidator(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str
    quantity: float
    price: float

    MAX_QTY: ClassVar[float] = 1_000_000
    MAX_PRICE: ClassVar[float] = 10_000_000

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        if v.upper() not in ALLOWED_SYMBOLS:
            raise ValueError(f"Symbol {v} not allowed")
        return v.upper()

    @field_validator("quantity")
    @classmethod
    def validate_quantity(cls, v: float) -> float:
        if v <= 0 or v > cls.MAX_QTY:
            raise ValueError("Quantity out of bounds")
        return v

    @field_validator("price")
    @classmethod
    def validate_price(cls, v: float) -> float:
        if v <= 0 or v > cls.MAX_PRICE:
            raise ValueError("Price out of bounds")
        return v

