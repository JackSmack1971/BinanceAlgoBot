from decimal import Decimal
from typing import Iterable
from src.validation.financial_validator import (
    FinancialValidator,
    TradingLimits,
    ValidationError,
)

ALLOWED_INTERVALS: set[str] = {"1m", "3m", "5m", "15m", "30m", "1h"}

# Example symbol-specific limits; in production these should come from config
_SYMBOL_LIMITS = {
    "BTCUSDT": {"tick_size": Decimal("0.01"), "min_qty_usd": Decimal("10"), "reference_price": Decimal("30000"), "max_deviation_pct": Decimal("0.05")},
    "ETHUSDT": {"tick_size": Decimal("0.01"), "min_qty_usd": Decimal("10"), "reference_price": Decimal("2000"), "max_deviation_pct": Decimal("0.05")},
}

_DEFAULT_LIMITS = TradingLimits(
    max_position_size_usd=Decimal("100000"),
    max_order_value_usd=Decimal("1000000"),
    max_daily_volume_usd=Decimal("5000000"),
    max_leverage=Decimal("5"),
    min_order_size_usd=Decimal("10"),
    symbol_limits=_SYMBOL_LIMITS,
    max_open_positions=10,
    max_concentration_pct=Decimal("0.1"),
)

_validator = FinancialValidator(_DEFAULT_LIMITS)


def validate_symbol(symbol: str) -> str:
    return _validator.validate_symbol(symbol)


def validate_quantity(quantity: float, symbol: str = "BTCUSDT", portfolio_value: float = 0.0) -> float:
    qty = _validator.validate_quantity(quantity, symbol, Decimal(str(portfolio_value)))
    return float(qty)


def validate_price(price: float, symbol: str, market_price: float) -> float:
    pr = _validator.validate_price(price, symbol, Decimal(str(market_price)))
    return float(pr)


def validate_timeframe(interval: str) -> str:
    if interval not in ALLOWED_INTERVALS:
        raise ValidationError(f"Invalid timeframe: {interval}")
    return interval


def validate_risk(value: float) -> float:
    if value <= 0 or value > 1:
        raise ValidationError("Risk parameter must be between 0 and 1")
    return float(value)


def sanitize_input(value: str) -> str:
    if not isinstance(value, str):
        raise ValidationError("Input must be a string")
    return value.replace(";", "").replace("--", "").strip()
