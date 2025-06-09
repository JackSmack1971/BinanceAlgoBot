from typing import Iterable

ALLOWED_INTERVALS = {
    '1m', '3m', '5m', '15m', '30m', '1h'
}


def validate_symbol(symbol: str) -> str:
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string")
    if not symbol.endswith(("USDT", "BTC", "ETH")):
        raise ValueError("Invalid trading pair")
    return symbol.upper()


def validate_quantity(quantity: float) -> float:
    if not isinstance(quantity, (int, float)):
        raise ValueError("Quantity must be a number")
    if quantity <= 0:
        raise ValueError("Quantity must be positive")
    return float(quantity)


def validate_timeframe(interval: str) -> str:
    if interval not in ALLOWED_INTERVALS:
        raise ValueError(f"Invalid timeframe: {interval}")
    return interval


def validate_risk(value: float) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError("Risk parameter must be a number")
    if value <= 0 or value > 1:
        raise ValueError("Risk parameter must be between 0 and 1")
    return float(value)


def sanitize_input(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Input must be a string")
    return value.replace(";", "").replace("--", "").strip()
