import pytest
from decimal import Decimal
from src.validation.financial_validator import (
    FinancialValidator,
    TradingLimits,
    ValidationError,
)

LIMITS = TradingLimits(
    max_position_size_usd=Decimal("100000"),
    max_order_value_usd=Decimal("1000000"),
    max_daily_volume_usd=Decimal("5000000"),
    max_leverage=Decimal("5"),
    min_order_size_usd=Decimal("10"),
    symbol_limits={
        "BTCUSDT": {
            "tick_size": Decimal("0.01"),
            "min_qty_usd": Decimal("10"),
            "reference_price": Decimal("30000"),
            "max_deviation_pct": Decimal("0.05"),
        }
    },
    max_open_positions=10,
    max_concentration_pct=Decimal("0.1"),
)

validator = FinancialValidator(LIMITS)


def test_validate_price_success():
    price = validator.validate_price(30000, "BTCUSDT", Decimal("30010"))
    assert isinstance(price, Decimal)


def test_validate_price_deviation():
    with pytest.raises(ValidationError):
        validator.validate_price(32000, "BTCUSDT", Decimal("30000"))


def test_validate_quantity_bounds():
    qty = validator.validate_quantity(0.001, "BTCUSDT", Decimal("200000"))
    assert qty == Decimal("0.00100000")


def test_validate_quantity_exceeds_position():
    with pytest.raises(ValidationError):
        validator.validate_quantity(10, "BTCUSDT", Decimal("1000"))
