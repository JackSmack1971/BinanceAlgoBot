import pytest
from validation import (
    validate_symbol,
    validate_quantity,
    validate_timeframe,
    validate_risk,
    sanitize_input,
)


def test_validate_symbol_success():
    assert validate_symbol("btcusdt") == "BTCUSDT"


def test_validate_symbol_failure():
    with pytest.raises(ValueError):
        validate_symbol("BTC")


def test_validate_quantity():
    assert validate_quantity(1.5) == 1.5
    with pytest.raises(ValueError):
        validate_quantity(-1)


def test_validate_timeframe():
    assert validate_timeframe("1m") == "1m"
    with pytest.raises(ValueError):
        validate_timeframe("2h")


def test_validate_risk():
    assert validate_risk(0.5) == 0.5
    with pytest.raises(ValueError):
        validate_risk(1.5)


def test_sanitize_input():
    assert sanitize_input(" test;") == "test"
    with pytest.raises(ValueError):
        sanitize_input(123)
