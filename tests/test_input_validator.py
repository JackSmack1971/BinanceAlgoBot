import hmac
from hashlib import sha256
import pytest
from pydantic import ValidationError

from src.validation.input_validator import (
    TradingSymbolValidator,
    TradingOrderValidator,
    verify_signature,
)


def test_trading_symbol_validator_success():
    assert TradingSymbolValidator(symbol="BTCUSDT").symbol == "BTCUSDT"


def test_trading_symbol_validator_failure():
    with pytest.raises(ValidationError):
        TradingSymbolValidator(symbol="BAD")


def test_trading_order_validator_bounds():
    order = TradingOrderValidator(symbol="ETHUSDT", quantity=1, price=10)
    assert order.quantity == 1
    with pytest.raises(ValidationError):
        TradingOrderValidator(symbol="ETHUSDT", quantity=-1, price=10)


def test_verify_signature():
    secret = "s3cr3t"
    msg = "test"
    sig = hmac.new(secret.encode(), msg.encode(), sha256).hexdigest()
    assert verify_signature(msg, sig, secret)
