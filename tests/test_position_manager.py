import pytest

from position_manager import PositionManager
from exceptions import OrderError


def test_open_and_close_position():
    manager = PositionManager(1000, 0.1)
    manager.open_position("BTCUSDT", "buy", 10, 1)
    assert manager.get_position()["symbol"] == "BTCUSDT"
    balance = manager.get_balance()
    profit = manager.close_position("BTCUSDT", 12)
    assert profit == 2
    assert manager.get_position() is None
    assert manager.get_balance() > balance


def test_open_position_twice_raises():
    manager = PositionManager(1000, 0.1)
    manager.open_position("BTCUSDT", "buy", 10, 1)
    with pytest.raises(OrderError):
        manager.open_position("BTCUSDT", "buy", 10, 1)


def test_close_wrong_symbol_raises():
    manager = PositionManager(1000, 0.1)
    manager.open_position("BTCUSDT", "buy", 10, 1)
    with pytest.raises(OrderError):
        manager.close_position("ETHUSDT", 11)
