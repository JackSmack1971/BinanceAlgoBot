import pytest
from position_manager import PositionManager
from unittest.mock import MagicMock

def test_position_manager_creation():
    position_manager = PositionManager(initial_balance=10000, risk_per_trade=0.01)
    assert position_manager is not None

def test_position_manager_open_position():
    position_manager = PositionManager(initial_balance=10000, risk_per_trade=0.01)
    position_manager.open_position("BTC", "buy", 10000, 1)
    assert position_manager.get_position() is not None
    assert position_manager.get_position()["symbol"] == "BTC"
    assert position_manager.get_position()["side"] == "buy"
    assert position_manager.get_position()["entry_price"] == 10000
    assert position_manager.get_position()["size"] == 1

def test_position_manager_close_position():
    position_manager = PositionManager(initial_balance=10000, risk_per_trade=0.01)
    position_manager.open_position("BTC", "buy", 10000, 1)
    position_manager.close_position("BTC", 10100)
    assert position_manager.get_position() is None