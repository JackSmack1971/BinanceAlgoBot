import pytest

from src.execution.smart_execution.performance_tracker import PerformanceTracker


def test_tracker_average_slippage():
    tracker = PerformanceTracker()
    req = {"symbol": "BTCUSDT", "price": 10.0}
    res = {"price": 10.5}
    tracker.add_record(req, res)
    assert tracker.average_slippage() == 0.5
