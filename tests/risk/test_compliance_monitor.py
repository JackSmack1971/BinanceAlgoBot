import pytest

from src.risk.compliance_monitor import ComplianceMonitor, Trade, ComplianceViolation


def test_compliance_monitor() -> None:
    monitor = ComplianceMonitor()
    valid = [Trade(symbol="BTCUSDT", quantity=1000, side="buy")]
    monitor.check_trades(valid)

    invalid = [Trade(symbol="ETHUSDT", quantity=1_500_000, side="sell")]
    with pytest.raises(ComplianceViolation):
        monitor.check_trades(invalid)
