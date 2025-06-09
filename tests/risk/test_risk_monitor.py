import pytest

from src.risk.risk_monitor import RiskMonitor, RiskEvent


@pytest.mark.asyncio
async def test_anomaly_detection() -> None:
    monitor = RiskMonitor(window_size=20)
    for _ in range(15):
        await monitor.record_event(RiskEvent(position_value=1000, trade_volume=10, pnl=0.01))
    assert not await monitor.detect_anomaly()
    await monitor.record_event(RiskEvent(position_value=1_000_000, trade_volume=10, pnl=0.01))
    assert await monitor.detect_anomaly()
