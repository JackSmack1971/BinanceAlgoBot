import asyncio
import pytest

from src.risk.kill_switch import KillSwitch


class DummyEngine:
    def __init__(self) -> None:
        self.cancelled = False
        self.closed = False

    async def cancel_all_orders(self) -> None:
        self.cancelled = True

    async def close_positions(self, order_by_loss: bool = False) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_kill_switch() -> None:
    engine = DummyEngine()
    ks = KillSwitch([engine])
    await ks.activate()
    assert engine.cancelled and engine.closed
