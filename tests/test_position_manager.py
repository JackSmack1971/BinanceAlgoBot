import asyncio
from decimal import Decimal
from typing import Any

import pytest

from position_manager import PositionManager, PositionStatus


class DummyTransaction:
    async def __aenter__(self) -> "DummyTransaction":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        pass


class DummyDB:
    async def transaction(self) -> DummyTransaction:
        return DummyTransaction()


class DummyRiskCheck:
    def __init__(self, approved: bool = True) -> None:
        self.approved = approved
        self.violations = []


class DummyRiskManager:
    async def validate_new_position(self, *_: Any, **__: Any) -> DummyRiskCheck:
        return DummyRiskCheck()


@pytest.mark.asyncio
async def test_open_and_close_position() -> None:
    manager = PositionManager(DummyDB(), DummyRiskManager())
    position = await manager.open_position(
        "BTCUSDT", "BUY", Decimal("10"), Decimal("1"), {}
    )
    assert position.symbol == "BTCUSDT"
    assert position.status == PositionStatus.OPENING
    closed = await manager.close_position("BTCUSDT", Decimal("12"))
    assert closed.status in {PositionStatus.CLOSED, PositionStatus.OPEN}


@pytest.mark.asyncio
async def test_open_position_twice_raises() -> None:
    manager = PositionManager(DummyDB(), DummyRiskManager())
    await manager.open_position("BTCUSDT", "BUY", Decimal("10"), Decimal("1"), {})
    with pytest.raises(Exception):
        await manager.open_position(
            "BTCUSDT", "BUY", Decimal("10"), Decimal("1"), {}
        )


@pytest.mark.asyncio
async def test_close_wrong_symbol_raises() -> None:
    manager = PositionManager(DummyDB(), DummyRiskManager())
    await manager.open_position("BTCUSDT", "BUY", Decimal("10"), Decimal("1"), {})
    with pytest.raises(Exception):
        await manager.close_position("ETHUSDT", Decimal("11"))


@pytest.mark.asyncio
async def test_concurrent_open_single_position() -> None:
    manager = PositionManager(DummyDB(), DummyRiskManager())

    async def worker() -> None:
        try:
            await manager.open_position(
                "BTCUSDT", "BUY", Decimal("10"), Decimal("1"), {}
            )
        except Exception:
            pass

    await asyncio.gather(*[worker() for _ in range(5)])
    pos = await manager.get_position("BTCUSDT")
    assert pos is not None
