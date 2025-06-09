import asyncio
import logging
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime
from enum import Enum
import uuid
from contextlib import asynccontextmanager

from exceptions import OrderError

logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    OPENING = "opening"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"


@dataclass
class Position:
    """Thread-safe position representation"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: str = ""
    entry_price: Decimal = Decimal("0")
    quantity: Decimal = Decimal("0")
    current_price: Decimal = Decimal("0")
    status: PositionStatus = PositionStatus.OPENING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    max_loss_usd: Optional[Decimal] = None
    orders: List[str] = field(default_factory=list)
    fills: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.symbol or not self.side:
            raise ValueError("Symbol and side are required")
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")

    @property
    def market_value(self) -> Decimal:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> Decimal:
        if self.side == "BUY":
            return (self.current_price - self.entry_price) * self.quantity
        return (self.entry_price - self.current_price) * self.quantity


class AtomicPositionManager:
    """Thread-safe position manager with atomic operations"""

    def __init__(self, database_manager: Any, risk_manager: Any) -> None:
        self._positions: Dict[str, Position] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        self.db = database_manager
        self.risk_manager = risk_manager

    @asynccontextmanager
    async def _symbol_lock(self, symbol: str):
        async with self._global_lock:
            if symbol not in self._locks:
                self._locks[symbol] = asyncio.Lock()
            lock = self._locks[symbol]
        async with lock:
            yield

    async def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: Decimal,
        quantity: Decimal,
        risk_params: Dict[str, Any],
    ) -> Position:
        async with self._symbol_lock(symbol):
            existing_position = await self._get_position_atomic(symbol)
            if existing_position and existing_position.status in (
                PositionStatus.OPENING,
                PositionStatus.OPEN,
            ):
                raise OrderError(f"Position already exists for {symbol}: {existing_position.id}")

            risk_check = await self.risk_manager.validate_new_position(
                symbol, side, entry_price, quantity, self._positions
            )
            if not risk_check.approved:
                raise OrderError(f"Position violates risk limits: {risk_check.violations}")

            position = Position(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                quantity=quantity,
                current_price=entry_price,
                status=PositionStatus.OPENING,
                stop_loss=risk_params.get("stop_loss"),
                take_profit=risk_params.get("take_profit"),
                max_loss_usd=risk_params.get("max_loss_usd"),
            )

            async with self.db.transaction() as txn:
                await self._save_position_to_db(position, txn)

            self._positions[symbol] = position
            logger.info("Position opened atomically: %s for %s", position.id, symbol)
            return position

    async def update_position(self, symbol: str, **updates: Any) -> Position:
        async with self._symbol_lock(symbol):
            position = self._positions.get(symbol)
            if not position:
                raise OrderError(f"No position found for {symbol}")

            for field_name, value in updates.items():
                if field_name == "quantity" and value <= 0:
                    raise ValueError("Quantity must be positive")
                if field_name == "current_price" and value <= 0:
                    raise ValueError("Price must be positive")

            old_position = position
            updated_fields = {**position.__dict__, **updates, "updated_at": datetime.utcnow()}
            updated_position = Position(**updated_fields)

            async with self.db.transaction() as txn:
                await self._save_position_to_db(updated_position, txn)
                await self._archive_position_change(old_position, updated_position, txn)

            self._positions[symbol] = updated_position
            return updated_position

    async def close_position(
        self, symbol: str, exit_price: Decimal, close_quantity: Optional[Decimal] = None
    ) -> Position:
        async with self._symbol_lock(symbol):
            position = self._positions.get(symbol)
            if not position:
                raise OrderError(f"No position found for {symbol}")
            if position.status != PositionStatus.OPEN:
                raise OrderError(f"Cannot close position in status: {position.status}")

            close_qty = close_quantity or position.quantity
            if close_qty > position.quantity:
                raise ValueError("Cannot close more than position size")

            if position.side == "BUY":
                realized_pnl = (exit_price - position.entry_price) * close_qty
            else:
                realized_pnl = (position.entry_price - exit_price) * close_qty

            async with self.db.transaction() as txn:
                if close_qty == position.quantity:
                    position.status = PositionStatus.CLOSED
                    position.quantity = Decimal("0")
                    del self._positions[symbol]
                else:
                    position.quantity -= close_qty
                    position.updated_at = datetime.utcnow()
                await self._record_position_close(position, exit_price, close_qty, realized_pnl, txn)
                await self._save_position_to_db(position, txn)

            logger.info("Position closed atomically: %s of %s", close_qty, symbol)
            return position

    async def get_position(self, symbol: str) -> Optional[Position]:
        async with self._symbol_lock(symbol):
            return self._positions.get(symbol)

    async def get_all_positions(self) -> Dict[str, Position]:
        async with self._global_lock:
            return self._positions.copy()

    async def emergency_close_all_positions(self) -> List[Position]:
        async with self._global_lock:
            closed_positions = []
            for symbol, position in list(self._positions.items()):
                if position.status == PositionStatus.OPEN:
                    try:
                        current_price = await self._get_market_price(symbol)
                        closed_position = await self.close_position(symbol, current_price)
                        closed_positions.append(closed_position)
                    except Exception as exc:  # pragma: no cover - best effort
                        logger.error("Failed to emergency close %s: %s", symbol, exc)
            return closed_positions

    async def _get_position_atomic(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    async def _save_position_to_db(self, position: Position, txn: Any) -> None:
        pass  # Database persistence implementation

    async def _archive_position_change(
        self, old_position: Position, new_position: Position, txn: Any
    ) -> None:
        pass

    async def _record_position_close(
        self,
        position: Position,
        exit_price: Decimal,
        quantity: Decimal,
        realized_pnl: Decimal,
        txn: Any,
    ) -> None:
        pass

    async def _get_market_price(self, symbol: str) -> Decimal:
        raise NotImplementedError


PositionManager = AtomicPositionManager
