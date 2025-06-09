from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from src.risk import (
    AdvancedCircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    RateLimitError,
)
from src.monitoring import AuditTrailRecorder

__all__ = [
    "OrderStatus",
    "OrderType",
    "OrderFill",
    "TradingOrder",
    "RobustExecutionEngine",
]


class ExecutionError(Exception):
    """Base execution error"""


class ExchangeConnectionError(ExecutionError):
    """Raised when the exchange is unreachable"""


class InsufficientBalanceError(ExecutionError):
    """Raised when account balance is insufficient"""


class OrderRejectedError(ExecutionError):
    """Raised when an order is rejected by risk checks"""


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


@dataclass
class OrderFill:
    """Represents a single order fill"""

    fill_id: str
    order_id: str
    price: Decimal
    quantity: Decimal
    commission: Decimal
    commission_asset: str
    timestamp: datetime
    trade_id: str


@dataclass
class TradingOrder:
    """Comprehensive order representation"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_order_id: str = field(
        default_factory=lambda: f"BOT_{int(time.time() * 1000)}"
    )
    exchange_order_id: Optional[str] = None
    symbol: str = ""
    side: str = ""
    order_type: OrderType = OrderType.MARKET
    quantity: Decimal = Decimal("0")
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = Decimal("0")
    remaining_quantity: Decimal = Decimal("0")
    avg_fill_price: Decimal = Decimal("0")
    total_commission: Decimal = Decimal("0")
    created_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=datetime.utcnow)
    fills: List[OrderFill] = field(default_factory=list)
    error_count: int = 0
    last_error: Optional[str] = None
    retry_count: int = 0
    max_slippage_pct: Decimal = Decimal("0.05")
    execution_timeout_seconds: int = 300

    def __post_init__(self) -> None:
        self.remaining_quantity = self.quantity

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def is_active(self) -> bool:
        return self.status in {
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
        }

    @property
    def fill_percentage(self) -> Decimal:
        if self.quantity == 0:
            return Decimal("0")
        return (self.filled_quantity / self.quantity) * 100

    def add_fill(self, fill: OrderFill) -> None:
        self.fills.append(fill)
        self.filled_quantity += fill.quantity
        self.remaining_quantity = self.quantity - self.filled_quantity
        self.total_commission += fill.commission
        total_value = sum(f.price * f.quantity for f in self.fills)
        self.avg_fill_price = (
            total_value / self.filled_quantity
            if self.filled_quantity > 0
            else Decimal("0")
        )
        if self.remaining_quantity <= 0:
            self.status = OrderStatus.FILLED
            self.filled_at = datetime.utcnow()
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
        self.updated_at = datetime.utcnow()


class ExecutionMetrics:
    """Collect simple execution metrics"""

    def __init__(self) -> None:
        self.total_orders: int = 0

    def record_execution(self, order: TradingOrder) -> None:
        self.total_orders += 1


class RobustExecutionEngine:
    """Comprehensive trade execution engine"""

    def __init__(
        self,
        exchange_interface: Any,
        position_manager: Any,
        risk_manager: Any,
        database_manager: Any,
        breaker: AdvancedCircuitBreaker | None = None,
        audit_trail: AuditTrailRecorder | None = None,
    ) -> None:
        self.exchange = exchange_interface
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.db = database_manager
        self.breaker = breaker or AdvancedCircuitBreaker(
            "execution_engine",
            CircuitBreakerConfig(max_requests_per_minute=30, request_timeout=10.0),
        )
        self._active_orders: Dict[str, TradingOrder] = {}
        self._order_lock = asyncio.Lock()
        self._reconciliation_interval = timedelta(minutes=5)
        self.execution_metrics = ExecutionMetrics()
        self.logger = logging.getLogger(__name__)
        self._reconciliation_task: Optional[asyncio.Task[Any]] = None
        self.audit_trail = audit_trail or AuditTrailRecorder()

    async def start(self) -> None:  # pragma: no cover
        self._reconciliation_task = asyncio.create_task(self._reconciliation_loop())
        self.logger.info("Execution engine started")

    async def execute_order(self, order_params: Dict[str, Any]) -> TradingOrder:
        order = await self._create_order(order_params)
        await self._validate_order(order)
        risk_check = await self.risk_manager.validate_order(order)
        if not getattr(risk_check, "approved", False):
            order.status = OrderStatus.REJECTED
            order.last_error = f"Risk check failed: {getattr(risk_check, 'reason', '')}"
            await self._save_order(order)
            raise OrderRejectedError(order.last_error)
        async with self._order_lock:
            self._active_orders[order.id] = order
        try:
            await self._submit_order_with_retry(order)
            await self._monitor_order_execution(order)
            await self._finalize_order_execution(order)
            return order
        except Exception as exc:
            await self._handle_execution_error(order, exc)
            raise
        finally:
            async with self._order_lock:
                self._active_orders.pop(order.id, None)

    async def _submit_order_with_retry(
        self, order: TradingOrder, max_retries: int = 3
    ) -> None:
        for attempt in range(max_retries + 1):
            try:
                order.submitted_at = datetime.utcnow()
                order.status = OrderStatus.SUBMITTED

                async def _call() -> Dict[str, Any]:
                    return await self.exchange.place_order(
                        symbol=order.symbol,
                        side=order.side,
                        type=order.order_type.value,
                        quantity=str(order.quantity),
                        price=str(order.price) if order.price else None,
                        stopPrice=str(order.stop_price) if order.stop_price else None,
                        newClientOrderId=order.client_order_id,
                    )

                response = await self.breaker.call(_call)
                order.exchange_order_id = response.get("orderId")
                if not order.exchange_order_id:
                    raise ExecutionError("No order ID returned")
                await self._save_order(order)
                self.logger.info("Order submitted: %s", order.id)
                return
            except Exception as exc:
                order.error_count += 1
                order.last_error = str(exc)
                order.retry_count = attempt
                if attempt < max_retries:
                    delay = 2**attempt
                    self.logger.warning(
                        "Submit failed, retrying in %ss: %s", delay, exc
                    )
                    await asyncio.sleep(delay)
                else:
                    order.status = OrderStatus.FAILED
                    await self._save_order(order)
                    raise ExecutionError(f"Failed to submit order: {exc}")

    async def _monitor_order_execution(self, order: TradingOrder) -> None:
        start = datetime.utcnow()
        timeout = timedelta(seconds=order.execution_timeout_seconds)
        while order.is_active and datetime.utcnow() - start < timeout:
            try:

                async def _call() -> Dict[str, Any]:
                    return await self.exchange.get_order(
                        symbol=order.symbol, orderId=order.exchange_order_id
                    )

                exchange_order = await self.breaker.call(_call)
                await self._update_order_from_exchange(order, exchange_order)
                if not order.is_active:
                    break
                await asyncio.sleep(1 if order.order_type == OrderType.MARKET else 5)
            except Exception as exc:
                self.logger.error("Monitor error for %s: %s", order.id, exc)
                await asyncio.sleep(5)
        if order.is_active:
            await self._handle_order_timeout(order)

    async def _update_order_from_exchange(
        self, order: TradingOrder, data: Dict[str, Any]
    ) -> None:
        status_map = {
            "NEW": OrderStatus.SUBMITTED,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }
        order.status = status_map.get(data.get("status", "").upper(), order.status)
        fills = data.get("fills", [])
        for fill_data in fills[len(order.fills) :]:
            fill = OrderFill(
                fill_id=fill_data.get("tradeId", str(uuid.uuid4())),
                order_id=order.id,
                price=Decimal(fill_data["price"]),
                quantity=Decimal(fill_data["qty"]),
                commission=Decimal(fill_data.get("commission", "0")),
                commission_asset=fill_data.get("commissionAsset", ""),
                timestamp=datetime.utcnow(),
                trade_id=fill_data.get("tradeId", ""),
            )
            order.add_fill(fill)
        order.updated_at = datetime.utcnow()
        await self._save_order(order)

    async def _validate_execution_slippage(self, order: TradingOrder) -> None:
        if not order.price or not order.fills:
            return
        expected = order.price
        actual = order.avg_fill_price
        slippage = (
            (actual - expected) / expected * 100
            if order.side == "BUY"
            else (expected - actual) / expected * 100
        )
        if slippage > order.max_slippage_pct:
            self.logger.warning(
                "High slippage %.2f%% on order %s", float(slippage), order.id
            )
            await self._record_slippage_event(order, slippage)

    async def _finalize_order_execution(self, order: TradingOrder) -> None:
        if not order.fills:
            return
        for fill in order.fills:
            await self.position_manager.update_position_from_fill(
                symbol=order.symbol,
                side=order.side,
                price=fill.price,
                quantity=fill.quantity,
                commission=fill.commission,
                order_id=order.id,
            )
        self.execution_metrics.record_execution(order)
        await self._generate_execution_report(order)

    async def _handle_execution_error(
        self, order: TradingOrder, error: Exception
    ) -> None:
        order.status = OrderStatus.FAILED
        order.last_error = str(error)
        order.error_count += 1
        await self._save_order(order)
        if order.exchange_order_id:
            try:

                async def _call() -> Any:
                    return await self.exchange.cancel_order(
                        symbol=order.symbol, orderId=order.exchange_order_id
                    )

                await self.breaker.call(_call)
                self.logger.info("Cancelled order %s", order.id)
            except Exception as cancel_error:
                self.logger.error("Cancel failed for %s: %s", order.id, cancel_error)
        if isinstance(error, (ExchangeConnectionError, InsufficientBalanceError)):
            await self._send_critical_error_alert(order, error)

    async def _reconciliation_loop(self) -> None:  # pragma: no cover
        while True:
            try:
                await self._reconcile_active_orders()
                await asyncio.sleep(self._reconciliation_interval.total_seconds())
            except Exception as exc:
                self.logger.error("Reconciliation error: %s", exc)
                await asyncio.sleep(60)

    async def _reconcile_active_orders(self) -> None:  # pragma: no cover
        async with self._order_lock:
            orders = list(self._active_orders.values())
        for order in orders:
            if order.exchange_order_id:
                try:

                    async def _call() -> Dict[str, Any]:
                        return await self.exchange.get_order(
                            symbol=order.symbol, orderId=order.exchange_order_id
                        )

                    data = await self.breaker.call(_call)
                    await self._update_order_from_exchange(order, data)
                except Exception as exc:
                    self.logger.error("Failed to reconcile %s: %s", order.id, exc)

    async def cancel_order(self, order_id: str) -> bool:  # pragma: no cover
        async with self._order_lock:
            order = self._active_orders.get(order_id)
        if not order or not order.is_active:
            return False
        try:
            if order.exchange_order_id:

                async def _call() -> Any:
                    return await self.exchange.cancel_order(
                        symbol=order.symbol, orderId=order.exchange_order_id
                    )

                await self.breaker.call(_call)
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.utcnow()
            await self._save_order(order)
            return True
        except Exception as exc:
            self.logger.error("Failed to cancel %s: %s", order_id, exc)
            return False

    async def _handle_order_timeout(
        self, order: TradingOrder
    ) -> None:  # pragma: no cover
        order.status = OrderStatus.EXPIRED
        await self._save_order(order)

    async def _send_critical_error_alert(
        self, order: TradingOrder, error: Exception
    ) -> None:  # pragma: no cover
        self.logger.error("Critical error on %s: %s", order.id, error)

    async def _record_slippage_event(
        self, order: TradingOrder, slippage: Decimal
    ) -> None:  # pragma: no cover
        self.logger.info("Recorded slippage %.2f%% for %s", float(slippage), order.id)

    async def _generate_execution_report(
        self, order: TradingOrder
    ) -> None:  # pragma: no cover
        self.logger.info("Execution report generated for %s", order.id)

    async def _create_order(self, params: Dict[str, Any]) -> TradingOrder:
        order = TradingOrder(
            symbol=params["symbol"],
            side=params["side"],
            order_type=OrderType(params.get("order_type", "market")),
            quantity=Decimal(str(params["quantity"])),
            price=(
                Decimal(str(params.get("price", "0"))) if params.get("price") else None
            ),
            stop_price=(
                Decimal(str(params.get("stop_price", "0")))
                if params.get("stop_price")
                else None
            ),
        )
        return order

    async def _validate_order(self, order: TradingOrder) -> None:
        if not order.symbol or order.quantity <= 0:
            raise OrderRejectedError("Invalid order parameters")

    async def _save_order(self, order: TradingOrder) -> None:
        async with self.db.transaction():
            await self.db.save_order(order)
        await self.audit_trail.record_event(
            order.status.value,
            json.loads(json.dumps(order.__dict__, default=str)),
        )
