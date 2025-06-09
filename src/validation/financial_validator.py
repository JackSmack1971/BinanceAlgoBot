from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from enum import Enum
import logging
import re
from typing import Dict, Any, Union, Optional

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation failures"""


class RiskLimitType(Enum):
    POSITION_SIZE = "position_size"
    ORDER_VALUE = "order_value"
    DAILY_VOLUME = "daily_volume"
    LEVERAGE = "leverage"


@dataclass
class TradingLimits:
    max_position_size_usd: Decimal
    max_order_value_usd: Decimal
    max_daily_volume_usd: Decimal
    max_leverage: Decimal
    min_order_size_usd: Decimal
    symbol_limits: Dict[str, Dict[str, Decimal]]
    max_open_positions: int
    max_concentration_pct: Decimal


class FinancialValidator:
    def __init__(self, trading_limits: TradingLimits):
        self.limits = trading_limits
        self.decimal_places = 8
        self._symbol_regex = re.compile(r"^[A-Z0-9]{6,12}$")

    def _to_decimal(self, value: Union[str, float, Decimal]) -> Decimal:
        return Decimal(str(value)).quantize(Decimal(10) ** -self.decimal_places)

    def validate_price(self, price: Union[str, float, Decimal], symbol: str,
                       current_market_price: Decimal) -> Decimal:
        try:
            price_dec = self._to_decimal(price)
            if price_dec <= 0:
                raise ValidationError("Price must be positive")
            limits = self.limits.symbol_limits.get(symbol, {})
            tick = limits.get("tick_size", Decimal("0.00000001"))
            if (price_dec % tick) != 0:
                raise ValidationError(f"Price not aligned with tick size {tick}")
            max_dev = limits.get("max_deviation_pct", Decimal("0.05"))
            if current_market_price > 0 and abs(price_dec - current_market_price) > current_market_price * max_dev:
                raise ValidationError("Price deviates from market price too much")
            return price_dec
        except Exception as exc:  # noqa: BLE001
            logger.error("Price validation failed: %s", exc)
            raise

    def validate_quantity(self, quantity: Union[str, float, Decimal], symbol: str,
                          current_portfolio_value: Decimal) -> Decimal:
        try:
            qty = self._to_decimal(quantity)
            if qty <= 0:
                raise ValidationError("Quantity must be positive")
            sym_limits = self.limits.symbol_limits.get(symbol, {})
            min_usd = sym_limits.get("min_qty_usd", self.limits.min_order_size_usd)
            ref_price = sym_limits.get("reference_price", Decimal("0"))
            order_value = qty * ref_price
            if ref_price > 0 and order_value < min_usd:
                raise ValidationError("Order value below minimum size")
            if ref_price > 0 and order_value > self.limits.max_position_size_usd:
                raise ValidationError("Order value exceeds max position size")
            if current_portfolio_value > 0 and order_value > current_portfolio_value * self.limits.max_concentration_pct:
                raise ValidationError("Position concentration limit breached")
            return qty
        except Exception as exc:  # noqa: BLE001
            logger.error("Quantity validation failed: %s", exc)
            raise

    def validate_order_value(self, price: Decimal, quantity: Decimal, side: str,
                             symbol: str) -> bool:
        try:
            value = (price * quantity).quantize(Decimal(10) ** -self.decimal_places)
            if value > self.limits.max_order_value_usd:
                raise ValidationError("Order value exceeds limit")
            # Daily volume and leverage checks can be added here
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Order value validation failed: %s", exc)
            raise

    def validate_symbol(self, symbol: str) -> str:
        if not self._symbol_regex.match(symbol):
            logger.error("Symbol format invalid: %s", symbol)
            raise ValidationError("Invalid symbol format")
        if symbol not in self.limits.symbol_limits:
            logger.error("Symbol not allowed: %s", symbol)
            raise ValidationError("Symbol not allowed")
        return symbol

    def validate_trading_parameters(self, params: Dict[str, Any]) -> Dict[str, Decimal]:
        symbol = self.validate_symbol(params["symbol"])
        price = self.validate_price(params["price"], symbol, params.get("market_price", Decimal("0")))
        quantity = self.validate_quantity(params["quantity"], symbol, params.get("portfolio_value", Decimal("0")))
        self.validate_order_value(price, quantity, params.get("side", "buy"), symbol)
        return {"symbol": symbol, "price": price, "quantity": quantity}

    def check_risk_limits(self, order_data: Dict[str, Any], current_positions: Dict[str, Decimal],
                          account_balance: Decimal) -> Dict[str, bool]:
        try:
            total_exposure = sum(current_positions.values()) + order_data["price"] * order_data["quantity"]
            breached = total_exposure > account_balance * self.limits.max_leverage
            return {"leverage_limit": not breached}
        except Exception as exc:  # noqa: BLE001
            logger.error("Risk limit check failed: %s", exc)
            raise
