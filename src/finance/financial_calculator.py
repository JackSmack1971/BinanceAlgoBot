from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, getcontext, ROUND_HALF_UP, ROUND_DOWN, ROUND_UP
from typing import Any, Dict, List, Optional, Union

getcontext().prec = 28

class FinancialCalculator:
    """Precision financial calculator for trading operations."""

    PRICE_PRECISION = 8
    QUANTITY_PRECISION = 8
    PERCENTAGE_PRECISION = 6
    USD_PRECISION = 2

    def __init__(self) -> None:
        self.commission_rates: Dict[str, Decimal] = {
            "maker": Decimal("0.001"),
            "taker": Decimal("0.001"),
        }
        self.tick_sizes: Dict[str, Decimal] = {
            "BTCUSDT": Decimal("0.01"),
            "ETHUSDT": Decimal("0.01"),
            "ADAUSDT": Decimal("0.0001"),
        }

    @staticmethod
    def to_decimal(value: Union[str, int, float, Decimal], precision: int = 8) -> Decimal:
        if isinstance(value, Decimal):
            return value.quantize(Decimal("0." + "0" * precision))
        decimal_value = Decimal(str(value))
        return decimal_value.quantize(Decimal("0." + "0" * precision))

    def calculate_order_value(self, price: Union[str, float, Decimal], quantity: Union[str, float, Decimal]) -> Decimal:
        precise_price = self.to_decimal(price, self.PRICE_PRECISION)
        precise_quantity = self.to_decimal(quantity, self.QUANTITY_PRECISION)
        order_value = precise_price * precise_quantity
        return order_value.quantize(Decimal("0." + "0" * self.USD_PRECISION))

    def calculate_commission(self, order_value: Decimal, commission_type: str = "taker") -> Decimal:
        commission_rate = self.commission_rates.get(commission_type, self.commission_rates["taker"])
        commission = order_value * commission_rate
        return commission.quantize(Decimal("0." + "0" * self.USD_PRECISION), rounding=ROUND_UP)

    def calculate_net_order_value(self, price: Decimal, quantity: Decimal, side: str, commission_type: str = "taker") -> Dict[str, Decimal]:
        gross_value = self.calculate_order_value(price, quantity)
        commission = self.calculate_commission(gross_value, commission_type)
        net_value = gross_value + commission if side.upper() == "BUY" else gross_value - commission
        return {"gross_value": gross_value, "commission": commission, "net_value": net_value}

    def calculate_profit_loss(self, entry_price: Decimal, exit_price: Decimal, quantity: Decimal, side: str) -> Dict[str, Decimal]:
        entry_price = self.to_decimal(entry_price, self.PRICE_PRECISION)
        exit_price = self.to_decimal(exit_price, self.PRICE_PRECISION)
        quantity = self.to_decimal(quantity, self.QUANTITY_PRECISION)
        price_diff = exit_price - entry_price if side.upper() == "BUY" else entry_price - exit_price
        absolute_pnl = price_diff * quantity
        if side.upper() == "BUY":
            percentage_pnl = (exit_price / entry_price - Decimal("1")) * Decimal("100")
        else:
            percentage_pnl = (entry_price / exit_price - Decimal("1")) * Decimal("100")
        entry_commission = self.calculate_commission(entry_price * quantity)
        exit_commission = self.calculate_commission(exit_price * quantity)
        total_commission = entry_commission + exit_commission
        net_pnl = absolute_pnl - total_commission
        return {
            "absolute_pnl": absolute_pnl.quantize(Decimal("0." + "0" * self.USD_PRECISION)),
            "percentage_pnl": percentage_pnl.quantize(Decimal("0." + "0" * self.PERCENTAGE_PRECISION)),
            "total_commission": total_commission,
            "net_pnl": net_pnl.quantize(Decimal("0." + "0" * self.USD_PRECISION)),
        }

    def calculate_portfolio_value(self, positions: Dict[str, Dict[str, Any]], current_prices: Dict[str, Decimal]) -> Dict[str, Any]:
        total_value = Decimal("0")
        total_unrealized_pnl = Decimal("0")
        total_commission_paid = Decimal("0")
        position_values: Dict[str, Dict[str, Decimal]] = {}
        for symbol, position in positions.items():
            current_price = self.to_decimal(current_prices.get(symbol, 0), self.PRICE_PRECISION)
            quantity = self.to_decimal(position["quantity"], self.QUANTITY_PRECISION)
            entry_price = self.to_decimal(position["entry_price"], self.PRICE_PRECISION)
            side = position["side"]
            market_value = self.calculate_order_value(current_price, quantity)
            pnl_data = self.calculate_profit_loss(entry_price, current_price, quantity, side)
            position_values[symbol] = {
                "market_value": market_value,
                "unrealized_pnl": pnl_data["net_pnl"],
                "percentage_pnl": pnl_data["percentage_pnl"],
            }
            total_value += market_value
            total_unrealized_pnl += pnl_data["net_pnl"]
            total_commission_paid += pnl_data["total_commission"]
        return {
            "total_portfolio_value": total_value,
            "total_unrealized_pnl": total_unrealized_pnl,
            "total_commission_paid": total_commission_paid,
            "position_values": position_values,
        }

    def calculate_position_size(self, account_balance: Decimal, risk_per_trade: Decimal, entry_price: Decimal, stop_loss_price: Decimal) -> Decimal:
        account_balance = self.to_decimal(account_balance, self.USD_PRECISION)
        risk_per_trade = self.to_decimal(risk_per_trade, self.PERCENTAGE_PRECISION)
        entry_price = self.to_decimal(entry_price, self.PRICE_PRECISION)
        stop_loss_price = self.to_decimal(stop_loss_price, self.PRICE_PRECISION)
        risk_amount = account_balance * (risk_per_trade / Decimal("100"))
        price_risk = abs(entry_price - stop_loss_price)
        if price_risk == Decimal("0"):
            raise ValueError("Stop loss price cannot equal entry price")
        position_size = risk_amount / price_risk
        return position_size.quantize(Decimal("0." + "0" * self.QUANTITY_PRECISION))

    def calculate_sharpe_ratio(self, returns: List[Decimal], risk_free_rate: Decimal = Decimal("0.02")) -> Decimal:
        if not returns:
            return Decimal("0")
        decimal_returns = [self.to_decimal(r, self.PERCENTAGE_PRECISION) for r in returns]
        mean_return = sum(decimal_returns) / Decimal(str(len(decimal_returns)))
        if len(decimal_returns) < 2:
            return Decimal("0")
        variance_sum = sum((r - mean_return) ** 2 for r in decimal_returns)
        variance = variance_sum / Decimal(str(len(decimal_returns) - 1))
        std_dev = variance.sqrt()
        if std_dev == Decimal("0"):
            return Decimal("0")
        excess_return = mean_return - risk_free_rate
        sharpe_ratio = excess_return / std_dev
        return sharpe_ratio.quantize(Decimal("0." + "0" * self.PERCENTAGE_PRECISION))

    def calculate_maximum_drawdown(self, portfolio_values: List[Decimal]) -> Dict[str, Decimal]:
        if len(portfolio_values) < 2:
            return {"max_drawdown": Decimal("0"), "max_drawdown_pct": Decimal("0")}
        values = [self.to_decimal(v, self.USD_PRECISION) for v in portfolio_values]
        peak = values[0]
        max_drawdown = Decimal("0")
        max_drawdown_pct = Decimal("0")
        for value in values[1:]:
            if value > peak:
                peak = value
            drawdown = peak - value
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                if peak > Decimal("0"):
                    max_drawdown_pct = (drawdown / peak) * Decimal("100")
        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown_pct.quantize(Decimal("0." + "0" * self.PERCENTAGE_PRECISION)),
        }

    def round_to_tick_size(self, price: Decimal, symbol: str) -> Decimal:
        tick_size = self.tick_sizes.get(symbol, Decimal("0.01"))
        return (price / tick_size).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * tick_size

    def validate_financial_precision(self, value: Decimal, max_precision: int) -> bool:
        sign, digits, exponent = value.as_tuple()
        decimal_places = max(0, -int(exponent))
        return decimal_places <= max_precision

    def format_currency(self, value: Decimal, currency: str = "USD") -> str:
        if currency == "USD":
            return f"${value:,.2f}"
        return f"{value:,.8f} {currency}"

    def format_percentage(self, value: Decimal) -> str:
        return f"{value:.2f}%"

class PrecisePosition:
    """Position class with precise decimal calculations."""

    def __init__(self, symbol: str, side: str, entry_price: Union[str, float, Decimal], quantity: Union[str, float, Decimal]) -> None:
        self.calculator = FinancialCalculator()
        self.symbol = symbol
        self.side = side
        self.entry_price = self.calculator.to_decimal(entry_price, self.calculator.PRICE_PRECISION)
        self.quantity = self.calculator.to_decimal(quantity, self.calculator.QUANTITY_PRECISION)
        self.current_price = self.entry_price
        self.realized_pnl = Decimal("0")

    def update_price(self, new_price: Union[str, float, Decimal]) -> None:
        self.current_price = self.calculator.to_decimal(new_price, self.calculator.PRICE_PRECISION)

    @property
    def market_value(self) -> Decimal:
        return self.calculator.calculate_order_value(self.current_price, self.quantity)

    @property
    def unrealized_pnl(self) -> Dict[str, Decimal]:
        return self.calculator.calculate_profit_loss(self.entry_price, self.current_price, self.quantity, self.side)

    def close_position(self, exit_price: Union[str, float, Decimal], close_quantity: Optional[Union[str, float, Decimal]] = None) -> Dict[str, Decimal]:
        exit_price_decimal = self.calculator.to_decimal(exit_price, self.calculator.PRICE_PRECISION)
        close_qty = self.calculator.to_decimal(close_quantity or self.quantity, self.calculator.QUANTITY_PRECISION)
        if close_qty > self.quantity:
            raise ValueError("Cannot close more than position size")
        pnl_data = self.calculator.calculate_profit_loss(self.entry_price, exit_price_decimal, close_qty, self.side)
        self.quantity -= close_qty
        self.realized_pnl += pnl_data["net_pnl"]
        return pnl_data

class PrecisePerformanceAnalyzer:
    """Performance analyzer using precise decimal calculations."""

    def __init__(self) -> None:
        self.calculator = FinancialCalculator()
        self.trades: List[Dict[str, Any]] = []
        self.daily_returns: List[Decimal] = []

    def add_trade(self, entry_price: Decimal, exit_price: Decimal, quantity: Decimal, side: str, timestamp: datetime) -> None:
        pnl_data = self.calculator.calculate_profit_loss(entry_price, exit_price, quantity, side)
        trade_data = {
            "timestamp": timestamp,
            "symbol": "BTCUSDT",
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "pnl": pnl_data["net_pnl"],
            "pnl_pct": pnl_data["percentage_pnl"],
            "commission": pnl_data["total_commission"],
        }
        self.trades.append(trade_data)

    def calculate_performance_metrics(self) -> Dict[str, Decimal]:
        if not self.trades:
            return {}
        returns = [trade["pnl_pct"] for trade in self.trades]
        total_trades = len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade["pnl"] > 0)
        losing_trades = total_trades - winning_trades
        win_rate = (Decimal(str(winning_trades)) / Decimal(str(total_trades))) * Decimal("100")
        total_pnl = sum(trade["pnl"] for trade in self.trades)
        total_commission = sum(trade["commission"] for trade in self.trades)
        avg_win = Decimal("0")
        avg_loss = Decimal("0")
        if winning_trades > 0:
            avg_win = sum(trade["pnl"] for trade in self.trades if trade["pnl"] > 0) / Decimal(str(winning_trades))
        if losing_trades > 0:
            avg_loss = abs(sum(trade["pnl"] for trade in self.trades if trade["pnl"] < 0)) / Decimal(str(losing_trades))
        profit_factor = avg_win / avg_loss if avg_loss > 0 else Decimal("0")
        sharpe_ratio = self.calculator.calculate_sharpe_ratio(returns)
        return {
            "total_trades": Decimal(str(total_trades)),
            "win_rate": win_rate.quantize(Decimal("0.01")),
            "total_pnl": total_pnl,
            "total_commission": total_commission,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor.quantize(Decimal("0.01")),
            "sharpe_ratio": sharpe_ratio,
        }

class PrecisionMigrationHelper:
    """Helper to migrate existing float-based calculations."""

    @staticmethod
    def migrate_strategy_calculations(strategy_file_path: str) -> str:
        migration_guide = f"""
MIGRATION CHECKLIST for {strategy_file_path}:
1. Replace all float arithmetic with Decimal:
   OLD: profit_pct = (exit_price / entry_price - 1)
   NEW: profit_pct = calculator.calculate_profit_loss(entry_price, exit_price, quantity, side)['percentage_pnl']
2. Use calculator methods for all financial operations:
   OLD: commission = order_value * 0.001
   NEW: commission = calculator.calculate_commission(order_value, 'taker')
3. Convert input parameters to Decimal:
   OLD: def calculate_position_size(self, price: float, balance: float)
   NEW: def calculate_position_size(self, price: Decimal, balance: Decimal)
4. Use proper rounding for display:
   OLD: f"P&L: {{pnl:.2f}}"
   NEW: f"P&L: {{calculator.format_currency(pnl)}}"
5. Validate precision in calculations:
   - Use calculator.validate_financial_precision() for critical values
   - Round to appropriate tick sizes before submitting orders
   - Maintain audit trail of precision in calculations
"""
        return migration_guide

