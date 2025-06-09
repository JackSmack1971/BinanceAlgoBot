# execution_engine.py
import logging
from typing import Optional
import pandas as pd
from strategies.base_strategy import BaseStrategy
from exchange_interface import ExchangeInterface
from position_manager import PositionManager
from decimal import Decimal
from exceptions import BaseTradingException, ExchangeError, TradeExecutionError
from src.execution import RobustExecutionEngine, OrderType

logger = logging.getLogger(__name__)

from utils import handle_error
from execution_engine_interface import ExecutionEngineInterface

class ExecutionEngine(ExecutionEngineInterface):
    @handle_error
    def __init__(self, exchange_interface: ExchangeInterface, strategy: BaseStrategy, quantity: float, position_manager: PositionManager):
        """
        Initialize the execution engine.

        Args:
            exchange_interface (ExchangeInterface): Exchange interface
            strategy (BaseStrategy): Trading strategy to execute
            quantity (float): Trade quantity
            position_manager (PositionManager): Position manager
        """
        self.exchange_interface = exchange_interface
        self.strategy = strategy
        self.quantity = quantity
        self.position_manager = position_manager
        self._robust_engine = RobustExecutionEngine(
            exchange_interface,
            position_manager,
            position_manager.risk_manager,
            position_manager.db,
        )
        self._engine_started = False

        logger.info(f"Initialized ExecutionEngine for {strategy.symbol} with quantity={quantity}")

    async def execute_trades(self, signals=None):
        """
        Execute trades based on signals from the strategy.

        Args:
            signals (pd.DataFrame, optional): DataFrame containing signals.
                                             If None, signals will be generated.
        """
        try:
            if signals is None:
                logger.info(f"Generating signals for {self.strategy.symbol}")
                signals = self.strategy.run()

            if signals is None or signals.empty:
                logger.warning("No signals to execute trades")
                return

            # Get the latest signal
            latest_signal = signals['signal'].iloc[-1]
            previous_signal = signals['signal'].iloc[-2] if len(signals) > 1 else 0

            # Execute trade only if the signal changed
            if latest_signal != previous_signal:
                # If the latest signal is 1.0, execute a buy order
                if latest_signal == 1.0:
                    await self._execute_buy()
                # If the latest signal is -1.0, execute a sell order
                elif latest_signal == -1.0:
                    await self._execute_sell()

        except Exception as exc:
            logger.error("Error during trade execution: %s", exc, exc_info=True)
            raise TradeExecutionError(
                f"Could not execute trades for symbol {self.strategy.symbol}: {exc}"
            ) from exc

    async def _execute_buy(self):
        """Execute a buy order."""
        try:
            logger.info(
                f"Executing buy order for {self.strategy.symbol}, quantity={self.quantity}"
            )
            from config import get_config
            if get_config("use_testnet"):
                logger.info(
                    f"TEST MODE: Would place buy order for {self.strategy.symbol}, quantity={self.quantity}"
                )
                return
            if not self._engine_started:
                await self._robust_engine.start()
                self._engine_started = True
            await self._robust_engine.execute_order(
                {
                    "symbol": self.strategy.symbol,
                    "side": "BUY",
                    "quantity": Decimal(str(self.quantity)),
                    "order_type": OrderType.MARKET.value,
                }
            )
        except Exception as exc:
            logger.error("Error during buy execution: %s", exc, exc_info=True)
            raise TradeExecutionError(
                f"Could not execute buy order for symbol {self.strategy.symbol} and quantity {self.quantity}: {exc}"
            ) from exc

    async def cancel_all_orders(self) -> None:
        """Cancel all open orders for the strategy symbol."""
        try:
            await self.exchange_interface.cancel_all_orders(self.strategy.symbol)
            logger.info("Cancelled all orders for %s", self.strategy.symbol)
        except Exception as exc:
            logger.error("Failed to cancel orders: %s", exc, exc_info=True)
            raise TradeExecutionError("Cancel orders failed") from exc

    async def close_positions(self, order_by_loss: bool = False) -> None:
        """Close open positions based on current risk."""
        try:
            position = await self.position_manager.get_position(self.strategy.symbol)
            if position:
                await self.position_manager.close_position(position.symbol, Decimal("0"))
                logger.info("Closed position for %s", position["symbol"])
        except Exception as exc:
            logger.error("Failed to close positions: %s", exc, exc_info=True)
            raise TradeExecutionError("Close positions failed") from exc

    async def _execute_sell(self):
        """Execute a sell order."""
        try:
            logger.info(
                f"Executing sell order for {self.strategy.symbol}, quantity={self.quantity}"
            )
            from config import get_config
            if get_config("use_testnet"):
                logger.info(
                    f"TEST MODE: Would place sell order for {self.strategy.symbol}, quantity={self.quantity}"
                )
                return
            if not self._engine_started:
                await self._robust_engine.start()
                self._engine_started = True
            await self._robust_engine.execute_order(
                {
                    "symbol": self.strategy.symbol,
                    "side": "SELL",
                    "quantity": Decimal(str(self.quantity)),
                    "order_type": OrderType.MARKET.value,
                }
            )
        except Exception as exc:
            logger.error("Error during sell execution: %s", exc, exc_info=True)
            raise TradeExecutionError(
                f"Could not execute sell order for symbol {self.strategy.symbol} and quantity {self.quantity}: {exc}"
            ) from exc
