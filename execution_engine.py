# execution_engine.py
import logging
from typing import Optional
import pandas as pd
from strategies.base_strategy import BaseStrategy
from exchange_interface import ExchangeInterface
from position_manager import PositionManager
from decimal import Decimal
from exceptions import BaseTradingException, ExchangeError, TradeExecutionError

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
            logger.info(f"Executing buy order for {self.strategy.symbol}, quantity={self.quantity}")

            # Check if we're in a test environment
            from config import get_config
            if get_config('use_testnet'):
                # In testnet, just log the action
                logger.info(f"TEST MODE: Would place buy order for {self.strategy.symbol}, quantity={self.quantity}")
                return

            # Execute the buy order
            order = await self.exchange_interface.place_order(
                symbol=self.strategy.symbol,
                side="buy",
                quantity=self.quantity,
                order_type="market"
            )

            logger.info(f"Buy order executed: {order}")

            # Open position in PositionManager
            await self.position_manager.open_position(
                symbol=self.strategy.symbol,
                side="BUY",
                entry_price=Decimal("0"),  # Replace with actual price
                quantity=Decimal(str(self.quantity)),
                risk_params={},
            )

        except Exception as exc:
            logger.error("Error during buy execution: %s", exc, exc_info=True)
            raise TradeExecutionError(
                f"Could not execute buy order for symbol {self.strategy.symbol} and quantity {self.quantity}: {exc}"
            ) from exc

        # Store trade history in the database
        try:
            from database.trade_history_repository import TradeHistoryRepository
            trade_history_repo = TradeHistoryRepository()
            trade_history_repo.insert_trade_history(
                strategy_id=self.strategy.id,  # Assuming strategy object has an 'id' attribute
                entry_time=pd.Timestamp.now(),
                exit_time=pd.Timestamp.now(),
                position_type="buy",
                entry_price=0.0,  # Replace with actual entry price
                exit_price=0.0,  # Replace with actual exit price
                profit_pct=0.0,  # Replace with actual profit percentage
                duration=0.0,  # Replace with actual duration
                commission_fee=0.0  # Replace with actual commission fee
            )
        except Exception as exc:
            logger.error("Error storing trade history: %s", exc, exc_info=True)

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
            logger.info(f"Executing sell order for {self.strategy.symbol}, quantity={self.quantity}")

            # Check if we're in a test environment
            from config import get_config
            if get_config('use_testnet'):
                # In testnet, just log the action
                logger.info(f"TEST MODE: Would place sell order for {self.strategy.symbol}, quantity={self.quantity}")
                return

            # Execute the sell order
            order = await self.exchange_interface.place_order(
                symbol=self.strategy.symbol,
                side="sell",
                quantity=self.quantity,
                order_type="market"
            )

            logger.info(f"Sell order executed: {order}")

            # Close position in PositionManager
            await self.position_manager.close_position(
                symbol=self.strategy.symbol,
                exit_price=Decimal("0")  # Replace with actual price
            )

        except Exception as exc:
            logger.error("Error during sell execution: %s", exc, exc_info=True)
            raise TradeExecutionError(
                f"Could not execute sell order for symbol {self.strategy.symbol} and quantity {self.quantity}: {exc}"
            ) from exc

        # Store trade history in the database
        try:
            from database.trade_history_repository import TradeHistoryRepository
            trade_history_repo = TradeHistoryRepository()
            trade_history_repo.insert_trade_history(
                strategy_id=self.strategy.id,  # Assuming strategy object has an 'id' attribute
                entry_time=pd.Timestamp.now(),
                exit_time=pd.Timestamp.now(),
                position_type="sell",
                entry_price=0.0,  # Replace with actual entry price
                exit_price=0.0,  # Replace with actual exit price
                profit_pct=0.0,  # Replace with actual profit percentage
                duration=0.0,  # Replace with actual duration
                commission_fee=0.0  # Replace with actual commission fee
            )
        except Exception as exc:
            logger.error("Error storing trade history: %s", exc, exc_info=True)
