# execution_engine.py
import logging
from typing import Optional
import pandas as pd
from strategy_factory import Strategy  # Fixed import to use the abstract Strategy class
from exchange_interface import ExchangeInterface

logger = logging.getLogger(__name__)

class TradeExecutionError(Exception):
    """Exception raised when there is an error executing a trade."""
    pass

class ExecutionEngine:
    def __init__(self, exchange_interface: ExchangeInterface, strategy: Strategy, quantity: float):
        """
        Initialize the execution engine.
        
        Args:
            exchange_interface (ExchangeInterface): Exchange interface
            strategy (Strategy): Trading strategy to execute
            quantity (float): Trade quantity
        """
        self.exchange_interface = exchange_interface
        self.strategy = strategy
        self.quantity = quantity
        
        logger.info(f"Initialized ExecutionEngine for {strategy.symbol} with quantity={quantity}")

    def execute_trades(self, signals=None):
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
                    self._execute_buy()
                # If the latest signal is -1.0, execute a sell order
                elif latest_signal == -1.0:
                    self._execute_sell()
                    
        except Exception as e:
            logger.error(f"An error occurred during trade execution: {e}", exc_info=True)
            raise TradeExecutionError(f"Could not execute trades for symbol {self.strategy.symbol}") from e

    def _execute_buy(self):
        """Execute a buy order."""
        try:
            logger.info(f"Executing buy order for {self.strategy.symbol}, quantity={self.quantity}")
            
            # Check if we're in a test environment
            from config import get_config
            if get_config('use_testnet', True):
                # In testnet, just log the action
                logger.info(f"TEST MODE: Would place buy order for {self.strategy.symbol}, quantity={self.quantity}")
                return
            
            # Execute the buy order
            order = self.exchange_interface.place_order(
                symbol=self.strategy.symbol,
                side="buy",
                quantity=self.quantity,
                order_type="market"
            )
            
            logger.info(f"Buy order executed: {order}")
            
        except Exception as e:
            logger.error(f"An error occurred during buy execution: {e}", exc_info=True)
            raise TradeExecutionError(f"Could not execute buy order for symbol {self.strategy.symbol} and quantity {self.quantity}") from e

        # Store trade history in the database
        from database.trade_history_repository import TradeHistoryRepository
        trade_history_repo = TradeHistoryRepository()
        trade_history_repo.insert_trade_history(
            strategy_id=self.strategy.id,  # Assuming strategy object has an 'id' attribute
            entry_time=pd.Timestamp.now(),  # Replace with actual entry time
            exit_time=pd.Timestamp.now(),  # Replace with actual exit time
            position_type="buy",  # Replace with actual position type
            entry_price=0.0,  # Replace with actual entry price
            exit_price=0.0,  # Replace with actual exit price
            profit_pct=0.0,  # Replace with actual profit percentage
            duration=0.0,  # Replace with actual duration
            commission_fee=0.0  # Replace with actual commission fee
        )

    def _execute_sell(self):
        """Execute a sell order."""
        try:
            logger.info(f"Executing sell order for {self.strategy.symbol}, quantity={self.quantity}")
            
            # Check if we're in a test environment
            from config import get_config
            if get_config('use_testnet', True):
                # In testnet, just log the action
                logger.info(f"TEST MODE: Would place sell order for {self.strategy.symbol}, quantity={self.quantity}")
                return
            
            # Execute the sell order
            order = self.exchange_interface.place_order(
                symbol=self.strategy.symbol,
                side="sell",
                quantity=self.quantity,
                order_type="market"
            )
            
            logger.info(f"Sell order executed: {order}")
            
        except Exception as e:
            logger.error(f"An error occurred during sell execution: {e}", exc_info=True)
            raise TradeExecutionError(f"Could not execute sell order for symbol {self.strategy.symbol} and quantity {self.quantity}") from e

        # Store trade history in the database
        from database.trade_history_repository import TradeHistoryRepository
        trade_history_repo = TradeHistoryRepository()
        trade_history_repo.insert_trade_history(
            strategy_id=self.strategy.id,  # Assuming strategy object has an 'id' attribute
            entry_time=pd.Timestamp.now(),  # Replace with actual entry time
            exit_time=pd.Timestamp.now(),  # Replace with actual exit time
            position_type="sell",  # Replace with actual position type
            entry_price=0.0,  # Replace with actual entry price
            exit_price=0.0,  # Replace with actual exit price
            profit_pct=0.0,  # Replace with actual profit percentage
            duration=0.0,  # Replace with actual duration
            commission_fee=0.0  # Replace with actual commission fee
        )
