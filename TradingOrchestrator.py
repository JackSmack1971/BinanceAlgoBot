# TradingOrchestrator.py
import logging
import asyncio
import time
from typing import Dict, List, Optional
from binance.client import Client
from configuration_service import ConfigurationService
from BinanceExchangeInterface import BinanceExchangeInterface

from config import get_config
from execution_engine import ExecutionEngine
from risk_management import RiskManagement
from strategy_factory import StrategyFactory, Strategy
from signal_manager import SignalManager

logger = logging.getLogger(__name__)

class TradingOrchestrator:
    """
    Main service orchestrator that coordinates the interaction between
    strategy, execution, and risk management components.
    """
    
    def __init__(self, config_service: ConfigurationService, initial_balance: float, risk_per_trade: float, symbols: List[str] = None, intervals: List[str] = None):
        """
        Initialize the trading orchestrator.

        Args:
            config_service (ConfigurationService): Configuration service
            initial_balance (float): Initial account balance.
            risk_per_trade (float): Risk per trade.
            symbols (List[str], optional): List of trading symbols. Defaults to config value if None.
            intervals (List[str], optional): List of timeframe intervals. Defaults to config value if None.
        """
        self.config_service = config_service
        
        # Use default symbol if none provided
        if symbols is None:
            symbols = [self.config_service.get_config('default_symbol')]
        self.symbols = symbols
        
        # Use default interval if none provided
        if intervals is None:
            intervals = [self.config_service.get_config('default_interval')]
        self.intervals = intervals
        
        # Initialize the client
        self.client = None

        # Initialize PositionManager
        self.position_manager = PositionManager(initial_balance=initial_balance, risk_per_trade=risk_per_trade)
        
        # Dictionary to store strategy, execution, and risk management components for each symbol-interval pair
        self.components: Dict[str, Dict] = {}
        
        # Initialize running state
        self.is_running = False

        # Initialize SignalManager
        self.signal_manager = SignalManager()
        
        logger.info(f"Initialized TradingOrchestrator with symbols={self.symbols}, intervals={self.intervals}")
    
    def initialize(self):
        """
        Initialize the Binance client and all trading components.
        """
        try:
            # Initialize Binance client
            use_testnet = self.config_service.get_config('use_testnet')
            api_key = self.config_service.get_config('api_key')
            api_secret = self.config_service.get_config('secret_key')
            if use_testnet:
                self.client = Client(api_key, api_secret, testnet=True)
                logger.info("Using Binance testnet")
            else:
                self.client = Client(api_key, api_secret)
                logger.info("Using Binance production API")
        
            # Get the default strategy type from config
            default_strategy_type = self.config_service.get_config("default_strategy_type")

            # Get initial balance and risk per trade from config
            initial_balance = float(self.config_service.get_config("initial_balance"))
            risk_per_trade = float(self.config_service.get_config("risk_per_trade"))
        
            # Initialize components for each symbol-interval pair
            for symbol in self.symbols:
                for interval in self.intervals:
                    # Create a unique key for each symbol-interval pair
                    key = f"{symbol}_{interval}"
                
                    # Create strategy using the factory
                    strategy = StrategyFactory.create_strategy(
                        default_strategy_type,
                        self.client,
                        symbol,
                        interval
                    )
                
                    if strategy is None:
                        logger.error(f"Failed to create strategy for {key}")
                        continue
                
                    # Initialize components
                    engine = ExecutionEngine(self.client, strategy, self.config_service.get_config("default_quantity"), self.position_manager)
                    risk_management = RiskManagement(self.client, strategy, engine, self.config_service.get_config("max_risk_per_trade"))

                    # Store components in a dictionary
                    self.components[key] = {
                        "strategy": strategy,
                        "execution_engine": engine,
                        "risk_management": risk_management
                    }

                
                    logger.info(f"Initialized components for {key}")
        
            logger.info("All components initialized successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing components: {e}", exc_info=True)
            raise BaseTradingException(f"Error initializing components: {e}") from e
            return False
    
    async def start_trading(self):
        """
        Start the trading process for all configured symbols and intervals.
        """
        if not self.client:
            logger.error("Client not initialized. Call initialize() first.")
            return False

        try:
            self.is_running = True
            logger.info("Starting trading process")

            while self.is_running:
                for key, components in self.components.items():
                    await self._process_component(key, components)

                # Sleep for a configured time before the next iteration
                await asyncio.sleep(60)  # Check for new signals every minute

            logger.info("Trading process stopped")
            return True

        except Exception as e:
            logger.error(f"Error in trading process: {e}", exc_info=True)
            raise BaseTradingException(f"Error in trading process: {e}") from e
            self.is_running = False
            return False

    async def _process_component(self, key: str, components: Dict):
        """
        Process a single component (strategy, execution engine, risk management) for a given symbol-interval pair.
        """
        try:
            # Get components
            strategy = components["strategy"]
            execution_engine = components["execution_engine"]
            risk_management = components["risk_management"]

            # Step 1: Update risk parameters
            logger.info(f"Managing risk for {key}")
            risk_management.manage_risk()

            # Step 2: Get trading signals
            logger.info(f"Generating signals for {key}")
            signals = strategy.run()

            if signals is not None and not signals.empty:
                # Step 3: Process signals with SignalManager
                logger.info(f"Processing signals for {key}")
                processed_signals = self.signal_manager.process_signal(signals)

                # Step 4: Execute trades based on processed signals
                logger.info(f"Executing trades for {key}")
                await execution_engine.execute_trades(processed_signals)
            else:
                logger.warning(f"No signals generated for {key}")

        except Exception as e:
            logger.error(f"Error processing {key}: {e}", exc_info=True)
            raise BaseTradingException(f"Error processing {key}: {e}") from e
            # Continue with the next symbol-interval pair
            pass

    
    def add_strategy(self, symbol: str, interval: str, strategy_type: str, **kwargs) -> bool:
        """
        Add a new strategy for a symbol-interval pair.
        
        Args:
            symbol (str): Trading symbol
            interval (str): Timeframe interval
            strategy_type (str): Strategy type identifier
            **kwargs: Additional parameters for the strategy
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            logger.error("Client not initialized. Call initialize() first.")
            return False
        
        key = f"{symbol}_{interval}"
        if key in self.components:
            logger.warning(f"{key} already exists in the components")
            return False
        
        try:
            # Create strategy using the factory
            strategy = StrategyFactory.create_strategy(
                strategy_type, 
                self.client, 
                symbol, 
                interval, 
                **kwargs
            )
            
            if strategy is None:
                logger.error(f"Failed to create strategy for {key}")
                return False
        
            # Initialize components
            engine = ExecutionEngine(self.client, strategy, get_config("default_quantity"))
            risk_management = RiskManagement(self.client, strategy, engine, get_config("max_risk_per_trade"))

            # Store components in the dictionary
            self.components[key] = {
                "strategy": strategy,
                "execution_engine": engine,
                "risk_management": risk_management
            }
            
            logger.info(f"Added {strategy_type} strategy for {key}")
            
            # Add symbol and interval to the lists if not already present
            if symbol not in self.symbols:
                self.symbols.append(symbol)
            if interval not in self.intervals:
                self.intervals.append(interval)
            
            return True
        except Exception as e:
            logger.error(f"Error adding strategy: {e}", exc_info=True)
            raise StrategyError(f"Error adding strategy: {e}") from e
            return False
    
    def change_strategy(self, symbol: str, interval: str, strategy_type: str, **kwargs) -> bool:
        """
        Change the strategy for an existing symbol-interval pair.
        
        Args:
            symbol (str): Trading symbol
            interval (str): Timeframe interval
            strategy_type (str): Strategy type identifier
            **kwargs: Additional parameters for the strategy
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            logger.error("Client not initialized. Call initialize() first.")
            return False
        
        key = f"{symbol}_{interval}"
        if key not in self.components:
            logger.warning(f"{key} not found in the components")
            return False
        
        try:
            # Create the new strategy
            strategy = StrategyFactory.create_strategy(
                strategy_type, 
                self.client, 
                symbol, 
                interval, 
                **kwargs
            )
            
            if strategy is None:
                logger.error(f"Failed to create strategy for {key}")
                return False
            
            # Get existing components
            execution_engine = self.components[key]["execution_engine"]
            risk_management = self.components[key]["risk_management"]
            
            # Update the strategy
            execution_engine.strategy = strategy
            risk_management.strategy = strategy
            
            # Update the components dictionary
            self.components[key]["strategy"] = strategy
            
            logger.info(f"Changed strategy for {key} to {strategy_type}")
            return True
        except Exception as e:
            logger.error(f"Error changing strategy: {e}", exc_info=True)
            raise StrategyError(f"Error changing strategy: {e}") from e
            return False
            
    def get_account_balance(self):
        """
        Get the current account balance.
        
        Returns:
            float: Available balance in USDT or None if there was an error
        """
        if not self.client:
            logger.error("Client not initialized. Call initialize() first.")
            return None
            
        try:
            account_info = self.client.get_account()
            balances = {asset['asset']: float(asset['free']) for asset in account_info['balances'] 
                       if float(asset['free']) > 0}
            
            # Return USDT balance as the default
            return balances.get('USDT', 0.0)
            
        except Exception as e:
            logger.error(f"Error getting account balance: {e}", exc_info=True)
            raise ExchangeError(f"Error getting account balance: {e}") from e
            return None
            
    def stop_trading(self):
        """
        Stop the trading process gracefully.
        """
        if not self.is_running:
            logger.warning("Trading is not running")
            return
            
        try:
            logger.info("Stopping trading process")
            self.is_running = False
            
            # Cancel any open orders
            for key, components in self.components.items():
                try:
                    # Get the symbol from the key
                    symbol = key.split('_')[0]
                    
                    # Cancel open orders for the symbol
                    open_orders = self.client.get_open_orders(symbol=symbol)
                    
                    if open_orders:
                        logger.info(f"Cancelling {len(open_orders)} open orders for {symbol}")
                        
                        for order in open_orders:
                            self.client.cancel_order(
                                symbol=symbol,
                                orderId=order['orderId']
                            )
                            logger.info(f"Cancelled order {order['orderId']} for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error cancelling orders for {key}: {e}", exc_info=True)
                    raise ExchangeError(f"Error cancelling orders for {key}: {e}") from e
            
            logger.info("Trading stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping trading: {e}", exc_info=True)
            raise BaseTradingException(f"Error stopping trading: {e}") from e
