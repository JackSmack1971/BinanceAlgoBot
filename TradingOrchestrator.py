import logging
import time
from typing import Dict, List, Optional
from binance.client import Client

from config import TRADING_CONFIG, RISK_CONFIG, LOGGING_CONFIG
from execution_engine import ExecutionEngine
from risk_management import RiskManagement
from strategy_factory import StrategyFactory, Strategy

logger = logging.getLogger(__name__)

class TradingOrchestrator:
    """
    Main service orchestrator that coordinates the interaction between
    strategy, execution, and risk management components.
    """
    
    def __init__(self, api_key: str, api_secret: str, symbols: List[str] = None, intervals: List[str] = None):
        """
        Initialize the trading orchestrator.
        
        Args:
            api_key (str): Binance API key
            api_secret (str): Binance API secret
            symbols (List[str], optional): List of trading symbols. Defaults to config value if None.
            intervals (List[str], optional): List of timeframe intervals. Defaults to config value if None.
        """
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Use default symbol if none provided
        if symbols is None:
            symbols = [TRADING_CONFIG["default_symbol"]]
        self.symbols = symbols
        
        # Use default interval if none provided
        if intervals is None:
            from config import BINANCE_CONSTANTS
            intervals = [BINANCE_CONSTANTS["KLINE_INTERVAL_15MINUTE"]]
        self.intervals = intervals
        
        # Initialize the client
        self.client = None
        
        # Dictionary to store strategy, execution, and risk management components for each symbol-interval pair
        self.components: Dict[str, Dict] = {}
        
        # Initialize running state
        self.is_running = False
        
        logger.info(f"Initialized TradingOrchestrator with symbols={self.symbols}, intervals={self.intervals}")
    
    def initialize(self):
        """
        Initialize the Binance client and all trading components.
        """
        try:
            # Initialize Binance client
            from config import API_CONFIG
            if API_CONFIG["use_testnet"]:
                self.client = Client(self.api_key, self.api_secret, testnet=True)
                logger.info("Using Binance testnet")
            else:
                self.client = Client(self.api_key, self.api_secret)
                logger.info("Using Binance production API")
            
            # Get the default strategy type from config
            from config import TRADING_CONFIG
            default_strategy_type = TRADING_CONFIG.get("default_strategy_type", "btc")
            
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
                    execution_engine = ExecutionEngine(self.client, strategy, TRADING_CONFIG["default_quantity"])
                    risk_management = RiskManagement(self.client, strategy, execution_engine, RISK_CONFIG["max_risk_per_trade"])
                    
                    # Store components in a dictionary
                    self.components[key] = {
                        "strategy": strategy,
                        "execution_engine": execution_engine,
                        "risk_management": risk_management
                    }
                    
                    logger.info(f"Initialized components for {key}")
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}", exc_info=True)
            return False
    
    def start_trading(self):
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
                            # Step 3: Execute trades based on signals
                            logger.info(f"Executing trades for {key}")
                            execution_engine.execute_trades(signals)
                        else:
                            logger.warning(f"No signals generated for {key}")
                    
                    except Exception as e:
                        logger.error(f"Error processing {key}: {e}", exc_info=True)
                        # Continue with the next symbol-interval pair
                        continue
                
                # Sleep for a configured time before the next iteration
                time.sleep(60)  # Check for new signals every minute
                
            logger.info("Trading process stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error in trading process: {e}", exc_info=True)
            self.is_running = False
            return False
    
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
            execution_engine = ExecutionEngine(self.client, strategy, TRADING_CONFIG["default_quantity"])
            risk_management = RiskManagement(self.client, strategy, execution_engine, RISK_CONFIG["max_risk_per_trade"])
            
            # Store components in the dictionary
            self.components[key] = {
                "strategy": strategy,
                "execution_engine": execution_engine,
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
            return False
