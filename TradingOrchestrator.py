import logging
import time
from typing import Dict, List, Optional
from binance.client import Client

from config import TRADING_CONFIG, RISK_CONFIG, LOGGING_CONFIG
from data_feed import DataFeed
from btcstrategy import BTCStrategy
from execution_engine import ExecutionEngine
from risk_management import RiskManagement

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
            
            # Initialize components for each symbol-interval pair
            for symbol in self.symbols:
                for interval in self.intervals:
                    # Create a unique key for each symbol-interval pair
                    key = f"{symbol}_{interval}"
                    
                    # Initialize components
                    strategy = BTCStrategy(self.client, symbol, interval)
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
    
    def stop_trading(self):
        """
        Stop the trading process.
        """
        logger.info("Stopping trading process")
        self.is_running = False
    
    def get_account_info(self) -> Optional[Dict]:
        """
        Get account information from Binance.
        
        Returns:
            Dict: Account information
        """
        if not self.client:
            logger.error("Client not initialized. Call initialize() first.")
            return None
        
        try:
            account_info = self.client.get_account()
            return account_info
        except Exception as e:
            logger.error(f"Error getting account information: {e}", exc_info=True)
            return None
    
    def get_account_balance(self, asset: str = 'USDT') -> Optional[float]:
        """
        Get account balance for a specific asset.
        
        Args:
            asset (str): Asset symbol. Defaults to 'USDT'.
            
        Returns:
            float: Asset balance
        """
        account_info = self.get_account_info()
        if not account_info:
            return None
        
        try:
            for balance in account_info['balances']:
                if balance['asset'] == asset:
                    return float(balance['free'])
            
            logger.warning(f"Asset {asset} not found in account")
            return 0.0
        except Exception as e:
            logger.error(f"Error getting account balance: {e}", exc_info=True)
            return None
    
    def add_symbol_interval(self, symbol: str, interval: str) -> bool:
        """
        Add a new symbol-interval pair to the trading orchestrator.
        
        Args:
            symbol (str): Trading symbol
            interval (str): Timeframe interval
            
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
            # Initialize components
            strategy = BTCStrategy(self.client, symbol, interval)
            execution_engine = ExecutionEngine(self.client, strategy, TRADING_CONFIG["default_quantity"])
            risk_management = RiskManagement(self.client, strategy, execution_engine, RISK_CONFIG["max_risk_per_trade"])
            
            # Store components in the dictionary
            self.components[key] = {
                "strategy": strategy,
                "execution_engine": execution_engine,
                "risk_management": risk_management
            }
            
            logger.info(f"Added {key} to the components")
            
            # Add symbol and interval to the lists if not already present
            if symbol not in self.symbols:
                self.symbols.append(symbol)
            if interval not in self.intervals:
                self.intervals.append(interval)
            
            return True
        except Exception as e:
            logger.error(f"Error adding {key}: {e}", exc_info=True)
            return False
    
    def remove_symbol_interval(self, symbol: str, interval: str) -> bool:
        """
        Remove a symbol-interval pair from the trading orchestrator.
        
        Args:
            symbol (str): Trading symbol
            interval (str): Timeframe interval
            
        Returns:
            bool: True if successful, False otherwise
        """
        key = f"{symbol}_{interval}"
        if key not in self.components:
            logger.warning(f"{key} not found in the components")
            return False
        
        try:
            # Remove the components
            del self.components[key]
            logger.info(f"Removed {key} from the components")
            
            # Check if the symbol and interval are used by other components
            symbol_used = False
            interval_used = False
            
            for k in self.components.keys():
                s, i = k.split('_')
                if s == symbol:
                    symbol_used = True
                if i == interval:
                    interval_used = True
            
            # Remove symbol and interval from the lists if not used by other components
            if not symbol_used and symbol in self.symbols:
                self.symbols.remove(symbol)
            if not interval_used and interval in self.intervals:
                self.intervals.remove(interval)
            
            return True
        except Exception as e:
            logger.error(f"Error removing {key}: {e}", exc_info=True)
            return False
    
    def backtest(self, symbol: str, interval: str, start_date: str, end_date: str) -> Optional[Dict]:
        """
        Backtest the strategy for a specific symbol and interval.
        
        Args:
            symbol (str): Trading symbol
            interval (str): Timeframe interval
            start_date (str): Start date for backtesting (format: 'YYYY-MM-DD')
            end_date (str): End date for backtesting (format: 'YYYY-MM-DD')
            
        Returns:
            Dict: Backtesting results
        """
        if not self.client:
            logger.error("Client not initialized. Call initialize() first.")
            return None
        
        try:
            # Create a temporary strategy for backtesting
            strategy = BTCStrategy(self.client, symbol, interval)
            
            # Get historical data
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_date,
                end_str=end_date
            )
            
            import pandas as pd
            data = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignored'
            ])
            data = data.astype(float)
            
            # Calculate indicators
            data = strategy.indicators.calculate_all(data)
            
            # Generate signals
            for i in range(1, len(data)):
                if data['close'].iloc[i] > data['ema'].iloc[i] and data['rsi'].iloc[i] > 50:
                    data.loc[data.index[i], 'signal'] = 1.0  # Buy signal
                elif data['close'].iloc[i] < data['ema'].iloc[i] and data['rsi'].iloc[i] < 50:
                    data.loc[data.index[i], 'signal'] = -1.0  # Sell signal
                else:
                    data.loc[data.index[i], 'signal'] = 0.0  # No signal
            
            # Calculate returns
            data['position'] = data['signal'].shift(1).fillna(0)
            data['returns'] = data['close'].pct_change() * data['position']
            
            # Calculate metrics
            total_return = data['returns'].sum()
            sharpe_ratio = data['returns'].mean() / data['returns'].std() * (252 ** 0.5)  # Annualized Sharpe ratio
            max_drawdown = (data['returns'].cumsum() - data['returns'].cumsum().cummax()).min()
            
            # Summarize results
            results = {
                'symbol': symbol,
                'interval': interval,
                'start_date': start_date,
                'end_date': end_date,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'num_trades': (data['signal'] != 0).sum(),
                'data': data
            }
            
            logger.info(f"Backtesting completed for {symbol}_{interval}")
            return results
            
        except Exception as e:
            logger.error(f"Error during backtesting: {e}", exc_info=True)
            return None


if __name__ == "__main__":
    import os
    import logging.config
    
    # Configure logging
    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'standard': {
                'format': LOGGING_CONFIG['format']
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': LOGGING_CONFIG['level'],
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': LOGGING_CONFIG['level'],
                'formatter': 'standard',
                'filename': LOGGING_CONFIG['log_file'],
                'mode': 'a',
            }
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['console', 'file'] if LOGGING_CONFIG['log_to_file'] else ['console'],
                'level': LOGGING_CONFIG['level'],
                'propagate': True
            }
        }
    })
    
    # Get API credentials from environment variables
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    if not api_key or not api_secret:
        logger.error("API credentials not found in environment variables")
        exit(1)
    
    # Initialize the trading orchestrator
    orchestrator = TradingOrchestrator(api_key, api_secret)
    
    if orchestrator.initialize():
        try:
            # Print account balance
            balance = orchestrator.get_account_balance()
            logger.info(f"Account balance: {balance} USDT")
            
            # Start trading
            orchestrator.start_trading()
            
        except KeyboardInterrupt:
            logger.info("Trading stopped by user")
            orchestrator.stop_trading()
        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)
            orchestrator.stop_trading()
    else:
        logger.error("Failed to initialize trading orchestrator")
