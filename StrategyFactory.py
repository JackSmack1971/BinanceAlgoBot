import logging
from typing import Dict, Any, Type, Optional
from binance.client import Client
from abc import ABC, abstractmethod

from data_feed import DataFeed
from indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    
    def __init__(self, client: Client, symbol: str, interval: str):
        """
        Initialize the strategy.
        
        Args:
            client (Client): Binance API client
            symbol (str): Trading symbol
            interval (str): Timeframe interval
        """
        self.client = client
        self.symbol = symbol
        self.interval = interval
        self.data_feed = DataFeed(client, symbol, interval)
        
        logger.info(f"Initialized {self.__class__.__name__} for {symbol} on {interval}")
    
    @abstractmethod
    def calculate_indicators(self):
        """
        Calculate indicators required for the strategy.
        
        Returns:
            pd.DataFrame: DataFrame with calculated indicators
        """
        pass
    
    @abstractmethod
    def run(self):
        """
        Generate trading signals based on the strategy.
        
        Returns:
            pd.DataFrame: DataFrame with trading signals
        """
        pass


class BTCStrategy(Strategy):
    """
    Bitcoin trading strategy based on EMA, RSI, ATR, and VWAP indicators.
    """
    
    def __init__(self, client: Client, symbol: str, interval: str, 
                 ema_window: int = None, rsi_window: int = None, 
                 atr_window: int = None, vwap_window: int = None):
        """
        Initialize the BTC trading strategy.
        
        Args:
            client (Client): Binance API client
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            interval (str): Timeframe interval (e.g., '15m')
            ema_window (int, optional): EMA window. Defaults to config value if None.
            rsi_window (int, optional): RSI window. Defaults to config value if None.
            atr_window (int, optional): ATR window. Defaults to config value if None.
            vwap_window (int, optional): VWAP window. Defaults to config value if None.
        """
        super().__init__(client, symbol, interval)
        
        from config import INDICATOR_CONFIG
        
        # Use provided parameters or defaults from config
        self.ema_window = ema_window if ema_window else INDICATOR_CONFIG["ema_window"]
        self.rsi_window = rsi_window if rsi_window else INDICATOR_CONFIG["rsi_window"]
        self.atr_window = atr_window if atr_window else INDICATOR_CONFIG["atr_window"]
        self.vwap_window = vwap_window if vwap_window else INDICATOR_CONFIG["vwap_window"]
        
        # Initialize indicators
        self.indicators = TechnicalIndicators(
            ema_window=self.ema_window,
            rsi_window=self.rsi_window,
            atr_window=self.atr_window,
            vwap_window=self.vwap_window
        )
    
    def calculate_indicators(self):
        """
        Retrieve data and calculate all indicators.
        
        Returns:
            pd.DataFrame: DataFrame with all indicators calculated
        """
        try:
            data = self.data_feed.get_data()
            if data is not None:
                data = self.indicators.calculate_all(data)
            return data
        except Exception as e:
            logger.error(f"An error occurred while calculating indicators: {e}", exc_info=True)
            return None

    def run(self):
        """
        Generate trading signals based on calculated indicators.
        
        Returns:
            pd.DataFrame: DataFrame with calculated indicators and generated signals.
        """
        try:
            # Get data with calculated indicators
            data = self.calculate_indicators()
            
            if data is None or data.empty:
                logger.warning(f"No data available to generate trading signals for {self.symbol}")
                return None
            
            # Initialize signal column with 0.0 (no signal/hold)
            data['signal'] = 0.0
            
            # Get configuration values
            from config import STRATEGY_CONFIG
            rsi_buy = STRATEGY_CONFIG["rsi_buy_threshold"]
            rsi_sell = STRATEGY_CONFIG["rsi_sell_threshold"]
            volatility_increase = STRATEGY_CONFIG["volatility_increase_factor"]
            volatility_decrease = STRATEGY_CONFIG["volatility_decrease_factor"]
            rsi_oversold = INDICATOR_CONFIG["rsi_oversold"]
            
            # Trading rules based on indicators
            for i in range(1, len(data)):
                # Rule 1: Price crosses above VWAP and RSI is above threshold -> Buy signal
                if (data['close'].iloc[i] > data['vwap'].iloc[i] and 
                    data['close'].iloc[i-1] <= data['vwap'].iloc[i-1] and 
                    data['rsi'].iloc[i] > rsi_buy):
                    data.loc[data.index[i], 'signal'] = 1.0
                
                # Rule 2: Price crosses below VWAP and RSI is below threshold -> Sell signal
                elif (data['close'].iloc[i] < data['vwap'].iloc[i] and 
                      data['close'].iloc[i-1] >= data['vwap'].iloc[i-1] and 
                      data['rsi'].iloc[i] < rsi_sell):
                    data.loc[data.index[i], 'signal'] = -1.0
                
                # Rule 3: Price is above EMA, but ATR is increasing significantly -> Sell signal (volatility exit)
                elif (data['close'].iloc[i] > data['ema'].iloc[i] and 
                      data['atr'].iloc[i] > data['atr'].iloc[i-1] * volatility_increase):
                    data.loc[data.index[i], 'signal'] = -1.0
                
                # Rule 4: Price is below EMA, but volatility is decreasing -> Buy signal (potential reversal)
                elif (data['close'].iloc[i] < data['ema'].iloc[i] and 
                      data['atr'].iloc[i] < data['atr'].iloc[i-1] * volatility_decrease and
                      data['rsi'].iloc[i] > rsi_oversold):
                    data.loc[data.index[i], 'signal'] = 1.0
            
            # Add a column for position to track the current position based on signals
            data['position'] = 0
            
            # Initialize the first position based on the first signal
            if data['signal'].iloc[0] != 0:
                data.loc[data.index[0], 'position'] = data['signal'].iloc[0]
            
            # Update positions based on signals
            for i in range(1, len(data)):
                # If there's a new signal, update the position
                if data['signal'].iloc[i] != 0:
                    data.loc[data.index[i], 'position'] = data['signal'].iloc[i]
                # Otherwise, maintain the previous position
                else:
                    data.loc[data.index[i], 'position'] = data['position'].iloc[i-1]
            
            logger.info(f"Generated trading signals for {self.symbol} on {self.interval} timeframe")
            return data
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}", exc_info=True)
            return None


class EMAcrossStrategy(Strategy):
    """
    Strategy based on EMA crossovers.
    """
    
    def __init__(self, client: Client, symbol: str, interval: str, 
                 fast_ema_window: int = None, slow_ema_window: int = None):
        """
        Initialize the EMA crossover strategy.
        
        Args:
            client (Client): Binance API client
            symbol (str): Trading symbol
            interval (str): Timeframe interval
            fast_ema_window (int, optional): Fast EMA window. Defaults to 9 if None.
            slow_ema_window (int, optional): Slow EMA window. Defaults to 21 if None.
        """
        super().__init__(client, symbol, interval)
        
        # Use provided parameters or defaults
        self.fast_ema_window = fast_ema_window if fast_ema_window else 9
        self.slow_ema_window = slow_ema_window if slow_ema_window else 21
        
        # Initialize indicators (we'll still use the TechnicalIndicators class, but override the calculate_indicators method)
        self.indicators = TechnicalIndicators()
    
    def calculate_indicators(self):
        """
        Retrieve data and calculate EMA indicators for crossover strategy.
        
        Returns:
            pd.DataFrame: DataFrame with EMA indicators calculated
        """
        try:
            from ta.trend import EMAIndicator
            
            # Get data
            data = self.data_feed.get_data()
            if data is None or data.empty:
                return None
            
            # Calculate fast and slow EMAs
            fast_ema = EMAIndicator(close=data['close'], window=self.fast_ema_window, fillna=True)
            slow_ema = EMAIndicator(close=data['close'], window=self.slow_ema_window, fillna=True)
            
            # Add EMAs to the dataframe
            data['fast_ema'] = fast_ema.ema_indicator()
            data['slow_ema'] = slow_ema.ema_indicator()
            
            return data
        except Exception as e:
            logger.error(f"An error occurred while calculating indicators: {e}", exc_info=True)
            return None
    
    def run(self):
        """
        Generate trading signals based on EMA crossovers.
        
        Returns:
            pd.DataFrame: DataFrame with trading signals
        """
        try:
            # Get data with calculated indicators
            data = self.calculate_indicators()
            
            if data is None or data.empty:
                logger.warning(f"No data available to generate trading signals for {self.symbol}")
                return None
            
            # Initialize signal column with 0.0 (no signal/hold)
            data['signal'] = 0.0
            
            # Generate signals based on EMA crossovers
            for i in range(1, len(data)):
                # Fast EMA crosses above slow EMA -> Buy signal
                if (data['fast_ema'].iloc[i] > data['slow_ema'].iloc[i] and 
                    data['fast_ema'].iloc[i-1] <= data['slow_ema'].iloc[i-1]):
                    data.loc[data.index[i], 'signal'] = 1.0
                
                # Fast EMA crosses below slow EMA -> Sell signal
                elif (data['fast_ema'].iloc[i] < data['slow_ema'].iloc[i] and 
                      data['fast_ema'].iloc[i-1] >= data['slow_ema'].iloc[i-1]):
                    data.loc[data.index[i], 'signal'] = -1.0
            
            # Add a column for position to track the current position based on signals
            data['position'] = 0
            
            # Initialize the first position based on the first signal
            if data['signal'].iloc[0] != 0:
                data.loc[data.index[0], 'position'] = data['signal'].iloc[0]
            
            # Update positions based on signals
            for i in range(1, len(data)):
                # If there's a new signal, update the position
                if data['signal'].iloc[i] != 0:
                    data.loc[data.index[i], 'position'] = data['signal'].iloc[i]
                # Otherwise, maintain the previous position
                else:
                    data.loc[data.index[i], 'position'] = data['position'].iloc[i-1]
            
            logger.info(f"Generated trading signals for {self.symbol} on {self.interval} timeframe")
            return data
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}", exc_info=True)
            return None


class MACDStrategy(Strategy):
    """
    Strategy based on MACD (Moving Average Convergence Divergence).
    """
    
    def __init__(self, client: Client, symbol: str, interval: str, 
                 fast_window: int = None, slow_window: int = None, signal_window: int = None):
        """
        Initialize the MACD strategy.
        
        Args:
            client (Client): Binance API client
            symbol (str): Trading symbol
            interval (str): Timeframe interval
            fast_window (int, optional): Fast window. Defaults to 12 if None.
            slow_window (int, optional): Slow window. Defaults to 26 if None.
            signal_window (int, optional): Signal window. Defaults to 9 if None.
        """
        super().__init__(client, symbol, interval)
        
        # Use provided parameters or defaults
        self.fast_window = fast_window if fast_window else 12
        self.slow_window = slow_window if slow_window else 26
        self.signal_window = signal_window if signal_window else 9
    
    def calculate_indicators(self):
        """
        Retrieve data and calculate MACD indicators.
        
        Returns:
            pd.DataFrame: DataFrame with MACD indicators calculated
        """
        try:
            from ta.trend import MACD
            
            # Get data
            data = self.data_feed.get_data()
            if data is None or data.empty:
                return None
            
            # Calculate MACD
            macd = MACD(
                close=data['close'], 
                window_fast=self.fast_window, 
                window_slow=self.slow_window, 
                window_sign=self.signal_window,
                fillna=True
            )
            
            # Add MACD components to the dataframe
            data['macd'] = macd.macd()
            data['macd_signal'] = macd.macd_signal()
            data['macd_diff'] = macd.macd_diff()
            
            return data
        except Exception as e:
            logger.error(f"An error occurred while calculating indicators: {e}", exc_info=True)
            return None
    
    def run(self):
        """
        Generate trading signals based on MACD.
        
        Returns:
            pd.DataFrame: DataFrame with trading signals
        """
        try:
            # Get data with calculated indicators
            data = self.calculate_indicators()
            
            if data is None or data.empty:
                logger.warning(f"No data available to generate trading signals for {self.symbol}")
                return None
            
            # Initialize signal column with 0.0 (no signal/hold)
            data['signal'] = 0.0
            
            # Generate signals based on MACD crossovers
            for i in range(1, len(data)):
                # MACD line crosses above signal line -> Buy signal
                if (data['macd'].iloc[i] > data['macd_signal'].iloc[i] and 
                    data['macd'].iloc[i-1] <= data['macd_signal'].iloc[i-1]):
                    data.loc[data.index[i], 'signal'] = 1.0
                
                # MACD line crosses below signal line -> Sell signal
                elif (data['macd'].iloc[i] < data['macd_signal'].iloc[i] and 
                      data['macd'].iloc[i-1] >= data['macd_signal'].iloc[i-1]):
                    data.loc[data.index[i], 'signal'] = -1.0
            
            # Add a column for position to track the current position based on signals
            data['position'] = 0
            
            # Initialize the first position based on the first signal
            if data['signal'].iloc[0] != 0:
                data.loc[data.index[0], 'position'] = data['signal'].iloc[0]
            
            # Update positions based on signals
            for i in range(1, len(data)):
                # If there's a new signal, update the position
                if data['signal'].iloc[i] != 0:
                    data.loc[data.index[i], 'position'] = data['signal'].iloc[i]
                # Otherwise, maintain the previous position
                else:
                    data.loc[data.index[i], 'position'] = data['position'].iloc[i-1]
            
            logger.info(f"Generated trading signals for {self.symbol} on {self.interval} timeframe")
            return data
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}", exc_info=True)
            return None


class StrategyFactory:
    """
    Factory class for creating different trading strategies.
    """
    
    # Dictionary to store registered strategy classes
    _strategies: Dict[str, Type[Strategy]] = {}
    
    @classmethod
    def register_strategy(cls, strategy_type: str, strategy_class: Type[Strategy]):
        """
        Register a strategy class with the factory.
        
        Args:
            strategy_type (str): Strategy type identifier
            strategy_class (Type[Strategy]): Strategy class
        """
        cls._strategies[strategy_type] = strategy_class
        logger.info(f"Registered strategy type: {strategy_type}")
    
    @classmethod
    def get_strategy_types(cls) -> list:
        """
        Get a list of all registered strategy types.
        
        Returns:
            list: List of strategy types
        """
        return list(cls._strategies.keys())
    
    @classmethod
    def create_strategy(cls, strategy_type: str, client: Client, symbol: str, interval: str, **kwargs) -> Optional[Strategy]:
        """
        Create a strategy instance based on the strategy type.
        
        Args:
            strategy_type (str): Strategy type identifier
            client (Client): Binance API client
            symbol (str): Trading symbol
            interval (str): Timeframe interval
            **kwargs: Additional parameters for the strategy
            
        Returns:
            Strategy: Strategy instance or None if the strategy type is not registered
        """
        if strategy_type not in cls._strategies:
            logger.error(f"Unknown strategy type: {strategy_type}")
            return None
        
        strategy_class = cls._strategies[strategy_type]
        try:
            strategy = strategy_class(client, symbol, interval, **kwargs)
            logger.info(f"Created strategy: {strategy_type} for {symbol} on {interval}")
            return strategy
        except Exception as e:
            logger.error(f"Error creating strategy: {e}", exc_info=True)
            return None


# Register strategies with the factory
StrategyFactory.register_strategy("btc", BTCStrategy)
StrategyFactory.register_strategy("ema_cross", EMAcrossStrategy)
StrategyFactory.register_strategy("macd", MACDStrategy)
